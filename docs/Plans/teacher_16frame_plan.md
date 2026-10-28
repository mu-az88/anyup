# Fix: Teacher always sees 16 native-stride frames

**Repo:** `mu-az88/anyup`, branch `fix/teacher-full-16frame-clips` (branched from `3Dconv` @ `e696743`). Training pipeline: `Train_3D_full.ipynb` at the repo root. Cells are referred to by their markdown headings ("cell 3" … "cell 14").

**Scope:** implement only this fix. Don't fix other issues you notice; add them to **Follow-ups** at the end.

**Comment rule:** on lines that control compute (frames, stride, batch size, resolution, token count), add a short comment and name the lines that depend on it.

**Log rule:** when done, add an entry to `docs/DECISIONS.md` (Date / Branch / Decision / Where / Affects).

## Problem

VideoMAE-base (`MCG-NJU/videomae-base`) was pretrained on 16-frame clips sampled at stride 4 (8 tubelets × 196 = 1568 tokens). The curriculum (T = 2→4→8) feeds the teacher 2/4/8 consecutive frames (1/2/4 tubelets), so:

1. **The teacher runs out of distribution, so the GT is degraded.** `T_tok != 8` in every stage, so the interpolation branch in `extract_teacher_features` (cell 12) always runs.
2. **The interpolated positions are meaningless.** `get_sinusoid_encoding_table` builds a 1D sinusoid over the flattened (t, h, w) index (verified in transformers 5.5.4, `modeling_videomae.py:80,106`). Reshaping it to `(8, 196)` and interpolating along time blends sinusoids that match no real position.
3. **Train and eval don't match.** Exp 2 (`results/exp2_temporal_consistency.ipynb`, cell 3) runs the teacher on full 16-frame clips. Training uses degraded short-clip features; evaluation uses proper ones.
4. **The frame timing is wrong.** The loader (cell 9, `_sample_frames`) reads consecutive frames (stride 1, ~0.64 s of UCF101). Pretraining used stride 4. Fixing only the clip length still leaves the teacher with clips that have about 4× less motion than it was trained on.

## Fix

Always run the teacher on 16 frames at `cfg.frame_stride` (default 4). The student gets a random T-frame, tubelet-aligned window. GT and input features are the matching tubelet slice of the 16-frame pass. The position table is never modified.

## Steps

**0. Verify first (on Colab).** Print `transformers.__version__`, `type(encoder.embeddings.position_embeddings)` and its shape, and confirm `VideoMAEEmbeddings.forward` adds the table without any `interpolate_pos_encoding` path. Expected: a plain `torch.Tensor` (not a Parameter) of shape `(1, 1568, 768)`. If anything differs, stop and tell the user.

**1. Create `anyup/data/training/teacher.py`:**

```python
import random
import torch
import torch.nn.functional as F


@torch.no_grad()
def extract_teacher_features(encoder, video, device=None):
    """video (B, 16, 3, H, W) -> (B, D, 8, 14, 14)"""
    c = encoder.config
    B, T, C, H, W = video.shape
    if T != c.num_frames:                                  # 16 — teacher native; loader must load this many
        raise ValueError(f"teacher needs {c.num_frames} frames, got {T}")
    pv = video.to(device or next(encoder.parameters()).device)
    if (H, W) != (c.image_size, c.image_size):             # 224; crop_low (112) is upscaled here
        pv = F.interpolate(pv.reshape(B * T, C, H, W), size=(c.image_size, c.image_size),
                           mode="bilinear", align_corners=False).view(B, T, C, c.image_size, c.image_size)
    g = c.image_size // c.patch_size                       # 14 tokens per side
    out = encoder(pixel_values=pv).last_hidden_state       # (B, 1568, D) — position table untouched
    return out.reshape(B, T // c.tubelet_size, g, g, c.hidden_size).permute(0, 4, 1, 2, 3).contiguous()


def sample_clip_indices(total, n_frames, stride, rng=random):
    """n_frames indices at `stride`; falls back to the largest stride that fits, then clamps (pads) at the end."""
    s = max(1, min(stride, (total - 1) // (n_frames - 1)))  # stride 4 → span 61 frames; short clips shrink s
    start = rng.randint(0, max(0, total - ((n_frames - 1) * s + 1)))
    return [min(start + i * s, total - 1) for i in range(n_frames)]


def sample_student_window(t_student, teacher_t, tubelet, rng=random):
    """Random tubelet-aligned window. Returns (f0, tok0, n_tok)."""
    if t_student % tubelet or not tubelet <= t_student <= teacher_t:
        raise ValueError(f"invalid t_student={t_student}")
    n_tok = t_student // tubelet                           # T=2→1, 4→2, 8→4 tubelets
    tok0 = rng.randint(0, teacher_t // tubelet - n_tok)
    return tok0 * tubelet, tok0, n_tok


def take_window(frames, feats, f0, tok0, n_tok, tubelet):
    """frames (B, 16, ...) and feats (B, D, 8, h, w) -> matching window of each."""
    return frames[:, f0:f0 + n_tok * tubelet], feats[:, :, tok0:tok0 + n_tok].contiguous()
```

**2. In the notebook:**

- **cell 2:** set `REPO_BRANCH = "fix/teacher-full-16frame-clips"`. The clone is skipped if `/content/anyup` already exists, so restart the runtime (or delete that folder) first, otherwise `teacher.py` won't be found.
- **cell 3 (config):**
  - Add `frame_stride: int = 4` under `# ── T-curriculum` with the comment `# teacher clip stride (VideoMAE pretrain = 4); 16 frames span 61 source frames`.
  - In the Colab overrides, set `cfg.experiment_name = "run_03"` and `cfg.resume = None`. Right now `resume` points at `run_02/anyup3d_step6000.pth`; leaving it would load weights trained on degraded GT and write into the run_02 folder.
- **cell 6:** after the encoder loads, add `TEACHER_T = encoder.config.num_frames  # 16 — every build_loader call and the window sampling depend on this`.
- **cell 9 (dataset):**
  - Give `UCF101TopNDataset.__init__` a `stride` argument.
  - In `_sample_frames`, replace the consecutive `start`/`wanted` logic with `idxs = sample_clip_indices(total, self.t, self.stride)`.
  - Decode until `fi >= idxs[-1]`, keep the frames whose index is in `set(idxs)`, then build the list in `idxs` order. Indices can repeat on short clips.
  - Change the fallback to `total = stream.frames or self.t * self.stride`. Keep the existing padding loop.
  - `build_loader` passes `stride=cfg.frame_stride`.
- **cell 9 and cell 13:** pass `TEACHER_T` instead of `current_stage.t` (or `8`) to **all five** `build_loader` calls: the quick test (cell 9), initial, curriculum switch, StopIteration, and validation (cell 13).
- **cell 12:** delete the old `extract_teacher_features`, including the position-embedding interpolation. Import the four functions from `anyup.data.training.teacher` instead. Leave `get_rgb_diff_gate` unchanged.
- **cell 13 (training step):**

```python
crop_high, crop_low = sample_tube_crop(video)                     # (B, 16, ...)
f0, tok0, n_tok = sample_student_window(current_stage.t, TEACHER_T, TUBELET_SIZE)  # student T drives VRAM

gt_full  = extract_teacher_features(encoder, crop_high, device)   # (B, 768, 8, 14, 14)
gui_full = extract_teacher_features(encoder, crop_low,  device)
crop_high_s, gt_feats       = take_window(crop_high, gt_full,  f0, tok0, n_tok, TUBELET_SIZE)
crop_low_s,  guidance_feats = take_window(crop_low,  gui_full, f0, tok0, n_tok, TUBELET_SIZE)
T_tok = n_tok
```

  - Downsample the **sliced** `guidance_feats` to 7×7, as now.
  - Use `crop_high_s` as the AnyUp guide, and pass `crop_low_s` to `get_rgb_diff_gate`. The full `crop_low` gives 7 pairs instead of `n_tok - 1` and breaks the gate shape.
  - Self-consistency: add noise to the full `crop_low`, run the teacher, slice `[:, :, tok0:tok0+n_tok]`, then downsample to 7×7.
- **cell 13 (validation):** use a fixed, centered window of the largest stage size, so `loss/val_rec` stays comparable across curriculum stages:
  - `VAL_N_TOK = max(s.t for s in cfg.t_schedule) // TUBELET_SIZE` (4).
  - `tok0 = (TEACHER_T // TUBELET_SIZE - VAL_N_TOK) // 2` (2), and `f0 = tok0 * TUBELET_SIZE`.
  - Use `batch_size=1` for the validation loader.
- **cell 14 (smoke test):** leave unchanged. It doesn't call the teacher.

## Tests: `tests/test_teacher_16frame.py`

CPU-only, using a tiny random VideoMAE (no download). All tests below were checked against the code above.

```python
import random
import pytest
import torch
from transformers import VideoMAEConfig, VideoMAEModel
from anyup.data.training.teacher import (extract_teacher_features, sample_clip_indices,
                                         sample_student_window, take_window)


@pytest.fixture(scope="module")
def teacher():
    cfg = VideoMAEConfig(num_frames=16, tubelet_size=2, hidden_size=32, num_hidden_layers=1,
                         num_attention_heads=2, intermediate_size=64)
    return VideoMAEModel(cfg).eval()


@pytest.mark.parametrize("size", [224, 112])
def test_shape(teacher, size):
    assert extract_teacher_features(teacher, torch.randn(1, 16, 3, size, size)).shape == (1, 32, 8, 14, 14)


@pytest.mark.parametrize("t", [2, 4, 8])
def test_rejects_short_clips(teacher, t):
    with pytest.raises(ValueError):
        extract_teacher_features(teacher, torch.randn(1, t, 3, 224, 224))


def test_position_table_unchanged(teacher):
    before = teacher.embeddings.position_embeddings.clone()
    extract_teacher_features(teacher, torch.randn(1, 16, 3, 224, 224))
    assert torch.equal(teacher.embeddings.position_embeddings, before)


@pytest.mark.parametrize("total,stride", [(300, 4), (61, 4), (40, 2), (40, 4)])
def test_clip_indices_stride(total, stride):
    rng = random.Random(0)
    for _ in range(100):
        idx = sample_clip_indices(total, 16, stride, rng)
        steps = {b - a for a, b in zip(idx, idx[1:])}
        assert len(idx) == 16 and len(steps) == 1 and 1 <= steps.pop() <= stride and idx[-1] < total


def test_clip_indices_short_clip_pads():
    idx = sample_clip_indices(10, 16, 4, random.Random(0))
    assert len(idx) == 16 and idx == sorted(idx) and idx[-1] == 9    # last frame repeated


@pytest.mark.parametrize("t", [2, 4, 8])
def test_window_aligned_and_in_bounds(t):
    rng = random.Random(0)
    for _ in range(200):
        f0, tok0, n = sample_student_window(t, 16, 2, rng)
        assert f0 == 2 * tok0 and n == t // 2 and tok0 + n <= 8


@pytest.mark.parametrize("t", [2, 4, 8])
def test_frames_match_tokens(t):
    frames = torch.arange(16.).view(1, 16, 1, 1, 1)
    feats = torch.arange(8.).view(1, 1, 8, 1, 1)
    f0, tok0, n = sample_student_window(t, 16, 2, random.Random(1))
    fw, tw = take_window(frames, feats, f0, tok0, n, 2)
    assert torch.equal(fw.flatten()[::2] // 2, tw.flatten())   # tubelet k covers frames 2k, 2k+1
```

## Done when

- Step 0 is confirmed, and `pytest tests/test_teacher_16frame.py` passes.
- No position-embedding interpolation remains, all five `build_loader` calls use `TEACHER_T`, and the loader returns `(B, 16, 3, 256, 256)`.
- A debug run (with `start_step` values lowered temporarily, then restored) completes one step at T=2, 4 and 8 with finite losses. Report peak VRAM **and it/s** per stage. Stride-4 decoding reads about 61 frames per clip, so check whether the loader (`num_workers=2`) becomes the bottleneck.
- `docs/DECISIONS.md` has an entry for this change.

**Note:** retrain from the 2D init (`cfg.resume = None`, new `experiment_name`). Don't resume run_02 checkpoints; they were trained against degraded GT.

## Follow-ups (known, out of scope)

- **Student T gap (decided: note only):** Exp 2 runs the student on 8 tubelets; training stops at 4 (T=8). The commented-out `TStage(t=16)` would close the gap on an A100.
- **Guide frames differ:** training passes 2 guide frames per tubelet (`crop_high_s`). Exp 2 passes 1 (`frames[::2]`). AnyUp average-pools the image encoder to `t = T_tok` (`anyup/model.py:86-95`), so this runs, but the inputs differ in distribution.
- **Exp 2 sampling:** Exp 2 spreads 16 frames over the whole clip (`np.linspace`). Training now uses stride 4. That's closer than before, but not identical.
- **Validation randomness:** validation still uses random spatial crops and random temporal starts, so `val_rec` is noisy.
- **Smoke test arguments (cell 14):** `anyup(dummy_low_5d, dummy_high_5d, ...)` passes the image as features. It only works because the asserted shapes happen to match.
