# T01 — Unified Backbone Loader

| Field | Value |
|---|---|
| Task ID | T01 |
| Plan | P1 Experiments (`plan.md`), Phase A.1 |
| Status | todo |
| Depends on | — |
| Blocks | A.2 Upsampler wrapper, A.3–A.5 |

## Context

AnyUp3D extends AnyUp (2D feature upsampling) to video with temporal consistency. The M.Sc. thesis is accepted; the goal now is a publishable paper. The experiments are the current bottleneck.

Right now each experiment notebook loads its backbone and model in its own way. That makes results harder to trust and compare, and adding a new backbone means touching every notebook.

**Goal of this task:** every experiment loads backbones through one shared interface, receives features in one standard format, and upsamples them through one shared helper. Adding a new backbone should mean writing one small adapter and nothing else.

We are going through the experiment notebooks one at a time. For each one, we record what it needs from the loader, so the interface covers all experiments before we lock it.

---

## Agreed Design

### 1. One entry point, adapters behind it (registry pattern)

```python
bb = BackboneLoader.load("videomae", device="cuda", input_size=224, num_frames=16)
# optional sugar, same thing underneath:
bb = BackboneLoader.VideoMAE(device="cuda", input_size=224, num_frames=16)

out = bb(frames)      # frames: (T, 3, H, W)
out.features          # (T', C, h, w) - RAW, UNNORMALIZED (AnyUp normalizes internally)
out.guide             # (T', 3, H, W) - frames aligned to the T' feature frames
out.meta              # dict: patch_size, temporal_stride/tubelet, feat_dim, grid (h, w), name, ...
```

- Each backbone is a small adapter class, e.g. `VideoMAEAdapter`, `SwinTAdapter`, `DINOv2Adapter`. It is registered by name.
- The adapter owns all backbone-specific quirks: preprocessing and normalization constants, frame-count constraints, and the token-to-grid reshape.
- Every adapter returns the **same output object**. Experiment code only touches `features`, `guide`, and `meta`.
- Parameters are almost identical across backbones: name, device, input size, number of frames. Backbone-specific extras go in `**kwargs`.

### 2. Output size is NOT a backbone parameter

The backbone produces its native feature grid. Output size is the **upsampler's** target resolution. Keep these separate:

- `BackboneLoader` decides which backbone, the input size, and the frames.
- A shared upsampling helper (e.g. `Upsampler`) decides which upsampler (AnyUp2D, AnyUp3D, bilinear baseline, ...) and the output size.

```python
up = Upsampler.load("anyup3d", ckpt_path=..., device="cuda")
hr = up(out.guide, out.features, output_size=(H, W), q_chunk_size=...)
```

With this split, all experiments also upsample identically, and backbone × upsampler combinations come for free.

### 3. Temporal alignment is reported, never guessed

This is the most likely source of silent bugs across experiments.

- VideoMAE has tubelet size 2, so 16 input frames become 8 feature frames.
- Video Swin-T has its own temporal stride.
- DINOv2 is per-frame, so T' = T.

Each adapter reports its temporal stride in `meta` and returns a `guide` already aligned to T'. The alignment rule (which input frame represents each feature frame) must be explicit and documented per adapter.

### 4. Portability

- Start as **one self-contained `.py` module**, also pasteable as a single notebook cell.
- Later, move it into the repo, e.g. `anyup/backbones.py` and `anyup/upsamplers.py`.

---

## Backbones to Support First

| Name | Source | Notes |
|---|---|---|
| `videomae` | HF `MCG-NJU/videomae-base` | `last_hidden_state` (1, 1568, 768) → reshape (8, 14, 14, 768) → permute (8, 768, 14, 14). 16 frames, 224 input, tubelet 2, patch 16. |
| `swin_t` | torchvision Video Swin-T, Kinetics-400 weights | Used in Exp 3 as a zero-shot universality test. Verify the temporal stride and output grid from the code. |
| `dinov2` | DINOv2 (per-frame image backbone) | Patch 14; frames processed independently. |

---

## Known Gotchas

See `docs/context/project_context.md` (Implementation Gotchas and Environment). Those rules apply here.

---

## Coding Conventions (required)

- **Every line that controls compute cost gets a short comment.** This covers batch size, resolution, number of frames, token length, chunk size, attention window, and kernel size. Example: `# ↓ reduce to save memory`. The comment must also name any other lines that depend on that value, so it is clear what else to change to avoid crashes.
- Prefer targeted edits to existing code over full rewrites. Show only what changed.
- Verify implementation-dependent facts (kernel sizes, resolutions, strides) against the actual code before relying on them.
- Avoid unnecessary complexity. Give honest assessments, with no premature "it works" claims; test before claiming.

---

## Workflow for This Task

1. Read experiment notebook 1. Extract exactly how it loads the backbone and model, and what it consumes downstream: shapes, frame counts, normalization, resolutions.
2. Record its requirements in the table below.
3. Repeat for notebooks 2 and 3.
4. Finalize the `BackboneLoader` / `Upsampler` interface so it covers all recorded requirements.
5. Implement the adapters. Sanity-check each one: output shapes, T' alignment, and that features fed through the new path **match the old notebook's features numerically** on the same input (regression check before switching over).
6. Refactor each notebook to use the shared module.

### Requirements Tracker (fill in as notebooks are reviewed)

| Experiment | Backbone(s) | Input frames / size | Needs from loader | Upsampler(s) | Output size | Notes |
|---|---|---|---|---|---|---|
| Exp 1 (cosine fidelity) | ? | ? | ? | ? | ? | |
| Exp 2 (temporal cosine, DAVIS) | ? | ? | ? | ? | ? | |
| Exp 3 (label propagation J&F, DAVIS) | Video Swin-T | ? | ? | AnyUp2D vs AnyUp3D? | ? | |

### Done when

- One module, with one entry point for all backbones.
- All three experiments load backbones and upsamplers through it.
- Old vs new feature outputs match on the same inputs.
- Adding a backbone requires writing only one adapter.

---

## Q&A (decisions agreed before implementation)

_To be filled in the Q&A round before implementation starts._

## Implementation Log

_Short notes per step: what changed, files touched, test results (actual numbers/shapes)._
