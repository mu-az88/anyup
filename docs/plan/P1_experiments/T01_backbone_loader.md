# T01 — Unified Backbone Loader

| Field | Value |
|---|---|
| Task ID | T01 |
| Plan | P1 Experiments (`plan.md`), Phase A.1 |
| Branch | `experiments-backbone-loader` |
| Status | done (2026-10-30) |
| Depends on | — |
| Blocks | T02 Video reader, T03 Upsampler wrapper, A.3–A.5 |

## Goal

Every experiment loads its backbone through one entry point and gets features in one standard format (`features`, `guide`, `meta`). Adding a backbone means writing one small adapter and nothing else.

## Scope

- **In scope:** `BackboneLoader` with three adapters (`videomae`, `swin_t`, `dinov2`), the standard output object, and a Colab check notebook with regression checks against the old notebook code.
- **Out of scope:**
  - Video reading, chunking long videos, and last-chunk padding. That is **T02**. T01 adapters process one chunk `(T, 3, H, W)`.
  - The upsampler wrapper (AnyUp2D / AnyUp3D / bilinear, `window_t`, checkpoint loading, the `TStage` stub). That is **T03**.
  - DAVIS/ADE20K annotation loading. That is A.3.
  - Refactoring the experiment notebooks to use the module. That happens after T03, once backbones, the video reader and upsamplers are all shared.

## Findings From Reviewing the Notebooks

| | Exp 1 (`results/exp1_spatial_upsampling_quality.ipynb`) | Exp 2 (`results/exp2_temporal_consistency.ipynb`) | Exp 3 (`results/exp3_product_in_action_FIXED.ipynb`) |
|---|---|---|---|
| Task | ADE20K linear-probe mIoU (3 seeds, 3 epochs) | Adjacent-frame temporal cosine, DAVIS-2017 val | Label-propagation J&F, DAVIS-2017 val |
| Backbone | DINOv2-base (HF `facebook/dinov2-base`), single images | VideoMAE (`MCG-NJU/videomae-base`) | Video Swin-T (torchvision `KINETICS400_V1`), `norm` layer hook |
| Backbone input | 1 image, 224×224, ImageNet-norm | 16 frames (linspace over the clip), 224, ImageNet-norm | Every 12th frame, 224, ImageNet-norm; each frame duplicated into a 2-frame clip |
| Features | (B, 768, 16, 16), CLS dropped | (8, 768, 14, 14) | (T, 768, 7, 7), T' = T |
| Guide | ImageNet-norm, 224 | ImageNet-norm, 224, `frames[::2]` | **raw [0,1]**, 56×56 |
| AnyUp3D ckpt / `window_t` | `step500` / **3** | `step6500` / 4 | `step6500` / 4 |
| Upsamplers | bilinear, AnyUp2D, AnyUp3D (T=1) | AnyUp2D, AnyUp3D | bilinear, AnyUp3D |
| Output size | 224 | 224 (guide size) | 56 |

Issues found (logged in `plan.md` → Open Questions / Phase B):

1. Training used `window_t=4` (`Train_3D_full.ipynb`). Exp 1 ran with `window_t=3` and the `step500` checkpoint.
2. Training feeds ImageNet-normalized guides (`anyup/data/video_dataset.py:214`). Exp 3 fed raw [0,1] guides.
3. In Exp 3, the `FEAT_SIZE = 23` comment is wrong; the actual Swin-T grid at 224 is 7×7.
4. AnyUp3D was trained on VideoMAE features (`config/anyup3d_train.yaml:12`), so DINOv2 and Swin-T are both zero-shot backbones.

## Q&A (decisions agreed before implementation)

| # | Question | Answer |
|---|---|---|
| 1 | Format of `out.guide`? | **ImageNet-normalized**, matching training. Exp 3's raw-[0,1] guide is treated as a bug; its numbers may change in Phase B. |
| 2 | Canonical `window_t`? | **Configurable, default 4.** Belongs to the upsampler, so it goes in T03. |
| 3 | Swin-T temporal handling? | **Both modes**, kwarg `temporal_mode='per_frame' \| 'clip'`, default `per_frame` (reproduces Exp 3). |
| 4 | VideoMAE guide alignment (tubelet 2)? | **First frame of each tubelet** (`frames[::2]`), as in Exp 2. |
| 5 | T01 scope? | **Backbones only.** The video reader is T02; the Upsampler is T03. |
| 6 | Input to `bb(frames)`? | **Tensor (T, 3, H, W), float RGB in [0,1], any size.** One chunk. The adapter resizes and normalizes. Reading and chunking are T02. |
| 7 | Guide resolution? | **kwarg `guide_size`, default = backbone input size.** The guide is resized from the original input frames, not from the 224 tensor. |
| 8 | DINOv2 source? | **timm via the repo wrapper** (`anyup/backbone/vit_wrapper.py`, `PretrainedViTWrapper`). All experiments and notebooks use the wrapper. The check must confirm pretrained weights are loaded. |
| 9 | Role of Exp 1? | **Keep it as the spatial no-regression check.** Re-run in Phase B with `window_t=4` and the canonical checkpoint. |
| 10 | Module location? | **`anyup/backbones.py`** |
| 11 | How to test (no torch/GPU locally)? | **Colab check notebook**: you run it and paste back the output. |
| 12 | Fix wrong Exp 1/2 descriptions in the docs? | **Yes**: fix them and log the issues. |
| 13 | The open questions (fidelity source, Exp 4 notebook, ...)? | **Parked**: not needed for T01. |
| 14 | Swin-T backbone input normalization? (The torchvision Kinetics weights use their own stats; Exp 3 used ImageNet.) | **Kinetics constants by default**, kwarg `norm='kinetics' \| 'imagenet'`. The regression check runs with `'imagenet'` to match Exp 3. Exp 3 changes in Phase B. |
| 15 | Notebook refactor in T01? | **No**: after T03. |
| 16 | VideoMAE frame count? | **Exactly 16, else ValueError.** Padding short chunks is T02's job. |
| 17 | Swin `clip`-mode guide? | **`frames[::2]`** (first frame of each tubelet), same as VideoMAE. |
| 18 | Regression check input? | **(a)** an exact check on PIL-pre-resized 224 frames; **(b)** the full path (tensor resize vs PIL), reported as a number, not pass/fail. |

## Design

### API

```python
from anyup.backbones import BackboneLoader

bb  = BackboneLoader.load("videomae", device="cuda", input_size=224)   # registry entry point
out = bb(frames)        # frames: (T, 3, H, W) float [0,1], any H, W
out.features            # (T', C, h, w) raw, unnormalized
out.guide               # (T', 3, gH, gW) ImageNet-normalized, aligned to T'
out.meta                # dict, see below
BackboneLoader.available()   # -> ["dinov2", "swin_t", "videomae"]
```

- Registry: `@register("name")` on each adapter class. `BackboneLoader.load(name, **kw)` builds it.
- Common kwargs: `device`, `input_size` (backbone input resolution), `guide_size` (None means `input_size`). Backbone-specific extras go in `**kwargs`.
- Backbones are frozen (`eval()`, `requires_grad_(False)`). The forward pass runs under `torch.no_grad()`. Outputs are float32 on `device`.
- Resizing uses `torchvision.transforms.functional.resize(..., antialias=True)`, bilinear, applied to the [0,1] tensor before ImageNet normalization.
- `meta` keys: `name`, `feat_dim`, `grid` (h, w), `patch_size`, `temporal_stride`, `t_in`, `t_out`, `frame_index` (for each feature frame, the index of the input frame used as its guide), `input_size`, `guide_size`.

### Adapters

| Name | Source | Input constraint | Output | Temporal | Guide frames |
|---|---|---|---|---|---|
| `videomae` | HF `VideoMAEModel` `MCG-NJU/videomae-base` | T must be 16 (fixed position embeddings) | `last_hidden_state` (1, 1568, 768) → (8, 768, 14, 14) | stride 2 | `frames[::2]` |
| `swin_t` (`per_frame`) | torchvision `swin3d_t(KINETICS400_V1)`, `norm` output (B, T, H, W, C) → permute. Computed as `patch_embed → pos_drop → features → norm` (head not run). Input norm: Kinetics stats (kwarg `norm`). | any T ≥ 1 | (T, 768, 7, 7) | stride 1 (each frame duplicated into a 2-frame clip, batched) | all frames |
| `swin_t` (`clip`) | same | any T ≥ 1 | (ceil(T/2), 768, 7, 7) | stride 2 | `frames[::2]` |
| `dinov2` | `PretrainedViTWrapper("vit_base_patch14_dinov2.lvd142m")` (kwarg `variant` = small/base/large) | any T ≥ 1 (images are T=1); frames batched | (T, 768, 16, 16) at 224 | stride 1 | all frames |

## Steps

1. Implement `anyup/backbones.py`: output dataclass, registry, shared preprocessing (resize + normalize + guide), and the three adapters.
2. Write `tests/T01_backbone_loader_check.ipynb` (Colab):
   - Shapes and `meta` for every adapter, including T=1 for DINOv2 and odd T for Swin clip mode.
   - Guide is ImageNet-normalized, at the right size, and `frame_index` is correct.
   - Regression check (see below).
3. You run it on Colab and paste back the output. I log the results here.
4. Update `plan.md` (Task Index) and this file. Close out.

## Tests / Regression Checks

- **VideoMAE vs Exp 2 code:** same 16 DAVIS frames.
  - (a) Exact check: feed the new adapter frames already resized to 224 the same way as Exp 2 (PIL resize). The features must match Exp 2's `extract_videomae_features` (max abs diff ≈ 0).
  - (b) Full-path check: report the cosine similarity and max diff between our tensor resize and Exp 2's PIL resize. Small, not bitwise, is expected.
- **Swin-T per-frame vs Exp 3 code:** the same (a)/(b) checks against `extract_swin_features` on the same sampled frames.
- **DINOv2 (timm wrapper):**
  - Pretrained weights confirmed: the wrapper's weights equal a fresh `timm.create_model(name, pretrained=True)`, and differ from `pretrained=False`.
  - Semantic sanity check: cosine(features of an image, features of its horizontal flip) is much higher than cosine(image, noise).
  - Shapes for T=1 and T>1.
  - No comparison to Exp 1's HF features: we're switching to the wrapper, and Exp 1 is re-run in Phase B.

## Done When

- `anyup/backbones.py` exists with one entry point and three adapters.
- The Colab check passes: shapes, alignment and guide norm are correct; VideoMAE and Swin per-frame match the old code exactly on pre-resized input; DINOv2 has its pretrained weights confirmed.
- Adding a backbone requires only one new adapter class.

## Implementation Log

_Updated live during implementation. Per step: what changed, files touched, test results (actual numbers/shapes). Also log mid-implementation questions and answers, changed decisions (old → new, why), and anything added or removed. Sections above are edited in place to match, so the finished file is a report of what was actually done._

### Step 1: `anyup/backbones.py` (written 2026-10-30)

- New file `anyup/backbones.py`: `BackboneOutput` dataclass, `register` decorator plus `BackboneLoader.load` / `.available()`, shared `resize_frames` (bilinear + antialias, a no-op when already at size) and `normalize_frames` (same arithmetic as `T.Normalize`), base `BackboneAdapter` (freeze, input check, guide built from the ORIGINAL frames), and the adapters `VideoMAEAdapter`, `SwinTAdapter`, `DINOv2Adapter`.
- Swin features are computed by calling `patch_embed → pos_drop → features → norm` directly instead of a forward hook. This is equivalent to Exp 3's hook on `norm` and skips the classification head. The regression check verifies it.
- The Swin clip-mode temporal stride is read from `model.patch_embed.proj.stride[0]` (= 2).
- DINOv2 `feat_dim` is taken from `PretrainedViTWrapper.embed_dim`; the input size must be a multiple of 14.
- Syntax checked locally (no torch here). Not yet run.

### Step 2: Colab check notebook (written 2026-10-30)

- New file `tests/T01_backbone_loader_check.ipynb`. It uses the first DAVIS-2017 val clip, with Exp 2 sampling (16 linspace frames) and Exp 3 sampling (every 12th frame).
- Exact-check tolerance: max abs diff < 1e-3 and worst-frame cosine > 0.99999. This allows for CPU vs GPU normalization arithmetic and cuDNN nondeterminism.
- Checks: shapes / meta / frame_index / guide norm for all adapters, including videomae `guide_size=56`, VideoMAE rejecting T≠16, Swin clip mode with odd T=5 → 3, and DINOv2 with T=1 and T=5; VideoMAE vs Exp 2 (a)/(b) plus guide vs Exp 2 guides; Swin per-frame vs Exp 3 (a)/(b), plus kinetics-vs-imagenet diff (info only); DINOv2 weights equal timm `pretrained=True` and differ from `pretrained=False`, plus a flip-vs-noise sanity check.
- The notebook clones from GitHub, so the new code must be pushed before running. ~~Branch `3Dconv`~~ → changed to **`experiments-backbone-loader`** (see the fix below).
- You confirmed the local experiment notebooks (`results/exp*.ipynb`) are up to date, so the Exp 2/3 code copied into the check notebook is the current reference.
- You commit and push. Next: run `tests/T01_backbone_loader_check.ipynb` on Colab (GPU), paste back the Summary cell output, and log the results here as Step 3.

### Fix: check notebook branch (2026-10-30)

- First Colab run failed: `ModuleNotFoundError: No module named 'anyup.backbones'`. Cause: commit `7d58f63` was pushed to the branch `experiments-backbone-loader`, but the notebook cloned `3Dconv`, and with an existing `/content/anyup` it only ran `git pull` on the current branch.
- Changed Cell 1 of `tests/T01_backbone_loader_check.ipynb`: `REPO_BRANCH = "experiments-backbone-loader"`; for an existing clone it now runs `fetch` + `checkout -B <branch> origin/<branch>`; it raises if `anyup` is already imported (the runtime must be restarted).

### Step 3: Colab check results (2026-10-30)

Environment: torch 2.11.0+cu130, torchvision 0.26.0+cu130, transformers 5.20.2, timm 1.0.30, Tesla T4. Clip: `bike-packing`. **16 checks, 0 failed.**

| Check | Result |
|---|---|
| videomae T=16 | features (8, 768, 14, 14), guide (8, 3, 224, 224), frame_index [0, 2, …, 14]: PASS |
| videomae guide_size=56 | guide (8, 3, 56, 56): PASS |
| videomae rejects T≠16 | ValueError raised: PASS |
| swin_t per_frame T=5 | (5, 768, 7, 7), frame_index [0..4]: PASS |
| swin_t clip T=5 | (3, 768, 7, 7), frame_index [0, 2, 4]: PASS |
| dinov2 T=1 / T=5 | (1, 768, 16, 16) / (5, 768, 16, 16): PASS |
| videomae vs Exp 2 (a) exact | max_abs 0.0: PASS |
| videomae guide vs Exp 2 guides | max_abs 0.0: PASS |
| swin_t per_frame vs Exp 3 (a) exact, norm=imagenet | max_abs 0.0: PASS (confirms calling `patch_embed → features → norm` directly = Exp 3's hook) |
| dinov2 weights == timm pretrained=True | 174 tensors equal: PASS; differ from random init: PASS |
| dinov2 sanity | cos(img, flip) 0.9689 vs cos(img, noise) −0.0013: PASS |
| videomae (b) full path, tensor vs PIL resize | INFO: max_abs 0.31, min_cos 0.99988 |
| swin_t (b) full path, tensor vs PIL resize | INFO: max_abs 1.39, min_cos 0.9935 |
| swin_t kinetics vs imagenet norm | INFO: max_abs 2.16, min_cos 0.9726 |

Notes:
- The new path reproduces the old Exp 2 / Exp 3 features **bit-exactly** when given the same resized input.
- The resize method alone changes features slightly (Swin-T cos 0.9935, more sensitive than VideoMAE 0.99988). That's not a bug, but it is one more expected source of difference vs the thesis numbers in Phase B, alongside the Kinetics norm (cos 0.9726), the normalized guide, `window_t`, and using all frames.

## Final Summary

- Delivered `anyup/backbones.py` (`BackboneLoader` + `videomae`, `swin_t`, `dinov2` adapters) and `tests/T01_backbone_loader_check.ipynb`.
- Differences from the original plan: the Swin input norm became Kinetics by default (Q14); Swin features are computed via direct layer calls instead of a hook (verified identical); the check notebook uses the branch `experiments-backbone-loader`.
- Not done here (by design): video reading/chunking (T02), upsamplers (T03), notebook refactor (after T03).

