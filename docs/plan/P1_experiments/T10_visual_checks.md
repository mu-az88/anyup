# T10 — Presentation-Grade Visual Checks for Exp 1–3

| Field | Value |
|---|---|
| Task ID | T10 |
| Plan | P1 Experiments (`plan.md`), addendum after P1 close (Phase B wrap-up) |
| Branch | `experiments-in-action-test` |
| Status | in progress |
| Depends on | T05 (Exp 2/3 notebooks), T06/T07 (Exp 1 notebook), saved results on Drive |
| Blocks | — |

## Goal

Upgrade the existing optional visual cells of Exp 1–3 so that a non-expert can see AnyUp3D's benefit by eye, next to the baselines and the ground truth (where there is one). The outputs must be good enough for a slide: a PNG figure per experiment plus videos for Exp 2/3.

## Scope

- In scope:
  - Upgrade the visual cells in place: Exp 1 Cell 7, Exp 2 Cell 6, Exp 3 Cell 6. Update their config lines in Cell 2.
  - A small shared helper module `anyup/viz.py` with pytest in `tests/test_viz.py` (Q7).
  - `REPO_BRANCH` → `experiments-in-action-test` in Cell 1 of the three notebooks.
  - Documentation (added on request): `docs/experiments/build_docs.py` indexes `anyup.viz`; `doc_content.py` step texts for the visual cells and config cells; `project_context.md`, `plan.md`; rebuild of `experiments.html` after the commit.
  - Standalone mode: each visual cell runs after Cells 1–3 (Exp 1: 1–4) and needs no main loop, because it reads the saved CSV, masks and heads from Drive.
- Out of scope: any change to the protocols, metrics or main loops; Exp 0; zoom-in crops; new experiments; investigating `drift-straight` (S08). That clip is only *shown* here as the failure case.

## Q&A (decisions agreed before implementation)

| # | Question | Answer |
|---|---|---|
| 1 | Upgrade the existing cells, a new showcase notebook, or both? | **Upgrade the existing visual cells in place.** |
| 2 | Exp 1 figure: feature PCA, segmentation, or both? | **Both** (recommended). PCA always runs (no head needed, T4 OK). The segmentation part runs only if the head checkpoints exist and otherwise prints why it was skipped. It warns when the heads come from the T4 profile (barely trained, not representative). |
| 3 | Which clips / images? | **Automatic: the largest AnyUp3D gains plus one honest failure case** (recommended). The gain is measured against the strongest baseline, AnyUp2D. The failure case is the smallest or most negative gain. The selection is printed and saved. A manual override list is available in the config. |
| 4 | Video format? | **Both H.264 MP4 and GIF** (recommended). MP4 for slides and browsers (not `mp4v`, which often won't play); GIF for embedding anywhere. |
| 5 | Visual cells without re-running the main loop? | **Yes** (recommended). They use in-memory results if present, else load from Drive (CSV, `masks.zip`, head checkpoints). `VIZ_RUN_DIR` in the config overrides the results folder; for example, the T05 runs are saved under an older `RUN_NAME` without the `_n30` suffix. |
| 6 | Tracking? | **T10 as a P1 addendum** (recommended), branch `experiments-in-action-test`. |
| 7 | Where do the shared helpers go? | **`anyup/viz.py`** (recommended): PCA fit/project with *fixed* normalization, flicker map, mask overlay, label strip, MP4/GIF writer. Exp 1 and Exp 2 share them, and they can be tested with pytest. The experiment-specific plotting stays in the cells. |

## Design

### `anyup/viz.py` (new, small, no model code)

```python
class FeaturePCA:                                   # 3-component PCA → RGB with FIXED colour scaling
    fit(samples: Tensor (N, C)) -> self             # torch.pca_lowrank on a random sample; stores mean, components, p1/p99 per component
    rgb(feats: Tensor (T, C, H, W)) -> np.uint8 (T, H, W, 3)   # project on the feats' device, clip to stored p1/p99
def flicker_map(feats: Tensor (T, C, H, W)) -> np.float32 (T-1, H, W)   # 1 − per-pixel cosine(t, t+1), float64 math
def overlay(rgb (H, W, 3) uint8, mask (H, W) int, alpha=0.55) -> uint8   # DAVIS palette (as Exp 3 Cell 6 today)
def label_strip(img uint8, text: str, height=22) -> uint8                # title burned above a video panel (cv2.putText)
def write_mp4(frames: List[uint8 (H, W, 3)], path, fps)                  # imageio-ffmpeg, libx264, yuv420p, even dims
def write_gif(frames, path, fps, max_width=960)                          # PIL, downscaled
```

- **Why fixed scaling matters:** if each frame were normalized by its own min/max, the colours would flicker even for perfectly stable features, and the video would show an artefact of the plot instead of the model. The scale is fixed once per condition (p1/p99 of the fit sample) and kept for all frames.
- The PCA basis is fitted **jointly on all shown conditions**, so a colour means the same feature direction in every panel.

### Exp 1 — Cell 7 (spatial sharpness, no regression)

- **Image selection:** from the per-image mIoU in `exp1_per_item.csv` (seed `SEEDS[0]`): the top `N_EXAMPLES−1` images by AnyUp3D − AnyUp2D, plus the most negative one. Without a CSV (no main loop yet), it uses the first `N_EXAMPLES` val images and prints that.
- **Figure A, `exp1_features_pca.png`:** columns are Image | LR DINOv2 features (16², nearest-upsampled, to show the blockiness) | bilinear | AnyUp2D | AnyUp3D, all as PCA colours. One PCA is fitted per image, jointly across the columns. Message: AnyUp3D edges follow the object boundaries like AnyUp2D does, while bilinear is blurry.
- **Figure B, `exp1_segmentation.png`** (only if the heads exist): Image | GT | 3 predictions, with the per-image mIoU in each panel title. Predictions are blacked out where the GT is unlabelled (id 0, not scored), so the panels compare only what is scored.

### Exp 2 — Cell 6 (temporal stability)

- **Clip selection:** from per-clip `all` (`exp2_per_item.csv` or `rows`): the top `N_VIZ_CLIPS−1` by AnyUp3D − AnyUp2D, plus the smallest gain.
- **Per selected clip:**
  - Backbone + upsamplers on all chunks. Per chunk, project to PCA RGB and compute the flicker map on the GPU, keeping only the uint8 / float maps on the CPU. The full upsampled clip is never stored (52 × 768 × 224² fp32 ≈ 8 GB).
  - **Figure, `exp2_<clip>.png`:** row 1 is the input frames. Then for each of bilinear / AnyUp2D / AnyUp3D, a PCA row and a flicker row (shared colour scale and colourbar; dark = stable). The columns are `VIZ_N_FRAMES` consecutive feature frames from the middle of the clip. The titles carry the clip-mean temporal cosine per condition.
  - **Curve, `exp2_<clip>_curve.png`:** the adjacent-pair temporal cosine over time per condition (`TemporalCosineStream.per_pair`), with chunk boundaries as dashed lines. The honest boundary dip of AnyUp3D is visible here.
  - **Video, `exp2_<clip>.mp4` + `.gif`:** the whole clip. Top row: Input | bilinear | AnyUp2D | AnyUp3D PCA. Bottom row: blank | 3 flicker maps. Each panel is labelled. Flicker in the baselines should be visible as colour boiling.
- PCA fit: `PCA_SAMPLE` random tokens from the first chunk of every shown condition, jointly.

### Exp 3 — Cell 6 (segmentation by propagation, has GT)

- **Clip selection:** from per-clip J&F (`exp3_per_item.csv` or `rows`): the top `N_VISUAL_CLIPS−1` by AnyUp3D − AnyUp2D, plus the most negative one (expected `drift-straight`).
- **Masks:** the local `MASK_ROOT` if present, else unzip `<VIZ_RUN_DIR>/masks.zip` into it.
- **Figure, `exp3_side_by_side.png`:** one row per clip. Columns: Frame 0 + given GT ("input") | frame t + GT | bilinear | AnyUp2D | AnyUp3D. The time t is at `VISUAL_FRAME_FRAC=0.8` of the clip, because propagation errors build up over time. Each panel carries the per-clip J&F.
- **Video, `exp3_<clip>.mp4` + `.gif` for every selected clip:** GT | bilinear | AnyUp2D | AnyUp3D overlays, labelled with the condition and the per-clip J&F, plus a frame counter.

### Config additions (Cell 2), each compute line commented with its dependents

| Exp | Knob | Default | Effect |
|---|---|---|---|
| all | `RUN_VISUALS` | False (existing) | — |
| all | `VIZ_RUN_DIR` | None (= `OUT_DIR`) | results folder to read the CSV / masks / heads from |
| all | `VIZ_ITEMS` | None (= auto) | manual clip / image list |
| 1 | `N_EXAMPLES` | 4 (existing) | ↓ images = batch of one upsampler call per condition. Dependent: Q_CHUNK, IMG_SIZE |
| 1 | `PCA_SAMPLE` | 20,000 (new) | ↓ tokens per condition and image for the PCA fit; speed only |
| 2 | `N_VIZ_CLIPS` | 3 | ↓ clips fully re-run through backbone + 3 upsamplers |
| 2 | `VIZ_N_FRAMES` | 6 | ↓ columns of the static figure |
| 2 | `PCA_SAMPLE` | 50,000 (existing) | ↓ PCA fit speed only |
| 2 | ~~`VIZ_CLIP`~~ | removed | replaced by `VIZ_ITEMS` |
| 2/3 | `VIDEO_FPS` / `VISUAL_FPS` | 6 / 8 (Exp 3 existing knob, 4 → 8) | display only |
| 3 | `VISUAL_SIZE` | 224 → 240 | panel height; width now keeps the DAVIS aspect ratio (was squashed to 224²) |
| 3 | `N_VISUAL_CLIPS` | 4 (existing) | ↓ clips in the figure and videos (masks are read from disk, cheap) |
| 3 | `VISUAL_FRAME_FRAC` | 0.8 | which frame the stills show |

Outputs go to `<VIZ_RUN_DIR>/visuals/`, and a `visuals_selection.json` records the chosen items and why.

## Steps

1. `anyup/viz.py` + `tests/test_viz.py`. You run pytest on Colab.
2. `REPO_BRANCH` in Cell 1 and the config lines in Cell 2 of all three notebooks (targeted JSON edits; other cells untouched).
3. Exp 3 Cell 6 (the closest to done, cheapest to check: reads the saved masks).
4. Exp 2 Cell 6.
5. Exp 1 Cell 7.
6. You run each visual cell on Colab (T4) in standalone mode and look at the outputs. Then close-out.

## Tests / Regression Checks

pytest (`tests/test_viz.py`, CPU, synthetic):
- `flicker_map` on identical frames gives 0 everywhere; for f and −f it gives 2; for random input it equals `1 − metrics._pair_cosine` per pixel, and its mean over pixels matches the per-pair metric to < 1e-9 (same float64 math).
- `FeaturePCA`: the output is uint8 (T, H, W, 3). With the same features in every frame, all frames get **identical** RGB (fixed scaling, no per-frame normalization). Seeded fit → deterministic output.
- `overlay` leaves background pixels unchanged; `write_mp4` / `write_gif` write a non-empty, readable file with the right frame count (small synthetic clip).

Colab (by eye, plus printed checks):
- Exp 3: the selected clips and the failure clip are printed with their J&F (panel labels come from the same CSV/rows, so no separate equality check).
- Exp 2: the mean of the flicker map per condition ≈ 1 − the CSV `all` for that clip (the same pairs): PASS if |diff| < 1e-6, else WARN (not an error: a CSV from the pre-T07, non-deterministic T05 run may differ slightly). Saved in `visuals_selection.json`.
- The MP4 plays in a browser and on the Colab preview.
- The main loops and their numbers are unchanged (the visual cells only read results).

## Done When

- `anyup/viz.py` exists and pytest passes on Colab.
- The three upgraded visual cells run standalone on a T4 and write their PNGs, plus MP4/GIF for Exp 2/3, under `visuals/`.
- You have looked at the outputs and agree that they show the effect (or we log what doesn't).

## Implementation Log

_Updated live during implementation._

- **OK received ("I trust you lets start"); implementation started.** Status Q&A → in progress.
- **Step 1 (code + local test): new `anyup/viz.py`** — `FeaturePCA` (exact covariance + `eigh` in float64; eigenvector sign fixed for determinism; colour scale p1/p99 per condition, fixed for all frames; projection one frame at a time on the feats' device), `flicker_map` (reuses `metrics._unit_frame`, float64), `colorize`, `denorm_guide`, `davis_palette` / `overlay` (+ object contours), `label_strip`, `grid`, `resize_rgb`, `write_mp4` (imageio-ffmpeg, libx264, yuv420p, padded to even size), `write_gif` (PIL).
  - Added beyond the design: `colorize`, `denorm_guide`, `grid`, `resize_rgb`, `FeaturePCA.sample_tokens` (small layout/sampling helpers used by the cells).
  - **`tests/test_viz.py`: 10 passed, 0 skipped, 2.97 s** — run **locally** on CPU (scratch venv: CPU torch, numpy, opencv-headless, imageio-ffmpeg; not Colab). Covers: flicker = 0 on identical frames, = 2 on f / −f, per-pair mean = 1 − `temporal_cosine.per_pair` (< 1e-9); PCA uint8 (T, H, W, 3), static frames identical, deterministic, per-condition scale, no per-frame rescaling; overlay background unchanged; GIF 6 frames; MP4 6 frames, odd 41×31 padded to 42×32.
- **Steps 2–5 (code): notebooks edited** (targeted JSON edits via a scratchpad script; cells verified against `git HEAD`: only Cells 1, 2 and the visual cell changed in each notebook; cell count and metadata unchanged; all code cells parse).
  - Cell 1 (all three): `REPO_BRANCH = "experiments-in-action-test"`.
  - Cell 2: visual knobs as in the Design table. Changed from the design: Exp 2 `VIZ_CLIP` removed (→ `VIZ_ITEMS`); Exp 1 + `PCA_SAMPLE`; Exp 3 `VISUAL_FPS` 4 → 8 and `VISUAL_SIZE` 224² squashed → height 240 with the aspect ratio kept.
  - Exp 3 Cell 6, Exp 2 Cell 6, Exp 1 Cell 7 replaced (sources in the scratchpad `t10/*.py`).
  - Changed from the design: Exp 1 Figure B blacks out unlabelled GT pixels; the Exp 3 "panel J&F = CSV" check dropped (same source, trivially equal); Exp 2 flicker check prints WARN instead of failing above 1e-6.
- **Local dry run of the three cells** (scratchpad `t10/harness.py`: stub upsamplers / backbone chunks, synthetic DAVIS folders and ADE tensors, real `anyup.viz` / `davis` / `metrics`; CPU). **Not a test of the real models: it only checks that the cells run and lay out correctly.**
  - Exp 3: in-memory mode and standalone mode (other run folder, CSV + `masks.zip` only, 99 zip members extracted for the 3 selected clips) both write `exp3_side_by_side.png`, 3 × MP4 + GIF, `visuals_selection.json`; selection = top-2 gains + worst, as designed.
  - Exp 2 (stub at the real 224² size): figure, curve, MP4 (908×522) + GIF per clip; flicker check |diff| ≤ 1.7e-16 on all 6 (clip, condition) pairs.
  - Exp 1: in-memory (PER_IMAGE + heads, `T4_` run → warning printed, Figure B written), standalone from a CSV written by the real `save_results` (NaN image dropped from the ranking; heads missing → Figure B skipped with the paths), and no scores at all (first val images).
  - Layout bugs found and fixed in the dry run: Exp 2 colourbar overlapped a panel column and shrank the flicker panels (→ own colourbar axis); a leftover junk line in the Exp 3 cell; Exp 2 cell missed `import cv2`; Exp 2 video title said what AnyUp3D *should* look like (→ neutral description).
- **Docs update (your request: "update the docs with the new functions and the modifications").** Scope extended (Scope updated).
  - `docs/experiments/build_docs.py`: `("anyup.viz", "anyup/viz.py")` added to `HELPER_FILES`, so the page gets a "Shared modules" card per viz helper.
  - `docs/experiments/doc_content.py`: provenance "+ T10 visual cell" (Exp 1–3); the step texts of Cell 2 (new knobs) and the visual cells (Exp 1 Cell 7, Exp 2 Cell 6, Exp 3 Cell 6) were rewritten for the new behaviour; an Overview paragraph on the visual checks was added; Exp 0 Cell 1 text corrected ("identical in all four notebooks" → except `REPO_BRANCH`).
  - `docs/context/project_context.md`: one line on the visual checks. `plan.md`: T10 addendum note under Immediate Next Step.
  - `build_docs.py --report` (dirty tree, commit `6eff213`): parses all cells; 13 viz helpers found and linked to their cells: `FeaturePCA.sample_tokens` / `fit` / `rgb` (exp1[7], exp2[6]), `flicker_map`, `colorize` (exp2[6]), `denorm_guide` (exp1[7], exp2[6]), `overlay` (exp3[6]), `label_strip`, `grid`, `write_mp4`, `write_gif` (exp2[6], exp3[6]), `resize_rgb` (exp1[7], exp3[6]). Index: 196 definitions in 10 files.
  - **`experiments.html` NOT rebuilt yet:** the build pins helper links to the HEAD commit and refuses on uncommitted `anyup/` / `experiments/` changes; `anyup/viz.py` doesn't exist at `6eff213`, so the build must run after your commit (then commit the HTML).
- **Waiting for your Colab runs** (step 6). Status stays in progress.
