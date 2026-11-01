# T04 — Shared DAVIS Utilities + Metrics Module

| Field | Value |
|---|---|
| Task ID | T04 |
| Plan | P1 Experiments (`plan.md`), Phases A.3 + A.4 (+ part of A.5: results I/O) |
| Branch | `experiments-task02` (continued, your choice) |
| Status | in progress |
| Depends on | T01 (`BackboneOutput`), T02 (`FrameFolderSource`, `chunk_boundaries`), T03 (upsamplers, used only in the Colab check) |
| Blocks | T05 notebook refactor (Exp 1–3 on the shared modules) |

## Goal

Define every DAVIS data step and every score used by Exp 1–3 **once**, in two modules, so the refactored experiments (T05) are reproducible and match the official DAVIS protocol that reviewers expect.

## Scope

- In scope:
  - `anyup/davis.py`: download, clip list, paths, annotations, object one-hot, resizing, palette-PNG saving, label propagation (Exp 3 method).
  - `anyup/metrics.py`: official DAVIS J&F (vendored), temporal cosine, GAP cosine fidelity, ADE20K mIoU, paired statistics, results I/O.
  - `requirements.txt`: add `scipy`, `scikit-image`.
- Out of scope: changing the experiment notebooks (T05); ADE20K data loading (stays in Exp 1 until T05 decides); SAM 2; the canonical checkpoint question; seeds/config handling beyond what results I/O records (rest of A.5).

## Findings from reading the old notebooks (inputs to the Q&A)

- **Exp 3 scores frame 0** (`evaluate_condition` loops over all `idxs`, t=0 is the GT mask), so it scores neither as the thesis says (drop frame 0) nor as the official protocol does.
- **Exp 3 𝓕 is not the official boundary metric**: it finds boundaries by erosion and matches with an ellipse, where the official code uses `_seg2bmap` + `disk(bound_pix)` dilation. The `bound_th=0.008` tolerance is the same.
- **Exp 3 aggregation ≠ official**: per frame, mean over objects, then pooled frames. Official: per object, mean over its frames, then mean over all objects of all clips.
- **Official semi-supervised scoring drops the first and the last frame** (`davis2017/evaluation.py`: `all_gt_masks[:, 1:-1]`).
- **Sign-test p ≈ 9.3×10⁻¹⁰ = 0.5³⁰**, i.e. a one-sided sign test at 30/30. At 27/30 the one-sided p is ≈ 4.2×10⁻⁶. So that p cannot belong to the Exp 3 result as reported (open question, plan.md).
- Cosine fidelity (VideoMAE viz notebook): per-frame cosine between the global-average-pooled LR features and the GAP of the HR output. Its use as the source of 0.9899 → 0.9913 is still unconfirmed.
- Exp 1 qualitative figure (Cell 11) uses a fresh untrained head (`make_head()`), not the trained one. Fix in T05.
- Exp 2 Cell 6 uses an undefined `LOCAL_OUT` (crashes). Exp 3 loads AnyUp3D with `strict=False`. Both go away in T05 via T03's loader.

## Q&A (decisions agreed before implementation)

| # | Question | Answer |
|---|---|---|
| 1 | Plan order? | **T04 = DAVIS utils + metrics (A.3 + A.4); T05 = notebook refactor.** (Changed from "T04 = refactor": the refactor needs both.) |
| 2 | J&F source? | **Vendor the official code** (`db_eval_iou`, `db_eval_boundary`, `f_measure`, `_seg2bmap`, `db_statistics` from `davisvideochallenge/davis2017-evaluation`, BSD-3, header with attribution + commit). Colab compares it with the installed package. |
| 3 | Which frames are scored? | **Official: drop first and last frame** (`1:-1`). Differs from the thesis text (frame 0 only); logged in plan.md as an expected Phase B difference. |
| 4 | J&F aggregation? | **Official per-object**: per object, mean over its frames; J-Mean / F-Mean over all objects of all clips; J&F = (J + F) / 2; also Recall and Decay (`db_statistics`). Per-clip (mean over that clip's objects) and per-object values are saved for the paired tests. |
| 5 | Temporal cosine? | **Keep the Exp 2 definition** (per-pixel cosine between adjacent output frames, mean over pixels and pairs), **in float64**. Returns `all` (headline) plus `within` and `cross` chunk-boundary means (boundaries from `chunk_boundaries`). |
| 6 | Cosine fidelity? | **GAP definition from the viz notebook, float64**: per frame, `cos(GAP(lr_t), GAP(hr_t))`; returns per-frame values + mean. |
| 7 | Other metrics? | Your answer: "any related score that is not repeated in another experiment." **My reading, please confirm:** add the scores used by any experiment, each once: **ADE20K mIoU** (Exp 1, verbatim), **paired statistics**, and **results I/O**. |
| 8 | DAVIS utils scope? | Your answer: "everything that makes the experiment easier to reproduce and show in papers." **My reading, please confirm:** download helper, clip list + paths, annotations (+ one-hot, resize), palette-PNG saving at native res, label propagation. |
| 9 | Module layout / branch? | **`anyup/davis.py` + `anyup/metrics.py`, branch `experiments-task02`.** Label propagation goes in `davis.py` (it's the DAVIS VOS protocol). |
| 10 | Test sidedness? | **Two-sided default; one-sided (H1: ours > baseline) also returned.** |
| 11 | New deps? | **Add `scipy` and `scikit-image` to `requirements.txt`** (both preinstalled on Colab). Vendored code stays verbatim. |

## Design

### Files

- New: `anyup/davis.py`, `anyup/metrics.py`.
- New: `tests/test_davis.py`, `tests/test_metrics.py` (pytest, CPU, synthetic data, no downloads).
- New: `tests/T04_davis_metrics_check.ipynb` (Colab; real DAVIS, official package, regression vs Exp 2/3 code).
- Modified: `requirements.txt` (+ `scipy`, `scikit-image`).
- No change to `anyup/backbones.py`, `anyup/video.py`, `anyup/upsamplers.py`, or the `results/` notebooks.

### `anyup/davis.py`

```python
DAVIS_URL = "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"

download_davis(dest="/content", url=DAVIS_URL) -> Path        # skips if <dest>/DAVIS/ImageSets exists; zip in its own dir
list_clips(root, split="val") -> List[str]                    # ImageSets/2017/<split>.txt (val = 30 clips)
frames_dir(root, clip, res="480p") -> Path                    # JPEGImages/480p/<clip>
ann_dir(root, clip, res="480p") -> Path                       # Annotations/480p/<clip>
open_clip(root, clip) -> FrameFolderSource                    # T02 reader; frames (N, 3, H, W) [0, 1]
load_annotation(path) -> np.ndarray                           # palette PNG → (H, W) uint8 object ids (0 = bg)
load_annotations(root, clip) -> np.ndarray                    # (N, H, W) uint8, sorted like the frames
num_objects(ann0) -> int                                      # int(ann0.max()), as the official toolkit and Exp 3
one_hot_objects(masks, n_obj) -> np.ndarray                   # (N, H, W) → (n_obj, N, H, W) uint8, official layout
resize_labels(mask, size) -> np.ndarray                       # nearest (cv2.INTER_NEAREST), as Exp 3
save_mask_png(mask, ref_ann_path, out_path)                   # resize to native res (nearest) + reference palette (Exp 3)
propagate_labels(feats, ann0, tau=0.07, top_k=7, n_last=6)    # Exp 3 propagate_clip, see below
```

- `propagate_labels`: `feats (T, C, H, W)` (upsampled, raw), `ann0 (H0, W0)` uint8 → `(T, H, W)` int64. It is Exp 3's `propagate_clip` verbatim (L2-norm per location; context = frame 0 + last `n_last` frames; affinity / τ; top-k → −inf; softmax; label transfer; argmax), with constants as kwargs and computed on `feats.device`.
  - Compute-cost lines: the affinity is `(H·W) × ((1 + n_last)·H·W)` floats per frame. At 56² that's 3136 × 21952 ≈ 69M floats ≈ 275 MB. Dependent: the upsampler `output_size` (T03) and `n_last`; reduce one if the affinity OOMs.
  - Frame 0's prediction is the GT one-hot (as in Exp 3), so it must never be scored. The official `1:-1` drops it.

### `anyup/metrics.py`

```python
# --- vendored official DAVIS (BSD-3) ---
db_eval_iou(annotation, segmentation, void_pixels=None)
db_eval_boundary(annotation, segmentation, void_pixels=None, bound_th=0.008)
f_measure(...), _seg2bmap(...), db_statistics(per_frame_values)   # M (mean), R (recall > 0.5), D (decay)

# --- DAVIS J&F wrappers ---
eval_clip_jf(gt (N, H, W) uint8, pred (N, H, W) uint8) -> dict    # one-hot by num_objects(gt[0]); slice 1:-1;
                                                                  # per object: J[f], F[f] arrays + M/R/D
summarize_jf(per_clip: Dict[clip, dict]) -> dict                  # J/F M/R/D over all objects, J&F, per_clip, per_object

# --- feature metrics (float64) ---
temporal_cosine(feats (T, C, H, W), boundaries=None) -> dict      # all, within, cross, per_pair (T-1,)
gap_cosine_fidelity(lr (T, C, h, w), hr (T, C, H, W)) -> dict     # per_frame (T,), mean

# --- ADE20K mIoU (Exp 1, verbatim) ---
update_conf_mat(conf_mat, logits, targets, num_classes=150, ignore_index=0)
compute_miou(conf_mat) -> float                                    # %, classes present in GT only

# --- paired statistics ---
paired_stats(ours, base) -> dict   # n, n_improved, n_worse, n_tied, mean_delta, median_delta,
                                   # sign test (exact binomial, ties dropped) p two-sided + one-sided,
                                   # Wilcoxon signed-rank p two-sided + one-sided, paired t p two-sided

# --- results I/O ---
save_results(out_dir, rows, summary, config) -> (csv_path, json_path)   # per-clip CSV + summary JSON
                                   # (summary + config + env: torch/cuda versions, GPU, git commit)
```

- `temporal_cosine` works per pixel in float64: `F.normalize(feats.double(), dim=1)`, then the dot along C, so the T03 float32 > 1 problem can't happen. Memory: one float64 copy of the output (T·C·H·W·8 bytes, e.g. 41·768·224²·8 ≈ 12.6 GB). It is processed one pair at a time to keep peak memory at ~2 frames (dependent: `output_size`).
- `boundaries` are the positions from `anyup.video.chunk_boundaries(metas)`. The pair `(p−1, p)` is `cross` if `p` is in it.
- The vendored J&F works on native 480p masks (official). Predictions are resized to native with nearest (`save_mask_png` / `resize_labels`) before scoring.

## Steps

1. `requirements.txt` + `anyup/metrics.py` (vendored DAVIS code first, then wrappers, feature metrics, mIoU, stats, I/O).
2. `anyup/davis.py`.
3. pytest `tests/test_metrics.py`, `tests/test_davis.py`; you run them on Colab and paste the output.
4. Colab notebook `tests/T04_davis_metrics_check.ipynb` (clones `experiments-task02`, fetch + checkout); you run it and paste the summary.
5. Update `plan.md` and this file. Close out.

## Tests / Regression Checks

pytest (CPU, synthetic):
- J/F: identical masks → 1.0; disjoint → 0.0; empty GT + empty pred → 1.0 (official convention); `1:-1` slicing (N=5 → 3 frames scored); per-object aggregation on a hand-computed 2-object example; multi-object one-hot layout.
- `db_statistics` on a hand-computed array (M, R, D).
- `temporal_cosine`: constant video → 1.0; identical frames vs ± flips → known values; within/cross split with `boundaries=[2]` on T=4; float64 result ≤ 1 for large identical tensors.
- `gap_cosine_fidelity`: hr = bilinear-upsampled lr → ≈ 1; known toy values.
- mIoU: equals Exp 1's `update_conf_mat`/`compute_miou` (copied into the test) on random logits/targets (exact).
- `paired_stats`: 30/30 one-sided sign p = 0.5³⁰ = 9.31×10⁻¹⁰; 27/30 one-sided ≈ 4.22×10⁻⁶; ties dropped; Wilcoxon / t equal `scipy.stats` directly.
- `propagate_labels` equals Exp 3's `propagate_clip` (copied into the test) on random features (exact, `torch.equal`).
- davis: `list_clips` / paths / `load_annotations` / `save_mask_png` round trip (palette kept, native size) on a tiny fake DAVIS tree in `tmp_path`.
- `save_results` writes CSV + JSON that read back equal.

Colab (real DAVIS from ETH, official package installed from GitHub):
- **Vendored = official package (exact):** `db_eval_iou` / `db_eval_boundary` from both on real GT masks vs perturbed masks (shifted, eroded, a dropped object), max_abs = 0.
- **Full official evaluation (exact):** bilinear Swin-T per-frame (T01 adapter, Exp 3 setting) → `propagate_labels` at `PROP_SIZE=56` on all 30 val clips, all frames → palette PNGs → the package's `DAVISEvaluation(task="semi-supervised")` vs our `summarize_jf` on the same PNGs: J-Mean, F-Mean, J&F, per-object J/F equal (|diff| < 1e-12). Bilinear only, so no checkpoint is needed.
- **Old vs official metric (INFO):** Exp 3's `compute_J` / `compute_F` + frame-pooled aggregation (incl. frame 0) on the same masks, to quantify the protocol change for the paper.
- **Propagation = Exp 3 `propagate_clip` (exact)** on `bear` (same bilinear features).
- **Temporal cosine (INFO):** float64 vs Exp 2 `adjacent_cosine_sim` (float32) on the same AnyUp2D VideoMAE `bear` output; AnyUp3D (112², `q_chunk_size=4096`, T4-safe) within / cross boundary split.
- **Fidelity (INFO):** GAP cosine for bilinear / AnyUp2D / AnyUp3D on `bear`.
- Summary saved to Drive (`results/T04_davis_metrics_check.json`: env, commit, every check).

## Done When

- `anyup/davis.py`, `anyup/metrics.py` exist with the API above; `requirements.txt` updated.
- pytest passes (Colab).
- Colab: vendored metric = official package (exact), full evaluation = `DAVISEvaluation` (exact), propagation = Exp 3 (exact).
- `plan.md` updated: Task Index; the expected Phase B differences (official `1:-1` + per-object J&F instead of frame 0 included + frame-pooled; official 𝓕 boundary); sign-test finding added to Open Questions.

## Implementation Log

_Updated live during implementation. Per step: what changed, files touched, test results (actual numbers/shapes). Also log mid-implementation questions and answers, changed decisions (old → new, why), and anything added or removed. Sections above are edited in place to match, so the finished file is a report of what was actually done._

- **OK received (Q7/Q8 readings confirmed by "yes, lets go").** Status Q&A → in progress.
- **Step 1 (code): `requirements.txt` + new `anyup/metrics.py`.**
  - Official source fetched from `davis2017-evaluation` master = commit `ac7c43fca936f9722837b7fbd337d284ba37004b`. Vendored `db_eval_iou`, `db_eval_boundary`, `f_measure`, `_seg2bmap` (`davis2017/metrics.py`) and `db_statistics` (`davis2017/utils.py`) with the BSD-3 text. **Checked byte-identical to the source** (substring check of each block, script in scratchpad).
  - Read the official pipeline (`evaluation.py`, `davis.py`, `results.py`, `evaluation_method.py`). `eval_clip_jf` replicates it: GT 255 (void) → 0, and semi-supervised passes no void mask; `n_objects = max(gt[0])`; slice `1:-1`; pred one-hot over the scored frames with `n_pred = max(pred[1:-1])`, zero-padded to `n_objects` (pred id > n_objects → `ValueError`, where the official code calls `sys.exit`); J/F per object per frame; `db_statistics` per object. `summarize_jf` = means over the flat per-object lists, J&F = (J-Mean + F-Mean)/2, as `evaluation_method.py`.
  - Added guards (no effect on valid inputs): N < 3 frames → error; > 255 scored frames → error (the vendored `db_statistics` casts ids to `uint8` and would overflow silently; DAVIS val clips are ≤ 104 frames).
  - `temporal_cosine` (float64, one pair at a time, `F.normalize` along C as Exp 2), `gap_cosine_fidelity` (float64 `F.cosine_similarity` of GAP vectors, as the viz notebook), `update_conf_mat` / `compute_miou` (Exp 1 verbatim, constants → kwargs), `paired_stats` (`scipy.stats.binomtest` / `wilcoxon` / `ttest_rel`), `environment()` + `save_results` (CSV per item + JSON {summary, config, env}; unknown types → `TypeError`, not silently stringified).
  - **Added beyond the design:** `opencv-python` in `requirements.txt` (cv2 is imported by both modules and wasn't listed; preinstalled on Colab).
  - `py_compile` passes; not run (no numpy/torch locally).
- **Step 2 (code): new `anyup/davis.py`.** `download_davis` (urllib + zipfile, zip kept in `<dest>/davis_zip/`), `list_clips`, `frames_dir` / `ann_dir` / `frame_paths` / `ann_paths`, `open_clip` (T02 `FrameFolderSource`), `load_annotation(s)` (count must equal the frame count), `num_objects`, `one_hot_objects`, `resize_labels`, `save_mask_png` (Exp 3, + error if id > 254), `propagate_labels` (Exp 3 `propagate_clip` + `build_label_maps_frame0`, constants `TAU`/`TOP_K`/`N_LAST_FRAMES` as module defaults and kwargs).
  - Differences from Exp 3 (none change the values on DAVIS inputs): runs on `feats.device` (Exp 3: CPU), returns CPU; `reshape` instead of `view`; **`ann0` containing void (255) → `ValueError`** (Exp 3's `ann0.max()` would create 255 object channels silently); `num_objects` ignores 255, as the official loader.
  - **Note for T05:** Exp 3 cached the upsampled features as **float16** (`torch.save(out.half())`) and propagated the float16-rounded values. The shared path propagates float32 directly, so Exp 3 masks can differ slightly even with the same features.
  - `py_compile` passes; not run.
- **Step 3 (tests): new `tests/test_metrics.py` (24 tests) and `tests/test_davis.py` (14 tests, 15 cases with parametrization).** Synthetic data, CPU, no downloads.
  - metrics: J/F identical → 1, empty pred → 0, empty GT + empty pred → 1, frames 0 and N−1 not scored, hand-computed per-object aggregation (J_M [1.0, 0.5] + a second clip with J 0 → J-Mean 0.5, per-clip 0.75 / 0), void → bg + pred zero-padding, pred id > n_objects → error, input errors, one-hot layout, `db_statistics` by hand (M 0.52, R 0.6, D 0.7); temporal cosine (constant → 1, ± flips → [1, −1, 1] with within 1 / cross −1, = Exp 2 `adjacent_cosine_sim` within 1e-6, 768×64² identical frames → |v−1| < 1e-12, errors); GAP fidelity (nearest ×4 → 1, negated → −1); mIoU = Exp 1 functions (`torch.equal` + equal value), perfect → 100; sign test 30/30 one-sided = 0.5³⁰, 27/30 one-sided = 4526/2³⁰ (4.22e-6) and two-sided = 2×; Wilcoxon / t / binomial = scipy with 2 ties; all tied → None; `save_results` round trip + `TypeError` on unknown types.
  - davis: fake DAVIS tree (2 clips × 5 frames, 32×48, palette PNGs, 2 objects): split-file order, sorted paths, `open_clip` shape, `download_davis` skips when present (invalid URL never touched), `load_annotations` + count mismatch error, `num_objects` ignores 255, one-hot = metrics layout, `resize_labels` = cv2 nearest, `save_mask_png` round trip (palette kept, native size) + id 300 error; **`propagate_labels` = Exp 3 `propagate_clip` (verbatim copy in the test), `torch.equal`, T = 2 and T = 9 (sliding context)**, frame 0 = resized GT, `top_k` changes the result, void in `ann0` → error.
  - `py_compile` passes; waiting for the Colab run.
