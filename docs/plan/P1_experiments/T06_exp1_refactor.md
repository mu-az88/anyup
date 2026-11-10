# T06 — Refactor Exp 1 (ADE20K Linear Probe) onto the Shared Modules

| Field | Value |
|---|---|
| Task ID | T06 |
| Plan | P1 Experiments (`plan.md`), Phase A.2+ (prepares Phase B) |
| Branch | `experiments-task02` (continued, your choice) |
| Status | done (full A100 run skipped → S02) |
| Depends on | T01 (`dinov2` adapter), T03 (upsamplers), T04 (`update_conf_mat`, `compute_miou`, `paired_stats`, `save_results`) |
| Blocks | Phase B re-run of Exp 1 |

## Goal

Rebuild Exp 1 (ADE20K linear-probe mIoU, DINOv2, T=1, bilinear vs AnyUp2D vs AnyUp3D) as a thin notebook on `anyup.backbones`, `anyup.upsamplers`, `anyup.metrics` and a new `anyup.ade20k`. A temporary `LEGACY` flag proves the refactored path equals the old notebook code on the same inputs.

## Scope

- In scope:
  - New `experiments/exp1_ade20k_probe.ipynb` (new protocol + `LEGACY` flag + A100/T4 profiles).
  - New `anyup/ade20k.py` + `tests/test_ade20k.py`.
  - `Upsampler.forward_images` in `anyup/upsamplers.py` + tests in `tests/test_upsamplers.py`.
  - `image_miou` in `anyup/metrics.py` + tests in `tests/test_metrics.py`.
  - Qualitative figure as one optional final cell, using the trained heads.
- Out of scope: Exp 2/3; Exp 4; native-resolution scoring (Q3); the canonical checkpoint decision; changing `results/*.ipynb`; Phase B interpretation of the new numbers; changing `AnyUp` itself.

## Findings in the old notebook

- **A.** `train_loader` has `shuffle=False`; its generator is unused. The 3 seeds differ only in head init, so std 0.01–0.05 and the t-test p = 0.0001 say little.
- **B.** The figure (Cell 11) uses an untrained `make_head()`.
- **C.** `AnyUp3DUpsampler` treats a (T, C, h, w) chunk as consecutive frames, so a batch of unrelated images would be mixed by the temporal layers. The old code used (B, C, 1, h, w).
- **D.** Images are squashed to 224² and mIoU is computed at 224² on nearest-resized masks.
- **E.** The feature cache is HF DINOv2, fp16; it must be rebuilt for the timm wrapper. The final LayerNorm is in both paths (`last_hidden_state`; wrapper `norm=True`).
- **F.** `precompute_split` skips if the cache files exist, so an interrupted run leaves a partly-zero cache that looks complete.
- **G.** Old thesis result: bilinear 39.54 ± 0.05, AnyUp2D 42.36 ± 0.02, AnyUp3D 41.93 ± 0.01 (`step500`).
- **H. `window_t` has no effect in Exp 1.** It is used only in `compute_attention_mask_3d`, as `|t_q − t_k| ≤ window_t`; at T=1 that is always true. So `plan.md`'s "ran with `window_t=3`" is not a reason for a different result; the `step500` checkpoint is. A pytest confirms this (see Tests).

## Q&A (decisions agreed before implementation)

| # | Question | Answer |
|---|---|---|
| 1 | Seeds? | **Seeded shuffle per seed**: seeds vary head init and data order. For a given seed, all conditions get the same init and order, so comparisons stay paired. |
| 2 | Batched upsampling? | **Image-batch path in the wrapper** (`forward_images`) + pytest against per-image calls. |
| 3 | Resolution protocol? | **Keep 224² square, mIoU at 224².** |
| 4 | Feature cache? | **fp16 on Drive, new path** tagged with the timm name and input size. |
| 5 | ADE20K loading? | **`anyup/ade20k.py`** + pytest. |
| 6 | Hyperparameters? | **Keep all as config values**: 1×1 conv head, AdamW lr 1e-3, wd 1e-4, StepLR step 2 / γ 0.1, 3 epochs, batch 16, seeds 42 / 123 / 456. |
| 7 | Legacy regression? | **Subset check**, old functions copied into the notebook, both paths on the same inputs: upsampled features, about 50 train steps, about 100 val images. |
| 8 | Statistics? | **Seeds + per-image**: mean ± std and `paired_stats` over seeds (n=3, reported as weak) + a per-image paired test on val (n=2000). |
| 9 | AnyUp3D settings? | **step6500 as a config value, `window_t=4`, `q_chunk_size=4096`** at 224². |
| 10 | Resume + figure? | **Per-(condition, seed) head checkpoints on Drive with resume; figure uses the trained heads**, optional final cell, off by default. |
| 11 | GPU? | **A100 for the real run, with a bool that switches to T4 for testing.** Every GPU-dependent value lives in a dictionary with one key per GPU, and each usage carries a comment with the options, e.g. `# A100: 16, T4: 4`. |
| 12 | Branch? | **`experiments-task02`** (continued). |

### Confirmed with this file

| # | Point | Proposal |
|---|---|---|
| 13 | Q7 said "identical losses". CUDA 2D cross-entropy is not guaranteed deterministic, so exact equality may fail with no bug. | **Exact** for upsampled features (max_abs = 0). **Tolerance** for training: per-step loss \|diff\| < 1e-4, final mIoU \|diff\| < 0.01; actual values reported. |
| 14 | T4 profile values | As in the table below: shrinks batch and data, so T4 runs are tests only. Their numbers are never compared with A100 numbers; the run name carries the GPU. |
| 15 | Cache location during training | **Built locally, stored on Drive (Q4), copied to local `/content` once per session.** Memmap reads and writes on Drive are slow, especially with shuffled access (new with Q1). |
| 16 | Per-image score | Per val image: `compute_miou` on that image's confusion matrix (classes present in its GT), final epoch only, averaged over seeds, then `paired_stats(anyup3d, X)`. This is not the dataset mIoU; it's labelled as such. Images with no labelled pixels are dropped and counted. |

## Design

### Files

- New: `anyup/ade20k.py`, `tests/test_ade20k.py`, `experiments/exp1_ade20k_probe.ipynb` (generated by a scratchpad script, as in T05).
- Modified: `anyup/upsamplers.py`, `tests/test_upsamplers.py`, `anyup/metrics.py`, `tests/test_metrics.py`.
- No change to `anyup/backbones.py`, `anyup/model.py`, `results/*.ipynb`.

### `anyup/upsamplers.py` addition

```python
def forward_images(self, chunk: BackboneOutput, output_size=None) -> UpsampleOutput
    """Unrelated images: chunk.features (B, C, h, w), chunk.guide (B, 3, gH, gW). Never mixes images."""
```

- Same checks and meta as `forward`, plus `meta["image_batch"]=True`. Each class gets `_upsample_images`:
  - AnyUp3D: `(B, C, h, w)` → `(B, C, 1, h, w)`, guide → `(B, 3, 1, gH, gW)`, one call, `squeeze(2)`. Same call as the old `upsample_anyup3d`.
  - AnyUp2D: one batched call (as the old `upsample_anyup2d`).
  - Bilinear: same as `_upsample`.
- `forward` is unchanged.

### `anyup/ade20k.py`

```python
ADE_URL = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
NUM_CLASSES, IGNORE_INDEX = 150, 0

prepare_ade20k(dest="/content", drive_zip=None, url=ADE_URL) -> Path   # skips if extracted; prefers the Drive zip, as old Cell 4
list_pairs(root, split) -> List[Tuple[Path, Path]]                     # split "train" / "val"; paired by file stem, raises on a mismatch
load_image(path, size) -> Tensor                                       # (3, size, size) float [0, 1], backbones.resize_frames
load_mask(path, size) -> Tensor                                        # (size, size) long, nearest, as old Cell 4
class ADE20KImages(Dataset)                                            # → frames (3, S, S) [0, 1], mask (S, S)
build_feature_cache(backbone, pairs, cache_dir, split, batch_size, num_workers)   # feats fp16 + masks uint8 + meta JSON
cache_complete(cache_dir, split) -> bool                               # meta present and both files have the expected byte size
copy_cache(src, dst, split)                                            # Drive ↔ local, meta last
class CachedFeats(Dataset)                                             # → guide (3, S, S) ImageNet-normalized, feats (C, h, w) fp32, mask (S, S)
```

- Resize uses `resize_frames` (T01) in the dataset, because ADE20K images differ in size and must be equal to batch. The adapter's own resize is then a no-op.
- `build_feature_cache` writes `*.part` files and renames them only when complete, then writes the meta last (fixes F). `<split>_meta.json` holds backbone id (timm name), size, N, feature shape and a SHA-1 of the image stems; `CachedFeats` raises if they disagree.
- Files: `<split>_feats.f16`, `<split>_masks.u8`, `<split>_meta.json`. The cache folder name holds timm name, size and N_TRAIN / N_VAL, so test and legacy subsets never touch the full cache.
- Guides are decoded from the JPEG per item, as old (a 224² fp32 guide cache would be about 12 GB).

### `anyup/metrics.py` addition

```python
def image_miou(logits, targets, num_classes=150, ignore_index=0) -> List[float]
    """Per-image mIoU in %, (B, K, H, W) + (B, H, W) → B values: update_conf_mat + compute_miou per image; NaN if no labelled pixel."""
```

### Notebook layout

1. Clone `experiments-task02` (fetch + `checkout -B`), Drive, deps, restart guard.
2. **Config:** `LEGACY`, `USE_T4`, `GPU_PROFILES`, paths, hyperparameters.
3. Imports, `set_seed`, `prepare_ade20k`, DINOv2, `build_feature_cache` (skips if complete), copy cache local, upsamplers.
4. `LEGACY` only: old functions (verbatim copy).
5. Train / eval functions, head checkpoints with resume.
6. Main loop: condition → seed → epoch.
7. `LEGACY` only: comparison cell, PASS/FAIL.
8. Summary, `paired_stats`, `save_results` (`exp1_per_item.csv`, `exp1_summary.json`, legacy checks included).
9. Optional figure (`RUN_VISUALS=False`).

### GPU profiles (Q11, Q14)

```python
USE_T4 = False   # ← True = T4 test run (small data, numbers not for reporting)
GPU = "T4" if USE_T4 else "A100"
GPU_PROFILES = {
    "A100": dict(BATCH_SIZE=16, Q_CHUNK=4096, N_TRAIN=None, N_VAL=None, NUM_EPOCHS=3, SEEDS=[42, 123, 456], NUM_WORKERS=4),
    "T4":   dict(BATCH_SIZE=4,  Q_CHUNK=2048, N_TRAIN=256,  N_VAL=64,   NUM_EPOCHS=1, SEEDS=[42],           NUM_WORKERS=2),
}
P = GPU_PROFILES[GPU]
BATCH_SIZE = P["BATCH_SIZE"]   # A100: 16, T4: 4 — ↓ reduce if OOM. Dependent: Q_CHUNK, IMG_SIZE
```

- Each value gets its own line like the last one, with the options and dependents in the comment. The same goes for later usages in the notebook.
- The cache covers exactly the `N_TRAIN` / `N_VAL` images of the run, in its own folder.
- The notebook prints the detected GPU and warns if it doesn't match `GPU`.

### New protocol

| Item | Value |
|---|---|
| Data | ADE20K train 20,210 / val 2,000, 224² square |
| Backbone | `dinov2` base (timm wrapper), input 224, feature (768, 16, 16), `guide_size=224`; zero-shot for AnyUp3D |
| Conditions | `bilinear`, `anyup2d`, `anyup3d` via `forward_images` |
| Head | `nn.Conv2d(768, 150, 1)` on 224² upsampled features |
| Training | Q6 values; `shuffle=True` with `torch.Generator().manual_seed(seed)` |
| Metric | `update_conf_mat` + `compute_miou` each epoch; `image_miou` at the final epoch |
| Rows | One per (image, condition, seed): `image_miou`. Dataset mIoU per (condition, seed, epoch) in the summary JSON |
| Summary | Mean ± std over seeds; `paired_stats(anyup3d, X)` over seeds and over images, X in bilinear / anyup2d |

- `set_seed(seed)` runs before `make_head()`, so a seed gives every condition the same init.
- The head checkpoint holds head, optimizer, scheduler, epoch, mIoU, generator state and, when complete, the per-image scores. So resume continues the same shuffle sequence, and a finished (condition, seed) needs no re-evaluation.
- Checkpoints live under `OUTPUT_DIR/<RUN_NAME>/head_ckpts/`; `RUN_NAME` holds the mode, GPU, checkpoint step and sizes, so runs can't resume each other.

### `LEGACY = True`

| Item | Value |
|---|---|
| Inputs | The same `CachedFeats` batches into both paths (T05 Q20). Image preprocessing and HF vs timm features are therefore not covered by the check; both are deliberate changes. |
| Conditions | All three, seed 42, `shuffle=False` (as old) |
| Subset | First `50 × BATCH_SIZE` train images (50 steps), first 100 val images, 1 epoch |
| Old code (verbatim copy) | `upsample_bilinear` / `upsample_anyup2d` / `upsample_anyup3d`, `make_head`, `set_seed`, `train_one_epoch`, `evaluate_miou` |
| Deviations from verbatim | Models come from the wrappers (the same objects); `q_chunk_size=Q_CHUNK` added to the old AnyUp3D call (as T05 Q17); `train_one_epoch` returns per-step losses and `evaluate_miou` also returns the conf-mat (needed for the checks) |

Checks:
- Upsampled features new = old, **max_abs = 0**, per condition, on the first train batch.
- Same head init: all head tensors equal before training.
- Per-step losses \|diff\| < 1e-4 and final mIoU \|diff\| < 0.01 (Q13); actual values reported.
- Conf-mat equality and trained-head max_abs reported as INFO (argmax can flip on float noise).

### Compute-cost lines (commented with options and dependents)

- `BATCH_SIZE`: images per upsampler + head step. Upsampled batch = B × 768 × 224² × 4 B (2.5 GB at 16), plus logits B × 150 × 224² × 4 B. Dependent: `Q_CHUNK`, `IMG_SIZE`. Changing it changes the protocol (Q6).
- `IMG_SIZE` (224): input, guide, output and mask size; a multiple of 14. Dependent: `FEAT_SIZE`, `BATCH_SIZE`, `Q_CHUNK`, cache path.
- `Q_CHUNK`: AnyUp3D query chunk → attention memory. Dependent: `BATCH_SIZE`, `IMG_SIZE`.
- `N_TRAIN`, `N_VAL`, `NUM_EPOCHS`, `SEEDS`: total upsampler calls. Dependent: run time, `LR_DECAY_STEP`.
- `NUM_WORKERS`: JPEG decode speed; no effect on results.

## Steps

1. `forward_images` + tests.
2. `image_miou` + tests.
3. `anyup/ade20k.py` + tests. You run pytest on Colab.
4. `experiments/exp1_ade20k_probe.ipynb`.
5. You run `LEGACY=True` with `USE_T4=True`, then `LEGACY=False` with `USE_T4=True` (smoke), and paste the outputs.
6. You run the full run on A100 (`USE_T4=False`) and paste the outputs.
7. Update `plan.md` and this file. Close out.

## Tests / Regression Checks

pytest (CPU, synthetic):
- `forward_images`: for each upsampler, output equals per-image `forward` at T=1 (rel 1e-4); changing image j leaves image i unchanged (proves no mixing); shape and meta; `forward` unchanged (existing tests).
- `window_t=3` and `window_t=4` give equal outputs at T=1 (finding H).
- `image_miou`: perfect → 100; equals `update_conf_mat` + `compute_miou` per image; only-ignore image → NaN.
- `ade20k`: on a tiny fake dataset, pairing by stem, mismatch raises, mask ids preserved (nearest), image range and shape, cache round trip (fp16 tolerance), an unfinished `.part` isn't treated as complete, meta mismatch raises, guide = `normalize_frames(resize_frames(...))`.

Colab:
- `LEGACY=True` checks as in Design.
- `LEGACY=False`, `USE_T4=True`: end to end, including one resume (re-run the main cell: everything skips, same results); rows and JSON written.
- Full A100 run: INFO only (Phase B interprets).

## Done When

- All pytest pass on Colab.
- The notebook runs end to end in both modes; legacy checks pass.
- Full results are on Drive (per-item CSV + summary JSON), with numbers in the Implementation Log. _(Skipped by Monzir → `docs/skipped_and_future_concerns.md` S02; the Colab resume check → S01.)_

## Implementation Log

_Updated live during implementation. Per step: what changed, files touched, test results (actual numbers/shapes). Also log mid-implementation questions and answers, changed decisions (old → new, why), and anything added or removed. Sections above are edited in place to match, so the finished file is a report of what was actually done._

- **OK received ("lets start"); Q13–16 accepted as proposed.** Status Q&A → in progress.
- **Step 1 (code):** `anyup/upsamplers.py`: `forward` body moved to `_run(chunk, output_size, upsample_fn)` (same lines, `no_grad` moved with it); new `forward_images` + `_upsample_images` per class (AnyUp3D `(B, C, 1, h, w)`, AnyUp2D one batched call, bilinear = `_upsample`). `tests/test_upsamplers.py`: + equals per-image ×3, never mixes images ×3, differs from `forward` for AnyUp3D, mismatched batch raises, `window_t` 3 = 4 at T=1.
- **Step 2 (code):** `anyup/metrics.py`: + `image_miou`. `tests/test_metrics.py`: + 2 tests.
- **Step 3 (code):** new `anyup/ade20k.py`, `tests/test_ade20k.py` (12 tests).
  - Changed from the design: cache is built on local disk and copied to Drive (old → new: was "build on Drive"; why: 8 GB memmap writes through the Drive mount are slow). Added `cache_complete`, `copy_cache`; the meta also holds a SHA-1 of the stems; files renamed `.f16` / `.u8` (raw memmaps, not `.npy`).
- **Step 4 (code): new `experiments/exp1_ade20k_probe.ipynb`** (10 cells; generated by scratchpad `make_exp1_nb.py`).
  - Changed from the design: the legacy comparison is Cell 7 and the summary Cell 8 (old → new: swapped; why: the legacy checks are saved in the summary JSON, as T05 Exp 3). Old functions: per-step losses and conf-mat added (Design table updated).
  - Added: per-epoch train / val time and peak GPU memory printed, and time stored in the summary, because neither the A100 runtime nor batch-16 memory is known.
- **Local CPU check (not in the original plan; scratchpad venv, Python 3.14, torch 2.15 CPU; Colab is Python 3.13):**
  - pytest `test_ade20k.py` + `test_upsamplers.py` + `test_metrics.py`: **76 passed, 0 failed, 1 warning** (the known 0/0 in `db_eval_iou`), 11.8 s. 23 of these are new (9 upsamplers, 2 metrics, 12 ade20k). The `window_t` 3 = 4 at T=1 test passes (random weights), so finding H holds, and `plan.md` is corrected.
  - Notebook Cells 2–9 dry run on CPU with a scratchpad harness: 12 fake images per split at 56², fake 16-channel backbone, random-init AnyUp weights, AnyUp2D from a local file; both modes, `RUN_VISUALS=True`. Not covered: Cell 1, `prepare_ade20k` download / unzip, the real DINOv2, GPU memory, real checkpoints, Drive.
  - `LEGACY=True`: 14 checks, 0 failed; upsampled max_abs 0, loss |diff| 0, conf-mat 0 cells differ, head max_abs ≤ 1.5e-8. On CPU and noise data, so this checks the logic only. CUDA behavior (Q13) needs the Colab run.
  - `LEGACY=False`: re-running the main cell skips everything with the same results; stopping after epoch 1 and resuming gives the same final mIoU as the uninterrupted run (exact, including the shuffle order). CSV has 24 rows = 4 images × 3 conditions × 2 seeds. mIoU ≈ 0 as expected on random labels.
  - **Bug found and fixed:** Cell 8 crashed formatting a `None` p-value when all pairs tie; added `fmt_p`. Re-run passes.
- **pytest run 1 (Colab, Python 3.13.16, pytest 8.4.2): `test_ade20k.py` + `test_upsamplers.py` + `test_metrics.py` 76 passed, 0 failed, 1 warning, 15.53 s.** Same result as the local run; the warning is the known 0/0 in `db_eval_iou`. Steps 1–3 done.
- **Exp 1 `LEGACY=True` run 1 (`USE_T4=True`, batch 4, `Q_CHUNK` 2048, 200 train / 100 val images, 50 steps, seed 42; real DINOv2 and step6500): 14 checks, 0 failed.** Saved to Drive `results/T06/exp1/legacy_T4_step6500_s224_b4_q2048_tr200_va100_e1/`. GPU type, commit, time and peak memory not in the pasted output.
  - **Upsampled new = old, max_abs = 0**, shape (4, 768, 224, 224), × bilinear / anyup2d / anyup3d (first train batch). Same head init (2 tensors each). The old and new paths use the same model objects.

  | condition | loss max \|diff\| (50 steps) | mIoU new | mIoU old | mIoU \|diff\| | conf-mat cells differing | head max_abs |
  |---|---|---|---|---|---|---|
  | bilinear | 4.77e-7 | 6.8880 | 6.8880 | 0 | 0 | 0 |
  | anyup2d | 2.03e-6 | 7.2969 | 7.2968 | 3.13e-5 | 26 | 8.21e-5 |
  | anyup3d | 2.38e-6 | 7.1286 | 7.1286 | 1.04e-5 | 12 | 3.23e-6 |

  - Q13 was needed: losses and heads are not bit-identical on CUDA, though all differences are far inside tolerance (loss 1e-4, mIoU 0.01). Bilinear heads are exact, while the AnyUp heads drift slightly; the cause is not established (float noise is the likely one, and only the first batch's features were compared).
  - INFO, 50-step heads, not results: dataset mIoU bilinear 6.89 / anyup2d 7.30 / anyup3d 7.13; per-image mIoU 26.33 / 28.28 / 27.42 (100 images, 0 dropped). The per-image score is higher because it averages only over the classes in each image, so the two are not comparable.
  - So on the same cached inputs, the refactored Exp 1 path reproduces the old code: upsampler calls, head, training step, scoring. Image preprocessing and HF vs timm features are outside the check (deliberate changes). Legacy item of Done When met.
- **Exp 1 `LEGACY=False` smoke run 1 (`USE_T4=True`: batch 4, `Q_CHUNK` 2048, 256 train / 64 val images, 1 epoch, seed 42, seeded shuffle): main loop 71 s, runs end to end.** Only the Cell 6 output was pasted (not the Cell 6 re-run or Cell 8).

  | condition | loss | mIoU | train (256 img) | val (64 img) | peak GPU memory |
  |---|---|---|---|---|---|
  | bilinear | 1.4673 | 5.97 | 7 s | 2 s | 1.0 GiB |
  | anyup2d | 1.3905 | 6.37 | 24 s | 5 s | 2.2 GiB |
  | anyup3d | 1.4070 | 6.10 | 26 s | 6 s | 1.7 GiB |

  - mIoU values are from 64 steps on 256 images: INFO only.
  - Memory at batch 4 ≤ 2.2 GiB (AnyUp2D the highest, as it has no query chunking). My estimate, not measured: batch 16 is about 4× the activations, ≈ 9 GiB, so it should fit on an A100.
  - Time at batch 4 on T4 ≈ 0.1 s per image for AnyUp2D / AnyUp3D. Extrapolated (not measured): full protocol = 182k train + 18k val image passes per condition ≈ 5.5 h per AnyUp condition on a T4; the A100 time is unknown until its first epoch prints.
- **Skipped by Monzir (2026-11-08):** the Colab resume check + Cell 8 output of the smoke run (→ S01) and the full A100 run (→ S02), both recorded in `docs/skipped_and_future_concerns.md`.
- **Close-out.** Done When met except the full results (S02): pytest 76 passed on Colab; the notebook ran end to end in both modes on T4; legacy checks 14/14. Status → done (full run skipped); `plan.md` updated.
  - Left over: S01, S02; the cause of the small AnyUp head drift in the legacy check (float noise is likely, not verified); GPU type and commit of the Colab runs not in the pasted outputs; the `LEGACY` flag stays until Phase B is accepted.
