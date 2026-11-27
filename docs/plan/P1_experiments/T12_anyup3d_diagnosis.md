# T12 — AnyUp3D Diagnosis at a Usable Resolution (Exp 3, Problem 2)

| Field | Value |
|---|---|
| Task ID | T12 |
| Plan | P1 Experiments (`plan.md`), addendum after P1 close (diagnostics for Phase B) |
| Branch | `experiments-in-action-test` (same as T10 / T11, Monzir's choice) |
| Status | in progress |
| Depends on | T11 (Test E setup, `propagate_e`, run 5 numbers), T04 (`davis`, `metrics`), T10 (`anyup.viz`) |
| Blocks | Any decision on the AnyUp3D fix (evaluation change vs retraining, `exp3_debug_findings.md` Problem 2) |

## Goal

Find out why AnyUp3D is below AnyUp2D (and bilinear) at a usable resolution (T11 run 5, DINOv2-B 448² → 112²: AnyUp2D 0.763 vs AnyUp3D 0.562, 11 % lost object-frames). One notebook with one cell per hypothesis and visuals, so Monzir can run it and we decide from the results.

## Facts found in the code before the Q&A (change `exp3_debug_findings.md` Problem 2)

What AnyUp3D saw in training (`Train_3D_full.ipynb` `TrainConfig`, checkpoint step 6500; `t_schedule` confirmed from the checkpoint in the T03 log):

- **Temporal context:** steps 0–5000 used T = 2 video frames = **1 feature frame** (VideoMAE tubelet 2), steps 5000–6500 used T = 4 = **2 feature frames**; the T = 8 stage starts at step 15000. So at inference (4–8 frames per call) the context is **longer** than in training, not shorter. Hypothesis 3 in the findings doc has the direction wrong.
- **Temporal RoPE:** `create_coordinates_3d` uses `linspace(0, 1, T)`, so the time spacing between frames depends on the number of frames per call (training: 1/1 or 1/3; 8 frames: 1/7).
- **Guide:** in training the guide had 2 frames per feature frame (all T video frames, `adaptive_avg_pool3d` pools Q and K to T/2) at 224² for a 14² output (16 px per output token). In Exp 3 / Test E: 1 guide frame per feature frame, guide 112² for a 112² output (1 px per token).
- **Scale:** training upsampled 7² → 14² (2×); `window_ratio = 0.1` then covers ≈ 2–3 LR tokens. At 32² → 112² (3.5×) it covers ≈ 7.

## Scope

- In scope: new `experiments/exp3_anyup3d_diag.ipynb` (diagnostic only; no change to the Exp 3 notebook). _Run 2 adds one opt-in change to `anyup/backbones.py` (VideoMAE at non-native input, Q&A 12) + `tests/test_videomae_pos.py`._
  - **Reference:** AnyUp2D and AnyUp3D (8 frames per call) on DINOv2-B 448² → 112², i.e. the T11 run-5 setting (regression check).
  - **Test T (temporal):** AnyUp3D with 1, 2 and 8 frames per call, and 8 frames with `window_t = 0` (temporal attention off, temporal convs still on).
  - **Test V (VideoMAE):** AnyUp3D's training backbone at its native 224 input, guide 2:1 as in training, 8 feature frames per call; AnyUp2D on the same features.
  - **Feature diagnostics** in every AnyUp3D run (computed on the fly, no extra resolution): per frame cos(AnyUp3D, AnyUp2D), input consistency of each upsampler (output avg-pooled back to the LR grid vs the LR features), by frame index and by position in the call.
  - Visuals per test (see Design).
  - **Run 2 (Test V448, Cell 6b):** VideoMAE at 448 (28² tokens) with the interpolated position table: AnyUp2D, AnyUp3D 8 per call, 1 per call, `window_t 0`; plus a 448-vs-224 feature check (DINOv2 as reference). Run flags per test cell in Cell 2.
- Out of scope: guide-resolution and window-ratio / upsampling-factor tests (declined, S11); resolution sweeps; bilinear; the mask fix (Problem 3); retraining; more than 3 clips.

## Q&A (decisions agreed before implementation)

| # | Question | Answer |
|---|---|---|
| 1 | Resolution for the DINOv2 tests | **448² → 112²** (AnyUp3D 8 frames per call fits on a T4; 112×200 caps at 4). No resolution sweep: "do not sweep resolutions, lets just go straight to the good resolution". |
| 2 | Clips | **parkour, bmx-trees, blackswan** (collapse + thin object + easy control). |
| 3 | Comparison | **AnyUp2D only** (no bilinear), as Monzir asked. |
| 4 | Which tests | **Temporal** (1 / 2 / 8 frames, `window_t = 0`) and **VideoMAE**. Guide-resolution and window/scale tests declined (→ S11). |
| 5 | VideoMAE input resolution | **224 native** (14² tokens; its position embeddings are fixed). Propagation grid 112². |
| 6 | VideoMAE guide | **2:1, as in training**: both video frames of each tubelet (frames 2k, 2k+1) for feature frame k. AnyUp2D gets frame 2k (one per feature frame). |
| 7 | VideoMAE propagation / scoring | **Propagate on feature frames, hold labels:** feature frame k predicts video frames 2k and 2k+1; official toolkit on all frames, same rule for both upsamplers. |
| 8 | VideoMAE AnyUp3D frames per call | **8 only** (one full VideoMAE chunk = 8 feature frames, 16 guide frames). |
| 9 | Tracking | **New task T12**, P1 addendum, on the existing branch `experiments-in-action-test`. |
| 10 | VideoMAE `N_LAST` (Monzir: "make the rest of decisions by your own") | **6 feature frames** (≈ 12 video frames): the same number of context feature maps as the DINOv2 runs, so the propagation cost and rule are the same; only the time span is longer. _My decision._ |
| 11 | Feature diagnostics (offered, not objected to) | **Kept** in every run: one extra AnyUp2D pass per chunk, no extra resolution. _My decision._ |
| 12 | Run 2 (Monzir: "low resolution corrupts the experiment … test videomae with good resolution, like 112x112"; flags to skip the other tests) | 112² was already the propagation grid; what was low is the **token grid** (224 / 16 = 14²; 112² tokens would need a 1792 input). **VideoMAE input 448 → 28² tokens** (≈ DINOv2 448's 32²; 512 → 32² is one knob away), position table interpolated bicubically per feature frame (standard ViT practice; zero-shot for VideoMAE). Same 4 variants as on DINOv2 (2D, 3D 8/call, 1/call, `window_t 0`). Flags `RUN_REF / RUN_T / RUN_V224 / RUN_V448` (default: only V224 + V448); outputs to `n3_v448`; Cell 7 shows run 1 results (marked *) for runs not repeated. _My decisions: 448, the two extra variants, the feature check._ |

## Design

Notebook `experiments/exp3_anyup3d_diag.ipynb`, standalone (needs no other notebook). Results to `/content/drive/MyDrive/results/T12/anyup3d_diag/`.

1. **Cell 1 · Setup** (as `exp3_debug_E.ipynb`, `REPO_BRANCH` = T12 branch). Prints the checkpoint's `global_step` and `t_schedule` (proof of the training facts above).
2. **Cell 2 · Config.** `CLIPS`, DINOv2 `INPUT = (448, 448)`, `GRID = (112, 112)`, `TAU 0.07 / TOP_K 7 / N_LAST 6` (Exp 3), `BB_CHUNK = 8`, `Q_CHUNK_3D = 1024`, `Q_CHUNK_PROP = 1024`, `RUNS_T = [(name, frames_per_call, window_t)]`, VideoMAE knobs, visual knobs. Every compute line commented with its dependents.
3. **Cell 3 · Helpers.**
   - `propagate_e` (copied from T11, already checked = `davis.propagate_labels`).
   - `run(name)` / `clip_features(spec, clip)` (one runner for both backbones): per clip, DINOv2 once per 8-frame chunk → AnyUp2D on the chunk (reference) → if `up3d`: AnyUp3D on sub-calls of `fpc` frames → per-frame diagnostics → only the evaluated condition's L2-normalized features are stored (T, N, C) on the GPU → `propagate_e` → official J&F. Stores the 112² predictions, the diagnostics, and one feature frame per clip (for the PCA figure).
   - VideoMAE branch of the same runner: VideoMAE 16-frame chunks (T02 padding) → 8 feature frames (14²). AnyUp3D: guide built from video frames (f, min(f+1, last)) per kept feature frame f at 112², ImageNet-normalized as the adapter does, called as `model(guide (1,3,2K,112,112), feats (1,768,K,14,14), output_size=(112,112))`. AnyUp2D: the adapter's guide (frame f). Propagation on the K feature frames; prediction for video frame t = feature frame t // 2. `N_LAST = 6` counts **feature** frames here (≈ 12 video frames), Q&A 10.
   - Diagnostics per frame: `cos_2d` = mean per-pixel cosine(AnyUp3D, AnyUp2D); `ic_3d`, `ic_2d` = mean per-pixel cosine(avg-pool(output → LR grid), LR features); `pos` = index of the frame inside its AnyUp3D call.
   - Per-frame J per object (native resolution) and lost object-frames for the plots (official `eval_clip_jf` + `summarize_jf` for the table).
4. **Cell 4 · Reference:** `ref2d` (AnyUp2D) and `t8` (AnyUp3D, 8 per call, `window_t 4`). Check: per-clip J&F = T11 run 5 (blackswan 0.937 / 0.880, bmx-trees 0.561 / 0.327, parkour 0.880 / 0.525; tolerance 0.002).
5. **Cell 5 · Test T:** `t1`, `t2`, `t8_wt0` (a second AnyUp3D with `window_t = 0`).
6. **Cell 6 · Test V:** `v_2d` (AnyUp2D), `v_3d` (AnyUp3D, guide 2:1).
6b. **Cell 6b · Test V448 (run 2):** `res_check` (first chunk per clip: features at 448 avg-pooled to the 224 grid vs features at 224, per-pixel cosine; VideoMAE and DINOv2), then `v448_2d`, `v448_3d`, `v448_t1`, `v448_wt0`. VideoMAE AnyUp3D now takes sub-calls of `fpc` feature frames (guide 2:1 per sub-call).
7. **Cell 7 · Results + visuals** (works on whatever finished):
   - Table: official J&F / J / F, lost object-frames, Δ vs AnyUp2D of the same backbone, per-clip J&F.
   - `bars.png`: J&F per run (DINOv2 group, VideoMAE group), lost % on the bars.
   - `per_frame_<clip>.png`: per-frame J (mean over objects) for every run, lost frames marked.
   - `diag_<clip>.png`: cos(AnyUp3D, AnyUp2D) and input consistency per frame for every AnyUp3D run, and the mean by position in the call (does the error grow at call edges or with more frames?).
   - `pca_<clip>.png`: one middle frame, LR features | AnyUp2D | every AnyUp3D run, one joint PCA (`viz.FeaturePCA`, fixed scale).
   - `masks_<clip>.png`: GT | AnyUp2D | every AnyUp3D run at `SHOW_FRACS` of the clip **and** the first frame where `t8` loses an object.
   - `<clip>.mp4` + `.gif`: GT | AnyUp2D | AnyUp3D runs side by side, labelled with the run and its clip J&F, plus a frame counter.
   - CSV / JSON via `metrics.save_results` (per clip + summary + per-frame diagnostics).

How to read the results (written into the notebook):
- `t1` ≈ AnyUp2D → the temporal part hurts; then `t2` vs `t8` vs `t8_wt0` says whether it is the context length (training saw 1–2) or the temporal attention.
- `t1` still bad → the spatial path / backbone domain; then Test V says whether AnyUp3D works on its own backbone.
- `v_3d` ≥ `v_2d` but DINOv2 bad → backbone shift (findings doc hypothesis 1).

## Steps

1. Notebook + local CPU dry run with stub backbones / upsamplers on synthetic DAVIS clips (cells run end to end; figures written). Real models are not runnable locally.
2. Monzir runs it on a T4 and pastes the outputs.
3. Interpretation → update `exp3_debug_findings.md` Problem 2 → decide the fix (new task).

## Tests / Regression Checks

- Local: `propagate_e` = `davis.propagate_labels` (exact, random features); the frames-per-call split covers every frame once, in order; VideoMAE guide index rule (f, min(f+1, last)) and the hold rule t → t // 2 on odd and even clip lengths; diagnostics on identical inputs give cos 1.
- Run 2, local: `tests/test_videomae_pos.py` (identity at the same size, shape, time not mixed, constant stays constant, (t, h, w) layout, wrong size raises); adapter with real HF `VideoMAEModel` code (random init): 224 output bit-identical to the plain HF forward, 448 gives (8, 768, 28, 28).
- Colab: Cell 4 reproduces T11 run 5 per-clip J&F (tolerance 0.002) → PASS / WARN; Cell 1 prints `global_step 6500` and the `t_schedule`.

## Done When

- The notebook runs end to end on a T4 and its outputs are logged here, with the reading of each test.

## Implementation Log

_Updated live during implementation._

- **Q&A done** (answers above). Tests declined by Monzir logged as **S11** in `docs/skipped_and_future_concerns.md`.
- **OK received** ("yes lets start on the same branch, please make the rest of decisions by your own cause I will be out for a while"). Branch = `experiments-in-action-test`; Q&A 10–11 decided by me. Status Q&A → in progress.
- **Step 1 (code + local checks): new `experiments/exp3_anyup3d_diag.ipynb`** (8 cells: intro + Cells 1–7; generated from scratchpad `t12/cells.py` by `t12/make_nb.py`; all code cells parse; Colab metadata from `exp3_debug_E.ipynb` without its widget state).
  - Cells as in Design. Implementation details: one runner `run(name)` + `clip_features(spec, clip)` for both backbones (instead of separate `run_dino` / `run_vmae`); AnyUp3D sub-calls through the wrapper on sliced `BackboneOutput`s (`up3d_split`); VideoMAE AnyUp3D called on `up3d.model` directly (the wrapper requires equal guide and feature T); a run that runs out of GPU memory stops, is listed as SKIPPED, and the next run starts; AnyUp3D mask caches cleared after each run; peak GPU memory and time per run recorded. Cell 3 also prints the checkpoint's `global_step` / `t_schedule` / `crop_size` / `img_size` / `window_t` (`CKPT_INFO`, saved in the summary JSON).
  - Dropped from the design: the Colab re-check of `propagate_e` vs `davis.propagate_labels` (at 112² the reference keeps the full (N, 7N) affinity twice ≈ 4.9 GB next to the feature store; the copy is verbatim T11 minus the window, which T11 checked on Colab, and the Cell 4 check vs T11 run 5 covers the whole path).
  - **Local checks (CPU, scratch venv; not Colab, not the real backbones):**
    - `propagate_e` (extracted from the cell source) = `davis.propagate_labels`: 0 of 5,760 grid pixels differ for query chunks 100 / 37 / 480 (random features, 12 frames, 20×24, 2 objects).
    - **Dry run of Cells 2–7** (`t12/harness.py`): stub DINOv2 / VideoMAE adapters (correct strides, VideoMAE 16 → 8 frames, `NUM_FRAMES` padding), **real AnyUp2D** (paper weights from `weights/anyup_paper.pth`) and **real AnyUp3D module** (random weights, `ANYUP3D_ARCH`), synthetic DAVIS clips named as the 3 clips (20 / 17 (odd, tests the VideoMAE padding + hold rule) / 18 frames, 120×214), small sizes (DINOv2 input 56×98 → 4×7 tokens, grid 16×28). All 7 runs and Cell 7 run end to end; the VideoMAE `frame_index == 0, 2, 4, …` assert and the feature-frame count assert hold; all outputs written (`bars.png`, `per_frame_*`, `diag_*`, `pca_*`, `masks_*` PNGs, 3 MP4 + GIF, `t12_diag_per_item.csv`, `t12_diag_summary.json`). Cell 4 check prints WARN as expected (stub numbers mean nothing).
    - Simulated interrupt (Cell 6 stopped after one clip of `v_2d`): Cell 7 runs, shows the 5 finished runs, lists `v_2d` as partial, prints `n/a` for the missing comparison.
    - Fixes from looking at the figures: per-frame legend moved outside (it hid the lost-object ticks); cos(3D, 2D) not printed / shown as "—" for AnyUp2D runs.
  - Not verified: real T4 memory and time (estimate ≈ 30–45 min: T11 run 5 took ≈ 250 s per condition for parkour at this size); real VideoMAE through the 2:1 guide path (shapes checked only with the stub).
- **`exp3_debug_findings.md` corrected:** Problem 2 hypothesis 3 said the inference context is shorter than in training; the checkpoint saw 1–2 feature frames per call, so it is longer. Rewritten with the training facts.
- **Waiting for Monzir's Colab run** (Step 2).
- **Step 2 (Colab T4 run, commit `e34ac8b`, all 7 runs finished, outputs in Drive `results/T12/anyup3d_diag/n3`).** Cell 4 check vs T11 run 5: 6/6 PASS. Peak GPU: ref2d 5.0, t8 10.4, t1 5.3, t2 6.1, t8_wt0 10.5, v_2d 3.4, v_3d 4.2 GiB; ≈ 23 min total.

  | run | J&F | Δ vs 2D | lost obj-frames | cos(3D,2D) | in-cons | parkour | bmx-trees | blackswan |
  |---|---|---|---|---|---|---|---|---|
  | ref2d | 0.735 | — | 0 % | — | 0.883 | 0.880 | 0.561 | 0.937 |
  | t8 | 0.514 | −0.220 | 15.2 % | 0.944 | 0.831 | 0.525 | 0.327 | 0.880 |
  | t1 | 0.617 | −0.117 | 13.2 % | 0.965 | 0.855 | 0.804 | 0.378 | 0.911 |
  | t2 | 0.539 | −0.195 | 18.5 % | 0.953 | 0.847 | 0.679 | 0.303 | 0.872 |
  | t8_wt0 | 0.602 | −0.132 | 8.3 % | 0.953 | 0.853 | 0.737 | 0.404 | 0.864 |
  | v_2d | 0.267 | — | 53.3 % | — | 0.978 | 0.119 | 0.085 | 0.778 |
  | v_3d | 0.385 | +0.118 | 24.2 % | 0.967 | 0.944 | 0.208 | 0.248 | 0.836 |

  Status stays in progress until Monzir agrees on the reading (Step 3).
