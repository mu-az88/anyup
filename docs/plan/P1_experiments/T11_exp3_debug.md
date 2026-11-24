# T11 — Exp 3 Debug: Find the Shared Bottleneck of Label Propagation

| Field | Value |
|---|---|
| Task ID | T11 |
| Plan | P1 Experiments (`plan.md`), addendum after P1 close (diagnostics for Phase B) |
| Branch | `experiments-in-action-test` (default; change `REPO_BRANCH` in Cell 1 if you make a T11 branch) |
| Status | in progress |
| Depends on | T04 (`davis`, `metrics`), T05 (Exp 3 protocol + its numbers as the reference), T10 (`anyup.viz`) |
| Blocks | Any Exp 3 protocol change (decided after these results) |

## Goal

Exp 3's absolute J&F is low for every upsampler (T05: bilinear 0.127, AnyUp2D 0.154, AnyUp3D 0.231). A diagnostic notebook tests, one by one, the steps all three conditions share, so we can see which one is the bottleneck. Every test has visuals.

## Scope

- In scope: new `experiments/exp3_debug.ipynb` (diagnostic only: it does not change `anyup/` or the Exp 3 notebook).
  - **Test A:** resolution ceiling. The GT masks are round-tripped through the propagation grid (native → grid → native, nearest) and scored, with no model. It also counts objects that vanish from the frame-0 mask at 56².
  - **Baseline + Test B:** the current protocol (Swin-T 224, 7×7, 56² propagation) and a **local spatial window** in the propagation (square neighbourhood as in DINO's `size_mask_neighborhood`), on the same features.
  - **Test C:** backbone swap to **DINOv2-base per frame** (224 → 16², 448 → 32²), everything else identical; with and without the window.
  - **Test D:** **Swin-T at 448** (14×14 grid instead of 7×7); with and without the window.
  - Summary table + bar chart + per-clip CSV / summary JSON on Drive.
- Out of scope: changing `davis.propagate_labels` or the Exp 3 notebook (only if a test shows a clear gain, in a later task); DINO ViT-S/8 (no adapter exists); aspect-preserving backbone input (only its *ceiling* is measured in Test A); training anything.

## Q&A (decisions)

Monzir asked directly for the notebook ("make me a code to test all these ideas … one by one in sequential cells … T4 … visuals with each test … I will run it and give you results"). There was no question round; the answers marked **(default)** are my picks, to be confirmed or changed after the first run.

| # | Question | Answer |
|---|---|---|
| 1 | Which tests? | **A, B, C, D**, as discussed (resolution ceiling, local window, DINOv2 swap, Swin at 448). Skip a test if it can't run on a T4. |
| 2 | Visuals? | **Each test has its own figures** (Monzir: "I need visuals with each test"). |
| 3 | Conditions | **(default)** `bilinear` + `anyup3d`; `anyup2d` is a config switch (+50 % time). The question is the *shared* bottleneck, so two conditions are enough. |
| 4 | Clips | **(default)** All 30 val clips; `N_CLIPS` to shrink. |
| 5 | Window radius | **(default)** `RADII = [6, 12]` grid cells at 56² (DINO uses 12 cells on its stride-8 grid ≈ 96 px; here 12 cells ≈ 103 px vertically on 480p). |
| 6 | Isolation | Each test changes **one** thing vs the baseline; propagation size stays 56², τ / top-k / context unchanged. C and D also run the window, so the combinations come for free (propagation only). |
| 7 | Scoring path | Predictions are nearest-resized to native in memory (`davis.resize_labels`, the same cv2 call as `save_mask_png`), no PNG write. Official `eval_clip_jf` + `summarize_jf`. |
| 8 | Sanity checks | (i) the window-free propagation in the notebook equals `davis.propagate_labels` exactly on one clip; (ii) the baseline reproduces T05 (bilinear 0.1272, AnyUp3D 0.2308; tol 0.002, GPU propagation). |

## Design

Notebook cells:

1. Setup (as the Exp 3 notebook, `REPO_BRANCH`).
2. Config (every compute knob commented with its dependents).
3. Imports, data, helpers: `propagate_local` (a copy of `davis.propagate_labels` + optional square window), `run_backbone` (backbone once per clip → upsampled features per condition → each propagation variant → score; stores 56² predictions for the figures), figure helpers (mask panels, per-clip scatter, LR feature PCA).
4. Test A: ceiling vs grid size (28², 56², 112², 56×100 aspect-kept) + lost objects + figure.
5. Baseline + Test B: Swin-T 224; variants `none`, `r6`, `r12`; checks (i), (ii); scatter + mask panels.
6. Test C: DINOv2 224 and 448; variants `none`, `r12`; scatter + mask panels + LR feature PCA (Swin 7×7 vs DINOv2 16² / 32²).
7. Test D: Swin-T 448 (chunk 8); variants `none`, `r12`; scatter + mask panels + LR feature PCA.
8. Summary: one table (J&F / J / F official per variant × condition, Δ vs baseline, paired stats on per-clip J&F), bar chart, `save_results`.

Outputs: `/content/drive/MyDrive/results/T11/exp3_debug/<RUN_NAME>/` (CSV, JSON, PNGs).

## Steps

1. Notebook + local checks (`propagate_local` = `davis.propagate_labels` without a window; Swin3D-T 448 shape; cell dry run with stubs).
2. You run it on a T4 and paste the printed results + figures.
3. Interpretation, then decide (new task) what enters Exp 3.

## Tests / Regression Checks

- Local: `propagate_local(radius=None)` = `davis.propagate_labels` (exact, random features); the window keeps ≥ `top_k` candidates; Swin3D-T (`weights=None`) output grid 14×14 at 448².
- Colab: checks (i) and (ii) above printed as PASS / WARN.

## Done When

- The notebook runs end to end on a T4 (or a test is reported as skipped with the reason), and its results are logged here.

## Implementation Log

- **Request received; notebook build started without a question round (Monzir's direct request).** Defaults recorded in Q&A.
- **Step 1 (code + local checks): new `experiments/exp3_debug.ipynb`** (9 cells; generated by scratchpad `t11/make_nb.py` from `t11/cells.py`; all code cells parse).
  - Cells as in Design. Added beyond the design: a CUDA out-of-memory guard in `run_backbone` (a test that OOMs on the T4 is removed from the results, printed as SKIPPED with the reason, and listed in the summary; the notebook continues). Cell 5 stops with a clear error if the baseline itself OOMs (C and D need it). A feature-grid figure (`C_grids.png`, `D_grids.png`: the token grid drawn on a frame + PCA colours of the LR features) for "see it yourself".
  - Checked against the code: Swin-T adapter = final `norm` stage, stride 32 → 7×7 at 224 (frames squashed to 224²); `propagate_labels` has no spatial window; `eval_clip_jf` raises on void (255) in a prediction → Test A maps void → 0 after the round trip; the DINOv2 wrapper has `dynamic_img_size=True` (448 accepted).
  - **Local checks (CPU, scratch venv; not Colab, not the real models):**
    - torchvision `swin3d_t(weights=None)` at 448² → (8, 1, 14, 14, 768): grid 14×14 as planned.
    - `propagate_local(radius=None)` = `davis.propagate_labels` exactly on random features (9 frames, 20², 2 objects); radius ≥ grid = no window (exact); radius 1 ≠ no window.
    - **Dry run of Cells 2–8** with stub backbones / upsamplers on 4 synthetic DAVIS clips (12 frames, 480×854, 2–3 objects incl. a planted 6×6 px object): every cell runs; check (i) PASS (0 differing pixels, real `davis.propagate_labels`); Test A flags the planted object (36 px → 0 px at 56²); all 13 outputs written (`A_ceiling`, `A_worst_clips`, `B_scatter`, `B_masks`, `C_scatter`, `C_masks`, `C_grids`, `D_scatter`, `D_masks`, `D_grids`, `summary` PNGs + CSV + JSON). The stub numbers mean nothing about the real models.
    - Figure fix from the dry run: Test A bar colours clashed with the reference lines and the legend covered the bars (→ blue bars, legend outside).
  - Not verified: real T4 memory and run time (estimates only, in the notebook intro); the OOM guard path itself (never triggered locally).
- **Waiting for the Colab run** (step 2).
