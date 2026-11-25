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
| 9 | Test E backbone (after run 1, Monzir: "run a test on bilinear with good resolution … and also anyup2d") | **DINOv2-B only.** Conditions bilinear + AnyUp2D (AnyUp3D left out: its full attention does not fit these sizes on a T4). |
| 10 | Test E resolutions | **Ladder, aspect kept:** DINOv2 input 224² → propagation 56² (= Test C dino224), 448² → 112², 448×798 → 112×200 (no squash). |
| 11 | Test E external reference | **Yes:** the published DINO propagation protocol (476×854 input, native 34×61 grid, no upsampler, window 12, top-5, τ 0.1, frame 0 + last 7, soft labels ×14 bilinear → argmax). DINO ViT-B/16 paper J&F 0.623 as an order-of-magnitude line. |

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
9. **Test E** (added after run 1; needs Cells 1–3 only): `propagate_e` = the `davis.propagate_labels` rule in query chunks (memory ∝ `E_Q_CHUNK`), optional window, returns soft labels; per variant and condition a full-clip normalized feature tensor on the GPU (backbone re-run per condition so only one set is held). Prints J&F / J / F, the ceiling of each grid (Test A rule) and **lost object-frames** (object in the GT, 0 predicted pixels). Checks (iii) `propagate_e` = `davis.propagate_labels` on the first clip, (iv) e224_p56 bilinear = Test C dino224. Figures `E_bars.png`, `E_masks.png` (2 best + 2 worst clips of the largest grid); results `exp3_debug_E_*`.

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
- **Step 2: Colab run 1** (T4, 30 clips, commit `2919e92`, `RUN_NAME p56_n30_bilinear-anyup3d`; outputs saved in the notebook, read locally).
  - Checks: (i) PASS (0 differing pixels, bike-packing); (ii) PASS (bilinear 0.1272 vs 0.1272, AnyUp3D 0.2309 vs 0.2308).
  - **A (ceiling, official J&F):** 28² 0.497, **56² 0.712**, 112² 0.873, 56×100 0.820. Lost frame-0 objects at 56²: lab-coat obj 1 (18 px → 0), obj 2 (14 px → 1). Worst ceilings: kite-surf 0.486, lab-coat 0.623, bmx-trees 0.576.
  - **B (window, Swin 224):** no effect: bilinear +0.0004 (w6) / +0.0003 (w12), AnyUp3D −0.0010 / −0.0014.
  - **C (DINOv2 224, 16²):** bilinear 0.3550 (+0.228, 29/30), AnyUp3D 0.3922 (+0.161, 29/30); window again ≈ 0. **dino448 SKIPPED: CUDA OOM** (tried to allocate 12.25 GiB with `CHUNK_SIZE_C=16`, `Q_CHUNK=None`).
  - **D (Swin-T 448, 14²):** bilinear 0.2117 (+0.085, 27/30), AnyUp3D 0.3289 (+0.098, 26/30); w12 AnyUp3D 0.3311.
  - AnyUp3D − bilinear: swin224 +0.104, swin448 +0.117, dino224 +0.037. AnyUp3D < bilinear clips at dino224: parkour (0.064 vs 0.328), drift-straight, bmx-trees, horsejump-high, judo, shooting.
  - Visuals (C_masks): many failures are **objects that vanish completely** (empty prediction), e.g. drift-chicane in every variant, breakdance swin224/bilinear.
  - `RuntimeWarning: invalid value encountered in divide` at `metrics.py:94` is harmless: the official `db_eval_iou` sets union = 0 frames to J = 1 right after.
  - Status of the run: open items = dino448 re-run (smaller chunk), next tests (decided with Monzir).
- **Step 3: Test E (Cell 9) added** + `E_*` config in Cell 2 + intro row. Local CPU dry run with stubs: check (iii) PASS (0 of 31,360 grid pixels differ); `propagate_e` = `davis.propagate_labels` on random features for query chunks 600 / 128 / 7 (0 differing pixels); radius ≥ grid = no window.
- **Colab run 2 (Cell 9, partial, pasted by Monzir):** check (iii) PASS (0 of 216,384 pixels, bike-packing). Finished: e224_p56 (452 s) and e448_p112 (3,104 s) on all 30 clips; e448x798_p112x200 at ≈ 4–6 min per clip (4 clips done when pasted). Clip-mean J&F from the printed lines (not the official object-weighted mean): e224_p56 bilinear 0.382 / AnyUp2D 0.454 (AnyUp2D > bilinear on 24/30); **e448_p112 bilinear 0.588 / AnyUp2D 0.686** (26/30). DINO reference not reached yet.
- **Step 4: Cell 10 added** (summary of finished clips after interrupting Cell 9, without restarting the runtime): frees the interrupted GPU tensors, optionally runs `dino_ref`, official tables on all clips + on the interrupted variant's finished clips, ceilings, lost object-frames, `E_bars.png`, `E_per_clip.png`, `E_masks.png`, `exp3_debug_E_*`. Local dry run with a simulated interrupt: runs end to end, partial variant handled (1 clip common to both conditions).
- **Colab run 2 deleted by Monzir** (interrupted before Cell 10 ran; nothing saved). Request: a standalone Test E notebook on 5 clips with visuals.
- **Step 5: new `experiments/exp3_debug_E.ipynb`** (6 cells; standalone, needs no other notebook; scratchpad `t11/e_cells.py` + `make_e_nb.py`). Same `propagate_e` / `run_e` as Cell 9. Changes vs Cell 9:
  - **5 clips (my pick, a config line):** blackswan, dog (easy), bmx-trees (thin object), drift-chicane (vanished everywhere in run 1), parkour (AnyUp3D collapse).
  - The DINO reference runs **first** (cheap, external check). Results cell is separate, so it also works after an interrupt (fair table on the clips every variant finished).
  - Visuals: `E_bars.png` (official J&F + grid ceilings + DINO paper line), `E_per_clip.png` (one line per clip across the ladder), `E_masks_<clip>.png` (GT + every variant, 2 frames per clip). Check iv dropped (needs 30 clips).
  - Outputs: `results/T11/exp3_debug_E/n5/`.
  - **Local CPU dry run** (stubs, synthetic DAVIS named as the 5 clips, small grids): equivalence check PASS (0 of 31,360 pixels differ); full run and simulated interrupt both run end to end and write all figures + CSV/JSON. Not run on Colab yet.
- **Colab run 3 (`exp3_debug_E.ipynb`, T4, 5 clips, commit `3701422`, 2,339 s; outputs in the notebook).** Equivalence check PASS (0 of 156,800 pixels). Official J&F (ceiling, lost object-frames of 410):
  - dino_ref 34×61 native: **0.581** (0.667, 5.1 %). Order of magnitude of DINO ViT-B/16 paper 0.623 (30 clips) → propagation + scoring code plausible.
  - e224_p56: bilinear 0.311 / AnyUp2D 0.398 (0.711; lost 22.9 % / 27.8 %).
  - e448_p112: bilinear 0.590 / AnyUp2D **0.763** (0.862; lost 0 / 0).
  - e448x798_p112x200: bilinear 0.663 / AnyUp2D **0.791** (0.888; lost 0 / 0).
  - drift-chicane: 0.037 / 0.012 at 224 → 0.623 / 0.780 at 448 (masks: the car is tracked).
  - Reading: resolution has a direct, large effect, and it removes the vanishing objects. The ladder changes the DINOv2 input (token grid 16² → 32² → 32×57) **and** the propagation grid together, so the two are not separated. dino_ref (34×61 tokens, coarser propagation grid than 56², no upsampler) beats e224_p56 by +0.27 → the token grid of the backbone input looks like the main factor. Isolation (e448 → p56, e224 → p112) not run.
- **Step 6: AnyUp3D at Test E sizes.**
  - `exp3_debug_E.ipynb`: `"anyup3d"` added to `E_CONDITIONS` (`CKPT_ANYUP3D`, `E_Q_CHUNK_3D` 1024, `E_CHUNK` 8); out-of-memory is now handled per condition (only that upsampler is skipped for the rest of the variant); AnyUp3D's mask cache is cleared after each variant. Dry run (stubs): full and simulated AnyUp3D OOM both run end to end. Not run on Colab.
  - Code facts: `CrossAttentionBlock3D` builds the **full** bool mask (T·H·W) × (T·h·w) on the **CPU** (`compute_attention_mask_3d`, `lru_cache`d) and copies it to the GPU on every call; `q_chunk_size` only chunks the attention, not the mask. Mask at chunk 8: ≈ 0.8 GB (112²), ≈ 2.4 GiB (112×200); chunk 16 at 112×200: ≈ 9.7 GiB (too big for Colab CPU RAM).
  - Monzir: "make the test for only anyup3d, I just want to test the memory now". **New `experiments/exp3_anyup3d_mem.ipynb`**: one chunk of parkour per variant × chunk {4, 8, 16} × q_chunk {None, 4096, 1024} (peak GPU, time, OOM stage); whole parkour clip on the Test E path (chunk 8, q_chunk 1024: feature store + AnyUp3D); heatmap figure + CSV/JSON. Guard: a case is not run if its mask would not fit the CPU RAM (2.5× mask > free RAM) or 0.8× GPU. Dry run with fake CUDA stats: runs end to end (stub numbers meaningless).
- **Colab run 4 (`exp3_anyup3d_mem.ipynb`, T4 14.6 GiB, parkour 100 frames, commit `f4b0dd7`).** One chunk, peak GiB (q_chunk None / 4096 / 1024):
  - 56²: chunk 4 1.16 / 1.16 / 1.16; chunk 8 2.21 / 1.92 / 1.92; chunk 16 7.59 / 3.46 / 3.46.
  - 112²: chunk 4 7.59 / 3.46 / 3.46; chunk 8 OOM / 6.54 / 6.54 (2.9 s); chunk 16 OOM / 12.69 / 12.69 (11.5 s).
  - 112×200: chunk 4 OOM / 5.87 / 5.87 (2.3 s); chunk 8 OOM / 11.34 / 11.34 (9.3 s); chunk 16 not run (mask 9.7 GiB, CPU RAM free 9.7 GiB).
  - (56² chunk 16 = 112² chunk 4 exactly: same query × key count, 50,176 × 4,096.) q_chunk 4096 and 1024 give the same peak → the full mask + per-call tensors dominate, not the attention rows.
  - Whole clip (chunk 8, q_chunk 1024; feature store + AnyUp3D): 56² peak 2.83 GiB (store 0.90), 112² 10.15 (store 3.59), **112×200 OOM** (store 6.41).
- **Step 7: `exp3_debug_E.ipynb` set for the real run with AnyUp3D:** `E_CHUNK_3D` per variant = 8 / 8 / **4 at 112×200** (estimate 5.9 + 6.4 ≈ 12.3 of 14.6 GiB). Consequence: AnyUp3D's temporal context is 4 frames at 112×200 (8 elsewhere; Exp 3 used 16). `CLIPS_E = None` → all 30. Dry run (stubs): runs end to end. Not run on Colab.
