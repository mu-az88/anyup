# P1 — Experiments: Big Picture Plan

## Where We Are

- **Project:** AnyUp3D extends AnyUp (2D, any-resolution feature upsampling) to video, adding temporal consistency through 3D/temporal components (`t_k=3`, windowed 3D cross-attention).
- **Status:** the M.Sc. thesis is accepted and closed. The aim now is the **strongest possible paper**. Venue and scope (one paper or several) are still to be decided with the supervisor.
- **Pre-publication plan:** about 6 weeks of technical work, then 2 weeks of writing and review. The approach is tiered: every milestone should leave the project in a publishable state.
- **Order of work:** (1) experiments, the current bottleneck → (2) training-pipeline fixes → (3) scaling data and backbones.

## Existing Experiments (thesis results, the baseline to reproduce)

| # | What it measures | Thesis result |
|---|---|---|
| Exp 1 | Spatial no-regression check: ADE20K linear-probe mIoU, DINOv2 single images (T=1), bilinear vs AnyUp2D vs AnyUp3D | see notebook results (Drive `exp1_results.json`) |
| (fidelity) | Cosine fidelity of upsampled features (source notebook unconfirmed, see Open Questions) | 0.9899 (step 250) → 0.9913 (step 6,000) |
| Exp 2 | Adjacent-frame temporal cosine similarity, 30 DAVIS-2017 clips, VideoMAE, AnyUp2D vs AnyUp3D | +0.081 improvement |
| Exp 3 | Video object segmentation via non-parametric label propagation (DAVIS-2017), zero-shot on Video Swin-T | Weighted J&F 0.266 → 0.347 (+0.081); 27/30 clips improved; the boundary gain (𝓕) is about 2× the region gain (𝓙) |
| Exp 4 | Efficiency | 1,540,736 params (5.88 MB), +75.5% vs AnyUp2D; about 2.7× slower and about 10.5× more memory |

### Evaluation details to keep

- **Exp 3:** non-parametric label propagation (no trained head), following the Jabri et al. and DINO protocols. Settings: `TAU=0.07`, `TOP_K=7`, `N_LAST_FRAMES=6`.
- **Exp 3:** **exclude frame 0** from scoring. Use the **official DAVIS boundary metric** (dilation tolerance, `bound_th=0.008`), not exact pixel overlap.
- **Exp 3:** low absolute J&F is expected. Frame it as a **lower bound**, not a weakness. Swin-T is a deliberate zero-shot universality test.
- A sign-test p-value is available (p ≈ 9.3×10⁻¹⁰ for a 30/30 result). **Check which result this refers to,** since Exp 3 reports 27/30.
- SAM 2 appears in the literature review as a key baseline but was never evaluated. Either address it or remove it.

## Issues Found in the Existing Experiments (from T01 review)

- Exp 1 ran with `window_t=3` (training used 4) and the `step500` checkpoint. **Must be re-run in Phase B.**
- Exp 3 fed raw [0,1] guides; training uses ImageNet-normalized guides. **Re-run in Phase B with normalized guides; the J&F numbers may change.**
- AnyUp3D was trained on VideoMAE features only, so DINOv2 and Swin-T results are zero-shot.

## Issues for P2 (training pipeline), found during P1

- `anyup/data/video_dataset.py:40` falls back to `torchvision.io.read_video`, which was removed in torchvision 0.26 (Colab, found in T02). The fallback is dead code: if decord fails, training crashes with an `ImportError`. Fix in P2 (decord only, as in `anyup/video.py`, or another reader).
- `train_3d.py` does not match the checkpoints in use (found in T03): it builds `AnyUp()` with the default `window_t=3` (`train_3d.py:394`) and saves under `model_state_dict`, while the run_01/run_02 checkpoints come from `Train_3D_full.ipynb` (`window_t=4`, key `"anyup"`, pickled `TStage`). Decide in P2 which training entry point is canonical.

## Open Questions

_Not needed for T01/T02; parked for later._

- Which notebook produced the cosine fidelity numbers (0.9899 → 0.9913)? Likely `results/VideoMAE features visualization exp/` (250 / 6000 steps). Unconfirmed.
- Where is the Exp 4 (efficiency) notebook? It's not in the repo.
- Canonical checkpoint: step6000 or step6500?
- Sign-test p-value (30/30) vs Exp 3's 27/30: which result does it belong to?
- SAM 2: evaluate it or remove it.

## The Problem to Fix

The experiments were built separately during the thesis. Backbone loading, model loading, data handling, and metrics may differ between notebooks. For a paper, a reviewer must be able to trust that every comparison is apples-to-apples and that the results are reproducible.

## Target State for the Experiments Part

### Phase A: Shared foundation (current step)

1. **Unified `BackboneLoader`** (task **T01**, see `T01_backbone_loader.md`): one entry point, one adapter per backbone, a standard output (`features`, `guide`, `meta`).
2. **Unified `Upsampler` wrapper:** AnyUp3D, AnyUp2D (per-frame), and simple baselines (bilinear, etc.) behind one call. This lets every experiment run any backbone × upsampler combination.
3. **Shared data utilities:** one DAVIS loader with consistent frame sampling, resolution and resizing, and annotation handling, reused by Exp 2 and Exp 3.
4. **Shared metrics module:** cosine fidelity, temporal consistency, and the official DAVIS J&F. Each is defined once. Compute cosines in float64 (or per pixel): one float32 cosine over a whole 768×112² frame gave 1.0034 for identical tensors (found in T03).
5. **Reproducibility:** fixed seeds, one config per run (backbone, checkpoint, sizes, frames, chunk size), and results saved to disk in a structured format (CSV or JSON per clip, plus a summary).

### Phase B: Re-verify the thesis results

- Re-run Exp 1–4 on the shared foundation and confirm the numbers match the thesis, or explain the differences. Expected differences: resize method (tensor vs PIL, Swin-T feature cos ≈ 0.993; measured in T01), Swin-T Kinetics input norm (cos ≈ 0.973), Exp 1 (`window_t` 3→4, ckpt step500→canonical, DINOv2 via timm wrapper instead of HF), Exp 3 (normalized guide), and Exp 2/3 (all frames instead of 16-linspace / every-12th, decided in T02). So Exp 2/3 thesis numbers won't be reproduced exactly; the new numbers replace them.
- Resolve the open checkpoint question (step6500 vs step6000) and fix one canonical checkpoint for the paper.

### Phase C: Strengthen the experiments for publication

These are suggestions, to confirm with the supervisor and against the paper's scope.

- **Backbone generality:** report the main metrics across several backbones (VideoMAE, Video Swin-T, DINOv2 per-frame, plus more if cheap). With the shared loader, this should cost only extra adapters.
- **Baselines:** for every metric, compare against bilinear, AnyUp2D per-frame, and ideally at least one other published feature upsampler. Decide on SAM 2.
- **Statistics:** report per-clip results and paired tests (sign test / Wilcoxon) consistently across experiments, not only for one of them.
- **Ablations (if not already in the thesis):** temporal kernel size `t_k`, temporal window, and the number of frames at inference.
- **Efficiency:** measure all methods under the same settings (GPU, resolution, frames), and include the `q_chunk_size` trade-off.

## Guiding Principles

- Each phase should leave the project in a publishable state; don't start Phase C until A and B are solid.
- Keep the experiments simple and honest. No premature success claims; regression-check every refactor against the old outputs.
- Every compute-controlling line in the code carries a short comment that lists its dependent lines (see Coding Rules in `CLAUDE.md`).

## Immediate Next Step

T03 (Phase A.2): unified `Upsampler` wrapper (AnyUp3D, AnyUp2D, bilinear), see `T03_upsampler_wrapper.md`.

## Task Index

Task IDs are global across all plans (P2 continues from the last P1 number). Status: `todo` → `Q&A` → `in progress` → `done` (or `blocked`).

| ID | Task | Phase | Status | File |
|---|---|---|---|---|
| T01 | Unified `BackboneLoader` | A.1 | done | `T01_backbone_loader.md` |
| T02 | Video reader + chunked backbone runner (all frames, any res/fps) | A.1/A.3 | done | `T02_video_reader.md` |
| T03 | Unified `Upsampler` wrapper (incl. `window_t` default 4, ckpt loading) | A.2 | in progress | `T03_upsampler_wrapper.md` |
| T04 | Shared DAVIS utilities + metrics module (official J&F, temporal cosine, fidelity, mIoU, paired stats, results I/O) | A.3 + A.4 | in progress | `T04_davis_metrics.md` |
| T05 | Refactor experiment notebooks to use shared backbones, upsamplers, DAVIS utils, metrics | A.2+ | todo (not yet created) | — |
| — | Reproducibility (seeds, configs; results I/O is in T04) | A.5 | not yet created | — |
