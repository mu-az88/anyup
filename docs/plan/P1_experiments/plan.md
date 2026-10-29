# P1 — Experiments: Big Picture Plan

## Where We Are

- **Project:** AnyUp3D extends AnyUp (2D, any-resolution feature upsampling) to video, adding temporal consistency through 3D/temporal components (`t_k=3`, windowed 3D cross-attention).
- **Status:** the M.Sc. thesis is accepted and closed. The aim now is the **strongest possible paper**. Venue and scope (one paper or several) are still to be decided with the supervisor.
- **Pre-publication plan:** about 6 weeks of technical work, then 2 weeks of writing and review. The approach is tiered: every milestone should leave the project in a publishable state.
- **Order of work:** (1) experiments, the current bottleneck → (2) training-pipeline fixes → (3) scaling data and backbones.

## Existing Experiments (thesis results, the baseline to reproduce)

| # | What it measures | Thesis result |
|---|---|---|
| Exp 1 | Cosine fidelity of upsampled features | 0.9899 (step 250) → 0.9913 (step 6,000) |
| Exp 2 | Temporal cosine similarity, 30 DAVIS-2017 clips | +0.081 improvement |
| Exp 3 | Video object segmentation via non-parametric label propagation (DAVIS-2017), zero-shot on Video Swin-T | Weighted J&F 0.266 → 0.347 (+0.081); 27/30 clips improved; the boundary gain (𝓕) is about 2× the region gain (𝓙) |
| Exp 4 | Efficiency | 1,540,736 params (5.88 MB), +75.5% vs AnyUp2D; about 2.7× slower and about 10.5× more memory |

### Evaluation details to keep

- **Exp 3:** non-parametric label propagation (no trained head), following the Jabri et al. and DINO protocols. Settings: `TAU=0.07`, `TOP_K=7`, `N_LAST_FRAMES=6`.
- **Exp 3:** **exclude frame 0** from scoring. Use the **official DAVIS boundary metric** (dilation tolerance, `bound_th=0.008`), not exact pixel overlap.
- **Exp 3:** low absolute J&F is expected. Frame it as a **lower bound**, not a weakness. Swin-T is a deliberate zero-shot universality test.
- A sign-test p-value is available (p ≈ 9.3×10⁻¹⁰ for a 30/30 result). **Check which result this refers to,** since Exp 3 reports 27/30.
- SAM 2 appears in the literature review as a key baseline but was never evaluated. Either address it or remove it.

## The Problem to Fix

The experiments were built separately during the thesis. Backbone loading, model loading, data handling, and metrics may differ between notebooks. For a paper, a reviewer must be able to trust that every comparison is apples-to-apples and that the results are reproducible.

## Target State for the Experiments Part

### Phase A: Shared foundation (current step)

1. **Unified `BackboneLoader`** (task **T01**, see `T01_backbone_loader.md`): one entry point, one adapter per backbone, a standard output (`features`, `guide`, `meta`).
2. **Unified `Upsampler` wrapper:** AnyUp3D, AnyUp2D (per-frame), and simple baselines (bilinear, etc.) behind one call. This lets every experiment run any backbone × upsampler combination.
3. **Shared data utilities:** one DAVIS loader with consistent frame sampling, resolution and resizing, and annotation handling, reused by Exp 2 and Exp 3.
4. **Shared metrics module:** cosine fidelity, temporal consistency, and the official DAVIS J&F. Each is defined once.
5. **Reproducibility:** fixed seeds, one config per run (backbone, checkpoint, sizes, frames, chunk size), and results saved to disk in a structured format (CSV or JSON per clip, plus a summary).

### Phase B: Re-verify the thesis results

- Re-run Exp 1–4 on the shared foundation and confirm the numbers match the thesis, or explain the differences.
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

T01 (Phase A.1): review the experiment notebooks one by one, fill in the requirements tracker in `T01_backbone_loader.md`, then implement the loader.

## Task Index

Task IDs are global across all plans (P2 continues from the last P1 number). Status: `todo` → `Q&A` → `in progress` → `done` (or `blocked`).

| ID | Task | Phase | Status | File |
|---|---|---|---|---|
| T01 | Unified `BackboneLoader` | A.1 | todo | `T01_backbone_loader.md` |
| — | Unified `Upsampler` wrapper | A.2 | not yet created | — |
| — | Shared DAVIS data utilities | A.3 | not yet created | — |
| — | Shared metrics module | A.4 | not yet created | — |
| — | Reproducibility (seeds, configs, results on disk) | A.5 | not yet created | — |
