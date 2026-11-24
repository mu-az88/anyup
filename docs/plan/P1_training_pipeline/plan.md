# P1_training — Training Pipeline Fixes: Big Picture Plan

## Where We Are

- **Numbering:** Muaaz's plans are `P1_training`, `P2_training`, … and his tasks `T01_training`, `T02_training`, …, kept separate from Munther's `P1, P2, …` / `T01, T02, …` (D-009). Munther's P1 (experiments) closed on 2026-11-10 (`docs/plan/P1_experiments/plan.md`).
- This plan fixes the training pipeline (`Train_3D_full.ipynb` + `anyup/data/training/`) before any retraining for the paper.
- **Branches:** `main` is the base. Each fix gets its own `fix/...` branch from `main`, is tested and committed there, and is merged into `main` only after its Colab debug pass (D-008).
- **Logging:** each task has a file here (from `docs/plan/_templates/task_template.md`) with its Q&A and Implementation Log. Cross-task decisions also go in `docs/DECISIONS.md`. Skipped checks go in `docs/skipped_and_future_concerns.md` (D-009).
- **This file is a stub.** Scope, phases and the order of the remaining work in this plan are still to be planned. Known candidates are in the "Issues for P2 (training pipeline)" section of Munther's P1 plan. That plan also names his next phase "P2, training-pipeline fixes", so agree with him who fixes what.

## Done Before This Template

- **Teacher 16-frame fix**, branch `fix/teacher-full-16frame-clips` (D-001, D-004–D-006). It was planned in the older format and kept as is: `teacher_16frame_plan.md` (this folder). Local work and tests are done (commit `c6d1036`). Colab Step 0 passed (D-019); debug run T=2/4 pass, validation ≈ 6.4 min per pass (D-021, D-023); T=8 moved to T04_training (S14), where it passes on a T4 (D-031). Done; merged into `main` (`f05ea4d`, D-026).

## Task Index

Task IDs run across all of Muaaz's plans (`P2_training` continues from the last `T…_training` number). Status: `todo` → `Q&A` → `in progress` → `done` (or `blocked`).

| ID | Task | Phase | Status | File |
|---|---|---|---|---|
| — | Teacher always sees 16 native-stride frames (pre-template) | — | done (D-023; T=8 → T04_training / S14); merged (D-026) | `teacher_16frame_plan.md` |
| T01_training | Temporal loss matches teacher dynamics | — | Q&A | `T01_training_temporal_dynamics_loss.md` |
| T02_training | Pass `window_t` from the config to the model (S13) | — | Q&A (Q1–Q3 answered; Q2 part done early) | `T02_training_window_t_passthrough.md` |
| T03_training | Retire `train_3d.py`; `Train_3D_full.ipynb` is the training entry point | — | Q&A (all answered) | `T03_training_retire_train_3d_script.md` |
| T04_training | Compute the image half once per training step, so T=8 fits on a T4 (S14) | — | done (D-031; T=8 10.26 GiB on a T4; not merged yet) | `T04_training_shared_image_encoding.md` |
| T05_training | Training monitor module: speed, memory, validation time, NaN and out-of-memory checks in every run (before T04) | — | done (D-028; merged, `547977c`) | `T05_training_train_monitor.md` |
