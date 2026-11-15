# Decisions & Modifications Log

One entry per modification, newest last. Before adding an entry, check earlier ones for anything it overrides or contradicts, and note it under **Affects**.

<!--
Entry format:

## D-NNN — Short title
- **Date:** YYYY-MM-DD
- **Branch:** branch name
- **Decision:** what was changed and why
- **Where:** file(s) / function(s) / cell(s)
- **Affects:** earlier decisions or code it overrides, interacts with, or depends on
-->

## D-001 — Teacher clips use a configurable frame stride, default 4
- **Date:** 2026-10-28
- **Branch:** fix/teacher-full-16frame-clips
- **Decision:** The teacher's 16-frame clip is sampled at `cfg.frame_stride` (default 4, matching VideoMAE-base pretraining) instead of consecutive frames. Short videos fall back to the largest stride that fits, then pad with the last frame. Consecutive frames gave the teacher about 4× less motion than it was pretrained on. Exp 2's linspace sampling was rejected because its spacing varies from video to video.
- **Where:** `sample_clip_indices` / `assemble_clip` in `anyup/data/training/teacher.py`, `UCF101TopNDataset._sample_frames` / `build_loader` (notebook cell 9), `TrainConfig.frame_stride` (cell 3). Implemented in D-006; plan: `docs/Plans/teacher_16frame_plan.md`.
- **Affects:** Changes what every training clip contains, so runs before this are not comparable to runs after it. Exp 2 still samples with linspace, so a smaller train/eval sampling mismatch remains (see the plan's Follow-ups).

## D-002 — Student T gap is documented, not fixed
- **Date:** 2026-10-28
- **Branch:** fix/teacher-full-16frame-clips
- **Decision:** Keep the curriculum at T = 2→4→8 (at most 4 tubelets), even though Exp 2 runs the student on 8 tubelets. Adding a T=16 stage was deferred to keep the teacher fix small; it needs about A100-class VRAM.
- **Where:** `cfg.t_schedule` (notebook cell 3), unchanged; recorded under Follow-ups in `docs/Plans/teacher_16frame_plan.md`.
- **Affects:** Exp 2 results still include extrapolation to a temporal length the student never trained on. Revisit when evaluating the teacher fix.

## D-003 — Temporal-loss fix goes on its own branch, after the teacher fix
- **Date:** 2026-10-28
- **Branch:** fix/teacher-full-16frame-clips
- **Decision:** Implement `docs/Plans/temporal_dynamics_loss_plan.md` on `fix/temporal-dynamics-loss`, branched from `3Dconv` after `fix/teacher-full-16frame-clips` is merged into it. Both plans edit the same training-step code, and keeping them separate lets each change be evaluated on its own.
- **Where:** branch workflow only; no code.
- **Affects:** The temporal-loss plan assumes `gt_feats` / `T_tok` come from the teacher fix's window slicing.

## D-004 — Plan clarifications before execution
- **Date:** 2026-10-29
- **Branch:** fix/teacher-full-16frame-clips
- **Decision:** Five changes after reviewing the plans against the code: (1) the loader skips frame indices that were never decoded and pads instead, so a short video doesn't silently become an all-zero clip through the catch-all `except`; (2) validation explicitly slices frames and teacher features with `take_window` on the fixed centered window, and the validation pass is timed; (3) the temporal plan's λ tuning at step ≈ 10000 happens after merging and isn't a "Done when" condition; (4) the teacher plan must update D-001's "Not implemented yet"; (5) the training-step snippet keeps the `torch.no_grad()` around `sample_tube_crop`.
- **Where:** `docs/Plans/teacher_16frame_plan.md` (log rule, cell 9, cell 13 training + validation, Done when), `docs/Plans/temporal_dynamics_loss_plan.md` (Step 5, Done when). Docs only; no code.
- **Affects:** Refines D-001 (the short-video padding behavior is unchanged, just made explicit for missing frames). Consistent with D-002 and D-003.

## D-005 — Frame selection moves into `teacher.py` so it can be tested
- **Date:** 2026-10-29
- **Branch:** fix/teacher-full-16frame-clips
- **Decision:** The teacher plan adds `assemble_clip(got, idxs)` to `anyup/data/training/teacher.py`, with 3 tests. Cell 9 calls it instead of building the frame list inline, so the short-video fix from D-004 is covered by pytest rather than only by the Colab debug run. It puts frames in index order, skips indices that were never decoded and repeats the last decoded frame; it returns `[]` if nothing was decoded, and the notebook's padding loop then fills zeros.
- **Where:** `docs/Plans/teacher_16frame_plan.md`: the `teacher.py` code block, cell 9 and cell 12 steps, `tests/test_teacher_16frame.py`. Docs only; no code.
- **Affects:** Implements D-004 point (1) as a function instead of inline notebook code; the behavior is the same.

## D-006 — Teacher always sees 16 native-stride frames (implemented)
- **Date:** 2026-10-29
- **Branch:** fix/teacher-full-16frame-clips
- **Decision:** Implemented `docs/Plans/teacher_16frame_plan.md`. VideoMAE now always runs on 16 frames at `cfg.frame_stride` (4) with its position table untouched. The student trains on a random tubelet-aligned T-frame window, and its GT and guidance features are the matching slice of the 16-frame teacher pass. The old `extract_teacher_features`, which interpolated position embeddings for short clips, is deleted. Validation uses a fixed centered 4-tubelet window at batch 1. The run is renamed `run_03` and `resume` is set to `None`.
- **Where:** new `anyup/data/training/teacher.py` + `tests/test_teacher_16frame.py`; `Train_3D_full.ipynb` cells 2 (`REPO_BRANCH`), 3 (`frame_stride`, overrides), 6 (`TEACHER_T`), 9 (dataset/loader), 12 (imports), 13 (training step, all `build_loader` calls, validation). Notebook outputs cleared.
- **Affects:** Implements D-001 and D-004/D-005. run_02 checkpoints and `loss/val_rec` values aren't comparable with run_03 (different GT, clip sampling and validation window). D-002's student-T gap still applies. D-003's temporal-loss branch can start once this is merged into `3Dconv`. The Colab Step 0 check and the GPU debug run (VRAM, it/s, validation time) are still to do.
