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
- **Where:** planned in `docs/Plans/teacher_16frame_plan.md`: `sample_clip_indices` in `anyup/data/training/teacher.py`, `UCF101TopNDataset._sample_frames` / `build_loader` (notebook cell 9), `TrainConfig.frame_stride` (cell 3). Not implemented yet.
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
