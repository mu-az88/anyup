# Decisions & Modifications Log

One entry per modification, newest last. Before adding an entry, check earlier ones for anything it overrides or contradicts, and note it under **Affects**.

From D-009 on, this log is used together with the task files in `docs/plan/<plan>/TNN_*.md` and `docs/skipped_and_future_concerns.md`. A task file holds that task's Q&A and Implementation Log. This file holds the decision record across tasks and branches. Entries name their task ID in **Where** when they belong to a task.

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
- **Where:** `sample_clip_indices` / `assemble_clip` in `anyup/data/training/teacher.py`, `UCF101TopNDataset._sample_frames` / `build_loader` (notebook cell 9), `TrainConfig.frame_stride` (cell 3). Implemented in D-006; plan: `docs/plan/P1_training_pipeline/teacher_16frame_plan.md`.
- **Affects:** Changes what every training clip contains, so runs before this are not comparable to runs after it. Exp 2 still samples with linspace, so a smaller train/eval sampling mismatch remains (see the plan's Follow-ups).

## D-002 — Student T gap is documented, not fixed
- **Date:** 2026-10-28
- **Branch:** fix/teacher-full-16frame-clips
- **Decision:** Keep the curriculum at T = 2→4→8 (at most 4 tubelets), even though Exp 2 runs the student on 8 tubelets. Adding a T=16 stage was deferred to keep the teacher fix small; it needs about A100-class VRAM.
- **Where:** `cfg.t_schedule` (notebook cell 3), unchanged; recorded under Follow-ups in `docs/plan/P1_training_pipeline/teacher_16frame_plan.md`.
- **Affects:** Exp 2 results still include extrapolation to a temporal length the student never trained on. Revisit when evaluating the teacher fix.

## D-003 — Temporal-loss fix goes on its own branch, after the teacher fix
- **Date:** 2026-10-28
- **Branch:** fix/teacher-full-16frame-clips
- **Decision:** Implement `docs/plan/P1_training_pipeline/T01_training_temporal_dynamics_loss.md` on `fix/temporal-dynamics-loss`, branched from `3Dconv` after `fix/teacher-full-16frame-clips` is merged into it. Both plans edit the same training-step code, and keeping them separate lets each change be evaluated on its own.
- **Where:** branch workflow only; no code.
- **Affects:** The temporal-loss plan assumes `gt_feats` / `T_tok` come from the teacher fix's window slicing.

## D-004 — Plan clarifications before execution
- **Date:** 2026-10-29
- **Branch:** fix/teacher-full-16frame-clips
- **Decision:** Five changes after reviewing the plans against the code: (1) the loader skips frame indices that were never decoded and pads instead, so a short video doesn't silently become an all-zero clip through the catch-all `except`; (2) validation explicitly slices frames and teacher features with `take_window` on the fixed centered window, and the validation pass is timed; (3) the temporal plan's λ tuning at step ≈ 10000 happens after merging and isn't a "Done when" condition; (4) the teacher plan must update D-001's "Not implemented yet"; (5) the training-step snippet keeps the `torch.no_grad()` around `sample_tube_crop`.
- **Where:** `docs/plan/P1_training_pipeline/teacher_16frame_plan.md` (log rule, cell 9, cell 13 training + validation, Done when), `docs/plan/P1_training_pipeline/T01_training_temporal_dynamics_loss.md` (Step 5, Done when). Docs only; no code.
- **Affects:** Refines D-001 (the short-video padding behavior is unchanged, just made explicit for missing frames). Consistent with D-002 and D-003.

## D-005 — Frame selection moves into `teacher.py` so it can be tested
- **Date:** 2026-10-29
- **Branch:** fix/teacher-full-16frame-clips
- **Decision:** The teacher plan adds `assemble_clip(got, idxs)` to `anyup/data/training/teacher.py`, with 3 tests. Cell 9 calls it instead of building the frame list inline, so the short-video fix from D-004 is covered by pytest rather than only by the Colab debug run. It puts frames in index order, skips indices that were never decoded and repeats the last decoded frame; it returns `[]` if nothing was decoded, and the notebook's padding loop then fills zeros.
- **Where:** `docs/plan/P1_training_pipeline/teacher_16frame_plan.md`: the `teacher.py` code block, cell 9 and cell 12 steps, `tests/test_teacher_16frame.py`. Docs only; no code.
- **Affects:** Implements D-004 point (1) as a function instead of inline notebook code; the behavior is the same.

## D-006 — Teacher always sees 16 native-stride frames (implemented)
- **Date:** 2026-10-29
- **Branch:** fix/teacher-full-16frame-clips
- **Decision:** Implemented `docs/plan/P1_training_pipeline/teacher_16frame_plan.md`. VideoMAE now always runs on 16 frames at `cfg.frame_stride` (4) with its position table untouched. The student trains on a random tubelet-aligned T-frame window, and its GT and guidance features are the matching slice of the 16-frame teacher pass. The old `extract_teacher_features`, which interpolated position embeddings for short clips, is deleted. Validation uses a fixed centered 4-tubelet window at batch 1. The run is renamed `run_03` and `resume` is set to `None`.
- **Where:** new `anyup/data/training/teacher.py` + `tests/test_teacher_16frame.py`; `Train_3D_full.ipynb` cells 2 (`REPO_BRANCH`), 3 (`frame_stride`, overrides), 6 (`TEACHER_T`), 9 (dataset/loader), 12 (imports), 13 (training step, all `build_loader` calls, validation). Notebook outputs cleared.
- **Affects:** Implements D-001 and D-004/D-005. run_02 checkpoints and `loss/val_rec` values aren't comparable with run_03 (different GT, clip sampling and validation window). D-002's student-T gap still applies. D-003's temporal-loss branch can start once this is merged into `3Dconv`. The Colab Step 0 check and the GPU debug run (VRAM, it/s, validation time) are still to do.

## D-007 — Munther's `3Dconv` work merged into the teacher-fix branch
- **Date:** 2026-11-15
- **Branch:** fix/teacher-full-16frame-clips
- **Decision:** Merged `origin/3Dconv` @ `6eff213` into the fix branch as merge commit `eb270d3`. That is Munther's P1 experiment tasks T01–T09 (29 Oct to 10 Nov): the new `anyup/` modules for backbones, video, upsamplers, metrics, DAVIS and ADE20K, the new `experiments/` notebooks, thesis edits, `CLAUDE.md`, the task template and the skipped-concerns file. This keeps the branch current before the teacher fix is committed. There were no conflicts: none of his files overlap the teacher-fix files. After the merge, `test_teacher_16frame.py` + `test_losses.py` pass (39/39, local CPU).
- **Where:** branch history only (merge `eb270d3`); teacher fix committed on top as `c6d1036`.
- **Affects:** D-006 is unchanged. Munther moved the `.pth` weights from the repo root to `weights/`. `Train_3D_full.ipynb` loads weights from `DRIVE_ROOT`, so it isn't affected. The test defaults are fixed in D-011. The teacher plan's Follow-up "Exp 2 sampling (linspace)" refers to the old `results/exp2_temporal_consistency.ipynb`, while the new `experiments/exp2_temporal_consistency.ipynb` reads all frames in chunks (T02). Deferred, as skipped item S12.

## D-008 — `main` becomes the base branch
- **Date:** 2026-11-15
- **Branch:** fix/teacher-full-16frame-clips
- **Decision:** Work targets `main`. Each fix is developed, tested and committed on its own `fix/...` branch and merged into `main` only after its Colab debug pass. `main` (`cb4dc52`, 2026-04-13) has one commit that `3Dconv` lacks: a merge of `fa2e910` + `0a60d95`. `3Dconv` contains both parents and an equivalent merge (`9d59c1d`). The two differ only in `tests/conftest.py`: `main`'s version defines `pytest_addoption` twice, and the second definition drops `--manifest_dir` / `--imagenet_val_dir`. `3Dconv`'s version is the correct one. So `3Dconv` → `main` has one conflict, in `tests/conftest.py`, and the resolution is to take `3Dconv`'s version. Not merged yet.
- **Where:** branch workflow; `docs/working rules/notebooks_version_control.md` (base branch example), `docs/plan/P1_training_pipeline/plan.md`, the Branch field of T01_training.
- **Affects:** Overrides D-003's base branch: `fix/temporal-dynamics-loss` branches from `main` after the teacher fix is merged into `main`, not from `3Dconv`. D-003's order (teacher fix first, temporal loss on its own branch) still holds. The teacher plan still says it was branched from `3Dconv` @ `e696743`; that is historically correct and is left as is (D-009). D-011 edits `tests/conftest.py` on this branch, on top of `3Dconv`'s version, so the `main` merge resolution stays "take `3Dconv`'s side."

## D-009 — Adopt Munther's task template alongside this log; one plan folder; separate numbering
- **Date:** 2026-11-15
- **Branch:** fix/teacher-full-16frame-clips
- **Decision:**
  - New tasks use Munther's workflow: a task file from `docs/plan/_templates/task_template.md`, the plan's Task Index, and `docs/skipped_and_future_concerns.md`. `docs/DECISIONS.md` is kept and used alongside them (see the note at the top).
  - **One plan folder:** `docs/Plans/` is removed, and everything lives in `docs/plan/<plan>/`.
  - **Separate numbering:** Muaaz's plans are `P1_training`, `P2_training`, … and his tasks `T01_training`, `T02_training`, …. Munther's `P1` / `T01…T09` and his own later numbers stay unchanged. The S-IDs in the skipped file remain one shared sequence.
  - The teacher plan is kept unchanged except for a header note marking it as the plan used, and moved to `docs/plan/P1_training_pipeline/teacher_16frame_plan.md`.
  - The temporal-loss plan is converted to the template as **T01_training** in **P1_training (Training pipeline)**. It was briefly T10 in `P2_training_pipeline` before the numbering was separated. Its content is unchanged except the base branch (D-008).
  - D-002 (student T gap) is added to the skipped file as S11, and the file's rules now name both Monzir and Muaaz.
- **Where:** `docs/plan/P1_training_pipeline/plan.md` (new), `docs/plan/P1_training_pipeline/T01_training_temporal_dynamics_loss.md` (moved from `docs/Plans/temporal_dynamics_loss_plan.md` and rewritten), `docs/plan/P1_training_pipeline/teacher_16frame_plan.md` (moved from `docs/Plans/`, header note only), `docs/skipped_and_future_concerns.md` (S11, rules wording), this file (top note, plan paths in D-001–D-006 updated to the new location).
- **Affects:** D-003 and D-004 point (3) are now recorded as T01_training Q&A rows 1 and 3. Munther's `CLAUDE.md` still describes only his `P1, P2, …` / global `T` numbering. Muaaz's `_training` numbering is documented in `docs/plan/P1_training_pipeline/plan.md`.

## D-010 — `CLAUDE.md`: decisions log + explicit-commit rule
- **Date:** 2026-11-15
- **Branch:** fix/teacher-full-16frame-clips
- **Decision:** Three changes to Munther's `CLAUDE.md`, at Muaaz's request:
  1. `docs/DECISIONS.md` is added to the docs tree, with a rule: every modification gets an entry, and earlier entries are checked for contradictions.
  2. Muaaz's separate numbering (D-009) is added to the docs tree and the numbering rules: `P<N>_training_<name>/` plans, `TNN_training_<name>.md` tasks, never sharing numbers with Munther's sequence.
  3. The commit rule. Old: "I create, commit, push and merge the branch myself … You don't commit; you tell me what to commit". New: the assistant does not commit until the developer explicitly says so ("commit the changes", "go ahead and commit"); the word "commit" must be in the request, and the context must mean committing now. Push and merge stay with the developer.
- **Where:** `CLAUDE.md`, "Docs Structure" section.
- **Affects:** D-009 (makes the decisions log and the `_training` numbering part of the shared rules). Munther's sessions now follow the same commit rule; he should be told.

## D-011 — Test checkpoint defaults point to `weights/`
- **Date:** 2026-11-15
- **Branch:** fix/teacher-full-16frame-clips
- **Decision:** `tests/conftest.py` defaults to `weights/anyup_multi_backbone.pth` and `weights/anyup3d_init.pth` (new `_WEIGHTS_DIR`) instead of the repo root, following Munther's move of the `.pth` files. The usage examples in `scripts/load_2d_weights.py` and `tests/test_weight_loading.py` are updated to match. Before this, `test_weight_loading.py` skipped silently.
- **Where:** `tests/conftest.py` (`_WEIGHTS_DIR`, `_DEFAULT_CKPT_2D/3D`, docstring, skip message), docstrings of `scripts/load_2d_weights.py` and `tests/test_weight_loading.py`.
- **Affects:** D-007. With the checkpoints found, `test_weight_loading.py` now **errors** (7/7). The tracked `weights/anyup3d_init.pth` (unchanged since `d3ad91b`, 2026-04-13) has `t_k=1` kernels (e.g. `aggregation.0.conv.weight` (128, 256, 1, 3, 3)). `AnyUp()` defaults to `t_k=3` since `d96010c` (2026-04-27; the default was `t_k=1` before). That mismatch predates this change, and the same file sat in the repo root before the move. Only this test uses `anyup3d_init.pth`: `Train_3D_full.ipynb` (cell 7, `build_3d_state_dict`) and `train_3d.py` (`load_2d_weights_into_3d`, `t_k=3`) inflate the 2D weights at `t_k=3` themselves, and the experiments load trained checkpoints. Not fixed here (fixed in D-013).

## D-012 — Colab export tracked as `train_3d_full.py`
- **Date:** 2026-11-15
- **Branch:** fix/teacher-full-16frame-clips
- **Decision:** The untracked `train_3d_full (1).py` (Colab "Download .py" export of `Train 3D full.ipynb`, 2026-10-18) is renamed to `train_3d_full.py` and staged, at Muaaz's request. **Purpose: a read-only reference for the assistant**, a flat view of the training notebook's structure and contents. It isn't meant to be run, edited or used as a training entry point.
- **Where:** `train_3d_full.py` (repo root).
- **Affects:** It predates the teacher fix (D-006): it still has `REPO_BRANCH = "3Dconv"`, `experiment_name`/`resume` for run_02, consecutive-frame loading in cell 9, and the old `extract_teacher_features` with position-embedding interpolation. So it shows the notebook **before** D-006, not the current cells 2, 3, 6, 9, 12 and 13. It also contains `!pip` / `!tar` lines, so it isn't valid Python. `docs/working rules/notebooks_version_control.md` names this file as an example of a drifting copy; here it is kept on purpose, as reference only. `Train_3D_full.ipynb` remains the training entry point.

## D-013 — `weights/anyup3d_init.pth` rebuilt at `t_k=3`
- **Date:** 2026-11-15
- **Branch:** fix/teacher-full-16frame-clips
- **Decision:** Rebuilt the tracked 3D init checkpoint with `scripts/load_2d_weights.py --checkpoint weights/anyup_multi_backbone.pth --output weights/anyup3d_init.pth --t_k 3`, so it matches `AnyUp()`'s default `t_k=3`. Summary: 46 tensors copied, 26 Conv2d→Conv3d center-init, 1 RoPE (2,d)→(3,d), 0 zero-init, 0 skipped, 0 mismatch. SHA-256 old `14e59b8b…` (`t_k=1`, `aggregation.0.conv.weight` (128, 256, 1, 3, 3)) → new `8d14bbf6…` ((128, 256, 3, 3, 3)). `test_weight_loading.py`: 7 errors → **7 passed**. (Windows note: the script's summary print needs `PYTHONIOENCODING=utf-8`, or it crashes on `→` before saving.)
- **Where:** `weights/anyup3d_init.pth`.
- **Affects:** Fixes the mismatch found in D-011. Training and the experiments don't read this file (D-011), so no trained or measured result changes. `weights/anyup_multi_backbone.pth` and `weights/anyup_2d.pth` are byte-identical (SHA-256 `b6cc407d…`). `anyup_2d.pth` is the 2D checkpoint the notebook loads (from Drive), so the test now checks the same 2D weights training starts from, assuming the Drive copy is that file too.

## D-014 — Empty TensorBoard log deleted
- **Date:** 2026-11-15
- **Branch:** fix/teacher-full-16frame-clips
- **Decision:** Deleted the untracked `checkpoints/anyup3d/logs/events.out.tfevents.1793195517.LAPTOP-A4DTS5PG.13688.0`. It was 88 bytes, only the event-file header and no logged values. It was created on this laptop on 2026-10-28 15:51, most likely by a local `train_3d.py` run (the only code that writes to `<checkpoint_dir>/logs` with `checkpoint_dir: checkpoints/anyup3d/`) that stopped before logging. It held no information.
- **Where:** `checkpoints/anyup3d/logs/` (the six tracked April log files are untouched).
- **Affects:** None. No `.gitignore` rule was added for `checkpoints/`, so new local logs will still show up as untracked.

## D-015 — `window_t` never reached the model: logged (S13), fix planned (T02_training)
- **Date:** 2026-11-15
- **Branch:** fix/teacher-full-16frame-clips
- **Decision:** `Train_3D_full.ipynb` cell 3 sets `window_t = 4`, but cell 7 never passes it to `AnyUp(...)`, in every notebook version in git. The model therefore used its default: `None` until `d96010c` (2026-04-27 12:54 UTC), 3 after. **Training is unaffected:** the mask allows `|Δ tubelet| ≤ window_t`, and training has at most 4 tubelets, so 3, 4 and `None` give the same mask. Evaluation with 8-tubelet chunks is affected. Munther's wrapper uses `window_t=4` ("training used 4"), a value the model was never constrained by, and how much Exp 0–3 change with it is unmeasured. Logged as S13. The fix is planned as T02_training (status Q&A, 3 open questions), and Munther is to be told.
- **Where:** `docs/skipped_and_future_concerns.md` (S13), `docs/plan/P1_training_pipeline/T02_training_window_t_passthrough.md` (new), `docs/plan/P1_training_pipeline/plan.md` (Task Index). No code changed.
- **Affects:** Munther's T03 (`anyup/upsamplers.py:40,47`) and `docs/context/project_context.md:28` state "training used 4", which is not what happened. Every checkpoint's saved `cfg` records `window_t=4`, not the value actually used. Related to D-002 / S11: a T=16 stage is where `window_t` would first affect training. Corrects my earlier claim in this session that step6500 was evaluated "with a different window than it was trained with": it was trained with no effective window, not a different one.

## D-016 — `window_t` answers (T02_training Q1–Q3); notes corrected now
- **Date:** 2026-11-15
- **Branch:** fix/teacher-full-16frame-clips
- **Decision:** Muaaz answered T02_training:
  - **Q1:** training uses `window_t = 3`.
  - **Q2:** correct the "training used 4" notes now.
  - **Q3:** cell 7 prints the model's real `window_t`.

  Q2 is done on this branch, ahead of the rest of T02, and touches only comments and notes:
  - `anyup/upsamplers.py`: the `ANYUP3D_ARCH` comment and the `window_t` line comment; the unit is "tubelets", not "frames".
  - `docs/context/project_context.md:28`.
  - `docs/experiments/doc_content.py`: the Exp 1 protocol line.
  - `docs/plan/P1_experiments/plan.md:32`: an inline correction.

  The wrapper's `window_t=4` value is unchanged. `tests/test_upsamplers.py` 34 passed. `docs/experiments/experiments.html` still has the old text: rebuild it with `build_docs.py` after committing (it refuses a dirty tree).
- **Where:** the files above; `docs/plan/P1_training_pipeline/T02_training_window_t_passthrough.md` (Q&A, Design, Steps, Log); S13 "how to check" updated.
- **Affects:** D-015. Changes Munther's code comments and docs (T03's `ANYUP3D_ARCH` note, `project_context.md`) at Muaaz's request; Munther's closed task files T01/T03/T06 are left as historical records. Cells 3/7 of the notebook are unchanged until T02_training is implemented on its own branch.

## D-017 — T03_training planned: retire `train_3d.py`
- **Date:** 2026-11-15
- **Branch:** fix/teacher-full-16frame-clips
- **Decision:** Task file for making `Train_3D_full.ipynb` the only 3D training entry point and retiring `train_3d.py`, its 3 test files (74 tests), `config/anyup3d_train.yaml`, and possibly the smoke-run files in `checkpoints/anyup3d/`. Status Q&A with 6 open questions; Q1 is Munther's agreement. Nothing moved yet.
- **Where:** `docs/plan/P1_training_pipeline/T03_training_retire_train_3d_script.md` (new), `plan.md` Task Index.
- **Affects:** Would answer Munther's P1 open question "which training entry point is canonical". Must not touch `train.py` (2D, hydra), `anyup/data/video_dataset.py` / `dataloader.py` (used elsewhere) or `train_3d_full.py` (D-012).

## D-018 — T03_training answers
- **Date:** 2026-11-15
- **Branch:** fix/teacher-full-16frame-clips
- **Decision:** Muaaz answered T03_training:
  - **Q1:** Munther agreed that `Train_3D_full.ipynb` is the only 3D training entry point.
  - **Q2:** move `train_3d.py` and `config/anyup3d_train.yaml` to `trash can/`.
  - **Q3:** delete its 3 test files (74 tests).
  - **Q4:** retire the reusable pieces with the file.
  - **Q5:** keep the smoke-run files in `checkpoints/anyup3d/`.
  - **Q6:** add a "retired" note to `docs/training_guide.md`.

  Nothing is moved yet. Implementation waits for the OK, on its own branch.
- **Where:** `docs/plan/P1_training_pipeline/T03_training_retire_train_3d_script.md` (Q&A, Scope, Design, Log), `plan.md` Task Index.
- **Affects:** D-017. Q5 keeps the `checkpoints/anyup3d/` files tracked, consistent with D-014 (no `.gitignore` rule).
