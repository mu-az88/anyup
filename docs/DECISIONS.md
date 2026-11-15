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

  The wrapper's `window_t=4` value is unchanged. `tests/test_upsamplers.py` 34 passed. `docs/experiments/experiments.html` was rebuilt with `build_docs.py` after committing (it refuses a dirty tree); it documents commit `99ff830`.
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

## D-019 — Teacher fix Step 0 passed on Colab
- **Date:** 2026-11-15
- **Branch:** fix/teacher-full-16frame-clips (pushed at `fa2f7fd`)
- **Decision:** Ran the teacher plan's Step 0 on Colab, after cells 1–6 in a scratch cell that wasn't saved. Result:
  - `transformers` 5.22.1 (local venv: 5.5.4);
  - `encoder.embeddings.position_embeddings` is a plain `Tensor`, not an `nn.Parameter`, of shape `(1, 1568, 768)`;
  - `VideoMAEEmbeddings.forward` contains no interpolation path. It adds `self.position_embeddings.detach().type_as(...)` directly to the patch embeddings, then applies the optional `bool_masked_pos` mask.

  Same behavior as 5.5.4, as the plan expected, so the fix's assumption holds on Colab: the teacher needs exactly 16 frames × 224² (1568 tokens), and the table is never modified.
- **Where:** Colab only; no code changed. `teacher_16frame_plan.md` header note and `plan.md` Task Index updated.
- **Affects:** D-006 "Done when": Step 0 done; the GPU debug run (T=2/4/8 peak VRAM, it/s, validation time) is still open.

## D-020 — UCF101 top-10 archive rebuilt on Drive
- **Date:** 2026-11-15
- **Branch:** fix/teacher-full-16frame-clips (no code change)
- **Decision:** `MyDrive/anyup3d/ucf101_top10.tar.gz` had been deleted by accident, so cell 9 raised `FileNotFoundError`. It was rebuilt in a separate Colab session with the steps of the original `Data_Preparation_UCF101.ipynb` (deleted in `b42d019`; it came from sayakpaul/Action-Recognition-in-TensorFlow):
  - official UCF101 + `trainlist01` / `testlist01`;
  - the 10 classes with the most training videos.

  Three fixes, so the archive matches what cell 9 expects:
  - the files sit inside a top-level `ucf101_top10/` folder (cell 9 extracts to `/content/` and reads `/content/ucf101_top10/`);
  - real gzip (`tar czf`; the old notebook used `tar cf` under a `.gz` name, which cell 9's `tar -xzf` would reject);
  - a pandas ≥ 2 fix for reading the top-10 class names.

  Result: 10 classes (BoxingPunchingBag, CricketShot, Drumming, HorseRiding, PlayingCello, PlayingDhol, PlayingGuitar, Punch, ShavingBeard, TennisSwing), **1171 train / 459 test videos**, 965 MB. These counts equal the original notebook's saved output `((1171, 2), (459, 2))`, and the only class names saved from the original data (`PlayingCello`, `BoxingPunchingBag`, in the training notebook's old cell 9 output) are both in the list. The files can't be proven byte-identical to the deleted archive.
- **Where:** Google Drive only. The rebuild cell is not in the repo.
- **Affects:** run_03 data. It should be the same as the data run_02 used. run_02 and run_03 aren't comparable anyway (D-006). The validation timing in the teacher debug run uses this test split (459 clips).

## D-021 — Teacher fix GPU debug run (T4): T=2 and T=4 pass, T=8 runs out of memory
- **Date:** 2026-11-15
- **Branch:** fix/teacher-full-16frame-clips (pushed at `fa2f7fd`; no code change)
- **Decision:** Ran the debug pass on a Colab **Tesla T4 (14.6 GiB)**, with temporary scratch cells 12.5/13.5 that weren't saved. The run used 15 steps, 5 per stage, start steps 0/5/10. The plan says 1 step per stage (start steps 0/1/2); 5 steps were used so the speed isn't dominated by start-up. Debug output went to the `debug_teacher` folders on Drive.
  - Cell 9: archive extracted (the rebuilt one, D-020); loader batch `(2, 16, 3, 256, 256)`.
  - **T=2, batch 2:** 5 steps, all losses finite; median 2.87 s/step (0.35 it/s); first step 6.0 s; peak 9.91 GiB allocated / 10.06 GiB reserved (69%).
  - **T=4, batch 1:** 5 steps, all losses finite; median 2.45 s/step (0.41 it/s); first step 3.0 s; peak 9.94 / 10.16 GiB (70%).
  - Curriculum switches T=2→4 at step 5 and 4→8 at step 10, as configured. `loss/temporal` = 0 (warmup), as expected.
  - **T=8, batch 1: `CUDA out of memory` on its first step** (step 10), in the self-consistency forward `pred_aug = anyup(crop_high_5d, gt_aug_low, …)` (cell 13), inside `query_encoder`'s Conv3d. 14.28 GiB allocated by PyTorch at the time.
  - Validation was not reached, so its time is unmeasured.
- **Where:** Colab only. `teacher_16frame_plan.md` header and `plan.md` Task Index updated.
- **Affects:**
  - D-006 "Done when" is **not met** for T=8 (OOM) or the validation time. T=2/4 meet it.
  - Inference, not measured: peak memory tracks the student's frames per step (T=2 × B2 and T=4 × B1 are both 4 frames, with equal peaks; T=8 × B1 is 8 frames), and the self-consistency pass runs the whole student a second time on the same image, keeping its activations for backward. The teacher runs under `no_grad`. So the OOM is most likely the existing memory budget of the training step at 8 frames, not the teacher fix. run_02 stopped near step 6500, before the T=8 stage (`start_step=15000`), so T=8 had probably never run on a T4.
  - As configured, run_03 on a T4 would crash at step 15000. At ≈ 0.4 it/s, 50k steps ≈ 35 h or more. Related to D-002 / S11 (student T gap).

## D-022 — T=8 OOM handled separately: teacher fix accepted on T=2/4; T04_training planned; validation timing rerun
- **Date:** 2026-11-15
- **Branch:** fix/teacher-full-16frame-clips
- **Decision:** Muaaz approved all three recommendations after D-021:
  1. The teacher fix counts as working on T=2/4. Its T=8 debug check moves to T04_training (skipped item S14).
  2. **T04_training** (task file written, Q&A with 5 questions): compute AnyUp's image half (image encoder + RoPE, `query_encoder`, `key_encoder`) once per training step and reuse it for the main and self-consistency passes. It is **not an architecture change**: same layers, weights, checkpoint format and outputs; gradients equal in math, with only rounding-level differences.
  3. Rerun the debug pass with validation at the end of the T=4 stage (`val_every_n_steps=10`), so the validation time is measured before T=8 runs out of memory.
- **Where:** `docs/plan/P1_training_pipeline/T04_training_shared_image_encoding.md` (new), `plan.md` Task Index, `teacher_16frame_plan.md` header note, `docs/skipped_and_future_concerns.md` (S14).
- **Affects:** Changes D-006's "Done when" for merging the teacher fix: T=8 is no longer required there (it's S14 / T04_training). The validation time is still required. T04_training would change `anyup/model.py`, which Munther's experiments use through `forward` (Q2). `forward`'s outputs must stay equal.

## D-023 — Validation time measured; teacher fix "Done when" met
- **Date:** 2026-11-15
- **Branch:** fix/teacher-full-16frame-clips (no code change)
- **Decision:** Reran the T4 debug pass with validation after step 9 (end of the T=4 stage), per D-022, with temporary cells 12.5/13.5 that weren't saved. Results:
  - **Validation:** 50 of 459 test clips in 42 s = **0.83 s/clip → full pass ≈ 6.4 min**. Batch 1, fixed centred 4-tubelet window, no gradients. Peak during validation 1.31 GiB allocated (10.16 GiB reserved, cached from training). `val_rec` = 0.1230 at step 10.
  - T=2: median 2.93 s/step (0.34 it/s), peak 9.91 / 10.06 GiB. T=4: 2.44 s/step (0.41 it/s), peak 9.94 / 10.16 GiB. Both match D-021.
  - **All 10 training-step losses are identical to D-021's run** (same seed), so these steps are reproducible run to run on this setup.
  - T=8 ran out of memory again at step 10, as expected (S14 → T04_training).
  
  With this, every remaining item of the teacher plan's "Done when" is met, except T=8, which D-022 moved out:
  - Step 0 (D-019);
  - pytest 20/20;
  - no position-embedding interpolation;
  - all loaders give 16 frames;
  - finite losses at T=2/4 with VRAM and it/s reported;
  - validation time;
  - log entries.
  
  The teacher fix is **done** and ready to merge into `main` (after `3Dconv` → `main`, D-008).
- **Where:** Colab only; `teacher_16frame_plan.md` header note and `plan.md` Task Index updated.
- **Affects:** Not changed in this branch (the plan says to report, not change): with `val_every_n_steps=1000`, a 50k-step run does about 50 validation passes ≈ 5.3 h, on top of ≈ 35 h of training at ≈ 0.4 it/s. It is a candidate for a later task (fewer or smaller validation passes) if run time matters. Relates to D-021 (T4 throughput) and T04_training (T=8 memory).

## D-024 — Colab saves of the notebook merged, then reverted; debug cells saved as a file
- **Date:** 2026-11-15
- **Branch:** fix/teacher-full-16frame-clips
- **Decision:** Two Colab "Save a copy in GitHub" commits (`c3a242e` 13:25, `0a179db` 14:20, author `mu-az88`) had changed `Train_3D_full.ipynb` on the remote branch, so pushing `5e01baa` was rejected. They added:
  - a "Cell 0" heading and an empty last cell;
  - the Step 0 check cell 6.5 and debug cells 12.5/13.5 (first-run version);
  - saved outputs in 17 cells, plus widget metadata;
  - `window_t: int = 4` → `3` in cell 3.

  Muaaz chose to keep them in history but not in the notebook:
  1. Merged them (`ae88541`, no force push).
  2. Restored the notebook to the reviewed `fa2f7fd` in a new commit (`dc63e01`). The Colab version stays in history as the record of the D-021/D-023 runs.
  3. `window_t = 3` is done in T02_training (with cell 7 passing it).
  4. The debug cells (validation-rerun version, D-023) are saved as `scripts/debug/teacher_debug_cells.py`. It's not importable (it wraps notebook globals); you paste it in, and it keeps the measurement method the same for later runs (T04_training before/after).
- **Where:** branch history (`ae88541`, `dc63e01`); `scripts/debug/teacher_debug_cells.py` (new); `T02_training_window_t_passthrough.md` (log).
- **Affects:**
  - The notebook rules (`docs/working rules/notebooks_version_control.md`: no fixes in Colab, clear outputs) are upheld. The notebook equals the version that the debug runs actually tested.
  - D-012 (`train_3d_full.py`, reference only) is unchanged.
  - Muaaz wants debugging and monitoring in the real training run as a habit, if it costs no memory and little time. A pluggable module that replaces these pasted cells is to be planned as a task.

## D-025 — Debugging as a habit: rule in `CLAUDE.md`; T05_training (training monitor) planned
- **Date:** 2026-11-15
- **Branch:** fix/teacher-full-16frame-clips
- **Decision:** At Muaaz's request:
  - **`CLAUDE.md` Coding Rules, two new rules:**
    1. Whenever useful debugging code can be added (checks, measurements, clear error messages), add it, so every problem is easy to isolate and detect. It must cost no GPU memory and little time, measured.
    2. Such checks live in importable modules with tests (training: `anyup/data/training/monitor.py`), never as temporary cells saved in a notebook.
  - **Working rules:** the same rule added to `docs/working rules/notebooks_version_control.md`.
  - **T05_training planned:** a `TrainMonitor` used in every real training run. It records per-stage it/s, peak allocated/reserved memory, validation time and time left; stops on NaN or infinite losses with context; and explains out-of-memory crashes. One config switch. Status Q&A, 8 questions. It is to be done **before** T04_training, which now measures its before/after through it.
- **Where:** `CLAUDE.md`, `docs/working rules/notebooks_version_control.md`, `docs/plan/P1_training_pipeline/T05_training_train_monitor.md` (new), `T04_training_shared_image_encoding.md` (Depends on, Step 4), `plan.md` Task Index.
- **Affects:** It replaces the paste-in debug cells for future runs. `scripts/debug/teacher_debug_cells.py` (D-024) stays as the historical method behind D-021/D-023. D-010 (`CLAUDE.md` edits) is extended. Munther's sessions read the same `CLAUDE.md` rule.

## D-026 — `3Dconv` and the teacher fix merged into `main`
- **Date:** 2026-11-15
- **Branch:** main
- **Decision:** Muaaz approved both merges (D-008 workflow):
  1. **`3Dconv` → `main`** as `9685b13`. The one conflict, `tests/conftest.py`, was resolved with `3Dconv`'s version (`main`'s defined `pytest_addoption` twice). The merged tree equals `origin/3Dconv` (`6eff213`) exactly.
  2. **`fix/teacher-full-16frame-clips` → `main`** (this merge commit; branch head `a0c3ac6`). No conflicts, and the merged tree equals the fix branch's tree.

  Tests on the merged `main` (local CPU venv, `pytest tests -m "not slow"`): **361 passed, 6 skipped, 1 failed, 2 collection errors**.
  - The failure is `test_smoke_train3d.py::test_train3d_debug_smoke`: `train_3d.py` crashes printing `λ` on the Windows console (cp1256 `UnicodeEncodeError`). It predates both merges, is local to Windows, and the script is being retired (T03_training).
  - The 2 collection errors are `test_davis.py` and `test_metrics.py`: `anyup/metrics.py` imports `cv2`, which the local venv lacks; Colab has OpenCV. A local environment gap, not a code change.
  - The smoke test also wrote two empty 88-byte TensorBoard files into `checkpoints/anyup3d/logs/` (removed). That confirms D-014's guess about where the 2026-10-28 file came from.
- **Where:** `main` history (`9685b13` + this merge).
- **Affects:**
  - `main` now holds Munther's P1 work and the teacher fix (D-006), so new branches start from `main` (T01–T05_training).
  - `main` is **not pushed yet** at the time of writing.
  - Munther should know that `main` is now the base. `3Dconv` stays as it is on GitHub.
