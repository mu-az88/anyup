# T03_training — Retire `train_3d.py`; `Train_3D_full.ipynb` Is the Training Entry Point

| Field | Value |
|---|---|
| Task ID | T03_training |
| Plan | P1_training Training pipeline (`plan.md`) |
| Branch | `chore/retire-train-3d-script`, from `main` `1cec4e7` (created by the assistant, as in T02 Q4) |
| Status | done (D-034); merged into `main` (`7820013`) |
| Depends on | — (Munther agreed, Q1) |
| Blocks | — |

## Goal

Have one training pipeline. `Train_3D_full.ipynb` (tuned for memory, file handling and the other fixes, and the source of every real checkpoint) is the entry point. `train_3d.py`, the original design that was never used for a real run, is retired together with everything that only exists to serve it.

## Scope

- In scope:
  - `train_3d.py` (repo root).
  - Its tests: `tests/test_smoke_train3d.py`, `tests/test_train_infrastructure.py`, `tests/test_gt_extractor_integration.py`.
  - Its config: `config/anyup3d_train.yaml`.
  - Answering Munther's open question in `docs/plan/P1_experiments/plan.md` ("Decide in P2 which training entry point is canonical") with a pointer to this task.
  - Notes that tell people to run it: `docs/training_guide.md` (lines 177, 180).
- Out of scope:
  - `train.py` (the 2D AnyUp training, adapted from JAFAR, hydra + `config/base.yaml` and `config/{backbone,dataloader,dataset,model,optimizer}/`). It's a separate pipeline; nothing here touches it.
  - `anyup/data/video_dataset.py`, `anyup/data/dataloader.py` and their tests: also used by `anyup/backbones.py` and others, so they stay.
  - `anyup/utils/seed.py` and `tests/test_seed.py` (shared; `train_3d.py` only imports them).
  - `train_3d_full.py`: stays as the read-only reference (D-012).
  - Moving notebook logic into `anyup/` modules.
  - The smoke-run files in `checkpoints/anyup3d/` (Q5: keep).

## Findings

- **A. No checkpoint came from it.** run_01/run_02 come from `Train_3D_full.ipynb` (key `"anyup"`, pickled `TStage`, `cfg`). `train_3d.py` saves `model_state_dict` (Munther, P1 plan "Issues for P2").
- **B. Its training task is degenerate.** `pred_features = model(video, gt_features, output_size=(H_tok, W_tok))` (`train_3d.py:480`): the model gets the teacher's 14×14 features as input and must output 14×14 against the same features. Input equals target, so there is nothing to learn. The notebook downsamples low-res-crop features to 7×7 and upsamples to 14×14 against high-res-crop features.
- **C. It doesn't run as documented.** The default `--config configs/anyup3d_train.yaml` points to a folder that doesn't exist (it's `config/`). `test_smoke_train3d.py` and `docs/training_guide.md` use the same wrong path.
- **D. It is behind the notebook on every fix.** It has the old teacher with position-embedding interpolation (fixed in the notebook by D-006). It has no real validation (`run_validation` returns `inf`) and a random-noise stub loader when `video_dataset` is missing. It runs in fp32 with no LR schedule. It builds `AnyUp()` without `window_t` (S13).
- **E. Its tests cover only its own code.** `test_train_infrastructure.py` tests `TCurriculumScheduler`, `save/load_checkpoint`, `BestCheckpointTracker`, `temporal_lambda`, the stub loader, `get_loader`, `build_gt_extractor` and `run_validation` (all from `train_3d.py`, which it imports via `exec`), plus 4 `set_seed` tests that duplicate `tests/test_seed.py`. `test_gt_extractor_integration.py` also `exec`s `train_3d.py`. None of this code runs in the notebook.
- **F. Some of it is well built.** The step-monotone curriculum scheduler, the atomic checkpoint write (`.tmp` + `os.replace`), best-checkpoint tracking, and `load_checkpoint`'s `t_k` center-expansion when kernel sizes differ. The notebook has simpler equivalents of the first three; it has no `t_k` expansion on resume.
- **G. pytest has no `testpaths`** (`pyproject.toml` `[tool.pytest.ini_options]` sets only `markers`). A test file moved under `trash can/` would still be collected by a bare `pytest` and fail.
- **H. Its smoke runs left files in the repo.** `test_smoke_train3d.py` (first test) and earlier manual runs write to the real `checkpoints/anyup3d/`. Tracked there: `final.pth` and `step_000010.pth` (≈ 10 MB each, 2026-04-14/15, 10 debug steps) and 6 TensorBoard logs from April. They're smoke-run output, not trained models.

## Q&A (decisions agreed before implementation)

| # | Question | Answer |
|---|---|---|
| 1 | Has Munther agreed that `Train_3D_full.ipynb` is the only training entry point and `train_3d.py` is retired? (His file; it answers an open question in his P1 plan.) | **Yes, he agreed** (Muaaz, 2026-11-15). |
| 2 | Retire how? (a) move to `trash can/` (his convention for retired files; git history keeps everything either way); (b) delete. | **(a) move to `trash can/`** (Muaaz, 2026-11-15). |
| 3 | The three test files? (a) delete them (they test only retired code; history keeps them); (b) move them to `trash can/` **and** add `testpaths = ["tests"]` to `pyproject.toml`, so pytest doesn't collect them (Finding G). | **(a) delete** (Muaaz, 2026-11-15). |
| 4 | The reusable pieces (Finding F): (a) retire them with the file; (b) first move them into an `anyup/` module with their tests. The notebook doesn't use them today. | **(a) retire them** with the file (Muaaz, 2026-11-15). If a later task moves notebook logic into modules, take them from `trash can/train_3d.py` or git history then. |
| 5 | The smoke-run files in `checkpoints/anyup3d/` (Finding H): (a) remove them from git (`git rm`) and add `checkpoints/` to `.gitignore`; (b) leave them. | **(b) keep them** (Muaaz, 2026-11-15). Untouched; no `.gitignore` change. |
| 6 | `docs/training_guide.md` (an older reference guide, per `CLAUDE.md`): (a) add a short note at the `train_3d.py` section saying it's retired, pointing to `Train_3D_full.ipynb`; (b) leave it. | **(a) add the note** (Muaaz, 2026-11-15). |

## Design

- `git mv train_3d.py "trash can/train_3d.py"`; `git mv config/anyup3d_train.yaml "trash can/anyup3d_train.yaml"`.
- `git rm tests/test_smoke_train3d.py tests/test_train_infrastructure.py tests/test_gt_extractor_integration.py`.
- `docs/plan/P1_experiments/plan.md`, "Issues for P2" bullet on `train_3d.py`: add an inline `_(…)_` note, "decided in T03_training: `Train_3D_full.ipynb` is canonical; `train_3d.py` retired".
- `docs/training_guide.md`: a note above the `train_3d.py` commands.
- `checkpoints/anyup3d/` is not touched (Q5).

## Steps

1. Before removing anything: run the full `pytest` and save the pass/fail/skip counts (baseline).
2. Move / remove files as answered.
3. Search the repo for any remaining reference to the retired names (`train_3d.py`, `train_3d import`, `anyup3d_train.yaml`, `build_gt_extractor`, `TCurriculumScheduler`) outside `trash can/`, git history and closed task files; fix or report each.
4. Run the full `pytest` again.

## Tests / Regression Checks

- Full `pytest` before vs after: the only differences are the 74 removed tests (smoke 2, infrastructure 65, GT-extractor integration 7). No other test changes status.
- `grep` (Step 3) finds no live reference.
- No change to `Train_3D_full.ipynb`, `anyup/` or `train.py` (`git diff --stat` shows only the files listed in Design).

## Done When

- All Steps done, with both pytest counts recorded below.
- Munther's open question in his P1 plan has its answer note.
- `docs/DECISIONS.md` entry; this file's log and the Task Index are up to date.

## Implementation Log

_Updated live during implementation. Per step: what changed, files touched, test results (actual numbers/shapes). Also log mid-implementation questions and answers, changed decisions (old → new, why), and anything added or removed. Sections above are edited in place to match, so the finished file is a report of what was actually done._

- **2026-11-15: task file written** (D-017), from reading the code only; nothing has been run or moved.
- **2026-11-15: Q1–Q6 answered** (agreed / move / delete / retire / keep / add note). Scope and Design updated: the Q5 option removed, `checkpoints/anyup3d/` out of scope (D-018). Waiting for your OK to implement on `chore/retire-train-3d-script`.
- **2026-11-26: OK'd** (Muaaz: "do T03"). Re-checked first: Q1–Q6 unchanged; branch `chore/retire-train-3d-script` created from `main` `1cec4e7`. Status → in progress.
- **2026-11-26: Step 1 done (baseline).** Full `pytest` (local venv, Python 3.12.10, torch 2.11.0+cpu, `--continue-on-collection-errors`): **415 passed, 5 skipped, 2 errors**, 420 collected, 307 s. The 2 errors are `test_davis.py` / `test_metrics.py` collection (`cv2` not in the local venv; known, unrelated). The three files to delete hold 2 + 65 + 7 = 74 tests, as in Tests.
  - Side effect (Finding H): `test_smoke_train3d.py` overwrote the tracked `checkpoints/anyup3d/final.pth` and `step_000010.pth` and wrote a new 5 KB tfevents file in `checkpoints/anyup3d/logs/`. Restored both `.pth` files to `HEAD` (`git checkout --`) and deleted the new tfevents file (created by this run), so `checkpoints/anyup3d/` is unchanged (Q5).
- **2026-11-26: Step 2 done.** `git mv train_3d.py "trash can/train_3d.py"`, `git mv config/anyup3d_train.yaml "trash can/anyup3d_train.yaml"`, `git rm` of the three test files.
- **2026-11-26: Step 3 done.** `git grep` for the retired names outside `trash can/`: **no reference in any `.py`, `.ipynb`, `.yaml`, `.toml` or other code/config file.** Doc references:
  - Fixed: `docs/training_guide.md` (note above the Step 4 commands, Q6); `docs/plan/P1_experiments/plan.md:39` (inline `_(…)_` answer note, Munther's open question; it also points to D-033 for the `window_t=4` claim in the same bullet).
  - Left as historical records: `docs/DECISIONS.md` (D-007, D-017, D-031, D-033), closed task files (P1_experiments T01 `config/anyup3d_train.yaml:12`, T02, T03, T07; P1_training T02), and this file.
- **2026-11-26: Step 4 done.** Full `pytest` again: **341 passed, 5 skipped, 2 errors** (same 2 `cv2` errors), 346 collected, 31 s. Collected test IDs before vs after: the only removed IDs are the 74 from the three deleted files (7 / 2 / 65), none added; passed 415 → 341 (−74), skipped 5 → 5. `checkpoints/anyup3d/` unchanged (the smoke test is gone).
  - `git diff --stat`: only the files in Design + the two doc notes (`Train_3D_full.ipynb`, `anyup/`, `train.py` untouched).
- **2026-11-26: close-out.** All Done When items met: Steps 1–4 with both pytest counts recorded; the answer note in Munther's P1 plan; D-034; this log and the Task Index. Status → done. Left: the merge into `main` (your call). Note: tests now run in about 31 s instead of about 5 min, and a full local `pytest` no longer modifies tracked files in `checkpoints/anyup3d/`.
- **2026-11-26: merged into `main`** (`7820013`, merge commit, `--no-ff`; branch commit `9973d07`) at Muaaz's request ("commit, merge. push and delete the branch"). Full `pytest` on the merged `main`: 341 passed, 5 skipped, 2 errors (the `cv2` ones), 31.9 s, as on the branch. Branch `chore/retire-train-3d-script` deleted locally (never pushed to GitHub). `CLAUDE.md`'s Active Context change was left out of this commit (open question to Muaaz: commit it or move it to `CLAUDE.local.md`).
