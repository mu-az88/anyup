# T07 — Reproducibility Audit + LEGACY Removal

| Field | Value |
|---|---|
| Task ID | T07 |
| Plan | P1 Experiments (`plan.md`), Phase A.5 |
| Branch | `experiments-task02` (continued, your choice) |
| Status | in progress |
| Depends on | T04 (`environment`, `save_results`), T05/T06 (experiment notebooks) |
| Blocks | Phase B (re-runs use the cleaned notebooks) |

## Goal

Close Phase A.5: every experiment run is seeded the same way (one shared, deterministic `set_seed`), and its results JSON says exactly which code, libraries, checkpoint and data produced it. Also remove the `LEGACY` regression code from the three notebooks, since its job (proving the refactor) is done.

## Scope

- In scope:
  - `anyup/utils/seed.py`: warn when the CUBLAS setting comes too late (no other behavior change; `train_3d.py` uses it too).
  - `anyup/metrics.py` `environment()`: library versions, git branch, dirty flag.
  - `anyup/upsamplers.py`: SHA-256 of the AnyUp3D checkpoint and AnyUp2D weights in the meta.
  - `anyup/ade20k.py`: public `cache_meta` (for the stems SHA-1 in Exp 1's config).
  - All three `experiments/*.ipynb`: shared seeding, CUBLAS env in Cell 1, LEGACY code removed, run names that can't overwrite older results, fuller config.
  - Tests for the module changes; 3 T4 smoke runs.
  - `anyup/ade20k.py` `prepare_ade20k`: refuse zip members outside `ADEChallengeData2016/` (added after a security review, see log).
- Out of scope: pinning library versions (Q2); re-running any legacy check (retired → S04); Phase B runs; `results/*.ipynb`; `train_3d.py`.

## Findings

- **A.** Each notebook defines its own `set_seed`; only Exp 1 sets the cuDNN flags. `anyup/utils/seed.py:21` exists but isn't used by the notebooks.
- **B.** `seed.py` sets `CUBLAS_WORKSPACE_CONFIG` with `setdefault` inside `set_seed`. cuBLAS reads it when CUDA starts, so in a notebook (CUDA already started) it has no effect; deterministic mode then can't make cuBLAS deterministic.
- **C.** `environment()` records Python, torch, CUDA, GPU and the commit, but not timm / transformers / torchvision / numpy etc., which Cell 1 installs unpinned.
- **D.** The checkpoint is identified only by path + step.
- **E. Exp 2 / Exp 3 run names don't include `N_CLIPS`.** A 2-clip smoke run writes to the same Drive folder as the full 30-clip T05 run and **overwrites its CSV/JSON** (`new_g224_c16_o8_q4096`, `new_p56_c16_o0_qNone_kinetics`). The T07 smoke runs would have hit this.
- **F.** Exp 1's config lacks `repo_branch` and the data identity; Exp 2/3 have the clip list.

## Q&A (decisions agreed before implementation)

| # | Question | Answer |
|---|---|---|
| 1 | Seeding? | **Shared `anyup.utils.seed.set_seed(seed, deterministic=True)` in all three notebooks**; CUBLAS env set before CUDA starts (Cell 1). |
| 2 | Library versions? | **Record all** in `environment()`; installs stay unpinned. |
| 3 | Checkpoint identity? | **SHA-256 in the wrapper meta** for AnyUp3D and AnyUp2D. |
| 4 | Git / data identity? | **All:** branch + dirty flag in `environment()`; `repo_branch` and ADE20K stems SHA-1 in Exp 1's config. |
| 5 | Legacy re-check? | **None. Legacy checks are retired for good** (their job was to confirm the new pipeline). Logged as S04. |
| 6 | LEGACY code? | **Remove it** from all three notebooks; note in the skipped file (S04). |
| 7 | Smoke runs? | **Yes, 3 T4 smoke runs** (Exp 1 `USE_T4=True`; Exp 2/3 `N_CLIPS=2`). |
| 8 | Branch? | **`experiments-task02`**. |

### Confirmed with this file

| # | Point | Proposal |
|---|---|---|
| 9 | Finding E: new run names | Drop the `legacy`/`new` prefix and add the clip count: Exp 2 `g224_c16_o8_q4096_n30` (smoke `..._n2`), Exp 3 `p56_c16_o0_qNone_kinetics_n30`. Exp 1 already has `tr…_va…`. T05's folders on Drive stay as they are; new runs never write into them. |
| 10 | Dirty flag | `git status --porcelain --untracked-files=no`: only modified tracked files count. Colab's `pip install -e` creates untracked egg-info files, which would otherwise always say dirty. |
| 11 | Versions recorded | torchvision, timm, transformers, numpy, scipy, scikit-image, opencv-python(-headless), pillow, decord (via `importlib.metadata`; `None` if not installed). |

## Design

### `anyup/utils/seed.py`

- Unchanged behavior. New: if `deterministic=True`, CUDA is already initialized and `CUBLAS_WORKSPACE_CONFIG` was not set before, `warnings.warn(...)` that cuBLAS stays non-deterministic and the variable must be set before CUDA starts.

### `anyup/metrics.py`

```python
environment() -> Dict   # + "packages": {name: version | None}, "git_branch", "git_dirty" (bool | None), "cublas_workspace_config"
```

### `anyup/upsamplers.py`

```python
file_sha256(path) -> str                     # streamed, 1 MB blocks
AnyUp3DUpsampler meta: + ckpt_sha256
AnyUp2DUpsampler meta: + weights_sha256      # local file, or the torch.hub cache file of ANYUP2D_URL
```

### `anyup/ade20k.py`

```python
cache_meta(cache_dir, split) -> Optional[Dict]   # public read of <split>_meta.json (None if incomplete)
```

### Notebooks (all three)

- Cell 1 first lines: `os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"` before anything imports torch.
- Local `set_seed` removed → `from anyup.utils.seed import set_seed`; `set_seed(SEED)` (Exp 1: per seed, as now).
- **LEGACY removed:** the flag, legacy-only constants (`COS_TOL`, `N_LEGACY_FRAMES`, `FRAME_STEP`, `LEGACY_STEPS`, `LEGACY_N_VAL`, `LOSS_TOL`, `MIOU_TOL`), the old-code cell, the comparison cell, `record` / `RESULTS`, every `if LEGACY:` branch, and the `legacy_checks` summary key. Header markdown updated. New-protocol lines are kept unchanged.
- Run names as Q9. Config: `repo_branch` everywhere; Exp 1 also `anyup3d_meta` / `anyup2d_meta` (as Exp 2/3, includes the hashes) and `ade20k_stems_sha1` (train / val).
- All three notebooks are edited cell by cell by one script (scratchpad `t07_edit_notebooks.py`, exact-match replacements, fails on any miss).
- Exp 2/3: `RUN_NAME` / `OUT_DIR` (Exp 3 also `MASK_ROOT`) move from Cell 2 to Cell 3, right after `CLIPS`, so the name holds the actual clip count `len(CLIPS)`.
- Cells renumbered after the deletions (Exp 1: 8 cells, Exp 2: 7, Exp 3: 7); "(Cell N)" references in config comments updated.

### Removal check (regression check of the refactor)

For each notebook: a line diff old → new of every code cell. **Pass** = the only changes are deleted legacy lines plus the listed edits (seeding, CUBLAS line, run name, config, header text). I report the diff summary per cell. The 3 smoke runs then show the new-protocol path still runs.

## Steps

1. `seed.py` warning, `environment()`, `file_sha256` + meta, `cache_meta`; tests. You run pytest on Colab.
2. Exp 1 notebook (generator), Exp 2/3 notebooks (scripted edits); removal check diff.
3. You run the 3 T4 smoke runs and paste the outputs (incl. the `env` / meta part of one summary JSON).
4. Update `plan.md`, `project_context.md` (CUBLAS gotcha), skipped file; close out.

## Tests / Regression Checks

pytest (CPU):
- `environment()`: has `packages` with the listed names, `git_branch`, `git_dirty`; dirty flips to True after modifying a tracked file in a temporary git repo (and stays False with only an untracked file).
- `file_sha256` = `hashlib.sha256` of the bytes; AnyUp3D meta `ckpt_sha256` equals it for the test checkpoint; AnyUp2D with `weights=` gives the local file's hash.
- `set_seed` on CPU unchanged (existing `test_seed.py`); the warning can't be tested without CUDA (covered by the smoke run output).
- `cache_meta`: dict for a complete cache, None otherwise.

Colab: the 3 smoke runs end to end on T4; `env.packages`, `git_branch`, `git_dirty`, hashes present in the JSON; no CUBLAS warning.

## Done When

- pytest passes on Colab.
- All three notebooks contain no LEGACY code, pass the removal check, and run end to end on T4.
- S04 is in the skipped file; `plan.md` and `project_context.md` updated.

## Implementation Log

_Updated live during implementation. Per step: what changed, files touched, test results (actual numbers/shapes). Also log mid-implementation questions and answers, changed decisions (old → new, why), and anything added or removed. Sections above are edited in place to match, so the finished file is a report of what was actually done._

- **OK received ("lets do it now"); Q9–11 accepted as proposed.** Status Q&A → in progress.
- **Step 1 (code):**
  - `anyup/utils/seed.py`: warning when `deterministic=True`, CUDA already initialized and `CUBLAS_WORKSPACE_CONFIG` unset; behavior otherwise unchanged.
  - `anyup/metrics.py`: `environment(repo=None)` + `git_branch`, `git_dirty` (tracked files only), `cublas_workspace_config`, `packages` (`ENV_PACKAGES`, via `importlib.metadata`); helpers `_git`, `_package_versions`.
  - `anyup/upsamplers.py`: `file_sha256`; AnyUp3D meta `ckpt_sha256`; AnyUp2D meta `weights_sha256` (hub cache file `<hub dir>/checkpoints/anyup_paper.pth`, or the local `weights` file).
  - `anyup/ade20k.py`: public `cache_meta`.
  - Tests: `test_metrics.py` + `test_environment_git_and_packages` (temp git repo: clean → untracked file still clean → modified tracked file dirty), `test_environment_outside_git`; `test_upsamplers.py` + `test_checkpoint_hashes_in_meta`; `test_ade20k.py` + `test_cache_meta`.
  - **Added (not in the plan): zip member check in `prepare_ade20k`.** An automated security review flagged the T06 download: plain HTTP, no integrity check, and `extractall` into `/content`, which also holds the `/content/anyup` checkout, so a tampered zip could overwrite repo code. Fix: `_check_members` refuses any member outside `ADEChallengeData2016/` (absolute, `..`, other folders) before extracting. Tests: `test_zip_with_foreign_member_is_refused` ×4, `test_clean_zip_extracts`. HTTPS and a pinned SHA-256 are open (asked); `davis.download_davis` uses the same `extractall` pattern (HTTPS URL), not changed.
  - Local pytest (scratchpad venv, CPU, Python 3.14, torch 2.15): `test_ade20k.py` + `test_upsamplers.py` + `test_metrics.py` + `test_seed.py` **92 passed, 1 skipped (CUDA seed test, no GPU), 1 warning** (known 0/0 in `db_eval_iou`).
- **Step 2 (code): LEGACY removed from all three notebooks** (script; the scratchpad was cleared between sessions, so the Exp 1 generator was gone: old → new "regenerate Exp 1", why: no generator left; design updated).
  - Removal check (code lines vs `HEAD` = `8d34f11`): Exp 1 463 → 314 lines (166 removed, 17 added/changed), Exp 2 294 → 184 (127 / 17), Exp 3 411 → 208 (221 / 18). **Every added/changed line is a planned edit**: CUBLAS lines, `set_seed` import, run names, `N_CLIPS` comment, config, `clip_source` / `idx` / `clip_idx` simplified to all frames, cell header renumbering. No other new-protocol line changed. No `LEGACY` / `RESULTS` left; all code cells compile.
  - Exp 1 CPU dry run (new scratchpad harness, fake data, `USE_T4=True` with tiny sizes, 2 seeds × 2 epochs, `RUN_VISUALS=True`): Cells 2–7 run; re-running the main cell resumes with the same results; summary JSON has `anyup3d_meta.ckpt_sha256`, `anyup2d_meta.weights_sha256`, `ade20k_stems_sha1` (train / val), `repo_branch`, and env `git_branch`, `git_dirty`, `cublas_workspace_config`, `packages`. Exp 2/3 not dry-run (need DAVIS + backbones): the smoke runs cover them.
- **Waiting for Colab:** pytest, then the 3 T4 smoke runs.

