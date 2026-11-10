# T07 — Reproducibility Audit + LEGACY Removal

| Field | Value |
|---|---|
| Task ID | T07 |
| Plan | P1 Experiments (`plan.md`), Phase A.5 |
| Branch | `experiments-task02` (continued, your choice) |
| Status | done |
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
- **pytest run 1 (Colab, Python 3.13.16, pytest 8.4.2): `test_ade20k.py` + `test_upsamplers.py` + `test_metrics.py` + `test_seed.py` 93 passed, 0 failed, 0 skipped, 1 warning, 15.44 s.** The CUDA seed test skipped locally runs and passes on the Colab GPU. The warning is the known 0/0 in `db_eval_iou`. Step 1 done.
- **Exp 1 smoke run 1 (T4, `USE_T4=True`: batch 4, 256 train / 64 val, 1 epoch, seed 42): main loop 83 s, runs end to end.** Only the main-loop output was pasted (no summary / JSON yet).

  | condition | loss | mIoU | train | val | peak | T06 smoke (same settings, before T07) |
  |---|---|---|---|---|---|---|
  | bilinear | 1.4673 | 5.97 | 11 s | 2 s | **4.1 GiB** | 1.4673 / 5.97, 7 s, 1.0 GiB |
  | anyup2d | 1.3905 | 6.37 | 25 s | 6 s | 2.2 GiB | 1.3905 / 6.37, 24 s, 2.2 GiB |
  | anyup3d | 1.4070 | 6.10 | 31 s | 8 s | 1.7 GiB | 1.4070 / 6.10, 26 s, 1.7 GiB |

  - **Loss and mIoU equal the T06 smoke run at the printed precision** (4 / 2 decimals), so removing LEGACY and switching to deterministic seeding didn't change the Exp 1 path at that level.
  - One expected warning: `nll_loss2d_forward_out_cuda_template does not have a deterministic implementation` (`warn_only=True`, as T06 Q13 anticipated). Exp 1 training is therefore still not bit-deterministic on CUDA.
  - **Unexplained: bilinear peak memory 1.0 → 4.1 GiB**; the AnyUp conditions are unchanged. Likely from deterministic algorithms (a different kernel or workspace for the first condition), not verified. Harmless on a T4 at batch 4; the A100 batch-16 estimate (≈ 9 GiB) is still only an estimate.
  - Runtime +5 to +60 % per condition (bilinear 7 → 11 s, anyup3d 26 → 31 s), consistent with deterministic kernels; one run each, so noisy.
- **Exp 1 smoke run 1, Cell 6 (summary):** dataset mIoU bilinear 5.97 / anyup2d 6.37 / anyup3d 6.10 (1 seed, std 0); per-image mIoU 21.55 / 23.09 / 22.25 (64 images, 0 dropped). anyup3d vs bilinear per image Δ +0.697, better 38/64, sign p2 0.0722, Wilcoxon p2 0.00277; vs anyup2d Δ −0.846, better 23/64, sign p2 0.0925, Wilcoxon p2 0.00409. INFO only (64 steps). Saved to `results/T06/exp1/T4_step6500_s224_b4_q2048_tr256_va64_e1/` (new name without the `new_` prefix, as Q9).
- **Exp 2 smoke run 1 (T4, `N_CLIPS=2`): runs end to end; saved to `results/T05/exp2/g224_c16_o8_q4096_n2/`**, its own `_n2` folder, so T05's 30-clip `new_g224_c16_o8_q4096` is no longer at risk (finding E fixed).
  - Means over the 2 clips (all / within / cross): raw 0.8976 / 0.8943 / 0.9100, bilinear 0.9266 / 0.9241 / 0.9361, anyup2d 0.9321 / 0.9303 / 0.9391, anyup3d 0.9825 / 0.9874 / 0.9636.
  - **Equal to the T05 smoke run at 4 decimals:** the means of T05's per-clip values (bike-packing, blackswan) are raw 0.8976, bilinear 0.9266, anyup2d 0.9321, anyup3d 0.9825 (all), anyup3d within 0.9874 / cross 0.9636. So LEGACY removal + deterministic seeding left the Exp 2 path unchanged at that precision.
  - Paired: anyup3d vs raw / bilinear / anyup2d Δ +0.0849 / +0.0559 / +0.0503, improved 2/2, p2 0.5 (minimum at n = 2).
- **Exp 3 smoke run 1 (T4, `N_CLIPS=2`): ran through to the saved summary JSON** (`time_utc` 12:34:13). Its J&F lines were not pasted, so they are not compared with T05 here.
- **JSON fields (Exp 2 full JSON + Exp 3 `env` / `anyup3d_meta`, same Colab image):** all new fields are written.
  - `env`: commit `fe68b70`, `git_branch` experiments-task02, **`git_dirty` false** (so the untracked egg-info files from `pip install -e` are correctly ignored, Q10), `cublas_workspace_config` ":4096:8"; Python 3.13.16, torch 2.14.1+cu130, CUDA 13.0, Tesla T4; packages torchvision 0.29.1+cu130, timm 1.0.31, transformers 5.22.1, numpy 2.4.6, scipy 1.16.3, scikit-image 0.25.2, opencv-python / -headless 5.0.0.93, pillow 11.3.0, decord not installed (`null`; DAVIS frames are JPEG folders, decord isn't needed).
  - `anyup3d_meta.ckpt_sha256` (`anyup3d_step6500.pth`) = `dbdc32f26e8d5c4021536c875c5249464cd0b9306c3afae0f6eebfa0466149e3`, same in Exp 2 and Exp 3. `anyup2d_meta.weights_sha256` (`anyup_paper.pth`) = `9d035c0f27114a6f32bdd3d8ed93b6cd39dad4b8f8bf94e69fddbfbf022901b2`. Both recorded in `project_context.md`.
  - Exp 2 config has `RUN_NAME` `g224_c16_o8_q4096_n2`, `repo_branch`, and no `LEGACY` key.
- **No CUBLAS warning in any of the 3 runs** (Monzir). The only warning was Exp 1's expected `nll_loss2d` one.
- **ADE20K zip integrity: not checked, by Monzir's choice** ("do not check the zip, it's ok to just use it") → `docs/skipped_and_future_concerns.md` S05. The member check added in Step 1 stays. `davis.download_davis` left unchanged (no member check), also noted in S05.
- **Close-out.** All Done When items met: pytest 93 passed on Colab; no LEGACY code in the three notebooks; removal check passed; all three ran end to end on T4, with Exp 1 and Exp 2 equal to their pre-T07 smoke runs at the printed precision; S04 + S05 logged; `plan.md` and `project_context.md` updated. Status → done.
  - Left over: Exp 1 is still not bit-deterministic on CUDA (`nll_loss2d`, warn only); the bilinear peak-memory jump 1.0 → 4.1 GiB in Exp 1 is unexplained; Exp 3's smoke J&F not compared with T05 (not pasted).

