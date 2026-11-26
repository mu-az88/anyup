# T02_training — Pass `window_t` from the Config to the Model

| Field | Value |
|---|---|
| Task ID | T02_training |
| Plan | P1_training Training pipeline (`plan.md`) |
| Branch | `fix/window-t-passthrough`, from `main` `b6dac9c` (Q4; the teacher fix, T04 and T01 are merged) |
| Status | done (D-033); merged into `main` (`dfb14dc`) |
| Depends on | — (independent code; branch order only avoids notebook merge conflicts with the teacher fix) |
| Blocks | Any training stage with more than 4 tubelets (T=16, skipped item S11): there `window_t` starts to matter |

## Goal

Make the `window_t` set in the training config actually reach the model, so the trained model and the `cfg` saved in every checkpoint agree. Also record what past runs really used.

## Scope

- In scope:
  - `Train_3D_full.ipynb` cell 7: pass `window_t=cfg.window_t` to `AnyUp(...)`.
  - New `tests/test_window_t.py`: shows when `window_t` changes the output and when it doesn't (see Tests).
  - Facts in this file and in S13 about which value run_01/run_02 effectively used.
  - Correcting the "training used 4" notes (Q2): comments in `anyup/upsamplers.py`, `docs/context/project_context.md`, `docs/experiments/doc_content.py`, and an inline correction in `docs/plan/P1_experiments/plan.md`. Munther's closed task files (T01, T03, T06) stay as historical records.
- Out of scope:
  - The wrapper's evaluation value (`anyup/upsamplers.py`, `window_t=4`): unchanged.
  - Measuring how Exp 0–3 change with `window_t` (S13).
  - `train_3d.py`, which builds `AnyUp()` with defaults in the same way.

## Findings

- **A. The setting never reaches the model.** Cell 3 sets `window_t: int = 4`. Cell 7 builds `AnyUp(input_dim, qk_dim, num_heads, window_ratio, kernel_size_lfu, t_k)` without `window_t`, so the model uses its default. This is true in every notebook version in git (`7e12492`, `2e249d2`, `e696743`, `c6d1036`).
- **B. The default changed during run_02.** `anyup/model.py` defaulted to `window_t=None` (no temporal limit) until `d96010c` (2026-04-27 12:54 UTC), and to `3` after that. Colab clones the repo fresh in each runtime. So each run_02 session used `None` or `3`, depending on when it cloned. By 2026-04-30, run_02 had already been resumed from step 2000 (saved notebook output).
- **C. In training this made no difference.** The temporal mask allows `|t_q − t_k| ≤ window_t`, where `t` is the tubelet index (`anyup/modules/attention_masking3d.py:103-107`; `t` = the features' `T_tok`, `anyup/model.py:83-88`). Training has at most 4 tubelets (T=8 frames), so the largest distance is 3. So `window_t` = 3, 4 and `None` all give the same full temporal mask, and run_01/run_02 effectively trained with **no temporal restriction**.
- **D. In evaluation it can make a difference.** With 8-tubelet chunks (16 frames, as in Exp 2/3), the three values differ:
  - 3 allows distances up to 3, the largest the model ever saw in training;
  - 4 (Munther's wrapper default) allows up to 4;
  - `None` allows up to 7.
  
  The model was never trained above 4 tubelets (D-002, S11), so none of these is "the trained value". Which one evaluation should use is an open choice.
- **E. The records are misleading.** Every checkpoint's saved `cfg` says `window_t=4`, because cell 11 saves `vars(cfg)`, and the T03 wrapper and `project_context.md` repeat "training used 4". `window_t` is not in the state dict, so a wrong value loads silently.

## Q&A (decisions agreed before implementation)

| # | Question | Answer |
|---|---|---|
| 1 | Which value should training pass from now on? (a) keep `cfg.window_t = 4`; (b) set it to 3; (c) `None`. At T ≤ 8 frames all three train identically (Finding C). The choice only matters once a T=16 stage exists | **3** (Muaaz, 2026-11-15): `cfg.window_t = 3`, the largest tubelet distance training actually sees. |
| 2 | Do we correct Munther's notes ("training used 4" in `anyup/upsamplers.py:40,47` and `project_context.md:28`) in this task, or does Munther change them? | **I do it now** (Muaaz, 2026-11-15), ahead of the rest of this task, on `fix/teacher-full-16frame-clips`. Done; see the Implementation Log. Only comments and notes change; the wrapper's value stays 4 (changing it would change Exp 0–3 results, which is S13's measurement). |
| 3 | Should the cell 7 print line also show the model's real `window_t` (`anyup.cross_decode.window_t`), so a run's log proves which value was used? | **Yes** (Muaaz, 2026-11-15). |
| 4 | Who creates the branch? | **The assistant**, locally from `main` `b6dac9c` (Muaaz, 2026-11-26); Muaaz pushes. |
| 5 | Colab check? The model default is already `window_t=3` (`anyup/model.py:25`), so passing `cfg.window_t = 3` changes nothing at runtime, and the pytest already proves 3 / 4 / `None` are equal at 4 tubelets. | **Cells 1–7 + the new print** (Muaaz, 2026-11-26): run up to the model build in Colab and check that it prints `window_t=3`. No training step, no added GPU memory. Replaces the old Step 3 (a T=8 debug step with equal losses). |

## Design

Cell 3 (Q1):

```python
    window_t:         int   = 3      # temporal attention window in tubelets (|Δt| ≤ window_t); passed in cell 7; no effect while the largest stage has ≤ window_t + 1 tubelets (cfg.t_schedule)
```

Cell 7:

```python
anyup = AnyUp(
    input_dim=cfg.input_dim,
    qk_dim=cfg.qk_dim,
    num_heads=cfg.num_heads,
    window_ratio=cfg.window_ratio,
    kernel_size_lfu=cfg.kernel_size_lfu,
    t_k=cfg.t_k,
    window_t=cfg.window_t,   # temporal attention window (tubelets); no effect while T_tok ≤ window_t + 1 (cfg.t_schedule); not stored in the checkpoint
).to(device)
print(f"AnyUp3D: t_k={anyup.t_k}  window_t={anyup.cross_decode.window_t}")   # Q3: the values the model really uses
```

## Steps

1. `tests/test_window_t.py` (CPU, small random model); run locally.
2. Cell 3: `window_t = 3` (Q1). Cell 7: pass `window_t` and print the model's values (Q3). Clear outputs.
3. Cell 2: `REPO_BRANCH = "fix/window-t-passthrough"` for the Colab check. You commit and push. Colab (Q5): run cells 1–7; the cell 7 print must show `t_k=3  window_t=3`. Then `REPO_BRANCH` back to `"main"` on the branch, before the merge (as T04 / T01).

## Tests / Regression Checks

`tests/test_window_t.py`:
- Same weights, same input, 4 tubelets: outputs with `window_t` = 3, 4 and `None` are equal (`torch.equal`). This confirms Finding C, so the change can't alter training at T ≤ 8.
- Same weights, 8 tubelets: `window_t=3` and `window_t=None` give different outputs. This confirms the value matters above 4 tubelets.
- `AnyUp(window_t=k).cross_decode.window_t == k`.

## Done When

- pytest passes; cell 3 has `window_t = 3` and cell 7 passes it; the Colab check (Q5) prints `window_t=3`.
- S13 updated with the facts above. `docs/DECISIONS.md` entry. This file's log and the Task Index are up to date.

## Implementation Log

_Updated live during implementation. Per step: what changed, files touched, test results (actual numbers/shapes). Also log mid-implementation questions and answers, changed decisions (old → new, why), and anything added or removed. Sections above are edited in place to match, so the finished file is a report of what was actually done._

- **2026-11-15: task file written** (D-015). Findings A–E come from reading the code and the git history only; nothing has been run yet.
- **2026-11-15: Q1–Q3 answered** (3 / correct the notes now / yes). Design and Steps updated to match (D-016).
- **2026-11-15: Q2 done early**, on `fix/teacher-full-16frame-clips` (D-016). The wrapper value stays 4; only comments and notes change:
  - `anyup/upsamplers.py`: the `ANYUP3D_ARCH` comment says training never passed `window_t`, and that 4 is an evaluation choice. The `window_t` line comment says "tubelets" (it said "frames") and drops "training used 4".
  - `docs/context/project_context.md:28`: rewritten with the facts (default `None` → 3, all the same at ≤ 4 tubelets, unmeasured at 8).
  - `docs/experiments/doc_content.py`: Exp 1 protocol line corrected. `docs/experiments/experiments.html` rebuilt after commit (`build_docs.py`, clean tree, documents `99ff830`).
  - `docs/plan/P1_experiments/plan.md:32`: inline correction note, in Munther's `_(…)_` style.
  - Check: `tests/test_upsamplers.py` 34 passed; both edited `.py` files parse.
- Status stays `Q&A` until you OK the remaining steps (cells 3/7, test, Colab check). That work goes on its own branch after the teacher fix merges.
- **2026-11-15: a Colab save set cell 3 `window_t` 4 → 3** (commit `c3a242e`), without cell 7 passing it to the model, so it had no effect. Reverted with the rest of the Colab saves (`dc63e01`, D-024). Q1's value 3 is applied in Step 2 of this task, together with the cell 7 change.
- **2026-11-26: Q4–Q5 answered** (assistant creates the branch / Colab check = cells 1–7 + print). Re-checked against the current code first: cell 7 still builds `AnyUp(...)` without `window_t`, cell 3 still has `window_t: int = 4`, `anyup/model.py:25` defaults to 3, `anyup.cross_decode.window_t` exists (`cross_attention3d.py:81`). Branch `fix/window-t-passthrough` created from `main` `b6dac9c`. Steps 3 and Done When updated (old: T=8 debug step with equal losses → new: the print check, Q5). Waiting for your OK on this file.
- **2026-11-26: file OK'd** (Muaaz: "yes implement"). Status Q&A → in progress.
- **2026-11-26: Step 1 done.** New `tests/test_window_t.py` (CPU, `AnyUp(qk_dim=32, num_heads=4, kernel_size_lfu=5, t_k=3, window_ratio=0.1)`, seed 0; one set of weights loaded `strict=True` into the `window_t` = 3 / 4 / `None` models, since `window_t` isn't a weight): `test_constructor_stores_window_t` ×3 (`model.window_t` and `cross_decode.window_t`), `test_equal_outputs_up_to_4_tubelets` (T_tok 1 / 2 / 4, `torch.equal`), `test_8_tubelets_window_t_matters` (T_tok 8, 16 frames).
  - **Local CPU (venv, Python 3.12.10, torch 2.11.0+cpu): 5 passed.** Measured max |Δ| vs `window_t=3`, output (1, 24, T_tok, 8, 8): T_tok 4 → 4: 0.0, `None`: 0.0; T_tok 8 → 4: 0.627, `None`: 1.183. Confirms Findings C and D on a random model.
- **2026-11-26: Step 2 done** (`Train_3D_full.ipynb`, +4 / −2 lines; edited via the parsed JSON, the file's formatting is unchanged):
  - Cell 2: `REPO_BRANCH = "fix/window-t-passthrough"` (back to `"main"` before the merge).
  - Cell 3: `window_t: int = 3` (was 4) with a comment (tubelets, passed in cell 7, no effect at ≤ window_t + 1 tubelets, not in the weights).
  - Cell 7: `window_t=cfg.window_t` passed to `AnyUp(...)`; new print `AnyUp3D: t_k=…  window_t=…` from `anyup.cross_decode.window_t` (Q3).
  - Checks (local): valid JSON, no outputs saved, all code cells parse. Cell 3 + cell 7's model build exec'd on CPU: prints `AnyUp3D: t_k=3  window_t=3`. `test_window_t.py` + `test_shared_image_encoding.py` + `test_temporal_dynamics_loss.py`: 27 passed.
  - Committed and pushed at Muaaz's request ("commit and push"): `5155d9f` on `fix/window-t-passthrough` (new on GitHub).
- **2026-11-26: Step 3 done: Colab check passes** (Q5; Tesla T4, PyTorch 2.14.1+cu130, fresh clone). Cell 2: `Code commit: 5155d9f feat: T02_training pass cfg.window_t (=3) to AnyUp and print it; window_t tests`. Cell 7: `AnyUp3D: t_k=3  window_t=3`; the 2D checkpoint loads (73 tensors, temporal centre-init); trainable params 1,540,736 (= Exp 0's count, so the architecture is unchanged).
- **2026-11-26: close-out.** All Done When items met: `test_window_t.py` 5 passed (local CPU); cell 3 `window_t = 3`, cell 7 passes it; the Colab print shows `window_t=3`; S13 updated (stays open: the evaluation measurement is S13's, not this task's); D-033 in `docs/DECISIONS.md`; this log and the Task Index up to date. Status → done.
  - `REPO_BRANCH` in cell 2 back to `"main"` on this branch, before the merge (as T04 / T01). Notebook check: valid JSON, no outputs, all code cells parse.
  - Left: the merge into `main` (your call). S13 (which `window_t` evaluation should use; Munther's wrapper still uses 4) stays open.
  - `docs/context/project_context.md:28`: left unchanged at close-out (outside Scope); updated after the merge at Muaaz's request, see below.
- **2026-11-26: merged into `main`** (`dfb14dc`, merge commit, `--no-ff`) at Muaaz's request ("commit and merge, push main and delete the branch"). Checked on the merged `main`: `REPO_BRANCH = "main"`; `test_window_t.py` + `test_shared_image_encoding.py` + `test_temporal_dynamics_loss.py` + `test_losses.py` + `test_train_monitor.py` + `test_upsamplers.py` 98 passed (local CPU, 39 s).
  - Docs after the merge (Muaaz: "update the docs"): `project_context.md:28` now says training passes `cfg.window_t = 3` since D-033; Task Index; D-033 follow-up.
  - Branch `fix/window-t-passthrough` deleted, local + GitHub (tip `b6a6934`, an ancestor of `main`).
  - Munther informed by an email drafted for Muaaz (training now passes 3; his wrapper's 4 unchanged; S13 open). Sending it is Muaaz's.
