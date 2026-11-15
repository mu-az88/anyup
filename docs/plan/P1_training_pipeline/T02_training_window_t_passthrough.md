# T02_training — Pass `window_t` from the Config to the Model

| Field | Value |
|---|---|
| Task ID | T02_training |
| Plan | P1_training Training pipeline (`plan.md`) |
| Branch | `fix/window-t-passthrough`, branched from `main` after the teacher fix is merged into `main` (D-008) |
| Status | Q&A |
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
3. Colab: one debug step at T=8 (4 tubelets). The losses should equal a run without the change, with the same seed and batch (Finding C).

## Tests / Regression Checks

`tests/test_window_t.py`:
- Same weights, same input, 4 tubelets: outputs with `window_t` = 3, 4 and `None` are equal (`torch.equal`). This confirms Finding C, so the change can't alter training at T ≤ 8.
- Same weights, 8 tubelets: `window_t=3` and `window_t=None` give different outputs. This confirms the value matters above 4 tubelets.
- `AnyUp(window_t=k).cross_decode.window_t == k`.

## Done When

- pytest passes; cell 7 passes `window_t`; the Colab debug step reproduces the unchanged losses.
- S13 updated with the facts above. `docs/DECISIONS.md` entry. This file's log and the Task Index are up to date.

## Implementation Log

_Updated live during implementation. Per step: what changed, files touched, test results (actual numbers/shapes). Also log mid-implementation questions and answers, changed decisions (old → new, why), and anything added or removed. Sections above are edited in place to match, so the finished file is a report of what was actually done._

- **2026-11-15: task file written** (D-015). Findings A–E come from reading the code and the git history only; nothing has been run yet.
- **2026-11-15: Q1–Q3 answered** (3 / correct the notes now / yes). Design and Steps updated to match (D-016).
- **2026-11-15: Q2 done early**, on `fix/teacher-full-16frame-clips` (D-016). The wrapper value stays 4; only comments and notes change:
  - `anyup/upsamplers.py`: the `ANYUP3D_ARCH` comment says training never passed `window_t`, and that 4 is an evaluation choice. The `window_t` line comment says "tubelets" (it said "frames") and drops "training used 4".
  - `docs/context/project_context.md:28`: rewritten with the facts (default `None` → 3, all the same at ≤ 4 tubelets, unmeasured at 8).
  - `docs/experiments/doc_content.py`: Exp 1 protocol line corrected. **`docs/experiments/experiments.html` is not rebuilt yet**: `build_docs.py` refuses a dirty tree, so rebuild it after commit.
  - `docs/plan/P1_experiments/plan.md:32`: inline correction note, in Munther's `_(…)_` style.
  - Check: `tests/test_upsamplers.py` 34 passed; both edited `.py` files parse.
- Status stays `Q&A` until you OK the remaining steps (cells 3/7, test, Colab check). That work goes on its own branch after the teacher fix merges.
