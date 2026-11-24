# T01_training — Temporal Loss Matches Teacher Dynamics

| Field | Value |
|---|---|
| Task ID | T01_training |
| Plan | P1_training Training pipeline (`plan.md`) |
| Branch | `fix/temporal-dynamics-loss`, from `main` `ee17ef0` after the T04_training merge (Q6; the teacher fix is merged, D-026; base `main` since D-008) |
| Status | done (D-032) |
| Depends on | Teacher 16-frame fix (`teacher_16frame_plan.md` in this folder, D-006; merged, D-026): `gt_feats` / `T_tok` come from its window slicing. T04_training (merged first, Q6): also edits cell 13. T05_training monitor (D-028): used by the debug run |
| Blocks | — |

## Goal

Replace the temporal loss in the training step, which rewards frozen features, with one that matches the teacher's change over time. Also start the λ warmup when the loss can first take effect.

## Scope

- In scope:
  - New `anyup/data/training/temporal_dynamics_loss.py` + `tests/test_temporal_dynamics_loss.py`.
  - `Train_3D_full.ipynb` (repo root): cell 8 (`get_temporal_lambda` warmup start), cell 13 (temporal block of the training step + one new log scalar). Cells are referred to by their markdown headings ("cell 3" … "cell 14").
  - The loss must work for any `T_tok`, so it does not depend on the teacher fix.
  - `Train_3D_full.ipynb` cells 2 / 3, for the debug run only (Q7, as T04 / T05 did): `REPO_BRANCH` = this branch until the merge, and a commented debug-run block.
- Out of scope:
  - `anyup/data/training/losses.py` and `tests/test_losses.py` (untouched; `losses.py:153` already has a `temporal_consistency_loss` with a different signature, used by `combined_loss`).
  - Choosing a new `lambda_temporal_consistency` (needs a run to step ≈ 10000; after merge, see Q3).
  - Any other issue noticed on the way: add it to the Implementation Log as a follow-up, don't fix it here.

## Findings

The current temporal loss is computed inline in the training step (cell 13). It doesn't use `anyup/data/training/losses.py`:

```python
feat_diff = pred_feats[:, :, 1:] - pred_feats[:, :, :-1]
loss_temp = (feat_diff.pow(2) * gate).mean() * t_lam
```

- **A. It rewards freezing.** It penalizes any change over time, so its optimum is static features. The teacher's features change legitimately, so this fights the reconstruction loss and causes oversmoothing.
- **B. Its scale is arbitrary.** It runs on raw features, so its size scales with feature norm², and λ has no stable meaning.
- **C. Its warmup never takes effect.** `get_temporal_lambda` (cell 8) ramps λ over steps 0–5000. The loss only activates once `T_tok > 1`, which first happens at the T=4 stage (`start_step=5000`). By then the ramp is already finished, so the full λ hits at the stage switch.

## Q&A (decisions agreed before implementation)

| # | Question | Answer |
|---|---|---|
| 1 | Branch and order? | Own branch `fix/temporal-dynamics-loss`, after the teacher fix, so each change is evaluated on its own (D-003, 2026-10-28). Base branch is now `main`, not `3Dconv` (D-008, 2026-11-15). |
| 2 | New module or extend `losses.py`? | **New module** `temporal_dynamics_loss.py`; `losses.py` untouched (plan, 2026-10-28). |
| 3 | Is λ retuning part of Done When? | **No.** It needs a real run to step ≈ 10000, so it happens after merge (D-004 point 3, 2026-10-29). |
| 4 | Auto-tune λ? | **No.** Report `loss/temp_over_rec` and suggest a value; you decide (plan, 2026-10-28). |
| 5 | RGB-diff gate? | **Not used by the new loss.** Keep `get_rgb_diff_gate` defined (cell 12) for an optional ablation, and keep the cell 14 smoke test that uses it (plan, 2026-10-28). |
| 6 | T04_training also edits cell 13 and isn't merged yet. How does the branch start? | **Merge T04 first** (Muaaz, 2026-11-24): Muaaz merges `fix/shared-image-encoding` into `main`, then creates `fix/temporal-dynamics-loss` from the new `main`. |
| 7 | Debug-run settings? (T04 logged losses only every 50 steps, so none at T=8; the 5000-step λ warmup keeps λ ≈ 0 in a short run.) | **T04-style + logging/warmup** (Muaaz, 2026-11-24): commented block in cell 3 + `REPO_BRANCH` = task branch; 90 steps, T=2/4/8 from steps 0/30/60 (batch 2/1/1), validation off, `log_every_n_steps = 10`, `temporal_lambda_warmup_steps = 20` (λ full from step 50, inside the T=4 stage). Memory estimate + OOM likelihood before the run (`CLAUDE.md`). |

## Design

**Loss:** penalize the mismatch between the student's and the teacher's change, on unit-normalized features:
`mean ‖Δ norm(pred) − Δ norm(gt)‖²`. This equals the change in error between frames: stable errors cost nothing, while flicker and freezing are penalized. Values lie in [0, 16] per location (summed over channels).

`anyup/data/training/temporal_dynamics_loss.py`:

```python
import torch
import torch.nn.functional as F


def temporal_dynamics_loss(pred, gt, gate=None):
    """pred, gt: (B, C, T_tok, H, W). Optional gate: (B, 1, T_tok-1, H, W)."""
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch {tuple(pred.shape)} vs {tuple(gt.shape)}")
    if pred.shape[2] < 2:                          # T_tok=1: no pairs (avoids NaN from an empty mean)
        return pred.new_zeros((), dtype=torch.float32)
    p = F.normalize(pred.float(), dim=1)           # direction only → scale-free; float32 for precision
    g = F.normalize(gt.float(), dim=1)
    err = ((p[:, :, 1:] - p[:, :, :-1]) - (g[:, :, 1:] - g[:, :, :-1])).pow(2).sum(1, keepdim=True)
    return (err * gate.float() if gate is not None else err).mean()
```

**Cell 8:** start the warmup at the first stage with at least 2 tubelets:

```python
TEMP_START_STEP = min(s.start_step for s in cfg.t_schedule
                      if s.t // TUBELET_SIZE >= 2)          # 5000 — first stage with ≥2 tubelets; depends on cfg.t_schedule

def get_temporal_lambda(step: int) -> float:
    warmup = cfg.temporal_lambda_warmup_steps
    if warmup <= 0:
        return cfg.lambda_temporal_consistency
    return cfg.lambda_temporal_consistency * min(1.0, max(0, step - TEMP_START_STEP) / warmup)
```

**Cell 13 (training step):** replace the temporal block:

```python
loss_temp = torch.tensor(0.0, device=device)
t_lam = get_temporal_lambda(step)
if T_tok > 1 and t_lam > 0:
    loss_temp = temporal_dynamics_loss(pred_feats, gt_feats) * t_lam   # same gt_feats window as loss_rec
```

Remove the old `feat_diff` code and the gate-resize branch.

**Cell 13 logging:** next to `loss/temporal`, add
`writer.add_scalar("loss/temp_over_rec", loss_temp.item() / max(loss_rec.item(), 1e-8), step)`.

## Steps

0. T04_training merged into `main` and `fix/temporal-dynamics-loss` created from it (Q6; done by the assistant at Muaaz's request).
1. Create `anyup/data/training/temporal_dynamics_loss.py` and `tests/test_temporal_dynamics_loss.py`; run pytest locally.
2. Cell 8: `TEMP_START_STEP` + new `get_temporal_lambda`.
3. Cell 13: replace the temporal block; import `temporal_dynamics_loss` (cell 12).
4. Cell 13: log `loss/temp_over_rec`.
5. Clear notebook outputs. Debug-run setup (Q7): `REPO_BRANCH` = this branch in cell 2, commented T01 block in cell 3; memory estimate + OOM likelihood to Muaaz for an OK. You commit and push; then the Colab debug run (see Done When). After the merge: `REPO_BRANCH` back to `"main"` (or before it, on the branch, as T04 did).
6. **After merge (not part of Done When):** a real run to step ≈ 10000. Report `loss/temp_over_rec` early in the T=4 stage once warmup finishes. Target roughly 0.1–0.3. Suggest a `lambda_temporal_consistency` value (Q4).

## Tests / Regression Checks

`tests/test_temporal_dynamics_loss.py` (all 8 checked against the Design code on 2026-10-29):

```python
import pytest
import torch
from anyup.data.training.temporal_dynamics_loss import temporal_dynamics_loss as L

SHAPE = (2, 16, 4, 3, 3)
static = lambda x: x[:, :, :1].expand_as(x).clone()   # same tubelet repeated → no temporal change


def test_zero_when_equal():
    gt = torch.randn(SHAPE)
    assert L(gt.clone(), gt).item() == pytest.approx(0, abs=1e-6)


def test_stable_error_costs_nothing():
    assert L(static(torch.randn(SHAPE)), static(torch.randn(SHAPE))).item() == pytest.approx(0, abs=1e-6)


def test_freezing_penalized():
    gt = torch.randn(SHAPE)
    assert L(static(gt), gt).item() > 0.1          # the old loss gives 0 here


def test_flicker_penalized():
    a, b = torch.randn(2, 16, 1, 3, 3), torch.randn(2, 16, 1, 3, 3)
    assert L(torch.cat([a, b, a, b], 2), static(torch.randn(SHAPE))).item() > 0.1


def test_scale_invariant():
    p, g = torch.randn(SHAPE), torch.randn(SHAPE)
    assert L(p * 50, g * 0.1).item() == pytest.approx(L(p, g).item(), rel=1e-4)


def test_single_tubelet_is_zero_not_nan():
    assert L(torch.randn(2, 16, 1, 3, 3), torch.randn(2, 16, 1, 3, 3)).item() == 0.0


def test_gate_zeroes_pairs():
    p, g = torch.randn(SHAPE), torch.randn(SHAPE)
    assert L(p, g, gate=torch.zeros(2, 1, 3, 3, 3)).item() == 0.0


def test_bf16_and_gradient():
    p = torch.randn(SHAPE, requires_grad=True)
    out = L(p.bfloat16(), torch.randn(SHAPE).bfloat16())
    out.backward()
    assert out.dtype == torch.float32 and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
```

Added in Step 1: `test_shape_errors` (shape mismatch, non-5D input, wrong gate shape → `ValueError`), and an autouse fixture seeding `torch.manual_seed(0)` so the threshold tests are reproducible.

Regression: `tests/test_losses.py` still passes, unchanged.

Colab debug run (Q7; through the T05 monitor): the commented T01 block in cell 3 (90 steps, T=2/4/8 from 0/30/60, `log_every_n_steps = 10`, `temporal_lambda_warmup_steps = 20`, validation off). Check: `loss/temporal` = 0 in the T=2 stage, finite and non-zero at T=4 and T=8; λ = 0 at step 30 (`TEMP_START_STEP`) and full from step 50; `loss/temp_over_rec` logged. Report the ratio at full λ for T=4 and T=8. This is only a sanity check after 90 steps, not the Step 6 tuning number.

## Done When

- `pytest tests/test_temporal_dynamics_loss.py` passes (9 tests), and `tests/test_losses.py` still passes (untouched).
- The old `feat_diff` loss is gone from the training step.
- `get_temporal_lambda(TEMP_START_STEP) == 0`, and it reaches full λ at `TEMP_START_STEP + temporal_lambda_warmup_steps`.
- The Colab debug run above passes, and its ratio is reported.
- `docs/DECISIONS.md` has an entry; under **Affects** it notes that this replaces the RGB-gated `feat_diff` loss and changes the meaning of `lambda_temporal_consistency`.
- This file's Implementation Log and the Task Index in `plan.md` are up to date.

## Implementation Log

_Updated live during implementation. Per step: what changed, files touched, test results (actual numbers/shapes). Also log mid-implementation questions and answers, changed decisions (old → new, why), and anything added or removed. Sections above are edited in place to match, so the finished file is a report of what was actually done._

- **2026-11-15: converted to the task template** (D-009). Moved from `docs/Plans/temporal_dynamics_loss_plan.md` (first as T10 in `P2_training_pipeline`, then renamed to Muaaz's numbering: T01_training in `P1_training_pipeline`). Problem → Findings A–C; earlier decisions → Q&A 1–5; Step 5 (λ retuning) → Step 6, marked after merge. Changed: base branch `3Dconv` → `main` (D-008). Code, tests and Done When are otherwise unchanged. Status `Q&A`: waiting for your OK on this file before implementing.
- **2026-11-24: file approved** (Muaaz: "the T01 plan is good"). Checked against the current notebook first: T04 only changed how `pred_feats` is computed (`encode_image` / `decode`), the temporal block (cell 13) is still the `feat_diff` + RGB-gate code of Finding A, `get_temporal_lambda` (cell 8) still ramps from step 0 (Finding C), `TUBELET_SIZE` (cell 11) is defined before cell 8 runs, and the schedule is still T=2/4/8 from 0/5000/15000 (`TEMP_START_STEP` = 5000). Two new questions: **Q6** (branch after the T04 merge) and **Q7** (debug-run settings), both answered; Scope, Steps (new Step 0) and the debug-run test updated in place. These edits were made on `fix/shared-image-encoding`, uncommitted, and move with the checkout to the new branch. Waiting for the T04 merge and the new branch.
- **2026-11-24: Step 0 done** (Muaaz chose: the assistant merges, pushes and branches). `fix/shared-image-encoding` merged into `main` (`726dd0c`, merge commit; post-merge docs `ee17ef0`; both pushed; 31 tests passed on the merged tree). `fix/temporal-dynamics-loss` created from `ee17ef0`; this file's approval edits moved with it (stash + pop). Status → in progress.
- **2026-11-24: Step 1 done.** New `anyup/data/training/temporal_dynamics_loss.py`: the Design code, plus (added, not in the Design) a `ValueError` for non-5D input and for a gate that isn't `(B, 1, T_tok-1, H, W)` (shape compares only, no GPU cost). New `tests/test_temporal_dynamics_loss.py`: the 8 Design tests verbatim + `test_shape_errors`, with a seed fixture (`torch.manual_seed(0)`).
  - **Local CPU (Python 3.12, torch 2.11.0+cpu): `test_temporal_dynamics_loss.py` 9 passed + `test_losses.py` 19 passed (untouched), 5.1 s.**
  - Values behind the threshold tests (seed 0): equal 0.0, stable error 0.0, freezing 2.057, flicker 1.794 (both > 0.1); scale test 4.1488 = 4.1488; random 768-ch features at T_tok 4, 14²: 3.99 (bound 16).
- **2026-11-24: Steps 2–4 done** (`Train_3D_full.ipynb`, exact one-occurrence replacements on the raw JSON; +16 / −11 lines):
  - Cell 8: `TEMP_START_STEP = min((s.start_step for s in cfg.t_schedule if s.t // TUBELET_SIZE >= 2), default=0)` + the Design's `get_temporal_lambda`, and a print of the warmup range. Added (not in the Design): `default=0`, so a schedule without a ≥ 2-tubelet stage doesn't crash (the loss never runs then).
  - Cell 12: `from anyup.data.training.temporal_dynamics_loss import temporal_dynamics_loss`.
  - Cell 13: the `feat_diff` + RGB-gate block (11 lines) → `loss_temp = temporal_dynamics_loss(pred_feats, gt_feats) * t_lam`. `get_rgb_diff_gate` stays defined (Q5); `crop_low_s` stays (returned by `take_window` with `guidance_feats`).
  - Cell 13 logging: `loss/temp_over_rec` after `loss/temporal`.
  - Checks (local): valid JSON, all code cells parse, no outputs saved, no `feat_diff` left in cell 13. The notebook's own new lines exec'd on fakes: default schedule `TEMP_START_STEP` 5000, λ(5000) = 0, λ(7500) = 0.1, λ(10000) = 0.2; Q7 schedule 30, λ(30) = 0, λ(50) = 0.2. Cell 13 block: T_tok 1 → 0.0; T_tok 2 / 4 → 0.0338 / 0.0339, float32 (pred = gt + 0.3·noise, λ 0.2).
- **2026-11-24: Step 5 setup (Q7).** Cell 2: `REPO_BRANCH = "fix/temporal-dynamics-loss"` (back to `"main"` before the merge). Cell 3: commented "T01_training debug run" block after T04's: `experiment_name = "t01_temporal_dyn"` (+ `checkpoint_dir` / `log_dir`), `max_steps = 90`, `val_every_n_steps = 10_000`, `log_every_n_steps = 10`, `temporal_lambda_warmup_steps = 20`, schedule T=2 bs 2 / T=4 bs 1 / T=8 bs 1 from 0 / 30 / 60. Cell 3 parses, and the block parses when uncommented.
  - **Memory estimate for the run:** the same steps as T04's T4 run (D-031: 5.79 / 5.79 / 10.26 GiB reserved), plus the new loss. Bytes autograd keeps for the backward pass (CPU, saved-tensor hooks, bf16 inputs as under autocast, batch 1, 14²): new loss 1.7 / 4.0 MiB at T_tok 2 / 4 vs old 0.3 / 0.9 MiB → **+1.4 / +3.2 MiB**. Expected peak ≈ 10.3 GiB at T=8, 4.3 GiB below the T4's 14.6. **OOM likelihood: low** (the same run fit with 4.3 GiB to spare; the change is ~1/1000 of that). `log_every_n_steps = 10` is CPU-side only.
  - **Estimate and run OK'd by Muaaz (2026-11-24).** Waiting for the commit + push and the Colab run.
- **2026-11-24: Step 5 done: Colab debug run passes** (Tesla T4, PyTorch 2.14.1+cu130, code `c48f00e`, the commented T01 block in cell 3). Run 1 completed; the runtime disconnected after `monitor_history.csv` and the TensorBoard scalars (event dir `runs/t01_temporal_dyn/2026-11-24_13-20-39/`) were read, so Muaaz ran it again (run 2; the monitor started a new history, "kept 0 of 9 rows", so the CSV on Drive is now run 2's). The numbers below are from run 1's CSV and events.
  - Cell 8: `Temporal loss: λ warmup from step 30 over 20 steps (full λ = 0.2 from step 50)`.

    | step | T | λ | loss_rec | loss_temporal (×λ) | raw loss (÷λ) | `loss/temp_over_rec` | grad_norm |
    |---|---|---|---|---|---|---|---|
    | 0 / 10 / 20 | 2 | 0 | 0.0804 / 0.0881 / 0.0627 | 0.0 | — | 0.0 | 0.090 / 0.030 / 0.020 |
    | 30 | 4 | 0 | 0.0929 | 0.0 | — | 0.0 | 0.030 |
    | 40 | 4 | 0.1 | 0.1037 | 0.0204 | 0.204 | 0.1968 | 0.040 |
    | 50 | 4 | 0.2 | 0.1005 | 0.0767 | 0.383 | 0.7627 | 0.131 |
    | 60 | 8 | 0.2 | 0.1031 | 0.0534 | 0.267 | 0.5178 | 0.048 |
    | 70 | 8 | 0.2 | 0.1038 | 0.0390 | 0.195 | 0.3755 | 0.028 |
    | 80 | 8 | 0.2 | 0.0955 | 0.0346 | 0.173 | 0.3624 | 0.088 |

  - **Checks, all pass:** `loss/temporal` = 0 in the T=2 stage (1 tubelet) and at step 30 (λ = 0 at `TEMP_START_STEP`); finite and non-zero at T=4 and T=8; λ 0.1 at step 40 and full 0.2 from step 50, as designed; `loss/temp_over_rec` logged, and it equals loss_temporal / loss_rec from the CSV (e.g. step 80: 0.03460 / 0.09548 = 0.3624). `loss_total` = rec + inp + aug + temporal to 1e-8 (steps 50, 80).
  - **Ratio at full λ (reported, not tuned; Q3):** T=4 0.76 (step 50, 1 point), T=8 0.52 → 0.38 → 0.36 (steps 60–80). Above Step 6's target of 0.1–0.3, but this is 90 steps with batch 1 and one sample per point; the raw loss falls from 0.38 to 0.17 over the same steps. The λ value is for Step 6 (after the merge) to decide.
  - **Memory and speed (both runs):** reserved 5.79 / 5.79 / 10.26 GiB (40 / 40 / 70 % of 14.6 GiB) at T=2 / 4 / 8, peak alloc 5.70 / 5.40 / 10.05 GiB: the same as T04's run (D-031) to the printed precision, so the new loss costs nothing measurable (estimate +3.2 MiB). Median s/step run 1 2.283 / 1.899 / 2.852, run 2 2.546 / 1.881 / 2.890. Skipped steps 0; grad_scale 65536 throughout; monitor overhead 0.050 % / 0.038 %. No OOM.
  - Run 2's final progress bar equals run 1's (loss 0.1355, rec 0.0955, inp 0.0021, lr 6.85e-06): the run repeats to the shown 4 digits.
  - Correction: from run 1's progress bar I had estimated temporal ≈ 0.037 and ratio ≈ 0.39 at step 80 (assuming aug ≈ 0.0008); the CSV gives aug 0.0032, temporal 0.0346, ratio 0.36.
  - All Done When items are now met. Waiting for Muaaz's OK to close out.
- **2026-11-24: close-out** (Muaaz: "go ahead"). All Done When items met: 9 + 19 tests pass; `feat_diff` gone from cell 13; λ(`TEMP_START_STEP`) = 0 and full at `TEMP_START_STEP + warmup` (local check and the Colab run); the debug run passes, ratio reported; D-032 in `docs/DECISIONS.md`; this log and the Task Index up to date. Status → done.
  - `REPO_BRANCH` in cell 2 back to `"main"` (main's original line) **before the merge, on this branch**, as T04 did, so `main` needs no follow-up commit. The T01 block in cell 3 stays commented, like the T04 and T05 blocks.
  - Left: the merge into `main` (Muaaz's call); Step 6 after the merge (a run to step ≈ 10000, then λ from `loss/temp_over_rec`; the debug run's 0.36–0.76 is above the 0.1–0.3 target, so expect a lower λ, but decide on the real run).
  - Not verified: the Step 6 behaviour (whether the loss actually improves temporal consistency) can only show on a long run; the debug run only shows the loss is wired correctly, finite and cheap.
