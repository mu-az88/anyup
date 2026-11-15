# T01_training — Temporal Loss Matches Teacher Dynamics

| Field | Value |
|---|---|
| Task ID | T01_training |
| Plan | P1_training Training pipeline (`plan.md`) |
| Branch | `fix/temporal-dynamics-loss`, branched from `main` after the teacher fix is merged into `main` (D-003, base changed by D-008) |
| Status | Q&A |
| Depends on | Teacher 16-frame fix (`teacher_16frame_plan.md` in this folder, D-006): `gt_feats` / `T_tok` come from its window slicing |
| Blocks | — |

## Goal

Replace the temporal loss in the training step, which rewards frozen features, with one that matches the teacher's change over time. Also start the λ warmup when the loss can first take effect.

## Scope

- In scope:
  - New `anyup/data/training/temporal_dynamics_loss.py` + `tests/test_temporal_dynamics_loss.py`.
  - `Train_3D_full.ipynb` (repo root): cell 8 (`get_temporal_lambda` warmup start), cell 13 (temporal block of the training step + one new log scalar). Cells are referred to by their markdown headings ("cell 3" … "cell 14").
  - The loss must work for any `T_tok`, so it does not depend on the teacher fix.
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

1. Create `anyup/data/training/temporal_dynamics_loss.py` and `tests/test_temporal_dynamics_loss.py`; run pytest locally.
2. Cell 8: `TEMP_START_STEP` + new `get_temporal_lambda`.
3. Cell 13: replace the temporal block; import `temporal_dynamics_loss` (cell 12).
4. Cell 13: log `loss/temp_over_rec`.
5. Clear notebook outputs. You commit and push; then the Colab debug run (see Done When).
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

Regression: `tests/test_losses.py` still passes, unchanged.

Colab debug run: with stage `start_step`s lowered temporarily (then restored), reach T=4 and check `loss/temporal` is finite and non-zero, and `loss/temp_over_rec` is logged. Report the ratio. This is only a sanity check: warmup hasn't finished, so it isn't the Step 6 tuning number.

## Done When

- `pytest tests/test_temporal_dynamics_loss.py` passes, and `tests/test_losses.py` still passes (untouched).
- The old `feat_diff` loss is gone from the training step.
- `get_temporal_lambda(TEMP_START_STEP) == 0`, and it reaches full λ at `TEMP_START_STEP + temporal_lambda_warmup_steps`.
- The Colab debug run above passes, and its ratio is reported.
- `docs/DECISIONS.md` has an entry; under **Affects** it notes that this replaces the RGB-gated `feat_diff` loss and changes the meaning of `lambda_temporal_consistency`.
- This file's Implementation Log and the Task Index in `plan.md` are up to date.

## Implementation Log

_Updated live during implementation. Per step: what changed, files touched, test results (actual numbers/shapes). Also log mid-implementation questions and answers, changed decisions (old → new, why), and anything added or removed. Sections above are edited in place to match, so the finished file is a report of what was actually done._

- **2026-11-15: converted to the task template** (D-009). Moved from `docs/Plans/temporal_dynamics_loss_plan.md` (first as T10 in `P2_training_pipeline`, then renamed to Muaaz's numbering: T01_training in `P1_training_pipeline`). Problem → Findings A–C; earlier decisions → Q&A 1–5; Step 5 (λ retuning) → Step 6, marked after merge. Changed: base branch `3Dconv` → `main` (D-008). Code, tests and Done When are otherwise unchanged. Status `Q&A`: waiting for your OK on this file before implementing.
