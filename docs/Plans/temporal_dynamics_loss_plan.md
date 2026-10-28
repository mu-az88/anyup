# Fix: Temporal loss matches teacher dynamics

**Repo:** `mu-az88/anyup`, branch `fix/temporal-dynamics-loss`, created **after** `fix/teacher-full-16frame-clips` is merged into `3Dconv` (branch from the updated `3Dconv`). Training pipeline: `Train_3D_full.ipynb` at the repo root. Cells are referred to by their markdown headings ("cell 3" … "cell 14").

**Scope:** implement only this fix. Don't fix other issues you notice; list them at the end. The loss itself must work for any `T_tok`, so it doesn't depend on the teacher fix.

**Log rule:** when done, add an entry to `docs/DECISIONS.md`. Under **Affects**, note that it replaces the RGB-gated `feat_diff` loss and changes the meaning of `lambda_temporal_consistency`.

## Problem

The current temporal loss is computed inline in the training step (cell 13). It doesn't use `anyup/data/training/losses.py`:

```python
feat_diff = pred_feats[:, :, 1:] - pred_feats[:, :, :-1]
loss_temp = (feat_diff.pow(2) * gate).mean() * t_lam
```

1. **It rewards freezing.** It penalizes any change over time, so its optimum is static features. The teacher's features change legitimately, so this fights the reconstruction loss and causes oversmoothing.
2. **Its scale is arbitrary.** It runs on raw features, so its size scales with feature norm², and λ has no stable meaning.
3. **Its warmup never takes effect.** `get_temporal_lambda` (cell 8) ramps λ over steps 0–5000. The loss only activates once `T_tok > 1`, which first happens at the T=4 stage (`start_step=5000`). By then the ramp is already finished, so the full λ hits at the stage switch.

## Fix

Penalize the mismatch between the student's and the teacher's change, on unit-normalized features:
`mean ‖Δ norm(pred) − Δ norm(gt)‖²`. This equals the change in error between frames: stable errors cost nothing, while flicker and freezing are penalized. Values lie in [0, 16] per location (summed over channels).

Start the λ warmup at the first stage that has at least 2 tubelets.

## Steps

**1. Create `anyup/data/training/temporal_dynamics_loss.py`.** Use a new name: `anyup/data/training/losses.py:153` already defines `temporal_consistency_loss`, with a different signature, and it's used by `combined_loss` and `tests/test_losses.py`. Don't modify `losses.py`.

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

**2. cell 8:** make the warmup start when the temporal loss can first fire:

```python
TEMP_START_STEP = min(s.start_step for s in cfg.t_schedule
                      if s.t // TUBELET_SIZE >= 2)          # 5000 — first stage with ≥2 tubelets; depends on cfg.t_schedule

def get_temporal_lambda(step: int) -> float:
    warmup = cfg.temporal_lambda_warmup_steps
    if warmup <= 0:
        return cfg.lambda_temporal_consistency
    return cfg.lambda_temporal_consistency * min(1.0, max(0, step - TEMP_START_STEP) / warmup)
```

**3. cell 13 (training step):** replace the temporal block:

```python
loss_temp = torch.tensor(0.0, device=device)
t_lam = get_temporal_lambda(step)
if T_tok > 1 and t_lam > 0:
    loss_temp = temporal_dynamics_loss(pred_feats, gt_feats) * t_lam   # same gt_feats window as loss_rec
```

Remove the old `feat_diff` code and the gate-resize branch. Keep `get_rgb_diff_gate` defined (cell 12, for an optional ablation), and keep the cell 14 smoke test that uses it.

**4. Logging (cell 13):** next to `loss/temporal`, add
`writer.add_scalar("loss/temp_over_rec", loss_temp.item() / max(loss_rec.item(), 1e-8), step)`.

**5. λ needs retuning.** Report `loss/temp_over_rec` early in the T=4 stage, once warmup finishes (step ≈ 10000). The target is roughly 0.1–0.3. Suggest a value for `lambda_temporal_consistency`, but don't auto-tune or change it without asking.

## Tests: `tests/test_temporal_dynamics_loss.py`

All tests below were checked against the code above.

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

## Done when

- `pytest tests/test_temporal_dynamics_loss.py` passes, and `tests/test_losses.py` still passes (it should be untouched).
- The old `feat_diff` loss is gone from the training step.
- `get_temporal_lambda(TEMP_START_STEP) == 0` and it reaches full λ at `TEMP_START_STEP + temporal_lambda_warmup_steps`.
- A debug run reaching T=4 (with `start_step` lowered temporarily, then restored) shows a finite, non-zero `loss/temporal` and logs `loss/temp_over_rec`. Report the ratio.
- `docs/DECISIONS.md` has an entry for this change.
