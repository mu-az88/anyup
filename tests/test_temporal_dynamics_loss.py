"""T01_training: tests for anyup.data.training.temporal_dynamics_loss (CPU, synthetic)."""
import pytest
import torch
from anyup.data.training.temporal_dynamics_loss import temporal_dynamics_loss as L

SHAPE = (2, 16, 4, 3, 3)                              # (B, C, T_tok, H, W)
static = lambda x: x[:, :, :1].expand_as(x).clone()   # same tubelet repeated → no temporal change


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


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


def test_shape_errors():                            # added in Step 1: clear errors for wrong inputs
    p = torch.randn(SHAPE)
    with pytest.raises(ValueError, match="shape mismatch"):
        L(p, torch.randn(2, 16, 3, 3, 3))
    with pytest.raises(ValueError, match="expected"):
        L(torch.randn(2, 16, 3, 3), torch.randn(2, 16, 3, 3))
    with pytest.raises(ValueError, match="gate"):
        L(p, torch.randn(SHAPE), gate=torch.ones(2, 1, 4, 3, 3))   # T_tok pairs, not T_tok-1
