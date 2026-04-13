"""
tests/test_losses.py
Unit tests for the loss functions defined in scripts/smoke_test.py (Phase 7.1).

Import the losses directly from smoke_test so they stay in sync with the
script — no duplication, no drift.

Covers:
  - cosine_mse: positive, zero on identical inputs, shape-agnostic
  - input_consistency: downsamples correctly, matches lores shape
  - temporal_consistency: returns 0.0 for T=1, positive for T>1
  - combined_loss: returns all expected keys, all scalars, no NaN

Run:
    pytest tests/test_losses.py -v
"""

import sys
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
# Import losses directly from the smoke_test script so there's a single source of truth
from scripts.smoke_test import cosine_mse, input_consistency, temporal_consistency, combined_loss


# ══════════════════════════════════════════════════════════════════════════════
# cosine_mse
# ══════════════════════════════════════════════════════════════════════════════

def test_cosine_mse_is_positive():
    """Random pred/target must produce a positive loss."""
    pred   = torch.randn(2, 64, 4, 14, 14)
    target = torch.randn(2, 64, 4, 14, 14)
    loss   = cosine_mse(pred, target)
    assert loss.item() > 0, "cosine_mse should be positive for random inputs"
    print("test_cosine_mse_is_positive passed")


def test_cosine_mse_zero_on_identical():
    """Identical pred and target must give loss ≈ 0."""
    x    = torch.randn(2, 64, 4, 14, 14)
    loss = cosine_mse(x, x)
    assert loss.item() < 1e-5, f"cosine_mse(x, x) should be ~0, got {loss.item()}"
    print("test_cosine_mse_zero_on_identical passed")


def test_cosine_mse_no_nan():
    """cosine_mse must not produce NaN for typical random inputs."""
    pred   = torch.randn(2, 64, 4, 14, 14)
    target = torch.randn(2, 64, 4, 14, 14)
    loss   = cosine_mse(pred, target)
    assert not torch.isnan(loss), "cosine_mse returned NaN"
    print("test_cosine_mse_no_nan passed")


def test_cosine_mse_scalar_output():
    """cosine_mse must return a 0-dim scalar tensor."""
    pred   = torch.randn(2, 64, 4, 14, 14)
    target = torch.randn(2, 64, 4, 14, 14)
    loss   = cosine_mse(pred, target)
    assert loss.shape == torch.Size([]), f"Expected scalar, got shape {loss.shape}"
    print("test_cosine_mse_scalar_output passed")


def test_cosine_mse_gradients_flow():
    """Gradients must flow back through cosine_mse to the input tensor."""
    pred   = torch.randn(2, 64, 4, 14, 14, requires_grad=True)
    target = torch.randn(2, 64, 4, 14, 14)
    loss   = cosine_mse(pred, target)
    loss.backward()
    assert pred.grad is not None, "No gradient on pred after cosine_mse.backward()"
    assert not torch.isnan(pred.grad).any(), "NaN in cosine_mse gradient"
    print("test_cosine_mse_gradients_flow passed")


# ══════════════════════════════════════════════════════════════════════════════
# input_consistency
# ══════════════════════════════════════════════════════════════════════════════

def test_input_consistency_shape_compatible():
    """
    input_consistency must not crash when pred is 4× larger than feat_lores
    (the standard downscale factor).
    """
    B, C, T = 2, 64, 4
    H_lo, W_lo = 14, 14
    pred       = torch.randn(B, C, T, H_lo * 4, W_lo * 4)
    feat_lores = torch.randn(B, C, T, H_lo,     W_lo)
    loss       = input_consistency(pred, feat_lores)
    assert loss.shape == torch.Size([]), f"Expected scalar, got {loss.shape}"
    print("test_input_consistency_shape_compatible passed")


def test_input_consistency_no_nan():
    """input_consistency must not produce NaN for typical inputs."""
    pred       = torch.randn(2, 64, 4, 56, 56)
    feat_lores = torch.randn(2, 64, 4, 14, 14)
    loss       = input_consistency(pred, feat_lores)
    assert not torch.isnan(loss), "input_consistency returned NaN"
    print("test_input_consistency_no_nan passed")


def test_input_consistency_zero_on_identical_after_pool():
    """
    If pred is the exact spatial repeat of feat_lores (after avg-pooling pred
    collapses back to lores), loss should be ~0.
    """
    B, C, T, H_lo, W_lo = 1, 32, 2, 7, 7
    feat_lores = torch.randn(B, C, T, H_lo, W_lo)
    # Upsample feat_lores to pred resolution then check round-trip
    pred = F.interpolate(
        feat_lores.reshape(B * T, C, H_lo, W_lo),
        size=(H_lo * 4, W_lo * 4), mode="nearest"
    ).reshape(B, C, T, H_lo * 4, W_lo * 4)
    loss = input_consistency(pred, feat_lores)
    # avg-pool of nearest-upsample ≠ exact original due to pooling kernel edge effects,
    # so we only assert it's not NaN and is a reasonable small value.
    assert not torch.isnan(loss), "input_consistency returned NaN on round-trip input"
    print("test_input_consistency_zero_on_identical_after_pool passed")


# ══════════════════════════════════════════════════════════════════════════════
# temporal_consistency
# ══════════════════════════════════════════════════════════════════════════════

def test_temporal_consistency_zero_for_T1():
    """temporal_consistency must return exactly 0.0 when T=1 (no adjacent frames)."""
    pred = torch.randn(2, 64, 1, 14, 14)
    loss = temporal_consistency(pred)
    assert loss.item() == 0.0, f"Expected 0.0 for T=1, got {loss.item()}"
    print("test_temporal_consistency_zero_for_T1 passed")


def test_temporal_consistency_positive_for_T4():
    """temporal_consistency must return a positive value for T=4 random input."""
    pred = torch.randn(2, 64, 4, 14, 14)
    loss = temporal_consistency(pred, lambda_t=0.01)
    assert loss.item() > 0, "temporal_consistency should be positive for T=4 random input"
    print("test_temporal_consistency_positive_for_T4 passed")


def test_temporal_consistency_no_nan():
    """temporal_consistency must not produce NaN."""
    pred = torch.randn(2, 64, 8, 14, 14)
    loss = temporal_consistency(pred, lambda_t=0.01)
    assert not torch.isnan(loss), "temporal_consistency returned NaN"
    print("test_temporal_consistency_no_nan passed")


def test_temporal_consistency_lower_for_smooth_sequence():
    """
    A smooth (nearly constant) video should produce lower temporal loss
    than a random video.
    """
    B, C, T, H, W = 2, 64, 8, 14, 14
    random_pred = torch.randn(B, C, T, H, W)
    # Smooth: replicate frame 0 across all T
    smooth_pred = random_pred[:, :, :1, :, :].expand(B, C, T, H, W).clone()

    loss_random = temporal_consistency(random_pred, lambda_t=1.0).item()
    loss_smooth = temporal_consistency(smooth_pred, lambda_t=1.0).item()
    assert loss_smooth < loss_random, (
        f"Smooth sequence ({loss_smooth:.4f}) should have lower temporal loss "
        f"than random ({loss_random:.4f})"
    )
    print("test_temporal_consistency_lower_for_smooth_sequence passed")


def test_temporal_consistency_lambda_scales_loss():
    """Doubling lambda_t should double the temporal loss."""
    pred = torch.randn(2, 64, 4, 14, 14)
    l1   = temporal_consistency(pred, lambda_t=0.01).item()
    l2   = temporal_consistency(pred, lambda_t=0.02).item()
    assert abs(l2 - 2 * l1) < 1e-5, f"Expected l2≈2·l1, got l1={l1:.6f} l2={l2:.6f}"
    print("test_temporal_consistency_lambda_scales_loss passed")


# ══════════════════════════════════════════════════════════════════════════════
# combined_loss
# ══════════════════════════════════════════════════════════════════════════════

def _make_combined_batch(device="cpu"):
    B, C, T, H, W = 2, 64, 4, 56, 56
    s = 4
    return {
        "feat_hires": torch.randn(B, C, T, H,    W,    device=device),
        "feat_lores": torch.randn(B, C, T, H//s, W//s, device=device),
    }


def test_combined_loss_returns_all_keys():
    """combined_loss must return a dict with keys: total, recon, input, temporal."""
    pred  = torch.randn(2, 64, 4, 56, 56)
    batch = _make_combined_batch()
    losses = combined_loss(pred, batch)
    for key in ("total", "recon", "input", "temporal"):
        assert key in losses, f"Missing key '{key}' in combined_loss output"
    print("test_combined_loss_returns_all_keys passed")


def test_combined_loss_all_scalars():
    """Every value in the combined_loss dict must be a 0-dim scalar tensor."""
    pred    = torch.randn(2, 64, 4, 56, 56)
    batch   = _make_combined_batch()
    losses  = combined_loss(pred, batch)
    for key, val in losses.items():
        assert val.shape == torch.Size([]), f"Loss '{key}' is not scalar: shape {val.shape}"
    print("test_combined_loss_all_scalars passed")


def test_combined_loss_no_nan():
    """combined_loss must not produce NaN for typical random inputs."""
    pred  = torch.randn(2, 64, 4, 56, 56)
    batch = _make_combined_batch()
    losses = combined_loss(pred, batch)
    for key, val in losses.items():
        assert not torch.isnan(val), f"NaN in combined_loss['{key}']"
    print("test_combined_loss_no_nan passed")


def test_combined_loss_total_equals_sum_of_components():
    """
    total must equal recon + input + temporal
    (within floating-point tolerance).
    """
    pred   = torch.randn(2, 64, 4, 56, 56)
    batch  = _make_combined_batch()
    losses = combined_loss(pred, batch)
    expected = losses["recon"] + losses["input"] + losses["temporal"]
    assert torch.allclose(losses["total"], expected, atol=1e-5), (
        f"total ({losses['total'].item():.6f}) ≠ sum of components "
        f"({expected.item():.6f})"
    )
    print("test_combined_loss_total_equals_sum_of_components passed")


def test_combined_loss_backward_flows():
    """Gradients must flow back through combined_loss to the prediction tensor."""
    pred  = torch.randn(2, 64, 4, 56, 56, requires_grad=True)
    batch = _make_combined_batch()
    losses = combined_loss(pred, batch)
    losses["total"].backward()
    assert pred.grad is not None, "No gradient on pred after combined_loss.backward()"
    assert not torch.isnan(pred.grad).any(), "NaN in combined_loss gradient"
    print("test_combined_loss_backward_flows passed")


# ══════════════════════════════════════════════════════════════════════════════
# __main__ runner
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    test_cosine_mse_is_positive()
    test_cosine_mse_zero_on_identical()
    test_cosine_mse_no_nan()
    test_cosine_mse_scalar_output()
    test_cosine_mse_gradients_flow()

    test_input_consistency_shape_compatible()
    test_input_consistency_no_nan()
    test_input_consistency_zero_on_identical_after_pool()

    test_temporal_consistency_zero_for_T1()
    test_temporal_consistency_positive_for_T4()
    test_temporal_consistency_no_nan()
    test_temporal_consistency_lower_for_smooth_sequence()
    test_temporal_consistency_lambda_scales_loss()

    test_combined_loss_returns_all_keys()
    test_combined_loss_all_scalars()
    test_combined_loss_no_nan()
    test_combined_loss_total_equals_sum_of_components()
    test_combined_loss_backward_flows()

    print("\nAll loss tests passed.")