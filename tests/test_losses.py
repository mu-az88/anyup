# tests/test_losses.py

import pytest
import torch
import torch.nn.functional as F
import torch.nn as nn
from anyup.data.training.losses import (
    cos_mse_loss,
    input_consistency_loss,
    self_consistency_loss,
    temporal_consistency_loss,
    combined_loss,
    get_lambda3,
)
from anyup.data.training.augmentations import augment_guidance_video


# ─────────────────────────────────────────────
# Shared model stub for combined_loss tests
# ─────────────────────────────────────────────

class _IdentityModel(nn.Module):
    """Returns p tiled to (B, T, H, W, C) — ignores V. For testing only."""
    def forward(self, p, V):
        B, T, h, w, C = p.shape
        H, W = V.shape[2], V.shape[3]  # ↑ H,W from guidance video
        p_up = p.permute(0, 1, 4, 2, 3).reshape(B * T, C, h, w)
        p_up = F.interpolate(p_up, size=(H, W), mode="nearest")
        return p_up.reshape(B, T, C, H, W).permute(0, 1, 3, 4, 2)


# ─────────────────────────────────────────────
# 4.1  cos_mse_loss
# ─────────────────────────────────────────────

def test_cos_mse_zero_on_identical():
    """Identical inputs must yield loss = 0."""
    B, T, h, w, C = 2, 4, 7, 7, 768  # ↑ T=4 early curriculum, C=768 ViT-B
    x = torch.randn(B, T, h, w, C)
    loss = cos_mse_loss(x, x)
    assert loss.item() < 1e-5, f"Expected ~0, got {loss.item()}"
    print(f"PASS  test_cos_mse_zero_on_identical: loss={loss.item():.2e}")


def test_cos_mse_positive_on_different():
    """Random pred vs target must yield loss > 0."""
    B, T, h, w, C = 2, 4, 7, 7, 768
    pred   = torch.randn(B, T, h, w, C)
    target = torch.randn(B, T, h, w, C)
    loss = cos_mse_loss(pred, target)
    assert loss.item() > 0.0, f"Expected positive loss, got {loss.item()}"
    print(f"PASS  test_cos_mse_positive_on_different: loss={loss.item():.4f}")


def test_cos_mse_opposite_vectors():
    """Perfectly anti-parallel features → cosine term = 2, MSE term = 4."""
    B, T, h, w, C = 1, 1, 1, 1, 4   # ↓ minimal size — just testing the math
    pred   = torch.ones(B, T, h, w, C)
    target = -torch.ones(B, T, h, w, C)
    loss = cos_mse_loss(pred, target)
    expected = 2.0 + 4.0
    assert abs(loss.item() - expected) < 1e-4, f"Expected {expected}, got {loss.item()}"
    print(f"PASS  test_cos_mse_opposite_vectors: loss={loss.item():.4f} (expected {expected})")


def test_cos_mse_t1_matches_2d_reduction():
    """With T=1, loss must equal the manual 2D equivalent."""
    B, h, w, C = 2, 7, 7, 768
    pred   = torch.randn(B, h, w, C)
    target = torch.randn(B, h, w, C)
    loss_3d = cos_mse_loss(pred.unsqueeze(1), target.unsqueeze(1))
    pf, tf  = pred.reshape(B, h * w, C), target.reshape(B, h * w, C)
    cos_loss_2d = (1 - F.cosine_similarity(pf, tf, dim=-1)).mean()
    mse_loss_2d = F.mse_loss(pf, tf)
    loss_2d = cos_loss_2d + mse_loss_2d
    assert abs(loss_3d.item() - loss_2d.item()) < 1e-5, \
        f"T=1 mismatch: 3D={loss_3d.item():.6f}, 2D={loss_2d.item():.6f}"
    print(f"PASS  test_cos_mse_t1_matches_2d_reduction: 3D={loss_3d.item():.6f} == 2D={loss_2d.item():.6f}")


# ─────────────────────────────────────────────
# 4.2  input_consistency_loss
# ─────────────────────────────────────────────

def test_input_consistency_zero_on_consistent():
    """
    q_pred = nearest-upsample of p → downsampling back recovers p → loss ≈ 0.
    """
    B, T, h, w, C = 2, 4, 7, 7, 768
    scale = 4                           # ↑ upsampling factor
    H, W  = h * scale, w * scale

    p = torch.randn(B, T, h, w, C)
    p_up = p.permute(0, 1, 4, 2, 3).reshape(B * T, C, h, w)
    p_up = F.interpolate(p_up, size=(H, W), mode="nearest")
    p_up = p_up.reshape(B, T, C, H, W).permute(0, 1, 3, 4, 2)

    loss = input_consistency_loss(p_up, p)
    assert loss.item() < 1e-4, f"Expected ~0, got {loss.item()}"
    print(f"PASS  test_input_consistency_zero_on_consistent: loss={loss.item():.2e}")


def test_input_consistency_positive_on_random():
    """Random high-res output vs random p must give positive loss."""
    B, T, h, w, C = 2, 4, 7, 7, 768
    H, W = 28, 28

    q_pred = torch.randn(B, T, H, W, C)
    p      = torch.randn(B, T, h, w, C)
    loss = input_consistency_loss(q_pred, p)
    assert loss.item() > 0.0, f"Expected positive loss, got {loss.item()}"
    print(f"PASS  test_input_consistency_positive_on_random: loss={loss.item():.4f}")


def test_input_consistency_t1_shape():
    """T=1 warmup must work without shape errors."""
    B, T, h, w, C = 2, 1, 7, 7, 768   # ↑ T=1 single-frame warmup
    H, W = 28, 28

    q_pred = torch.randn(B, T, H, W, C)
    p      = torch.randn(B, T, h, w, C)
    loss = input_consistency_loss(q_pred, p)
    assert loss.ndim == 0, "Loss must be a scalar"
    print(f"PASS  test_input_consistency_t1_shape: loss={loss.item():.4f}")


# ─────────────────────────────────────────────
# 4.3  self_consistency_loss
# ─────────────────────────────────────────────

def test_self_consistency_zero_on_identical_video():
    """
    No augmentation (std=0, brightness/contrast=1) → both branches identical → loss ≈ 0.
    """
    class IdentityModel(nn.Module):
        def forward(self, p, V):
            B, T, h, w, C = p.shape
            H, W = V.shape[2], V.shape[3]
            p_up = p.permute(0, 1, 4, 2, 3).reshape(B * T, C, h, w)
            p_up = F.interpolate(p_up, size=(H, W), mode="nearest")
            return p_up.reshape(B, T, C, H, W).permute(0, 1, 3, 4, 2)

    B, T, h, w, C = 2, 4, 7, 7, 64    # ↑ small C for test speed
    H, W = 28, 28
    p = torch.randn(B, T, h, w, C)
    V = torch.rand(B, T, H, W, 3)

    loss = self_consistency_loss(
        IdentityModel(), p, V,
        brightness_range=(1.0, 1.0),
        contrast_range=(1.0, 1.0),
        noise_std=0.0,
    )
    assert loss.item() < 1e-5, f"Expected ~0, got {loss.item()}"
    print(f"PASS  test_self_consistency_zero_on_identical_video: loss={loss.item():.2e}")


def test_self_consistency_positive_on_augmented():
    """With real augmentation, a V-sensitive model should give positive loss."""
    class NoisySensitiveModel(nn.Module):
        def forward(self, p, V):
            B, T, H, W, _ = V.shape
            C = p.shape[-1]
            return V.mean(dim=-1, keepdim=True).expand(B, T, H, W, C)

    B, T, h, w, C = 2, 4, 7, 7, 64
    H, W = 28, 28
    p = torch.randn(B, T, h, w, C)
    V = torch.rand(B, T, H, W, 3)

    loss = self_consistency_loss(NoisySensitiveModel(), p, V)
    assert loss.item() > 0.0, f"Expected positive loss, got {loss.item()}"
    print(f"PASS  test_self_consistency_positive_on_augmented: loss={loss.item():.4f}")


def test_self_consistency_no_grad_on_aug_branch():
    """Augmented video must not carry gradients."""
    B, T, H, W = 2, 4, 28, 28
    V = torch.rand(B, T, H, W, 3, requires_grad=False)
    V_aug = augment_guidance_video(V)
    assert not V_aug.requires_grad, "Augmented video must not carry gradients"
    print(f"PASS  test_self_consistency_no_grad_on_aug_branch")


# ─────────────────────────────────────────────
# 4.4  temporal_consistency_loss
# ─────────────────────────────────────────────

def test_temporal_consistency_t1_returns_zero():
    """T=1 warmup — no adjacent pairs, must return exactly 0."""
    B, T, H, W, C = 2, 1, 28, 28, 64
    q_pred = torch.randn(B, T, H, W, C)
    V      = torch.rand(B, T, H, W, 3)
    loss   = temporal_consistency_loss(q_pred, V)
    assert loss.item() == 0.0, f"Expected 0.0, got {loss.item()}"
    print(f"PASS  test_temporal_consistency_t1_returns_zero: loss={loss.item()}")


def test_temporal_consistency_zero_on_static_identical_features():
    """Static scene + identical features across time → loss = 0."""
    B, T, H, W, C = 2, 4, 28, 28, 64
    V      = torch.rand(B, 1, H, W, 3).expand(B, T, H, W, 3).contiguous()
    q_pred = torch.randn(B, 1, H, W, C).expand(B, T, H, W, C).contiguous()
    loss = temporal_consistency_loss(q_pred, V, rgb_diff_threshold=0.05)
    assert loss.item() < 1e-5, f"Expected ~0, got {loss.item()}"
    print(f"PASS  test_temporal_consistency_zero_on_static_identical_features: loss={loss.item():.2e}")


def test_temporal_consistency_gate_blocks_fast_motion():
    """All pairs above threshold → all gated → loss = 0."""
    B, T, H, W, C = 2, 4, 28, 28, 64
    V = torch.zeros(B, T, H, W, 3)
    V[:, 1::2, ...] = 1.0          # alternating black/white → diff=1.0 >> threshold
    q_pred = torch.randn(B, T, H, W, C)
    loss = temporal_consistency_loss(q_pred, V, rgb_diff_threshold=0.05)
    assert loss.item() == 0.0, f"Expected 0.0 (all gated), got {loss.item()}"
    print(f"PASS  test_temporal_consistency_gate_blocks_fast_motion: loss={loss.item()}")


def test_temporal_consistency_positive_on_flickering_features():
    """Static RGB (gate passes all pairs) + different features each frame → positive loss."""
    B, T, H, W, C = 2, 4, 28, 28, 64
    V      = torch.rand(B, 1, H, W, 3).expand(B, T, H, W, 3).contiguous()
    q_pred = torch.randn(B, T, H, W, C)
    loss = temporal_consistency_loss(q_pred, V, rgb_diff_threshold=0.05)
    assert loss.item() > 0.0, f"Expected positive loss, got {loss.item()}"
    print(f"PASS  test_temporal_consistency_positive_on_flickering_features: loss={loss.item():.4f}")


# ─────────────────────────────────────────────
# 4.5  combined_loss + get_lambda3
# ─────────────────────────────────────────────

def test_get_lambda3_warmup():
    """λ3 must be 0 at step=0, max at step>=warmup_steps, linear in between."""
    assert get_lambda3(0,    0.01, 1000) == 0.0
    assert get_lambda3(500,  0.01, 1000) == pytest.approx(0.005, abs=1e-6)
    assert get_lambda3(1000, 0.01, 1000) == pytest.approx(0.01,  abs=1e-6)
    assert get_lambda3(2000, 0.01, 1000) == pytest.approx(0.01,  abs=1e-6)  # capped
    print("PASS  test_get_lambda3_warmup")


def test_combined_loss_returns_dict():
    """combined_loss must return a dict with all expected keys."""
    B, T, H, W, C = 2, 4, 28, 28, 64   # ↑ T=4 early curriculum
    h, w = 7, 7

    q_pred   = torch.randn(B, T, H, W, C, requires_grad=True)
    q_target = torch.randn(B, T, H, W, C)  # ← same resolution as q_pred
    p        = torch.randn(B, T, h, w, C)
    V        = torch.rand(B, T, H, W, 3)

    out = combined_loss(q_pred, q_target, p, V, model=_IdentityModel(), step=0)

    expected_keys = {"total", "reconstruction", "input", "self", "temporal", "lambda3"}
    assert set(out.keys()) == expected_keys, f"Missing keys: {expected_keys - set(out.keys())}"
    print("PASS  test_combined_loss_returns_dict")


def test_combined_loss_total_has_grad():
    """total loss must carry a grad_fn so .backward() works."""
    B, T, H, W, C = 2, 4, 28, 28, 64
    h, w = 7, 7

    q_pred   = torch.randn(B, T, H, W, C, requires_grad=True)
    q_target = torch.randn(B, T, H, W, C)  # ← same resolution as q_pred
    p        = torch.randn(B, T, h, w, C)
    V        = torch.rand(B, T, H, W, 3)

    out = combined_loss(q_pred, q_target, p, V, model=_IdentityModel(), step=500)
    assert out["total"].grad_fn is not None, "total loss has no grad_fn"
    out["total"].backward()
    assert q_pred.grad is not None, "No gradient flowed to q_pred"
    print("PASS  test_combined_loss_total_has_grad")


def test_combined_loss_lambda3_zero_at_step0():
    """At step=0, lambda3=0 → temporal term contributes nothing to total."""
    B, T, H, W, C = 2, 4, 28, 28, 64
    h, w = 7, 7

    q_pred   = torch.randn(B, T, H, W, C, requires_grad=True)
    q_target = torch.randn(B, T, H, W, C)  # ← same resolution as q_pred
    p        = torch.randn(B, T, h, w, C)
    V        = torch.rand(B, T, H, W, 3)

    out = combined_loss(
        q_pred, q_target, p, V,
        model=_IdentityModel(),
        step=0,
        lambda3_max=0.01,
        warmup_steps=1000,
    )
    assert out["lambda3"] == 0.0, f"Expected λ3=0 at step 0, got {out['lambda3']}"
    print(f"PASS  test_combined_loss_lambda3_zero_at_step0: λ3={out['lambda3']}")


def test_combined_loss_t1_warmup():
    """T=1 curriculum stage: temporal loss = 0, total still computes correctly."""
    B, T, H, W, C = 2, 1, 28, 28, 64  # ↑ T=1 warmup
    h, w = 7, 7

    q_pred   = torch.randn(B, T, H, W, C, requires_grad=True)
    q_target = torch.randn(B, T, H, W, C)  # ← same resolution as q_pred
    p        = torch.randn(B, T, h, w, C)
    V        = torch.rand(B, T, H, W, 3)

    out = combined_loss(q_pred, q_target, p, V, model=_IdentityModel(), step=2000)
    assert out["temporal"].item() == 0.0, "T=1: temporal loss must be 0"
    assert out["total"].grad_fn is not None
    print(f"PASS  test_combined_loss_t1_warmup: temporal={out['temporal'].item()}")


if __name__ == "__main__":
    test_cos_mse_zero_on_identical()
    test_cos_mse_positive_on_different()
    test_cos_mse_opposite_vectors()
    test_cos_mse_t1_matches_2d_reduction()

    test_input_consistency_zero_on_consistent()
    test_input_consistency_positive_on_random()
    test_input_consistency_t1_shape()

    test_self_consistency_zero_on_identical_video()
    test_self_consistency_positive_on_augmented()
    test_self_consistency_no_grad_on_aug_branch()

    test_temporal_consistency_t1_returns_zero()
    test_temporal_consistency_zero_on_static_identical_features()
    test_temporal_consistency_gate_blocks_fast_motion()
    test_temporal_consistency_positive_on_flickering_features()

    test_get_lambda3_warmup()
    test_combined_loss_returns_dict()
    test_combined_loss_total_has_grad()
    test_combined_loss_lambda3_zero_at_step0()
    test_combined_loss_t1_warmup()

    print("\nAll tests passed.")