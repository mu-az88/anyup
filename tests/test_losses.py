
import torch
from anyup.data.training.losses import cos_mse_loss
from anyup.data.training.losses import input_consistency_loss
import torch.nn.functional as F
from anyup.data.training.losses import temporal_consistency_loss

def test_cos_mse_zero_on_identical():
    """Identical inputs must yield loss = 0."""
    B, T, h, w, C = 2, 4, 7, 7, 768  # ↑ T=4 matches early curriculum stage (task 5.3)
                                       # ↑ h=w=7 matches ViT-B/16 patch grid at 112px input
                                       # ↑ C=768 matches VideoMAE / DINOv2 ViT-B feature dim
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
    """Perfectly anti-parallel features → cosine term = 2, plus MSE contribution."""
    B, T, h, w, C = 1, 1, 1, 1, 4   # ↓ minimal size — just testing the math
    pred   = torch.ones(B, T, h, w, C)
    target = -torch.ones(B, T, h, w, C)
    loss = cos_mse_loss(pred, target)
    # cos_loss = 1 - (-1) = 2; mse = mean((1-(-1))^2) = 4
    expected = 2.0 + 4.0
    assert abs(loss.item() - expected) < 1e-4, f"Expected {expected}, got {loss.item()}"
    print(f"PASS  test_cos_mse_opposite_vectors: loss={loss.item():.4f} (expected {expected})")


def test_cos_mse_t1_matches_2d_reduction():
    """With T=1, loss should equal what you'd get treating (B,h,w,C) as the 2D case."""
    import torch.nn.functional as F
    B, h, w, C = 2, 7, 7, 768
    pred   = torch.randn(B, h, w, C)
    target = torch.randn(B, h, w, C)
    # 3D call with T=1
    loss_3d = cos_mse_loss(pred.unsqueeze(1), target.unsqueeze(1))
    # Manual 2D equivalent
    pf, tf  = pred.reshape(B, h*w, C), target.reshape(B, h*w, C)
    cos_loss_2d = (1 - F.cosine_similarity(pf, tf, dim=-1)).mean()
    mse_loss_2d = F.mse_loss(pf, tf)
    loss_2d = cos_loss_2d + mse_loss_2d
    assert abs(loss_3d.item() - loss_2d.item()) < 1e-5, \
        f"T=1 mismatch: 3D={loss_3d.item():.6f}, 2D={loss_2d.item():.6f}"
    print(f"PASS  test_cos_mse_t1_matches_2d_reduction: 3D={loss_3d.item():.6f} == 2D={loss_2d.item():.6f}")

def test_input_consistency_zero_on_consistent():
    """
    If q_pred is a perfect spatial upsampling of p (nearest-neighbor tiled),
    downsampling it back should recover p exactly → loss ≈ 0.
    """
    B, T, h, w, C = 2, 4, 7, 7, 768    # ↑ h,w = low-res (e.g. ViT patch grid at 112px)
    scale = 4                            # ↑ upsampling factor; change if model factor differs
    H, W = h * scale, w * scale         # ↑ high-res dims derived from low-res × scale

    p = torch.randn(B, T, h, w, C)

    # Build a q_pred that is a block-constant upsampling of p
    # permute to (B*T, C, h, w), upsample, permute back
    p_up = p.permute(0, 1, 4, 2, 3).reshape(B * T, C, h, w)  # (B*T, C, h, w)
    p_up = F.interpolate(p_up, size=(H, W), mode="nearest")   # (B*T, C, H, W)
    p_up = p_up.reshape(B, T, C, H, W).permute(0, 1, 3, 4, 2)  # (B, T, H, W, C)

    loss = input_consistency_loss(p_up, p)
    assert loss.item() < 1e-4, f"Expected ~0, got {loss.item()}"
    print(f"PASS  test_input_consistency_zero_on_consistent: loss={loss.item():.2e}")


def test_input_consistency_positive_on_random():
    """Random high-res output vs random p must give positive loss."""
    B, T, h, w, C = 2, 4, 7, 7, 768
    H, W = 28, 28                       # ↑ 4× upsampling; keep consistent with model config

    q_pred = torch.randn(B, T, H, W, C)
    p      = torch.randn(B, T, h, w, C)

    loss = input_consistency_loss(q_pred, p)
    assert loss.item() > 0.0, f"Expected positive loss, got {loss.item()}"
    print(f"PASS  test_input_consistency_positive_on_random: loss={loss.item():.4f}")


def test_input_consistency_t1_shape():
    """T=1 (warmup curriculum) must work without shape errors."""
    B, T, h, w, C = 2, 1, 7, 7, 768   # ↑ T=1 = single-frame warmup stage (task 5.3)
    H, W = 28, 28

    q_pred = torch.randn(B, T, H, W, C)
    p      = torch.randn(B, T, h, w, C)

    loss = input_consistency_loss(q_pred, p)
    assert loss.ndim == 0, "Loss must be a scalar"
    print(f"PASS  test_input_consistency_t1_shape: loss={loss.item():.4f}")


def test_self_consistency_zero_on_identical_video():
    """
    If augmentation produces the exact same video (std=0, brightness/contrast=1),
    both branches output the same features → loss ≈ 0.
    """
    import torch.nn as nn
    from anyup.data.training.losses import self_consistency_loss

    # Minimal identity model stub — returns p tiled to high-res, ignoring V
    class IdentityModel(nn.Module):
        def forward(self, p, V):
            B, T, h, w, C = p.shape        # ↑ h,w = low-res
            H, W = V.shape[2], V.shape[3]  # ↑ H,W = high-res (from guidance video)
            p_up = p.permute(0,1,4,2,3).reshape(B*T, C, h, w)
            p_up = F.interpolate(p_up, size=(H, W), mode="nearest")
            return p_up.reshape(B, T, C, H, W).permute(0,1,3,4,2)

    B, T, h, w, C = 2, 4, 7, 7, 64    # ↑ small C for test speed
    H, W = 28, 28                      # ↑ 4× upsampling
    p = torch.randn(B, T, h, w, C)
    V = torch.rand(B, T, H, W, 3)     # float in [0,1]

    model = IdentityModel()

    loss = self_consistency_loss(
        model, p, V,
        brightness_range=(1.0, 1.0),   # no brightness jitter
        contrast_range=(1.0, 1.0),     # no contrast jitter
        noise_std=0.0,                 # no noise
    )
    assert loss.item() < 1e-5, f"Expected ~0, got {loss.item()}"
    print(f"PASS  test_self_consistency_zero_on_identical_video: loss={loss.item():.2e}")


def test_self_consistency_positive_on_augmented():
    """With real augmentation, a non-trivial model should give positive loss."""
    import torch.nn as nn
    from anyup.data.training.losses import self_consistency_loss

    class NoisySensitiveModel(nn.Module):
        """Model that directly uses V pixel values — maximally sensitive to augmentation."""
        def forward(self, p, V):
            B, T, H, W, _ = V.shape
            C = p.shape[-1]
            # project V channels to C and return — not a real model, just for test
            out = V.mean(dim=-1, keepdim=True).expand(B, T, H, W, C)
            return out

    B, T, h, w, C = 2, 4, 7, 7, 64
    H, W = 28, 28
    p = torch.randn(B, T, h, w, C)
    V = torch.rand(B, T, H, W, 3)

    model = NoisySensitiveModel()
    loss = self_consistency_loss(model, p, V)   # default augmentation strength
    assert loss.item() > 0.0, f"Expected positive loss, got {loss.item()}"
    print(f"PASS  test_self_consistency_positive_on_augmented: loss={loss.item():.4f}")


def test_self_consistency_no_grad_on_aug_branch():
    """Augmented branch output must have requires_grad=False."""
    from anyup.data.training.augmentations import augment_guidance_video

    B, T, H, W = 2, 4, 28, 28
    V = torch.rand(B, T, H, W, 3, requires_grad=False)
    V_aug = augment_guidance_video(V)
    # V_aug should be a plain tensor with no grad history
    assert not V_aug.requires_grad, "Augmented video must not carry gradients"
    print(f"PASS  test_self_consistency_no_grad_on_aug_branch")




def test_temporal_consistency_t1_returns_zero():
    """T=1 warmup stage — no pairs exist, must return exactly 0."""
    B, T, H, W, C = 2, 1, 28, 28, 64  # ↑ T=1 = curriculum warmup (task 5.3)
    q_pred = torch.randn(B, T, H, W, C)
    V      = torch.rand(B, T, H, W, 3)
    loss   = temporal_consistency_loss(q_pred, V)
    assert loss.item() == 0.0, f"Expected 0.0, got {loss.item()}"
    print(f"PASS  test_temporal_consistency_t1_returns_zero: loss={loss.item()}")


def test_temporal_consistency_zero_on_static_identical_features():
    """
    Static scene (identical RGB frames) + identical features across time → loss = 0.
    Gate should pass all pairs (diff=0 < threshold) and cos-mse(q, q) = 0.
    """
    B, T, H, W, C = 2, 4, 28, 28, 64  # ↑ T=4 = early curriculum stage
    # identical frames across time — gate will pass all pairs
    V      = torch.rand(B, 1, H, W, 3).expand(B, T, H, W, 3).contiguous()
    q_pred = torch.randn(B, 1, H, W, C).expand(B, T, H, W, C).contiguous()

    loss = temporal_consistency_loss(q_pred, V, rgb_diff_threshold=0.05)
    assert loss.item() < 1e-5, f"Expected ~0, got {loss.item()}"
    print(f"PASS  test_temporal_consistency_zero_on_static_identical_features: loss={loss.item():.2e}")


def test_temporal_consistency_gate_blocks_fast_motion():
    """
    If all consecutive RGB frames differ above threshold, every pair is gated out.
    Loss should return 0 (no active pairs), regardless of feature values.
    """
    B, T, H, W, C = 2, 4, 28, 28, 64
    # alternating black/white frames — guaranteed large RGB diff
    V = torch.zeros(B, T, H, W, 3)
    V[:, 1::2, ...] = 1.0          # every other frame is white → diff = 1.0 >> threshold

    q_pred = torch.randn(B, T, H, W, C)  # random features — irrelevant since all gated

    loss = temporal_consistency_loss(q_pred, V, rgb_diff_threshold=0.05)
    assert loss.item() == 0.0, f"Expected 0.0 (all gated), got {loss.item()}"
    print(f"PASS  test_temporal_consistency_gate_blocks_fast_motion: loss={loss.item()}")


def test_temporal_consistency_positive_on_flickering_features():
    """
    Static RGB (gate passes all pairs) + different features each frame → positive loss.
    """
    B, T, H, W, C = 2, 4, 28, 28, 64
    V      = torch.rand(B, 1, H, W, 3).expand(B, T, H, W, 3).contiguous()  # static scene
    q_pred = torch.randn(B, T, H, W, C)   # independent random features per frame

    loss = temporal_consistency_loss(q_pred, V, rgb_diff_threshold=0.05)
    assert loss.item() > 0.0, f"Expected positive loss, got {loss.item()}"
    print(f"PASS  test_temporal_consistency_positive_on_flickering_features: loss={loss.item():.4f}")


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
    print("\nAll tests passed.")


