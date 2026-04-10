
import torch
from anyup.data.training.losses import cos_mse_loss


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


if __name__ == "__main__":
    test_cos_mse_zero_on_identical()
    test_cos_mse_positive_on_different()
    test_cos_mse_opposite_vectors()
    test_cos_mse_t1_matches_2d_reduction()
    print("\nAll tests passed.")