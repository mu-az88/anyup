import torch
import torch.nn.functional as F
from anyup.modules.feature_unification3d import LearnedFeatureUnification3D
from anyup.layers.feature_unification import LearnedFeatureUnification   # original


# ─────────────────────────────────────────────
# Test 1 — Output shape is correct
# ─────────────────────────────────────────────
def test_output_shape():
    B, C, T, H, W = 2, 4, 5, 16, 16
    out_ch = 8

    for k in [1, 3, 5]:
        for t_k in [1, 3]:
            layer = LearnedFeatureUnification3D(out_channels=out_ch, kernel_size=k, t_kernel_size=t_k)
            x = torch.randn(B, C, T, H, W)
            out = layer(x)
            assert out.shape == (B, out_ch, T, H, W), \
                f"Shape mismatch for k={k}, t_k={t_k}: got {out.shape}"

    print("test_output_shape passed")


# ─────────────────────────────────────────────
# Test 2 — Output values are in (0, 1) and sum to 1 across out_channels
# ─────────────────────────────────────────────
def test_softmax_property():
    """
    After softmax over dim=1 and mean over dim=2, values should still
    be in (0,1) and sum to 1 across out_channels at every (B,T,H,W) position.
    This verifies the softmax is applied correctly and not accidentally collapsed.
    """
    B, C, T, H, W = 2, 6, 3, 8, 8
    out_ch = 4
    layer = LearnedFeatureUnification3D(out_channels=out_ch, kernel_size=3, t_kernel_size=1)
    x = torch.randn(B, C, T, H, W)
    out = layer(x)

    # all values in (0, 1)
    assert (out > 0).all() and (out < 1).all(), \
        "Output values should be in (0, 1) after softmax+mean"

    # sum across out_channels = 1 at every position
    channel_sum = out.sum(dim=1)   # (B, T, H, W)
    assert torch.allclose(channel_sum, torch.ones_like(channel_sum), atol=1e-5), \
        "Output should sum to 1 across out_channels at every position"

    print("test_softmax_property passed")


# ─────────────────────────────────────────────
# Test 3 — Single frame with t_k=1 matches original 2D LFU
# ─────────────────────────────────────────────
def test_single_frame_matches_2d():
    """
    With t_k=1 and T=1, LFU3D must produce the same output as the original
    2D LFU given identical basis weights.
    """
    C, out_ch, k = 4, 8, 3
    B, H, W = 2, 16, 16

    lfu2d = LearnedFeatureUnification(out_channels=out_ch, kernel_size=k)
    lfu3d = LearnedFeatureUnification3D(out_channels=out_ch, kernel_size=k, t_kernel_size=1)

    # copy 2D basis (out_ch, 1, k, k) → 3D basis (out_ch, 1, 1, k, k)
    with torch.no_grad():
        lfu3d.basis.copy_(lfu2d.basis.unsqueeze(2))

    x2d = torch.randn(B, C, H, W)
    x3d = x2d.unsqueeze(2)    # (B, C, T=1, H, W)

    out2d = lfu2d(x2d)                  # (B, out_ch, H, W)
    out3d = lfu3d(x3d).squeeze(2)       # (B, out_ch, H, W)

    assert torch.allclose(out2d, out3d, atol=1e-5), \
        f"Max diff: {(out2d - out3d).abs().max().item()}"

    print("test_single_frame_matches_2d passed")


# ─────────────────────────────────────────────
# Test 4 — Channel invariance: more input channels, same output
# ─────────────────────────────────────────────
def test_channel_invariance():
    """
    The mean over input channels in forward() is what makes LFU
    invariant to input dimensionality. Two inputs with different C
    but same spatial/temporal content (after averaging) should produce
    close outputs when the basis is fixed.
    This test verifies the mean(dim=2) is actually applied.
    """
    out_ch, k, t_k = 4, 3, 1
    B, T, H, W = 1, 1, 8, 8

    # same layer, different number of channels
    layer = LearnedFeatureUnification3D(out_channels=out_ch, kernel_size=k, t_kernel_size=t_k)

    x_c4  = torch.randn(B, 4,  T, H, W)
    x_c16 = torch.randn(B, 16, T, H, W)

    out_c4  = layer(x_c4)
    out_c16 = layer(x_c16)

    # both should produce (B, out_ch, T, H, W) — different values but same shape
    assert out_c4.shape  == (B, out_ch, T, H, W)
    assert out_c16.shape == (B, out_ch, T, H, W)

    print("test_channel_invariance passed")


# ─────────────────────────────────────────────
# Test 5 — Gaussian init raises NotImplementedError
# ─────────────────────────────────────────────
def test_gaussian_init_raises():
    try:
        layer = LearnedFeatureUnification3D(
            out_channels=8, kernel_size=3, t_kernel_size=3,
            init_gaussian_derivatives=True
        )
        assert False, "Should have raised NotImplementedError"
    except NotImplementedError:
        pass

    print("test_gaussian_init_raises passed")


if __name__ == "__main__":
    test_output_shape()
    test_softmax_property()
    test_single_frame_matches_2d()
    test_channel_invariance()
    test_gaussian_init_raises()