import torch
import torch.nn as nn
from anyup.layers.convolutions import ResBlock          # original 2D
from anyup.modules import ResBlock3D

# ─────────────────────────────────────────────
# Test 1 — Output shape is correct
# ─────────────────────────────────────────────
def test_output_shape():
    B, T, H, W = 2, 5, 16, 16

    cases = [
        dict(in_channels=8,  out_channels=8,  kernel_size=1, t_kernel_size=1),
        dict(in_channels=8,  out_channels=16, kernel_size=1, t_kernel_size=1),  # channel change → conv shortcut
        dict(in_channels=8,  out_channels=8,  kernel_size=3, t_kernel_size=1),
        dict(in_channels=8,  out_channels=8,  kernel_size=3, t_kernel_size=3),
    ]

    for cfg in cases:
        block = ResBlock3D(**cfg, num_groups=8, norm_fn=nn.GroupNorm, activation_fn=nn.SiLU)
        x = torch.randn(B, cfg["in_channels"], T, H, W)
        out = block(x)
        assert out.shape == (B, cfg["out_channels"], T, H, W), \
            f"Shape mismatch for {cfg}: got {out.shape}"

    print("test_output_shape passed")


# ─────────────────────────────────────────────
# Test 2 — Residual connection is active
# ─────────────────────────────────────────────
def test_residual_connection():
    """
    Zero out all weights in self.block so it outputs zero.
    The output should then equal the shortcut path only.
    This directly verifies the + in forward() is wired correctly.
    """
    B, C, T, H, W = 1, 8, 3, 8, 8
    block = ResBlock3D(in_channels=C, out_channels=C, kernel_size=1, t_kernel_size=1)

    # zero out all block weights so block(x) = 0
    with torch.no_grad():
        for p in block.block.parameters():
            p.zero_()

    x = torch.randn(B, C, T, H, W)

    # shortcut is Identity when in_ch == out_ch and use_conv_shortcut=False
    expected = x
    out = block(x)

    assert torch.allclose(out, expected, atol=1e-6), \
        "Residual connection broken — output does not match shortcut when block is zeroed"

    print("test_residual_connection passed")


# ─────────────────────────────────────────────
# Test 3 — Single frame with t_k=1 matches original 2D ResBlock
# ─────────────────────────────────────────────
def test_single_frame_matches_2d():
    """
    With t_k=1 and T=1, ResBlock3D must produce the same output as
    the original 2D ResBlock given identical weights.
    Uses kernel_size=1 to avoid reflect padding differences.
    """
    B, C, H, W = 2, 8, 16, 16

    rb2d = ResBlock(
        in_channels=C, out_channels=C,
        kernel_size=1, num_groups=8,
        pad_mode="zeros", norm_fn=nn.GroupNorm, activation_fn=nn.SiLU
    )
    rb3d = ResBlock3D(
        in_channels=C, out_channels=C,
        kernel_size=1, t_kernel_size=1, num_groups=8,
        pad_mode="zeros", norm_fn=nn.GroupNorm, activation_fn=nn.SiLU
    )

    # copy weights: Conv2d(C,C,1,1) → Conv3d(C,C,1,1,1)
    with torch.no_grad():
        params2d = list(rb2d.block.parameters())
        params3d = list(rb3d.block.parameters())
        for p2, p3 in zip(params2d, params3d):
            if p2.dim() == 4:
                # Conv2d weight (out, in, k, k) → Conv3d weight (out, in, 1, k, k)
                p3.copy_(p2.unsqueeze(2))
            else:
                # GroupNorm weight/bias: same shape
                p3.copy_(p2)

        # shortcut: both are Identity here so nothing to copy

    x2d = torch.randn(B, C, H, W)
    x3d = x2d.unsqueeze(2)             # (B, C, T=1, H, W)

    out2d = rb2d(x2d)
    out3d = rb3d(x3d).squeeze(2)       # back to (B, C, H, W)

    assert torch.allclose(out2d, out3d, atol=1e-5), \
        f"Max diff: {(out2d - out3d).abs().max().item()}"

    print("test_single_frame_matches_2d passed")


# ─────────────────────────────────────────────
# Test 4 — Temporal dimension is preserved across all t_k values
# ─────────────────────────────────────────────
def test_temporal_preservation():
    """
    T must be the same before and after the block for all t_k values.
    This catches off-by-one errors in temporal padding.
    """
    B, C, H, W = 1, 8, 8, 8

    for T in [1, 3, 5, 8]:
        for t_k in [1, 3]:
            block = ResBlock3D(
                in_channels=C, out_channels=C,
                kernel_size=1, t_kernel_size=t_k,
                pad_mode="zeros"
            )
            x = torch.randn(B, C, T, H, W)
            out = block(x)
            assert out.shape[2] == T, \
                f"T={T} not preserved with t_k={t_k}: got T={out.shape[2]}"

    print("test_temporal_preservation passed")


# ─────────────────────────────────────────────
# Test 5 — pad_mode='reflect' works without error for k>1
# ─────────────────────────────────────────────
def test_reflect_pad_mode():
    """
    Verifies SpatialReflectConv3d is correctly wired in when pad_mode='reflect'.
    Checks shape only — correctness of reflect padding is covered by Unit 1 tests.
    """
    B, C, T, H, W = 2, 8, 4, 16, 16
    block = ResBlock3D(
        in_channels=C, out_channels=C,
        kernel_size=3, t_kernel_size=1,
        pad_mode="reflect", num_groups=8,
        norm_fn=nn.GroupNorm, activation_fn=nn.SiLU
    )
    x = torch.randn(B, C, T, H, W)
    out = block(x)
    assert out.shape == (B, C, T, H, W)

    print("test_reflect_pad_mode passed")


if __name__ == "__main__":
    test_output_shape()
    test_residual_connection()
    test_single_frame_matches_2d()
    test_temporal_preservation()
    test_reflect_pad_mode()