import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from anyup.model import AnyUp
from anyup.modules import RoPE3D
from anyup.modules.create_coordinates3d import create_coordinates_3d
from anyup.modules.cross_attention3d import CrossAttentionBlock3D


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_model(qk_dim=32, num_heads=4):
    return AnyUp(
        input_dim=3,
        qk_dim=qk_dim,
        kernel_size=1,
        kernel_size_lfu=3,
        window_ratio=0.0,
        num_heads=num_heads,
        t_k=1,
    ).eval()


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — Flatten step: (B, C, T, H, W) → (B, T*H*W, C)
# ─────────────────────────────────────────────────────────────────────────────

def test_flatten_shape():
    """
    The permute(0,2,3,4,1).reshape(...) in forward must produce exactly
    (B, T*H*W, C) tokens for the RoPE call.
    """
    B, C, T, H, W = 2, 16, 3, 8, 8
    enc = torch.randn(B, C, T, H, W)

    flat = enc.permute(0, 2, 3, 4, 1).reshape(B, -1, C)

    assert flat.shape == (B, T * H * W, C), \
        f"Expected ({B}, {T*H*W}, {C}), got {flat.shape}"
    print("test_flatten_shape passed")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — Restore step: (B, T*H*W, C) → (B, C, T, H, W)
# ─────────────────────────────────────────────────────────────────────────────

def test_restore_shape():
    B, C, T, H, W = 2, 16, 3, 8, 8
    tokens = torch.randn(B, T * H * W, C)

    restored = tokens.view(B, T, H, W, C).permute(0, 4, 1, 2, 3)

    assert restored.shape == (B, C, T, H, W), \
        f"Expected ({B}, {C}, {T}, {H}, {W}), got {restored.shape}"
    print("test_restore_shape passed")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — Flatten → restore is a lossless round-trip
# ─────────────────────────────────────────────────────────────────────────────

def test_flatten_restore_roundtrip():
    """
    Applying flatten then restore should recover the original tensor exactly.
    This confirms the axis ordering is self-consistent.
    """
    B, C, T, H, W = 2, 16, 3, 8, 8
    enc = torch.randn(B, C, T, H, W)

    flat = enc.permute(0, 2, 3, 4, 1).reshape(B, -1, C)
    restored = flat.view(B, T, H, W, C).permute(0, 4, 1, 2, 3)

    assert torch.allclose(enc, restored), \
        f"Round-trip failed; max diff: {(enc - restored).abs().max().item()}"
    print("test_flatten_restore_roundtrip passed")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — output_size default uses spatial dims only (H, W), not T
# ─────────────────────────────────────────────────────────────────────────────

def test_output_size_default_is_spatial_only():
    """
    For a 5D image (B, C, T, H, W), image.shape[-2:] must return (H, W),
    not (T, H) or any other pair.
    """
    B, C, T, H, W = 2, 3, 4, 64, 64
    image = torch.zeros(B, C, T, H, W)

    output_size = image.shape[-2:]

    assert tuple(output_size) == (H, W), \
        f"Expected ({H}, {W}), got {tuple(output_size)}"
    print("test_output_size_default_is_spatial_only passed")


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — create_coordinates_3d is called with the correct enc dims
# ─────────────────────────────────────────────────────────────────────────────

def test_coords_shape_matches_enc():
    """
    The coordinates tensor must have one entry per spatiotemporal token,
    i.e. shape (1, T*H*W, 3), matching what enc.shape[-3:] produces.
    """
    T, H, W = 3, 8, 6
    coords = create_coordinates_3d(T, H, W)

    assert coords.shape == (1, T * H * W, 3), \
        f"Expected (1, {T*H*W}, 3), got {coords.shape}"

    # Also check device/dtype forwarding works
    enc_like = torch.zeros(1, 16, T, H, W)
    coords2 = create_coordinates_3d(
        enc_like.shape[-3], enc_like.shape[-2], enc_like.shape[-1],
        device=enc_like.device, dtype=enc_like.dtype,
    )
    assert coords2.device == enc_like.device
    assert coords2.dtype == enc_like.dtype
    print("test_coords_shape_matches_enc passed")


# ─────────────────────────────────────────────────────────────────────────────
# Test 6 — RoPE application preserves the token sequence length and channel dim
# ─────────────────────────────────────────────────────────────────────────────

def test_rope_preserves_shape():
    """
    After the full flatten → RoPE → restore sandwich the output must have
    the same shape as the input enc tensor.
    """
    B, C, T, H, W = 2, 32, 3, 8, 8
    enc = torch.randn(B, C, T, H, W)

    rope = RoPE3D(C)
    rope._device_weight_init()

    coords = create_coordinates_3d(T, H, W)

    flat = enc.permute(0, 2, 3, 4, 1).reshape(B, -1, C)
    flat = rope(flat, coords)
    restored = flat.view(B, T, H, W, C).permute(0, 4, 1, 2, 3)

    assert restored.shape == enc.shape, \
        f"Expected {enc.shape}, got {restored.shape}"
    print("test_rope_preserves_shape passed")


# ─────────────────────────────────────────────────────────────────────────────
# Test 7 — Integration: full forward pass shape (multi-frame)
# ─────────────────────────────────────────────────────────────────────────────

def test_forward_output_shape_multiframe():
    """
    End-to-end forward pass with T>1. Output should be
    (B, C_feat, T, H_out, W_out) where (H_out, W_out) is the image resolution.
    """
    B, T = 1, 3
    qk_dim, C_feat = 32, 64
    H_img, W_img = 16, 16
    H_feat, W_feat = 8, 8

    model = _make_model(qk_dim=qk_dim)
    image = torch.randn(B, 3, T, H_img, W_img)
    features = torch.randn(B, C_feat, T, H_feat, W_feat)

    with torch.no_grad():
        out = model(image, features)

    expected = (B, C_feat, T, H_img, W_img)
    assert out.shape == expected, f"Expected {expected}, got {out.shape}"
    print("test_forward_output_shape_multiframe passed")


# ─────────────────────────────────────────────────────────────────────────────
# Test 8 — Integration: explicit output_size is respected
# ─────────────────────────────────────────────────────────────────────────────

def test_forward_explicit_output_size():
    """
    When output_size=(H_out, W_out) is passed explicitly, the spatial dims
    of the output must match it, not the image dims.
    """
    B, T = 1, 2
    qk_dim, C_feat = 32, 64
    H_img, W_img = 16, 16
    H_feat, W_feat = 8, 8
    H_out, W_out = 32, 32   # upsampling beyond the image resolution

    model = _make_model(qk_dim=qk_dim)
    image = torch.randn(B, 3, T, H_img, W_img)
    features = torch.randn(B, C_feat, T, H_feat, W_feat)

    with torch.no_grad():
        out = model(image, features, output_size=(H_out, W_out))

    expected = (B, C_feat, T, H_out, W_out)
    assert out.shape == expected, f"Expected {expected}, got {out.shape}"
    print("test_forward_explicit_output_size passed")


# ─────────────────────────────────────────────────────────────────────────────
# Test 9 — Integration: T=1 (single frame) runs without error
# ─────────────────────────────────────────────────────────────────────────────

def test_forward_single_frame():
    """
    T=1 is the degenerate case of the 3D model; it must still run correctly
    and produce the right output shape.
    """
    B, T = 1, 1
    qk_dim, C_feat = 32, 64
    H_img, W_img = 16, 16
    H_feat, W_feat = 8, 8

    model = _make_model(qk_dim=qk_dim)
    image = torch.randn(B, 3, T, H_img, W_img)
    features = torch.randn(B, C_feat, T, H_feat, W_feat)

    with torch.no_grad():
        out = model(image, features)

    expected = (B, C_feat, T, H_img, W_img)
    assert out.shape == expected, f"Expected {expected}, got {out.shape}"
    print("test_forward_single_frame passed")


# ─────────────────────────────────────────────────────────────────────────────
# Test 10 — Integration: temporal dimension is never altered
# ─────────────────────────────────────────────────────────────────────────────

def test_forward_temporal_dim_preserved():
    """
    AnyUp performs spatial-only upsampling. For every T value the output's
    T must equal the input features' T.
    """
    qk_dim, C_feat = 32, 64
    model = _make_model(qk_dim=qk_dim)

    for T in [1, 2, 5]:
        image = torch.randn(1, 3, T, 16, 16)
        features = torch.randn(1, C_feat, T, 8, 8)

        with torch.no_grad():
            out = model(image, features)

        assert out.shape[2] == T, \
            f"T={T}: expected output T={T}, got {out.shape[2]}"

    print("test_forward_temporal_dim_preserved passed")


if __name__ == "__main__":
    test_flatten_shape()
    test_restore_shape()
    test_flatten_restore_roundtrip()
    test_output_size_default_is_spatial_only()
    test_coords_shape_matches_enc()
    test_rope_preserves_shape()
    test_forward_output_shape_multiframe()
    test_forward_explicit_output_size()
    test_forward_single_frame()
    test_forward_temporal_dim_preserved()
