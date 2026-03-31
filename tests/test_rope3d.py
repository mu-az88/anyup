import torch
import torch.nn as nn

from anyup.modules import RoPE3D
from anyup.modules.RoPE3d import RoPE3D
from anyup.modules import create_coordinates3d
from anyup.modules.create_coordinates3d import create_coordinate_3d


# ─────────────────────────────────────────────
# Test 1 — freqs parameter shape is (3, dim)
# ─────────────────────────────────────────────
def test_freqs_shape():
    """
    freqs must be (3, dim) — one row per axis (z, x, y).
    The matrix multiply coords @ freqs only works if freqs
    has exactly as many rows as coords has coordinate components.
    """
    for dim in [12, 64, 128]:
        rope = RoPE3D(dim=dim)
        assert rope.freqs.shape == (3, dim), \
            f"Expected freqs shape (3, {dim}), got {rope.freqs.shape}"

    print("test_freqs_shape passed")


# ─────────────────────────────────────────────
# Test 2 — _device_weight_init fills correct blocks
# ─────────────────────────────────────────────
def test_weight_init_block_structure():
    """
    After _device_weight_init, each axis must occupy its own
    dedicated third of the dim columns, with zeros everywhere else.

    dim=12 → each block is 4 columns wide:
      z row: cols  0.. 3 nonzero, rest zero
      x row: cols  4.. 7 nonzero, rest zero
      y row: cols  8..11 nonzero, rest zero
    """
    dim = 12
    rope = RoPE3D(dim=dim)
    rope._device_weight_init()

    block = dim // 3   # 4

    freqs = rope.freqs.data

    # z row — first block nonzero, rest zero
    assert (freqs[0, :block]  != 0).all(),  "z block should be nonzero"
    assert (freqs[0, block:]  == 0).all(),  "z row outside its block should be zero"

    # x row — middle block nonzero, rest zero
    assert (freqs[1, block:2*block] != 0).all(), "x block should be nonzero"
    assert (freqs[1, :block]        == 0).all(), "x row before its block should be zero"
    assert (freqs[1, 2*block:]      == 0).all(), "x row after its block should be zero"

    # y row — last block nonzero, rest zero
    assert (freqs[2, 2*block:] != 0).all(), "y block should be nonzero"
    assert (freqs[2, :2*block] == 0).all(), "y row outside its block should be zero"

    print("test_weight_init_block_structure passed")


# ─────────────────────────────────────────────
# Test 3 — Output shape is correct
# ─────────────────────────────────────────────
def test_output_shape():
    """
    RoPE3D must return a tensor with the exact same shape as its input x.
    Position encoding rotates the vector — it never changes dimensionality.
    """
    dim = 32
    rope = RoPE3D(dim=dim)
    rope._device_weight_init()

    cases = [
        (1, 1, 4, 4),   # B=1, T=1
        (2, 3, 8, 8),   # B=2, T=3
        (1, 5, 14, 14), # typical video input
    ]

    for B, T, H, W in cases:
        x      = torch.randn(B, T * H * W, dim)
        coords = create_coordinate_3d(T, H, W)                       # (1, T*H*W, 3)
        coords = coords.expand(B, -1, -1)                            # (B, T*H*W, 3)
        out    = rope(x, coords)
        assert out.shape == x.shape, \
            f"Shape mismatch for B={B},T={T},H={H},W={W}: got {out.shape}"

    print("test_output_shape passed")


# ─────────────────────────────────────────────
# Test 4 — Rotation preserves vector magnitude
# ─────────────────────────────────────────────
def test_rotation_preserves_norm():
    """
    Rotation is a rigid transformation — it changes direction but never length.
    ||RoPE(x)|| must equal ||x|| at every position.
    If this fails, RoPE is scaling the vector, which is wrong.
    """
    dim = 32
    B, T, H, W = 2, 3, 8, 8

    rope = RoPE3D(dim=dim)
    rope._device_weight_init()

    x      = torch.randn(B, T * H * W, dim)
    coords = create_coordinate_3d(T, H, W).expand(B, -1, -1)
    out    = rope(x, coords)

    norm_in  = x.norm(dim=-1)
    norm_out = out.norm(dim=-1)

    assert torch.allclose(norm_in, norm_out, atol=1e-5), \
        f"Norm not preserved. Max diff: {(norm_in - norm_out).abs().max().item()}"

    print("test_rotation_preserves_norm passed")


# ─────────────────────────────────────────────
# Test 5 — Different positions produce different outputs
# ─────────────────────────────────────────────
def test_different_positions_differ():
    """
    Two patches with identical content but different coordinates
    must produce different outputs after RoPE.
    This is the whole point — position is baked into the vector direction.
    """
    dim = 32
    rope = RoPE3D(dim=dim)
    rope._device_weight_init()

    x = torch.randn(1, dim).unsqueeze(0)   # same content vector, shape (1, 1, dim)

    coord_a = torch.tensor([[[0.0, 0.0, 0.0]]])   # position A
    coord_b = torch.tensor([[[0.5, 0.5, 0.5]]])   # position B

    out_a = rope(x, coord_a)
    out_b = rope(x, coord_b)

    assert not torch.allclose(out_a, out_b, atol=1e-5), \
        "Different positions produced identical outputs — position encoding is broken"

    print("test_different_positions_differ passed")


# ─────────────────────────────────────────────
# Test 6 — Same position always produces same output
# ─────────────────────────────────────────────
def test_same_position_same_output():
    """
    RoPE is deterministic — calling it twice with the same x and coords
    must give identical results. No randomness should survive after init.
    """
    dim = 32
    B, T, H, W = 1, 2, 4, 4

    rope = RoPE3D(dim=dim)
    rope._device_weight_init()

    x      = torch.randn(B, T * H * W, dim)
    coords = create_coordinate_3d(T, H, W).expand(B, -1, -1)

    out1 = rope(x, coords)
    out2 = rope(x, coords)

    assert torch.allclose(out1, out2, atol=1e-6), \
        "RoPE produced different outputs for the same input — non-deterministic behaviour"

    print("test_same_position_same_output passed")


# ─────────────────────────────────────────────
# Test 7 — Single frame (T=1) matches original 2D RoPE
# ─────────────────────────────────────────────
def test_single_frame_matches_2d():
    """
    With T=1, RoPE3D must produce the same output as the original 2D RoPE
    given identical weights. This is the primary correctness anchor —
    3D is a strict extension of 2D, not a replacement.

    Weight transfer:
      2D freqs shape: (2, dim)  — rows for x and y
      3D freqs shape: (3, dim)  — rows for z, x, y

    We copy the 2D weights into the x and y rows of 3D (rows 1 and 2),
    and zero out the z row (row 0) so the temporal axis has no effect.
    With T=1, z coordinate is always 0.0, so the z row contributes
    zero to the angle regardless — outputs must then match exactly.
    """
    dim = 12
    H, W = 7, 9
    B = 2

    rope2d = RoPE(dim=dim)
    rope3d = RoPE3D(dim=dim)

    rope2d._device_weight_init()
    rope3d._device_weight_init()

    # copy 2D x and y rows into 3D x and y rows (rows 1 and 2)
    with torch.no_grad():
        rope3d.freqs[0].zero_()               # z row → zero (no temporal contribution)
        rope3d.freqs[1].copy_(rope2d.freqs[0])  # x row ← 2D row 0
        rope3d.freqs[2].copy_(rope2d.freqs[1])  # y row ← 2D row 1

    x = torch.randn(B, H * W, dim)

    coords2d = create_coordinate(H, W).expand(B, -1, -1)          # (B, H*W, 2)
    coords3d = create_coordinate_3d(1, H, W).expand(B, -1, -1)    # (B, H*W, 3)

    out2d = rope2d(x, coords2d)
    out3d = rope3d(x, coords3d)

    assert torch.allclose(out2d, out3d, atol=1e-5), \
        f"Single-frame 3D output does not match 2D. Max diff: {(out2d - out3d).abs().max().item()}"

    print("test_single_frame_matches_2d passed")


if __name__ == "__main__":
    test_freqs_shape()
    test_weight_init_block_structure()
    test_output_shape()
    test_rotation_preserves_norm()
    test_different_positions_differ()
    test_same_position_same_output()
    test_single_frame_matches_2d()
