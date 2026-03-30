import torch
import pytest

from anyup.utils.img import create_coordinate
from anyup.modules import create_coordinate_3d


# ─────────────────────────────────────────────
# Test 1 — Output shape is correct (2D)
# ─────────────────────────────────────────────
def test_2d_output_shape():
    cases = [(1, 1), (4, 6), (8, 8), (14, 14)]
    for h, w in cases:
        coords = create_coordinate(h, w)
        assert coords.shape == (1, h * w, 2), \
            f"Shape mismatch for h={h}, w={w}: got {coords.shape}"

    print("test_2d_output_shape passed")


# ─────────────────────────────────────────────
# Test 2 — Output shape is correct (3D)
# ─────────────────────────────────────────────
def test_3d_output_shape():
    cases = [(1, 1, 1), (3, 4, 6), (5, 8, 8), (4, 14, 14)]
    for t, h, w in cases:
        coords = create_coordinate_3d(t, h, w)
        assert coords.shape == (1, t * h * w, 3), \
            f"Shape mismatch for t={t}, h={h}, w={w}: got {coords.shape}"

    print("test_3d_output_shape passed")


# ─────────────────────────────────────────────
# Test 3 — Coordinate ranges are correct (3D)
# ─────────────────────────────────────────────
def test_3d_coordinate_range_default():
    """All three axes must span exactly [0, 1] with the default start/end."""
    coords = create_coordinate_3d(4, 5, 6)  # t, h, w all different to catch axis mix-ups

    for axis, name in enumerate(["z (temporal)", "x (height)", "y (width)"]):
        lo = coords[..., axis].min().item()
        hi = coords[..., axis].max().item()
        assert lo == pytest.approx(0.0), f"{name} min is {lo}, expected 0.0"
        assert hi == pytest.approx(1.0), f"{name} max is {hi}, expected 1.0"

    print("test_3d_coordinate_range_default passed")


# ─────────────────────────────────────────────
# Test 4 — Custom start/end is respected (3D)
# ─────────────────────────────────────────────
def test_3d_coordinate_range_custom():
    coords = create_coordinate_3d(3, 4, 5, start=-1.0, end=1.0)

    for axis in range(3):
        assert coords[..., axis].min().item() == pytest.approx(-1.0)
        assert coords[..., axis].max().item() == pytest.approx(1.0)

    print("test_3d_coordinate_range_custom passed")


# ─────────────────────────────────────────────
# Test 5 — Axis order follows (T, H, W) convention
# ─────────────────────────────────────────────
def test_3d_axis_order():
    """
    Use asymmetric t/h/w so each axis has a unique linspace.
    Verify that dim-0 of the coord tuple tracks T, dim-1 tracks H, dim-2 tracks W.
    With indexing='ij', z varies slowest (outermost), y varies fastest (innermost).
    """
    t, h, w = 2, 3, 4
    coords = create_coordinate_3d(t, h, w)  # (1, t*h*w, 3)
    coords = coords.view(t, h, w, 3)

    z_expected = torch.linspace(0.0, 1.0, t)
    x_expected = torch.linspace(0.0, 1.0, h)
    y_expected = torch.linspace(0.0, 1.0, w)

    # z (temporal) must be constant across h and w
    assert torch.allclose(coords[:, 0, 0, 0], z_expected), \
        "z axis does not correspond to the temporal dimension"

    # x (height) must be constant across t and w
    assert torch.allclose(coords[0, :, 0, 1], x_expected), \
        "x axis does not correspond to the height dimension"

    # y (width) must be constant across t and h
    assert torch.allclose(coords[0, 0, :, 2], y_expected), \
        "y axis does not correspond to the width dimension"

    print("test_3d_axis_order passed")


# ─────────────────────────────────────────────
# Test 6 — Single frame (T=1) matches 2D output
# ─────────────────────────────────────────────
def test_3d_single_frame_matches_2d():
    """
    With T=1, create_coordinate_3d must produce (x, y) values identical
    to create_coordinate — the z axis is a trivial 0.0 singleton.
    """
    h, w = 7, 9
    coords2d = create_coordinate(h, w)            # (1, h*w, 2)
    coords3d = create_coordinate_3d(1, h, w)      # (1, 1*h*w, 3)

    # x and y components must match exactly
    assert torch.allclose(coords2d[..., 0], coords3d[..., 1], atol=1e-6), \
        "x component differs between 2D and 3D (T=1)"
    assert torch.allclose(coords2d[..., 1], coords3d[..., 2], atol=1e-6), \
        "y component differs between 2D and 3D (T=1)"

    # z must be the scalar start value (0.0 by default)
    assert (coords3d[..., 0] == 0.0).all(), \
        "z component is not 0.0 for T=1"

    print("test_3d_single_frame_matches_2d passed")


# ─────────────────────────────────────────────
# Test 7 — dtype and device are forwarded (3D)
# ─────────────────────────────────────────────
def test_3d_dtype_and_device():
    coords_f64 = create_coordinate_3d(2, 4, 4, dtype=torch.float64)
    assert coords_f64.dtype == torch.float64, \
        f"Expected float64, got {coords_f64.dtype}"

    coords_cpu = create_coordinate_3d(2, 4, 4, device=torch.device("cpu"))
    assert coords_cpu.device.type == "cpu"

    print("test_3d_dtype_and_device passed")


# ─────────────────────────────────────────────
# Test 8 — Batch dim is always 1 (3D)
# ─────────────────────────────────────────────
def test_3d_batch_dim_is_one():
    coords = create_coordinate_3d(3, 5, 7)
    assert coords.shape[0] == 1, \
        f"Expected batch dim 1, got {coords.shape[0]}"

    print("test_3d_batch_dim_is_one passed")


# ─────────────────────────────────────────────
# Test 9 — No NaN or Inf in output (3D)
# ─────────────────────────────────────────────
def test_3d_no_nan_or_inf():
    for t, h, w in [(1, 1, 1), (4, 8, 8), (8, 14, 14)]:
        coords = create_coordinate_3d(t, h, w)
        assert torch.isfinite(coords).all(), \
            f"Non-finite values found for t={t}, h={h}, w={w}"

    print("test_3d_no_nan_or_inf passed")


# ─────────────────────────────────────────────────────
# Test 10 — Values are unique per position (3D)
# ─────────────────────────────────────────────────────
def test_3d_unique_coordinates():
    """
    Each (z, x, y) triplet must be unique — no two spatial positions
    should share the same coordinate vector.
    """
    t, h, w = 3, 4, 5
    coords = create_coordinate_3d(t, h, w).squeeze(0)  # (t*h*w, 3)

    n = coords.shape[0]
    unique_rows = torch.unique(coords, dim=0)
    assert unique_rows.shape[0] == n, \
        f"Expected {n} unique coordinates, got {unique_rows.shape[0]}"

    print("test_3d_unique_coordinates passed")


if __name__ == "__main__":
    test_2d_output_shape()
    test_3d_output_shape()
    test_3d_coordinate_range_default()
    test_3d_coordinate_range_custom()
    test_3d_axis_order()
    test_3d_single_frame_matches_2d()
    test_3d_dtype_and_device()
    test_3d_batch_dim_is_one()
    test_3d_no_nan_or_inf()
    test_3d_unique_coordinates()