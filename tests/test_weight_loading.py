"""
tests/test_weight_loading.py
----------------------------
Task 2.2 — Single-frame equivalence after loading 2D weights into AnyUp3D.

Every test follows the same invariant:
    unsqueeze T=1 into the 3D model input → squeeze output back → allclose to 2D.

Fixtures
--------
Two ways to supply checkpoints (in order of priority):
  1. Pytest CLI options:
       pytest tests/test_weight_loading.py \
           --ckpt2d anyup_multi_backbone.pth \
           --ckpt3d anyup3d_init.pth
  2. Environment variables:  ANYUP_CKPT_2D  /  ANYUP_CKPT_3D

If neither checkpoint is reachable the entire module is skipped cleanly.

NOTE: If any fixture or forward call raises because of an API mismatch with
your updated 3D implementation, paste the relevant function's updated code
and we'll adjust the tensor manipulation here accordingly.
"""

import pytest
import torch
import torch.nn.functional as F


def _load_sd(path: str) -> dict:
    """Load a state dict from a .pth file, unwrapping common wrapper keys."""
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(raw, dict):
        for key in ("model", "state_dict", "weights", "params"):
            if key in raw and isinstance(raw[key], dict):
                return raw[key]
        # bare state dict
        if all(isinstance(v, torch.Tensor) for v in raw.values()):
            return raw
    raise ValueError(f"Cannot parse checkpoint at {path}")


@pytest.fixture(scope="session")
def model_2d(ckpt_paths):
    """
    The original 2D AnyUp model with 2D weights loaded.

    ⚠ If your repo no longer has a separate 2D model class, paste the class
    here and we will adjust.  The fixture falls back to torch.hub if
    anyup.model_2d is not importable.
    """
    path_2d, _ = ckpt_paths
    try:
        from anyup.model_2d import AnyUp as AnyUp2D       # frozen 2D class
        model = AnyUp2D()
        model.load_state_dict(_load_sd(path_2d), strict=False)
    except ImportError:
        model = torch.hub.load(
            "wimmerth/anyup", "anyup",
            map_location="cpu", source="github"
        )
    model.eval()
    return model


@pytest.fixture(scope="session")
def model_3d(ckpt_paths):
    """AnyUp3D with the adapted checkpoint from load_2d_weights.py."""
    _, path_3d = ckpt_paths
    from anyup.model import AnyUp as AnyUp3D
    model = AnyUp3D()                                      # uses your default args
    model.load_state_dict(_load_sd(path_3d), strict=True)
    model.eval()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

ATOL = 1e-5   # tolerance for all equivalence checks


def _max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_rope_single_frame(model_2d, model_3d):
    """
    RoPE3D with T=1 and zeroed z-row must match RoPE2D output exactly.
    Mirrors test_single_frame_matches_2d from tests/test_rope3d.py.
    """
    from anyup.utils.img import create_coordinate
    from anyup.modules import create_coordinates_3d

    H, W, B, dim = 7, 9, 2, model_3d.qk_dim

    x = torch.randn(B, H * W, dim)
    coords_2d = create_coordinate(H, W).expand(B, -1, -1)         # (B, H*W, 2)
    coords_3d = create_coordinates_3d(1, H, W).expand(B, -1, -1)  # (B, H*W, 3)

    with torch.no_grad():
        out_2d = model_2d.rope(x, coords_2d)
        out_3d = model_3d.rope(x, coords_3d)

    diff = _max_diff(out_2d, out_3d)
    assert torch.allclose(out_2d, out_3d, atol=ATOL), (
        f"RoPE single-frame equivalence failed — max diff: {diff:.2e}"
    )
    print(f"\ntest_rope_single_frame passed  (max_diff={diff:.2e})")


def test_image_encoder_single_frame(model_2d, model_3d):
    """
    image_encoder: 2D input (B,3,H,W) vs 3D input (B,3,1,H,W).squeeze(2).
    """
    B, H, W = 2, 64, 64
    img_2d = torch.randn(B, 3, H, W)
    img_3d = img_2d.unsqueeze(2)   # (B, 3, 1, H, W) ← ↑ T=1; adjust if your model uses a different dim order

    with torch.no_grad():
        enc_2d = model_2d.image_encoder(img_2d)    # (B, C, H', W')
        enc_3d = model_3d.image_encoder(img_3d)    # (B, C, 1, H', W')

    enc_3d_sq = enc_3d.squeeze(2)
    diff = _max_diff(enc_2d, enc_3d_sq)
    assert torch.allclose(enc_2d, enc_3d_sq, atol=ATOL), (
        f"image_encoder single-frame equivalence failed — max diff: {diff:.2e}"
    )
    print(f"\ntest_image_encoder_single_frame passed  (max_diff={diff:.2e})")


def test_key_encoder_single_frame(model_2d, model_3d):
    """key_encoder with T=1 must match 2D exactly."""
    B, C, H, W = 2, model_3d.qk_dim, 32, 32
    x_2d = torch.randn(B, C, H, W)
    x_3d = x_2d.unsqueeze(2)

    with torch.no_grad():
        out_2d = model_2d.key_encoder(x_2d)
        out_3d = model_3d.key_encoder(x_3d)

    out_3d_sq = out_3d.squeeze(2)
    diff = _max_diff(out_2d, out_3d_sq)
    assert torch.allclose(out_2d, out_3d_sq, atol=ATOL), (
        f"key_encoder single-frame equivalence failed — max diff: {diff:.2e}"
    )
    print(f"\ntest_key_encoder_single_frame passed  (max_diff={diff:.2e})")


def test_query_encoder_single_frame(model_2d, model_3d):
    """query_encoder with T=1 must match 2D exactly."""
    B, C, H, W = 2, model_3d.qk_dim, 32, 32
    x_2d = torch.randn(B, C, H, W)
    x_3d = x_2d.unsqueeze(2)

    with torch.no_grad():
        out_2d = model_2d.query_encoder(x_2d)
        out_3d = model_3d.query_encoder(x_3d)

    out_3d_sq = out_3d.squeeze(2)
    diff = _max_diff(out_2d, out_3d_sq)
    assert torch.allclose(out_2d, out_3d_sq, atol=ATOL), (
        f"query_encoder single-frame equivalence failed — max diff: {diff:.2e}"
    )
    print(f"\ntest_query_encoder_single_frame passed  (max_diff={diff:.2e})")


def test_key_features_encoder_single_frame(model_2d, model_3d):
    """
    key_features_encoder (LFU front-end) with T=1.
    Input is L2-normalized before being passed in — matching the forward() call.
    """
    B, feat_dim, H, W = 2, 768, 14, 14   # ← adjust feat_dim to match your test extractor

    feats_2d = torch.randn(B, feat_dim, H, W)
    feats_3d = feats_2d.unsqueeze(2)      # (B, feat_dim, 1, H, W)

    # Normalize along channel dim — same as in model.upsample()
    feats_2d_n = F.normalize(feats_2d, dim=1)
    feats_3d_n = F.normalize(feats_3d, dim=1)

    with torch.no_grad():
        out_2d = model_2d.key_features_encoder(feats_2d_n)
        out_3d = model_3d.key_features_encoder(feats_3d_n)

    out_3d_sq = out_3d.squeeze(2)
    diff = _max_diff(out_2d, out_3d_sq)
    assert torch.allclose(out_2d, out_3d_sq, atol=ATOL), (
        f"key_features_encoder single-frame equivalence failed — max diff: {diff:.2e}"
    )
    print(f"\ntest_key_features_encoder_single_frame passed  (max_diff={diff:.2e})")


def test_aggregation_single_frame(model_2d, model_3d):
    """aggregation (256→128 merge network) with T=1."""
    B, C, H, W = 2, 2 * model_3d.qk_dim, 14, 14   # C = 2×qk_dim after cat(K_img, K_feat)
    x_2d = torch.randn(B, C, H, W)
    x_3d = x_2d.unsqueeze(2)

    with torch.no_grad():
        out_2d = model_2d.aggregation(x_2d)
        out_3d = model_3d.aggregation(x_3d)

    out_3d_sq = out_3d.squeeze(2)
    diff = _max_diff(out_2d, out_3d_sq)
    assert torch.allclose(out_2d, out_3d_sq, atol=ATOL), (
        f"aggregation single-frame equivalence failed — max diff: {diff:.2e}"
    )
    print(f"\ntest_aggregation_single_frame passed  (max_diff={diff:.2e})")


def test_full_forward_single_frame(model_2d, model_3d):
    """
    End-to-end forward pass with T=1.

    If your updated forward() has a different signature (e.g. output_size is
    passed differently, or the 3D model expects a tuple vs a 2-tuple), paste
    the updated forward() signature and we will adjust here.
    """
    B           = 1
    H, W        = 224, 224   # ↓ reduce to save memory if needed
    feat_dim    = 768         # ↓ must match your feature extractor's output dim
    h, w        = 14, 14     # ↓ low-res feature spatial size
    out_h, out_w = 28, 28    # ↓ target upsample size (affects attention sequence length)

    img_2d  = torch.randn(B, 3, H, W)
    img_3d  = img_2d.unsqueeze(2)              # (B, 3, 1, H, W)
    feat_2d = torch.randn(B, feat_dim, h, w)
    feat_3d = feat_2d.unsqueeze(2)             # (B, feat_dim, 1, h, w)

    with torch.no_grad():
        out_2d = model_2d(img_2d,  feat_2d, output_size=(out_h, out_w))
        out_3d = model_3d(img_3d,  feat_3d, output_size=(out_h, out_w))

    # out_3d expected shape: (B, qk_dim, 1, out_h, out_w)
    out_3d_sq = out_3d.squeeze(2)
    diff = _max_diff(out_2d, out_3d_sq)
    assert torch.allclose(out_2d, out_3d_sq, atol=1e-4), (   # slightly looser: accumulates over full graph
        f"full_forward single-frame equivalence failed — max diff: {diff:.2e}"
    )
    print(f"\ntest_full_forward_single_frame passed  (max_diff={diff:.2e})")