"""T04_training: AnyUp.forward split into encode_image (image half) + decode (feature half).

Regression check (CLAUDE.md): the new forward must give exactly the same outputs as the old one,
and a training step that shares one image encoding between two passes must give the same parameter
gradients as two separate forward calls (up to float rounding, Q4: rtol=1e-5, atol=1e-7).

`_reference_forward` is a verbatim copy of AnyUp.forward + AnyUp.upsample before T04 (main 9e43ba2),
with `self` -> `model`. CPU only, small random model, fixed seed.
"""
import os
import sys
import warnings

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from anyup.model import AnyUp
from anyup.modules import create_coordinates_3d

RTOL, ATOL = 1e-5, 1e-7          # Q4: gradient tolerance (outputs must be exactly equal)
SEED = 0

# (image frames T, feature tokens T_tok): training uses T = 2 * T_tok (VideoMAE tubelet 2);
# (1, 1) is the single-image case (Exp 1).
FRAME_CASES = [(1, 1), (2, 1), (4, 2), (8, 4)]


# ── old code, verbatim (main 9e43ba2, anyup/model.py:82-123) ────────────────────────────────────
def _reference_upsample(model, enc_img, feats, out_size, vis_attn=False, q_chunk_size=None):
    b, c, t, h, w = feats.shape

    # Q
    q = F.adaptive_avg_pool3d(
        model.query_encoder(enc_img),
        output_size=(t, *out_size),
    )

    # K
    k = F.adaptive_avg_pool3d(
        model.key_encoder(enc_img),
        output_size=(t, h, w),
    )
    k = torch.cat([k, model.key_features_encoder(F.normalize(feats, dim=1))], dim=1)
    k = model.aggregation(k)

    # V
    v = feats

    if vis_attn:
        warnings.warn("Visualisation of attention maps is not supported for 3D cross-attention.")
        vis_attn = False

    return model.cross_decode(q, k, v, store_attn=vis_attn, q_chunk_size=q_chunk_size)


def _reference_forward(model, image, features, output_size=None, vis_attn=False, q_chunk_size=None):
    output_size = output_size if output_size is not None else image.shape[-2:]

    enc = model.image_encoder(image)               # (B, C, T, H, W)
    t, h, w = enc.shape[-3], enc.shape[-2], enc.shape[-1]

    coords = create_coordinates_3d(t, h, w, device=enc.device, dtype=enc.dtype)

    # Flatten spatiotemporal tokens: (B, C, T, H, W) -> (B, T*H*W, C)
    enc = enc.permute(0, 2, 3, 4, 1).reshape(enc.shape[0], -1, enc.shape[1])
    enc = model.rope(enc, coords)

    # Restore 5D layout: (B, T*H*W, C) -> (B, C, T, H, W)
    enc = enc.view(enc.shape[0], t, h, w, enc.shape[-1]).permute(0, 4, 1, 2, 3)

    return _reference_upsample(model, enc, features, output_size, vis_attn=vis_attn, q_chunk_size=q_chunk_size)
# ── end of old code ─────────────────────────────────────────────────────────────────────────────


def _model():
    """Small AnyUp with the real kernel settings (kernel_size_lfu=5, t_k=3, window_t=3)."""
    torch.manual_seed(SEED)
    return AnyUp(qk_dim=32, num_heads=4, kernel_size_lfu=5, t_k=3, window_t=3, window_ratio=0.1)


def _inputs(T, T_tok, B=1, C=24, img=32, low=4, seed=SEED + 1):
    """image (B, 3, T, img, img); features (B, C, T_tok, low, low). C=24 stands in for 768."""
    g = torch.Generator().manual_seed(seed)
    image = torch.randn(B, 3, T, img, img, generator=g)
    feats = torch.randn(B, C, T_tok, low, low, generator=g)
    return image, feats


@pytest.mark.parametrize("T,T_tok", FRAME_CASES)
def test_forward_unchanged(T, T_tok):
    model = _model().eval()
    image, feats = _inputs(T, T_tok)
    with torch.no_grad():
        ref = _reference_forward(model, image, feats, output_size=(8, 8))
        new = model(image, feats, output_size=(8, 8))
    assert new.shape == ref.shape == (1, 24, T_tok, 8, 8)
    assert torch.equal(new, ref), f"max_abs {(new - ref).abs().max().item():.3e}"


def test_forward_unchanged_batch_default_size_and_chunks():
    """Batch 2, output_size=None (→ image size), and q_chunk_size: still exactly equal."""
    model = _model().eval()
    image, feats = _inputs(4, 2, B=2, img=16)
    with torch.no_grad():
        for kw in ({}, {"q_chunk_size": 7}):
            ref = _reference_forward(model, image, feats, **kw)
            new = model(image, feats, **kw)
            assert new.shape == (2, 24, 2, 16, 16)
            assert torch.equal(new, ref), (kw, (new - ref).abs().max().item())


def test_vis_attn_warns_and_matches():
    model = _model().eval()
    image, feats = _inputs(2, 1)
    with torch.no_grad():
        ref = _reference_forward(model, image, feats, output_size=(8, 8))
        with pytest.warns(UserWarning, match="not supported for 3D"):
            new = model(image, feats, output_size=(8, 8), vis_attn=True)
    assert torch.equal(new, ref)


@pytest.mark.parametrize("T,T_tok", FRAME_CASES)
def test_encode_decode_equals_forward(T, T_tok):
    model = _model().eval()
    image, feats = _inputs(T, T_tok)
    with torch.no_grad():
        enc = model.encode_image(image, feats.shape, (8, 8))
        assert torch.equal(model.decode(enc, feats), model(image, feats, output_size=(8, 8)))


def test_decode_rejects_mismatched_features():
    """Reuse is only valid for features on the same grid (Finding B); a wrong grid must fail clearly."""
    model = _model().eval()
    image, feats = _inputs(4, 2)
    with torch.no_grad():
        enc = model.encode_image(image, feats.shape, (8, 8))
        with pytest.raises(ValueError, match="encode_image"):
            model.decode(enc, feats[:, :, :1])                # T_tok 2 → 1


@pytest.mark.parametrize("T,T_tok", [(4, 2), (8, 4)])
def test_shared_encoding_gradients_match_two_forwards(T, T_tok):
    """Cell 13 shape of a step: pass 1 (clean features) + pass 2 (noisy features, target = pass 1 detached).
    Old: two forward calls. New: one encode_image, two decode calls. Parameter gradients must agree (Q4)."""
    image, feats = _inputs(T, T_tok)
    feats_aug = feats + 0.05 * torch.randn(feats.shape, generator=torch.Generator().manual_seed(SEED + 2))

    def loss_fn(pred, pred_aug):
        return (pred ** 2).mean() + ((pred_aug - pred.detach()) ** 2).mean()

    old = _model().train()
    p1 = _reference_forward(old, image, feats, output_size=(8, 8))
    p2 = _reference_forward(old, image, feats_aug, output_size=(8, 8))
    loss_old = loss_fn(p1, p2)
    loss_old.backward()

    new = _model().train()
    enc = new.encode_image(image, feats.shape, (8, 8))
    q1 = new.decode(enc, feats)
    q2 = new.decode(enc, feats_aug)
    loss_new = loss_fn(q1, q2)
    loss_new.backward()

    assert torch.equal(q1, p1) and torch.equal(q2, p2)       # forward values: same ops, same order
    assert torch.equal(loss_new, loss_old)
    n_checked = 0
    for (name, a), (_, b) in zip(old.named_parameters(), new.named_parameters()):
        if a.grad is None:
            assert b.grad is None, name
            continue
        torch.testing.assert_close(b.grad, a.grad, rtol=RTOL, atol=ATOL, msg=lambda m: f"{name}: {m}")
        n_checked += 1
    assert n_checked > 0
