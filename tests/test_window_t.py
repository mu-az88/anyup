"""T02_training: when does `window_t` (temporal attention window, in tubelets) change AnyUp's output?

Finding C: the mask allows |t_q - t_k| <= window_t, so with <= 4 tubelets (training's max, T=8 frames)
window_t = 3, 4 and None give the same full mask, and outputs are equal. With 8 tubelets they differ.
`window_t` is not in the state dict, so the same weights load into every variant.
CPU only, small random model, fixed seed.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from anyup.model import AnyUp

SEED = 0
WINDOWS = [3, 4, None]


def _model(window_t):
    return AnyUp(qk_dim=32, num_heads=4, kernel_size_lfu=5, t_k=3, window_t=window_t, window_ratio=0.1)


def _models_same_weights():
    torch.manual_seed(SEED)
    ref = _model(WINDOWS[0]).eval()
    models = {WINDOWS[0]: ref}
    for w in WINDOWS[1:]:
        m = _model(w).eval()
        m.load_state_dict(ref.state_dict(), strict=True)   # window_t isn't a weight: same state dict fits
        models[w] = m
    return models


def _inputs(T_tok, B=1, C=24, img=32, low=4):
    g = torch.Generator().manual_seed(SEED + 1)
    image = torch.randn(B, 3, 2 * T_tok, img, img, generator=g)   # (B, 3, T, img, img); T = 2·T_tok (tubelet 2)
    feats = torch.randn(B, C, T_tok, low, low, generator=g)       # (B, C, T_tok, low, low)
    return image, feats


@pytest.mark.parametrize("window_t", WINDOWS)
def test_constructor_stores_window_t(window_t):
    m = _model(window_t)
    assert m.window_t == window_t
    assert m.cross_decode.window_t == window_t   # the value the mask actually uses (cell 7 prints it)


def test_equal_outputs_up_to_4_tubelets():
    models = _models_same_weights()
    for T_tok in (1, 2, 4):                      # training stages T=2/4/8 → 1/2/4 tubelets
        image, feats = _inputs(T_tok)
        with torch.no_grad():
            outs = {w: m(image, feats, output_size=(8, 8)) for w, m in models.items()}
        for w in WINDOWS[1:]:
            assert torch.equal(outs[w], outs[WINDOWS[0]]), f"T_tok={T_tok}: window_t={w} != {WINDOWS[0]}"


def test_8_tubelets_window_t_matters():
    models = _models_same_weights()
    image, feats = _inputs(8)                    # 16 frames, as the Exp 2/3 evaluation chunks
    with torch.no_grad():
        out3 = models[3](image, feats, output_size=(8, 8))
        outN = models[None](image, feats, output_size=(8, 8))
    assert out3.shape == outN.shape == (1, 24, 8, 8, 8)
    assert not torch.equal(out3, outN)
    assert (out3 - outN).abs().max().item() > 1e-4
