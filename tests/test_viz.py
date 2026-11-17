"""
T10 pytest: visual helpers (anyup/viz.py).

CPU only, synthetic data. Run on Colab:

    python -m pytest tests/test_viz.py -v
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch

from anyup.metrics import temporal_cosine
from anyup.viz import (FeaturePCA, colorize, davis_palette, denorm_guide, flicker_map, grid, label_strip,
                       overlay, write_gif, write_mp4)

SEED = 0
T, C, H, W = 5, 16, 12, 10   # ↓ tiny synthetic clip


def rand_feats(seed=SEED, shape=(T, C, H, W)):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


# ─── flicker_map ────────────────────────────────────────────────────────────

def test_flicker_identical_frames_zero():
    f = rand_feats()[:1].repeat(T, 1, 1, 1)
    fl = flicker_map(f)
    assert fl.shape == (T - 1, H, W)
    assert np.abs(fl).max() < 1e-12


def test_flicker_flip_is_two():
    f0 = rand_feats()[0]
    fl = flicker_map(torch.stack([f0, -f0]))
    assert np.allclose(fl, 2.0, atol=1e-12)


def test_flicker_mean_equals_temporal_cosine():
    f = rand_feats()
    fl = flicker_map(f)
    ref = temporal_cosine(f)["per_pair"]
    assert np.abs((1 - fl.mean(axis=(1, 2))) - np.array(ref)).max() < 1e-9


def test_flicker_errors():
    with pytest.raises(ValueError):
        flicker_map(rand_feats()[:1])


# ─── FeaturePCA ─────────────────────────────────────────────────────────────

def test_pca_shape_dtype_and_static_frames_identical():
    f = rand_feats()
    pca = FeaturePCA().fit({"a": FeaturePCA.sample_tokens(f, 200)})
    rgb = pca.rgb(f, "a")
    assert rgb.shape == (T, H, W, 3) and rgb.dtype == np.uint8
    static = f[:1].repeat(T, 1, 1, 1)
    rs = pca.rgb(static, "a")
    assert all(np.array_equal(rs[0], rs[t]) for t in range(T))   # fixed scale: no per-frame normalization


def test_pca_deterministic_and_per_key_scale():
    f1, f2 = rand_feats(1), 3 * rand_feats(2)
    s = {"a": FeaturePCA.sample_tokens(f1, 300, seed=0), "b": FeaturePCA.sample_tokens(f2, 300, seed=0)}
    p1, p2 = FeaturePCA().fit(s), FeaturePCA().fit(s)
    assert np.array_equal(p1.rgb(f1, "a"), p2.rgb(f1, "a"))
    assert not torch.equal(p1.scale["a"][1], p1.scale["b"][1])     # different conditions keep their own scale


def test_pca_scale_shared_across_time():
    # a frame-only change in magnitude must change the colours (no per-frame rescaling)
    f = rand_feats()
    pca = FeaturePCA().fit({"a": FeaturePCA.sample_tokens(f, 300)})
    g = f.clone()
    g[1] = g[0] * 0.2
    r = pca.rgb(g, "a")
    assert not np.array_equal(r[0], r[1])


# ─── images / video ─────────────────────────────────────────────────────────

def test_overlay_background_unchanged():
    rgb = np.random.default_rng(SEED).integers(0, 255, (H, W, 3), dtype=np.uint8)
    mask = np.zeros((H, W), np.uint8)
    mask[3:7, 3:7] = 1
    out = overlay(rgb, mask, contour=False)
    assert np.array_equal(out[mask == 0], rgb[mask == 0])
    assert not np.array_equal(out[mask == 1], rgb[mask == 1])
    assert tuple(davis_palette()[1]) == (128, 0, 0)


def test_layout_helpers():
    img = np.zeros((20, 30, 3), np.uint8)
    assert label_strip(img, "x").shape == (42, 30, 3)
    assert grid([[img, img], [img]], pad=4).shape == (44, 64, 3)
    assert colorize(np.zeros((2, 5, 6)), 0, 1).shape == (2, 5, 6, 3)
    g = torch.zeros(2, 3, 5, 6)
    assert denorm_guide(g, (0.5, 0.5, 0.5), (0.2, 0.2, 0.2)).max() == 128


def test_write_gif_and_mp4(tmp_path):
    frames = [np.full((31, 41, 3), 20 * i, np.uint8) for i in range(6)]   # odd size → padded for H.264
    gif = write_gif(frames, tmp_path / "a.gif", fps=4)
    from PIL import Image
    im = Image.open(gif)
    assert im.n_frames == 6
    pytest.importorskip("imageio_ffmpeg")
    import imageio.v2 as imageio
    mp4 = write_mp4(frames, tmp_path / "a.mp4", fps=4)
    assert mp4.stat().st_size > 0
    r = imageio.get_reader(mp4)
    assert r.count_frames() == 6
    assert r.get_meta_data()["size"] == (42, 32)
