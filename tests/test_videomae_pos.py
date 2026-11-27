"""T12: VideoMAE position-table interpolation (anyup.backbones.interpolate_videomae_pos). CPU, no HF download."""
import pytest
import torch

from anyup.backbones import interpolate_videomae_pos


def _table(t=8, h=14, w=14, C=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, t * h * w, C, generator=g)


def test_same_size_is_identity():
    tab = _table()
    assert interpolate_videomae_pos(tab, 8, (14, 14), (14, 14)) is tab


def test_shape_and_dtype():
    out = interpolate_videomae_pos(_table(), 8, (14, 14), (28, 28))
    assert out.shape == (1, 8 * 28 * 28, 16) and out.dtype == torch.float32


def test_time_not_mixed():
    # each feature frame's table is resized on its own: changing frame 3 changes only frame 3 of the output
    a = _table()
    b = a.clone().reshape(8, 14 * 14, 16)
    b[3] += 1.0
    b = b.reshape(1, -1, 16)
    oa = interpolate_videomae_pos(a, 8, (14, 14), (28, 28)).reshape(8, 28 * 28, 16)
    ob = interpolate_videomae_pos(b, 8, (14, 14), (28, 28)).reshape(8, 28 * 28, 16)
    diff = (oa - ob).abs().amax(dim=(1, 2))
    assert diff[3] > 0.5 and torch.all(diff[torch.arange(8) != 3] == 0)


def test_spatially_constant_stays_constant():
    # a table that is constant in (h, w) per frame stays that constant after resizing
    v = torch.randn(8, 1, 1, 16).expand(8, 14, 14, 16).reshape(1, -1, 16)
    out = interpolate_videomae_pos(v, 8, (14, 14), (28, 28)).reshape(8, 28, 28, 16)
    assert torch.allclose(out, out[:, :1, :1].expand_as(out), atol=1e-5)


def test_layout_is_t_h_w():
    # a table that varies only along w keeps that: output varies only along w, and is monotone like the input
    w_ramp = torch.linspace(0, 1, 14)[None, None, :, None].expand(8, 14, 14, 1).reshape(1, -1, 1)
    out = interpolate_videomae_pos(w_ramp, 8, (14, 14), (28, 28)).reshape(8, 28, 28)
    assert torch.allclose(out, out[:, :1, :].expand_as(out), atol=1e-6)        # constant along h
    assert torch.all(out[0, 0, 1:] >= out[0, 0, :-1] - 1e-6)                   # monotone along w


def test_wrong_token_count_raises():
    with pytest.raises(ValueError):
        interpolate_videomae_pos(_table(h=14, w=14), 8, (16, 16), (28, 28))
