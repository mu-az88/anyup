"""
T02: video reader + chunked backbone runner (anyup/video.py).

Uses fake adapters (no downloads, CPU). Frame k of the synthetic video has every pixel
equal to k / 1000, so each output's feature value identifies which input frame produced it.
"""

import sys

import pytest
import torch
import torch.nn as nn
from PIL import Image

from anyup.backbones import BackboneOutput
from anyup.video import (
    DEFAULT_CHUNK_SIZE, FrameFolderSource, TensorSource, chunk_boundaries,
    chunk_starts, iter_backbone_chunks, open_video,
)


# ---------------------------------------------------------------------------
# Fake adapters
# ---------------------------------------------------------------------------

class FakeAdapter(nn.Module):
    """Variable-T backbone. Output k = input frame frame_index[k]; feature value = k_global*1000 → frame id."""

    name = "fake"
    temporal_stride = 1

    def __init__(self, temporal_stride=1):
        super().__init__()
        self.temporal_stride = temporal_stride
        self.calls = []   # T of every call

    def forward(self, frames):
        T = frames.shape[0]
        self.calls.append(T)
        frame_index = list(range(0, T, self.temporal_stride))
        feats = frames[frame_index, :1, :2, :2] * 1000             # (T', 1, 2, 2), value = frame id
        return BackboneOutput(features=feats, guide=frames[frame_index],
                              meta=dict(name=self.name, frame_index=frame_index, t_in=T, t_out=len(frame_index)))


class FakeFixed16(FakeAdapter):
    """VideoMAE-like: exactly 16 frames, tubelet 2."""

    NUM_FRAMES = 16
    name = "fake16"

    def __init__(self):
        super().__init__(temporal_stride=2)

    def forward(self, frames):
        assert frames.shape[0] == 16, frames.shape
        return super().forward(frames)


def _video(n, h=4, w=6):
    return torch.arange(n, dtype=torch.float32).div(1000).view(n, 1, 1, 1).expand(n, 3, h, w).clone()   # (n, 3, h, w)


def _ids(out):
    return out.features[:, 0, 0, 0].round().long().tolist()   # producing input frame id per output


def _run(bb, n, **kw):
    return list(iter_backbone_chunks(bb, _video(n), **kw))


# ---------------------------------------------------------------------------
# Coverage: every frame exactly once, global indices correct
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 15, 16, 17, 33, 48])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("overlap", [0, 2, 4, 8])
def test_coverage_variable_t(n, stride, overlap):
    bb = FakeAdapter(temporal_stride=stride)
    outs = _run(bb, n, overlap=overlap)
    gidx = [f for o in outs for f in o.meta["frame_index"]]
    assert gidx == list(range(0, n, stride))                               # every (tubelet-first) frame exactly once, in order
    for o in outs:
        assert _ids(o) == o.meta["frame_index"]                            # outputs come from the right frames
        assert o.guide.shape[0] == o.features.shape[0] == o.meta["t_out"]
    assert all(o.meta["n_pad"] == 0 for o in outs)
    assert [o.meta["n_ctx"] for o in outs] == [0] + [overlap] * (len(outs) - 1)
    assert outs[-1].meta["is_last"] and not any(o.meta["is_last"] for o in outs[:-1])


@pytest.mark.parametrize("n", [16, 17, 33, 48, 82])
@pytest.mark.parametrize("overlap", [0, 2, 4, 8])
def test_coverage_fixed16(n, overlap):
    bb = FakeFixed16()
    outs = _run(bb, n, overlap=overlap)
    gidx = [f for o in outs for f in o.meta["frame_index"]]
    assert gidx == list(range(0, n, 2))                                    # tubelet-aligned since overlap is even
    for o in outs:
        assert _ids(o) == o.meta["frame_index"]
    assert all(t == 16 for t in bb.calls)


def test_earlier_chunk_wins():
    bb = FakeAdapter()
    outs = _run(bb, 20, chunk_size=8, overlap=4)                           # starts 0, 4, 8, 12
    assert [o.meta["chunk_start"] for o in outs] == [0, 4, 8, 12]
    assert [o.meta["frame_index"] for o in outs] == [list(range(0, 8)), [8, 9, 10, 11],
                                                     [12, 13, 14, 15], [16, 17, 18, 19]]
    assert bb.calls == [8, 8, 8, 8]


def test_overlap_zero_equals_plain_chunking():
    a = _run(FakeAdapter(), 37, chunk_size=10)
    b = _run(FakeAdapter(), 37, chunk_size=10, overlap=0)
    for x, y in zip(a, b):
        assert torch.equal(x.features, y.features) and x.meta["frame_index"] == y.meta["frame_index"]
    assert [o.meta["frame_index"][0] for o in a] == [0, 10, 20, 30]


# ---------------------------------------------------------------------------
# Fixed-T padding and errors
# ---------------------------------------------------------------------------

def test_fixed16_last_chunk_padding():
    bb = FakeFixed16()
    outs = _run(bb, 33)                                                    # 16 + 16 + (1 real + 15 pad)
    assert [o.meta["n_valid"] for o in outs] == [16, 16, 1]
    assert [o.meta["n_pad"] for o in outs] == [0, 0, 15]
    assert outs[-1].meta["t_out"] == 1 and outs[-1].meta["frame_index"] == [32]
    assert [o.meta["t_out"] for o in outs] == [8, 8, 1]


def test_fixed16_odd_tail_keeps_tubelet_with_real_first_frame():
    outs = _run(FakeFixed16(), 31)                                         # last chunk: 15 real + 1 pad
    assert outs[-1].meta["n_pad"] == 1
    assert outs[-1].meta["frame_index"] == list(range(16, 31, 2))          # tubelet (30, pad) kept, guide = frame 30


def test_fixed16_errors():
    with pytest.raises(ValueError, match="at least 16"):
        _run(FakeFixed16(), 15)
    with pytest.raises(ValueError, match="chunk_size=16"):
        _run(FakeFixed16(), 32, chunk_size=8)
    with pytest.raises(ValueError, match="multiple of temporal_stride"):
        _run(FakeFixed16(), 32, overlap=3)


@pytest.mark.parametrize("overlap", [-1, 16, 20])
def test_overlap_range_errors(overlap):
    with pytest.raises(ValueError, match="overlap must satisfy"):
        _run(FakeAdapter(), 40, chunk_size=16, overlap=overlap)


def test_default_chunk_size_and_short_video():
    bb = FakeAdapter()
    outs = _run(bb, 40)
    assert bb.calls == [DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_SIZE, 40 - 2 * DEFAULT_CHUNK_SIZE]
    bb = FakeAdapter()
    outs = _run(bb, 5)                                                     # shorter than one chunk: OK, no padding
    assert bb.calls == [5] and outs[0].meta["frame_index"] == list(range(5))


def test_chunk_starts():
    assert chunk_starts(16, 16, 0) == [0]
    assert chunk_starts(17, 16, 0) == [0, 16]
    assert chunk_starts(20, 8, 4) == [0, 4, 8, 12]
    assert chunk_starts(1, 16, 0) == [0]


def test_chunk_boundaries():
    outs = _run(FakeFixed16(), 33)                                         # t_out 8, 8, 1
    assert chunk_boundaries([o.meta for o in outs]) == [8, 16]
    assert chunk_boundaries([o.meta for o in _run(FakeAdapter(), 10)]) == []


def test_chunk_meta():
    outs = _run(FakeAdapter(), 20, chunk_size=8)
    m = outs[1].meta
    assert m["chunk_id"] == 1 and m["chunk_start"] == 8 and m["n_chunks"] == 3
    assert m["n_frames_total"] == 20 and m["native_size"] == (4, 6) and m["source"] == "tensor"
    assert m["local_frame_index"] == list(range(8)) and m["fps"] is None


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def test_tensor_source_uint8_and_float():
    u8 = torch.randint(0, 256, (3, 3, 5, 7), dtype=torch.uint8)
    assert torch.equal(TensorSource(u8).get([0, 2]), u8[[0, 2]].float() / 255)
    with pytest.raises(ValueError):
        TensorSource(torch.zeros(3, 5, 7))


def test_frame_folder_matches_tensor(tmp_path):
    u8 = torch.randint(0, 256, (5, 3, 6, 9), dtype=torch.uint8)
    for k in range(5):
        Image.fromarray(u8[k].permute(1, 2, 0).numpy()).save(tmp_path / f"{k:05d}.png")   # lossless
    src = open_video(tmp_path, ext=".png")
    assert isinstance(src, FrameFolderSource)
    assert len(src) == 5 and src.native_size == (6, 9) and src.fps is None
    assert torch.equal(src.get(range(5)), TensorSource(u8).get(range(5)))  # max abs diff = 0


def test_frame_folder_sorted_order(tmp_path):
    for k in [3, 0, 2, 1]:
        Image.new("RGB", (4, 4), (k * 10, 0, 0)).save(tmp_path / f"{k:05d}.png")
    src = FrameFolderSource(tmp_path, ext=".png")
    assert (src.get(range(4))[:, 0, 0, 0] * 255).round().tolist() == [0, 10, 20, 30]


def test_video_file_without_decord_raises(tmp_path, monkeypatch):
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"")
    monkeypatch.setitem(sys.modules, "decord", None)                       # makes `import decord` raise ImportError
    with pytest.raises(ImportError, match="pip install eva-decord"):
        open_video(f)


def test_open_video_errors(tmp_path):
    with pytest.raises(ValueError, match="No"):
        open_video(tmp_path)                                               # empty dir
    with pytest.raises(FileNotFoundError):
        open_video(tmp_path / "missing.mp4")
