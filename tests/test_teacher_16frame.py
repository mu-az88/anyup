import random
import pytest
import torch
from transformers import VideoMAEConfig, VideoMAEModel
from anyup.data.training.teacher import (assemble_clip, extract_teacher_features, sample_clip_indices,
                                         sample_student_window, take_window)


@pytest.fixture(scope="module")
def teacher():
    cfg = VideoMAEConfig(num_frames=16, tubelet_size=2, hidden_size=32, num_hidden_layers=1,
                         num_attention_heads=2, intermediate_size=64)
    return VideoMAEModel(cfg).eval()


@pytest.mark.parametrize("size", [224, 112])
def test_shape(teacher, size):
    assert extract_teacher_features(teacher, torch.randn(1, 16, 3, size, size)).shape == (1, 32, 8, 14, 14)


@pytest.mark.parametrize("t", [2, 4, 8])
def test_rejects_short_clips(teacher, t):
    with pytest.raises(ValueError):
        extract_teacher_features(teacher, torch.randn(1, t, 3, 224, 224))


def test_position_table_unchanged(teacher):
    before = teacher.embeddings.position_embeddings.clone()
    extract_teacher_features(teacher, torch.randn(1, 16, 3, 224, 224))
    assert torch.equal(teacher.embeddings.position_embeddings, before)


@pytest.mark.parametrize("total,stride", [(300, 4), (61, 4), (40, 2), (40, 4)])
def test_clip_indices_stride(total, stride):
    rng = random.Random(0)
    for _ in range(100):
        idx = sample_clip_indices(total, 16, stride, rng)
        steps = {b - a for a, b in zip(idx, idx[1:])}
        assert len(idx) == 16 and len(steps) == 1 and 1 <= steps.pop() <= stride and idx[-1] < total


def test_clip_indices_short_clip_pads():
    idx = sample_clip_indices(10, 16, 4, random.Random(0))
    assert len(idx) == 16 and idx == sorted(idx) and idx[-1] == 9    # last frame repeated


@pytest.mark.parametrize("t", [2, 4, 8])
def test_window_aligned_and_in_bounds(t):
    rng = random.Random(0)
    for _ in range(200):
        f0, tok0, n = sample_student_window(t, 16, 2, rng)
        assert f0 == 2 * tok0 and n == t // 2 and tok0 + n <= 8


@pytest.mark.parametrize("t", [2, 4, 8])
def test_frames_match_tokens(t):
    frames = torch.arange(16.).view(1, 16, 1, 1, 1)
    feats = torch.arange(8.).view(1, 1, 8, 1, 1)
    f0, tok0, n = sample_student_window(t, 16, 2, random.Random(1))
    fw, tw = take_window(frames, feats, f0, tok0, n, 2)
    assert torch.equal(fw.flatten()[::2] // 2, tw.flatten())   # tubelet k covers frames 2k, 2k+1


def test_assemble_clip_missing_tail():
    idxs = list(range(0, 64, 4))                           # header claimed 64 frames → 0, 4, …, 60
    got = {i: torch.tensor(float(i)) for i in idxs if i <= 30}   # video actually ended at frame 30
    clip = [c.item() for c in assemble_clip(got, idxs)]
    assert clip == idxs[:8] + [28.0] * 8                   # no KeyError, no zeros: last real frame repeated


def test_assemble_clip_keeps_order_and_repeats():
    idxs = sample_clip_indices(10, 16, 4, random.Random(0))     # short clip → repeated indices
    got = {i: torch.tensor(float(i)) for i in set(idxs)}
    assert [c.item() for c in assemble_clip(got, idxs)] == idxs


def test_assemble_clip_nothing_decoded():
    assert assemble_clip({}, list(range(16))) == []        # notebook's padding loop fills zeros
