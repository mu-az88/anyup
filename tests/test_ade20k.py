"""
T06 pytest: shared ADE20K utilities (anyup/ade20k.py).

CPU only, tiny fake dataset on disk, no downloads. Run on Colab:

    python -m pytest tests/test_ade20k.py -v
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch
from PIL import Image

from anyup import ade20k
from anyup.backbones import BackboneOutput, IMAGENET_MEAN, IMAGENET_STD, normalize_frames, resize_frames

SEED = 0
N = 5              # ↓ fake images per split
S = 28             # ↓ cache image size; fake backbone grid = S // 14
C = 8              # ↓ fake feature channels


def make_root(tmp_path, n=N):
    rng = np.random.default_rng(SEED)
    root = tmp_path / "ADEChallengeData2016"
    for folder in ("training", "validation"):
        (root / "images" / folder).mkdir(parents=True)
        (root / "annotations" / folder).mkdir(parents=True)
        for i in range(n):
            stem = f"ADE_{folder}_{i:04d}"
            h, w = 30 + 2 * i, 40 - i            # images differ in size, as real ADE20K
            Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)).save(root / "images" / folder / f"{stem}.jpg")
            Image.fromarray(rng.integers(0, 151, (h, w), dtype=np.uint8)).save(root / "annotations" / folder / f"{stem}.png")
    return root


class FakeBackbone:
    """Deterministic per-image features: mean colour broadcast over channels and the grid."""
    name, timm_name, input_size = "fake", "fake_timm", (S, S)

    def __init__(self, fail_after=None):
        self.calls, self.fail_after = 0, fail_after

    def __call__(self, frames):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("interrupted")
        g = S // 14
        feats = frames.mean(dim=(1, 2, 3))[:, None, None, None].expand(-1, C, g, g).clone()
        return BackboneOutput(features=feats, guide=frames, meta={})


@pytest.fixture
def root(tmp_path):
    return make_root(tmp_path)


def test_list_pairs_by_stem(root):
    pairs = ade20k.list_pairs(root, "train")
    assert len(pairs) == N
    assert all(i.stem == m.stem for i, m in pairs)
    assert [i.stem for i, _ in pairs] == sorted(i.stem for i, _ in pairs)
    assert len(ade20k.list_pairs(root, "val")) == N


def test_list_pairs_mismatch_raises(root):
    next((root / "annotations" / "training").glob("*.png")).rename(root / "annotations" / "training" / "other.png")
    with pytest.raises(ValueError, match="pair up"):
        ade20k.list_pairs(root, "train")
    with pytest.raises(ValueError, match="split"):
        ade20k.list_pairs(root, "test")


def test_load_image_range_and_shape(root):
    img, _ = ade20k.list_pairs(root, "train")[1]
    x = ade20k.load_image(img, S)
    assert x.shape == (3, S, S) and x.dtype == torch.float32
    assert 0.0 <= x.min() and x.max() <= 1.0


def test_load_mask_keeps_ids(tmp_path):
    arr = np.zeros((40, 40), dtype=np.uint8)
    arr[:, :20], arr[:, 20:] = 7, 150
    Image.fromarray(arr).save(tmp_path / "m.png")
    m = ade20k.load_mask(tmp_path / "m.png", 20)
    assert m.shape == (20, 20) and m.dtype == torch.long
    assert set(m.unique().tolist()) == {7, 150}


def test_guide_matches_adapter_arithmetic(root):
    img, _ = ade20k.list_pairs(root, "train")[0]
    x = ade20k.load_image(img, S)
    ref = normalize_frames(resize_frames(x[None], S), IMAGENET_MEAN, IMAGENET_STD)[0]
    assert torch.equal(ade20k.guide_from_image(x), ref)


def test_cache_round_trip(root, tmp_path):
    pairs = ade20k.list_pairs(root, "train")
    d = tmp_path / "cache"
    ade20k.build_feature_cache(FakeBackbone(), pairs, d, "train", batch_size=2, num_workers=0)
    assert ade20k.cache_complete(d, "train")
    ds = ade20k.CachedFeats(d, "train", pairs, S, backbone_id="fake_timm")
    assert len(ds) == N
    for i in range(N):
        guide, feat, mask = ds[i]
        frame = ade20k.load_image(pairs[i][0], S)
        assert feat.shape == (C, S // 14, S // 14) and feat.dtype == torch.float32
        assert torch.allclose(feat, frame.mean().expand_as(feat), atol=1e-3)     # fp16 rounding
        assert torch.equal(guide, ade20k.guide_from_image(frame))
        assert torch.equal(mask, ade20k.load_mask(pairs[i][1], S))


def test_complete_cache_is_skipped(root, tmp_path):
    pairs = ade20k.list_pairs(root, "train")
    d = tmp_path / "cache"
    ade20k.build_feature_cache(FakeBackbone(), pairs, d, "train", batch_size=2, num_workers=0)
    bb = FakeBackbone()
    ade20k.build_feature_cache(bb, pairs, d, "train", batch_size=2, num_workers=0)
    assert bb.calls == 0


def test_interrupted_build_is_not_complete(root, tmp_path):
    pairs = ade20k.list_pairs(root, "train")
    d = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="interrupted"):
        ade20k.build_feature_cache(FakeBackbone(fail_after=1), pairs, d, "train", batch_size=2, num_workers=0)
    assert not ade20k.cache_complete(d, "train")
    with pytest.raises(ValueError, match="no complete"):
        ade20k.CachedFeats(d, "train", pairs, S)
    ade20k.build_feature_cache(FakeBackbone(), pairs, d, "train", batch_size=2, num_workers=0)   # rebuild works
    assert ade20k.cache_complete(d, "train")


def test_truncated_file_is_not_complete(root, tmp_path):
    pairs = ade20k.list_pairs(root, "train")
    d = tmp_path / "cache"
    ade20k.build_feature_cache(FakeBackbone(), pairs, d, "train", batch_size=2, num_workers=0)
    f = d / "train_feats.f16"
    f.write_bytes(f.read_bytes()[:-2])
    assert not ade20k.cache_complete(d, "train")


def test_meta_mismatch_raises(root, tmp_path):
    pairs = ade20k.list_pairs(root, "train")
    d = tmp_path / "cache"
    ade20k.build_feature_cache(FakeBackbone(), pairs, d, "train", batch_size=2, num_workers=0)
    with pytest.raises(ValueError, match="doesn't match"):
        ade20k.CachedFeats(d, "train", pairs[:-1], S)                  # other images
    with pytest.raises(ValueError, match="doesn't match"):
        ade20k.CachedFeats(d, "train", pairs, S, backbone_id="other")  # other backbone
    meta = d / "train_meta.json"
    m = json.loads(meta.read_text())
    assert m["backbone"] == "fake_timm" and m["feat_shape"] == [C, S // 14, S // 14]


def test_copy_cache(root, tmp_path):
    pairs = ade20k.list_pairs(root, "train")
    a, b = tmp_path / "a", tmp_path / "b"
    with pytest.raises(ValueError, match="no complete"):
        ade20k.copy_cache(a, b, "train")
    ade20k.build_feature_cache(FakeBackbone(), pairs, a, "train", batch_size=2, num_workers=0)
    ade20k.copy_cache(a, b, "train")
    assert ade20k.cache_complete(b, "train")
    assert (a / "train_feats.f16").read_bytes() == (b / "train_feats.f16").read_bytes()


def test_prepare_skips_when_extracted(root, tmp_path):
    assert ade20k.prepare_ade20k(tmp_path, url="http://invalid.invalid/x.zip") == root
