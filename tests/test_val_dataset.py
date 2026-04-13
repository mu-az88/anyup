"""
Tests for ValidationDataset and the val split manifest.

Synthetic tests run with no external data.
Real-data tests require:
    pytest tests/test_val_dataset.py --manifest_dir configs/ --imagenet_val_dir /data/imagenet/val
"""

import pytest
import torch
from pathlib import Path
from unittest.mock import patch
from torch.utils.data import DataLoader
from anyup.data.val_dataset import ValidationDataset, val_collate_fn


# ── CLI fixtures (options registered in conftest.py) ──────────────────────────
@pytest.fixture(scope="module")
def manifest_dir(request):
    return Path(request.config.getoption("--manifest_dir"))

@pytest.fixture(scope="module")
def imagenet_val_dir(request):
    val = request.config.getoption("--imagenet_val_dir")
    return Path(val) if val else None


# ── Synthetic fixtures ────────────────────────────────────────────────────────
NUM_FRAMES   = 4
SPATIAL_SIZE = (112, 112)
BATCH_SIZE   = 2


def _fake_frames(n=NUM_FRAMES, h=SPATIAL_SIZE[0], w=SPATIAL_SIZE[1]):
    """Return a (T, H, W, 3) float32 tensor in [0, 1]."""
    return torch.rand(n, h, w, 3)


@pytest.fixture()
def synthetic_manifest(tmp_path):
    """
    Writes train.txt / val.txt with synthetic (non-existent) paths,
    and patches ValidationDataset._load_video_frames so no real file is needed.
    Returns the tmp_path directory.
    """
    train_paths = [f"/fake/video_{i:03d}.mp4" for i in range(18)]
    val_paths   = [f"/fake/video_{i:03d}.mp4" for i in range(18, 20)]

    (tmp_path / "train.txt").write_text("\n".join(train_paths) + "\n")
    (tmp_path / "val.txt").write_text(  "\n".join(val_paths)   + "\n")
    return tmp_path


@pytest.fixture()
def synthetic_imagenet(tmp_path):
    """
    Creates a tiny directory of real PNG images so imagenet mode can be tested
    without an actual ImageNet download.
    """
    from PIL import Image
    import numpy as np

    img_dir = tmp_path / "imagenet_val" / "n01234"
    img_dir.mkdir(parents=True)
    for i in range(3):
        arr = (np.random.rand(64, 64, 3) * 255).astype("uint8")
        Image.fromarray(arr).save(img_dir / f"img_{i}.jpg")
    return tmp_path / "imagenet_val"


# ── Synthetic: no overlap ─────────────────────────────────────────────────────
def test_no_train_val_overlap_synthetic(synthetic_manifest):
    train_paths = set((synthetic_manifest / "train.txt").read_text().splitlines())
    val_paths   = set((synthetic_manifest / "val.txt").read_text().splitlines())
    overlap     = train_paths & val_paths
    assert len(overlap) == 0, f"Found {len(overlap)} clips in both splits"


# ── Synthetic: video mode shape ───────────────────────────────────────────────
def test_video_mode_output_shape_synthetic(synthetic_manifest):
    val_txt = synthetic_manifest / "val.txt"
    with patch.object(ValidationDataset, "_load_video_frames",
                      return_value=_fake_frames()):
        ds   = ValidationDataset(mode="video", manifest_path=val_txt,
                                 num_frames=NUM_FRAMES, spatial_size=SPATIAL_SIZE)
        item = ds[0]

    assert item["frames"].shape == (NUM_FRAMES, *SPATIAL_SIZE, 3)
    assert item["frames"].dtype == torch.float32


# ── Synthetic: imagenet mode shape ────────────────────────────────────────────
def test_imagenet_mode_output_shape_synthetic(synthetic_imagenet):
    ds   = ValidationDataset(mode="imagenet", imagenet_val_dir=synthetic_imagenet,
                             spatial_size=SPATIAL_SIZE)
    item = ds[0]
    assert item["frames"].shape == (1, *SPATIAL_SIZE, 3)
    assert item["frames"].dtype == torch.float32


# ── Synthetic: collation shape ────────────────────────────────────────────────
def test_collation_shape_synthetic(synthetic_manifest):
    val_txt = synthetic_manifest / "val.txt"
    with patch.object(ValidationDataset, "_load_video_frames",
                      return_value=_fake_frames()):
        ds     = ValidationDataset(mode="video", manifest_path=val_txt,
                                   num_frames=NUM_FRAMES, spatial_size=SPATIAL_SIZE)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=val_collate_fn)
        batch  = next(iter(loader))

    expected = (BATCH_SIZE, NUM_FRAMES, *SPATIAL_SIZE, 3)
    assert batch["frames"].shape == expected
    assert "paths" in batch and len(batch["paths"]) == BATCH_SIZE


# ── Synthetic: val_collate_fn mode field ─────────────────────────────────────
def test_collate_preserves_mode(synthetic_manifest):
    val_txt = synthetic_manifest / "val.txt"
    with patch.object(ValidationDataset, "_load_video_frames",
                      return_value=_fake_frames()):
        ds    = ValidationDataset(mode="video", manifest_path=val_txt,
                                  num_frames=NUM_FRAMES, spatial_size=SPATIAL_SIZE)
        batch = val_collate_fn([ds[0], ds[0]])
    assert batch["mode"] == "video"


# ── Real-data tests (skip when manifests absent) ──────────────────────────────
def test_no_train_val_overlap(manifest_dir):
    train_txt = manifest_dir / "train.txt"
    val_txt   = manifest_dir / "val.txt"
    if not train_txt.exists() or not val_txt.exists():
        pytest.skip("Manifests not yet generated — run prepare_val_split.py first")

    train_paths = set(train_txt.read_text().splitlines())
    val_paths   = set(val_txt.read_text().splitlines())
    overlap     = train_paths & val_paths
    assert len(overlap) == 0, f"Found {len(overlap)} clips in both train and val!"


def test_video_mode_output_shape(manifest_dir):
    val_txt = manifest_dir / "val.txt"
    if not val_txt.exists():
        pytest.skip("val.txt not found")

    ds   = ValidationDataset(mode="video", manifest_path=val_txt,
                             num_frames=NUM_FRAMES, spatial_size=SPATIAL_SIZE)
    item = ds[0]
    assert item["frames"].shape == (NUM_FRAMES, *SPATIAL_SIZE, 3)
    assert item["frames"].dtype == torch.float32


def test_imagenet_mode_output_shape(imagenet_val_dir):
    if imagenet_val_dir is None or not imagenet_val_dir.exists():
        pytest.skip("--imagenet_val_dir not provided or doesn't exist")

    SPATIAL_SIZE_224 = (224, 224)
    ds   = ValidationDataset(mode="imagenet", imagenet_val_dir=imagenet_val_dir,
                             spatial_size=SPATIAL_SIZE_224)
    item = ds[0]
    assert item["frames"].shape == (1, *SPATIAL_SIZE_224, 3)


def test_collation_shape(manifest_dir):
    val_txt = manifest_dir / "val.txt"
    if not val_txt.exists():
        pytest.skip("val.txt not found")

    ds     = ValidationDataset(mode="video", manifest_path=val_txt,
                               num_frames=NUM_FRAMES, spatial_size=SPATIAL_SIZE)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=val_collate_fn)
    batch  = next(iter(loader))
    assert batch["frames"].shape == (BATCH_SIZE, NUM_FRAMES, *SPATIAL_SIZE, 3)
