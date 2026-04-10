"""
Tests for ValidationDataset and the val split manifest.
Run: pytest tests/test_val_dataset.py --manifest_dir configs/ --imagenet_val_dir /data/imagenet/val
"""

import pytest
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from anyup.data.val_dataset import ValidationDataset, val_collate_fn


# ── CLI options (registered in conftest.py) ────────────────────────────────────
def pytest_addoption(parser):
    parser.addoption("--manifest_dir",     default="configs/")
    parser.addoption("--imagenet_val_dir", default=None)


@pytest.fixture(scope="module")
def manifest_dir(request):
    return Path(request.config.getoption("--manifest_dir"))

@pytest.fixture(scope="module")
def imagenet_val_dir(request):
    val = request.config.getoption("--imagenet_val_dir")
    return Path(val) if val else None


# ── Test 1: No overlap between train and val splits ────────────────────────────
def test_no_train_val_overlap(manifest_dir):
    train_txt = manifest_dir / "train.txt"
    val_txt   = manifest_dir / "val.txt"

    if not train_txt.exists() or not val_txt.exists():
        pytest.skip("Manifests not yet generated — run prepare_val_split.py first")

    train_paths = set(train_txt.read_text().splitlines())
    val_paths   = set(val_txt.read_text().splitlines())
    overlap     = train_paths & val_paths

    assert len(overlap) == 0, f"Found {len(overlap)} clips in both train and val!"
    print(f"PASSED: no overlap — train={len(train_paths)}, val={len(val_paths)}")


# ── Test 2: Video mode output shape ───────────────────────────────────────────
def test_video_mode_output_shape(manifest_dir):
    val_txt = manifest_dir / "val.txt"
    if not val_txt.exists():
        pytest.skip("val.txt not found")

    NUM_FRAMES   = 4          # ↓ keep small for fast test
    SPATIAL_SIZE = (112, 112) # ↓ keep small for fast test

    ds = ValidationDataset(
        mode="video",
        manifest_path=val_txt,
        num_frames=NUM_FRAMES,
        spatial_size=SPATIAL_SIZE,
    )
    item = ds[0]

    assert "frames" in item
    assert item["frames"].shape == (NUM_FRAMES, SPATIAL_SIZE[0], SPATIAL_SIZE[1], 3), \
        f"Unexpected shape: {item['frames'].shape}"
    assert item["frames"].dtype == torch.float32
    print(f"PASSED: video mode shape {item['frames'].shape}")


# ── Test 3: ImageNet mode output shape ────────────────────────────────────────
def test_imagenet_mode_output_shape(imagenet_val_dir):
    if imagenet_val_dir is None or not imagenet_val_dir.exists():
        pytest.skip("--imagenet_val_dir not provided or doesn't exist")

    SPATIAL_SIZE = (224, 224)
    ds = ValidationDataset(
        mode="imagenet",
        imagenet_val_dir=imagenet_val_dir,
        spatial_size=SPATIAL_SIZE,
    )
    item = ds[0]

    assert item["frames"].shape == (1, SPATIAL_SIZE[0], SPATIAL_SIZE[1], 3), \
        f"Unexpected shape: {item['frames'].shape}"
    print(f"PASSED: imagenet mode shape {item['frames'].shape}")


# ── Test 4: Collation produces correct batch shape ────────────────────────────
def test_collation_shape(manifest_dir):
    val_txt = manifest_dir / "val.txt"
    if not val_txt.exists():
        pytest.skip("val.txt not found")

    BATCH_SIZE   = 2  # ↓ keep small; collation correctness doesn't depend on batch size
    NUM_FRAMES   = 4  # ↓ must match dataset num_frames
    SPATIAL_SIZE = (112, 112)

    ds     = ValidationDataset(mode="video", manifest_path=val_txt,
                               num_frames=NUM_FRAMES, spatial_size=SPATIAL_SIZE)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=val_collate_fn)
    batch  = next(iter(loader))

    expected = (BATCH_SIZE, NUM_FRAMES, SPATIAL_SIZE[0], SPATIAL_SIZE[1], 3)
    assert batch["frames"].shape == expected, f"Unexpected batch shape: {batch['frames'].shape}"
    print(f"PASSED: collated batch shape {batch['frames'].shape}")


if __name__ == "__main__":
    test_no_train_val_overlap(Path("configs/"))
    print("Run remaining tests with pytest and --manifest_dir / --imagenet_val_dir flags.")