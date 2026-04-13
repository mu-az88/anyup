
import torch
import tempfile, subprocess, os
from torch.utils.data import DataLoader

from anyup.data.video_dataset import VideoDataset
from anyup.data.crop_sampler import SpatiotemporalCropSampler
from anyup.data.dataloader import (
    ClipDataset, collate_clip_items, build_dataloader, extract_batch_features
)
from anyup.data.feature_extractor import PerFrameFeatureExtractor
import torch.nn as nn


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_dummy_video(path: str, n_frames: int = 32, fps: int = 8,
                      h: int = 224, w: int = 224) -> None:
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi",
        "-i", f"color=c=red:size={w}x{h}:rate={fps}:duration={n_frames/fps}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", path,
    ]
    subprocess.run(cmd, check=True)


def _make_video_dir(tmp_dir: str, n_clips: int = 4) -> str:
    for i in range(n_clips):
        _make_dummy_video(os.path.join(tmp_dir, f"clip_{i}.mp4"))
    return tmp_dir


class _StubEncoder(nn.Module):
    patch_size = 14
    def forward_features(self, x):
        B, C, H, W = x.shape
        h, w = H // 14, W // 14
        return {"x_norm_patchtokens": torch.zeros(B, h * w, 384, device=x.device)}


def _make_extractor() -> PerFrameFeatureExtractor:
    ext = PerFrameFeatureExtractor.__new__(PerFrameFeatureExtractor)
    nn.Module.__init__(ext)                        # ← initializes _modules, _parameters, etc.
    object.__setattr__(ext, "patch_size", 14)      # plain int — bypass nn.Module.__setattr__
    object.__setattr__(ext, "device", torch.device("cpu"))
    object.__setattr__(ext, "cache_p", True)
    object.__setattr__(ext, "max_cache", 64)
    object.__setattr__(ext, "_cache", {})
    ext.encoder = _StubEncoder()                   # nn.Module assignment — now safe
    return ext


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_clip_dataset_item_shapes():
    """ClipDataset.__getitem__ must return correct shapes."""
    with tempfile.TemporaryDirectory() as d:
        _make_video_dir(d, n_clips=2)
        video_ds = VideoDataset(d, n_frames=8, spatial_size=(224, 224))
        sampler  = SpatiotemporalCropSampler(n_frames=4, crop_size=(7, 7))
        ds       = ClipDataset(video_ds, sampler, p_spatial_size=(28, 28))

        item = ds[0]
        assert item.video_full.shape == (8, 224, 224, 3),  item.video_full.shape
        assert item.video_crop.shape == (4, 56, 56, 3),    item.video_crop.shape  # 7 * (224//28) = 7*8 = 56
        assert item.clip_idx == 0
        print("PASS test_clip_dataset_item_shapes")


def test_collate_output_shapes():
    """collate_clip_items must produce (B, ...) tensors."""
    with tempfile.TemporaryDirectory() as d:
        _make_video_dir(d, n_clips=4)
        video_ds = VideoDataset(d, n_frames=8, spatial_size=(224, 224))
        sampler  = SpatiotemporalCropSampler(n_frames=4, crop_size=(7, 7))
        ds       = ClipDataset(video_ds, sampler, p_spatial_size=(28, 28))

        items  = [ds[i] for i in range(3)]
        batch  = collate_clip_items(items)

        assert batch["video_full"].shape == (3, 8, 224, 224, 3),  batch["video_full"].shape
        assert batch["video_crop"].shape == (3, 4,  56,  56, 3),  batch["video_crop"].shape
        assert len(batch["clip_indices"]) == 3
        print("PASS test_collate_output_shapes")


def test_collate_coords_are_lists():
    """Every coords entry must be a list of length B after collation."""
    with tempfile.TemporaryDirectory() as d:
        _make_video_dir(d, n_clips=4)
        video_ds = VideoDataset(d, n_frames=8, spatial_size=(224, 224))
        sampler  = SpatiotemporalCropSampler(n_frames=4, crop_size=(7, 7))
        ds       = ClipDataset(video_ds, sampler, p_spatial_size=(28, 28))

        B = 3
        items = [ds[i] for i in range(B)]
        batch = collate_clip_items(items)

        for key, val in batch["coords"].items():
            assert isinstance(val, list) and len(val) == B, \
                f"coords[{key!r}] has wrong type/length: {type(val)}, len={len(val)}"
        print("PASS test_collate_coords_are_lists")


def test_dataloader_batch_shapes():
    """Full DataLoader must yield correctly shaped batches."""
    with tempfile.TemporaryDirectory() as d:
        _make_video_dir(d, n_clips=6)
        loader = build_dataloader(
            root=d,
            n_frames=4,
            spatial_size=(224, 224),
            crop_size=(7, 7),
            p_spatial_size=(28, 28),
            batch_size=2,
            num_workers=0,   # 0 workers for test stability
            shuffle=False,
        )
        batch = next(iter(loader))
        assert batch["video_full"].shape == (2, 4, 224, 224, 3)
        assert batch["video_crop"].shape == (2, 4,  56,  56, 3)
        print("PASS test_dataloader_batch_shapes")


def test_extract_batch_features_shapes():
    """extract_batch_features must return (B, T, h_p, w_p, C) and (B, T', h_gt, w_gt, C)."""
    with tempfile.TemporaryDirectory() as d:
        _make_video_dir(d, n_clips=4)
        loader = build_dataloader(
            root=d,
            n_frames=4,
            spatial_size=(224, 224),
            crop_size=(7, 7),
            p_spatial_size=(28, 28),
            batch_size=2,
            num_workers=0,
            shuffle=False,
        )
        batch     = next(iter(loader))
        extractor = _make_extractor()
        device    = torch.device("cpu")

        p, q_hat = extract_batch_features(
            batch, extractor, p_spatial_size=(28, 28), device=device
        )
        # p:     (B, T,  h_p=28//14*14... wait: 28 pixels / 14 patch = 2 patches per side)
        # Actually: extractor resizes video to (28, 28) then encodes → 28//14=2 patches per side
        # q_hat: crop is (56, 56) → 56//14 = 4 patches per side
        assert p.shape     == (2, 4, 2, 2, 384),  f"p shape: {p.shape}"
        assert q_hat.shape == (2, 4, 4, 4, 384),  f"q_hat shape: {q_hat.shape}"
        print("PASS test_extract_batch_features_shapes — p:", p.shape, "q_hat:", q_hat.shape)


def test_no_gpu_tensors_from_dataloader():
    """DataLoader workers must return CPU tensors only."""
    with tempfile.TemporaryDirectory() as d:
        _make_video_dir(d, n_clips=4)
        loader = build_dataloader(
            root=d, n_frames=4, batch_size=2, num_workers=0, shuffle=False
        )
        batch = next(iter(loader))
        assert batch["video_full"].device.type == "cpu"
        assert batch["video_crop"].device.type == "cpu"
        print("PASS test_no_gpu_tensors_from_dataloader")


def test_p_cache_reduces_calls():
    """Second epoch over same clips should hit the cache for p extraction."""
    with tempfile.TemporaryDirectory() as d:
        _make_video_dir(d, n_clips=2)
        loader = build_dataloader(
            root=d, n_frames=4, batch_size=2, num_workers=0, shuffle=False
        )
        extractor = _make_extractor()
        device    = torch.device("cpu")

        batch = next(iter(loader))
        # First pass — cold cache
        extract_batch_features(batch, extractor, (28, 28), device)
        cache_size_after_epoch1 = len(extractor._cache)

        # Second pass — same batch, cache should already be warm
        extract_batch_features(batch, extractor, (28, 28), device)
        cache_size_after_epoch2 = len(extractor._cache)

        assert cache_size_after_epoch1 == cache_size_after_epoch2, \
            "Cache grew on second pass — cache hits not working"
        print("PASS test_p_cache_reduces_calls — cache size stable:", cache_size_after_epoch1)


if __name__ == "__main__":
    test_clip_dataset_item_shapes()
    test_collate_output_shapes()
    test_collate_coords_are_lists()
    test_dataloader_batch_shapes()
    test_extract_batch_features_shapes()
    test_no_gpu_tensors_from_dataloader()
    test_p_cache_reduces_calls()