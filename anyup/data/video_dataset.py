
from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms.functional import normalize
import decord # lazy import — optional dependency

# ---------------------------------------------------------------------------
# ImageNet normalisation constants (same as 2D AnyUp)
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------

def _load_clip_decord(
    path: str,
    frame_indices: List[int],
) -> torch.Tensor:
    """Load specific frames with decord. Returns (T, H, W, 3) uint8."""
    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(path, ctx=decord.cpu(0))
    frames = vr.get_batch(frame_indices)          # (T, H, W, 3) uint8 tensor
    return frames


def _load_clip_torchvision(
    path: str,
    frame_indices: List[int],
) -> torch.Tensor:
    """Load specific frames with torchvision. Returns (T, H, W, 3) uint8."""
    from torchvision.io import read_video
    # read_video returns (T, H, W, 3) uint8
    video, _, _ = read_video(path, output_format="THWC", pts_unit="sec")
    # clamp indices in case the video is shorter than expected
    frame_indices = [min(i, len(video) - 1) for i in frame_indices]
    return video[frame_indices]


def _try_load_clip(path: str, frame_indices: List[int]) -> torch.Tensor:
    """Try decord first, fall back to torchvision."""
    try:
        return _load_clip_decord(path, frame_indices)
    except Exception:
        return _load_clip_torchvision(path, frame_indices)


# ---------------------------------------------------------------------------
# Spatial resize helper
# ---------------------------------------------------------------------------

def _resize_clip(frames: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """
    Resize (T, H, W, 3) uint8 → (T, H_out, W_out, 3) uint8.
    Uses bilinear interpolation via F.interpolate.
    """
    T, H, W, C = frames.shape
    # F.interpolate expects (N, C, H, W)
    x = frames.permute(0, 3, 1, 2).float()        # (T, 3, H, W)
    x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    x = x.permute(0, 2, 3, 1)                     # (T, H_out, W_out, 3)
    return x.to(torch.uint8)


# ---------------------------------------------------------------------------
# Frame sampling strategies
# ---------------------------------------------------------------------------

def _uniform_stride_indices(total_frames: int, n_frames: int, stride: int) -> List[int]:
    """
    Sample n_frames with a fixed stride, starting at a random offset.
    Clamps to [0, total_frames-1] if the clip would exceed the video length.
    """
    max_start = max(0, total_frames - (n_frames - 1) * stride - 1)
    start = random.randint(0, max_start)
    return [min(start + i * stride, total_frames - 1) for i in range(n_frames)]


def _random_contiguous_indices(total_frames: int, n_frames: int) -> List[int]:
    """Sample n_frames contiguous frames from a random offset."""
    max_start = max(0, total_frames - n_frames)
    start = random.randint(0, max_start)
    return list(range(start, start + n_frames))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class VideoDataset(Dataset):
    """
    PyTorch Dataset that loads short video clips from a flat directory
    (or any glob-able tree) of video files.

    Returns
    -------
    torch.Tensor
        Shape ``(T, H, W, 3)``, dtype ``float32``, ImageNet-normalized,
        values roughly in [-2.1, 2.6].

    Parameters
    ----------
    root : str | Path
        Root directory to scan for videos (recursive).
    n_frames : int
        Number of frames T to return per clip.
    spatial_size : tuple[int, int]
        (H, W) to resize every frame to.
    stride : int
        Temporal stride between sampled frames. ``stride=1`` gives contiguous
        frames; higher values give sparser temporal coverage.
    min_video_frames : int
        Videos shorter than this are skipped at scan time.
    transform : Callable | None
        Optional additional transform applied to the final
        ``(T, H, W, 3)`` float tensor (e.g. random horizontal flip).
    """

    def __init__(
        self,
        root: str | Path,
        n_frames: int = 8,
        spatial_size: Tuple[int, int] = (224, 224),
        stride: int = 1,
        min_video_frames: int = 16,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.root = Path(root)
        self.n_frames = n_frames
        self.spatial_size = spatial_size
        self.stride = stride
        self.min_video_frames = min_video_frames
        self.transform = transform

        self.video_paths: List[Path] = self._scan()
        if len(self.video_paths) == 0:
            raise RuntimeError(f"No valid video files found under {self.root}")

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def _scan(self) -> List[Path]:
        paths = []
        for ext in VIDEO_EXTENSIONS:
            paths.extend(self.root.rglob(f"*{ext}"))
        paths = sorted(paths)

        # Filter by minimum length using a cheap metadata read
        valid = []
        for p in paths:
            n = self._get_frame_count(str(p))
            if n >= self.min_video_frames:
                valid.append(p)

        return valid

    @staticmethod
    def _get_frame_count(path: str) -> int:
        """Return approximate frame count using decord or torchvision."""
        try:
            import decord
            vr = decord.VideoReader(path, ctx=decord.cpu(0))
            return len(vr)
        except Exception:
            pass
        try:
            from torchvision.io import read_video_timestamps
            pts, _ = read_video_timestamps(path, pts_unit="pts")
            return len(pts)
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.video_paths[idx]
        n_total = self._get_frame_count(str(path))

        # Sample frame indices
        min_required = (self.n_frames - 1) * self.stride + 1
        if n_total >= min_required:
            indices = _uniform_stride_indices(n_total, self.n_frames, self.stride)
        else:
            # Video too short for the requested stride — fall back to contiguous
            indices = _random_contiguous_indices(n_total, self.n_frames)

        # Load frames: (T, H, W, 3) uint8
        frames = _try_load_clip(str(path), indices)

        # Resize if needed
        if frames.shape[1] != self.spatial_size[0] or frames.shape[2] != self.spatial_size[1]:
            frames = _resize_clip(frames, self.spatial_size)

        # uint8 [0, 255] → float32 [0.0, 1.0]
        clip = frames.float() / 255.0              # (T, H, W, 3)

        # ImageNet normalization: applied per-channel across (T, H, W)
        # normalize() expects (C, *) so we permute temporarily
        mean = torch.tensor(IMAGENET_MEAN)         # (3,)
        std  = torch.tensor(IMAGENET_STD)          # (3,)
        clip = (clip - mean) / std                 # broadcast over (T, H, W, 3)

        # Optional extra transform
        if self.transform is not None:
            clip = self.transform(clip)

        return clip                                # (T, H, W, 3) float32