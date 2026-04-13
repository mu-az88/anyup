"""
Validation dataset with two modes:

  mode="video"  — loads short clips from val.txt manifest (T > 1 stages)
  mode="imagenet" — loads single images from ImageNet val directory (T=1 warmup),
                    allowing direct comparison to published 2D AnyUp numbers.

Both modes return the same dict structure so the validation loop is mode-agnostic.
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from typing import Literal

try:
    import decord                              # fast video reader; preferred over torchvision
    _DECORD_AVAILABLE = True
except ImportError:
    _DECORD_AVAILABLE = False
    import torchvision.io as tvio              # fallback


# ── ImageNet normalization — must match whatever the feature extractor expects ──
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class ValidationDataset(Dataset):
    """
    Unified validation dataset for AnyUp3D.

    Args:
        mode:          "video" for clip-level validation, "imagenet" for single-frame.
        manifest_path: Path to val.txt (used only in "video" mode).
        imagenet_val_dir: Root of ImageNet val split, e.g. /data/imagenet/val
                          (used only in "imagenet" mode).
        num_frames:    T — number of frames to sample per clip (video mode only).
                       Set to match the current curriculum stage.  # ↓ cost: reduce T to save memory
        frame_stride:  Temporal stride when sampling frames.       # ↓ cost: increase stride to sample fewer frames
        spatial_size:  (H, W) to resize frames to before returning. # ↓ cost: reduce to save memory; affects eval resolution
    """

    def __init__(
        self,
        mode: Literal["video", "imagenet"],
        manifest_path: Path | None = None,
        imagenet_val_dir: Path | None = None,
        num_frames: int = 8,          # ↓ memory cost: tied to T in the curriculum scheduler
        frame_stride: int = 2,        # ↓ memory cost: higher stride = fewer frames read from disk
        spatial_size: tuple[int, int] = (224, 224),  # ↓ memory cost: larger → more tokens in encoder
    ):
        self.mode         = mode
        self.num_frames   = num_frames    # ↓ memory: also used in collation — change together with batch size
        self.frame_stride = frame_stride
        self.spatial_size = spatial_size  # ↓ memory: encoder patch count = (H/patch_size) * (W/patch_size) * T

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.resize    = transforms.Resize(spatial_size)

        if mode == "video":
            assert manifest_path is not None, "manifest_path required for video mode"
            manifest_path = Path(manifest_path)
            assert manifest_path.exists(), f"Val manifest not found: {manifest_path}"
            self.paths = [Path(l.strip()) for l in manifest_path.read_text().splitlines() if l.strip()]

        elif mode == "imagenet":
            assert imagenet_val_dir is not None, "imagenet_val_dir required for imagenet mode"
            imagenet_val_dir = Path(imagenet_val_dir)
            assert imagenet_val_dir.exists(), f"ImageNet val dir not found: {imagenet_val_dir}"
            IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPEG"}
            self.paths = sorted(
                p for p in imagenet_val_dir.rglob("*")
                if p.suffix in IMAGE_EXTENSIONS
            )
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'video' or 'imagenet'.")

        print(f"[ValidationDataset] mode={mode}, {len(self.paths)} items")

    # ── Length ─────────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.paths)

    # ── Single item ────────────────────────────────────────────────────────────
    def __getitem__(self, idx: int) -> dict:
        path = self.paths[idx]

        if self.mode == "video":
            frames = self._load_video_frames(path)     # (T, H, W, 3) float32 in [0,1]
        else:
            frames = self._load_image_as_clip(path)    # (1, H, W, 3) float32 in [0,1]

        # Normalize each frame with ImageNet stats
        # frames: (T, H, W, 3) → normalize → (T, H, W, 3)
        frames = torch.stack([
            self.normalize(f.permute(2, 0, 1))        # (3, H, W) for normalize, then back
                 .permute(1, 2, 0)                    # back to (H, W, 3)
            for f in frames
        ])  # shape: (T, H, W, 3)   ↓ T is the sequence length cost driver

        return {
            "frames": frames,          # (T, H, W, 3) — main input to AnyUp3D
            "path":   str(path),       # for debugging / logging
            "mode":   self.mode,
        }

    # ── Video loading ──────────────────────────────────────────────────────────
    def _load_video_frames(self, path: Path) -> torch.Tensor:
        """
        Returns (T, H, W, 3) float32 tensor in [0, 1].
        Samples self.num_frames frames with self.frame_stride spacing.
        """
        if _DECORD_AVAILABLE:
            return self._load_with_decord(path)
        else:
            return self._load_with_torchvision(path)

    def _load_with_decord(self, path: Path) -> torch.Tensor:
        import decord
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(str(path), num_threads=1)  # ↓ cost: num_threads; 1 is safe for dataloader workers

        total_frames = len(vr)
        # Center the sampled window in the clip
        window = self.num_frames * self.frame_stride   # ↓ memory: window size scales with both num_frames and stride
        start  = max(0, (total_frames - window) // 2)
        indices = list(range(start, min(total_frames, start + window), self.frame_stride))
        indices = indices[:self.num_frames]            # ↓ token count: capped at num_frames (= T in attention)

        # Pad by repeating last frame if clip is shorter than requested
        while len(indices) < self.num_frames:
            indices.append(indices[-1])

        frames = vr.get_batch(indices).float() / 255.0  # (T, H, W, 3) in [0,1]
        frames = torch.stack([
            self.resize(f.permute(2, 0, 1)).permute(1, 2, 0)  # resize to spatial_size
            for f in frames
        ])  # (T, H, W, 3)   ↓ H, W set by spatial_size — changing it changes all downstream token counts
        return frames

    def _load_with_torchvision(self, path: Path) -> torch.Tensor:
        video, _, _ = tvio.read_video(str(path), pts_unit="sec")  # (T_full, H, W, 3) uint8
        total_frames = video.shape[0]
        window  = self.num_frames * self.frame_stride
        start   = max(0, (total_frames - window) // 2)
        indices = list(range(start, min(total_frames, start + window), self.frame_stride))
        indices = indices[:self.num_frames]

        while len(indices) < self.num_frames:
            indices.append(indices[-1])

        frames = video[indices].float() / 255.0       # (T, H, W, 3)
        frames = torch.stack([
            self.resize(f.permute(2, 0, 1)).permute(1, 2, 0)
            for f in frames
        ])
        return frames

    # ── ImageNet single-image loading ──────────────────────────────────────────
    def _load_image_as_clip(self, path: Path) -> torch.Tensor:
        """
        Loads one image and wraps it as a single-frame clip: (1, H, W, 3).
        Used in imagenet mode for T=1 warmup validation.
        """
        from PIL import Image
        img = Image.open(path).convert("RGB")
        img_t = self.to_tensor(img)                    # (3, H, W) in [0,1]
        img_t = self.resize(img_t)                     # (3, H, W) at spatial_size
        img_t = img_t.permute(1, 2, 0).unsqueeze(0)   # (1, H, W, 3)
        return img_t


# ── Collation ──────────────────────────────────────────────────────────────────
def val_collate_fn(batch: list[dict]) -> dict:
    """
    Stacks a list of val items into a batch.
    All items must have the same T, H, W (enforced by spatial_size + num_frames).
    """
    return {
        "frames": torch.stack([item["frames"] for item in batch]),  # (B, T, H, W, 3)
        "paths":  [item["path"] for item in batch],
        "mode":   batch[0]["mode"],
    }