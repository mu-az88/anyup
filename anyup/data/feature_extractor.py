
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Supported encoder registry
# ---------------------------------------------------------------------------

# Maps encoder name → (hub_repo, model_name, patch_size)
# patch_size determines the spatial downsampling factor: h_gt = H_crop // patch_size
_ENCODER_REGISTRY: Dict[str, Tuple[str, str, int]] = {
    "dinov2_vits14": ("facebookresearch/dinov2", "dinov2_vits14", 14),
    "dinov2_vitb14": ("facebookresearch/dinov2", "dinov2_vitb14", 14),
    "dinov2_vitl14": ("facebookresearch/dinov2", "dinov2_vitl14", 14),
}

# ImageNet stats — must match VideoDataset normalization
_MEAN = torch.tensor([0.485, 0.456, 0.406])
_STD  = torch.tensor([0.229, 0.224, 0.225])


# ---------------------------------------------------------------------------
# Encoder wrapper
# ---------------------------------------------------------------------------

class PerFrameFeatureExtractor(nn.Module):
    """
    Extracts dense patch features from a video crop, one frame at a time,
    using a frozen DINOv2 (or compatible ViT) encoder.

    Produces ground-truth features ``q̂ ∈ R^{T' × h_gt × w_gt × C}``
    where ``h_gt = H_crop // patch_size``, ``w_gt = W_crop // patch_size``.

    Parameters
    ----------
    encoder_name : str
        Key into ``_ENCODER_REGISTRY``. Default: ``"dinov2_vits14"``.
    device : torch.device | str
        Device to run the encoder on. Must match the training device.
        ↑ encoder stays here for the full training run — don't move it.
    cache_p : bool
        Whether to cache full-video coarse features ``p`` to avoid
        re-extracting features for the same clip across epochs.
        Cache lives in CPU RAM to avoid occupying GPU memory.
        ↑ set False if you're RAM-constrained or dataset is huge.
    max_cache_size : int
        Maximum number of (clip_idx, frame_key) entries in the cache.
        Each entry is a ``(T, h_p, w_p, C)`` float32 tensor on CPU.
        ↑ increase if you have >32 GB RAM and a small dataset.
    """

    def __init__(
        self,
        encoder_name: str = "dinov2_vits14",
        device: torch.device | str = "cuda",
        cache_p: bool = True,
        max_cache_size: int = 512,         # ↓ reduce if RAM is tight
    ) -> None:
        super().__init__()

        if encoder_name not in _ENCODER_REGISTRY:
            raise ValueError(
                f"Unknown encoder '{encoder_name}'. "
                f"Available: {list(_ENCODER_REGISTRY.keys())}"
            )

        repo, name, patch_size = _ENCODER_REGISTRY[encoder_name]
        self.patch_size = patch_size       # spatial downsampling factor; h_gt = H // patch_size

        # Load and freeze the encoder
        encoder = torch.hub.load(repo, name, pretrained=True)
        encoder = encoder.to(device)
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad_(False)        # frozen — no gradients flow through here

        self.encoder = encoder
        self.device  = torch.device(device)

        # Optional LRU-style cache for coarse feature volumes p
        self.cache_p      = cache_p
        self.max_cache    = max_cache_size  # ↑ more entries = less re-extraction
        self._cache: Dict[str, torch.Tensor] = {}   # key → (T, h_p, w_p, C) on CPU

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Encode a single RGB frame.

        Parameters
        ----------
        frame : Tensor (H, W, 3) float32, ImageNet-normalised

        Returns
        -------
        Tensor (h_gt, w_gt, C)
        """
        H, W, _ = frame.shape

        # Validate spatial dims are divisible by patch_size
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"Frame size ({H}, {W}) must be divisible by "
                f"patch_size={self.patch_size}. "
                f"Resize the crop to a multiple of {self.patch_size} before extraction."
            )

        h_gt = H // self.patch_size   # output height in patches; scales with patch_size
        w_gt = W // self.patch_size   # output width  in patches

        # (H, W, 3) → (1, 3, H, W) for encoder
        x = frame.permute(2, 0, 1).unsqueeze(0).to(self.device)   # (1, 3, H, W)

        # DINOv2 forward_features returns a dict; patch tokens are under 'x_norm_patchtokens'
        # Shape: (1, h_gt*w_gt, C) — flat sequence of patch tokens, no CLS token
        out = self.encoder.forward_features(x)
        patch_tokens = out["x_norm_patchtokens"]  # (1, h_gt*w_gt, C)

        # Reshape flat patch sequence → spatial grid
        C = patch_tokens.shape[-1]                # encoder embedding dim; 384 for ViT-S
        feat = patch_tokens[0].reshape(h_gt, w_gt, C)   # (h_gt, w_gt, C)

        return feat  # stays on self.device

    @torch.no_grad()
    def _encode_clip(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Encode multiple frames independently.

        Parameters
        ----------
        frames : Tensor (T', H, W, 3) float32, ImageNet-normalised

        Returns
        -------
        Tensor (T', h_gt, w_gt, C) on self.device
        """
        T = frames.shape[0]    # number of frames to encode; ↑ scales linearly with GPU time

        feats = []
        for t in range(T):     # loop over T' frames; each is a separate encoder forward pass
            feat = self._encode_frame(frames[t])   # (h_gt, w_gt, C)
            feats.append(feat)

        return torch.stack(feats, dim=0)   # (T', h_gt, w_gt, C)

    @staticmethod
    def _cache_key(clip_idx: int, frame_indices: List[int]) -> str:
        """Stable string key for the (clip, frames) pair."""
        frames_str = ",".join(str(i) for i in sorted(frame_indices))
        return f"{clip_idx}|{frames_str}"

    def _maybe_evict_cache(self) -> None:
        """Drop oldest entry if cache is full (FIFO eviction)."""
        while len(self._cache) >= self.max_cache:   # ↑ max_cache controls RAM usage
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_gt(
        self,
        video_crop: torch.Tensor,   # (T', H_c, W_c, 3) float32, ImageNet-normalised
    ) -> torch.Tensor:
        """
        Extract GT features from the sampled crop.

        Parameters
        ----------
        video_crop : Tensor (T', H_c, W_c, 3)
            RGB crop returned by ``SpatiotemporalCropSampler``.
            Must already be ImageNet-normalised (done by ``VideoDataset``).

        Returns
        -------
        q_hat : Tensor (T', h_gt, w_gt, C) on self.device
            Dense patch features — the supervision target for AnyUp3D.
        """
        return self._encode_clip(video_crop)

    @torch.no_grad()
    def extract_p(
        self,
        video_full: torch.Tensor,   # (T, H, W, 3) float32
        clip_idx: int,
        frame_indices: List[int],
        target_spatial_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Extract (or retrieve from cache) the full coarse feature volume ``p``.

        The full video is downsampled spatially before encoding, producing
        coarse features at a lower resolution than ``extract_gt``.

        Parameters
        ----------
        video_full : Tensor (T, H, W, 3)
            The full guidance video.
        clip_idx : int
            Dataset index of this clip — used as cache key.
        frame_indices : List[int]
            Which frames were selected — part of the cache key.
        target_spatial_size : (h_p, w_p) | None
            If given, resize video_full spatially to this size before
            encoding. Must be divisible by patch_size.
            If None, encode at full resolution (expensive).
            ↓ smaller = coarser p, less GPU memory, faster extraction.

        Returns
        -------
        p : Tensor (T, h_p, w_p, C) on CPU (for storage), move to device before use.
        """
        key = self._cache_key(clip_idx, frame_indices)

        if self.cache_p and key in self._cache:
            return self._cache[key]   # cache hit — skip re-extraction

        # Resize spatially if requested
        if target_spatial_size is not None:
            T, H, W, C3 = video_full.shape
            # F.interpolate needs (N, C, H, W)
            x = video_full.permute(0, 3, 1, 2).float()    # (T, 3, H, W)
            x = F.interpolate(                             # ↓ target_spatial_size controls feature resolution
                x,
                size=target_spatial_size,                  # (h_p, w_p); smaller = cheaper
                mode="bilinear",
                align_corners=False,
            )
            video_to_encode = x.permute(0, 2, 3, 1)       # (T, h_p*ps, w_p*ps, 3)
        else:
            video_to_encode = video_full                   # full resolution — expensive for large H,W

        # Encode all T frames
        p = self._encode_clip(video_to_encode)             # (T, h_p, w_p, C); T frames on GPU

        # Store on CPU to free GPU memory
        p_cpu = p.cpu()

        if self.cache_p:
            self._maybe_evict_cache()
            self._cache[key] = p_cpu                       # (T, h_p, w_p, C) on CPU RAM

        return p_cpu

    def clear_cache(self) -> None:
        """Manually clear the feature cache (e.g., between epochs)."""
        self._cache.clear()