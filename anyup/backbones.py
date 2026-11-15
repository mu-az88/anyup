"""
Unified backbone loader for AnyUp3D experiments (task T01).

One entry point, one adapter per backbone, one standard output:

    bb  = BackboneLoader.load("videomae", device="cuda")
    out = bb(frames)          # frames: (T, 3, H, W) float RGB in [0, 1], any H, W
    out.features              # (T', C, h, w)    raw, unnormalized (AnyUp normalizes internally)
    out.guide                 # (T', 3, gH, gW)  ImageNet-normalized, aligned to T'
    out.meta                  # dict, see BackboneOutput

An adapter processes ONE chunk of frames. Reading videos and splitting long
videos into chunks is done by the video reader (T02), not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# ImageNet stats — guide normalization (matches training, anyup/data/video_dataset.py)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# torchvision Swin3D_T_Weights.KINETICS400_V1 training stats
KINETICS_MEAN = (0.43216, 0.394666, 0.37645)
KINETICS_STD  = (0.22803, 0.22145, 0.216989)

Size = Union[int, Tuple[int, int]]


# ---------------------------------------------------------------------------
# Output object
# ---------------------------------------------------------------------------

@dataclass
class BackboneOutput:
    """
    features : (T', C, h, w) float32, raw backbone features (no L2 / layer norm added)
    guide    : (T', 3, gH, gW) float32, ImageNet-normalized frames aligned to features
    meta     : name, feat_dim, grid (h, w), patch_size, temporal_stride, t_in, t_out,
               frame_index (input frame index used as guide for each feature frame),
               input_size, guide_size, plus adapter-specific keys
    """
    features: torch.Tensor
    guide: torch.Tensor
    meta: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, Type["BackboneAdapter"]] = {}


def register(name: str) -> Callable[[Type["BackboneAdapter"]], Type["BackboneAdapter"]]:
    def deco(cls):
        if name in _REGISTRY:
            raise ValueError(f"Backbone {name!r} already registered")
        _REGISTRY[name] = cls
        cls.name = name
        return cls
    return deco


class BackboneLoader:
    """Entry point: ``BackboneLoader.load(name, **kwargs)``."""

    @staticmethod
    def load(name: str, **kwargs) -> "BackboneAdapter":
        if name not in _REGISTRY:
            raise ValueError(f"Unknown backbone {name!r}. Available: {BackboneLoader.available()}")
        return _REGISTRY[name](**kwargs)

    @staticmethod
    def available() -> List[str]:
        return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Shared preprocessing
# ---------------------------------------------------------------------------

def _to_hw(size: Size) -> Tuple[int, int]:
    return (size, size) if isinstance(size, int) else tuple(size)


def resize_frames(frames: torch.Tensor, size: Size) -> torch.Tensor:
    """
    frames : (T, 3, H, W) float [0, 1]
    Returns: (T, 3, size_h, size_w), bilinear + antialias. No-op if already at size.
    """
    hw = _to_hw(size)
    if tuple(frames.shape[-2:]) == hw:
        return frames
    return TF.resize(frames, list(hw), interpolation=TF.InterpolationMode.BILINEAR, antialias=True)


def normalize_frames(frames: torch.Tensor, mean, std) -> torch.Tensor:
    """(T, 3, H, W) [0, 1] → normalized. Same arithmetic as torchvision T.Normalize."""
    mean = torch.as_tensor(mean, dtype=frames.dtype, device=frames.device)[:, None, None]
    std  = torch.as_tensor(std,  dtype=frames.dtype, device=frames.device)[:, None, None]
    return (frames - mean) / std


# ---------------------------------------------------------------------------
# Base adapter
# ---------------------------------------------------------------------------

class BackboneAdapter(nn.Module):
    """
    Subclasses implement ``_extract(x) -> (features, frame_index)`` where
    x is the (T, 3, input_h, input_w) backbone-normalized input on device.
    """

    name: str = ""
    feat_dim: int = 0
    patch_size: int = 0
    temporal_stride: int = 1
    input_mean = IMAGENET_MEAN
    input_std  = IMAGENET_STD

    def __init__(self, device: str = "cuda", input_size: Size = 224, guide_size: Optional[Size] = None):
        super().__init__()
        self.device = torch.device(device)
        self.input_size = _to_hw(input_size)   # ↓ backbone input resolution; sets the feature grid (h, w) and memory
        self.guide_size = _to_hw(guide_size) if guide_size is not None else self.input_size  # guide resolution = AnyUp output size by default

    def _freeze(self, model: nn.Module) -> nn.Module:
        model = model.to(self.device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    def _check_frames(self, frames: torch.Tensor) -> None:
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError(f"frames must be (T, 3, H, W), got {tuple(frames.shape)}")
        if not frames.is_floating_point():
            raise ValueError(f"frames must be float in [0, 1], got dtype {frames.dtype}")

    def _extract(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> BackboneOutput:
        self._check_frames(frames)
        frames = frames.to(self.device, torch.float32)                                   # (T, 3, H, W) [0, 1]

        x = normalize_frames(resize_frames(frames, self.input_size), self.input_mean, self.input_std)  # (T, 3, iH, iW)
        feats, frame_index = self._extract(x)                                             # (T', C, h, w)

        guide = resize_frames(frames[frame_index], self.guide_size)                       # resized from ORIGINAL frames
        guide = normalize_frames(guide, IMAGENET_MEAN, IMAGENET_STD)                      # (T', 3, gH, gW)

        meta = dict(
            name=self.name,
            feat_dim=feats.shape[1],
            grid=tuple(feats.shape[-2:]),
            patch_size=self.patch_size,
            temporal_stride=self.temporal_stride,
            t_in=frames.shape[0],
            t_out=feats.shape[0],
            frame_index=list(frame_index),
            input_size=self.input_size,
            guide_size=self.guide_size,
        )
        meta.update(self._extra_meta())
        return BackboneOutput(features=feats.float().contiguous(), guide=guide, meta=meta)

    def _extra_meta(self) -> Dict:
        return {}


# ---------------------------------------------------------------------------
# VideoMAE
# ---------------------------------------------------------------------------

@register("videomae")
class VideoMAEAdapter(BackboneAdapter):
    """
    HF VideoMAE ViT-B/16, tubelet 2. Input must be exactly NUM_FRAMES frames at 224
    (position embeddings are fixed). 16 frames → 8 feature frames; guide = frames[::2].
    """

    NUM_FRAMES = 16   # fixed by videomae-base position embeddings; padding of short chunks is T02's job
    feat_dim = 768
    patch_size = 16
    temporal_stride = 2

    def __init__(self, device="cuda", input_size=224, guide_size=None, model_id: str = "MCG-NJU/videomae-base"):
        super().__init__(device, input_size, guide_size)
        from transformers import VideoMAEModel
        self.model_id = model_id
        self.model = self._freeze(VideoMAEModel.from_pretrained(model_id))
        if self.input_size != (self.model.config.image_size,) * 2:
            raise ValueError(f"VideoMAE needs input_size={self.model.config.image_size}, got {self.input_size}")
        self.temporal_stride = self.model.config.tubelet_size

    def _extract(self, x):
        T = x.shape[0]
        if T != self.NUM_FRAMES:
            raise ValueError(f"VideoMAE needs exactly {self.NUM_FRAMES} frames per chunk, got {T}")
        h, w = self.input_size[0] // self.patch_size, self.input_size[1] // self.patch_size   # 14 × 14 at 224
        t_out = T // self.temporal_stride                                                       # 8 for T=16

        tokens = self.model(pixel_values=x.unsqueeze(0)).last_hidden_state.squeeze(0)  # (1, T, 3, H, W) → (t_out*h*w, C)
        feats = tokens.reshape(t_out, h, w, -1).permute(0, 3, 1, 2)                    # (t_out, C, h, w)
        frame_index = list(range(0, T, self.temporal_stride))                          # first frame of each tubelet
        return feats, frame_index

    def _extra_meta(self):
        return dict(model_id=self.model_id)


# ---------------------------------------------------------------------------
# Video Swin-T
# ---------------------------------------------------------------------------

@register("swin_t")
class SwinTAdapter(BackboneAdapter):
    """
    torchvision Video Swin-T (Kinetics-400), output of the final `norm` layer.

    temporal_mode:
      "per_frame" — each frame duplicated into a 2-frame clip, all frames batched → T' = T
                    (as in Exp 3; no temporal context from the backbone)
      "clip"      — native forward over the whole chunk → T' = ceil(T / 2); guide = frames[::2]
    norm: "kinetics" (the weights' own stats, default) | "imagenet" (what Exp 3 used)
    """

    feat_dim = 768
    patch_size = 32   # effective spatial stride: patch 4 × three 2× merges → 7 × 7 at 224

    def __init__(self, device="cuda", input_size=224, guide_size=None,
                 temporal_mode: str = "per_frame", norm: str = "kinetics"):
        super().__init__(device, input_size, guide_size)
        from torchvision.models.video import swin3d_t, Swin3D_T_Weights
        if temporal_mode not in ("per_frame", "clip"):
            raise ValueError(f"temporal_mode must be 'per_frame' or 'clip', got {temporal_mode!r}")
        if norm not in ("kinetics", "imagenet"):
            raise ValueError(f"norm must be 'kinetics' or 'imagenet', got {norm!r}")
        self.temporal_mode = temporal_mode
        self.norm = norm
        self.input_mean, self.input_std = (KINETICS_MEAN, KINETICS_STD) if norm == "kinetics" else (IMAGENET_MEAN, IMAGENET_STD)
        self.model = self._freeze(swin3d_t(weights=Swin3D_T_Weights.KINETICS400_V1))
        self.temporal_stride = 1 if temporal_mode == "per_frame" else self.model.patch_embed.proj.stride[0]  # tubelet 2

    def _features(self, clips):
        """clips: (B, 3, T, H, W) → norm output permuted to (B, C, T', h, w). Same as Exp 3's norm hook."""
        m = self.model
        x = m.patch_embed(clips)   # (B, T', H/4, W/4, 96)
        x = m.pos_drop(x)
        x = m.features(x)          # (B, T', h, w, 768)
        x = m.norm(x)
        return x.permute(0, 4, 1, 2, 3)

    def _extract(self, x):
        T = x.shape[0]
        if self.temporal_mode == "per_frame":
            clips = x.unsqueeze(2).repeat(1, 1, 2, 1, 1)   # (T, 3, 2, H, W) — all T frames in ONE batch; chunk size (T02) sets memory
            feats = self._features(clips).squeeze(2)        # (T, C, 1, h, w) → (T, C, h, w)
            frame_index = list(range(T))
        else:
            clip = x.permute(1, 0, 2, 3).unsqueeze(0)       # (1, 3, T, H, W)
            feats = self._features(clip).squeeze(0).permute(1, 0, 2, 3)  # (C, T', h, w) → (T', C, h, w)
            frame_index = list(range(0, T, self.temporal_stride))       # first frame of each tubelet
        return feats, frame_index

    def _extra_meta(self):
        return dict(temporal_mode=self.temporal_mode, norm=self.norm)


# ---------------------------------------------------------------------------
# DINOv2 (timm, via the repo wrapper)
# ---------------------------------------------------------------------------

@register("dinov2")
class DINOv2Adapter(BackboneAdapter):
    """
    DINOv2 through anyup.backbone.PretrainedViTWrapper (timm, pretrained=True).
    Per-frame image backbone: all T frames in one batch → T' = T. Images are T = 1.
    """

    VARIANTS = {
        "small": "vit_small_patch14_dinov2.lvd142m",
        "base":  "vit_base_patch14_dinov2.lvd142m",
        "large": "vit_large_patch14_dinov2.lvd142m",
    }
    patch_size = 14
    temporal_stride = 1

    def __init__(self, device="cuda", input_size=224, guide_size=None, variant: str = "base"):
        super().__init__(device, input_size, guide_size)
        from anyup.backbone import PretrainedViTWrapper
        if variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {list(self.VARIANTS)}, got {variant!r}")
        self.variant = variant
        self.timm_name = self.VARIANTS[variant]
        self.model = self._freeze(PretrainedViTWrapper(self.timm_name))
        self.feat_dim = self.model.embed_dim
        if any(s % self.patch_size for s in self.input_size):
            raise ValueError(f"DINOv2 input_size must be a multiple of {self.patch_size}, got {self.input_size}")

    def _extract(self, x):
        feats, _cls = self.model(x)   # (T, C, h, w) — all T frames in ONE batch; chunk size (T02) sets memory; 16 × 16 at 224
        return feats, list(range(x.shape[0]))

    def _extra_meta(self):
        return dict(variant=self.variant, timm_name=self.timm_name)
