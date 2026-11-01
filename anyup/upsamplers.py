"""
Unified upsampler wrapper for AnyUp3D experiments (task T03).

One entry point, one class per upsampler, one call per T02 chunk:

    up  = UpsamplerLoader.load("anyup3d", ckpt_path=".../anyup3d_step6500.pth", device="cuda")
    for chunk in iter_backbone_chunks(bb, src):       # BackboneOutput: features (T, C, h, w), guide (T, 3, gH, gW)
        res = up(chunk)                               # output_size=None → guide size
        res.features                                  # (T, C, H, W) on the chunk's device
        res.meta                                      # chunk meta + upsampler keys

Inputs are used as given: raw backbone features (AnyUp normalizes internally) and the
ImageNet-normalized guide produced by the backbone adapter (T01). AnyUp3D gets one
chunk per call, with no context across chunk boundaries (T02 Q13).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import BackboneOutput
from .model import AnyUp
from .model_2d import AnyUp2D

Size = Union[int, Tuple[int, int]]

ANYUP2D_URL = "https://github.com/wimmerth/anyup/releases/download/checkpoint/anyup_paper.pth"

# AnyUp3D architecture as trained (Train_3D_full.ipynb TrainConfig; Exp 2 Cell 1).
# window_t is NOT stored in the checkpoint: a wrong value loads silently with strict=True.
ANYUP3D_ARCH = dict(
    input_dim=3,
    qk_dim=128,           # ↓ attention width; must match the checkpoint (strict load fails otherwise)
    kernel_size=1,
    kernel_size_lfu=5,    # mandatory 5 (LFU kernel); must match the checkpoint
    window_ratio=0.1,     # ↓ spatial attention window (fraction of the grid) → attention memory; not in the checkpoint
    window_t=4,           # ↓ temporal attention window (frames) → attention memory; not in the checkpoint, training used 4
    num_heads=4,          # must divide qk_dim
    t_k=3,                # temporal conv kernel; must match the checkpoint
)


# ---------------------------------------------------------------------------
# Output object
# ---------------------------------------------------------------------------

@dataclass
class UpsampleOutput:
    """
    features : (T, C, H, W) float32, upsampled features, on the input chunk's device
    meta     : copy of the chunk's meta + upsampler, output_size, upsampler-specific keys
    """
    features: torch.Tensor
    meta: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, Type["Upsampler"]] = {}


def register_upsampler(name: str) -> Callable[[Type["Upsampler"]], Type["Upsampler"]]:
    def deco(cls):
        if name in _REGISTRY:
            raise ValueError(f"Upsampler {name!r} already registered")
        _REGISTRY[name] = cls
        cls.name = name
        return cls
    return deco


class UpsamplerLoader:
    """Entry point: ``UpsamplerLoader.load(name, **kwargs)``."""

    @staticmethod
    def load(name: str, **kwargs) -> "Upsampler":
        if name not in _REGISTRY:
            raise ValueError(f"Unknown upsampler {name!r}. Available: {UpsamplerLoader.available()}")
        return _REGISTRY[name](**kwargs)

    @staticmethod
    def available() -> List[str]:
        return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Checkpoint loading (AnyUp3D)
# ---------------------------------------------------------------------------

class TStage:
    """
    Stub for the T-curriculum stage pickled inside the run_01/run_02 checkpoints
    (Train_3D_full.ipynb, saved as ``__main__.TStage`` inside ``cfg["t_schedule"]``).
    Allowlisted for ``torch.load(weights_only=True)``; it carries data only.
    """

    def __init__(self, t=None, start_step=None, batch_size=None):
        self.t = t
        self.start_step = start_step
        self.batch_size = batch_size

    def __repr__(self):
        return f"TStage(t={self.t}, start_step={self.start_step}, batch_size={self.batch_size})"


TStage.__module__ = "__main__"   # must match the pickled global name "__main__.TStage"


def load_anyup3d_checkpoint(path: str) -> Dict:
    """Safe load: weights_only=True, only tensors/primitives + the TStage stub; no code runs from the file."""
    with torch.serialization.safe_globals([TStage]):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict) or "anyup" not in ckpt:
        keys = list(ckpt) if isinstance(ckpt, dict) else type(ckpt).__name__
        raise KeyError(f"checkpoint {path!r} has no 'anyup' state dict; keys: {keys}")
    return ckpt


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

def _to_hw(size: Size) -> Tuple[int, int]:
    return (size, size) if isinstance(size, int) else tuple(size)


class Upsampler(nn.Module):
    """
    Subclasses implement ``_upsample(feats, guide, out_hw) -> (T, C, H, W)`` with
    feats (T, C, h, w) and guide (T, 3, gH, gW) already on ``self.device``.
    """

    name: str = ""

    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device)

    def _freeze(self, model: nn.Module) -> nn.Module:
        model = model.to(self.device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    def _upsample(self, feats: torch.Tensor, guide: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
        raise NotImplementedError

    def _extra_meta(self) -> Dict:
        return {}

    @torch.no_grad()
    def forward(self, chunk: BackboneOutput, output_size: Optional[Size] = None) -> UpsampleOutput:
        feats, guide = chunk.features, chunk.guide
        if feats.ndim != 4 or guide.ndim != 4 or guide.shape[1] != 3 or feats.shape[0] != guide.shape[0]:
            raise ValueError(f"expected features (T, C, h, w) and guide (T, 3, gH, gW) with the same T, "
                             f"got {tuple(feats.shape)} and {tuple(guide.shape)}")
        in_device = feats.device
        # ↓ output resolution → memory/time of every upsampler; default = guide size (T01 guide_size).
        #   Dependent: AnyUp3D q_chunk_size (reduce it when this grows).
        out_hw = _to_hw(output_size) if output_size is not None else tuple(guide.shape[-2:])

        out = self._upsample(feats.to(self.device, torch.float32),
                             guide.to(self.device, torch.float32), out_hw)       # (T, C, H, W)

        meta = dict(chunk.meta)
        meta.update(upsampler=self.name, output_size=out_hw)
        meta.update(self._extra_meta())
        return UpsampleOutput(features=out.float().to(in_device).contiguous(), meta=meta)


# ---------------------------------------------------------------------------
# AnyUp3D
# ---------------------------------------------------------------------------

@register_upsampler("anyup3d")
class AnyUp3DUpsampler(Upsampler):
    """One AnyUp3D call per chunk (all T feature frames together)."""

    def __init__(self, ckpt_path: str, device: str = "cuda", q_chunk_size: Optional[int] = None, **arch):
        super().__init__(device)
        unknown = set(arch) - set(ANYUP3D_ARCH)
        if unknown:
            raise TypeError(f"unknown AnyUp3D arch args {sorted(unknown)}; allowed: {sorted(ANYUP3D_ARCH)}")
        self.arch = {**ANYUP3D_ARCH, **arch}
        # ↓ cross-attention query chunk (None = all queries at once) → peak memory.
        #   Dependent: T02 chunk_size (T per call) and output_size; reduce this when either grows.
        self.q_chunk_size = q_chunk_size

        ckpt = load_anyup3d_checkpoint(ckpt_path)
        model = AnyUp(**self.arch)
        model.load_state_dict(ckpt["anyup"], strict=True)       # raises on any missing/unexpected key
        self.model = self._freeze(model)
        self.ckpt_path = str(ckpt_path)
        self.ckpt_step = ckpt.get("global_step")

    def _upsample(self, feats, guide, out_hw):
        feat_vol = feats.permute(1, 0, 2, 3).unsqueeze(0)      # (T, C, h, w)    → (1, C, T, h, w)
        guide_vol = guide.permute(1, 0, 2, 3).unsqueeze(0)     # (T, 3, gH, gW)  → (1, 3, T, gH, gW)
        out = self.model(guide_vol, feat_vol, output_size=out_hw,
                         q_chunk_size=self.q_chunk_size)       # guide first → (1, C, T, H, W)
        return out.squeeze(0).permute(1, 0, 2, 3)              # (1, C, T, H, W) → (T, C, H, W)

    def _extra_meta(self):
        return dict(ckpt_path=self.ckpt_path, ckpt_step=self.ckpt_step,
                    q_chunk_size=self.q_chunk_size, arch=dict(self.arch))


# ---------------------------------------------------------------------------
# AnyUp2D (per frame)
# ---------------------------------------------------------------------------

@register_upsampler("anyup2d")
class AnyUp2DUpsampler(Upsampler):
    """Published AnyUp (2D), one frame per call (as Exp 2 run_anyup2d)."""

    def __init__(self, device: str = "cuda", weights: Optional[str] = None):
        super().__init__(device)
        self.weights = weights or ANYUP2D_URL
        if weights is None:
            sd = torch.hub.load_state_dict_from_url(ANYUP2D_URL, map_location="cpu", weights_only=True)
        else:
            sd = torch.load(weights, map_location="cpu", weights_only=True)
        model = AnyUp2D()
        model.load_state_dict(sd, strict=True)
        self.model = self._freeze(model)

    def _upsample(self, feats, guide, out_hw):
        outs = []
        for t in range(feats.shape[0]):                                        # one frame per call
            outs.append(self.model(guide[t:t + 1], feats[t:t + 1], output_size=out_hw))   # (1, C, H, W)
        return torch.cat(outs, dim=0)                                          # (T, C, H, W)

    def _extra_meta(self):
        return dict(weights=self.weights)


# ---------------------------------------------------------------------------
# Bilinear
# ---------------------------------------------------------------------------

@register_upsampler("bilinear")
class BilinearUpsampler(Upsampler):
    """F.interpolate bilinear, align_corners=False (as Exp 3 upsample_bilinear); guide unused."""

    def _upsample(self, feats, guide, out_hw):
        return F.interpolate(feats, size=out_hw, mode="bilinear", align_corners=False)   # (T, C, h, w) → (T, C, H, W)
