"""
scripts/load_2d_weights.py
--------------------------
Load a pretrained 2D AnyUp checkpoint into an AnyUp3D model.

Shape-adaptation rules
-----------------------
1. Conv2d → Conv3d  (SpatialReflectConv3d, ResBlock3D internals)
   2D weight : (out, in, kH, kW)
   3D weight : (out, in, kT, kH, kW)
   Strategy  : center-init — zero the 3D tensor, copy 2D into the
               middle temporal slice [kT//2].

2. LearnedFeatureUnification (LFU) Conv weight
   Same center-init as (1).

3. RoPE freqs
   2D weight : (2, dim)   — rows [x, y]
   3D weight : (3, dim)   — rows [z, x, y]
   Strategy  : copy 2D rows → 3D rows 1 and 2; zero row 0 (temporal).

4. Key remapping — wrapper attribute insertion
   SpatialReflectConv3d and ResBlock3D wrap their conv as self.conv,
   inserting an extra ".conv." segment into every parameter path.

   Example:
     2D key : image_encoder.0.weight
     3D key : image_encoder.0.conv.weight

   We build a lookup table that maps each 3D key → its 2D counterpart
   by stripping known wrapper segments before doing any shape matching.

5. New-only 3D params (not present in the 2D checkpoint after remapping)
   Strategy  : zero-init (logged explicitly).

Usage
-----
    python scripts/load_2d_weights.py \
        --checkpoint anyup_multi_backbone.pth \
        --output     anyup3d_init.pth \
        [--t_k 3]
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# Key remapping
# ──────────────────────────────────────────────────────────────────────────────

def _remap_3d_key_to_2d(key_3d: str) -> str:
    """
    Convert a 3D model parameter key to its expected 2D checkpoint key.

    Two kinds of name changes exist between the 2D and 3D models:

    1. SpatialReflectConv3d and ResBlock3D store their inner conv as self.conv,
       adding '.conv.' into every path.  Strip it to recover the 2D name.

    2. CrossAttentionBlock3D renamed the query context conv from conv2d → conv3d
       to reflect that it is now a Conv3d layer.  Reverse that rename.

    Examples
    --------
    image_encoder.0.conv.weight         → image_encoder.0.weight
    image_encoder.1.block.2.conv.weight → image_encoder.1.block.2.weight
    cross_decode.conv3d.weight          → cross_decode.conv2d.weight
    rope.freqs                          → rope.freqs   (unchanged)
    """
    return (
        key_3d
        .replace(".conv.weight", ".weight")
        .replace(".conv.bias",   ".bias")
        .replace(".conv3d.",     ".conv2d.")
    )


# ──────────────────────────────────────────────────────────────────────────────
# Shape adapters
# ──────────────────────────────────────────────────────────────────────────────

def _adapt_conv_weight(src: torch.Tensor, dst_shape: torch.Size) -> torch.Tensor:
    """
    Center-initialize a Conv3d weight from a Conv2d weight.

    src : (out, in, kH, kW)
    dst : (out, in, kT, kH, kW)

    The 2D kernel is placed at the temporal center; all other slices are zero.
    For kT=1 this is a trivial unsqueeze(2).
    """
    assert src.dim() == 4, f"Expected 4D Conv2d weight, got {src.shape}"
    assert len(dst_shape) == 5, f"Expected 5D Conv3d shape, got {dst_shape}"

    out_ch, in_ch, k_t, kH, kW = dst_shape
    assert src.shape == (out_ch, in_ch, kH, kW), (
        f"Spatial dims mismatch: src {tuple(src.shape)} vs dst {tuple(dst_shape)}"
    )

    dst = torch.zeros(dst_shape, dtype=src.dtype)
    center = k_t // 2      # middle temporal slice — all weight lives here
    dst[:, :, center, :, :] = src
    return dst


def _adapt_rope_freqs(src: torch.Tensor, dst_shape: torch.Size) -> torch.Tensor:
    """
    Expand RoPE freqs (2, dim) → (3, dim).

    Row layout
      2D : [x_freqs, y_freqs]
      3D : [z_freqs, x_freqs, y_freqs]

    Row 0 (z / temporal) is zeroed — at T=1 z-coords are 0 so the row
    contributes nothing, giving exact 2D equivalence.
    """
    assert src.shape[0] == 2 and dst_shape[0] == 3
    assert src.shape[1] == dst_shape[1], (
        f"RoPE dim mismatch: {src.shape[1]} vs {dst_shape[1]}"
    )
    dst = torch.zeros(dst_shape, dtype=src.dtype)
    dst[1] = src[0]   # x row
    dst[2] = src[1]   # y row
    # dst[0] stays zero  (temporal / z row)
    return dst


# ──────────────────────────────────────────────────────────────────────────────
# Status tags
# ──────────────────────────────────────────────────────────────────────────────

class S:
    DIRECT   = "direct"    # exact shape match, copied as-is
    CONV_3D  = "conv3d"    # Conv2d → Conv3d center-init
    ROPE     = "rope"      # RoPE (2,d) → (3,d)
    ZERO     = "zero"      # new 3D-only param, zero-initialized
    SKIPPED  = "skipped"   # 2D key not present in 3D model
    MISMATCH = "mismatch"  # unhandled shape difference


# ──────────────────────────────────────────────────────────────────────────────
# Core builder
# ──────────────────────────────────────────────────────────────────────────────

def build_3d_state_dict(
    ckpt_2d: Dict[str, torch.Tensor],
    model_3d: nn.Module,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    """
    Build a complete state dict for model_3d from a 2D checkpoint.

    Returns
    -------
    adapted_sd : ready for model_3d.load_state_dict(strict=True)
    status     : key → S.*  (for the summary printout)
    """
    model_sd   = model_3d.state_dict()
    adapted_sd : Dict[str, torch.Tensor] = {}
    status     : Dict[str, str] = {}

    for key_3d, dst_tensor in model_sd.items():
        dst_shape = dst_tensor.shape

        # ── Step 1: resolve the matching 2D key ───────────────────────────
        key_2d = _remap_3d_key_to_2d(key_3d)   # strip wrapper segments

        src: Optional[torch.Tensor] = ckpt_2d.get(key_2d)
        if src is None and key_2d != key_3d:
            # Also try the raw 3D key in case the ckpt was already adapted
            src = ckpt_2d.get(key_3d)

        if src is None:
            # Genuinely new 3D-only parameter
            adapted_sd[key_3d] = torch.zeros_like(dst_tensor)
            status[key_3d] = S.ZERO
            continue

        # ── Step 2: adapt the tensor to the 3D shape ──────────────────────

        # Exact match — no shape change needed
        if src.shape == dst_shape:
            adapted_sd[key_3d] = src.clone()
            status[key_3d] = S.DIRECT
            continue

        # RoPE freqs: (2, dim) → (3, dim)
        if key_3d.endswith("freqs") and src.dim() == 2 and src.shape[0] == 2 and dst_shape[0] == 3:
            adapted_sd[key_3d] = _adapt_rope_freqs(src, dst_shape)
            status[key_3d] = S.ROPE
            continue

        # Conv2d → Conv3d: (out, in, kH, kW) → (out, in, kT, kH, kW)
        if src.dim() == 4 and len(dst_shape) == 5:
            try:
                adapted_sd[key_3d] = _adapt_conv_weight(src, dst_shape)
                status[key_3d] = S.CONV_3D
            except AssertionError as e:
                print(f"  [ERROR] {key_3d}: {e}", file=sys.stderr)
                adapted_sd[key_3d] = dst_tensor.clone()
                status[key_3d] = S.MISMATCH
            continue

        # Unhandled shape mismatch
        print(
            f"  [WARN] {key_3d}: src {tuple(src.shape)} → dst {tuple(dst_shape)} "
            "not handled; keeping model default.",
            file=sys.stderr,
        )
        adapted_sd[key_3d] = dst_tensor.clone()
        status[key_3d] = S.MISMATCH

    # Track 2D-only keys that have no counterpart in the 3D model
    all_2d_keys_used = {_remap_3d_key_to_2d(k) for k in model_sd}
    for key in ckpt_2d:
        if key not in all_2d_keys_used and key not in model_sd:
            status[key] = S.SKIPPED

    return adapted_sd, status


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────

def _print_summary(status: Dict[str, str]) -> None:
    from collections import Counter
    counts = Counter(status.values())

    print("\n" + "=" * 72)
    print("  Weight loading summary")
    print("=" * 72)
    print(f"  {'direct':12s} {counts[S.DIRECT]:5d}  (shape-matched, copied as-is)")
    print(f"  {'conv3d':12s} {counts[S.CONV_3D]:5d}  (Conv2d→Conv3d center-init)")
    print(f"  {'rope':12s} {counts[S.ROPE]:5d}  (RoPE (2,d)→(3,d), z-row zeroed)")
    print(f"  {'zero':12s} {counts[S.ZERO]:5d}  (new 3D-only param, zero-initialized)")
    print(f"  {'skipped':12s} {counts[S.SKIPPED]:5d}  (2D-only key, ignored)")
    print(f"  {'mismatch':12s} {counts[S.MISMATCH]:5d}  (unhandled — kept model default)")
    print("=" * 72)

    if counts[S.MISMATCH] > 0:
        print("\n  [!] Mismatch keys (need manual handling):")
        for k, v in status.items():
            if v == S.MISMATCH:
                print(f"      {k}")

    if counts[S.ZERO] > 0:
        print("\n  [i] Zero-initialized (genuinely new 3D-only params):")
        for k, v in status.items():
            if v == S.ZERO:
                print(f"      {k}")

    print()


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def load_2d_weights_into_3d(
    checkpoint_path: str,
    output_path: str = None,
    t_k: int = 3,
    model_kwargs: dict = None,
) -> nn.Module:
    from anyup.model import AnyUp

    model_kwargs = model_kwargs or {}
    model = AnyUp(t_k=t_k, **model_kwargs)

    print(f"Loading 2D checkpoint: {checkpoint_path}")
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    if isinstance(raw, dict):
        for key in ("model", "state_dict", "weights", "params"):
            if key in raw and isinstance(raw[key], dict):
                ckpt_2d = raw[key]
                break
        else:
            ckpt_2d = raw   # bare state dict
    else:
        raise ValueError(f"Unexpected checkpoint type: {type(raw)}")

    print(f"  2D checkpoint has {len(ckpt_2d)} tensors")
    print(f"  3D model has      {len(model.state_dict())} tensors")

    adapted_sd, status = build_3d_state_dict(ckpt_2d, model)

    missing, unexpected = model.load_state_dict(adapted_sd, strict=True)
    assert not missing,    f"Missing keys: {missing}"
    assert not unexpected, f"Unexpected keys: {unexpected}"

    _print_summary(status)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_path)
        print(f"Saved adapted 3D checkpoint to: {output_path}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Load 2D AnyUp weights into AnyUp3D")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to 2D AnyUp .pth file")
    parser.add_argument("--output", required=True,
                        help="Where to write the adapted 3D checkpoint")
    parser.add_argument("--t_k", type=int, default=3,
                        help="Temporal kernel size for Conv3d layers (default: 3)")
    args = parser.parse_args()
    load_2d_weights_into_3d(args.checkpoint, args.output, args.t_k)


if __name__ == "__main__":
    main()