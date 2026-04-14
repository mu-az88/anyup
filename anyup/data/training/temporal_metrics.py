"""
Temporal coherence metrics for AnyUp3D evaluation.

Main metric: temporal_feature_variance
  For each spatial position (h, w), measure the variance of upsampled features
  across T frames. Lower = more temporally stable = better.

Baseline comparison: apply AnyUp2D independently per frame (no temporal context),
  compute the same metric — the 3D model should score lower.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


@torch.no_grad()
def temporal_feature_variance(
    features: Tensor,               # (B, T, H, W, C) upsampled features
    rgb_frames: Tensor | None = None,  # (B, T, H, W, 3) optional — for static scene gating
    static_threshold: float = 0.02, # ↓ gate sensitivity: lower = stricter definition of "static"
) -> dict:
    """
    Computes per-position feature variance across T frames.

    If rgb_frames is provided, applies a static scene gate: only spatial positions
    where consecutive RGB frames are similar (mean diff < static_threshold) are
    included. This avoids penalizing the model for fast motion or scene cuts.

    Returns a dict with scalar summary metrics.
    """
    B, T, H, W, C = features.shape  # ↓ T is the main cost driver here

    if T < 2:
        raise ValueError("temporal_feature_variance requires T >= 2")

    # ── Optional static scene gate ─────────────────────────────────────────────
    if rgb_frames is not None:
        # Compute mean absolute RGB difference between consecutive frames
        # rgb_frames: (B, T, H, W, 3)
        rgb_diff = (rgb_frames[:, 1:] - rgb_frames[:, :-1]).abs().mean(dim=-1)  # (B, T-1, H, W)
        # Gate: True where the scene is approximately static
        static_mask = rgb_diff < static_threshold   # (B, T-1, H, W) bool
        # Expand to cover all T frames: a position is static if ALL consecutive
        # pairs around it are static — conservative but clean
        frame_static = torch.cat([
            static_mask[:, :1],                    # first frame uses first pair
            (static_mask[:, :-1] & static_mask[:, 1:]),  # middle frames
            static_mask[:, -1:],                   # last frame uses last pair
        ], dim=1)                                  # (B, T, H, W)
    else:
        frame_static = torch.ones(B, T, H, W, dtype=torch.bool, device=features.device)

    # ── Per-position variance across T ────────────────────────────────────────
    # features: (B, T, H, W, C)
    # Variance over the T dimension, then mean over C
    feat_var = features.var(dim=1)                 # (B, H, W, C) — variance across frames
    feat_var = feat_var.mean(dim=-1)               # (B, H, W)   — mean over feature dim

    # Apply static mask — use the AND across T frames at each position
    # frame_static: (B, T, H, W) → collapse T → (B, H, W)
    position_static = frame_static.all(dim=1)      # (B, H, W) — True where all frames are static

    # ── Summary stats ─────────────────────────────────────────────────────────
    n_static = position_static.sum().item()        # number of valid (static) positions

    if n_static == 0:
        # No static positions found — threshold may be too strict, or clip has lots of motion
        return {
            "mean_var":         float("nan"),
            "median_var":       float("nan"),
            "n_static_positions": 0,
            "static_fraction":  0.0,
        }

    gated_var = feat_var[position_static]          # (N_static,) — only static positions

    return {
        "mean_var":             gated_var.mean().item(),
        "median_var":           gated_var.median().item(),
        "n_static_positions":   int(n_static),
        "static_fraction":      n_static / (B * H * W),  # fraction of positions that were static
    }


@torch.no_grad()
def temporal_cosine_drift(
    features: Tensor,               # (B, T, H, W, C)
) -> dict:
    """
    Complementary metric: mean cosine distance between adjacent frames.
    Measures how much features drift frame-to-frame.
    Lower = more temporally coherent.

    Unlike variance, this is directional — it captures drift rather than scatter.
    """
    B, T, H, W, C = features.shape

    if T < 2:
        raise ValueError("temporal_cosine_drift requires T >= 2")

    # Adjacent frame pairs: f_t and f_{t+1}
    f_curr = features[:, :-1].reshape(-1, C)       # (B*(T-1)*H*W, C)  ↓ sequence length scales with T
    f_next = features[:, 1: ].reshape(-1, C)       # (B*(T-1)*H*W, C)

    cos_sim = F.cosine_similarity(f_curr, f_next, dim=-1)  # (B*(T-1)*H*W,)
    cos_drift = 1.0 - cos_sim                      # 0 = identical, 2 = opposite

    return {
        "mean_cosine_drift":   cos_drift.mean().item(),
        "median_cosine_drift": cos_drift.median().item(),
    }


@torch.no_grad()
def compare_3d_vs_2d_baseline(
    model_3d,                        # AnyUp3D — processes (B, T, H, W, 3) jointly
    model_2d,                        # AnyUp2D — processes (B, H, W, 3) per frame
    frames: Tensor,                  # (B, T, H, W, 3) input RGB
    device: torch.device,
    static_threshold: float = 0.02,
) -> dict:
    """
    Runs both models on the same clip and computes temporal coherence metrics
    for each. Returns a comparison dict — 3D model should have lower variance
    and lower cosine drift than the 2D baseline.
    """
    frames = frames.to(device)
    B, T, H, W, _ = frames.shape    # ↓ T and H,W are the main memory cost drivers

    # ── 3D model forward ──────────────────────────────────────────────────────
    feat_3d = model_3d(frames)       # (B, T, H', W', C)

    # ── 2D baseline: run per frame independently ───────────────────────────────
    feat_2d_frames = []
    for t in range(T):               # ↓ loop over T — if T is large, this is the bottleneck
        f = model_2d(frames[:, t])   # (B, H', W', C) — single frame
        feat_2d_frames.append(f)
    feat_2d = torch.stack(feat_2d_frames, dim=1)  # (B, T, H', W', C)

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics_3d  = temporal_feature_variance(feat_3d,  frames, static_threshold)
    metrics_2d  = temporal_feature_variance(feat_2d,  frames, static_threshold)
    drift_3d    = temporal_cosine_drift(feat_3d)
    drift_2d    = temporal_cosine_drift(feat_2d)

    return {
        "3d": {**metrics_3d, **drift_3d},
        "2d": {**metrics_2d, **drift_2d},
        "improvement": {
            "var_reduction":   metrics_2d["mean_var"]         - metrics_3d["mean_var"],
            "drift_reduction": drift_2d["mean_cosine_drift"]  - drift_3d["mean_cosine_drift"],
        },
    }