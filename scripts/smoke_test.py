"""
scripts/smoke_test.py
Phase 7.1 — End-to-end smoke test.

Runs 10 training steps on purely synthetic data with debug=True.
Checks: no NaNs, no OOM, no shape mismatches, all losses compute and backprop.

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --device cpu      # for machines without CUDA
    python scripts/smoke_test.py --steps 20        # extend if desired
"""

import argparse
import math
import sys
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, ".")
from anyup.modules.cross_attention3d import CrossAttentionBlock3D
from anyup.utils.seed import set_seed


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic data factory
# ══════════════════════════════════════════════════════════════════════════════

def make_batch(B, C_feat, T, H, W, C_img=3, device="cuda"):
    """
    Returns a dict mirroring what the real DataLoader would produce.
    All values are random Gaussians — enough to verify shapes and gradients.

    Shapes
    ------
    rgb_hires  : (B, C_img, T, H,   W)     full-res RGB video
    rgb_lores  : (B, C_img, T, H//s, W//s) downsampled RGB for key branch
    feat_lores : (B, C_feat, T, H//s, W//s) GT low-res features (encoder output)
    feat_hires : (B, C_feat, T, H,   W)     GT high-res features (regression target)
    """
    s = 4                                    # ↓ spatial downscale factor — must match model config
    return {
        "rgb_hires":  torch.randn(B, C_img,  T, H,    W,    device=device),
        "rgb_lores":  torch.randn(B, C_img,  T, H//s, W//s, device=device),
        "feat_lores": torch.randn(B, C_feat, T, H//s, W//s, device=device),
        "feat_hires": torch.randn(B, C_feat, T, H,    W,    device=device),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Minimal forward model (wires CrossAttentionBlock3D directly)
# Replace this section with your real AnyUp3D once it is available.
# ══════════════════════════════════════════════════════════════════════════════

class _MinimalPipeline(nn.Module):
    """
    Lightweight stand-in for AnyUp3D that exercises the same code paths:
      q (high-res query)  ← small conv on rgb_hires projected to QK dim
      k (low-res key)     ← small conv on feat_lores projected to QK dim
      v (low-res value)   ← feat_lores (no projection; C_v = C_feat)
      output              ← CrossAttentionBlock3D → 1×1 conv to C_feat
    Swap this out for the real AnyUp3D forward once that class exists.
    """
    def __init__(self, C_img, C_feat, C_qk=64, num_heads=4,
                 window_ratio=0.15, window_t=1):
        super().__init__()
        # Query encoder stub (replaces LearnedFeatureUnification + image encoder)
        self.q_proj = nn.Conv3d(C_img, C_qk, kernel_size=1)   # ↓ increase C_qk if underfitting
        # Key encoder stub
        self.k_proj = nn.Conv3d(C_feat, C_qk, kernel_size=1)  # must match q_proj out channels
        # Core attention block
        self.attn = CrossAttentionBlock3D(
            qk_dim=C_qk,
            num_heads=num_heads,            # ↓ reduce if OOM
            window_ratio=window_ratio,      # ↓ reduce to save memory (also affects mask shape)
            window_t=window_t,              # None = no temporal windowing
        )
        # Output projection to feature space
        self.out_proj = nn.Conv3d(C_feat, C_feat, kernel_size=1)

    def forward(self, batch):
        q = self.q_proj(batch["rgb_hires"])    # (B, C_qk, T, H, W)
        k = self.k_proj(batch["feat_lores"])   # (B, C_qk, T, H//s, W//s)
        v = batch["feat_lores"]                # (B, C_feat, T, H//s, W//s)
        out = self.attn(q, k, v)               # (B, C_feat, T, H, W)
        out = self.out_proj(out)
        return out                             # (B, C_feat, T, H, W)


# ══════════════════════════════════════════════════════════════════════════════
# Loss functions
# ══════════════════════════════════════════════════════════════════════════════

def cosine_mse(pred, target):
    """L_reconstruction: element-wise cosine + MSE in feature space."""
    pred_n  = F.normalize(pred,   dim=1)
    target_n = F.normalize(target, dim=1)
    l_cos = (1 - (pred_n * target_n).sum(dim=1)).mean()
    l_mse = F.mse_loss(pred, target)
    return l_cos + l_mse


def input_consistency(pred, feat_lores):
    """
    L_input-consistency: downsampled prediction should match GT low-res features.
    Average-pools pred to lores resolution and compares.
    """
    _, _, T, H_lo, W_lo = feat_lores.shape
    pred_down = F.adaptive_avg_pool3d(pred, (T, H_lo, W_lo))  # ↑ T preserved; only H,W pooled
    return cosine_mse(pred_down, feat_lores)


def temporal_consistency(pred, lambda_t=0.01):
    """
    L_temporal-consistency: adjacent frames should be similar.
    Only applied when T > 1. lambda_t ↓ reduces temporal pressure early in training.
    """
    if pred.shape[2] < 2:       # T < 2 → skip (e.g. warmup stage)
        return pred.new_tensor(0.0)
    f_t   = pred[:, :, 1:, :, :]   # frames 1..T-1
    f_tm1 = pred[:, :, :-1, :, :]  # frames 0..T-2
    return lambda_t * cosine_mse(f_t, f_tm1)


def combined_loss(pred, batch, lambda_input=0.1, lambda_temporal=0.01):
    """
    Full combined loss: L_recon + λ1·L_input + λ3·L_temporal.
    Returns dict of individual components for logging.
    """
    l_recon   = cosine_mse(pred, batch["feat_hires"])
    l_input   = lambda_input    * input_consistency(pred, batch["feat_lores"])
    l_temporal = temporal_consistency(pred, lambda_t=lambda_temporal)
    total     = l_recon + l_input + l_temporal
    return {
        "total":    total,
        "recon":    l_recon,
        "input":    l_input,
        "temporal": l_temporal,
    }


# ══════════════════════════════════════════════════════════════════════════════
# NaN / shape checks
# ══════════════════════════════════════════════════════════════════════════════

def _check_no_nan(tensor, name):
    if torch.isnan(tensor).any():
        raise ValueError(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        raise ValueError(f"Inf detected in {name}")


def _check_shape(tensor, expected, name):
    if tuple(tensor.shape) != tuple(expected):
        raise ValueError(f"Shape mismatch for {name}: expected {expected}, got {list(tensor.shape)}")


def _check_grad(model, step):
    """Verify at least one parameter has a non-None, non-zero gradient."""
    any_grad = False
    for name, p in model.named_parameters():
        if p.grad is not None:
            _check_no_nan(p.grad, f"grad/{name}")
            any_grad = True
    if not any_grad:
        raise RuntimeError(f"Step {step}: no gradients found — backward may have been skipped")


# ══════════════════════════════════════════════════════════════════════════════
# Smoke test loop
# ══════════════════════════════════════════════════════════════════════════════

def run_smoke_test(steps=10, device="cuda", seed=42):
    print("=" * 60)
    print("AnyUp3D — Phase 7.1 Smoke Test")
    print("=" * 60)

    set_seed(seed)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Device : {dev}")
    print(f"Seed   : {seed}")
    print(f"Steps  : {steps}")
    print()

    # ── Config (all compute-cost knobs are labelled) ───────────────────────
    B      = 2       # ↓ reduce to save memory
    C_feat = 64      # feature channel dim — must match encoder output
    C_qk   = 64      # QK projection dim  — must match C_feat here for _MinimalPipeline
    T      = 4       # ↓ reduce to save memory — also affects mask shape and temporal loss
    H      = 56      # ↓ reduce to save memory — also affects H//s (line below)
    W      = 56      # ↓ reduce to save memory — also affects W//s (line below)
    # Note: H and W must be divisible by the downscale factor s=4 (defined in make_batch)

    model = _MinimalPipeline(
        C_img=3, C_feat=C_feat, C_qk=C_qk, num_heads=4,
        window_ratio=0.15, window_t=1,     # ↓ window_ratio to save memory
    ).to(dev).train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)

    print(f"Parameters : {sum(p.numel() for p in model.parameters()):,}")
    print()

    failures = []
    loss_log = []

    for step in range(steps):
        try:
            # ── Data ──────────────────────────────────────────────────────
            batch = make_batch(B, C_feat, T, H, W, device=dev)

            # ── Forward ───────────────────────────────────────────────────
            optimizer.zero_grad()
            pred = model(batch)                 # (B, C_feat, T, H, W)

            # Shape check
            _check_shape(pred, (B, C_feat, T, H, W), "model output")

            # NaN check on forward activations
            _check_no_nan(pred, "model output")

            # ── Losses ────────────────────────────────────────────────────
            losses = combined_loss(pred, batch)

            for k, v in losses.items():
                _check_no_nan(v, f"loss/{k}")

            # ── Backward ──────────────────────────────────────────────────
            losses["total"].backward()

            # Gradient check
            _check_grad(model, step)

            # Gradient clipping (mirrors production training)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_val = losses["total"].item()
            loss_log.append(total_val)

            print(
                f"  Step {step:02d} | "
                f"total={total_val:.4f}  "
                f"recon={losses['recon'].item():.4f}  "
                f"input={losses['input'].item():.4f}  "
                f"temporal={losses['temporal'].item():.4f}  "
                f"✅"
            )

        except Exception as e:
            msg = f"Step {step}: {type(e).__name__}: {e}"
            failures.append(msg)
            print(f"  Step {step:02d} | ❌ {msg}")
            traceback.print_exc()

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    if failures:
        print(f"SMOKE TEST FAILED — {len(failures)} error(s):")
        for f in failures:
            print(f"  • {f}")
        sys.exit(1)
    else:
        print(f"SMOKE TEST PASSED — {steps} steps, no NaNs, no shape errors")
        print(f"  Loss range : {min(loss_log):.4f} → {max(loss_log):.4f}")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",  type=int,   default=10)
    parser.add_argument("--device", type=str,   default="cuda")
    parser.add_argument("--seed",   type=int,   default=42)
    args = parser.parse_args()
    run_smoke_test(steps=args.steps, device=args.device, seed=args.seed)