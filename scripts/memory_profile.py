"""
scripts/memory_profile.py
Phase 7.2 — GPU memory profiling at T=4, T=8, T=16.

Measures peak GPU memory for each T value (forward + backward).
If T=16 is tight, enables gradient checkpointing on CrossAttentionBlock3D
and reports the saving.

Usage:
    python scripts/memory_profile.py
    python scripts/memory_profile.py --budget_gb 20   # OOM threshold in GB
    python scripts/memory_profile.py --batch_size 2
"""

import argparse
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as torch_checkpoint

sys.path.insert(0, ".")
from anyup.modules.cross_attention3d import CrossAttentionBlock3D
from anyup.utils.seed import set_seed


# ══════════════════════════════════════════════════════════════════════════════
# Gradient-checkpointed wrapper for CrossAttentionBlock3D
# ══════════════════════════════════════════════════════════════════════════════

class CrossAttentionBlock3D_GradCkpt(CrossAttentionBlock3D):
    """
    Drop-in replacement for CrossAttentionBlock3D with gradient checkpointing.
    Trades recomputation at backward for ~30-50% activation memory reduction.
    Enable when T=16 causes OOM at your target batch size.

    Usage: swap CrossAttentionBlock3D → CrossAttentionBlock3D_GradCkpt in AnyUp3D.__init__
    """

    def forward(self, q, k, v, q_chunk_size=None, store_attn=None, **kwargs):
        # Wrap the parent forward in checkpoint.
        # use_reentrant=False avoids autograd graph corruption with non-leaf inputs.
        def _fwd(q_, k_, v_):
            return super(CrossAttentionBlock3D_GradCkpt, self).forward(
                q_, k_, v_, q_chunk_size=q_chunk_size, store_attn=store_attn, **kwargs
            )
        return torch_checkpoint.checkpoint(_fwd, q, k, v, use_reentrant=False)


# ══════════════════════════════════════════════════════════════════════════════
# Minimal pipeline (same as smoke_test, parameterised on attn_cls)
# ══════════════════════════════════════════════════════════════════════════════

class _PipelineForProfiling(nn.Module):
    def __init__(self, C_img, C_feat, C_qk, num_heads, window_ratio, window_t, attn_cls):
        super().__init__()
        self.q_proj   = nn.Conv3d(C_img,   C_qk,   kernel_size=1)
        self.k_proj   = nn.Conv3d(C_feat,  C_qk,   kernel_size=1)
        self.attn     = attn_cls(
            qk_dim=C_qk,
            num_heads=num_heads,        # ↓ reduce if OOM
            window_ratio=window_ratio,  # ↓ reduce to save memory (also affects mask shape)
            window_t=window_t,
        )
        self.out_proj = nn.Conv3d(C_feat, C_feat, kernel_size=1)

    def forward(self, rgb_hires, feat_lores):
        q   = self.q_proj(rgb_hires)   # (B, C_qk,  T, H,    W)
        k   = self.k_proj(feat_lores)  # (B, C_qk,  T, H//s, W//s)
        v   = feat_lores               # (B, C_feat, T, H//s, W//s)
        out = self.attn(q, k, v)       # (B, C_feat, T, H,    W)
        return self.out_proj(out)


# ══════════════════════════════════════════════════════════════════════════════
# Single-T profiling run
# ══════════════════════════════════════════════════════════════════════════════

def _bytes_to_gb(n):
    return n / 1024 ** 3


def profile_one(T, B, C_feat, C_qk, H, W, num_heads, window_ratio, window_t,
                attn_cls, device):
    """Returns peak allocated GPU memory in GB for one forward+backward pass."""
    set_seed(0)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    s = 4   # ↓ spatial downscale — must match model config; also update H//s, W//s below

    model = _PipelineForProfiling(
        C_img=3, C_feat=C_feat, C_qk=C_qk,
        num_heads=num_heads, window_ratio=window_ratio,
        window_t=window_t, attn_cls=attn_cls,
    ).to(device).train()

    rgb_hires  = torch.randn(B, 3,      T, H,    W,    device=device)   # ↑ H,W → ↑ memory
    feat_lores = torch.randn(B, C_feat, T, H//s, W//s, device=device)   # depends on H,W,s above

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    optimizer.zero_grad()

    pred = model(rgb_hires, feat_lores)

    # Simple surrogate loss
    target = torch.randn_like(pred)
    loss   = F.mse_loss(pred, target)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    peak_gb = _bytes_to_gb(torch.cuda.max_memory_allocated(device))

    del model, rgb_hires, feat_lores, pred, loss
    torch.cuda.empty_cache()

    return peak_gb


# ══════════════════════════════════════════════════════════════════════════════
# Main sweep
# ══════════════════════════════════════════════════════════════════════════════

def run_memory_profile(budget_gb=20.0, batch_size=2):
    print("=" * 60)
    print("AnyUp3D — Phase 7.2 Memory Profiling")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("No CUDA device found — memory profiling requires a GPU.")
        sys.exit(0)

    device = torch.device("cuda")
    total_gb = _bytes_to_gb(torch.cuda.get_device_properties(device).total_memory)
    print(f"GPU          : {torch.cuda.get_device_name(device)}")
    print(f"Total VRAM   : {total_gb:.1f} GB")
    print(f"Budget limit : {budget_gb:.1f} GB")
    print(f"Batch size   : {batch_size}")
    print()

    # ── Shared config (change these to match your real model) ─────────────
    C_feat       = 64     # feature channel dim
    C_qk         = 64     # QK projection dim — must match C_feat for _PipelineForProfiling
    H            = 56     # ↓ reduce to save memory — also update W and the s//factor comment
    W            = 56     # ↓ reduce to save memory — must equal H for square crops
    num_heads    = 4      # ↓ reduce if OOM
    window_ratio = 0.15   # ↓ reduce to save memory (also affects mask shape)
    window_t     = 1      # None = no temporal window

    T_values = [4, 8, 16]  # curriculum stages — must match T-curriculum config

    results = {}   # T → {"plain": gb, "ckpt": gb}

    for T in T_values:
        row = {}
        label_plain = f"T={T:2d} plain"
        try:
            gb = profile_one(
                T, batch_size, C_feat, C_qk, H, W,
                num_heads, window_ratio, window_t,
                CrossAttentionBlock3D, device,
            )
            row["plain"] = gb
            oom = gb > budget_gb
            flag = f"{'⚠️  OOM RISK' if oom else '✅'}"
            print(f"  {label_plain} : {gb:.2f} GB  {flag}")
        except torch.cuda.OutOfMemoryError:
            row["plain"] = None
            print(f"  {label_plain} : 💥 OOM")

        # Run checkpointed version if plain was tight or OOM'd
        if row["plain"] is None or row["plain"] > budget_gb * 0.85:
            label_ckpt = f"T={T:2d} ckpt "
            try:
                gb_ckpt = profile_one(
                    T, batch_size, C_feat, C_qk, H, W,
                    num_heads, window_ratio, window_t,
                    CrossAttentionBlock3D_GradCkpt, device,
                )
                row["ckpt"] = gb_ckpt
                saved = (row["plain"] or 0) - gb_ckpt
                print(f"  {label_ckpt} : {gb_ckpt:.2f} GB  ✅  (saved {saved:.2f} GB vs plain)")
            except torch.cuda.OutOfMemoryError:
                row["ckpt"] = None
                print(f"  {label_ckpt} : 💥 OOM even with checkpointing — reduce B, H, W, or window_ratio")

        results[T] = row

    # ── Summary + recommendation ──────────────────────────────────────────
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for T, row in results.items():
        plain = f"{row['plain']:.2f} GB" if row.get("plain") is not None else "OOM"
        ckpt  = f"{row['ckpt']:.2f} GB"  if row.get("ckpt")  is not None else ("N/A" if "ckpt" not in row else "OOM")
        print(f"  T={T:2d}  plain={plain:10s}  ckpt={ckpt}")

    print()
    print("Recommendation:")
    t16 = results.get(16, {})
    if t16.get("plain") is not None and t16["plain"] <= budget_gb:
        print("  ✅ T=16 fits in budget without checkpointing. No changes needed.")
    elif t16.get("ckpt") is not None and t16["ckpt"] <= budget_gb:
        print("  ⚠️  Use CrossAttentionBlock3D_GradCkpt for T=16.")
        print("     In AnyUp3D.__init__, replace:")
        print("       CrossAttentionBlock3D(...)  →  CrossAttentionBlock3D_GradCkpt(...)")
        print("     (defined in scripts/memory_profile.py — copy to anyup/modules/)")
    else:
        print("  ❌ T=16 OOMs even with checkpointing.")
        print("     Options: reduce B (batch_size), H/W, window_ratio, or C_qk.")
        print("     Check which lines control these in this script header.")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget_gb",  type=float, default=20.0,
                        help="OOM risk threshold in GB (default: 20.0)")
    parser.add_argument("--batch_size", type=int,   default=2,
                        help="Batch size for profiling (default: 2)")
    args = parser.parse_args()
    run_memory_profile(budget_gb=args.budget_gb, batch_size=args.batch_size)