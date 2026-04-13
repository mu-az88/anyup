"""
scripts/verify_reproducibility.py
Phase 7.3 — Verify two runs with the same seed produce identical loss curves.

Runs the first 10 training steps twice from the same seed.
Asserts that every loss value is bit-for-bit identical across both runs.

Usage:
    python scripts/verify_reproducibility.py
    python scripts/verify_reproducibility.py --seed 1234 --steps 10
    python scripts/verify_reproducibility.py --device cpu   # determinism guaranteed on CPU
"""

import argparse
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from anyup.modules.cross_attention3d import CrossAttentionBlock3D
from anyup.utils.seed import set_seed


# ══════════════════════════════════════════════════════════════════════════════
# Reuse the same minimal pipeline from smoke_test
# ══════════════════════════════════════════════════════════════════════════════

class _MinimalPipeline(nn.Module):
    def __init__(self, C_img, C_feat, C_qk, num_heads, window_ratio, window_t):
        super().__init__()
        self.q_proj   = nn.Conv3d(C_img,  C_qk,  kernel_size=1)
        self.k_proj   = nn.Conv3d(C_feat, C_qk,  kernel_size=1)
        self.attn     = CrossAttentionBlock3D(
            qk_dim=C_qk,
            num_heads=num_heads,
            window_ratio=window_ratio,
            window_t=window_t,
        )
        self.out_proj = nn.Conv3d(C_feat, C_feat, kernel_size=1)

    def forward(self, rgb, feat):
        q   = self.q_proj(rgb)    # (B, C_qk,  T, H,    W)
        k   = self.k_proj(feat)   # (B, C_qk,  T, H//s, W//s)
        out = self.attn(q, k, feat)
        return self.out_proj(out)


def _run_steps(steps, seed, device, cfg):
    """
    Run `steps` training steps from scratch using `seed`.
    Returns list of per-step total loss values (Python floats).
    """
    set_seed(seed)

    B, C_feat, C_qk, T, H, W = (
        cfg["B"], cfg["C_feat"], cfg["C_qk"], cfg["T"], cfg["H"], cfg["W"]
    )
    s = 4   # ↓ spatial downscale — must match make_batch in smoke_test.py

    model = _MinimalPipeline(
        C_img=3, C_feat=C_feat, C_qk=C_qk,
        num_heads=cfg["num_heads"],
        window_ratio=cfg["window_ratio"],
        window_t=cfg["window_t"],
    ).to(device).train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)

    losses = []
    for _ in range(steps):
        rgb  = torch.randn(B, 3,      T, H,    W,    device=device)  # reproducible via seed
        feat = torch.randn(B, C_feat, T, H//s, W//s, device=device)  # depends on H,W,s

        optimizer.zero_grad()
        pred   = model(rgb, feat)
        target = torch.randn_like(pred)        # also reproducible — drawn from same RNG state
        loss   = F.mse_loss(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(loss.item())

    return losses


# ══════════════════════════════════════════════════════════════════════════════
# Verification
# ══════════════════════════════════════════════════════════════════════════════

def verify_reproducibility(seed=42, steps=10, device="cuda"):
    print("=" * 60)
    print("AnyUp3D — Phase 7.3 Reproducibility Verification")
    print("=" * 60)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Device : {dev}")
    print(f"Seed   : {seed}")
    print(f"Steps  : {steps}")
    print()

    # ── Config ────────────────────────────────────────────────────────────
    cfg = dict(
        B=2,             # ↓ reduce to save memory
        C_feat=64,
        C_qk=64,
        T=4,             # ↓ reduce to save memory — also affects temporal loss
        H=56,            # ↓ reduce to save memory — must be divisible by s=4
        W=56,            # ↓ reduce to save memory — must equal H
        num_heads=4,     # ↓ reduce if OOM
        window_ratio=0.15,
        window_t=1,
    )

    print("Run 1 ...")
    losses_a = _run_steps(steps, seed, dev, cfg)

    print("Run 2 ...")
    losses_b = _run_steps(steps, seed, dev, cfg)

    print()
    print(f"{'Step':>5}  {'Run 1':>14}  {'Run 2':>14}  {'Match':>7}")
    print("-" * 50)
    all_match = True
    for i, (a, b) in enumerate(zip(losses_a, losses_b)):
        match = a == b      # bit-for-bit equality on float (same seed → same RNG state)
        all_match = all_match and match
        mark = "✅" if match else "❌"
        print(f"  {i:3d}  {a:14.8f}  {b:14.8f}  {mark}")

    print()
    print("=" * 60)
    if all_match:
        print(f"REPRODUCIBILITY VERIFIED — all {steps} steps match exactly ✅")
    else:
        n_bad = sum(a != b for a, b in zip(losses_a, losses_b))
        print(f"REPRODUCIBILITY FAILED — {n_bad}/{steps} steps differ ❌")
        print()
        print("Likely causes:")
        print("  • Non-deterministic op without a deterministic kernel")
        print("    → set warn_only=False in set_seed() to surface the op name")
        print("  • DataLoader num_workers > 0 with different worker seeds")
        print("    → set worker_init_fn=seed_worker (see PyTorch docs)")
        print("  • CUDA graph replay or async ops reordering gradient accumulation")
        print("    → run on CPU first to isolate GPU non-determinism")
        sys.exit(1)
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--steps",  type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    verify_reproducibility(seed=args.seed, steps=args.steps, device=args.device)