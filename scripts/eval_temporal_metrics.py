"""
Evaluates temporal coherence of AnyUp3D vs the 2D per-frame baseline.

Usage:
    python scripts/eval_temporal_coherence.py \
        --checkpoint_3d checkpoints/stage2_best.pth \
        --val_manifest configs/val.txt \
        --num_frames 8 \
        --batch_size 4 \
        --static_threshold 0.02
"""

import argparse
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from anyup.data.val_dataset import ValidationDataset, val_collate_fn
from anyup.data.training.temporal_metrics import compare_3d_vs_2d_baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_3d",    required=True,  type=Path)
    parser.add_argument("--val_manifest",     required=True,  type=Path)
    parser.add_argument("--num_frames",       default=8,      type=int)    # ↓ reduce to save memory
    parser.add_argument("--batch_size",       default=4,      type=int)    # ↓ reduce if OOM — also update num_workers
    parser.add_argument("--spatial_size",     default=224,    type=int)    # ↓ reduce to save memory
    parser.add_argument("--static_threshold", default=0.02,   type=float,
                        help="Mean RGB diff below which a position is considered static")
    parser.add_argument("--num_batches",      default=None,   type=int,
                        help="Limit eval to N batches for quick checks (None = full val set)")
    args = parser.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spatial = (args.spatial_size, args.spatial_size)  # ↓ tied to batch_size — change together
    print(f"Device: {device}")

    # ── Load AnyUp3D ──────────────────────────────────────────────────────────
    print(f"Loading 3D checkpoint: {args.checkpoint_3d}")
    ckpt_3d = torch.load(args.checkpoint_3d, map_location=device)

    from anyup.model import AnyUp
    model_3d = AnyUp()
    model_3d.load_state_dict(ckpt_3d["model"])
    model_3d.eval()
    model_3d.to(device)

    # ── Load AnyUp2D baseline ─────────────────────────────────────────────────
    print("Loading 2D baseline from torch.hub...")
    model_2d = torch.hub.load("wimmerth/anyup", "anyup")
    model_2d.eval()
    model_2d.to(device)

    # ── Val dataloader ────────────────────────────────────────────────────────
    val_ds = ValidationDataset(
        mode="video",
        manifest_path=args.val_manifest,
        num_frames=args.num_frames,       # ↓ memory: also update batch_size if you change this
        spatial_size=spatial,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,                    # ↓ reduce if CPU memory tight
        collate_fn=val_collate_fn,
    )

    # ── Eval loop ─────────────────────────────────────────────────────────────
    agg = {"3d": [], "2d": [], "improvement": []}

    for i, batch in enumerate(val_loader):
        if args.num_batches is not None and i >= args.num_batches:
            break

        frames = batch["frames"].to(device)   # (B, T, H, W, 3)

        result = compare_3d_vs_2d_baseline(
            model_3d=model_3d,
            model_2d=model_2d,
            frames=frames,
            device=device,
            static_threshold=args.static_threshold,
        )

        agg["3d"].append(result["3d"])
        agg["2d"].append(result["2d"])
        agg["improvement"].append(result["improvement"])

        if i % 10 == 0:
            print(f"Batch {i} | "
                  f"3D var={result['3d']['mean_var']:.4f} | "
                  f"2D var={result['2d']['mean_var']:.4f} | "
                  f"drift reduction={result['improvement']['drift_reduction']:.4f}")

    # ── Aggregate and report ───────────────────────────────────────────────────
    def mean_of(key, sub): return sum(d[sub][key] for d in zip(*[agg[sub]])) / len(agg[sub])

    def avg(dicts, key):
        vals = [d[key] for d in dicts if d[key] == d[key]]  # filter NaN
        return sum(vals) / len(vals) if vals else float("nan")

    print("\n" + "=" * 50)
    print("TEMPORAL COHERENCE RESULTS")
    print("=" * 50)
    print(f"{'Metric':<30} {'AnyUp3D':>10} {'2D Baseline':>12}")
    print("-" * 50)
    print(f"{'Mean feature variance':<30} {avg(agg['3d'], 'mean_var'):>10.4f} {avg(agg['2d'], 'mean_var'):>12.4f}")
    print(f"{'Median feature variance':<30} {avg(agg['3d'], 'median_var'):>10.4f} {avg(agg['2d'], 'median_var'):>12.4f}")
    print(f"{'Mean cosine drift':<30} {avg(agg['3d'], 'mean_cosine_drift'):>10.4f} {avg(agg['2d'], 'mean_cosine_drift'):>12.4f}")
    print(f"{'Median cosine drift':<30} {avg(agg['3d'], 'median_cosine_drift'):>10.4f} {avg(agg['2d'], 'median_cosine_drift'):>12.4f}")
    print("-" * 50)
    print(f"{'Var reduction (2D - 3D)':<30} {avg(agg['improvement'], 'var_reduction'):>10.4f}")
    print(f"{'Drift reduction (2D - 3D)':<30} {avg(agg['improvement'], 'drift_reduction'):>10.4f}")
    print("=" * 50)
    print("Positive values = 3D model is more temporally coherent than 2D baseline.")


if __name__ == "__main__":
    main()