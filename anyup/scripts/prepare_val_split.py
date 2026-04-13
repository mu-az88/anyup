"""
Scans a directory of video files and writes train.txt / val.txt manifests.
Run once before training. The split is deterministic given the same seed.

Usage:
    python scripts/prepare_val_split.py \
        --video_dir /data/kinetics400/clips \
        --output_dir configs/ \
        --val_fraction 0.05 \
        --seed 42
"""

import argparse
import random
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir",   required=True,  type=Path,
                        help="Root directory containing video clips (searched recursively)")
    parser.add_argument("--output_dir",  required=True,  type=Path,
                        help="Where to write train.txt and val.txt")
    parser.add_argument("--val_fraction", default=0.05, type=float,
                        help="Fraction of clips held out for validation (default 0.05 = 5%%)")
    parser.add_argument("--seed",        default=42,    type=int,
                        help="Random seed — fix this forever so the split never changes")
    args = parser.parse_args()

    # ── Collect all video paths ────────────────────────────────────────────────
    all_clips = sorted(                          # sort first for determinism across OSes
        p for p in args.video_dir.rglob("*")
        if p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not all_clips:
        raise RuntimeError(f"No video files found under {args.video_dir}")

    print(f"Found {len(all_clips)} clips total.")

    # ── Deterministic shuffle + split ─────────────────────────────────────────
    rng = random.Random(args.seed)               # isolated RNG — does not affect global state
    rng.shuffle(all_clips)

    n_val   = max(1, int(len(all_clips) * args.val_fraction))  # at least 1 val clip
    n_train = len(all_clips) - n_val

    train_clips = all_clips[:n_train]
    val_clips   = all_clips[n_train:]

    print(f"Split → train: {len(train_clips)}, val: {len(val_clips)}")

    # ── Write manifests ────────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_manifest = args.output_dir / "train.txt"
    val_manifest   = args.output_dir / "val.txt"

    train_manifest.write_text("\n".join(str(p) for p in train_clips) + "\n")
    val_manifest.write_text(  "\n".join(str(p) for p in val_clips)   + "\n")

    print(f"Manifests written to {args.output_dir}")
    print(f"  {train_manifest}")
    print(f"  {val_manifest}")


if __name__ == "__main__":
    main()