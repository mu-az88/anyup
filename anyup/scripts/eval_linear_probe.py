"""
Entry point for per-frame linear probing evaluation.

Supported datasets: ADE20K (150 classes), Cityscapes (19 classes)

Usage:
    # ADE20K
    python scripts/eval_linear_probe.py \
        --checkpoint checkpoints/stage1_best.pth \
        --dataset ade20k \
        --dataset_root /data/ade20k \
        --num_epochs 20 \
        --batch_size 16

    # Cityscapes
    python scripts/eval_linear_probe.py \
        --checkpoint checkpoints/stage1_best.pth \
        --dataset cityscapes \
        --dataset_root /data/cityscapes \
        --num_epochs 20 \
        --batch_size 16
"""

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ADE20K, Cityscapes

from anyup.data.training.linear_probe import LinearProbe, train_linear_probe


# ── Dataset configs ────────────────────────────────────────────────────────────
DATASET_CONFIGS = {
    "ade20k":     {"num_classes": 150, "ignore_index": 0},    # ADE20K uses 0 as ignore
    "cityscapes": {"num_classes": 19,  "ignore_index": 255},  # Cityscapes uses 255
}


# ── Wrapper to produce the {frames, labels} dict AnyUp3D expects ───────────────
class SegDatasetWrapper(Dataset):
    """
    Thin wrapper around torchvision seg datasets.
    Converts (image, mask) pairs into the {frames, labels} dict
    that train_linear_probe expects.

    frames: (1, H, W, 3)  float32 normalized  — T=1, single frame
    labels: (H, W)        int64               — pixel class IDs
    """

    def __init__(self, base_dataset, spatial_size: tuple[int, int]):
        self.ds          = base_dataset
        self.spatial_size = spatial_size  # ↓ cost: larger → more tokens; change together with model spatial_size

        self.img_transform = transforms.Compose([
            transforms.Resize(spatial_size),
            transforms.ToTensor(),                              # (3, H, W) in [0,1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std= [0.229, 0.224, 0.225]),  # ImageNet stats — match feature extractor
        ])
        self.label_transform = transforms.Compose([
            transforms.Resize(spatial_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.PILToTensor(),                          # (1, H, W) uint8 — nearest to preserve class IDs
        ])

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        img, mask = self.ds[idx]                               # PIL Image, PIL Image

        img_t  = self.img_transform(img)                       # (3, H, W)
        img_t  = img_t.permute(1, 2, 0).unsqueeze(0)          # (1, H, W, 3) — T=1 frame format

        mask_t = self.label_transform(mask).squeeze(0).long()  # (H, W) int64

        return {"frames": img_t, "labels": mask_t}


def build_feature_extractor(model, device):
    """
    Wraps AnyUp3D as a frozen feature extractor:
      input:  (B, 1, H, W, 3)  — T=1 single frame
      output: (B, H', W', C)   — upsampled features, T dim squeezed out
    """
    model.eval()
    model.to(device)

    @torch.no_grad()
    def extract(frames: torch.Tensor) -> torch.Tensor:
        out = model(frames)   # (B, 1, H', W', C)
        return out[:, 0]      # (B, H', W', C) — squeeze T=1

    return extract


def build_cityscapes_loaders(root: Path, spatial_size, batch_size: int):
    """Cityscapes uses 'fine' annotations and a specific mode kwarg."""
    train_ds = Cityscapes(str(root), split="train", mode="fine", target_type="semantic")
    val_ds   = Cityscapes(str(root), split="val",   mode="fine", target_type="semantic")
    train_ds = SegDatasetWrapper(train_ds, spatial_size)
    val_ds   = SegDatasetWrapper(val_ds,   spatial_size)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4),  # ↓ num_workers if CPU memory tight
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4),
    )


def build_ade20k_loaders(root: Path, spatial_size, batch_size: int):
    """ADE20K root should contain 'images/training' and 'images/validation' subdirs."""
    train_ds = ADE20K(str(root), split="train")
    val_ds   = ADE20K(str(root), split="val")
    train_ds = SegDatasetWrapper(train_ds, spatial_size)
    val_ds   = SegDatasetWrapper(val_ds,   spatial_size)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4),  # ↓ num_workers if CPU memory tight
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   required=True, type=Path)
    parser.add_argument("--dataset",      required=True, choices=["ade20k", "cityscapes"])
    parser.add_argument("--dataset_root", required=True, type=Path)
    parser.add_argument("--num_epochs",   default=20,    type=int)
    parser.add_argument("--batch_size",   default=16,    type=int)   # ↓ reduce if OOM
    parser.add_argument("--spatial_size", default=224,   type=int)   # ↓ reduce to save memory — also affects mIoU resolution
    parser.add_argument("--lr",           default=1e-3,  type=float)
    parser.add_argument("--feature_dim",  default=384,   type=int,
                        help="AnyUp3D output channel dim — must match model config")
    args = parser.parse_args()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg       = DATASET_CONFIGS[args.dataset]
    spatial   = (args.spatial_size, args.spatial_size)  # ↓ both dims tied — update batch_size if you change this

    print(f"Device: {device} | Dataset: {args.dataset} | Classes: {cfg['num_classes']}")

    # ── Load AnyUp3D ──────────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    from anyup.model import AnyUp
    model = AnyUp()
    model.load_state_dict(checkpoint["model"])
    feature_extractor = build_feature_extractor(model, device)

    # ── Build loaders ─────────────────────────────────────────────────────────
    if args.dataset == "ade20k":
        train_loader, val_loader = build_ade20k_loaders(
            args.dataset_root, spatial, args.batch_size)
    else:
        train_loader, val_loader = build_cityscapes_loaders(
            args.dataset_root, spatial, args.batch_size)

    # ── Train + evaluate probe ─────────────────────────────────────────────────
    probe = LinearProbe(feature_dim=args.feature_dim, num_classes=cfg["num_classes"])

    history = train_linear_probe(
        probe=probe,
        feature_extractor=feature_extractor,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=cfg["num_classes"],
        num_epochs=args.num_epochs,
        lr=args.lr,
        device=device,
        ignore_index=cfg["ignore_index"],
    )

    best_miou = max(history["val_miou"])
    print(f"\nBest val mIoU: {best_miou:.4f}")
    print("Compare against published 2D AnyUp number for the same encoder.")


if __name__ == "__main__":
    main()