"""
Linear probe for per-frame semantic segmentation evaluation.
Trains a single Linear layer on top of frozen AnyUp3D upsampled features,
then evaluates mIoU on a held-out set.

This is the go/no-go gate after Stage 1 (T=1) warmup:
mIoU here should match published 2D AnyUp numbers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional


class LinearProbe(nn.Module):
    """
    Single linear layer mapping upsampled feature dim → num_classes.
    Deliberately no hidden layers — this isolates feature quality from probe capacity.
    """

    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)  # ↓ cost: feature_dim driven by encoder choice

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, H, W, C) per-frame features at upsampled resolution
        returns: (B, num_classes, H, W) logits for cross-entropy loss
        """
        # (B, H, W, C) → (B, C, H, W) for F.cross_entropy
        logits = self.fc(x)               # (B, H, W, num_classes)
        return logits.permute(0, 3, 1, 2) # (B, num_classes, H, W)


def train_linear_probe(
    probe: LinearProbe,
    feature_extractor,           # callable: (B, 1, H, W, 3) → (B, H', W', C) — AnyUp3D in T=1 mode
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    num_epochs: int = 20,        # ↓ cost: reduce for quick sanity check; 20 is standard for probing
    lr: float = 1e-3,
    device: torch.device = torch.device("cuda"),
    ignore_index: int = 255,     # standard ignore label for Cityscapes / ADE20K
) -> dict:
    """
    Trains the linear probe and returns val mIoU after each epoch.
    feature_extractor must be frozen (no grad) — enforced inside this function.
    """

    probe = probe.to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)  # Adam is standard for linear probing

    history = {"train_loss": [], "val_miou": []}

    for epoch in range(num_epochs):
        # ── Training pass ──────────────────────────────────────────────────────
        probe.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            frames = batch["frames"].to(device)         # (B, 1, H, W, 3) — T=1 during warmup
            labels = batch["labels"].to(device)         # (B, H_label, W_label) int64

            # Extract upsampled features — frozen, no grad
            with torch.no_grad():
                features = feature_extractor(frames)    # (B, H', W', C)  ↓ H'W' = upsampled resolution

            logits = probe(features)                    # (B, num_classes, H', W')

            # Resize labels to match feature resolution if needed
            # ↓ label resolution: if H_label != H', bilinear resize here
            if labels.shape[-2:] != logits.shape[-2:]:
                labels = F.interpolate(
                    labels.unsqueeze(1).float(),
                    size=logits.shape[-2:],
                    mode="nearest",
                ).squeeze(1).long()

            loss = F.cross_entropy(logits, labels, ignore_index=ignore_index)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss  += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        history["train_loss"].append(avg_loss)

        # ── Validation pass ────────────────────────────────────────────────────
        miou = evaluate_miou(probe, feature_extractor, val_loader,
                             num_classes, device, ignore_index)
        history["val_miou"].append(miou)
        print(f"Epoch {epoch+1}/{num_epochs} | loss={avg_loss:.4f} | val mIoU={miou:.4f}")

    return history


@torch.no_grad()
def evaluate_miou(
    probe: LinearProbe,
    feature_extractor,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
    ignore_index: int = 255,
) -> float:
    """
    Computes mean IoU over the full loader.
    Uses a confusion matrix — numerically stable and handles class imbalance correctly.
    """
    probe.eval()

    # Confusion matrix: rows = GT class, cols = predicted class
    # ↓ memory: (num_classes, num_classes) — negligible even for 150 classes
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)

    for batch in loader:
        frames  = batch["frames"].to(device)   # (B, 1, H, W, 3)
        labels  = batch["labels"].to(device)   # (B, H_label, W_label)

        features = feature_extractor(frames)   # (B, H', W', C)
        logits   = probe(features)             # (B, num_classes, H', W')
        preds    = logits.argmax(dim=1)        # (B, H', W')

        # Resize labels to pred resolution
        if labels.shape[-2:] != preds.shape[-2:]:
            labels = F.interpolate(
                labels.unsqueeze(1).float(),
                size=preds.shape[-2:],
                mode="nearest",
            ).squeeze(1).long()

        # Accumulate confusion matrix — only on valid (non-ignore) pixels
        mask   = (labels != ignore_index)                        # (B, H', W') bool
        gt     = labels[mask]                                    # (N_valid,)
        pred   = preds[mask]                                     # (N_valid,)

        # ↓ cost: this scatter_add is O(N_valid) — main eval cost per batch
        indices = gt * num_classes + pred                        # flat confusion index
        confusion.view(-1).scatter_add_(
            0, indices, torch.ones_like(indices)
        )

    # mIoU from confusion matrix
    tp        = confusion.diagonal()                             # (num_classes,) true positives
    fn        = confusion.sum(dim=1) - tp                       # false negatives
    fp        = confusion.sum(dim=0) - tp                       # false positives
    iou       = tp.float() / (tp + fn + fp).float().clamp(min=1e-6)

    # Only average over classes that appear in GT (avoids inflating mIoU with absent classes)
    present   = confusion.sum(dim=1) > 0                        # (num_classes,) bool
    miou      = iou[present].mean().item()

    return miou