
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from anyup.data.video_dataset import VideoDataset
from anyup.data.crop_sampler import SpatiotemporalCropSampler, SpatiotemporalCrop
from anyup.data.feature_extractor import PerFrameFeatureExtractor


# ---------------------------------------------------------------------------
# Per-item return type from the Dataset
# ---------------------------------------------------------------------------

@dataclass
class VideoClipItem:
    """
    Single item returned by ``ClipDataset.__getitem__``.
    Contains only CPU tensors — no GPU work happens in workers.
    """
    video_full:  torch.Tensor   # (T, H, W, 3)  float32  ImageNet-normalised
    video_crop:  torch.Tensor   # (T', H_c, W_c, 3) float32
    p_dummy:     None           # placeholder — p is extracted on GPU in the training loop
    coords:      dict           # crop coordinates (scalars + lists)
    clip_idx:    int            # dataset index — needed for p cache key


# ---------------------------------------------------------------------------
# Dataset that wraps VideoDataset + SpatiotemporalCropSampler
# ---------------------------------------------------------------------------

class ClipDataset(Dataset):
    """
    Wraps ``VideoDataset`` with online spatiotemporal crop sampling.

    Each ``__getitem__`` call:
      1. Loads a full video clip (T, H, W, 3) via ``VideoDataset``
      2. Samples a spatiotemporal crop tube via ``SpatiotemporalCropSampler``
      3. Returns CPU tensors only — GPU feature extraction is deferred

    Parameters
    ----------
    video_dataset : VideoDataset
        Underlying video loader.
    sampler : SpatiotemporalCropSampler
        Crop sampler. ``n_frames`` should match the current T-curriculum value.
        ↓ reduce sampler.n_frames to save memory; also reduce batch_size.
    p_spatial_size : tuple[int, int]
        (h_p, w_p) — target size for the coarse feature volume.
        Must be divisible by the encoder patch_size (14 for DINOv2).
        ↓ reduce to get coarser p, less GPU memory during extraction.
    """

    def __init__(
        self,
        video_dataset: VideoDataset,
        sampler: SpatiotemporalCropSampler,
        p_spatial_size: Tuple[int, int] = (28, 28),  # ↓ reduce for coarser p; must be multiple of patch_size=14
    ) -> None:
        self.video_dataset   = video_dataset
        self.sampler         = sampler
        self.p_spatial_size  = p_spatial_size         # (h_p, w_p); drives spatial_ratio = H // h_p

    def __len__(self) -> int:
        return len(self.video_dataset)

    def __getitem__(self, idx: int) -> VideoClipItem:
        # Step 1: load full clip — (T, H, W, 3) float32 on CPU
        video_full = self.video_dataset[idx]          # (T, H, W, 3); T set by VideoDataset.n_frames

        # Step 2: build a dummy coarse feature volume for the sampler
        # We can't run the encoder here (no GPU in workers), so we create
        # a placeholder p at the target spatial resolution.
        # The sampler only needs p's shape to compute valid crop coordinates.
        T = video_full.shape[0]                       # number of frames; ↑ scales memory
        h_p, w_p = self.p_spatial_size                # coarse spatial resolution
        p_placeholder = torch.zeros(T, h_p, w_p, 1)  # (T, h_p, w_p, 1) — shape-only, no real features

        # Step 3: sample the spatiotemporal crop
        crop: SpatiotemporalCrop = self.sampler.sample(video_full, p_placeholder)

        return VideoClipItem(
            video_full=crop.video_full,   # (T, H, W, 3)
            video_crop=crop.video_crop,   # (T', H_c, W_c, 3); T' set by sampler.n_frames
            p_dummy=None,
            coords=crop.coords,
            clip_idx=idx,
        )


# ---------------------------------------------------------------------------
# Custom collate function
# ---------------------------------------------------------------------------

def collate_clip_items(items: List[VideoClipItem]) -> Dict:
    """
    Collate a list of ``VideoClipItem`` into a batch dict.

    Tensors are stacked along dim 0 → batch dimension B.
    ``coords`` dicts are collated key-by-key: scalar int/float values
    become lists; list values (e.g. frame_indices) become lists of lists.

    Returns
    -------
    dict with keys:
        video_full  : (B, T, H, W, 3)
        video_crop  : (B, T', H_c, W_c, 3)
        coords      : dict of lists (one entry per batch item)
        clip_indices: list[int]  length B
    """
    B = len(items)                                    # batch size; ↓ reduce to save memory

    video_full  = torch.stack([it.video_full  for it in items], dim=0)   # (B, T, H, W, 3)
    video_crop  = torch.stack([it.video_crop  for it in items], dim=0)   # (B, T', H_c, W_c, 3)

    # Collate coords: each value becomes a list of length B
    coords_batch: Dict[str, list] = {}
    for key in items[0].coords.keys():
        coords_batch[key] = [it.coords[key] for it in items]             # list of B values

    clip_indices = [it.clip_idx for it in items]

    return {
        "video_full":   video_full,    # (B, T, H, W, 3);   B×T×H×W×3 floats
        "video_crop":   video_crop,    # (B, T', H_c, W_c, 3); B×T'×H_c×W_c×3 floats
        "coords":       coords_batch,
        "clip_indices": clip_indices,  # list[int] length B
    }


# ---------------------------------------------------------------------------
# GPU-side feature extraction step (called in the training loop, not workers)
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_batch_features(
    batch: Dict,
    extractor: PerFrameFeatureExtractor,
    p_spatial_size: Tuple[int, int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run the frozen encoder on the batch — **must be called in the training loop**,
    not inside DataLoader workers.

    Extracts:
      - ``p``: coarse features for the full video, shape (B, T, h_p, w_p, C)
      - ``q_hat``: GT features for the crop, shape (B, T', h_gt, w_gt, C)

    Parameters
    ----------
    batch : dict
        Output of ``collate_clip_items``.
    extractor : PerFrameFeatureExtractor
        Frozen encoder. Already on ``device``.
    p_spatial_size : (h_p, w_p)
        Target spatial size for coarse features (passed through to extractor).
        ↓ reduce to get coarser p, cheaper extraction.
    device : torch.device
        Training device. Tensors are moved here before encoding.

    Returns
    -------
    p     : Tensor (B, T, h_p, w_p, C)  on device
    q_hat : Tensor (B, T', h_gt, w_gt, C) on device
    """
    video_full  = batch["video_full"]   # (B, T, H, W, 3) on CPU from DataLoader
    video_crop  = batch["video_crop"]   # (B, T', H_c, W_c, 3) on CPU
    clip_indices = batch["clip_indices"]
    frame_indices_batch = batch["coords"]["frame_indices"]  # list[list[int]] length B

    B = video_full.shape[0]            # batch size; ↑ scales GPU memory linearly

    p_list:     List[torch.Tensor] = []
    q_hat_list: List[torch.Tensor] = []

    for b in range(B):                 # loop over batch items; B forward passes for p, B*T' for q̂
        # ---- Coarse features p for full video ----
        # Returns CPU tensor (from cache or fresh extraction)
        p_cpu = extractor.extract_p(
            video_full=video_full[b],           # (T, H, W, 3)
            clip_idx=clip_indices[b],
            frame_indices=frame_indices_batch[b],
            target_spatial_size=p_spatial_size, # ↓ controls h_p, w_p; smaller = less memory
        )
        p_list.append(p_cpu.to(device))        # move to GPU for training

        # ---- GT features q̂ for the crop ----
        q_hat = extractor.extract_gt(
            video_crop=video_crop[b].to(device),  # (T', H_c, W_c, 3) → GPU
        )                                          # returns (T', h_gt, w_gt, C) on device
        q_hat_list.append(q_hat)

    p     = torch.stack(p_list,     dim=0)   # (B, T,  h_p,  w_p,  C)
    q_hat = torch.stack(q_hat_list, dim=0)   # (B, T', h_gt, w_gt, C)

    return p, q_hat


# ---------------------------------------------------------------------------
# Factory — builds the full DataLoader in one call
# ---------------------------------------------------------------------------

def build_dataloader(
    root: str,
    n_frames: int = 4,                          # ↓ T-curriculum: start at 1, grow to 16; reduce batch_size when increasing
    spatial_size: Tuple[int, int] = (224, 224), # ↓ reduce to save memory; also reduces H_c and p resolution
    crop_size: Tuple[int, int] = (7, 7),        # ↓ reduce crop_size to save attention memory (scales as h_c*w_c*T')
    temporal_stride: int = 1,                   # ↓ reduce for denser crops; affects temporal_consistency loss range
    p_spatial_size: Tuple[int, int] = (28, 28), # ↓ reduce for coarser p; must be multiple of patch_size=14
    batch_size: int = 4,                        # ↓ reduce if OOM; must reduce when n_frames increases
    num_workers: int = 4,                       # ↓ reduce if CPU-bound or RAM-constrained
    shuffle: bool = True,
    pin_memory: bool = True,                    # set False if not using CUDA
    min_video_frames: int = 16,
) -> DataLoader:
    """
    Build the full training DataLoader.

    Workers are CPU-only. GPU feature extraction is done separately via
    ``extract_batch_features`` inside the training loop.
    """
    video_ds = VideoDataset(
        root=root,
        n_frames=n_frames,                      # T; ↑ this when curriculum advances
        spatial_size=spatial_size,              # (H, W); ↓ reduces token count quadratically
        min_video_frames=min_video_frames,
    )

    sampler = SpatiotemporalCropSampler(
        n_frames=n_frames,                      # T'; must match video_ds.n_frames
        crop_size=crop_size,                    # (h_c, w_c) in feature space
        temporal_stride=temporal_stride,
    )

    clip_ds = ClipDataset(
        video_dataset=video_ds,
        sampler=sampler,
        p_spatial_size=p_spatial_size,          # sets spatial_ratio = H // h_p
    )

    return DataLoader(
        clip_ds,
        batch_size=batch_size,                  # ↓ reduce if OOM; also adjust when n_frames changes
        shuffle=shuffle,
        num_workers=num_workers,                # ↓ each worker holds one video in RAM
        collate_fn=collate_clip_items,
        pin_memory=pin_memory,                  # speeds up CPU→GPU transfer; requires CUDA
        persistent_workers=(num_workers > 0),  # avoids worker respawn overhead between epochs
        prefetch_factor=2 if num_workers > 0 else None,  # ↓ reduce if RAM is tight; each worker prefetches this many batches
    )