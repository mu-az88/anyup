
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Tuple

import torch


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class SpatiotemporalCrop:
    """
    Everything the training loop needs for one crop.

    Attributes
    ----------
    video_full : Tensor (T, H, W, 3)
        The full guidance video (unmodified). The model needs the full frame
        context even though we only supervise a sub-region.
    p_crop : Tensor (T', h_c, w_c, C)
        The coarse feature patch corresponding to the crop region.
    video_crop : Tensor (T', H_c, W_c, 3)
        The raw RGB frames for the crop region — fed to the GT encoder.
    coords : dict
        Crop coordinates (in both feature-space and video-space) for
        reconstructing which output tokens to supervise against GT.
        Keys:
            t0, t1          — temporal slice [t0 : t1] in frames
            fh0, fh1        — feature-space height slice
            fw0, fw1        — feature-space width slice
            vh0, vh1        — video-space height slice  (= fh * spatial_ratio)
            vw0, vw1        — video-space width slice
            spatial_ratio   — H / h_p (integer)
    """
    video_full: torch.Tensor   # (T, H, W, 3)
    p_crop:     torch.Tensor   # (T', h_c, w_c, C)
    video_crop: torch.Tensor   # (T', H_c, W_c, 3)
    coords:     dict


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

class SpatiotemporalCropSampler:
    """
    Samples a random 3D tube from a video clip and its coarse feature volume.

    The crop is defined in **feature space** first, then projected to video
    space via ``spatial_ratio = H / h_p``. This guarantees pixel-perfect
    alignment between ``p_crop`` and ``video_crop``.

    Parameters
    ----------
    n_frames : int
        Number of temporal frames T' in each crop.
        Controlled by the T-curriculum — pass the current curriculum value.
        ↓ reduce n_frames to save memory; also reduce batch size accordingly.
    crop_size : tuple[int, int]
        (h_c, w_c) in **feature space** — i.e., number of feature patches,
        not pixels.
        ↑ increase for larger receptive field; GPU memory scales as h_c * w_c * T'.
    temporal_stride : int
        Stride between sampled frames within the crop tube.
        stride=1 → contiguous frames; higher → sparser but longer span.
        ↓ reduce to save memory; also affects temporal_consistency loss range.
    """

    def __init__(
        self,
        n_frames: int = 4,                  # ↓ T-curriculum: start at 1, grow to 16
        crop_size: Tuple[int, int] = (7, 7), # ↓ reduce to save memory; scales attention O(h_c*w_c*T')
        temporal_stride: int = 1,           # ↓ reduce for denser temporal crops
    ) -> None:
        self.n_frames = n_frames            # T' — also the token sequence length contributor
        self.crop_h, self.crop_w = crop_size  # h_c, w_c in feature space
        self.temporal_stride = temporal_stride

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(
        self,
        video: torch.Tensor,   # (T, H, W, 3) — full guidance video
        p: torch.Tensor,       # (T, h_p, w_p, C) — coarse feature volume
    ) -> SpatiotemporalCrop:
        """
        Sample one spatiotemporal crop tube.

        Parameters
        ----------
        video : Tensor (T, H, W, 3)
        p     : Tensor (T, h_p, w_p, C)

        Returns
        -------
        SpatiotemporalCrop
        """
        T,  H,  W,  _  = video.shape   # video dimensions; H and W drive spatial_ratio
        Tp, h_p, w_p, C = p.shape      # feature volume dimensions; h_p, w_p set crop bounds

        # ---- Validate temporal consistency --------------------------------
        if T != Tp:
            raise ValueError(
                f"video T={T} and feature T={Tp} must match — "
                "AnyUp3D performs spatial-only upsampling."
            )

        # ---- Compute spatial ratio ----------------------------------------
        # This must be an integer — if it isn't, the features don't align.
        if H % h_p != 0 or W % w_p != 0:
            raise ValueError(
                f"Video spatial size ({H}, {W}) must be divisible by "
                f"feature spatial size ({h_p}, {w_p}). "
                f"Got H/h_p={H/h_p:.2f}, W/w_p={W/w_p:.2f}."
            )
        r_h = H // h_p   # spatial ratio height; ↑ means coarser features
        r_w = W // w_p   # spatial ratio width;  depends on encoder patch size

        # ---- Validate crop fits inside feature volume ---------------------
        if self.crop_h > h_p or self.crop_w > w_p:
            raise ValueError(
                f"crop_size ({self.crop_h}, {self.crop_w}) exceeds "
                f"feature spatial size ({h_p}, {w_p})."
            )

        # ---- Sample temporal start ----------------------------------------
        # We need n_frames frames separated by temporal_stride.
        # Total span in the video = (n_frames - 1) * stride + 1
        span = (self.n_frames - 1) * self.temporal_stride + 1  # ↑ grows with n_frames and stride
        if span > T:
            raise ValueError(
                f"Requested clip span {span} (n_frames={self.n_frames}, "
                f"stride={self.temporal_stride}) exceeds video length T={T}. "
                "Reduce n_frames or temporal_stride, or use a longer video."
            )
        t0 = random.randint(0, T - span)                   # random temporal offset
        frame_indices = [t0 + i * self.temporal_stride for i in range(self.n_frames)]
        # frame_indices has length n_frames; this is the T' dimension
        t1 = frame_indices[-1] + 1                         # exclusive end for coord dict

        # ---- Sample spatial crop in feature space -------------------------
        fh0 = random.randint(0, h_p - self.crop_h)        # feature-space top-left height
        fw0 = random.randint(0, w_p - self.crop_w)        # feature-space top-left width
        fh1 = fh0 + self.crop_h                            # exclusive end
        fw1 = fw0 + self.crop_w                            # exclusive end

        # ---- Project to video space (pixel coordinates) ------------------
        vh0 = fh0 * r_h    # video top-left pixel row;    depends on r_h
        vh1 = fh1 * r_h    # video bottom pixel row;       must stay exact multiple
        vw0 = fw0 * r_w    # video top-left pixel column; depends on r_w
        vw1 = fw1 * r_w    # video bottom pixel column

        # ---- Slice the tensors -------------------------------------------
        # p_crop: select frames then spatial patch — (T', h_c, w_c, C)
        p_crop = p[frame_indices][:, fh0:fh1, fw0:fw1, :]  # T' set by n_frames; spatial by crop_size

        # video_crop: same frame selection, pixel-space patch — (T', H_c, W_c, 3)
        video_crop = video[frame_indices][:, vh0:vh1, vw0:vw1, :]  # H_c = crop_h * r_h

        # ---- Assemble coordinate record ----------------------------------
        coords = {
            "t0": t0,          "t1": t1,          # temporal extent in original video
            "frame_indices": frame_indices,        # exact frames sampled (for non-contiguous strides)
            "fh0": fh0,        "fh1": fh1,        # feature-space crop
            "fw0": fw0,        "fw1": fw1,
            "vh0": vh0,        "vh1": vh1,        # video-space crop (pixel coords)
            "vw0": vw0,        "vw1": vw1,
            "spatial_ratio_h": r_h,               # needed by training loop to locate output tokens
            "spatial_ratio_w": r_w,
        }

        return SpatiotemporalCrop(
            video_full=video,       # full video — model needs full context
            p_crop=p_crop,
            video_crop=video_crop,
            coords=coords,
        )