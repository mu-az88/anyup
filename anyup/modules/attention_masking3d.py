import torch
from typing import Optional
from functools import lru_cache


# ── Existing 2D code (unchanged) ─────────────────────────────────

def window2d(
        low_res,
        high_res,
        ratio,
        *,
        device="cpu"
):
    if isinstance(high_res, int):
        H = W = high_res
    else:
        H, W = high_res
    if isinstance(low_res, int):
        Lh = Lw = low_res
    else:
        Lh, Lw = low_res

    r_pos = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) / H
    c_pos = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) / W
    pos_r, pos_c = torch.meshgrid(r_pos, c_pos, indexing="ij")

    r_lo = (pos_r - ratio).clamp(0.0, 1.0)
    r_hi = (pos_r + ratio).clamp(0.0, 1.0)
    c_lo = (pos_c - ratio).clamp(0.0, 1.0)
    c_hi = (pos_c + ratio).clamp(0.0, 1.0)

    r0 = (r_lo * Lh).floor().long()
    r1 = (r_hi * Lh).ceil().long()
    c0 = (c_lo * Lw).floor().long()
    c1 = (c_hi * Lw).ceil().long()

    return torch.stack([r0, r1, c0, c1], dim=2)


@lru_cache
def compute_attention_mask(high_res_h, high_res_w, low_res_h, low_res_w,
                           window_size_ratio, device="cpu"):
    h, w = high_res_h, high_res_w
    h_, w_ = low_res_h, low_res_w

    windows = window2d(
        low_res=(h_, w_),
        high_res=(h, w),
        ratio=window_size_ratio,
        device=device
    )

    q = h * w

    r0 = windows[..., 0].reshape(q, 1)
    r1 = windows[..., 1].reshape(q, 1)
    c0 = windows[..., 2].reshape(q, 1)
    c1 = windows[..., 3].reshape(q, 1)

    rows = torch.arange(h_, device=device)
    cols = torch.arange(w_, device=device)

    row_ok = (rows >= r0) & (rows < r1)
    col_ok = (cols >= c0) & (cols < c1)

    attention_mask = (row_ok.unsqueeze(2) & col_ok.unsqueeze(1)) \
        .reshape(q, h_ * w_)

    return ~attention_mask


# ── New 3D mask ──────────────────────────────────────────────────

@lru_cache
def compute_attention_mask_3d(T, H_q, W_q,
                              H_k, W_k,
                              spatial_ratio,
                              window_t=None,
                              device="cpu"):
    """
    3D attention mask for (T, H, W) flattened sequences.

    Q and K share the same temporal dimension T (spatial upsampling only —
    no frame interpolation). Returns a boolean mask of shape
    (T*H_q*W_q, T*H_k*W_k) with True = blocked.

    Flatten order is (t, h, w) — T outermost.

    Parameters
    ----------
    spatial_ratio : float
        Window ratio for H, W dimensions (same as 2D window_size_ratio).
    window_t : int or None
        Attends to keys at frames where |t_q - t_k| <= window_t.
        None means no temporal restriction (attend to all frames).
    """
    spatial_blocked = compute_attention_mask(
        H_q, W_q, H_k, W_k, spatial_ratio, device=device
    )
    spatial_allowed = ~spatial_blocked

    if window_t is None:
        temporal_allowed = torch.ones(T, T, dtype=torch.bool, device=device)
    else:
        t_idx = torch.arange(T, device=device)
        temporal_allowed = (t_idx[:, None] - t_idx[None, :]).abs() <= window_t

    combined = (
        temporal_allowed[:, None, :, None]
        & spatial_allowed[None, :, None, :]
    )  # (T, H_q*W_q, T, H_k*W_k)

    combined = combined.reshape(T * H_q * W_q, T * H_k * W_k)
    return ~combined  # True = blocked