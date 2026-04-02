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
def compute_attention_mask_3d(T_q, H_q, W_q,
                              T_k, H_k, W_k,
                              spatial_ratio,
                              window_t=None,
                              device="cpu"):
    """
    3D attention mask for (T, H, W) flattened sequences.

    Returns a boolean mask of shape (T_q*H_q*W_q, T_k*H_k*W_k)
    with True = blocked (same convention as compute_attention_mask).

    Flatten order is (t, h, w) — T outermost.

    Parameters
    ----------
    spatial_ratio : float
        Window ratio for H, W dimensions (same as 2D window_size_ratio).
    window_t : int or None
        Each query at frame t_q attends to keys at frames
        |t_q_normalized - t_k_normalized| mapped through the same
        floor/ceil quantisation as spatial, but using integer ± window.
        None means no temporal restriction (attend to all frames).
    """
    # ── spatial mask: reuse existing 2D ──
    # shape (H_q*W_q, H_k*W_k), True = blocked
    spatial_blocked = compute_attention_mask(
        H_q, W_q, H_k, W_k, spatial_ratio, device=device
    )
    spatial_allowed = ~spatial_blocked  # True = can attend

    # ── temporal mask ──
    # shape (T_q, T_k), True = can attend
    if window_t is None:
        temporal_allowed = torch.ones(T_q, T_k, dtype=torch.bool, device=device)
    else:
        tq = torch.arange(T_q, device=device)
        tk = torch.arange(T_k, device=device)
        # normalise to [0, 1) then map to key-frame indices
        # same logic as spatial: query center in normalised coords,
        # ± window_t key-frame steps
        if T_q == T_k:
            # common case: same temporal resolution
            temporal_allowed = (tq[:, None] - tk[None, :]).abs() <= window_t
        else:
            # different temporal resolutions: normalise query position
            # to key grid, then apply integer window
            tq_on_k = (tq.float() + 0.5) / T_q * T_k  # center of query frame in key-frame coords
            temporal_allowed = (tq_on_k[:, None] - (tk.float()[None, :] + 0.5)).abs() < (window_t + 0.5)

    # ── combine: outer product over (T, spatial) ──
    # temporal_allowed: (T_q, T_k)
    # spatial_allowed:  (H_q*W_q, H_k*W_k)
    # target:           (T_q * H_q*W_q,  T_k * H_k*W_k)
    combined = (
        temporal_allowed[:, None, :, None]    # (T_q, 1,      T_k, 1)
        & spatial_allowed[None, :, None, :]   # (1,   HqWq,   1,   HkWk)
    )  # → (T_q, HqWq, T_k, HkWk)

    combined = combined.reshape(T_q * H_q * W_q, T_k * H_k * W_k)
    return ~combined  # True = blocked