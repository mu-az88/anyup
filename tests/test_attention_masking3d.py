import torch
from anyup.modules.attention_masking3d import compute_attention_mask, compute_attention_mask_3d


# ─────────────────────────────────────────────
# Test 1 — T=1 matches 2D mask exactly
# ─────────────────────────────────────────────
def test_single_frame_matches_2d():
    """
    With T_q=T_k=1, the 3D mask should be identical to the 2D mask.
    """
    H_q, W_q = 8, 12
    H_k, W_k = 4, 6
    ratio = 0.15

    mask_2d = compute_attention_mask(H_q, W_q, H_k, W_k, ratio)
    mask_3d = compute_attention_mask_3d(1, H_q, W_q, 1, H_k, W_k,
                                        spatial_ratio=ratio, window_t=None)

    assert mask_3d.shape == mask_2d.shape, \
        f"Shape mismatch: {mask_3d.shape} vs {mask_2d.shape}"
    assert torch.equal(mask_3d, mask_2d), \
        f"Masks differ at {(mask_3d != mask_2d).sum().item()} positions"

    print("test_single_frame_matches_2d passed")


# ─────────────────────────────────────────────
# Test 2 — Temporal exclusion with window_t=0
# ─────────────────────────────────────────────
def test_temporal_exclusion():
    """
    With window_t=0 and T_q=T_k=4, a query at frame t can only attend
    to keys at frame t. All cross-frame entries must be blocked.
    """
    T = 4
    H_q, W_q = 4, 4
    H_k, W_k = 2, 2
    ratio = 0.5  # generous spatial window so spatial isn't the bottleneck

    mask = compute_attention_mask_3d(T, H_q, W_q, T, H_k, W_k,
                                     spatial_ratio=ratio, window_t=0)

    S_q = H_q * W_q  # 16
    S_k = H_k * W_k  # 4

    for t_q in range(T):
        for t_k in range(T):
            # extract the (S_q, S_k) sub-block for this frame pair
            block = mask[t_q * S_q : (t_q + 1) * S_q,
                         t_k * S_k : (t_k + 1) * S_k]
            if t_q == t_k:
                # same frame: should have some unblocked entries (spatial window allows it)
                assert not block.all(), \
                    f"Frame pair ({t_q},{t_k}): entirely blocked but should have visible entries"
            else:
                # different frame: entirely blocked
                assert block.all(), \
                    f"Frame pair ({t_q},{t_k}): has {(~block).sum().item()} unblocked entries, expected 0"

    print("test_temporal_exclusion passed")


# ─────────────────────────────────────────────
# Test 3 — Shape and dtype
# ─────────────────────────────────────────────
def test_shape_and_dtype():
    T_q, H_q, W_q = 3, 6, 8
    T_k, H_k, W_k = 5, 3, 4

    mask = compute_attention_mask_3d(T_q, H_q, W_q, T_k, H_k, W_k,
                                     spatial_ratio=0.2, window_t=1)

    expected = (T_q * H_q * W_q, T_k * H_k * W_k)
    assert mask.shape == expected, f"Shape {mask.shape} != {expected}"
    assert mask.dtype == torch.bool, f"dtype {mask.dtype} != torch.bool"

    print("test_shape_and_dtype passed")


if __name__ == "__main__":
    test_single_frame_matches_2d()
    test_temporal_exclusion()
    test_shape_and_dtype()