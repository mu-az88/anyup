import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from anyup.modules.cross_attention3d import CrossAttentionBlock3D
from anyup.layers.attention.chunked_attention import CrossAttentionBlock


def test_output_shape():
    B, C_qk, C_v = 2, 128, 384
    T_q, H_q, W_q = 4, 16, 16
    T_k, H_k, W_k = 4, 8, 8

    block = CrossAttentionBlock3D(qk_dim=128, num_heads=4, window_ratio=0.1, window_t=1)
    q = torch.randn(B, C_qk, T_q, H_q, W_q)
    k = torch.randn(B, C_qk, T_k, H_k, W_k)
    v = torch.randn(B, C_v, T_k, H_k, W_k)

    out = block(q, k, v)
    assert out.shape == (B, C_v, T_q, H_q, W_q), \
        f"Expected {(B, C_v, T_q, H_q, W_q)}, got {out.shape}"
    print("test_output_shape passed")


def test_single_frame_matches_2d():
    qk_dim = 32
    num_heads = 4
    window_ratio = 0.15
    B = 1
    H_q, W_q = 8, 8
    H_k, W_k = 4, 4
    C_v = 64

    block2d = CrossAttentionBlock(qk_dim=qk_dim, num_heads=num_heads, window_ratio=window_ratio)
    block3d = CrossAttentionBlock3D(qk_dim=qk_dim, num_heads=num_heads,
                                    window_ratio=window_ratio, window_t=None)

    # Copy conv weights: Conv2d (C, C, 3, 3) → Conv3d (C, C, 1, 3, 3)
    block3d.conv3d.weight.data.copy_(block2d.conv2d.weight.data.unsqueeze(2))

    # Copy cross_attn parameters by name
    for name, param in block2d.cross_attn.named_parameters():
        block3d.cross_attn.get_parameter(name).data.copy_(param.data)

    q2d = torch.randn(B, qk_dim, H_q, W_q)
    k2d = torch.randn(B, qk_dim, H_k, W_k)
    v2d = torch.randn(B, C_v, H_k, W_k)

    q3d = q2d.unsqueeze(2)
    k3d = k2d.unsqueeze(2)
    v3d = v2d.unsqueeze(2)

    block2d.eval()
    block3d.eval()

    with torch.no_grad():
        out2d = block2d(q2d, k2d, v2d)
        out3d = block3d(q3d, k3d, v3d).squeeze(2)

    assert torch.allclose(out2d, out3d, atol=1e-5), \
        f"Max diff: {(out2d - out3d).abs().max().item()}"
    print("test_single_frame_matches_2d passed")


def test_window_t_none_vs_large_window():
    T = 4
    qk_dim = 32
    num_heads = 4
    window_ratio = 0.15
    B = 1
    H_q, W_q = 8, 8
    H_k, W_k = 4, 4
    C_v = 64

    torch.manual_seed(42)
    block_none = CrossAttentionBlock3D(qk_dim=qk_dim, num_heads=num_heads,
                                       window_ratio=window_ratio, window_t=None)
    torch.manual_seed(42)
    block_large = CrossAttentionBlock3D(qk_dim=qk_dim, num_heads=num_heads,
                                        window_ratio=window_ratio, window_t=T - 1)

    # Copy weights from block_none to block_large
    block_large.load_state_dict(block_none.state_dict())

    block_none.eval()
    block_large.eval()

    torch.manual_seed(0)
    q = torch.randn(B, qk_dim, T, H_q, W_q)
    k = torch.randn(B, qk_dim, T, H_k, W_k)
    v = torch.randn(B, C_v, T, H_k, W_k)

    with torch.no_grad():
        out_none = block_none(q, k, v)
        out_large = block_large(q, k, v)

    assert torch.allclose(out_none, out_large, atol=1e-6), \
        f"Max diff: {(out_none - out_large).abs().max().item()}"
    print("test_window_t_none_vs_large_window passed")


if __name__ == "__main__":
    test_output_shape()
    test_single_frame_matches_2d()
    test_window_t_none_vs_large_window()
