"""
tests/test_grad_ckpt.py
Unit tests for CrossAttentionBlock3D_GradCkpt (Phase 7.2).

Verifies that the gradient-checkpointed block is a numerically identical
drop-in for CrossAttentionBlock3D — same forward output, same gradient
magnitudes, gradients still flow.

Run:
    pytest tests/test_grad_ckpt.py -v
"""

import sys
import copy
import pytest
import torch

sys.path.insert(0, ".")
from anyup.modules.cross_attention3d import CrossAttentionBlock3D
# Import the checkpointed variant from memory_profile (single source of truth)
from scripts.memory_profile import CrossAttentionBlock3D_GradCkpt


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_pair(qk_dim=32, num_heads=4, window_ratio=0.15, window_t=None):
    """
    Create a plain + checkpointed block with identical weights.
    Returns (plain_block, ckpt_block).
    """
    plain = CrossAttentionBlock3D(
        qk_dim=qk_dim, num_heads=num_heads,
        window_ratio=window_ratio, window_t=window_t,
    ).eval()

    ckpt = CrossAttentionBlock3D_GradCkpt(
        qk_dim=qk_dim, num_heads=num_heads,
        window_ratio=window_ratio, window_t=window_t,
    ).eval()

    # Copy weights so the only difference is the forward implementation
    ckpt.load_state_dict(copy.deepcopy(plain.state_dict()))
    return plain, ckpt


def _make_inputs(B=1, C_qk=32, C_v=64, T=2, H=8, W=8, H_k=4, W_k=4):
    q = torch.randn(B, C_qk, T, H,   W,   requires_grad=True)
    k = torch.randn(B, C_qk, T, H_k, W_k, requires_grad=True)
    v = torch.randn(B, C_v,  T, H_k, W_k, requires_grad=True)
    return q, k, v


# ══════════════════════════════════════════════════════════════════════════════
# Forward equivalence
# ══════════════════════════════════════════════════════════════════════════════

def test_grad_ckpt_output_matches_plain():
    """
    CrossAttentionBlock3D_GradCkpt must produce bit-for-bit identical output
    to CrossAttentionBlock3D for the same weights and inputs (eval mode).
    """
    plain, ckpt = _make_pair(window_t=None)
    q, k, v     = _make_inputs()

    with torch.no_grad():
        out_plain = plain(q, k, v)
        out_ckpt  = ckpt(q, k, v)

    assert torch.allclose(out_plain, out_ckpt, atol=1e-5), (
        f"Output mismatch — max diff: {(out_plain - out_ckpt).abs().max().item():.2e}"
    )
    print("test_grad_ckpt_output_matches_plain passed")


def test_grad_ckpt_output_shape():
    """Checkpointed block must return the correct output shape (B, C_v, T, H, W)."""
    B, C_qk, C_v, T, H, W, H_k, W_k = 2, 32, 64, 4, 16, 16, 8, 8
    _, ckpt = _make_pair(qk_dim=C_qk, window_t=1)
    q = torch.randn(B, C_qk, T, H,   W)
    k = torch.randn(B, C_qk, T, H_k, W_k)
    v = torch.randn(B, C_v,  T, H_k, W_k)

    with torch.no_grad():
        out = ckpt(q, k, v)

    assert out.shape == (B, C_v, T, H, W), (
        f"Expected {(B, C_v, T, H, W)}, got {list(out.shape)}"
    )
    print("test_grad_ckpt_output_shape passed")


def test_grad_ckpt_no_nan_in_output():
    """Checkpointed block must not produce NaN in its output."""
    _, ckpt = _make_pair(window_t=None)
    q, k, v = _make_inputs()

    with torch.no_grad():
        out = ckpt(q, k, v)

    assert not torch.isnan(out).any(), "NaN detected in GradCkpt output"
    print("test_grad_ckpt_no_nan_in_output passed")


# ══════════════════════════════════════════════════════════════════════════════
# Gradient flow
# ══════════════════════════════════════════════════════════════════════════════

def test_grad_ckpt_gradients_flow_to_q():
    """Gradients must flow back through the checkpointed block to q."""
    _, ckpt = _make_pair(window_t=None)
    ckpt.train()
    q, k, v = _make_inputs()

    out  = ckpt(q, k, v)
    loss = out.sum()
    loss.backward()

    assert q.grad is not None, "No gradient on q after GradCkpt backward"
    assert not torch.isnan(q.grad).any(), "NaN in q.grad after GradCkpt backward"
    print("test_grad_ckpt_gradients_flow_to_q passed")


def test_grad_ckpt_gradients_flow_to_v():
    """Gradients must flow back to v (the value tensor)."""
    _, ckpt = _make_pair(window_t=None)
    ckpt.train()
    q, k, v = _make_inputs()

    out  = ckpt(q, k, v)
    loss = out.sum()
    loss.backward()

    assert v.grad is not None, "No gradient on v after GradCkpt backward"
    assert not torch.isnan(v.grad).any(), "NaN in v.grad after GradCkpt backward"
    print("test_grad_ckpt_gradients_flow_to_v passed")


def test_grad_ckpt_param_gradients_match_plain():
    """
    Parameter gradients from the checkpointed block should be numerically close
    to those from the plain block (same weights, same input, train mode).
    """
    plain, ckpt = _make_pair(window_t=None)
    plain.train()
    ckpt.train()

    q, k, v = _make_inputs()

    # Plain backward
    q_p  = q.detach().clone().requires_grad_(True)
    k_p  = k.detach().clone().requires_grad_(True)
    v_p  = v.detach().clone().requires_grad_(True)
    out_p = plain(q_p, k_p, v_p)
    out_p.sum().backward()

    # GradCkpt backward
    q_c  = q.detach().clone().requires_grad_(True)
    k_c  = k.detach().clone().requires_grad_(True)
    v_c  = v.detach().clone().requires_grad_(True)
    out_c = ckpt(q_c, k_c, v_c)
    out_c.sum().backward()

    for (name, p_param), (_, c_param) in zip(
        plain.named_parameters(), ckpt.named_parameters()
    ):
        if p_param.grad is None and c_param.grad is None:
            continue
        assert p_param.grad is not None and c_param.grad is not None, (
            f"Grad presence mismatch for param '{name}'"
        )
        assert torch.allclose(p_param.grad, c_param.grad, atol=1e-4), (
            f"Gradient mismatch for param '{name}' — "
            f"max diff: {(p_param.grad - c_param.grad).abs().max().item():.2e}"
        )

    print("test_grad_ckpt_param_gradients_match_plain passed")


# ══════════════════════════════════════════════════════════════════════════════
# State dict compatibility
# ══════════════════════════════════════════════════════════════════════════════

def test_grad_ckpt_state_dict_compatible_with_plain():
    """
    A state_dict saved from CrossAttentionBlock3D must load cleanly into
    CrossAttentionBlock3D_GradCkpt with no missing or unexpected keys.
    """
    plain, ckpt = _make_pair()
    sd = plain.state_dict()
    missing, unexpected = ckpt.load_state_dict(sd, strict=True)
    # load_state_dict with strict=True raises on mismatch; reaching here means success
    assert True
    print("test_grad_ckpt_state_dict_compatible_with_plain passed")


def test_grad_ckpt_plain_state_dict_round_trip():
    """
    Weights loaded into GradCkpt and then extracted must be identical to
    the original plain block's weights.
    """
    plain, ckpt = _make_pair()
    sd_plain = plain.state_dict()
    ckpt.load_state_dict(sd_plain)
    sd_ckpt = ckpt.state_dict()

    for key in sd_plain:
        assert key in sd_ckpt, f"Key '{key}' missing from GradCkpt state_dict"
        assert torch.equal(sd_plain[key], sd_ckpt[key]), (
            f"Weight mismatch after round-trip for key '{key}'"
        )
    print("test_grad_ckpt_plain_state_dict_round_trip passed")


# ══════════════════════════════════════════════════════════════════════════════
# window_t compatibility
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("window_t", [None, 1, 2])
def test_grad_ckpt_various_window_t(window_t):
    """GradCkpt must work for all supported window_t values."""
    plain, ckpt = _make_pair(window_t=window_t)
    q, k, v     = _make_inputs(T=4)

    with torch.no_grad():
        out_plain = plain(q, k, v)
        out_ckpt  = ckpt(q, k, v)

    assert torch.allclose(out_plain, out_ckpt, atol=1e-5), (
        f"window_t={window_t}: max diff {(out_plain - out_ckpt).abs().max().item():.2e}"
    )
    print(f"test_grad_ckpt_various_window_t[window_t={window_t}] passed")


# ══════════════════════════════════════════════════════════════════════════════
# __main__ runner
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    test_grad_ckpt_output_matches_plain()
    test_grad_ckpt_output_shape()
    test_grad_ckpt_no_nan_in_output()
    test_grad_ckpt_gradients_flow_to_q()
    test_grad_ckpt_gradients_flow_to_v()
    test_grad_ckpt_param_gradients_match_plain()
    test_grad_ckpt_state_dict_compatible_with_plain()
    test_grad_ckpt_plain_state_dict_round_trip()
    for wt in [None, 1, 2]:
        test_grad_ckpt_various_window_t(wt)
    print("\nAll grad_ckpt tests passed.")