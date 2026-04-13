"""
tests/test_smoke_pipeline.py
Unit tests for the _MinimalPipeline and guard functions in scripts/smoke_test.py
(Phase 7.1).

Covers:
  - _MinimalPipeline: correct output shape, no NaN, gradients flow
  - make_batch: correct tensor shapes, correct device placement
  - _check_no_nan: raises on NaN, raises on Inf, passes on clean tensor
  - _check_shape: raises on mismatch, passes on match
  - _check_grad: raises when no grad, passes after backward
  - Full 3-step integration: forward → loss → backward → optimizer step

Run:
    pytest tests/test_smoke_pipeline.py -v
"""

import sys
import pytest
import torch

sys.path.insert(0, ".")
from scripts.smoke_test import (
    _MinimalPipeline,
    make_batch,
    combined_loss,
    _check_no_nan,
    _check_shape,
    _check_grad,
)
from anyup.utils.seed import set_seed


# ══════════════════════════════════════════════════════════════════════════════
# make_batch
# ══════════════════════════════════════════════════════════════════════════════

def test_make_batch_shapes():
    """make_batch must produce tensors with the documented shapes."""
    B, C_feat, T, H, W, s = 2, 64, 4, 56, 56, 4
    batch = make_batch(B, C_feat, T, H, W, C_img=3, device="cpu")

    assert batch["rgb_hires"].shape  == (B, 3,      T, H,    W),    "rgb_hires shape wrong"
    assert batch["rgb_lores"].shape  == (B, 3,      T, H//s, W//s), "rgb_lores shape wrong"
    assert batch["feat_lores"].shape == (B, C_feat, T, H//s, W//s), "feat_lores shape wrong"
    assert batch["feat_hires"].shape == (B, C_feat, T, H,    W),    "feat_hires shape wrong"
    print("test_make_batch_shapes passed")


def test_make_batch_device_placement():
    """make_batch tensors must live on the requested device."""
    batch = make_batch(2, 64, 4, 56, 56, device="cpu")
    for key, t in batch.items():
        assert t.device.type == "cpu", f"Tensor '{key}' not on cpu"
    print("test_make_batch_device_placement passed")


def test_make_batch_no_nan():
    """make_batch tensors must not contain NaN (randn is well-behaved)."""
    batch = make_batch(2, 64, 4, 56, 56, device="cpu")
    for key, t in batch.items():
        assert not torch.isnan(t).any(), f"NaN in make_batch['{key}']"
    print("test_make_batch_no_nan passed")


# ══════════════════════════════════════════════════════════════════════════════
# _check_no_nan
# ══════════════════════════════════════════════════════════════════════════════

def test_check_no_nan_passes_on_clean():
    """_check_no_nan must not raise on a normal tensor."""
    t = torch.randn(4, 4)
    _check_no_nan(t, "clean_tensor")    # should not raise
    print("test_check_no_nan_passes_on_clean passed")


def test_check_no_nan_raises_on_nan():
    """_check_no_nan must raise ValueError when tensor contains NaN."""
    t = torch.tensor([1.0, float("nan"), 3.0])
    with pytest.raises(ValueError, match="NaN"):
        _check_no_nan(t, "nan_tensor")
    print("test_check_no_nan_raises_on_nan passed")


def test_check_no_nan_raises_on_inf():
    """_check_no_nan must raise ValueError when tensor contains Inf."""
    t = torch.tensor([1.0, float("inf"), 3.0])
    with pytest.raises(ValueError, match="Inf"):
        _check_no_nan(t, "inf_tensor")
    print("test_check_no_nan_raises_on_inf passed")


# ══════════════════════════════════════════════════════════════════════════════
# _check_shape
# ══════════════════════════════════════════════════════════════════════════════

def test_check_shape_passes_on_match():
    """_check_shape must not raise when shape matches."""
    t = torch.zeros(2, 64, 4, 56, 56)
    _check_shape(t, (2, 64, 4, 56, 56), "matching_tensor")    # should not raise
    print("test_check_shape_passes_on_match passed")


def test_check_shape_raises_on_mismatch():
    """_check_shape must raise ValueError when shape does not match."""
    t = torch.zeros(2, 64, 4, 56, 56)
    with pytest.raises(ValueError, match="Shape mismatch"):
        _check_shape(t, (2, 64, 4, 28, 28), "wrong_shape_tensor")
    print("test_check_shape_raises_on_mismatch passed")


# ══════════════════════════════════════════════════════════════════════════════
# _check_grad
# ══════════════════════════════════════════════════════════════════════════════

def test_check_grad_raises_when_no_grad():
    """_check_grad must raise RuntimeError when no parameter has a gradient."""
    model = torch.nn.Linear(4, 4)
    # Gradients were never computed
    with pytest.raises(RuntimeError, match="no gradients"):
        _check_grad(model, step=0)
    print("test_check_grad_raises_when_no_grad passed")


def test_check_grad_passes_after_backward():
    """_check_grad must not raise after a successful backward pass."""
    model = torch.nn.Linear(4, 4)
    x     = torch.randn(2, 4)
    loss  = model(x).sum()
    loss.backward()
    _check_grad(model, step=0)    # should not raise
    print("test_check_grad_passes_after_backward passed")


# ══════════════════════════════════════════════════════════════════════════════
# _MinimalPipeline
# ══════════════════════════════════════════════════════════════════════════════

def _make_pipeline_and_batch(B=1, C_feat=32, T=2, H=28, W=28):
    """Helper: create model + batch at small scale for fast unit tests."""
    model = _MinimalPipeline(
        C_img=3, C_feat=C_feat, C_qk=C_feat,
        num_heads=4,
        window_ratio=0.15,
        window_t=None,
    )
    batch = make_batch(B, C_feat, T, H, W, C_img=3, device="cpu")
    return model, batch


def test_minimal_pipeline_output_shape():
    """_MinimalPipeline must return (B, C_feat, T, H, W)."""
    B, C_feat, T, H, W = 1, 32, 2, 28, 28
    model, batch = _make_pipeline_and_batch(B, C_feat, T, H, W)
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (B, C_feat, T, H, W), (
        f"Expected {(B, C_feat, T, H, W)}, got {list(out.shape)}"
    )
    print("test_minimal_pipeline_output_shape passed")


def test_minimal_pipeline_no_nan():
    """_MinimalPipeline forward must not produce NaN."""
    model, batch = _make_pipeline_and_batch()
    with torch.no_grad():
        out = model(batch)
    assert not torch.isnan(out).any(), "NaN in _MinimalPipeline output"
    print("test_minimal_pipeline_no_nan passed")


def test_minimal_pipeline_gradients_flow():
    """Gradients must flow back through _MinimalPipeline to its parameters."""
    model, batch = _make_pipeline_and_batch()
    model.train()
    out  = model(batch)
    loss = out.sum()
    loss.backward()
    _check_grad(model, step=0)    # will raise if no grads
    print("test_minimal_pipeline_gradients_flow passed")


def test_minimal_pipeline_different_T():
    """Pipeline must handle T=1 (warmup) and T=8 without crashing."""
    for T in [1, 8]:
        model, batch = _make_pipeline_and_batch(T=T)
        with torch.no_grad():
            out = model(batch)
        assert out.shape[2] == T, f"T={T}: output temporal dim wrong: {out.shape}"
    print("test_minimal_pipeline_different_T passed")


# ══════════════════════════════════════════════════════════════════════════════
# Integration: 3-step training loop
# ══════════════════════════════════════════════════════════════════════════════

def test_integration_3_step_loop():
    """
    Full mini integration test: 3 steps of forward → combined_loss → backward
    → optimizer.step(), with NaN and shape checks at every step.
    This mirrors exactly what smoke_test.py does, but at minimal scale.
    """
    set_seed(42)
    B, C_feat, T, H, W = 1, 32, 2, 28, 28

    model = _MinimalPipeline(
        C_img=3, C_feat=C_feat, C_qk=C_feat,
        num_heads=4, window_ratio=0.15, window_t=None,
    ).train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)

    for step in range(3):
        batch = make_batch(B, C_feat, T, H, W, device="cpu")
        optimizer.zero_grad()

        pred = model(batch)
        _check_shape(pred, (B, C_feat, T, H, W), f"step {step} output")
        _check_no_nan(pred, f"step {step} output")

        losses = combined_loss(pred, batch)
        for k, v in losses.items():
            _check_no_nan(v, f"step {step} loss/{k}")

        losses["total"].backward()
        _check_grad(model, step)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    print("test_integration_3_step_loop passed")


def test_integration_loss_decreases_direction():
    """
    Over 5 steps on a fixed batch (no data shuffling), total loss should
    have a downward trend — confirms optimizer is functioning.
    We check that the last step's loss is lower than the first.
    """
    set_seed(0)
    B, C_feat, T, H, W = 1, 32, 2, 28, 28

    model     = _MinimalPipeline(
        C_img=3, C_feat=C_feat, C_qk=C_feat,
        num_heads=4, window_ratio=0.15, window_t=None,
    ).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)   # high LR for fast descent
    batch     = make_batch(B, C_feat, T, H, W, device="cpu")     # fixed batch

    loss_history = []
    for _ in range(5):
        optimizer.zero_grad()
        pred = model(batch)
        losses = combined_loss(pred, batch)
        losses["total"].backward()
        optimizer.step()
        loss_history.append(losses["total"].item())

    assert loss_history[-1] < loss_history[0], (
        f"Loss did not decrease: {loss_history[0]:.4f} → {loss_history[-1]:.4f}"
    )
    print("test_integration_loss_decreases_direction passed")


# ══════════════════════════════════════════════════════════════════════════════
# __main__ runner
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    test_make_batch_shapes()
    test_make_batch_device_placement()
    test_make_batch_no_nan()

    test_check_no_nan_passes_on_clean()
    test_check_no_nan_raises_on_nan()
    test_check_no_nan_raises_on_inf()

    test_check_shape_passes_on_match()
    test_check_shape_raises_on_mismatch()

    test_check_grad_raises_when_no_grad()
    test_check_grad_passes_after_backward()

    test_minimal_pipeline_output_shape()
    test_minimal_pipeline_no_nan()
    test_minimal_pipeline_gradients_flow()
    test_minimal_pipeline_different_T()

    test_integration_3_step_loop()
    test_integration_loss_decreases_direction()

    print("\nAll smoke pipeline tests passed.")