"""
tests/test_seed.py
Unit tests for anyup/utils/seed.py (Phase 7.3).

Covers:
  - Same seed → identical tensor samples from torch, numpy, random
  - Different seeds → different samples
  - CUDA seeds set correctly when GPU is available
  - deterministic=False skips cuDNN determinism flags without crashing

Run:
    pytest tests/test_seed.py -v
    pytest tests/test_seed.py -v -k "test_same_seed"
"""

import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, ".")
from anyup.utils.seed import set_seed


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _sample_all():
    """Draw one sample from each RNG after seed is set. Returns a tuple."""
    t = torch.randn(4).tolist()
    n = np.random.rand(4).tolist()
    r = [random.random() for _ in range(4)]
    return t, n, r


# ══════════════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════════════

def test_same_seed_torch_reproducible():
    """Two calls to set_seed with the same value produce identical torch samples."""
    set_seed(42)
    a = torch.randn(8)
    set_seed(42)
    b = torch.randn(8)
    assert torch.equal(a, b), f"Expected identical tensors, max diff: {(a - b).abs().max()}"
    print("test_same_seed_torch_reproducible passed")


def test_same_seed_numpy_reproducible():
    """Two calls to set_seed with the same value produce identical numpy samples."""
    set_seed(42)
    a = np.random.rand(8)
    set_seed(42)
    b = np.random.rand(8)
    assert np.array_equal(a, b), "numpy samples differ across identical seeds"
    print("test_same_seed_numpy_reproducible passed")


def test_same_seed_random_reproducible():
    """Two calls to set_seed with the same value produce identical Python random samples."""
    set_seed(42)
    a = [random.random() for _ in range(8)]
    set_seed(42)
    b = [random.random() for _ in range(8)]
    assert a == b, "random samples differ across identical seeds"
    print("test_same_seed_random_reproducible passed")


def test_different_seeds_diverge():
    """Different seeds must produce different torch samples."""
    set_seed(0)
    a = torch.randn(16)
    set_seed(1)
    b = torch.randn(16)
    assert not torch.equal(a, b), "Different seeds produced identical samples — RNG may be broken"
    print("test_different_seeds_diverge passed")


def test_all_rngs_in_sync():
    """
    After set_seed, all three RNGs (torch, numpy, random) draw reproducibly
    in the same interleaved sequence across two identical runs.
    """
    set_seed(99)
    run_a = _sample_all()

    set_seed(99)
    run_b = _sample_all()

    assert run_a[0] == run_b[0], "torch diverged in interleaved draw"
    assert run_a[1] == run_b[1], "numpy diverged in interleaved draw"
    assert run_a[2] == run_b[2], "random diverged in interleaved draw"
    print("test_all_rngs_in_sync passed")


def test_cuda_seed_set_when_available():
    """
    When CUDA is available, set_seed must set the CUDA seed so that
    GPU tensor generation is also reproducible.
    """
    if not torch.cuda.is_available():
        pytest.skip("No CUDA device — skipping GPU seed test")

    device = torch.device("cuda")
    set_seed(7)
    a = torch.randn(8, device=device)
    set_seed(7)
    b = torch.randn(8, device=device)
    assert torch.equal(a, b), f"CUDA RNG not reproducible — max diff: {(a - b).abs().max()}"
    print("test_cuda_seed_set_when_available passed")


def test_nondeterministic_mode_does_not_crash():
    """
    set_seed(seed, deterministic=False) must not raise even when cuDNN
    determinism flags are intentionally left off.
    """
    try:
        set_seed(42, deterministic=False)
    except Exception as e:
        pytest.fail(f"set_seed(deterministic=False) raised unexpectedly: {e}")

    # Basic sanity: torch still works after the call
    t = torch.randn(4)
    assert t.shape == (4,)
    print("test_nondeterministic_mode_does_not_crash passed")


def test_seed_zero_valid():
    """Seed=0 is a valid edge case and must not crash or produce all-zeros."""
    set_seed(0)
    t = torch.randn(8)
    assert not torch.all(t == 0), "seed=0 produced all-zero tensor"
    print("test_seed_zero_valid passed")


# ══════════════════════════════════════════════════════════════════════════════
# __main__ runner
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    test_same_seed_torch_reproducible()
    test_same_seed_numpy_reproducible()
    test_same_seed_random_reproducible()
    test_different_seeds_diverge()
    test_all_rngs_in_sync()
    test_cuda_seed_set_when_available()
    test_nondeterministic_mode_does_not_crash()
    test_seed_zero_valid()
    print("\nAll seed tests passed.")