"""
anyup/utils/seed.py
Phase 7.3 — Global reproducibility seed utility.

Call set_seed(seed) once at the top of train.py (before any model or data init).
The seed value should live in the training config so it is documented alongside
all other hyperparameters.

Usage in train.py:
    from anyup.utils.seed import set_seed
    set_seed(cfg.seed)          # immediately after cfg is loaded
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set seeds for Python random, numpy, and torch (CPU + GPU).

    Parameters
    ----------
    seed : int
        The seed value. Store this in your config so every run is documented.
    deterministic : bool
        If True, forces cuDNN into deterministic mode.
        ⚠️  May reduce throughput ~5-15% on some ops. Set False for speed if
        exact reproducibility across hardware isn't required.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)      # multi-GPU safety

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
        # CUBLAS workspace determinism (PyTorch ≥ 1.11)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        # warn_only=True: some ops have no deterministic kernel — they'll warn
        # rather than error, so training still runs. Flip to False for strict mode.