"""
tests/conftest.py
-----------------
Registers custom pytest CLI options used by test_weight_loading.py and
provides the shared ckpt_paths fixture.

Checkpoint resolution order:
  1. --ckpt2d / --ckpt3d CLI options
  2. ANYUP_CKPT_2D / ANYUP_CKPT_3D environment variables
  3. Default files in the project root:
       anyup_multi_backbone.pth  (2D)
       anyup3d_init.pth          (3D, auto-generated if missing)

If the 2D checkpoint is found but the 3D one is missing, load_2d_weights.py
is called automatically to produce it.  Tests are only skipped when no 2D
checkpoint can be located.
"""

import os
from pathlib import Path

import pytest

# Project root is one level above this file (tests/)
_PROJECT_ROOT = Path(__file__).parent.parent

_DEFAULT_CKPT_2D = _PROJECT_ROOT / "anyup_multi_backbone.pth"
_DEFAULT_CKPT_3D = _PROJECT_ROOT / "anyup3d_init.pth"


def pytest_addoption(parser):
    parser.addoption(
        "--ckpt2d",
        default=None,
        help="Path to the original 2D AnyUp checkpoint (.pth)",
    )
    parser.addoption(
        "--ckpt3d",
        default=None,
        help="Path to the adapted 3D checkpoint produced by load_2d_weights.py",
    )


@pytest.fixture(scope="session")
def ckpt_paths(request):
    """Resolve 2D and 3D checkpoint paths, auto-generating the 3D one if needed."""
    path_2d = (
        request.config.getoption("--ckpt2d", default=None)
        or os.environ.get("ANYUP_CKPT_2D")
        or (str(_DEFAULT_CKPT_2D) if _DEFAULT_CKPT_2D.exists() else None)
    )
    path_3d = (
        request.config.getoption("--ckpt3d", default=None)
        or os.environ.get("ANYUP_CKPT_3D")
        or (str(_DEFAULT_CKPT_3D) if _DEFAULT_CKPT_3D.exists() else None)
    )

    if not path_2d:
        pytest.skip(
            "No 2D checkpoint found. Pass --ckpt2d, set ANYUP_CKPT_2D, "
            f"or place anyup_multi_backbone.pth in {_PROJECT_ROOT}."
        )

    # Auto-generate the 3D checkpoint from the 2D one if it is missing.
    if not path_3d or not Path(path_3d).exists():
        out = str(_DEFAULT_CKPT_3D)
        print(f"\n[conftest] 3D checkpoint not found — generating {out} from {path_2d}")
        from scripts.load_2d_weights import load_2d_weights
        load_2d_weights(path_2d, out)
        path_3d = out

    return path_2d, path_3d
def pytest_addoption(parser):
    parser.addoption("--manifest_dir",     default="configs/")
    parser.addoption("--imagenet_val_dir", default=None)
