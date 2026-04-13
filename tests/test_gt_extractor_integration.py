"""
tests/test_gt_extractor_integration.py

Integration tests for build_gt_extractor with the real VideoMAE backbone.
Requires downloading MCG-NJU/videomae-base (~350 MB from HuggingFace).

These tests are guarded by @pytest.mark.slow — run with:
    pytest tests/test_gt_extractor_integration.py -v -m slow

To run ALL tests including slow ones:
    pytest -v -m slow
"""

import sys
import types
from types import SimpleNamespace

import pytest
import torch

# ── Check if the *real* transformers is available with VideoMAE ────────────────
# We can't use importlib.util.find_spec because other test files may have
# injected a stub ModuleType with no __spec__, which makes find_spec crash.
# Instead, temporarily remove any stub, probe, and restore.
_saved_tf = sys.modules.pop("transformers", None)
try:
    import importlib.util
    _HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
except Exception:
    _HAS_TRANSFORMERS = False
finally:
    if _saved_tf is not None:
        sys.modules["transformers"] = _saved_tf

# ── Import build_gt_extractor from train_3d.py via exec (same pattern as
#    test_train_infrastructure.py) ──────────────────────────────────────────────
import pathlib

# Stub only the modules that don't exist yet — leave real packages alone.
from unittest.mock import MagicMock

for _parent in ["anyup.modules", "anyup.data", "scripts"]:
    if _parent not in sys.modules:
        sys.modules[_parent] = types.ModuleType(_parent)

_STUB_MODULES = [
    "anyup.modules.anyup3d", "anyup.modules.losses3d",
    "scripts.load_2d_weights",
    "anyup.data.video_dataset",
]
for _mod in _STUB_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

# If the real transformers is available, force-reload it so that stubs
# injected by other test files (e.g. test_train_infrastructure.py) don't
# contaminate the exec namespace. If not available, stub it out.
if _HAS_TRANSFORMERS:
    import importlib
    if "transformers" in sys.modules:
        importlib.reload(sys.modules["transformers"])
    else:
        import transformers  # noqa: F401
else:
    if "transformers" not in sys.modules:
        sys.modules["transformers"] = types.ModuleType("transformers")
    sys.modules["transformers"].VideoMAEModel = MagicMock()
    sys.modules["transformers"].AutoImageProcessor = MagicMock()

if "omegaconf" not in sys.modules:
    sys.modules["omegaconf"] = types.ModuleType("omegaconf")
if "torch.utils.tensorboard" not in sys.modules:
    sys.modules["torch.utils.tensorboard"] = types.ModuleType("torch.utils.tensorboard")

sys.modules["anyup.modules.anyup3d"].AnyUp3D = MagicMock()
sys.modules["anyup.modules.losses3d"].combined_loss_3d = None
sys.modules["scripts.load_2d_weights"].load_2d_weights_into_3d = None
sys.modules["torch.utils.tensorboard"].SummaryWriter = MagicMock()
sys.modules["omegaconf"].OmegaConf = MagicMock()

_train_src = (pathlib.Path(__file__).parent.parent / "train_3d.py").read_text()
_g = {"__name__": "__test__", "__file__": str(pathlib.Path(__file__).parent.parent / "train_3d.py")}
exec(compile(_train_src, "train_3d.py", "exec"), _g)

build_gt_extractor = _g["build_gt_extractor"]


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.skipif(not _HAS_TRANSFORMERS, reason="transformers not installed")
class TestGTExtractorIntegration:
    """Integration tests that download and run the real VideoMAE backbone."""

    @pytest.fixture(scope="class")
    def extractor(self):
        """Build the GT extractor once for the whole class (avoids re-downloading)."""
        # Inject real transformers into the exec namespace so build_gt_extractor
        # picks up the real VideoMAEModel, not the stub.
        from transformers import VideoMAEModel, AutoImageProcessor
        _g["VideoMAEModel"] = VideoMAEModel
        _g["AutoImageProcessor"] = AutoImageProcessor

        cfg = SimpleNamespace(encoder="videomae", encoder_model="MCG-NJU/videomae-base")
        device = torch.device("cpu")
        extract = build_gt_extractor(cfg, device)
        yield extract

    # VideoMAE-base has fixed position embeddings for num_frames=16.
    # T_tok = 16//2 = 8, H_tok = 224//16 = 14, W_tok = 224//16 = 14.

    def test_output_shape_T16(self, extractor):
        """(1, 3, 16, 224, 224) → (1, 768, 8, 14, 14)"""
        video = torch.rand(1, 3, 16, 224, 224)
        feats = extractor(video)
        assert feats.shape == (1, 768, 8, 14, 14), f"Got {feats.shape}"

    def test_output_shape_batch2(self, extractor):
        """Batch size > 1 must work."""
        video = torch.rand(2, 3, 16, 224, 224)
        feats = extractor(video)
        assert feats.shape == (2, 768, 8, 14, 14), f"Got {feats.shape}"

    def test_output_dtype_float32(self, extractor):
        """Output should be float32 (same as input)."""
        video = torch.rand(1, 3, 16, 224, 224)
        feats = extractor(video)
        assert feats.dtype == torch.float32, f"Got {feats.dtype}"

    def test_output_no_nan(self, extractor):
        """Output must not contain NaN or Inf."""
        video = torch.rand(1, 3, 16, 224, 224)
        feats = extractor(video)
        assert not torch.isnan(feats).any(), "Output contains NaN"
        assert not torch.isinf(feats).any(), "Output contains Inf"

    def test_output_is_contiguous(self, extractor):
        """Output must be contiguous for downstream .view() calls."""
        video = torch.rand(1, 3, 16, 224, 224)
        feats = extractor(video)
        assert feats.is_contiguous(), "Output is not contiguous"

    def test_deterministic_output(self, extractor):
        """Same input must produce identical output (model is frozen)."""
        torch.manual_seed(42)
        video = torch.rand(1, 3, 16, 224, 224)
        feats1 = extractor(video)
        feats2 = extractor(video)
        assert torch.equal(feats1, feats2), "Same input produced different outputs"

    def test_no_gradients_on_output(self, extractor):
        """Extractor runs under no_grad — output should not require grad."""
        video = torch.rand(1, 3, 16, 224, 224)
        feats = extractor(video)
        assert not feats.requires_grad, "Output should not require gradients"
