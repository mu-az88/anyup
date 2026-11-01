"""
T03 pytest: unified upsampler wrapper (anyup/upsamplers.py).

CPU only, random-init models, tiny sizes, no downloads (AnyUp2D gets a local
random state dict via `weights=`). Run on Colab:

    python -m pytest tests/test_upsamplers.py -v
"""

import os
import pickle
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import torch.nn.functional as F

from anyup.backbones import BackboneOutput
from anyup.model import AnyUp
from anyup.model_2d import AnyUp2D
from anyup.upsamplers import (ANYUP3D_ARCH, TStage, UpsampleOutput, UpsamplerLoader,
                              load_anyup3d_checkpoint)

SEED = 0
T, C, h, w = 4, 16, 4, 4          # ↓ tiny chunk: T feature frames, C channels, h×w grid (CPU speed)
GH = GW = 16                      # ↓ guide size = default output size; AnyUp3D attention cost grows with T·GH·GW
SMALL_ARCH = dict(qk_dim=32)      # ↓ qk_dim 32 (GroupNorm needs a multiple of 8; num_heads 4 must divide it)
REL_TOL, MIN_COS = 1e-4, 0.99999  # q_chunk equivalence tolerance (relative error + worst-frame cosine, T02 Q22)


# ─── helpers ────────────────────────────────────────────────────────────────

def make_chunk(t=T, seed=SEED):
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn(t, C, h, w, generator=g)                  # (T, C, h, w) raw
    guide = torch.randn(t, 3, GH, GW, generator=g)                # (T, 3, gH, gW) "normalized"
    meta = dict(name="fake", t_out=t, frame_index=list(range(t)), chunk_id=0)
    return BackboneOutput(features=feats, guide=guide, meta=meta)


def save_anyup3d_ckpt(path, sd=None, extra=None, **arch):
    if sd is None:
        torch.manual_seed(SEED)
        sd = AnyUp(**{**ANYUP3D_ARCH, **SMALL_ARCH, **arch}).state_dict()
    ckpt = {"anyup": sd, "global_step": 123}
    ckpt.update(extra or {})
    torch.save(ckpt, path)
    return path


@dataclass
class _TrainTStage:
    """Same shape as Train_3D_full.ipynb's dataclass TStage."""
    t: int
    start_step: int
    batch_size: int


@pytest.fixture
def ckpt_with_tstage(tmp_path, monkeypatch):
    """Checkpoint pickling a dataclass under the global name __main__.TStage (as the real ones)."""
    monkeypatch.setattr(_TrainTStage, "__module__", "__main__")
    monkeypatch.setattr(_TrainTStage, "__qualname__", "TStage")
    monkeypatch.setattr(sys.modules["__main__"], "TStage", _TrainTStage, raising=False)
    cfg = {"window_t": 4, "t_schedule": [_TrainTStage(2, 0, 2), _TrainTStage(4, 5000, 1)]}
    path = save_anyup3d_ckpt(tmp_path / "ckpt.pth", extra={"cfg": cfg})
    monkeypatch.delattr(sys.modules["__main__"], "TStage")   # loading must not need a caller-side TStage
    return path


@pytest.fixture
def anyup3d(ckpt_with_tstage):
    return UpsamplerLoader.load("anyup3d", ckpt_path=ckpt_with_tstage, device="cpu", **SMALL_ARCH)


@pytest.fixture
def anyup2d_weights(tmp_path):
    torch.manual_seed(SEED)
    path = tmp_path / "anyup2d.pth"
    torch.save(AnyUp2D().state_dict(), path)
    return path


def rel_cos(a, b):
    rel = ((a - b).abs().max() / a.abs().max()).item()
    cos = F.cosine_similarity(a.flatten(1), b.flatten(1), dim=1).min().item()
    return rel, cos


# ─── registry ───────────────────────────────────────────────────────────────

def test_available():
    assert UpsamplerLoader.available() == ["anyup2d", "anyup3d", "bilinear"]


def test_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown upsampler"):
        UpsamplerLoader.load("nope")


# ─── shapes + meta ──────────────────────────────────────────────────────────

def _check_output(res, chunk, name, out_hw):
    assert isinstance(res, UpsampleOutput)
    assert res.features.shape == (chunk.features.shape[0], C, *out_hw)
    assert res.features.dtype == torch.float32
    assert res.features.device == chunk.features.device
    for k, v in chunk.meta.items():
        assert res.meta[k] == v
    assert res.meta["upsampler"] == name
    assert res.meta["output_size"] == out_hw


@pytest.mark.parametrize("output_size, out_hw", [(None, (GH, GW)), (8, (8, 8)), ((12, 20), (12, 20))])
def test_shapes_bilinear(output_size, out_hw):
    up = UpsamplerLoader.load("bilinear", device="cpu")
    chunk = make_chunk()
    _check_output(up(chunk, output_size=output_size), chunk, "bilinear", out_hw)


@pytest.mark.parametrize("output_size, out_hw", [(None, (GH, GW)), ((12, 20), (12, 20))])
def test_shapes_anyup3d(anyup3d, ckpt_with_tstage, output_size, out_hw):
    chunk = make_chunk()
    res = anyup3d(chunk, output_size=output_size)
    _check_output(res, chunk, "anyup3d", out_hw)
    assert res.meta["ckpt_path"] == str(ckpt_with_tstage)
    assert res.meta["ckpt_step"] == 123
    assert res.meta["q_chunk_size"] is None
    assert res.meta["arch"] == {**ANYUP3D_ARCH, **SMALL_ARCH}
    assert res.meta["arch"]["window_t"] == 4


@pytest.mark.parametrize("output_size, out_hw", [(None, (GH, GW)), ((12, 20), (12, 20))])
def test_shapes_anyup2d(anyup2d_weights, output_size, out_hw):
    up = UpsamplerLoader.load("anyup2d", device="cpu", weights=anyup2d_weights)
    chunk = make_chunk()
    res = up(chunk, output_size=output_size)
    _check_output(res, chunk, "anyup2d", out_hw)
    assert res.meta["weights"] == anyup2d_weights


def test_meta_is_a_copy():
    up = UpsamplerLoader.load("bilinear", device="cpu")
    chunk = make_chunk()
    up(chunk)
    assert "upsampler" not in chunk.meta


def test_mismatched_t_raises():
    up = UpsamplerLoader.load("bilinear", device="cpu")
    chunk = make_chunk()
    chunk.guide = chunk.guide[:-1]
    with pytest.raises(ValueError, match="same T"):
        up(chunk)


# ─── numerics ───────────────────────────────────────────────────────────────

def test_bilinear_exact():
    up = UpsamplerLoader.load("bilinear", device="cpu")
    chunk = make_chunk()
    ref = F.interpolate(chunk.features, size=(GH, GW), mode="bilinear", align_corners=False)
    assert torch.equal(up(chunk).features, ref)


def test_anyup3d_matches_direct_call(anyup3d):
    """Wrapper = one model call on (1, C, T, h, w), guide first (Exp 2 run_anyup3d)."""
    chunk = make_chunk()
    with torch.no_grad():
        ref = anyup3d.model(chunk.guide.permute(1, 0, 2, 3).unsqueeze(0),
                            chunk.features.permute(1, 0, 2, 3).unsqueeze(0))       # (1, C, T, GH, GW)
    ref = ref.squeeze(0).permute(1, 0, 2, 3)                                         # (T, C, GH, GW)
    assert torch.equal(anyup3d(chunk).features, ref)


def test_anyup3d_one_call_uses_temporal_context(anyup3d):
    """All T frames go into one call: changing frame 0 changes the output of frame 1 (t_k=3, window_t=4)."""
    a = make_chunk()
    b = make_chunk()
    b.features[0] += 1.0
    out_a, out_b = anyup3d(a).features, anyup3d(b).features
    assert not torch.equal(out_a[1], out_b[1])


def test_anyup3d_q_chunk_equivalence(ckpt_with_tstage):
    full = UpsamplerLoader.load("anyup3d", ckpt_path=ckpt_with_tstage, device="cpu", **SMALL_ARCH)
    qc = UpsamplerLoader.load("anyup3d", ckpt_path=ckpt_with_tstage, device="cpu",
                              q_chunk_size=100, **SMALL_ARCH)     # T·GH·GW = 1024 queries → 11 chunks
    chunk = make_chunk()
    a, b = full(chunk).features, qc(chunk).features
    rel, cos = rel_cos(a, b)
    assert rel < REL_TOL and cos > MIN_COS, (rel, cos)
    assert qc(chunk).meta["q_chunk_size"] == 100


def test_anyup2d_loop_exact(anyup2d_weights):
    """Wrapper = calling the model frame by frame by hand (Exp 2 run_anyup2d)."""
    up = UpsamplerLoader.load("anyup2d", device="cpu", weights=anyup2d_weights)
    chunk = make_chunk()
    with torch.no_grad():
        ref = torch.stack([up.model(chunk.guide[t].unsqueeze(0), chunk.features[t].unsqueeze(0)).squeeze(0)
                           for t in range(T)])                                       # (T, C, GH, GW)
    assert torch.equal(up(chunk).features, ref)


def test_models_frozen(anyup3d, anyup2d_weights):
    up2 = UpsamplerLoader.load("anyup2d", device="cpu", weights=anyup2d_weights)
    for up in (anyup3d, up2):
        assert not up.model.training
        assert all(not p.requires_grad for p in up.model.parameters())


# ─── checkpoint loading ─────────────────────────────────────────────────────

def test_tstage_checkpoint_loads_safely(ckpt_with_tstage):
    ckpt = load_anyup3d_checkpoint(ckpt_with_tstage)
    stages = ckpt["cfg"]["t_schedule"]
    assert all(type(s) is TStage for s in stages)
    assert [(s.t, s.start_step, s.batch_size) for s in stages] == [(2, 0, 2), (4, 5000, 1)]
    assert ckpt["global_step"] == 123


class _NotAllowed:
    pass


def test_non_allowlisted_object_fails(tmp_path):
    path = save_anyup3d_ckpt(tmp_path / "bad.pth", extra={"obj": _NotAllowed()})
    with pytest.raises(pickle.UnpicklingError):
        load_anyup3d_checkpoint(path)


def test_missing_anyup_key_raises(tmp_path):
    path = tmp_path / "nokey.pth"
    torch.save({"model_state_dict": {}, "global_step": 1}, path)
    with pytest.raises(KeyError, match="model_state_dict"):
        load_anyup3d_checkpoint(path)


def test_strict_missing_key_raises(tmp_path):
    torch.manual_seed(SEED)
    sd = AnyUp(**{**ANYUP3D_ARCH, **SMALL_ARCH}).state_dict()
    sd.pop(next(iter(sd)))
    path = save_anyup3d_ckpt(tmp_path / "missing.pth", sd=sd)
    with pytest.raises(RuntimeError, match="Missing key"):
        UpsamplerLoader.load("anyup3d", ckpt_path=path, device="cpu", **SMALL_ARCH)


def test_strict_unexpected_key_raises(tmp_path):
    torch.manual_seed(SEED)
    sd = AnyUp(**{**ANYUP3D_ARCH, **SMALL_ARCH}).state_dict()
    sd["extra.weight"] = torch.zeros(1)
    path = save_anyup3d_ckpt(tmp_path / "unexpected.pth", sd=sd)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        UpsamplerLoader.load("anyup3d", ckpt_path=path, device="cpu", **SMALL_ARCH)


def test_arch_mismatch_raises(tmp_path):
    """qk_dim 32 checkpoint into the default qk_dim 128 model → size mismatch."""
    path = save_anyup3d_ckpt(tmp_path / "small.pth")
    with pytest.raises(RuntimeError, match="size mismatch"):
        UpsamplerLoader.load("anyup3d", ckpt_path=path, device="cpu")


def test_unknown_arch_arg_raises(ckpt_with_tstage):
    with pytest.raises(TypeError, match="unknown AnyUp3D arch"):
        UpsamplerLoader.load("anyup3d", ckpt_path=ckpt_with_tstage, device="cpu", window=4)
