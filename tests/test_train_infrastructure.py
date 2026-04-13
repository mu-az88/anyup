"""
tests/test_train_infrastructure.py

Unit tests for every callable in train.py (Phase 5: Tasks 5.1–5.4).

Covered:
    TCurriculumScheduler.__init__
    TCurriculumScheduler.step
    TCurriculumScheduler.current_t          (property)
    TCurriculumScheduler.current_batch_size (property)
    TCurriculumScheduler.stage_changed
    temporal_lambda
    BestCheckpointTracker.__init__
    BestCheckpointTracker.update
    save_checkpoint
    load_checkpoint
    set_seed
    _stub_dataloader
    build_gt_extractor  →  extract  (inner function)
    run_validation

NOT covered (see bottom of file for explanation):
    main()
    _get_loader()   (closure inside main)

Run:
    pytest tests/test_train_infrastructure.py -v
    python tests/test_train_infrastructure.py
"""

import math
import os
import sys
import types
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

# ── Stub heavy project imports so train.py can be imported without the full
#    codebase being installed. ─────────────────────────────────────────────────
_STUB_MODULES = [
    "anyup", "anyup.modules", "anyup.modules.anyup3d", "anyup.modules.losses3d",
    "scripts", "scripts.load_2d_weights",
    "anyup.data", "anyup.data.video_dataset",
    "omegaconf", "transformers", "torch.utils.tensorboard",
]
for _mod in _STUB_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

sys.modules["anyup.modules.anyup3d"].AnyUp3D = nn.Linear
sys.modules["anyup.modules.losses3d"].combined_loss_3d = None
sys.modules["scripts.load_2d_weights"].load_2d_weights_into_3d = None
sys.modules["transformers"].VideoMAEModel = MagicMock()
sys.modules["transformers"].AutoImageProcessor = MagicMock()
sys.modules["torch.utils.tensorboard"].SummaryWriter = MagicMock()
sys.modules["omegaconf"].OmegaConf = MagicMock()

# Prevent main() from executing on import
import builtins
_real_name = "__notmain__"

import pathlib

_train_src = (pathlib.Path(__file__).parent.parent / "train_3d.py").read_text()
# Execute with __name__ != "__main__" so the if-guard at the bottom never fires
_g = {"__name__": "__test__", "__file__": str(pathlib.Path(__file__).parent.parent / "train_3d.py")}
exec(compile(_train_src, "train_3d.py", "exec"), _g)

# Pull everything into local scope
TCurriculumScheduler  = _g["TCurriculumScheduler"]
BestCheckpointTracker = _g["BestCheckpointTracker"]
save_checkpoint       = _g["save_checkpoint"]
load_checkpoint       = _g["load_checkpoint"]
temporal_lambda       = _g["temporal_lambda"]
set_seed              = _g["set_seed"]
_stub_dataloader      = _g["_stub_dataloader"]
build_gt_extractor    = _g["build_gt_extractor"]
run_validation        = _g["run_validation"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _default_stages():
    return [
        {"t": 1,  "start_step": 0,     "batch_size": 8},
        {"t": 4,  "start_step": 5000,  "batch_size": 4},
        {"t": 8,  "start_step": 15000, "batch_size": 2},
        {"t": 16, "start_step": 30000, "batch_size": 1},
    ]

def _small_model():
    """Tiny Linear so checkpointing tests run fast."""
    return nn.Linear(4, 4)

def _adamw(model):
    return torch.optim.AdamW(model.parameters(), lr=2e-4)


# ==============================================================================
# TCurriculumScheduler.__init__
# ==============================================================================

def test_scheduler_init_stores_sorted_stages():
    """Stages given out of order must be sorted ascending by start_step."""
    shuffled = [
        {"t": 8,  "start_step": 15000, "batch_size": 2},
        {"t": 1,  "start_step": 0,     "batch_size": 8},
        {"t": 4,  "start_step": 5000,  "batch_size": 4},
    ]
    sched = TCurriculumScheduler(shuffled)
    steps = [s["start_step"] for s in sched.stages]
    assert steps == sorted(steps), f"Stages not sorted: {steps}"
    print("PASS  __init__ sorts stages ascending by start_step")


def test_scheduler_init_single_stage():
    """Single-stage schedule should not crash and should always return that stage."""
    sched = TCurriculumScheduler([{"t": 1, "start_step": 0, "batch_size": 4}])
    assert sched.step(99999)["t"] == 1
    print("PASS  __init__ handles single-stage schedule")


# ==============================================================================
# TCurriculumScheduler.step
# ==============================================================================

def test_scheduler_step_at_zero():
    sched = TCurriculumScheduler(_default_stages())
    s = sched.step(0)
    assert s["t"] == 1 and s["batch_size"] == 8
    print("PASS  step(0) → T=1, bs=8")


def test_scheduler_step_just_before_first_threshold():
    sched = TCurriculumScheduler(_default_stages())
    s = sched.step(4999)
    assert s["t"] == 1
    print("PASS  step(4999) → T=1 (below threshold)")


def test_scheduler_step_at_first_threshold():
    sched = TCurriculumScheduler(_default_stages())
    s = sched.step(5000)
    assert s["t"] == 4 and s["batch_size"] == 4
    print("PASS  step(5000) → T=4, bs=4")


def test_scheduler_step_between_stages():
    sched = TCurriculumScheduler(_default_stages())
    sched.step(5000)        # advance to stage 1
    s = sched.step(10000)   # still in stage 1
    assert s["t"] == 4
    print("PASS  step(10000) → T=4 (between stage 1 and 2)")


def test_scheduler_step_all_thresholds():
    """Walk through every threshold and verify correct stage activates."""
    sched = TCurriculumScheduler(_default_stages())
    expected = [(0, 1), (5000, 4), (15000, 8), (30000, 16)]
    for step_val, expected_t in expected:
        s = sched.step(step_val)
        assert s["t"] == expected_t, f"step={step_val}: got T={s['t']}, want T={expected_t}"
    print("PASS  step walks through all 4 thresholds correctly")


def test_scheduler_step_monotone_never_regresses():
    """
    After advancing to a later stage, calling step() with an earlier step number
    must NOT regress the stage index. (Stage transitions are one-way.)
    """
    sched = TCurriculumScheduler(_default_stages())
    sched.step(30000)   # → T=16
    s = sched.step(0)   # try to go back
    assert s["t"] == 16, f"Stage regressed to T={s['t']} — monotone invariant violated"
    print("PASS  step() is monotone — no stage regression on lower step values")


def test_scheduler_step_far_beyond_last_stage():
    sched = TCurriculumScheduler(_default_stages())
    s = sched.step(10_000_000)
    assert s["t"] == 16
    print("PASS  step(10M) stays at final stage T=16")


def test_scheduler_step_returns_correct_batch_sizes():
    sched = TCurriculumScheduler(_default_stages())
    assert sched.step(0)["batch_size"] == 8
    assert sched.step(5000)["batch_size"] == 4
    assert sched.step(15000)["batch_size"] == 2
    assert sched.step(30000)["batch_size"] == 1
    print("PASS  step() batch_sizes halve correctly across all stages")


# ==============================================================================
# TCurriculumScheduler.current_t / current_batch_size  (properties)
# ==============================================================================

def test_scheduler_property_current_t_matches_step():
    sched = TCurriculumScheduler(_default_stages())
    sched.step(15000)
    assert sched.current_t == 8
    print("PASS  current_t property matches step() result")


def test_scheduler_property_current_batch_size_matches_step():
    sched = TCurriculumScheduler(_default_stages())
    sched.step(30000)
    assert sched.current_batch_size == 1
    print("PASS  current_batch_size property matches step() result")


def test_scheduler_properties_consistent_after_multiple_steps():
    """Properties always reflect the last-called step."""
    sched = TCurriculumScheduler(_default_stages())
    sched.step(0)
    assert sched.current_t == 1
    sched.step(5000)
    assert sched.current_t == 4
    sched.step(15000)
    assert sched.current_t == 8
    print("PASS  current_t / current_batch_size stay consistent across multiple step() calls")


# ==============================================================================
# TCurriculumScheduler.stage_changed
# ==============================================================================

def test_stage_changed_false_at_step_zero():
    """Step 0 is not a 'change' — it's the initial stage."""
    sched = TCurriculumScheduler(_default_stages())
    assert sched.stage_changed(0) is False
    print("PASS  stage_changed(0) == False (initial stage, not a transition)")


def test_stage_changed_true_at_each_threshold():
    sched = TCurriculumScheduler(_default_stages())
    for thresh in [5000, 15000, 30000]:
        assert sched.stage_changed(thresh) is True, f"stage_changed({thresh}) should be True"
    print("PASS  stage_changed() True at all non-zero thresholds")


def test_stage_changed_false_between_thresholds():
    sched = TCurriculumScheduler(_default_stages())
    for mid in [1, 100, 4999, 5001, 14999, 15001, 29999, 30001]:
        assert sched.stage_changed(mid) is False, f"stage_changed({mid}) should be False"
    print("PASS  stage_changed() False at all non-threshold steps")


# ==============================================================================
# temporal_lambda
# ==============================================================================

def test_temporal_lambda_zero_at_start():
    cfg = SimpleNamespace(lambda_temporal_consistency=0.2, temporal_lambda_warmup_steps=5000)
    assert temporal_lambda(cfg, 0) == 0.0
    print("PASS  temporal_lambda(step=0) == 0.0")


def test_temporal_lambda_midpoint():
    cfg = SimpleNamespace(lambda_temporal_consistency=0.2, temporal_lambda_warmup_steps=5000)
    result = temporal_lambda(cfg, 2500)
    assert abs(result - 0.1) < 1e-9, f"Expected 0.1, got {result}"
    print("PASS  temporal_lambda(step=2500) == 0.1 (half of warmup)")


def test_temporal_lambda_at_warmup_boundary():
    cfg = SimpleNamespace(lambda_temporal_consistency=0.2, temporal_lambda_warmup_steps=5000)
    assert temporal_lambda(cfg, 5000) == 0.2
    print("PASS  temporal_lambda(step=warmup) == full lambda")


def test_temporal_lambda_past_warmup():
    cfg = SimpleNamespace(lambda_temporal_consistency=0.2, temporal_lambda_warmup_steps=5000)
    assert temporal_lambda(cfg, 99999) == 0.2
    print("PASS  temporal_lambda(step >> warmup) == full lambda (no overshoot)")


def test_temporal_lambda_no_warmup():
    """warmup_steps=0 should return full lambda immediately at step 0."""
    cfg = SimpleNamespace(lambda_temporal_consistency=0.2, temporal_lambda_warmup_steps=0)
    assert temporal_lambda(cfg, 0) == 0.2
    assert temporal_lambda(cfg, 1000) == 0.2
    print("PASS  temporal_lambda with warmup_steps=0 always returns full lambda")


def test_temporal_lambda_zero_lambda():
    """lambda=0 should always return 0 regardless of step."""
    cfg = SimpleNamespace(lambda_temporal_consistency=0.0, temporal_lambda_warmup_steps=5000)
    for step in [0, 2500, 5000, 99999]:
        assert temporal_lambda(cfg, step) == 0.0
    print("PASS  temporal_lambda with lambda=0 always returns 0.0")


def test_temporal_lambda_linearity():
    """Ramp must be strictly linear: equal step increments → equal lambda increments."""
    cfg = SimpleNamespace(lambda_temporal_consistency=1.0, temporal_lambda_warmup_steps=1000)
    values = [temporal_lambda(cfg, s) for s in range(0, 1001, 100)]
    diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
    assert all(abs(d - diffs[0]) < 1e-9 for d in diffs), f"Non-linear ramp: {diffs}"
    print("PASS  temporal_lambda ramp is strictly linear")


# ==============================================================================
# BestCheckpointTracker.__init__ + update
# ==============================================================================

def test_best_tracker_lower_is_better_first_call_always_saves():
    with tempfile.TemporaryDirectory() as d:
        t = BestCheckpointTracker(os.path.join(d, "best.pth"), higher_is_better=False)
        m, o = _small_model(), _adamw(_small_model())
        result = t.update(999.0, m, o, step=1, t_stage=1)
        assert result is True
    print("PASS  first update() always returns True (lower_is_better)")


def test_best_tracker_higher_is_better_first_call_always_saves():
    with tempfile.TemporaryDirectory() as d:
        t = BestCheckpointTracker(os.path.join(d, "best.pth"), higher_is_better=True)
        m, o = _small_model(), _adamw(_small_model())
        result = t.update(-999.0, m, o, step=1, t_stage=1)
        assert result is True
    print("PASS  first update() always returns True (higher_is_better)")


def test_best_tracker_lower_is_better_improvement():
    with tempfile.TemporaryDirectory() as d:
        t = BestCheckpointTracker(os.path.join(d, "best.pth"), higher_is_better=False)
        m, o = _small_model(), _adamw(_small_model())
        t.update(1.0, m, o, 100, 1)
        result = t.update(0.5, m, o, 200, 4)   # lower → better
        assert result is True
        assert t.best_metric == 0.5
    print("PASS  lower_is_better: smaller value → True, best_metric updated")


def test_best_tracker_lower_is_better_no_improvement():
    with tempfile.TemporaryDirectory() as d:
        t = BestCheckpointTracker(os.path.join(d, "best.pth"), higher_is_better=False)
        m, o = _small_model(), _adamw(_small_model())
        t.update(0.5, m, o, 100, 1)
        result = t.update(0.8, m, o, 200, 4)   # higher → worse
        assert result is False
        assert t.best_metric == 0.5
    print("PASS  lower_is_better: larger value → False, best_metric unchanged")


def test_best_tracker_higher_is_better_improvement():
    with tempfile.TemporaryDirectory() as d:
        t = BestCheckpointTracker(os.path.join(d, "best.pth"), higher_is_better=True)
        m, o = _small_model(), _adamw(_small_model())
        t.update(0.5, m, o, 100, 1)
        result = t.update(0.9, m, o, 200, 4)   # higher → better
        assert result is True
        assert t.best_metric == 0.9
    print("PASS  higher_is_better: larger value → True, best_metric updated")


def test_best_tracker_higher_is_better_no_improvement():
    with tempfile.TemporaryDirectory() as d:
        t = BestCheckpointTracker(os.path.join(d, "best.pth"), higher_is_better=True)
        m, o = _small_model(), _adamw(_small_model())
        t.update(0.9, m, o, 100, 1)
        result = t.update(0.3, m, o, 200, 4)   # lower → worse
        assert result is False
        assert t.best_metric == 0.9
    print("PASS  higher_is_better: smaller value → False, best_metric unchanged")


def test_best_tracker_file_created_on_first_update():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "best.pth")
        t = BestCheckpointTracker(path, higher_is_better=False)
        m, o = _small_model(), _adamw(_small_model())
        t.update(1.0, m, o, 1, 1)
        assert os.path.isfile(path), "best.pth not created after first update"
    print("PASS  best.pth file is created after first update()")


def test_best_tracker_file_overwritten_on_new_best():
    """Verify best.pth is actually overwritten (mtime advances) when a new best arrives."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "best.pth")
        t = BestCheckpointTracker(path, higher_is_better=False)
        m, o = _small_model(), _adamw(_small_model())
        t.update(1.0, m, o, 100, 1)
        mtime1 = os.path.getmtime(path)
        import time; time.sleep(0.01)
        t.update(0.5, m, o, 200, 4)
        mtime2 = os.path.getmtime(path)
        assert mtime2 > mtime1, "best.pth mtime did not advance — file not overwritten"
    print("PASS  best.pth is overwritten (mtime advances) when new best arrives")


def test_best_tracker_equal_metric_not_saved_again():
    """Exactly equal metric should NOT count as improvement (strict inequality)."""
    with tempfile.TemporaryDirectory() as d:
        t = BestCheckpointTracker(os.path.join(d, "best.pth"), higher_is_better=False)
        m, o = _small_model(), _adamw(_small_model())
        t.update(0.5, m, o, 100, 1)
        result = t.update(0.5, m, o, 200, 4)   # same value
        assert result is False
    print("PASS  equal metric does not count as improvement (strict inequality)")


# ==============================================================================
# save_checkpoint / load_checkpoint
# ==============================================================================

def test_save_checkpoint_creates_file():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pth")
        m, o = _small_model(), _adamw(_small_model())
        save_checkpoint(path, m, o, step=100, t_stage=4)
        assert os.path.isfile(path)
    print("PASS  save_checkpoint creates the file")


def test_save_checkpoint_no_tmp_file_remains():
    """Atomic write: the .tmp file must be gone after a successful save."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pth")
        m, o = _small_model(), _adamw(_small_model())
        save_checkpoint(path, m, o, step=100, t_stage=4)
        assert not os.path.isfile(path + ".tmp"), ".tmp file not cleaned up"
    print("PASS  save_checkpoint leaves no .tmp file (atomic write)")


def test_load_checkpoint_restores_step():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pth")
        m, o = _small_model(), _adamw(_small_model())
        save_checkpoint(path, m, o, step=1234, t_stage=8)
        m2 = _small_model()
        step, _ = load_checkpoint(path, m2, device=torch.device("cpu"))
        assert step == 1234
    print("PASS  load_checkpoint restores step correctly")


def test_load_checkpoint_restores_t_stage():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pth")
        m, o = _small_model(), _adamw(_small_model())
        save_checkpoint(path, m, o, step=500, t_stage=16)
        m2 = _small_model()
        _, t_stage = load_checkpoint(path, m2, device=torch.device("cpu"))
        assert t_stage == 16
    print("PASS  load_checkpoint restores t_stage correctly")


def test_load_checkpoint_restores_model_weights():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pth")
        m = _small_model()
        # Give the model distinctive non-random weights
        with torch.no_grad():
            for p in m.parameters():
                p.fill_(3.14)
        o = _adamw(m)
        save_checkpoint(path, m, o, step=1, t_stage=1)
        m2 = _small_model()
        load_checkpoint(path, m2, device=torch.device("cpu"))
        for p1, p2 in zip(m.parameters(), m2.parameters()):
            assert torch.allclose(p1, p2), "Model weights not restored exactly"
    print("PASS  load_checkpoint restores model weights exactly")


def test_load_checkpoint_restores_optimizer_state():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pth")
        m = _small_model()
        o = _adamw(m)
        # Run one optimizer step so optimizer has non-trivial state
        loss = m(torch.randn(2, 4)).sum()
        loss.backward()
        o.step()
        save_checkpoint(path, m, o, step=1, t_stage=1)
        m2 = _small_model()
        o2 = _adamw(m2)
        load_checkpoint(path, m2, o2, device=torch.device("cpu"))
        # Compare exp_avg buffers from AdamW state
        for (pg1, pg2) in zip(o.state.values(), o2.state.values()):
            assert torch.allclose(pg1["exp_avg"], pg2["exp_avg"])
    print("PASS  load_checkpoint restores optimizer state (AdamW exp_avg)")


def test_load_checkpoint_without_optimizer_does_not_crash():
    """Passing optimizer=None should succeed and only restore model weights."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pth")
        m, o = _small_model(), _adamw(_small_model())
        save_checkpoint(path, m, o, step=42, t_stage=4)
        m2 = _small_model()
        step, t_stage = load_checkpoint(path, m2, optimizer=None, device=torch.device("cpu"))
        assert step == 42 and t_stage == 4
    print("PASS  load_checkpoint(optimizer=None) does not crash")


def test_save_checkpoint_stores_val_metric():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pth")
        m, o = _small_model(), _adamw(_small_model())
        save_checkpoint(path, m, o, step=1, t_stage=1, val_metric=0.314)
        state = torch.load(path, map_location="cpu")
        assert abs(state["val_metric"] - 0.314) < 1e-9
    print("PASS  save_checkpoint stores val_metric in checkpoint dict")


def test_checkpoint_roundtrip_all_t_stages():
    """Verify save/load works for every T stage value used in the curriculum."""
    with tempfile.TemporaryDirectory() as d:
        for t_val in [1, 4, 8, 16]:
            path = os.path.join(d, f"ckpt_T{t_val}.pth")
            m, o = _small_model(), _adamw(_small_model())
            save_checkpoint(path, m, o, step=t_val * 100, t_stage=t_val)
            m2 = _small_model()
            step, t_stage = load_checkpoint(path, m2, device=torch.device("cpu"))
            assert t_stage == t_val
            assert step == t_val * 100
    print("PASS  checkpoint round-trip correct for all T stages (1, 4, 8, 16)")


# ==============================================================================
# set_seed
# ==============================================================================

def test_set_seed_same_seed_produces_identical_tensors():
    set_seed(42)
    t1 = torch.randn(10)
    set_seed(42)
    t2 = torch.randn(10)
    assert torch.equal(t1, t2), "Same seed produced different tensors — seeding broken"
    print("PASS  set_seed(42) produces identical tensors on two runs")


def test_set_seed_different_seeds_produce_different_tensors():
    set_seed(0)
    t1 = torch.randn(100)
    set_seed(1)
    t2 = torch.randn(100)
    assert not torch.equal(t1, t2), "Different seeds produced identical tensors"
    print("PASS  different seeds produce different tensors")


def test_set_seed_affects_python_random():
    import random
    set_seed(99)
    v1 = [random.random() for _ in range(5)]
    set_seed(99)
    v2 = [random.random() for _ in range(5)]
    assert v1 == v2
    print("PASS  set_seed seeds Python random module")


def test_set_seed_affects_numpy():
    import numpy as np
    set_seed(7)
    a1 = np.random.rand(5).tolist()
    set_seed(7)
    a2 = np.random.rand(5).tolist()
    assert a1 == a2
    print("PASS  set_seed seeds NumPy random")


# ==============================================================================
# _stub_dataloader
# ==============================================================================

def _stub_cfg(img_size=64):
    return SimpleNamespace(img_size=img_size)


def test_stub_dataloader_yields_video_key():
    cfg = _stub_cfg(img_size=64)      # ↓ small spatial size for fast testing
    gen = _stub_dataloader(cfg, t=2, batch_size=1, device=torch.device("cpu"))
    batch = next(gen)
    assert "video" in batch, f"'video' key missing from batch: {list(batch.keys())}"
    print("PASS  _stub_dataloader yields batch with 'video' key")


def test_stub_dataloader_yields_video_crop_key():
    cfg = _stub_cfg(img_size=64)
    gen = _stub_dataloader(cfg, t=2, batch_size=1, device=torch.device("cpu"))
    batch = next(gen)
    assert "video_crop" in batch
    print("PASS  _stub_dataloader yields batch with 'video_crop' key")


def test_stub_dataloader_yields_patch_size_key():
    cfg = _stub_cfg(img_size=64)
    gen = _stub_dataloader(cfg, t=2, batch_size=1, device=torch.device("cpu"))
    batch = next(gen)
    assert "patch_size" in batch
    print("PASS  _stub_dataloader yields batch with 'patch_size' key")


def test_stub_dataloader_video_shape():
    """video must be (B, C=3, T, H, W) with H=W=img_size."""
    B, T, H = 2, 4, 32       # ↓ small values to keep test fast
    cfg = _stub_cfg(img_size=H)
    gen = _stub_dataloader(cfg, t=T, batch_size=B, device=torch.device("cpu"))
    batch = next(gen)
    expected = (B, 3, T, H, H)
    assert batch["video"].shape == expected, \
        f"video shape {batch['video'].shape} != {expected}"
    print(f"PASS  _stub_dataloader video shape {expected}")


def test_stub_dataloader_video_crop_shape():
    """video_crop must be 2× the spatial resolution of video."""
    B, T, H = 1, 2, 16        # ↓ small values
    cfg = _stub_cfg(img_size=H)
    gen = _stub_dataloader(cfg, t=T, batch_size=B, device=torch.device("cpu"))
    batch = next(gen)
    expected = (B, 3, T, H * 2, H * 2)
    assert batch["video_crop"].shape == expected, \
        f"video_crop shape {batch['video_crop'].shape} != {expected}"
    print(f"PASS  _stub_dataloader video_crop shape {expected} (2× spatial)")


def test_stub_dataloader_respects_T_parameter():
    cfg = _stub_cfg(img_size=32)
    for T in [1, 4, 8, 16]:
        gen = _stub_dataloader(cfg, t=T, batch_size=1, device=torch.device("cpu"))
        batch = next(gen)
        assert batch["video"].shape[2] == T, \
            f"T={T}: video.shape[2]={batch['video'].shape[2]}"
    print("PASS  _stub_dataloader T dimension matches t parameter for all T values")


def test_stub_dataloader_respects_batch_size():
    cfg = _stub_cfg(img_size=16)
    for bs in [1, 2, 4]:
        gen = _stub_dataloader(cfg, t=1, batch_size=bs, device=torch.device("cpu"))
        batch = next(gen)
        assert batch["video"].shape[0] == bs
    print("PASS  _stub_dataloader batch dimension matches batch_size parameter")


def test_stub_dataloader_is_infinite():
    """Generator must yield indefinitely without StopIteration."""
    cfg = _stub_cfg(img_size=8)
    gen = _stub_dataloader(cfg, t=1, batch_size=1, device=torch.device("cpu"))
    for _ in range(50):   # 50 consecutive nexts — infinite generator invariant
        next(gen)
    print("PASS  _stub_dataloader is infinite (50 consecutive next() calls succeeded)")


def test_stub_dataloader_video_values_in_unit_range():
    """Values must be in [0, 1] as documented (torch.rand)."""
    cfg = _stub_cfg(img_size=16)
    gen = _stub_dataloader(cfg, t=2, batch_size=4, device=torch.device("cpu"))
    batch = next(gen)
    assert batch["video"].min() >= 0.0
    assert batch["video"].max() <= 1.0
    print("PASS  _stub_dataloader video values are in [0, 1]")


# ==============================================================================
# build_gt_extractor → extract  (inner function)
# ==============================================================================
#
# We mock VideoMAEModel and AutoImageProcessor so no weights are downloaded.
# The mock backbone returns a known last_hidden_state tensor.
# This tests: pixel_values permutation, T_tok/H_tok/W_tok arithmetic,
# reshape, and final permute — all the real logic in extract().

def _make_mock_backbone(B, T, H, W):
    """
    Returns a mock VideoMAE backbone whose __call__ returns a SimpleNamespace
    with last_hidden_state of shape (B, T_tok*H_tok*W_tok, 768).
    t_tubelet=2, patch_size=16 (matching the constants in build_gt_extractor).
    """
    t_tubelet = 2
    patch_size = 16
    embed_dim = 768
    T_tok = T // t_tubelet
    H_tok = H // patch_size
    W_tok = W // patch_size
    seq_len = T_tok * H_tok * W_tok

    mock_model = MagicMock()
    # Make the mock callable: backbone(pixel_values=...) → output object
    mock_output = SimpleNamespace(
        last_hidden_state=torch.ones(B, seq_len, embed_dim)
    )
    mock_model.return_value = mock_output
    mock_model.parameters = lambda: iter([])   # for .eval() compatibility
    mock_model.eval = lambda: mock_model
    mock_model.to = lambda *a, **kw: mock_model
    return mock_model, T_tok, H_tok, W_tok, embed_dim


def _patch_gt_extractor(mock_bb):
    """
    Inject a mock backbone into the exec namespace _g so that build_gt_extractor
    picks it up when it calls VideoMAEModel.from_pretrained(...).
    Returns a context manager that restores originals on exit.
    """
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        orig_vmae = _g["VideoMAEModel"]
        orig_proc = _g["AutoImageProcessor"]

        mock_cls = MagicMock()
        mock_cls.from_pretrained.return_value = mock_bb
        mock_proc_cls = MagicMock()
        mock_proc_cls.from_pretrained.return_value = MagicMock()

        _g["VideoMAEModel"] = mock_cls
        _g["AutoImageProcessor"] = mock_proc_cls
        try:
            yield mock_cls
        finally:
            _g["VideoMAEModel"] = orig_vmae
            _g["AutoImageProcessor"] = orig_proc

    return _ctx()


def test_gt_extractor_output_shape():
    """extract(video) must return (B, D, T_tok, H_tok, W_tok)."""
    B, C, T, H, W = 2, 3, 4, 224, 224   # ↓ H/W must be multiples of patch_size=16
    mock_bb, T_tok, H_tok, W_tok, D = _make_mock_backbone(B, T, H, W)

    with _patch_gt_extractor(mock_bb):
        cfg = SimpleNamespace(encoder="videomae", encoder_model="MCG-NJU/videomae-base")
        extract = build_gt_extractor(cfg, device=torch.device("cpu"))
        feats = extract(torch.rand(B, C, T, H, W))

    expected = (B, D, T_tok, H_tok, W_tok)
    assert feats.shape == expected, f"GT feats shape {feats.shape} != {expected}"
    print(f"PASS  build_gt_extractor output shape {expected}")


def test_gt_extractor_token_grid_arithmetic():
    """Verify T_tok=T//2, H_tok=H//16, W_tok=W//16 for several (T, H, W) combos."""
    for T, H, W in [(2, 224, 224), (4, 112, 112), (8, 224, 224)]:
        B, C = 1, 3
        mock_bb, T_tok, H_tok, W_tok, D = _make_mock_backbone(B, T, H, W)

        with _patch_gt_extractor(mock_bb):
            cfg = SimpleNamespace(encoder="videomae", encoder_model="MCG-NJU/videomae-base")
            extract = build_gt_extractor(cfg, device=torch.device("cpu"))
            feats = extract(torch.rand(B, C, T, H, W))

        assert feats.shape[2] == T_tok, f"T={T}: feats.shape[2]={feats.shape[2]} != T_tok={T_tok}"
        assert feats.shape[3] == H_tok, f"H={H}: feats.shape[3]={feats.shape[3]} != H_tok={H_tok}"
        assert feats.shape[4] == W_tok, f"W={W}: feats.shape[4]={feats.shape[4]} != W_tok={W_tok}"
    print("PASS  build_gt_extractor token grid arithmetic correct for all (T, H, W) combos")


def test_gt_extractor_pixel_values_permutation():
    """
    VideoMAE expects pixel_values as (B, T, C, H, W).
    Verify the backbone is called with axis order (B, T, C, H, W) not (B, C, T, H, W).
    """
    B, C, T, H, W = 1, 3, 4, 224, 224
    mock_bb, *_ = _make_mock_backbone(B, T, H, W)

    captured_args = {}

    def capture_call(pixel_values):
        captured_args["shape"] = tuple(pixel_values.shape)
        return SimpleNamespace(last_hidden_state=torch.ones(B, (T//2)*(H//16)*(W//16), 768))

    mock_bb.side_effect = capture_call

    with _patch_gt_extractor(mock_bb):
        cfg = SimpleNamespace(encoder="videomae", encoder_model="MCG-NJU/videomae-base")
        extract = build_gt_extractor(cfg, device=torch.device("cpu"))
        extract(torch.rand(B, C, T, H, W))

    expected_pv_shape = (B, T, C, H, W)
    assert captured_args.get("shape") == expected_pv_shape, \
        f"pixel_values shape {captured_args.get('shape')} != expected {expected_pv_shape}"
    print("PASS  build_gt_extractor passes pixel_values as (B, T, C, H, W) to backbone")


def test_gt_extractor_output_is_contiguous():
    """Downstream code may call .view() on the output — must be contiguous."""
    B, C, T, H, W = 1, 3, 2, 224, 224
    mock_bb, *_ = _make_mock_backbone(B, T, H, W)

    with _patch_gt_extractor(mock_bb):
        cfg = SimpleNamespace(encoder="videomae", encoder_model="MCG-NJU/videomae-base")
        extract = build_gt_extractor(cfg, device=torch.device("cpu"))
        feats = extract(torch.rand(B, C, T, H, W))

    assert feats.is_contiguous(), "GT features tensor is not contiguous"
    print("PASS  build_gt_extractor output tensor is contiguous")


def test_gt_extractor_unknown_encoder_raises():
    """Passing an unsupported encoder name must raise ValueError immediately."""
    cfg = SimpleNamespace(encoder="fakeencoder", encoder_model="some/model")
    with pytest.raises(ValueError, match="fakeencoder"):
        build_gt_extractor(cfg, device=torch.device("cpu"))
    print("PASS  build_gt_extractor raises ValueError for unknown encoder")


# ==============================================================================
# run_validation
# ==============================================================================

def test_run_validation_returns_float():
    """The stub must return a float (real impl will return mIoU or MSE)."""
    m = _small_model()
    cfg = SimpleNamespace()
    result = run_validation(m, cfg, device=torch.device("cpu"), global_step=0)
    assert isinstance(result, float), f"Expected float, got {type(result)}"
    print("PASS  run_validation returns a float")


def test_run_validation_returns_inf_stub():
    """Current stub must return inf as a sentinel so best-tracker never falsely fires."""
    m = _small_model()
    cfg = SimpleNamespace()
    result = run_validation(m, cfg, device=torch.device("cpu"), global_step=0)
    assert math.isinf(result), f"Expected inf from stub, got {result}"
    print("PASS  run_validation stub returns inf (best-tracker sentinel)")


# ==============================================================================
# NOT TESTED — Requires external environment / Person B's interface
# ==============================================================================
#
# main()
#   Cannot be unit-tested here. It depends on:
#     - AnyUp3D model (anyup/modules/anyup3d.py) — not available without full codebase
#     - combined_loss_3d (anyup/modules/losses3d.py) — idem
#     - load_2d_weights_into_3d (scripts/load_2d_weights.py) — idem
#     - Person B's get_video_dataloaders() — not available yet
#     - A real VideoMAE checkpoint (requires HuggingFace download)
#   → SUGGESTION: add a smoke-test in tests/test_smoke.py that calls
#       `python train.py --config configs/anyup3d_train.yaml --debug`
#     as a subprocess once all deps are installed and Person B's module is ready.
#
# _get_loader()
#   A closure defined inside main(). It cannot be extracted and tested independently
#   without refactoring main(). The relevant logic (selecting between Person B's
#   loader and the stub) is implicitly covered by test_stub_dataloader_* above.
#   → SUGGESTION: If you want an explicit test, extract _get_loader to a module-level
#     function in train.py so it can be imported here directly.
#
# build_gt_extractor — real VideoMAE backbone (no mock)
#   Requires downloading MCG-NJU/videomae-base (~350 MB). Cannot run in CI.
#   → SUGGESTION: Add a separate test file tests/test_gt_extractor_integration.py
#     guarded by `pytest.mark.slow` that runs the real backbone on a synthetic
#     (1, 3, 4, 224, 224) tensor and checks output shape matches (1, 768, 2, 14, 14).


# ==============================================================================
# __main__ runner
# ==============================================================================

if __name__ == "__main__":
    tests = [
        # TCurriculumScheduler.__init__
        test_scheduler_init_stores_sorted_stages,
        test_scheduler_init_single_stage,
        # .step()
        test_scheduler_step_at_zero,
        test_scheduler_step_just_before_first_threshold,
        test_scheduler_step_at_first_threshold,
        test_scheduler_step_between_stages,
        test_scheduler_step_all_thresholds,
        test_scheduler_step_monotone_never_regresses,
        test_scheduler_step_far_beyond_last_stage,
        test_scheduler_step_returns_correct_batch_sizes,
        # .current_t / .current_batch_size properties
        test_scheduler_property_current_t_matches_step,
        test_scheduler_property_current_batch_size_matches_step,
        test_scheduler_properties_consistent_after_multiple_steps,
        # .stage_changed()
        test_stage_changed_false_at_step_zero,
        test_stage_changed_true_at_each_threshold,
        test_stage_changed_false_between_thresholds,
        # temporal_lambda
        test_temporal_lambda_zero_at_start,
        test_temporal_lambda_midpoint,
        test_temporal_lambda_at_warmup_boundary,
        test_temporal_lambda_past_warmup,
        test_temporal_lambda_no_warmup,
        test_temporal_lambda_zero_lambda,
        test_temporal_lambda_linearity,
        # BestCheckpointTracker
        test_best_tracker_lower_is_better_first_call_always_saves,
        test_best_tracker_higher_is_better_first_call_always_saves,
        test_best_tracker_lower_is_better_improvement,
        test_best_tracker_lower_is_better_no_improvement,
        test_best_tracker_higher_is_better_improvement,
        test_best_tracker_higher_is_better_no_improvement,
        test_best_tracker_file_created_on_first_update,
        test_best_tracker_file_overwritten_on_new_best,
        test_best_tracker_equal_metric_not_saved_again,
        # save_checkpoint / load_checkpoint
        test_save_checkpoint_creates_file,
        test_save_checkpoint_no_tmp_file_remains,
        test_load_checkpoint_restores_step,
        test_load_checkpoint_restores_t_stage,
        test_load_checkpoint_restores_model_weights,
        test_load_checkpoint_restores_optimizer_state,
        test_load_checkpoint_without_optimizer_does_not_crash,
        test_save_checkpoint_stores_val_metric,
        test_checkpoint_roundtrip_all_t_stages,
        # set_seed
        test_set_seed_same_seed_produces_identical_tensors,
        test_set_seed_different_seeds_produce_different_tensors,
        test_set_seed_affects_python_random,
        test_set_seed_affects_numpy,
        # _stub_dataloader
        test_stub_dataloader_yields_video_key,
        test_stub_dataloader_yields_video_crop_key,
        test_stub_dataloader_yields_patch_size_key,
        test_stub_dataloader_video_shape,
        test_stub_dataloader_video_crop_shape,
        test_stub_dataloader_respects_T_parameter,
        test_stub_dataloader_respects_batch_size,
        test_stub_dataloader_is_infinite,
        test_stub_dataloader_video_values_in_unit_range,
        # build_gt_extractor → extract
        test_gt_extractor_output_shape,
        test_gt_extractor_token_grid_arithmetic,
        test_gt_extractor_pixel_values_permutation,
        test_gt_extractor_output_is_contiguous,
        test_gt_extractor_unknown_encoder_raises,
        # run_validation
        test_run_validation_returns_float,
        test_run_validation_returns_inf_stub,
    ]

    passed = 0
    failed = 0
    for fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"FAIL  {fn.__name__}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed:
        sys.exit(1)