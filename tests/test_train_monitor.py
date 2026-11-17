"""T05_training: TrainMonitor on CPU with a fake clock, fake CUDA counters and a fake writer."""
import csv
import math
from types import SimpleNamespace

import pytest
import torch

from anyup.data.training.monitor import GIB, NonFiniteLossError, TrainMonitor

LOSSES = {"total": 1.0, "rec": 0.5, "inp": 0.2, "aug": 0.2, "temporal": 0.1}


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class FakeCuda:
    """Peak counters that the test sets; reset_peak_memory_stats() drops the peak to the current value."""
    def __init__(self, total_gib=15.0):
        self.total = int(total_gib * GIB)
        self.cur_a = self.cur_r = 0
        self.peak_a = self.peak_r = 0
        self.resets = 0

    def is_available(self):
        return True

    def get_device_properties(self, i):
        return SimpleNamespace(total_memory=self.total)

    def use(self, alloc_gib, reserved_gib):
        self.cur_a, self.cur_r = int(alloc_gib * GIB), int(reserved_gib * GIB)
        self.peak_a, self.peak_r = max(self.peak_a, self.cur_a), max(self.peak_r, self.cur_r)

    def max_memory_allocated(self):
        return self.peak_a

    def max_memory_reserved(self):
        return self.peak_r

    def memory_allocated(self):
        return self.cur_a

    def memory_reserved(self):
        return self.cur_r

    def reset_peak_memory_stats(self):
        self.peak_a, self.peak_r = self.cur_a, self.cur_r
        self.resets += 1


class NoCuda:
    def is_available(self):
        return False


class Writer:
    def __init__(self):
        self.scalars = {}

    def add_scalar(self, tag, v, step):
        self.scalars.setdefault(tag, []).append((step, v))


Stage = lambda t, s, b: SimpleNamespace(t=t, start_step=s, batch_size=b)   # noqa: E731
SCHEDULE = [Stage(2, 0, 2), Stage(4, 4, 1), Stage(8, 100, 1)]


def make(tmp_path=None, **kw):
    clock, cuda, writer = Clock(), FakeCuda(), Writer()
    args = dict(writer=writer, out_dir=tmp_path, log_every=2, total_steps=200, schedule=SCHEDULE,
                val_every=4, grad_clip=1.0, clock=clock, cuda=cuda)
    args.update(kw)
    return TrainMonitor(**args), clock, cuda, writer


def run_steps(mon, clock, steps, T, bs, dt, losses=LOSSES, grad_norm=0.5, cuda=None, mem=None):
    for step in steps:
        mon.step_start(step, T, bs)
        if cuda is not None and mem is not None:
            cuda.use(*mem)
        clock.t += dt
        mon.step_end(step, losses, grad_norm=grad_norm, grad_scale=65536.0, lr=1e-4)


def test_stage_timing_and_first_step_apart():
    mon, clock, cuda, _ = make()
    mon.step_start(0, 2, 2); clock.t += 9.0; mon.step_end(0, LOSSES)     # slow first step (warm-up)
    run_steps(mon, clock, range(1, 4), 2, 2, dt=2.0)
    mon.step_start(4, 4, 1); clock.t += 7.0; mon.step_end(4, LOSSES)
    run_steps(mon, clock, range(5, 8), 4, 1, dt=3.0)
    s = mon.summary()
    assert [(x["T"], x["batch_size"], x["first_step"], x["steps"]) for x in s] == [(2, 2, 0, 4), (4, 1, 4, 4)]
    assert s[0]["first_step_s"] == 9.0 and s[0]["median_s_per_step"] == 2.0 and s[0]["it_per_s"] == 0.5
    assert s[1]["first_step_s"] == 7.0 and s[1]["median_s_per_step"] == 3.0


def test_peak_reset_at_stage_switch_and_val_separate():
    mon, clock, cuda, writer = make()
    cuda.use(5, 6); cuda.use(0, 0)                     # transient setup peak (model load) must not count for stage 1
    run_steps(mon, clock, range(0, 4), 2, 2, dt=1.0, cuda=cuda, mem=(3, 4))
    mon.val_start(); cuda.use(1, 9); clock.t += 60; cuda.use(1, 1); mon.val_end(4, 10, 0.3)
    run_steps(mon, clock, range(4, 8), 4, 1, dt=1.0, cuda=cuda, mem=(7, 8))
    s = mon.summary()
    assert s[0]["peak_alloc_gib"] == pytest.approx(3) and s[0]["peak_reserved_gib"] == pytest.approx(4)
    assert s[1]["peak_alloc_gib"] == pytest.approx(7) and s[1]["peak_reserved_gib"] == pytest.approx(8)
    val = [r for r in mon.rows if r["kind"] == "val"][0]
    assert val["val_peak_reserved_gib"] == pytest.approx(9) and val["val_time_min"] == pytest.approx(1.0)
    assert val["val_clips"] == 10 and val["val_loss"] == pytest.approx(0.3)
    assert writer.scalars["val/time_min"] == [(4, 1.0)]


def test_nan_loss_raises_with_context():
    mon, clock, *_ = make()
    run_steps(mon, clock, [0], 2, 2, dt=1.0)
    mon.step_start(1, 2, 2); clock.t += 1
    with pytest.raises(NonFiniteLossError) as e:
        mon.step_end(1, {**LOSSES, "temporal": torch.tensor(float("nan"))})
    msg = str(e.value)
    assert "step 1" in msg and "temporal=nan" in msg and "T=2" in msg and "batch_size=2" in msg


def test_inf_loss_raises_and_check_before_save():
    mon, clock, *_ = make()
    run_steps(mon, clock, [0], 2, 2, dt=1.0)
    mon.step_start(1, 2, 2); clock.t += 1
    with pytest.raises(NonFiniteLossError):
        mon.step_end(1, {**LOSSES, "rec": float("inf")})
    with pytest.raises(NonFiniteLossError, match="before checkpoint save"):
        mon.check_before_save(1, {**LOSSES, "total": float("-inf")})
    mon.check_before_save(1, LOSSES)                   # finite → no error


def test_nan_check_only_at_logging_points_when_disabled():
    mon, clock, *_ = make(nan_every_step=False)
    run_steps(mon, clock, [0], 2, 2, dt=1.0)
    mon.step_start(1, 2, 2); clock.t += 1
    mon.step_end(1, {**LOSSES, "total": float("nan")})          # step 1 is not a logging point (log_every=2)
    mon.step_start(2, 2, 2); clock.t += 1
    with pytest.raises(NonFiniteLossError):
        mon.step_end(2, {**LOSSES, "total": float("nan")})


def test_nonfinite_grad_norm_counts_not_raises(tmp_path, capsys):
    mon, clock, *_ = make(tmp_path)
    run_steps(mon, clock, range(0, 3), 2, 2, dt=1.0, grad_norm=torch.tensor(float("inf")))
    run_steps(mon, clock, [3, 4], 2, 2, dt=1.0)
    assert mon.skipped_total == 3 and mon.summary()[0]["skipped_steps"] == 3
    assert "GradScaler skips" in capsys.readouterr().out
    rows = [r for r in mon.rows if r["kind"] == "train"]
    assert [r["skipped_steps"] for r in rows] == [1, 3, 3]
    assert math.isinf(rows[0]["grad_norm"])


def test_oom_report_text():
    mon, clock, cuda, _ = make()
    run_steps(mon, clock, [0, 1], 4, 1, dt=1.0, cuda=cuda, mem=(10, 12))
    mon.step_start(2, 4, 1)
    msg = mon.oom_report(torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    for s in ["step 2", "T=4", "batch_size=1", "allocated 10.00 GiB", "reserved 12.00 GiB", "15.00 GiB total",
              "Tried to allocate 2.00 GiB"]:
        assert s in msg, s


def test_disabled_does_nothing(tmp_path):
    mon, clock, cuda, writer = make(tmp_path / "out", enabled=False)
    run_steps(mon, clock, range(0, 4), 2, 2, dt=1.0, losses={**LOSSES, "total": float("nan")})
    mon.check_before_save(3, {"total": float("nan")})
    mon.val_start(); mon.val_end(4, 1, 0.1); mon.finish(4)
    assert mon.oom_report() == "" and mon.summary() == [] and writer.scalars == {}
    assert not (tmp_path / "out").exists() and cuda.resets == 0


def test_no_cuda_works(tmp_path):
    mon, clock, *_ = make(tmp_path, cuda=NoCuda())
    run_steps(mon, clock, range(0, 4), 2, 2, dt=1.0)
    mon.val_start(); clock.t += 6; mon.val_end(4, 2, 0.1)
    mon.finish(4)
    s = mon.summary()[0]
    assert s["peak_alloc_gib"] is None and s["peak_reserved_gib"] is None
    assert "allocated" not in mon.oom_report()
    assert (tmp_path / "dashboard_final.png").exists()


def test_scalar_tags():
    mon, clock, cuda, writer = make()
    run_steps(mon, clock, range(0, 4), 2, 2, dt=2.0, cuda=cuda, mem=(3, 6))
    assert set(writer.scalars) == {"perf/s_per_step", "perf/it_per_s", "memory/peak_alloc_gib",
                                   "memory/peak_reserved_gib", "memory/peak_reserved_pct",
                                   "train/grad_norm", "train/grad_scale", "train/skipped_steps"}
    assert [s for s, _ in writer.scalars["perf/s_per_step"]] == [0, 2]
    assert writer.scalars["perf/s_per_step"][1][1] == 2.0          # mean of steps 1, 2
    assert writer.scalars["memory/peak_reserved_pct"][0][1] == pytest.approx(40.0)


def test_history_csv_and_resume(tmp_path):
    mon, clock, *_ = make(tmp_path)
    run_steps(mon, clock, range(0, 4), 2, 2, dt=1.0)                # train rows at 0, 2
    mon.val_start(); mon.val_end(4, 3, 0.2)                          # val row at 4
    run_steps(mon, clock, range(4, 8), 4, 1, dt=1.0)                # train rows at 4, 6
    with open(tmp_path / "monitor_history.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(r["kind"], int(float(r["step"]))) for r in rows] == [("train", 0), ("train", 2), ("val", 4),
                                                                  ("train", 4), ("train", 6)]
    mon2, *_ = make(tmp_path)
    mon2.resume(4)                    # checkpoint at step 4: keep val@4, drop train@4 and @6
    assert [(r["kind"], r["step"]) for r in mon2.rows] == [("train", 0), ("train", 2), ("val", 4)]
    with open(tmp_path / "monitor_history.csv", newline="") as f:
        assert len(list(csv.DictReader(f))) == 3
    mon3, *_ = make(tmp_path)
    mon3.resume(0)                    # fresh run → empty history
    assert mon3.rows == []


def test_dashboard_files(tmp_path):
    mon, clock, cuda, _ = make(tmp_path)
    run_steps(mon, clock, range(0, 4), 2, 2, dt=1.0, cuda=cuda, mem=(3, 4))
    mon.val_start(); mon.val_end(4, 3, 0.2)
    assert (tmp_path / "dashboard.png").exists()
    run_steps(mon, clock, range(4, 6), 4, 1, dt=1.0, cuda=cuda, mem=(5, 6),
              grad_norm=float("nan"))                                # non-finite values in the history
    assert (tmp_path / "dashboard_T2_end.png").exists()
    mon.finish(6)
    assert (tmp_path / "dashboard_final.png").exists()
    assert (tmp_path / "dashboard.png").read_bytes() == (tmp_path / "dashboard_final.png").read_bytes()


def test_time_left_lower_bound():
    mon, clock, *_ = make()
    run_steps(mon, clock, range(0, 4), 2, 2, dt=2.0)
    mon.val_start(); clock.t += 30; mon.val_end(4, 3)
    train_s, val_s, lower = mon.time_left(4)
    assert train_s == (200 - 4) * 2.0 and lower is True             # T=4 and T=8 stages still ahead
    assert val_s == (200 // 4 - 4 // 4) * 30
    assert "lower bound" in mon.time_left_text(4)


def test_time_left_before_first_validation():
    """Run 1 bug: before any validation was timed, the estimate printed 'validation 0 h 00 min'."""
    mon, clock, *_ = make(schedule=[Stage(2, 0, 2)])                   # one stage → no stage lower bound
    run_steps(mon, clock, range(0, 4), 2, 2, dt=2.0)
    train_s, val_s, lower = mon.time_left(3)
    assert train_s == (200 - 3) * 2.0 and val_s is None and lower is False
    text = mon.time_left_text(3)
    assert "validation not timed yet" in text and text.startswith("time left ≥ ")
    mon.val_start(); clock.t += 30; mon.val_end(4, 3)
    train_s, val_s, lower = mon.time_left(4)
    assert val_s == (200 // 4 - 4 // 4) * 30 and "time left ≈ " in mon.time_left_text(4)
    mon2, clock2, *_ = make(schedule=[Stage(2, 0, 2)], val_every=None)  # no validation in the run
    run_steps(mon2, clock2, range(0, 4), 2, 2, dt=2.0)
    assert mon2.time_left(3)[1] == 0.0 and "time left ≈ " in mon2.time_left_text(3)


class FakeShell:
    def __init__(self):
        self.exc, self.handler, self.shown = None, None, []

    def set_custom_exc(self, exc, handler):
        self.exc, self.handler = exc, handler

    def showtraceback(self, exc_tuple, tb_offset=None):
        self.shown.append(exc_tuple)


def test_install_oom_handler(capsys):
    mon, clock, cuda, _ = make()
    run_steps(mon, clock, [0], 8, 1, dt=1.0, cuda=cuda, mem=(13, 14))
    mon.step_start(1, 8, 1)
    sh = FakeShell()
    assert mon.install_oom_handler(shell=sh) is True
    assert sh.exc == (torch.cuda.OutOfMemoryError,)
    err = torch.cuda.OutOfMemoryError("CUDA out of memory.")
    sh.handler(sh, type(err), err, None)
    out = capsys.readouterr().out
    assert "CUDA OUT OF MEMORY at step 1" in out and "T=8" in out
    assert sh.shown == [(type(err), err, None)]                     # normal traceback still shown


def test_install_oom_handler_outside_ipython_or_disabled():
    mon, *_ = make()
    assert mon.install_oom_handler() is False                       # pytest is not an IPython shell
    mon_off, *_ = make(enabled=False)
    assert mon_off.install_oom_handler(shell=FakeShell()) is False


class Timer:
    """perf_counter stand-in: every call advances 1 ms."""
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 0.001
        return self.t


def test_self_timed_overhead_excludes_draws(tmp_path):
    timer = Timer()
    mon, clock, *_ = make(tmp_path, timer=timer)
    run_steps(mon, clock, range(0, 4), 2, 2, dt=2.0)
    run_steps(mon, clock, range(4, 6), 4, 1, dt=2.0)       # stage switch → one draw inside step_start
    o = mon.overhead()
    assert o["step_s"] == pytest.approx(12.0)              # 6 steps × 2 s (fake clock)
    assert o["n_draws"] == 1 and o["draw_s"] == pytest.approx(0.001)
    # 12 timed calls of ≈ 1 ms each (fake timer); the draw's time is subtracted
    assert 0 < o["monitor_s"] < 0.05 and o["pct"] == pytest.approx(100 * o["monitor_s"] / 12.0)
    assert "own overhead" in mon.overhead_text() and "target < 1 %" in mon.overhead_text()


def test_overhead_disabled_is_zero():
    mon, *_ = make(enabled=False, timer=Timer())
    mon.step_start(0, 2, 2); mon.step_end(0, LOSSES)
    assert mon.overhead_s == 0.0
