"""Training monitor (T05_training): speed, memory, validation time, NaN / OOM checks,
training-health scalars and a PNG dashboard, built into every real training run.

CPU-side only: it reads the wall clock and the CUDA memory counters, and turns losses that
already exist into Python numbers. It never allocates GPU memory.

Files written to `out_dir` (cfg.log_dir on Drive):
    monitor_history.csv      one "train" row per logging point, one "val" row per validation (Q9)
    dashboard.png            latest dashboard, redrawn at every validation (Q3b, Q10)
    dashboard_T{t}_end.png   frozen at the end of each curriculum stage
    dashboard_final.png      frozen at the end of the run
"""
import csv
import functools
import math
import shutil
import statistics
import time
from pathlib import Path

import torch

GIB = 1024 ** 3
LOSS_KEYS = ("total", "rec", "inp", "aug", "temporal")
HISTORY_COLS = (["kind", "step", "T", "batch_size", "s_per_step", "it_per_s"]
                + [f"loss_{k}" for k in LOSS_KEYS]
                + ["lr", "grad_norm", "grad_scale", "skipped_steps",
                   "peak_alloc_gib", "peak_reserved_gib", "peak_reserved_pct",
                   "val_loss", "val_time_min", "val_clips", "val_peak_reserved_gib"])
MAX_SKIP_WARNINGS = 10   # per stage; further skipped steps are only counted (see the stage summary)


class NonFiniteLossError(RuntimeError):
    """A training loss became NaN or infinite (Q4)."""


def _num(v):
    """Tensor / number / None -> float or None. A 0-dim CUDA tensor costs one GPU wait."""
    if v is None:
        return None
    return float(v.item() if torch.is_tensor(v) else v)


def _finite(x):
    return x is not None and math.isfinite(x)


def _fmt_time(seconds):
    h, rem = divmod(int(round(seconds)), 3600)
    return f"{h} h {rem // 60:02d} min"


def _self_timed(fn):
    """Q7: add the time spent inside this monitor call (minus dashboard draws) to self.overhead_s."""
    @functools.wraps(fn)
    def wrap(self, *a, **k):
        if not self.enabled:
            return fn(self, *a, **k)
        t0, d0 = self.timer(), self.draw_s
        try:
            return fn(self, *a, **k)
        finally:
            self.overhead_s += self.timer() - t0 - (self.draw_s - d0)
    return wrap


class _Stage:
    def __init__(self, T, batch_size, first_step):
        self.T, self.batch_size, self.first_step = T, batch_size, first_step
        self.first_step_s = None          # first step of a stage: loader rebuild + warm-up, reported apart
        self.step_s = []                  # wall time of the other steps
        self.peak_alloc = 0
        self.peak_reserved = 0
        self.skipped = 0

    def median_s(self):
        return statistics.median(self.step_s) if self.step_s else None


class TrainMonitor:
    def __init__(self, writer=None, out_dir=None, log_every=50, total_steps=None, schedule=None,
                 val_every=None, grad_clip=None, nan_every_step=True, enabled=True,
                 clock=time.time, cuda=None, timer=time.perf_counter):
        """
        writer:         TensorBoard SummaryWriter (or None)
        out_dir:        folder for monitor_history.csv and the dashboards (None = no files)
        log_every:      cfg.log_every_n_steps (50): logging points for scalars + history rows
        total_steps:    last step of the run (for the time-left estimate)
        schedule:       cfg.t_schedule (objects with .t, .start_step, .batch_size)
        val_every:      cfg.val_every_n_steps (for the time-left estimate)
        grad_clip:      cfg.grad_clip_max_norm (drawn as a line on the dashboard)
        nan_every_step: Q6 — check losses / grad norm every step; False = at logging points only
        clock, cuda:    injectable for tests (default time.time, torch.cuda)
        timer:          self-timing clock for the overhead measurement (Q7; default time.perf_counter)
        """
        self.enabled = enabled
        self.writer = writer
        self.out_dir = Path(out_dir) if out_dir else None
        self.log_every = log_every
        self.total_steps = total_steps
        self.schedule = sorted(schedule or [], key=lambda s: s.start_step)
        self.val_every = val_every
        self.grad_clip = grad_clip
        self.nan_every_step = nan_every_step
        self.clock = clock
        self.timer = timer
        self.cuda = torch.cuda if cuda is None else cuda
        self.has_cuda = bool(self.cuda.is_available())
        self.gpu_total = self.cuda.get_device_properties(0).total_memory if self.has_cuda else None

        self.stage = None                 # current _Stage
        self.stages = []                  # finished + current
        self.rows = []                    # history rows (dicts), mirrored in monitor_history.csv
        self.skipped_total = 0
        self._t0 = None                   # wall time at step_start
        self._step = None
        self._window_s = []               # step times since the last logging point
        self._val_t0 = None
        self._last_val_s = None
        self.overhead_s = 0.0             # Q7: time inside monitor calls, dashboard draws excluded
        self.draw_s = 0.0                 # time spent drawing dashboards
        self.n_draws = 0
        if self.enabled and self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)

    # ── history file (Q9) ────────────────────────────────────────────────────
    @property
    def history_path(self):
        return self.out_dir / "monitor_history.csv" if self.out_dir else None

    def resume(self, start_step):
        """Reload the history of an earlier session and drop what the resumed run will redo.
        Train rows with step >= start_step go; val rows with step <= start_step stay (that validation
        ran on the checkpoint being resumed). start_step = 0 starts a fresh history."""
        if not (self.enabled and self.history_path and self.history_path.exists()):
            return
        with open(self.history_path, newline="") as f:
            rows = list(csv.DictReader(f))
        keep = []
        for r in rows:
            step = int(float(r["step"]))
            if (r["kind"] == "train" and step < start_step) or (r["kind"] == "val" and step <= start_step):
                keep.append({k: (r[k] if k == "kind" else (float(r[k]) if r.get(k, "") != "" else None))
                             for k in HISTORY_COLS})
        self.rows = keep
        last_skip = [r["skipped_steps"] for r in keep if r["kind"] == "train" and r["skipped_steps"] is not None]
        self.skipped_total = int(last_skip[-1]) if last_skip else 0
        self._rewrite_history()
        print(f"[monitor] resumed history at step {start_step}: kept {len(keep)} of {len(rows)} rows")

    def _rewrite_history(self):
        with open(self.history_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=HISTORY_COLS)
            w.writeheader()
            for r in self.rows:
                w.writerow({k: ("" if r.get(k) is None else r[k]) for k in HISTORY_COLS})

    def _append_row(self, row):
        row = {k: row.get(k) for k in HISTORY_COLS}
        self.rows.append(row)
        if not self.history_path:
            return
        new = not self.history_path.exists()
        with open(self.history_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=HISTORY_COLS)
            if new:
                w.writeheader()
            w.writerow({k: ("" if row[k] is None else row[k]) for k in HISTORY_COLS})

    # ── memory ───────────────────────────────────────────────────────────────
    def _read_peak_and_reset(self):
        """(peak allocated, peak reserved) in bytes since the last reset, then reset. (None, None) without CUDA."""
        if not self.has_cuda:
            return None, None
        a, r = self.cuda.max_memory_allocated(), self.cuda.max_memory_reserved()
        self.cuda.reset_peak_memory_stats()
        return a, r

    def _fold_peak(self):
        a, r = self._read_peak_and_reset()
        if a is not None and self.stage is not None:
            self.stage.peak_alloc = max(self.stage.peak_alloc, a)
            self.stage.peak_reserved = max(self.stage.peak_reserved, r)
        return a, r

    # ── steps ────────────────────────────────────────────────────────────────
    @_self_timed
    def step_start(self, step, T, batch_size):
        if not self.enabled:
            return
        if self.stage is None or (T, batch_size) != (self.stage.T, self.stage.batch_size):
            if self.stage is not None:
                self._fold_peak()
                self._print_stage_summary(self.stage, step)
                self.draw(f"dashboard_T{self.stage.T}_end.png", also_latest=True)
            else:
                self._read_peak_and_reset()      # drop setup peaks (model load etc.) from stage 1
            self.stage = _Stage(T, batch_size, step)
            self.stages.append(self.stage)
            self._window_s = []
        self._step = step
        self._t0 = self.clock()

    @_self_timed
    def step_end(self, step, losses, grad_norm=None, grad_scale=None, lr=None):
        """losses: {name: 0-dim tensor or float}. Raises NonFiniteLossError on a NaN/inf loss (Q4)."""
        if not self.enabled:
            return
        dt = self.clock() - self._t0
        st = self.stage
        if st.first_step_s is None:
            st.first_step_s = dt
        else:
            st.step_s.append(dt)
        self._window_s.append(dt)

        log_point = step % self.log_every == 0
        if not (self.nan_every_step or log_point):
            return
        vals = {k: _num(v) for k, v in losses.items()}
        self._raise_if_nonfinite(step, vals, "")
        gn = _num(grad_norm)
        if grad_norm is not None and not _finite(gn):
            self._skipped_step(step, gn)
        if log_point:
            self._log_point(step, vals, gn, _num(grad_scale), lr)

    @_self_timed
    def check_before_save(self, step, losses):
        """Q6 addition: never write a checkpoint after a NaN/inf loss."""
        if not self.enabled:
            return
        self._raise_if_nonfinite(step, {k: _num(v) for k, v in losses.items()}, " (before checkpoint save)")

    def _raise_if_nonfinite(self, step, vals, where):
        bad = {k: v for k, v in vals.items() if not _finite(v)}
        if bad:
            st = self.stage
            raise NonFiniteLossError(
                f"[monitor] non-finite loss at step {step}{where}: "
                + ", ".join(f"{k}={v}" for k, v in bad.items())
                + f" | T={st.T}, batch_size={st.batch_size}, stage started at step {st.first_step}. "
                + "Stopping: every step after this would train on a NaN/inf loss.")

    def _skipped_step(self, step, gn):
        """Q11: non-finite grad norm → the GradScaler skips this optimizer step (S15). Count, don't stop."""
        self.stage.skipped += 1
        self.skipped_total += 1
        if self.stage.skipped <= MAX_SKIP_WARNINGS:
            more = " (further skips this stage are only counted)" if self.stage.skipped == MAX_SKIP_WARNINGS else ""
            print(f"[monitor] WARNING step {step}: grad norm = {gn} → GradScaler skips this optimizer step "
                  f"(T={self.stage.T}; skipped so far: {self.skipped_total}){more}")

    def _log_point(self, step, vals, gn, scale, lr):
        st = self.stage
        s_per_step = statistics.mean(self._window_s) if self._window_s else None
        self._window_s = []
        a, r = self._fold_peak()
        row = {"kind": "train", "step": step, "T": st.T, "batch_size": st.batch_size,
               "s_per_step": s_per_step, "it_per_s": (1.0 / s_per_step) if s_per_step else None,
               "lr": lr, "grad_norm": gn, "grad_scale": scale, "skipped_steps": self.skipped_total,
               "peak_alloc_gib": a / GIB if a is not None else None,
               "peak_reserved_gib": r / GIB if r is not None else None,
               "peak_reserved_pct": 100.0 * r / self.gpu_total if r is not None else None}
        row.update({f"loss_{k}": vals.get(k) for k in LOSS_KEYS})
        self._append_row(row)
        self._scalars(step, {
            "perf/s_per_step": row["s_per_step"], "perf/it_per_s": row["it_per_s"],
            "memory/peak_alloc_gib": row["peak_alloc_gib"], "memory/peak_reserved_gib": row["peak_reserved_gib"],
            "memory/peak_reserved_pct": row["peak_reserved_pct"],
            "train/grad_norm": gn, "train/grad_scale": scale, "train/skipped_steps": self.skipped_total})

    def _scalars(self, step, d):
        if self.writer is None:
            return
        for tag, v in d.items():
            if v is not None:
                self.writer.add_scalar(tag, v, step)

    # ── validation ───────────────────────────────────────────────────────────
    @_self_timed
    def val_start(self):
        if not self.enabled:
            return
        self._fold_peak()                         # training peak so far → stage; then measure val alone
        self._val_t0 = self.clock()

    @_self_timed
    def val_end(self, step, n_clips, val_loss=None):
        if not self.enabled:
            return
        secs = self.clock() - self._val_t0
        self._last_val_s = secs
        _, r = self._read_peak_and_reset()        # val peak only; not folded into the training peak
        row = {"kind": "val", "step": step, "T": self.stage.T if self.stage else None,
               "batch_size": self.stage.batch_size if self.stage else None,
               "val_loss": _num(val_loss), "val_time_min": secs / 60, "val_clips": n_clips,
               "val_peak_reserved_gib": r / GIB if r is not None else None}
        self._append_row(row)
        self._scalars(step, {"val/time_min": secs / 60, "memory/val_peak_reserved_gib": row["val_peak_reserved_gib"]})
        print(f"[monitor] validation at step {step}: {n_clips} clips in {secs / 60:.1f} min"
              + (f", peak reserved {r / GIB:.2f} GiB" if r is not None else "")
              + f" | {self.time_left_text(step)}")
        self.draw("dashboard.png")
        print(self.overhead_text())

    # ── out of memory (Q5) ───────────────────────────────────────────────────
    def oom_report(self, err=None):
        """Print where the OOM happened. The caller re-raises the original error."""
        if not self.enabled:
            return ""
        st = self.stage
        parts = [f"[monitor] CUDA OUT OF MEMORY at step {self._step}"]
        if st is not None:
            parts.append(f"T={st.T}, batch_size={st.batch_size} (stage started at step {st.first_step})")
        if self.has_cuda:
            parts.append(f"allocated {self.cuda.memory_allocated() / GIB:.2f} GiB, "
                         f"reserved {self.cuda.memory_reserved() / GIB:.2f} GiB, "
                         f"peak reserved {self.cuda.max_memory_reserved() / GIB:.2f} GiB "
                         f"of {self.gpu_total / GIB:.2f} GiB total")
        if st is not None and st.peak_reserved:
            parts.append(f"stage peak reserved so far {st.peak_reserved / GIB:.2f} GiB")
        if err is not None:
            parts.append(f"error: {str(err).splitlines()[0] if str(err) else type(err).__name__}")
        msg = " | ".join(parts)
        print(msg)
        return msg

    def install_oom_handler(self, shell=None):
        """Jupyter/Colab: on a CUDA OOM in any later cell, print oom_report() and then the normal
        traceback (the error still stops the cell). Avoids re-indenting the training loop in a try/except.
        Returns False outside IPython (plain scripts must catch the error themselves)."""
        if not self.enabled:
            return False
        if shell is None:
            try:
                from IPython import get_ipython
                shell = get_ipython()
            except ImportError:
                shell = None
        if shell is None:
            print("[monitor] not running in IPython: OOM handler not installed")
            return False
        mon = self

        def _handler(sh, etype, value, tb, tb_offset=None):
            mon.oom_report(value)
            sh.showtraceback((etype, value, tb), tb_offset=tb_offset)

        shell.set_custom_exc((torch.cuda.OutOfMemoryError,), _handler)
        return True

    # ── summaries ────────────────────────────────────────────────────────────
    def summary(self):
        out = []
        for st in self.stages:
            med = st.median_s()
            out.append({"T": st.T, "batch_size": st.batch_size, "first_step": st.first_step,
                        "steps": len(st.step_s) + (st.first_step_s is not None),
                        "first_step_s": st.first_step_s, "median_s_per_step": med,
                        "it_per_s": (1.0 / med) if med else None,
                        "peak_alloc_gib": st.peak_alloc / GIB if self.has_cuda else None,
                        "peak_reserved_gib": st.peak_reserved / GIB if self.has_cuda else None,
                        "skipped_steps": st.skipped})
        return out

    def _print_stage_summary(self, st, step):
        med = st.median_s()
        mem = (f"peak alloc {st.peak_alloc / GIB:.2f} / reserved {st.peak_reserved / GIB:.2f} GiB "
               f"({100 * st.peak_reserved / self.gpu_total:.0f} % of {self.gpu_total / GIB:.1f} GiB)"
               if self.has_cuda else "memory n/a (no CUDA)")
        speed = (f"median {med:.3f} s/step ({1 / med:.2f} it/s)" if med else "no timed steps")
        first = f"first step {st.first_step_s:.1f} s" if st.first_step_s is not None else ""
        print(f"[monitor] stage T={st.T}, bs={st.batch_size} (steps {st.first_step}–{step - 1}): "
              f"{speed}; {first}; {mem}; skipped steps {st.skipped} | {self.time_left_text(step)}")

    def time_left_text(self, step):
        est = self.time_left(step)
        if est is None:
            return "time left: n/a"
        train_s, val_s, lower = est
        ge = "≥ " if lower else "≈ "
        return (f"time left {ge}{_fmt_time(train_s + val_s)} (training {_fmt_time(train_s)}, "
                f"validation {_fmt_time(val_s)}"
                f"{'; lower bound: later stages timed at the current speed' if lower else ''})")

    def time_left(self, step):
        """(train seconds, val seconds, is_lower_bound) or None. Stages not yet run use the current
        stage's speed, so the estimate is a lower bound while larger-T stages remain."""
        if not (self.enabled and self.total_steps and self.stage):
            return None
        med = self.stage.median_s()
        if med is None:
            return None
        remaining = max(0, self.total_steps - step)
        lower = any(s.start_step > step for s in self.schedule)
        val_s = 0.0
        if self.val_every and self._last_val_s is not None:
            n_val = self.total_steps // self.val_every - step // self.val_every
            val_s = max(0, n_val) * self._last_val_s
        return remaining * med, val_s, lower

    def finish(self, step):
        if not self.enabled or self.stage is None:
            return
        self._fold_peak()
        self._print_stage_summary(self.stage, step)
        if self.skipped_total:
            print(f"[monitor] {self.skipped_total} optimizer steps were skipped by the GradScaler (S15)")
        self.draw("dashboard_final.png", also_latest=True)
        print(self.overhead_text())

    def overhead(self):
        """Q7: monitor time (draws excluded) as % of the measured training-step time."""
        step_s = sum(sum(st.step_s) + (st.first_step_s or 0.0) for st in self.stages)
        return {"monitor_s": self.overhead_s, "step_s": step_s,
                "pct": 100.0 * self.overhead_s / step_s if step_s else None,
                "draw_s": self.draw_s, "n_draws": self.n_draws}

    def overhead_text(self):
        o = self.overhead()
        pct = f"{o['pct']:.3f} %" if o["pct"] is not None else "n/a"
        return (f"[monitor] own overhead: {o['monitor_s']:.2f} s = {pct} of {o['step_s'] / 60:.1f} min step time "
                f"(target < 1 %); dashboard draws: {o['n_draws']} in {o['draw_s']:.1f} s (not counted)")

    # ── dashboard (Q3b, Q10) ─────────────────────────────────────────────────
    def draw(self, name, also_latest=False):
        """Draw the dashboard to out_dir/name (and dashboard.png). CPU only; never raises."""
        if not (self.enabled and self.out_dir and self.rows):
            return None
        try:
            t0 = self.timer()
            path = self.out_dir / name
            draw_dashboard(self.rows, path, grad_clip=self.grad_clip,
                           gpu_total_gib=self.gpu_total / GIB if self.gpu_total else None)
            if also_latest and name != "dashboard.png":
                shutil.copyfile(path, self.out_dir / "dashboard.png")
            dt = self.timer() - t0
            self.draw_s += dt
            self.n_draws += 1
            print(f"[monitor] dashboard → {path} ({dt:.1f} s)")
            return path
        except Exception as e:                    # a plotting failure must never stop training
            print(f"[monitor] WARNING: dashboard not drawn ({type(e).__name__}: {e})")
            return None


def draw_dashboard(rows, path, grad_clip=None, gpu_total_gib=None):
    """Six panels on one step axis; vertical lines at stage switches. Uses Figure + Agg directly
    (no pyplot), so the notebook's inline backend is untouched and no figure stays open."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    train = [r for r in rows if r["kind"] == "train"]
    val = [r for r in rows if r["kind"] == "val"]

    def series(rs, key):
        pts = [(r["step"], r[key]) for r in rs if r.get(key) is not None]
        return [p[0] for p in pts], [p[1] for p in pts]

    switches = [b["step"] for a, b in zip(train, train[1:]) if a["T"] != b["T"]]

    fig = Figure(figsize=(14, 10), dpi=90)
    FigureCanvasAgg(fig)
    axes = fig.subplots(3, 2, sharex=True).ravel()

    ax = axes[0]
    for k in LOSS_KEYS:
        x, y = series(train, f"loss_{k}")
        pos = [(a, b) for a, b in zip(x, y) if math.isfinite(b) and b > 0]
        if pos:
            ax.plot(*zip(*pos), label=k, lw=1)
    ax.set_yscale("log")
    ax.set_title("train losses (log)")
    ax.legend(fontsize=7, ncol=3)

    ax = axes[1]
    x, y = series(val, "val_loss")
    ax.plot(x, y, "o-", ms=3, lw=1)
    ax.set_title("validation loss (loss/val_rec)")

    ax = axes[2]
    x, y = series(train, "grad_norm")
    fin = [(a, b) for a, b in zip(x, y) if math.isfinite(b)]
    if fin:
        ax.plot(*zip(*fin), lw=1)
    bad = [a for a, b in zip(x, y) if not math.isfinite(b)]
    for a in bad:
        ax.axvline(a, color="red", lw=0.8, alpha=0.6)
    if grad_clip is not None:
        ax.axhline(grad_clip, color="gray", ls="--", lw=1, label=f"clip {grad_clip}")
        ax.legend(fontsize=7)
    x, y = series(train, "skipped_steps")
    n_skip = int(y[-1]) if y else 0
    ax.set_title(f"grad norm before clipping (skipped steps: {n_skip}; red = non-finite)")

    ax = axes[3]
    x, y = series(train, "lr")
    ax.plot(x, y, lw=1)
    ax.set_title("learning rate")

    ax = axes[4]
    x, y = series(train, "s_per_step")
    ax.plot(x, y, lw=1)
    ax.set_title("s / step (mean per logging window)")

    ax = axes[5]
    x, y = series(train, "peak_reserved_gib")
    ax.plot(x, y, lw=1, label="train peak reserved")
    x, y = series(train, "peak_alloc_gib")
    ax.plot(x, y, lw=1, label="train peak allocated")
    x, y = series(val, "val_peak_reserved_gib")
    if x:
        ax.plot(x, y, "s", ms=3, label="val peak reserved")
    if gpu_total_gib:
        ax.axhline(gpu_total_gib, color="red", ls="--", lw=1, label=f"GPU total {gpu_total_gib:.1f} GiB")
    ax.set_title("GPU memory (GiB)")
    ax.legend(fontsize=7)

    for ax in axes:
        for s in switches:
            ax.axvline(s, color="k", ls=":", lw=0.8)
        ax.grid(alpha=0.3)
    for ax in axes[4:]:
        ax.set_xlabel("step")
    last = train[-1]["step"] if train else (val[-1]["step"] if val else 0)
    fig.suptitle(f"AnyUp3D training monitor — step {int(last)}"
                 + (f" (stage switches at {', '.join(str(int(s)) for s in switches)})" if switches else ""))
    fig.tight_layout()
    fig.savefig(path)
    return path
