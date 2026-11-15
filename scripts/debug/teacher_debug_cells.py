# Debug-run cells for Train_3D_full.ipynb: measure speed, GPU memory, losses and validation time.
#
# NOT an importable module: these cells read and wrap the notebook's own globals
# (cfg, TStage, DRIVE_ROOT, SummaryWriter, get_current_stage, build_loader, torch, Path).
# Paste "Cell 12.5" after cell 12 and "Cell 13.5" after cell 13, run cells 1-12, 12.5, 13, 13.5,
# then restart the runtime. Never save the notebook with these cells (D-024).
#
# This is the exact measurement used for D-021 / D-023 (teacher fix, Tesla T4), kept so later runs
# (e.g. T04_training's before/after) are measured the same way. A proper, pluggable training monitor
# that replaces these cells is proposed separately.
#
# Settings below = the D-023 rerun: 5 steps per stage (T=2/4/8), validation after step 9 (end of T=4).

# %% Cell 12.5
# ── Cell 12.5 (validation-timing rerun) — TEMPORARY. Don't save the notebook. ──
# Same short run as before, but validation runs at the end of the T=4 stage (after step 9),
# so it's measured before T=8 runs out of memory. Uses no GPU memory itself.
import time, math
from collections import defaultdict
from torch.utils.data import DataLoader, Subset

cfg.debug, cfg.debug_steps = True, 15
cfg.val_every_n_steps      = 10                          # validation after step 9 (still T=4)
cfg.log_every_n_steps      = 1
cfg.t_schedule = [TStage(t=2, start_step=0,  batch_size=2),   # steps 0–4
                  TStage(t=4, start_step=5,  batch_size=1),   # steps 5–9, then validation
                  TStage(t=8, start_step=10, batch_size=1)]   # step 10: expected to run out of memory on a T4
VAL_LIMIT = 50                                           # ↓ validation clips timed; the full-split time is estimated

ckpt_dir = Path(f"{DRIVE_ROOT}/checkpoints/debug_teacher"); ckpt_dir.mkdir(parents=True, exist_ok=True)

_scalars = defaultdict(list)
class _RecWriter(SummaryWriter):
    def add_scalar(self, tag, value, global_step=None, *a, **k):
        _scalars[tag].append((global_step, float(value)))
        return super().add_scalar(tag, value, global_step, *a, **k)
writer = _RecWriter(log_dir=f"{DRIVE_ROOT}/runs/debug_teacher")

_stamps, _peak, _t = [], {}, {}

def _unwrap(f):
    return getattr(f, "_orig", f)
_orig_stage, _orig_loader = _unwrap(get_current_stage), _unwrap(build_loader)

def _mem():
    return torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()

def _store_peak(T):
    a, r = _peak.get(T, (0, 0)); ma, mr = _mem()
    _peak[T] = (max(a, ma), max(r, mr))

def get_current_stage(step):                             # called at the start of every step
    s, now = _orig_stage(step), time.time()
    if _t.get("in_val"):                                 # validation ran at the end of the previous step and just ended
        _t["in_val"] = False
        _t["val1"], _t["peak_val"] = now, _mem()
    elif _stamps:
        _store_peak(_stamps[-1][1])                      # peak of the training step that just finished
    if not _stamps or _stamps[-1][1] != s.t:
        torch.cuda.reset_peak_memory_stats()             # each stage gets its own peak
    _stamps.append((now, s.t))
    return s
get_current_stage._orig = _orig_stage

def build_loader(*a, **k):                               # the test loader marks the start of validation
    ld = _orig_loader(*a, **k)
    if k.get("split") == "test":
        _t["n_val_full"] = len(ld.dataset)
        _t["n_val_used"] = min(VAL_LIMIT, len(ld.dataset))
        ld = DataLoader(Subset(ld.dataset, range(_t["n_val_used"])), batch_size=1, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
        _store_peak(_stamps[-1][1])                      # training peak of the step that just ran
        torch.cuda.reset_peak_memory_stats()             # measure validation on its own
        _t["in_val"], _t["val0"] = True, time.time()
    return ld
build_loader._orig = _orig_loader

torch.cuda.reset_peak_memory_stats()
print("Debug setup ready on", torch.cuda.get_device_name(0),
      f"({torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB)")

# %% Cell 13.5
# ── Cell 13.5 (validation-timing rerun) — TEMPORARY: print what cell 12.5 recorded. ──
import statistics as st
G = 2**30
steps = _stamps[1:]                                      # drop the call made before the loop
T_of  = {i: T for i, (_, T) in enumerate(steps)}

dur = defaultdict(list)
for (t0, T), (t1, _) in zip(steps, steps[1:]):
    if "val0" in _t and t0 < _t["val0"] < t1:            # the step before validation ends where validation starts
        t1 = _t["val0"]
    dur[T].append(t1 - t0)

total = torch.cuda.get_device_properties(0).total_memory
print(f"GPU: {torch.cuda.get_device_name(0)}  ({total / G:.1f} GiB total)\n")

print("── Training: speed and memory per stage ──")
for T in sorted({T for _, T in steps}):
    if T not in dur:
        print(f"T={T}: no completed step (out of memory or stopped)")
        continue
    d = dur[T][1:] or dur[T]
    a, r = _peak.get(T, (0, 0))
    print(f"T={T}: {len(dur[T])} steps | median {st.median(d):.2f} s/step ({1 / st.median(d):.2f} it/s) | "
          f"first step {dur[T][0]:.1f} s | peak {a / G:.2f} GiB allocated, {r / G:.2f} GiB reserved "
          f"({100 * r / total:.0f}% of GPU)")

print("\n── Validation (4-tubelet window, batch 1, no gradients) ──")
if "val0" in _t and "val1" in _t:
    sec = _t["val1"] - _t["val0"]
    per_clip = sec / _t["n_val_used"]
    a, r = _t["peak_val"]
    print(f"{_t['n_val_used']} of {_t['n_val_full']} clips in {sec:.0f} s ({per_clip:.2f} s/clip) "
          f"→ full pass ≈ {per_clip * _t['n_val_full'] / 60:.1f} min")
    print(f"peak during validation: {a / G:.2f} GiB allocated, {r / G:.2f} GiB reserved")
else:
    print("validation did not finish")

print("\n── Losses per step ──")
tags = ["loss/total", "loss/rec", "loss/inp", "loss/aug", "loss/temporal"]
vals = {t: dict(_scalars.get(t, [])) for t in tags}
print("step  T     total      rec      inp      aug  temporal")
for s in sorted(vals["loss/total"]):
    print(f"{s:4d} {T_of.get(s, '?'):>2} " + " ".join(f"{vals[t].get(s, float('nan')):8.4f}" for t in tags))
val = _scalars.get("loss/val_rec", [])
print("val_rec:", " ".join(f"step {s}: {x:.4f}" for s, x in val) if val else "not logged")
everything = [x for t in tags for _, x in _scalars.get(t, [])] + [x for _, x in val]
print(f"\nall losses finite: {all(math.isfinite(x) for x in everything)}  ({len(everything)} values)")
