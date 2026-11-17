# T05_training — Training Monitor Module (Debugging Built Into the Real Run)

| Field | Value |
|---|---|
| Task ID | T05_training |
| Plan | P1_training Training pipeline (`plan.md`) |
| Branch | `feat/train-monitor`, branched from `main` (`f05ea4d`, teacher fix merged; D-008) |
| Status | done |
| Depends on | — |
| Blocks | T04_training (its before/after memory numbers are measured through the monitor) |

## Goal

Make debugging part of every real training run, not something pasted in by hand. A small, tested `TrainMonitor` in `anyup/` records speed, GPU memory and validation time, and it catches bad losses and out-of-memory crashes with clear messages. The notebook only switches it on and calls it. It must cost no GPU memory and measurably little time (Muaaz, 2026-11-15; the `CLAUDE.md` rule "Debugging is a habit").

## Scope

- In scope:
  - New `anyup/data/training/monitor.py` (`TrainMonitor`) + `tests/test_train_monitor.py` (CPU).
  - Training-health numbers besides speed/memory: gradient norm (before clipping), GradScaler scale, count of skipped steps (Q3b, Q11).
  - A PNG dashboard on Drive (losses, gradient norm, lr, speed, memory) and its history CSV, reloaded on resume (Q3b, Q9, Q10).
  - `Train_3D_full.ipynb`: create the monitor (cell 12), call it at fixed points in cell 13 (incl. keeping `clip_grad_norm_`'s return value), and add the config switches (cell 3).
  - A Colab debug run proving the overhead is small and that the monitor reproduces D-021/D-023's numbers.
- Out of scope:
  - Munther's experiment notebooks (they could use it later).
  - Changing what training computes, any loss, or the validation protocol.
  - Retiring `scripts/debug/teacher_debug_cells.py` (D-024). It stays as the historical method. After this task, debug runs use the monitor.

## Findings

- **A. Today, debugging means pasted cells.** The teacher fix needed temporary cells 12.5/13.5 (D-021, D-023). They wrap notebook functions (`get_current_stage`, `build_loader`) to get timings, and were accidentally saved into the notebook through Colab (D-024).
- **B. Cell 13 measures nothing about itself.** It has no timing, no memory reading, no check for NaN or infinite losses, and an out-of-memory crash shows only PyTorch's message, without the stage, T or batch size.
- **C. What the debug runs showed is worth having in every run.** Per-stage it/s (0.34–0.41 on a T4), peak allocated/reserved memory (≈ 10.1 GiB at T=2/4; out of memory at T=8), validation time (≈ 6.4 min per pass), and whether losses are finite.
- **D. Cheap by construction.**
  - `torch.cuda.max_memory_allocated/reserved()` and `reset_peak_memory_stats()` read or reset counters; `time.time()` reads the clock. None of them allocates GPU memory.
  - Losses are already turned into Python numbers (`.item()`) at the existing logging points (every `log_every_n_steps` = 50), so reading them there adds no extra GPU wait.
  - `scaler.step(optimizer)` already waits for the GPU every step (it reads its found-inf flag with `.item()` before deciding to step), so a per-step NaN check (Q6) should add almost nothing. Q7 measures it.
- **E. Training health is half-logged (2026-11-16).** Cell 13 already writes `loss/total|rec|inp|aug|temporal`, `loss/val_rec`, `lr`, `curriculum/T|bs`, `data/epoch` to TensorBoard. Missing:
  - The gradient norm: `clip_grad_norm_(...)` computes it, but its return value is discarded.
  - The GradScaler scale. With bf16 autocast, the scaler's only effect is to silently skip steps whose gradients are inf/NaN (S15); nothing shows these skips today.
- **F. Saves and logging points don't line up.** `save_checkpoint` runs at `(step + 1) % 250 == 0` (steps 249, 499, …), logging at `step % 50 == 0` (250, 500, …). A check at logging points alone would catch a NaN only after it was saved (Q6 addition).
- **G. TensorBoard needs a live session.** On Colab it shows nothing after a disconnect unless reopened; a PNG on Drive can be looked at any time (Q3b).

## Q&A (decisions agreed before implementation)

| # | Question | Answer |
|---|---|---|
| 1 | Name and place: `TrainMonitor` in `anyup/data/training/monitor.py`, next to `teacher.py`? | **Yes: one class in one file, `anyup/data/training/monitor.py`** (Muaaz, 2026-11-16). He asked about a `debugging` folder; it was not made because it would hold one file and contradict `CLAUDE.md`. Revisit if Munther's experiments start using the monitor. |
| 2 | On by default (`cfg.monitor = True`)? | **On by default**, once the overhead test (Q7) passes (Muaaz, 2026-11-16). |
| 3 | What to record. Proposal: per step, wall time only (no GPU wait). Per logging point: mean s/step and it/s, peak allocated/reserved memory and % of the GPU, all to TensorBoard (`perf/…`, `memory/…`). Per validation: duration and clips. At each stage switch and at the end, a printed summary per stage, plus the estimated time left for the run. | **Yes to the proposal** (Muaaz, 2026-11-16). Plus: **graphs drawn**, if they don't significantly affect memory or latency, covering the loss and the training itself, not only memory and time. **3b → (b)** (Muaaz, 2026-11-16): TensorBoard (losses already logged in cell 13) + new `train/grad_norm` (the return value of `clip_grad_norm_`, now discarded) + `train/grad_scale` (S15), **plus a PNG dashboard on Drive** drawn at each validation, stage switch and the end. Asked first whether it costs memory: no GPU memory; CPU ≈ 0.5 MB history + ≈ 20–50 MB for ≈ 1 s per draw (estimates; Q7 run measures). |
| 4 | A loss becomes NaN or infinite: (a) stop with a clear error (which loss, step, T, batch size); (b) warn and skip the step. | **(a) stop with a clear error** (Muaaz, 2026-11-16). |
| 5 | Out of memory: print the stage, T, batch size, current and peak memory and the GPU total, then re-raise the normal error (no automatic retry or batch-size change). | **Yes** (Muaaz, 2026-11-16). |
| 6 | Where the checks run: at the existing logging points only (every 50 steps), or every step? Every step catches a NaN sooner but makes the CPU wait for the GPU each step. | **The proposal: NaN check every step if the overhead test shows < 1 %; otherwise at the logging points** (Muaaz, 2026-11-16; option (c), a GPU-side flag, was declined). **Plus a check right before every checkpoint save** (Muaaz, 2026-11-16): saves run at `(step + 1) % 250 == 0` (step 249, 499, …), logging at `step % 50 == 0` (250, 500, …), so without it a NaN from steps 201–249 would be saved before it is caught. |
| 7 | Overhead target and test: same short run with the monitor on vs off, median s/step difference < 1 %? | **Target < 1 %: yes. Method changed to self-timing within one run** (Muaaz, 2026-11-16; old → new: on/off pair → the monitor times its own calls, draws excluded, as % of step time; an on/off pair can't resolve 1 % through run-to-run noise). |
| 8 | Order: do this before T04_training, so T04 measures through it? | **Yes** (Muaaz, 2026-11-16). |
| 9 | Dashboard history: where does the monitor keep the numbers it plots (a disconnect loses RAM; a resumed run would draw only the steps after the resume)? | **CSV on Drive** (Muaaz, 2026-11-16): one row per logging point appended to `monitor_history.csv` in `cfg.log_dir`, reloaded on resume. |
| 10 | Dashboard PNG files: how many to keep? | **Latest + per stage** (Muaaz, 2026-11-16): `dashboard.png` overwritten at every draw, plus a frozen copy at each stage end and at the final step. |
| 11 | The gradient norm becomes inf/NaN (the GradScaler then silently skips the step, S15): stop, or count? | **Count and report** (Muaaz, 2026-11-16): count skipped steps, print a warning, show them on the dashboard and in the stage summary; training behaves as today. |
| 12 | (implementation) Catching an OOM: `try/except` around the loop body re-indents ≈ 140 lines of cell 13. Use that, or an IPython handler? | **IPython handler** (Muaaz, 2026-11-16): `mon.install_oom_handler()`, one line, no re-indentation; tested with a fake shell. |
| 13 | (implementation) Where to create the monitor: cell 12 (design), or the top of cell 13 where `max_steps` is defined? | **Top of cell 13** (Muaaz, 2026-11-16). Old → new: design said cell 12; moved so `max_steps` isn't repeated. |
| 14 | (implementation) How do the test-run settings get into Colab? | **Commented-out block in cell 3** (Muaaz, 2026-11-16), like `# cfg.debug = True`; uncommented in Colab only, never saved uncommented. |
| 15 | (implementation) Cell 2 clones `fix/teacher-full-16frame-clips` and skips an existing clone. Change? | **Branch + switch** (Muaaz, 2026-11-16): `REPO_BRANCH = "feat/train-monitor"` (→ `main` after the merge); an existing clone is fetched + checked out. |

## Design

### `anyup/data/training/monitor.py` (one class, Q1)

```python
class TrainMonitor:
    def __init__(self, writer=None, out_dir=None, log_every=50, total_steps=None, schedule=None,
                 grad_clip=None, nan_every_step=True, enabled=True,
                 clock=time.time, cuda=torch.cuda): ...   # clock/cuda injectable for tests; out_dir = cfg.log_dir
    def resume(self, start_step): ...                     # Q9: reload monitor_history.csv, drop rows with step >= start_step
    def step_start(self, step, T, batch_size): ...        # wall time; stage switch → stage summary + frozen dashboard (Q10) + reset peak memory
    def step_end(self, step, losses: dict, grad_norm=None, grad_scale=None): ...
        # every step (Q6, if nan_every_step): NaN/inf loss → raise (Q4); non-finite grad_norm → skipped-step count + warning (Q11)
        # at logging points: perf/ memory/ train/ scalars → writer; one row → monitor_history.csv (Q9)
    def check_before_save(self, step, losses): ...        # Q6 addition: NaN/inf loss → raise before a checkpoint is written
    def val_start(self): ...
    def val_end(self, step, n_clips, val_loss): ...       # val duration + memory → writer; redraw dashboard.png
    def oom_report(self, err): ...                        # Q5: context message, then the caller re-raises
    def finish(self, step): ...                           # final stage summary + dashboard_final.png
    def summary(self) -> dict: ...                        # per-stage numbers
```

What it records (Q3, Q3b, Q11):

| When | What | Where |
|---|---|---|
| Every step | wall time (no GPU wait); loss finite check (Q6); grad-norm finite check → skipped count (Q11) | RAM |
| Logging points (50) | mean s/step, it/s; peak allocated/reserved GiB and % of GPU total; grad norm; scaler scale; skipped steps so far | TensorBoard `perf/s_per_step`, `perf/it_per_s`, `memory/peak_alloc_gib`, `memory/peak_reserved_gib`, `memory/peak_reserved_pct`, `train/grad_norm`, `train/grad_scale`, `train/skipped_steps`; one row in `monitor_history.csv` (also the 5 losses and lr) |
| Validation | duration, clips, val loss, val peak memory | TensorBoard `val/time_min`, `memory/val_peak_reserved_gib`; CSV row; redraw `dashboard.png` |
| Stage switch / end | per-stage summary printed (median s/step, it/s, peak memory, skipped steps, estimated time left) | stdout; `dashboard_T{t}_end.png`, `dashboard_final.png` |

- Peak-memory counters are reset at each stage switch and around validation, so training and validation peaks are separate.
- The estimated time left uses the measured speed of finished stages and the current one. For stages not yet started it uses the current speed, so it's a lower bound (later stages have larger T); the printout says so.
- No CUDA → memory columns are empty and reported as unavailable; everything else works (CPU dry runs).
- `enabled=False` → every method returns at once; nothing is written.

### Dashboard (Q3b, Q10)

- One PNG, 6 panels, shared step axis, vertical lines at stage switches: (1) train losses (total, rec, inp, aug, temporal; log y); (2) val loss; (3) grad norm with the clip line (`cfg.grad_clip_max_norm`) and skipped steps as markers; (4) lr; (5) s/step; (6) peak reserved memory with the GPU total line.
- Drawn with `matplotlib.figure.Figure` + the Agg canvas directly (not `pyplot`), so it doesn't change the notebook's inline backend or keep figures open. matplotlib is imported at the first draw.
- Drawn only at validations (every 1000 steps), stage switches and the end; never inside a training step.
- Files in `cfg.log_dir` (Drive): `monitor_history.csv`, `dashboard.png` (latest), `dashboard_T{t}_end.png` per stage, `dashboard_final.png`.

### Notebook changes (targeted edits; training math unchanged)

- Cell 3 (`TrainConfig`): `monitor: bool = True` (Q2) and `monitor_nan_every_step: bool = True` (Q6; set to False if the Q7 test shows ≥ 1 % overhead).
- Cell 13, top (Q13, after `max_steps` and the `VAL_*` lines): create `mon = TrainMonitor(writer, out_dir=cfg.log_dir, log_every=cfg.log_every_n_steps, total_steps=max_steps, schedule=cfg.t_schedule, val_every=cfg.val_every_n_steps, grad_clip=cfg.grad_clip_max_norm, nan_every_step=cfg.monitor_nan_every_step, enabled=cfg.monitor)`; `mon.resume(global_step)`; `mon.install_oom_handler()` (Q12).
- Cell 13, at fixed points inside the loop (no wrapping of notebook functions, no re-indentation):

```python
mon.step_start(step, current_stage.t, current_stage.batch_size)   # after the stage-switch block, before next(loader_iter)
...                                                                # existing training step, unchanged
grad_norm = torch.nn.utils.clip_grad_norm_(anyup.parameters(), cfg.grad_clip_max_norm)   # was: return value discarded
...
losses = {"total": total_loss, "rec": loss_rec, "inp": loss_inp, "aug": loss_aug, "temporal": loss_temp}
mon.step_end(step, losses, grad_norm=grad_norm, lr=optimizer.param_groups[0]["lr"],
             grad_scale=scaler.get_scale() if step % cfg.log_every_n_steps == 0 else None)
...
if (step + 1) % cfg.save_every_n_steps == 0:
    mon.check_before_save(step, losses)                      # Q6 addition
    save_checkpoint(step + 1, data_epoch)
...
anyup.eval(); mon.val_start(); ...existing validation...; mon.val_end(step + 1, len(val_losses), mean val loss)
...
mon.finish(max_steps)                                        # before the final save
```

- Out of memory (Q12): no `try/except` in cell 13. `mon.install_oom_handler()` registers an IPython handler (`set_custom_exc`) that prints `oom_report()` and then the normal traceback for any CUDA OOM in a later cell (training or validation).

- The existing `writer.add_scalar("loss/…")` lines stay as they are (same tags as run_01/run_02).

## Steps

1. Branch `feat/train-monitor` from `main` (Muaaz creates it).
2. `monitor.py` + tests (fake clock, fake CUDA counters, fake writer, temp dir); run locally.
3. Notebook cells 3/12/13; clear outputs; CPU dry run (memory parts off).
4. Colab T4 short run (300 steps, T=2 → T=4 at step 150, one validation at step 200) with the monitor's self-timing (Q7). Check that it reproduces D-021/D-023's it/s and memory at T=2/4 and the validation time; look at the dashboard PNG; measure the draw time and RAM.
5. Update the docs that point to `scripts/debug/teacher_debug_cells.py` to say "use the monitor"; `DECISIONS.md` entry.

## Tests / Regression Checks

`tests/test_train_monitor.py`:
- With a fake clock: per-stage median and it/s are correct across a stage switch; the first step of each stage is reported separately.
- With fake CUDA counters: the peak is reset at stage switches; validation memory is measured separately from training.
- A NaN or infinite loss raises with the loss name, step and T in the message (Q4).
- `oom_report` text includes stage, T, batch size and memory numbers.
- `enabled=False` does nothing and writes nothing; no CUDA → no crash, and memory is reported as unavailable.
- Scalars go to the writer under the agreed tags.
- `check_before_save` raises on a NaN loss (Q6 addition).
- A non-finite grad norm does not raise; it increments the skipped count and appears in the CSV and summary (Q11).
- History CSV: rows appended at logging points; `resume(start_step)` reloads it and drops rows with step ≥ start_step (Q9).
- Dashboard: `dashboard.png` written at `val_end`; `dashboard_T{t}_end.png` at a stage switch; `dashboard_final.png` at `finish` (Q10); drawing with NaN/inf values in the history does not crash.

Colab: self-timed overhead < 1 % (Q7). Numbers within noise of D-021/D-023 on the same setup. Dashboard draw time and RAM reported.

## Done When

- Tests pass; the notebook uses the monitor at the agreed points; outputs are cleared.
- The Colab run's self-timed overhead meets the target, and the numbers match D-021/D-023.
- The dashboard PNGs and `monitor_history.csv` appear on Drive and show losses, grad norm, lr, speed and memory for the run.
- `docs/DECISIONS.md` entry; log and Task Index up to date.

## Implementation Log

_Updated live during implementation. Per step: what changed, files touched, test results (actual numbers/shapes). Also log mid-implementation questions and answers, changed decisions (old → new, why), and anything added or removed. Sections above are edited in place to match, so the finished file is a report of what was actually done._

- **2026-11-15: task file written** (D-025). Nothing implemented. Status `Q&A`: Q1–Q8 are open.
- **2026-11-16: Q&A round 1.** Answered: Q2, Q4, Q5, Q7, Q8 (all as proposed), Q3 (as proposed, plus graphs). Still open: Q1 (Muaaz asked whether it is one class, and suggested a `debugging` folder), Q3b (graph form), Q6 (proposal vs option (c)).
- **2026-11-16: Q&A round 2.** Q1 → one file (no folder). Q6 → the proposal. Q3b is still open: Muaaz wants graphs of the loss and of the training itself, not only memory and time.
- **2026-11-16: Q&A round 3.** Q6 + a NaN check before each checkpoint save (added). Found while reading cell 13: bf16 autocast runs with a `GradScaler`, which silently skips steps with inf/NaN gradients; Muaaz chose to log it, not fix it here → **S15**. Q3b still open (Muaaz asked whether the PNG dashboard costs memory).
- **2026-11-16: Q&A round 4.** Q3b → (b), TensorBoard + PNG dashboard (after the memory estimate: 0 GPU, ≈ 0.5 MB + a brief 20–50 MB CPU). New questions from the dashboard design: Q9 history CSV on Drive, Q10 latest + per-stage PNGs, Q11 count skipped steps (don't stop). Added Findings D–G, rewrote Scope, Design, Steps, Tests from the actual cells 3/12/13. All questions answered; task file waiting for Muaaz's OK.
- **2026-11-16: task file approved ("go ahead with the task T05 file").** Status Q&A → in progress. Same turn: the OOM-warning rule went into `CLAUDE.md` (D-027). Working on `main`'s working tree; Muaaz creates `feat/train-monitor` (the uncommitted changes move with `git switch -c`).
- **Step 2 (code): `anyup/data/training/monitor.py` + `tests/test_train_monitor.py`.**
  - `TrainMonitor` as designed, plus: `NonFiniteLossError` (Q4); `draw_dashboard()` as a module function (testable); `lr` passed to `step_end`; `val_every` (time-left estimate includes the remaining validations). Skip warnings are printed for the first 10 per stage (`MAX_SKIP_WARNINGS`), then only counted. A dashboard failure prints a warning and never stops training. `dashboard.png` is copied from the frozen PNG instead of being drawn twice.
  - Peak memory: read and reset at every logging point (per-window peak on the graph), folded into a per-stage max; around validation the training peak is folded first, then the val peak is read alone.
  - Tests (CPU, fake clock / CUDA counters / writer, temp dir): **13 passed, 11 s** (local venv, torch 2.11.0+cpu, matplotlib 3.10.8). Two failures on the first run were fixed: `resume` parsed `"0.0"` with `int()` (code bug → `int(float())`); the fake CUDA kept the setup memory as "current", so the reset couldn't drop it (test bug).
  - Synthetic full-run dashboard (1,050 rows = 50k steps / 50 + 50 validations): **draw 6.6 s on first call (incl. the matplotlib import), 3.0 s after; Python heap peak 27 MiB.** My earlier "≈ 1 s per draw" estimate was low; ≈ 50 draws × 3 s ≈ 2.5 min over a 35 h run, still negligible. 0 GPU memory.
- **Implementation questions Q12, Q13** (Muaaz, 2026-11-16): OOM via an IPython handler (not `try/except`); monitor created at the top of cell 13 (old → new: cell 12). Design updated in place. `install_oom_handler()` + 2 tests added: **15 passed, 8.5 s**.
- **Step 3 (code): `Train_3D_full.ipynb` cells 3 and 13**, edited by a scratchpad script (each old string asserted to occur once; JSON written with the file's own `indent=2`, round-trip checked byte-identical first). Diff: **+23 / −1 lines** (the `−1` is `clip_grad_norm_` now kept as `grad_norm`). All code cells parse; outputs and execution counts stay empty.
  - Cell 3: `monitor`, `monitor_nan_every_step` (both `True`).
  - Cell 13: create + `resume(global_step)` + `install_oom_handler()`; `step_start` after the stage-switch block (so data loading counts in the step time); `step_end` after `scheduler.step()`; `check_before_save` before `save_checkpoint`; `val_start` after `anyup.eval()`; `val_end` after the `loss/val_rec` line (existing line unchanged); `finish` before the final save.
  - GPU memory: none added. The `losses` dict holds the same 0-dim loss tensors the loop variables already hold (graph freed by `backward()`); `grad_norm` is one 0-dim tensor.
  - **CPU dry run not possible locally** (no UCF101 data, no VideoMAE weights on this machine). Moved into the Colab run (Step 4).
- **Q7 method changed (Muaaz, 2026-11-16): old → new = two runs on vs off → self-timing within one run.** Why: two Colab runs differ by more than 1 % from noise (video decoding, Drive I/O, validation), so an on/off pair can't resolve 1 %. Added `_self_timed` (on `step_start`, `step_end`, `check_before_save`, `val_start`, `val_end`; `time.perf_counter`, injectable `timer`), `overhead()` / `overhead_text()` printed at every validation and at `finish`. Dashboard draws are excluded and reported separately. Any GPU wait from the monitor's `.item()` falls inside its own time, so the number is pessimistic. Tests: + 2 → **17 passed, 9.5 s**.
- **Step 4 settings approved** (Muaaz, 2026-11-16; OOM estimate shown first per the D-027 rule: monitor 0 GPU memory; T=2 bs 2 / T=4 bs 1 ≈ 10.1 GiB reserved of 15 GiB as in D-021; validation as in D-023; T=8 not reachable; likelihood low): `max_steps=300`, `val_every_n_steps=200`, `t_schedule` T=2@0 bs 2, T=4@150 bs 1, T=8@10,000 (kept only so the validation window stays 8 frames), `experiment_name="t05_monitor_on"`.
- **Q14, Q15 (Muaaz, 2026-11-16):** test settings as a **commented-out block in cell 3** (uncomment in Colab only); **cell 2: `REPO_BRANCH = "feat/train-monitor"`** (back to `main` after the merge) **and an existing `/content/anyup` is switched with `fetch --depth 1` + `checkout -B … FETCH_HEAD`** (CLAUDE.md rule; it used to keep the old clone silently). Cell 2 now raises a clear error if the switch fails and prints the code commit used.
  - The test block also recomputes `checkpoint_dir` / `log_dir`: they are built from `experiment_name` earlier in the cell, so changing only the name would have written the test into `run_03`'s folders.
  - Notebook diff now **+46 / −3** (cells 2, 3, 13). All code cells parse; outputs empty. Executing cell 3 with the block uncommented gives `experiment_name=t05_monitor_on`, `max_steps=300`, `val_every_n_steps=200`, schedule `[(2, 0, 2), (4, 150, 1), (8, 10000, 1)]`, `log_dir=…/runs/t05_monitor_on`, `checkpoint_dir=…/checkpoints/t05_monitor_on`.
  - In the test run: stage switch at step 150, validation after step 199 (T=4 stage), checkpoint after step 249, `finish` at 300.
- **2026-11-17: committed and pushed.** `feat/train-monitor` created from `main` (`f05ea4d`), commit `423f61f` (8 files, +1011 / −40), pushed to `origin/feat/train-monitor`.
- **Step 4, Colab run 1 (Tesla T4 14.6 GiB, PyTorch 2.14.1+cu130, code commit `423f61f`, test block in cell 3): 300/300 steps in 20:18, no NaN stop, no OOM.**

  | | this run (monitor, 150 steps per stage) | D-021 / D-023 (5 steps per stage) |
  |---|---|---|
  | T=2 bs 2 median | 3.011 s/step (0.33 it/s), first 5.7 s | 2.87 / 2.93 s (0.35 / 0.34 it/s), first 6.0 s |
  | T=2 peak alloc / reserved | 9.91 / 10.10 GiB (69 %) | 9.91 / 10.06 GiB |
  | T=4 bs 1 median | 2.529 s/step (0.40 it/s), first 2.7 s | 2.45 / 2.44 s (0.41 it/s), first 3.0 s |
  | T=4 peak alloc / reserved | 9.98 / 10.20 GiB (70 %) | 9.94 / 10.16 GiB |
  | Validation | 459 clips in 6.4 min, peak reserved 10.16 GiB | ≈ 6.4 min (0.83 s/clip × 459) |

  - **Overhead: 0.11 s = 0.013 % of 13.9 min step time** (target < 1 %; 0.014 % at the validation). Dashboard draws: 3 in 3.7 s (0.9–1.8 s each), not counted. So Q6's per-step NaN check stays on (no fallback needed).
  - Skipped steps (GradScaler, S15): 0 in both stages.
  - Checkpoints at step 250 and 300 (final); dashboards `dashboard_T2_end.png`, `dashboard.png`, `dashboard_final.png` written to `runs/t05_monitor_on/`.
  - Both stages are 3–5 % slower than D-021/D-023. Not from the monitor (its own time is 0.08–0.11 s in total). The cause is not separated: a different Colab VM, and a median over 150 steps instead of 5. The T=4 peak is +0.04 GiB higher than in D-021, likewise not separated (more steps sampled for the peak, and a validation inside the stage).
  - **Found: time-left estimate shows "validation 0 h 00 min" before the first validation** (`time_left`, `monitor.py:398`: `_last_val_s is None` → `val_s = 0`). At step 150, one 6.4 min validation remained. In run_03 the estimate would leave out ≈ 5.3 h of validation until the first one. Display only. Also minor: "lower bound" printed because the T=8 stage at 10,000 is in the schedule, though it lies after `max_steps=300` (correct in a real run).
  - Not yet checked: the dashboard PNG contents, `monitor_history.csv` on Drive, the TensorBoard tags.
- **Run 1 outputs from Drive checked (2026-11-17; Muaaz copied them locally).**
  - `monitor_history.csv`: 7 rows (train at 0, 50, 100, 150, 200, 250; val at 200), all 22 columns filled. Matches the printed lines: val 459 clips in 6.37 min, `val_rec` 0.1035, val peak reserved 10.16 GiB; `grad_scale` 65536 throughout, `skipped_steps` 0; grad norm 0.016–0.090 (clip 1.0).
  - `dashboard_T2_end.png` (title "step 100") and `dashboard_final.png` (title "step 250", stage switch marked at 150): all 6 panels drawn (losses on a log axis, val loss, grad norm + clip line, lr, s/step, GPU memory + 14.6 GiB line). `dashboard.png` = `dashboard_final.png` (same size, 107,823 B), as designed.
  - Observations, no change made: the last history row is step 250, since logging runs at multiples of 50 and the loop ends at 299, so steps 251–299 aren't in the dashboard (only in the printed stage summary). `loss_temporal` is 0.0 at T=2, so the log-axis line starts at step 150. That comes from the training code, not the monitor. The empty val panel before the first validation shows a ±0.04 placeholder axis (cosmetic).
  - TensorBoard tags: checked in the next entry.
- **Run 1 TensorBoard file checked (2026-11-17):** one file, `events.out.tfevents.1794899319.c47065976d03.457.0` (5.5 KB), read with the local `tensorboard` 2.20.0 `EventAccumulator`.
  - **All 10 monitor tags are present:** `perf/s_per_step`, `perf/it_per_s`, `memory/peak_alloc_gib`, `memory/peak_reserved_gib`, `memory/peak_reserved_pct`, `train/grad_norm`, `train/grad_scale`, `train/skipped_steps` (6 points each, steps 0–250), and `val/time_min`, `memory/val_peak_reserved_gib` (1 point, step 200).
  - Values = `monitor_history.csv` at the same steps for every tag: max relative difference 4.8e-8 (float32 storage in tfevents).
  - The notebook's own tags are still written: `loss/{total,rec,inp,aug,temporal,val_rec}`, `lr`, `curriculum/T`, `curriculum/bs`, `data/epoch`. No non-scalar data.
  - Muaaz's `%tensorboard` view in Colab showed "No dashboards are active". The file has data, so it's a viewer issue (most likely Drive not mounted in that Colab session), not a monitor issue. Not followed up.
- **Step 5 (docs), 2026-11-17.** Docs that pointed to `scripts/debug/teacher_debug_cells.py` as the method to use now point to the monitor:
  - `scripts/debug/teacher_debug_cells.py` header: "SUPERSEDED (T05_training, D-028)". The file is kept as the historical method behind D-021 / D-023. No code changed.
  - `T04_training_shared_image_encoding.md` step 4: the monitor only, the fallback to the old cells removed. Added that run 1 reproduced D-021's memory within 0.04 GiB, so T04's before/after stays comparable to D-021.
  - Left as they are: the historical mentions in D-024 / D-025; `CLAUDE.md` and `docs/working rules/notebooks_version_control.md` already point to the monitor.
  - `docs/DECISIONS.md`: **D-028** (run 1 accepted, time-left fix, old cells superseded; overrides D-024's "same method for T04").
- **Fix (a) for the time-left bug (Muaaz chose (a), 2026-11-17).** `time_left` returns `val_s = None` while validations remain but none has been timed yet. `time_left_text` then prints "validation not timed yet" and marks the total "≥". When no validation is left in the run, `val_s` is 0.0 as before. The cosmetic "lower bound" from a stage beyond `max_steps` is left unchanged (option (b) not chosen).
  - Run 1's step-150 line now reads (replayed on CPU with fake timings): `time left ≥ 0 h 07 min (training 0 h 07 min, validation not timed yet; lower bound: later stages timed at the current speed)`.
  - Files: `anyup/data/training/monitor.py` (`time_left`, `time_left_text`), `tests/test_train_monitor.py` (+ `test_time_left_before_first_validation`). **18 passed, 9.8 s** (local CPU venv). The first run of the new test failed: it passed a tuple instead of a `Stage` (a test bug, fixed).
  - CPU-only text change, so no second Colab run is needed (agreed in the option list).
- **Close-out (2026-11-17).** Muaaz chose (a): the 3–5 % slower steps count as a match within Colab VM noise. Memory and validation time match D-021 / D-023; the monitor's own time is 0.11 s, so it isn't the cause. Not logged as a skipped item. All Done When items are met:
  - tests: 18 pass (local CPU); the notebook calls the monitor at the agreed points; outputs empty;
  - overhead 0.013 % (< 1 %); memory and validation time = D-021 / D-023, speed as above;
  - dashboards + `monitor_history.csv` on Drive, showing losses, grad norm, lr, speed and memory; TensorBoard tags = CSV;
  - D-028; this log and the Task Index updated.
  - Status in progress → **done**. Left for Muaaz: commit and push the fix + docs (on top of `423f61f`), merge `feat/train-monitor` into `main`, then set `REPO_BRANCH` in cell 2 back to `"main"`. Not verified: TensorBoard in the Colab UI (it showed no dashboards, see above); the 3–5 % speed gap's cause.
