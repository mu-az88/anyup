# T05_training — Training Monitor Module (Debugging Built Into the Real Run)

| Field | Value |
|---|---|
| Task ID | T05_training |
| Plan | P1_training Training pipeline (`plan.md`) |
| Branch | `feat/train-monitor`, branched from `main` after the teacher fix is merged into `main` (D-008) |
| Status | Q&A |
| Depends on | — |
| Blocks | T04_training (its before/after memory numbers are measured through the monitor) |

## Goal

Make debugging part of every real training run, not something pasted in by hand. A small, tested `TrainMonitor` in `anyup/` records speed, GPU memory and validation time, and it catches bad losses and out-of-memory crashes with clear messages. The notebook only switches it on and calls it. It must cost no GPU memory and measurably little time (Muaaz, 2026-11-15; the `CLAUDE.md` rule "Debugging is a habit").

## Scope

- In scope:
  - New `anyup/data/training/monitor.py` (`TrainMonitor`) + `tests/test_train_monitor.py` (CPU).
  - `Train_3D_full.ipynb`: create the monitor (cell 12), call it at fixed points in cell 13, and add one config switch (cell 3).
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

## Q&A (decisions agreed before implementation)

| # | Question | Answer |
|---|---|---|
| 1 | Name and place: `TrainMonitor` in `anyup/data/training/monitor.py`, next to `teacher.py`? | _open_. Proposal: yes |
| 2 | On by default (`cfg.monitor = True`)? | _open_. Proposal: on; the overhead test (Q7) must pass first |
| 3 | What to record. Proposal: per step, wall time only (no GPU wait). Per logging point: mean s/step and it/s, peak allocated/reserved memory and % of the GPU, all to TensorBoard (`perf/…`, `memory/…`). Per validation: duration and clips. At each stage switch and at the end, a printed summary per stage, plus the estimated time left for the run. | _open_ |
| 4 | A loss becomes NaN or infinite: (a) stop with a clear error (which loss, step, T, batch size); (b) warn and skip the step. | _open_. Proposal: (a); a bad loss usually means everything after it is wasted |
| 5 | Out of memory: print the stage, T, batch size, current and peak memory and the GPU total, then re-raise the normal error (no automatic retry or batch-size change). | _open_. Proposal: yes |
| 6 | Where the checks run: at the existing logging points only (every 50 steps), or every step? Every step catches a NaN sooner but makes the CPU wait for the GPU each step. | _open_. Proposal: NaN check every step **only if** the overhead test shows < 1 %; otherwise at logging points |
| 7 | Overhead target and test: same short run with the monitor on vs off, median s/step difference < 1 %? | _open_. Proposal: yes |
| 8 | Order: do this before T04_training, so T04 measures through it? | _open_. Proposal: yes |

## Design (sketch; final per the answers)

`anyup/data/training/monitor.py`:

```python
class TrainMonitor:
    def __init__(self, writer=None, enabled=True, clock=time.time, cuda=torch.cuda): ...   # clock/cuda injectable for tests
    def step_start(self, step, T, batch_size): ...      # wall time; stage switch → reset peak + print the previous stage's summary
    def step_end(self, step, losses: dict): ...         # NaN/inf guard (Q4/Q6); at logging points: perf/ + memory/ scalars
    def val_start(self): ...
    def val_end(self, n_clips): ...                     # val duration + memory
    def oom_report(self, err): ...                      # Q5: context message, then the caller re-raises
    def summary(self) -> dict: ...                      # per-stage numbers (also printed at the end)
```

On a machine without CUDA, the memory parts switch themselves off (CPU dry runs keep working).

Cell 13, at fixed points (no wrapping of notebook functions):

```python
mon.step_start(step, current_stage.t, current_stage.batch_size)
try:
    ...                                             # existing training step, unchanged
except torch.cuda.OutOfMemoryError as e:
    mon.oom_report(e); raise
mon.step_end(step, {"total": total_loss, "rec": loss_rec, "inp": loss_inp, "aug": loss_aug, "temporal": loss_temp})
...
mon.val_start(); ...existing validation...; mon.val_end(len(val_loader))
```

## Steps

1. `monitor.py` + tests (fake clock, fake CUDA counters); run locally.
2. Notebook cells 3/12/13; clear outputs; CPU dry run (memory parts off).
3. Colab T4 short run, monitor on vs off (Q7). Check that it reproduces D-021/D-023's it/s and memory at T=2/4 and the validation time.
4. Update the docs that point to `scripts/debug/teacher_debug_cells.py` to say "use the monitor".

## Tests / Regression Checks

`tests/test_train_monitor.py`:
- With a fake clock: per-stage median and it/s are correct across a stage switch; the first step of each stage is reported separately.
- With fake CUDA counters: the peak is reset at stage switches; validation memory is measured separately from training.
- A NaN or infinite loss raises with the loss name, step and T in the message (Q4).
- `oom_report` text includes stage, T, batch size and memory numbers.
- `enabled=False` does nothing and writes nothing; no CUDA → no crash, and memory is reported as unavailable.
- Scalars go to the writer under the agreed tags.

Colab: overhead < 1 % (Q7). Numbers within noise of D-021/D-023 on the same setup.

## Done When

- Tests pass; the notebook uses the monitor at the agreed points; outputs are cleared.
- The Colab on/off run meets the overhead target, and the numbers match D-021/D-023.
- `docs/DECISIONS.md` entry; log and Task Index up to date.

## Implementation Log

_Updated live during implementation. Per step: what changed, files touched, test results (actual numbers/shapes). Also log mid-implementation questions and answers, changed decisions (old → new, why), and anything added or removed. Sections above are edited in place to match, so the finished file is a report of what was actually done._

- **2026-11-15: task file written** (D-025). Nothing implemented. Status `Q&A`: Q1–Q8 are open.
