# T08 — Exp 0: Efficiency (renamed from Exp 4)

| Field | Value |
|---|---|
| Task ID | T08 |
| Plan | P1 Experiments (`plan.md`), Phase B wrap-up |
| Branch | `experiments-task02` |
| Status | in progress |
| Depends on | T03 (upsamplers), T07 (seeding, env/hash in results) |
| Blocks | T09 (interactive HTML documentation of Exp 0–3) |

## Goal

Build and run Exp 0, which measures the cost of each upsampler (parameters, size, run time, GPU memory), on the shared modules, and rename "Exp 4" to "Exp 0" in the docs. The repo has no Exp 4 notebook, so this is a new notebook, not a port.

## Scope

- In scope:
  - New `experiments/exp0_efficiency.ipynb`: params + size, timing and peak memory for bilinear / AnyUp2D / AnyUp3D, main row + sweep, results to Drive.
  - Rename Exp 4 → Exp 0 in `plan.md` and `project_context.md` (Q7).
- Out of scope: training cost (forward only, under `no_grad` as in the experiments); backbone cost (Q3); the checkpoint choice (S06, step6500 used); the HTML documentation (T09); `results/*.ipynb`.

## Findings

- **A.** No Exp 4 notebook in the repo. The only source is `results/VideoMAE features visualization exp/6000 steps of training/Ox/6000 steps 14 to 224.ipynb` Cell 5a, which prints `AnyUp3D parameters: 1,540,736` (1,540,736 × 4 B = 5.88 MiB, which matches the thesis "5.88 MB").
- **B.** The thesis claims "about 2.7× slower" and "about 10.5× more memory" vs AnyUp2D have no source in the repo: settings (GPU, size, frames, `q_chunk_size`) are unknown. Exp 0 measures them fresh; the thesis values are shown next to the new ones only for reference.

## Q&A (decisions agreed before implementation)

| # | Question | Answer |
|---|---|---|
| 1 | Split? | **T08 = Exp 0; T09 = HTML documentation.** |
| 2 | Setting? | **Exp 2 setting as the main row + a sweep.** |
| 3 | What is timed? | **Upsampler only, synthetic inputs** of the right shapes (no dataset, no backbone). |
| 4 | Method? | **CUDA events, warm-up then N timed runs, median + IQR**; params / MB from the model; A100 / T4 via `USE_T4` (per-GPU dict, as Exp 1). |
| 5 | Memory? | **Both: total peak and extra peak** (peak − memory in use just before the call); ratios on the extra peak. |
| 6 | Ratios? | **vs AnyUp2D only** (as the thesis), plus the thesis values for reference. |
| 7 | Rename? | **All docs**: `plan.md`, `project_context.md`, the new notebook and its results folder. T01–T07 task files stay as written; `plan.md` notes that "Exp 4" there means Exp 0. |
| 8 | Frames sweep? | **Yes, feature frames {4, 8, 16}.** |

### Confirmed with this file

| # | Point | Proposal |
|---|---|---|
| 9 | Seeding mode for timing | **Main row in both modes, sweep with `deterministic=False`.** `False` = normal inference cost (deterministic kernels slowed Exp 1 by +5 to +60 %, T07); `True` = the cost inside Exp 1–3 runs, and a test of T07's unexplained bilinear memory jump. Seeds are fixed in both modes (same inputs). Each row records its `mode`. |
| 10 | Branch | `experiments-task02` (as T05–T07). |
| 11 | Sweep grid | Output size {56, 112, 224} × feature frames {4, 8, 16} × AnyUp3D `q_chunk_size` {None, 1024, 4096} = 27 AnyUp3D rows; bilinear and AnyUp2D 9 rows each (`q_chunk_size` doesn't apply to them). An OOM is caught and recorded as `status="OOM"` (expected for `None` at 224², T03), so the sweep always finishes. |
| 12 | Where the measuring code lives | In the notebook (one `measure(fn, warmup, repeats)` helper, about 25 lines), not a new module; nothing else needs it. |

## Design

### Inputs

- Synthetic, fixed seed: features `(T, 768, 14, 14)` (VideoMAE grid and width), guide `(T, 3, S, S)` ImageNet-normalized random values, with S = output size (guide = output size, as Exp 2). T = feature frames.
- Real weights: AnyUp3D `anyup3d_step6500.pth` (S06), AnyUp2D `anyup_paper.pth`, via `UpsamplerLoader` (so the hashes go into the meta).

### Measurement (per row)

```python
def measure(fn, warmup, repeats):    # fn: one upsampler call on GPU inputs
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    times = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(); out = fn(); end.record(); torch.cuda.synchronize()
        times.append(start.elapsed_time(end)); del out
    peak = torch.cuda.max_memory_allocated()
    return times, base, peak         # → median / IQR ms, per-frame ms, total peak, extra peak = peak − base
```

- Rows: `part (main / sweep), mode (fast / deterministic), upsampler, out_size, frames, q_chunk, status, median_ms, iqr_ms, per_frame_ms, peak_total_mib, peak_extra_mib, repeats`.
- Model rows (summary): params, size MiB (params × 4 B), for each upsampler (bilinear 0).
- Summary: main row (224², 8 frames, AnyUp3D q 4096) ratios AnyUp3D / AnyUp2D for params, median time, per-frame time, extra peak memory; next to the thesis values (+75.5 %, 2.7×, 10.5×).
- `save_results(OUT_DIR, rows, summary, config, name="exp0")`; `RUN_NAME = f"{GPU}_main_s224_f8_q4096_sweep"`; `OUTPUT_ROOT = results/T08/exp0`.
- Optional cell (`RUN_VISUALS=False`): per-frame time vs output size, one line per (upsampler, frames).

### GPU profiles (as Exp 1)

```python
GPU_PROFILES = {
    "A100": dict(WARMUP=5, REPEATS=20, SWEEP=True),
    "T4":   dict(WARMUP=2, REPEATS=5,  SWEEP=True),
}
```

- Each value on its own line with `# A100: x, T4: y` and dependents. Sizes, frames and q values are the same on both GPUs (the point is comparable rows; OOMs are recorded).

### Notebook layout

1. CUBLAS env, clone `experiments-task02`, Drive, deps, restart guard.
2. Config (`USE_T4`, profiles, `MAIN`, sweep lists, paths, `SEED`).
3. Imports, `set_seed`, upsamplers, params table.
4. `measure` + `run_row` (OOM handling: `torch.cuda.OutOfMemoryError` → row with `status="OOM"`, `empty_cache`).
5. Main row for the three upsamplers, then the sweep.
6. Summary table (main row ratios vs AnyUp2D, thesis column), `save_results`.
7. Optional plot.

### Compute-cost lines

- `OUT_SIZES`, `FRAMES`: guide/output size and feature frames per call → memory/time of every upsampler (AnyUp3D attention grows with frames × S²). Dependent: `Q_CHUNKS` (None OOMs at large values).
- `Q_CHUNKS`: AnyUp3D query chunk → peak memory. Dependent: `OUT_SIZES`, `FRAMES`.
- `REPEATS`, `WARMUP`: run time only, no effect on memory.

## Steps

1. Notebook `experiments/exp0_efficiency.ipynb`; CPU dry run of the logic (fake CUDA timing).
2. Rename Exp 4 → Exp 0 in `plan.md` and `project_context.md`.
3. You run on T4 (`USE_T4=True`), then A100; paste the summary and the sweep table.
4. Update `plan.md`, this file; close out.

## Tests / Regression Checks

- No pytest: no module code changes.
- Sanity checks in the notebook (printed PASS / FAIL): AnyUp3D params = 1,540,736 (thesis value, finding A); output shapes `(T, 768, S, S)` for every non-OOM row; bilinear extra memory ≈ its output tensor size (T × 768 × S² × 4 B) within 10 %.
- CPU dry run before Colab (scratchpad harness, CUDA calls stubbed): loop, OOM path, CSV / JSON written.

## Done When

- Exp 0 runs end to end on T4 and A100 with every row `ok` or `OOM`, and the sanity checks pass.
- Results (CSV + JSON) on Drive, numbers in the Implementation Log.
- Exp 4 → Exp 0 rename done in `plan.md` and `project_context.md`.

## Implementation Log

_Updated live during implementation. Per step: what changed, files touched, test results (actual numbers/shapes). Also log mid-implementation questions and answers, changed decisions (old → new, why), and anything added or removed. Sections above are edited in place to match, so the finished file is a report of what was actually done._

- **OK received ("I agree on everything you said, you have my trust").** Q9 decided as option (b): the main row in both seeding modes, the sweep in `fast` mode (old → new: the file proposed `deterministic=False` for everything; why: (b) also measures the cost inside Exp 1–3 runs and tests T07's unexplained memory jump). Q10–12 accepted. Status Q&A → in progress.
- **Step 1 (code): new `experiments/exp0_efficiency.ipynb`** (8 cells; generated by scratchpad `make_exp0_nb.py`; all code cells compile).
  - Cells: 1 CUBLAS env + clone + Drive + deps; 2 config (`USE_T4`, `GPU_PROFILES` WARMUP / REPEATS, `MAIN`, sweep lists, `THESIS`); 3 imports, `set_mode` (`fast` explicitly undoes the deterministic flags, since `set_seed(deterministic=False)` leaves them on), upsamplers, params table + check; 4 `make_inputs`, `measure`, `run_row` (OOM → `status="OOM"`, `gc` + `empty_cache`); 5 main row (fast) → sweep (fast, small → large) → main row (deterministic), shape + bilinear-memory checks; 6 ratios vs AnyUp2D per mode, main table, `save_results` (`exp0_per_item.csv`, `exp0_summary.json`); 7 optional plot.
  - AnyUp3D's `q_chunk_size` is set per row on the wrapper (`up.q_chunk_size`) and reset to the main value before the config meta is saved.
  - CPU dry run (scratchpad harness: CUDA events / memory stubbed, random-init weights, sizes 16 / 32, frames 1 / 2, a forced OOM for `q=None` at 32): 22 rows (2 OOM, recorded and skipped correctly), summary + CSV + JSON + plot written. Shape check PASS (20 ok rows). The bilinear-memory check FAILs there only because the stubbed memory counter is constant; it needs the real GPU. Timings on CPU are meaningless.
  - **Parameters (architecture only, so already final): AnyUp3D 1,540,736 (5.88 MiB), AnyUp2D 878,080 (3.35 MiB) → +75.5 %, equal to the thesis.** Check PASS.
- **Step 2 (docs): Exp 4 → Exp 0** in `plan.md` (thesis table row, naming note, open question answered, "Re-run Exp 0–3", Task Index) and `project_context.md` (experiment list, thesis results line, open question). T01–T07 left as written (Q7).
- **Waiting for Colab:** T4 run (`USE_T4=True`), then A100.

