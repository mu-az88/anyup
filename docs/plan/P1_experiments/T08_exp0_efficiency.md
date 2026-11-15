# T08 — Exp 0: Efficiency (renamed from Exp 4)

| Field | Value |
|---|---|
| Task ID | T08 |
| Plan | P1 Experiments (`plan.md`), Phase B wrap-up |
| Branch | `experiments-task02` |
| Status | done |
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

- Exp 0 runs end to end on T4 and A100 with every row `ok` or `OOM`, and the sanity checks pass. _(A100 skipped → S07. One run done; the bilinear memory check fails only on the deterministic row, explained in the log.)_
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
- **A100 run skipped by Monzir (no A100 available now) → `docs/skipped_and_future_concerns.md` S07.** Done When item "runs on T4 and A100" reduced to T4 for now.
- **Colab run 1 (pasted Cells 5 + 6): 51 rows (3 OOM) in 747 s.** Saved to Drive `results/T08/exp0/A100_main_s224_f8_q4096_sweep/` (`exp0_per_item.csv`, `exp0_summary.json`).
  - **GPU: Tesla T4** (`env.gpu` in `exp0_summary.json`, checked after Monzir copied the file to `docs/experiments/results/`). The run had `USE_T4=False`, so it used the A100 profile (5 warm-up / 20 repeats instead of 2 / 5) and the run name starts with `A100_`. The numbers are valid T4 numbers (more repeats only tightens the median); only the name is wrong.
  - Checks: params PASS; output shapes PASS (48 ok rows are (T, 768, S, S)); **bilinear memory FAIL, max gap 6.000 over 11 rows**. Every `fast` bilinear row matches its output tensor exactly (e.g. 224² × 8: 1176.0 MiB = 8·768·224²·4 B; 56² × 4: 36.8 MiB). The only outlier is the `deterministic` main row: 8232.0 MiB = 7 × 1176 MiB, and 173.25 ms vs 9.50 ms. So under `use_deterministic_algorithms`, bilinear `F.interpolate` takes a different, slower path with extra memory. This is probably T07's unexplained bilinear memory jump. Not checked in the PyTorch source (no local torch).
  - Main row (224², 8 frames, AnyUp3D `q_chunk_size=4096`):

  | mode | upsampler | median ms (IQR) | ms / frame | peak MiB | extra MiB |
  |---|---|---|---|---|---|
  | fast | bilinear | 9.50 (0.14) | 1.19 | 1194.4 | 1176.0 |
  | fast | anyup2d | 604.84 (5.21) | 75.61 | 2402.4 | 2352.0 |
  | fast | **anyup3d** | 1458.71 (19.65) | 182.34 | 3836.9 | 3786.5 |
  | deterministic | bilinear | 173.25 (0.45) | 21.66 | 8282.4 | 8232.0 |
  | deterministic | anyup2d | 668.14 (2.09) | 83.52 | 2402.4 | 2352.0 |
  | deterministic | **anyup3d** | 1518.94 (6.24) | 189.87 | 3836.9 | 3786.5 |

  - Ratios AnyUp3D / AnyUp2D: params +75.5 % (thesis +75.5 %); `fast` time ×2.41, extra memory ×1.61, total ×1.60; `deterministic` time ×2.27, memory ×1.61. **Thesis: about ×2.7 time, ×10.5 memory.** The time ratio is close; the memory ratio is not.
  - Thesis-memory hypothesis (not verified, the thesis setting is unknown, finding B): the thesis numbers fit a `q_chunk_size=None` run. 112² × 8 frames, q None: memory 5812.4 / 590.3 = ×9.85, time 409.22 / 158.99 = ×2.57. With query chunking (1024 or 4096) at the same size, memory is ×2.22.
  - Sweep (fast), AnyUp3D:
    - OOM at q None for 112² × 16, 224² × 8, 224² × 16. All chunked rows fit; largest 224² × 16, q 4096: 4804 ms, extra 8769 MiB.
    - q 1024 vs 4096: same memory up to small differences, q 1024 is 3–7 % slower. q None is about as fast as chunked where it fits, but uses up to 4.4× more memory (112² × 8: 5812 vs 1311 MiB).
    - **Time per frame grows with frames for AnyUp3D only**: 224², q 4096: 133.3 / 187.2 / 300.3 ms per frame at T = 4 / 8 / 16; 56²: 17.9 / 21.3 / 27.5. Bilinear and AnyUp2D are flat per frame (AnyUp2D 224²: 79.7 / 79.8 / 79.7). Likely cause: `CrossAttention3D` runs `nn.MultiheadAttention` over all T·h·w keys with a boolean window mask (`anyup/modules/cross_attention3d.py`), so masked-out keys are still computed and the cost is superlinear in T. Not profiled.
    - AnyUp2D per frame is flat in T and scales with S² (56²: 10.6, 112²: 19.9, 224²: 79.7 ms).
  - Deterministic vs fast on the main row: AnyUp2D +10.5 % time, AnyUp3D +4.1 %, memory unchanged; bilinear ×18 time, ×7 memory (above).
- **Close-out.** Status → done (`plan.md` Task Index updated). Done When: notebook runs end to end, every row `ok` / `OOM`; params and shape checks pass; the bilinear memory check fails only on the deterministic row (explained above, a property of PyTorch's deterministic mode, not of the notebook); results on Drive; rename done.
  - GPU checked afterwards: Tesla T4 (see above). Open choice for Monzir: keep the `A100_` run name, or re-run with `USE_T4=True` so the name is right. S07 (real A100 run) stays open.
  - Copied by Monzir into `docs/experiments/results/` for T09: `exp0_summary.json`, `exp0_per_item.csv`.
  - Left for Phase C (paper efficiency table): the AnyUp3D superlinear cost in T (masked full attention), the q None vs chunked memory trade-off, and the thesis ×10.5 claim (it should be replaced by the measured value with its setting stated).

