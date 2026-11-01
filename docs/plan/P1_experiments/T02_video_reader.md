# T02 — Video Reader + Chunked Backbone Runner

| Field | Value |
|---|---|
| Task ID | T02 |
| Plan | P1 Experiments (`plan.md`), Phase A.1/A.3 |
| Branch | `experiments-task02` |
| Status | in progress (step 4 fixes after Colab run 1; waiting for run 2) |
| Depends on | T01 (backbone adapters) |
| Blocks | T03 Upsampler wrapper, notebook refactor |

## Goal

A helper that takes any video (any resolution, any frame rate) and runs a backbone on **every** frame, with no frame skipping. Long videos are split into chunks that are loaded and processed one at a time, so memory stays constant.

## Scope

- In scope: reading a video (JPEG frame folder, video file, or in-memory tensor) into [0,1] frame chunks; splitting it into fixed-size chunks per backbone, with optional overlap (default 0); handling the last chunk; yielding one `BackboneOutput` per chunk with global frame indices and chunk metadata. Also fixing the AnyUp3D chunk policy (1 backbone chunk = 1 AnyUp3D call), which T03 implements.
- Out of scope: the backbone adapters themselves (T01); the upsampler call (T03); DAVIS annotation loading (shared DAVIS utilities task); fps resampling (we always use native frames).

## Q&A (decisions agreed before implementation)

| # | Question | Answer |
|---|---|---|
| 1 | Own task or part of T01? | **Own task (T02).** T01 adapters take one chunk `(T, 3, H, W)`; T02 handles reading and chunking. |
| 2 | Frame use relative to the old Exp 2 (16 linspace) / Exp 3 (every 12th) protocols? | **Always all frames.** No skipping anywhere. Exp 2/3 are redefined on all frames, so the thesis numbers can't be reproduced exactly (logged in `plan.md`). |
| 3 | VideoMAE last chunk when the length isn't a multiple of 16? | **Pad with the last frame** up to 16, then drop the outputs of the padded frames. |
| 4 | Return type for long videos? | **Iterator of chunks:** yields one `BackboneOutput` per chunk, with global frame indices in `meta`. |
| 5 | Input sources? | **JPEG frame folders + video files + in-memory tensors**, all behind one source interface. |
| 6 | Decoding library for video files? | **decord only** (lazy per-chunk access). _(Changed: was "decord, with torchvision.io fallback"; torchvision 0.26 removed `read_video`, see Q23 and Log.)_ |
| 7 | Aspect ratio? | **Squash to square** (as in Exp 2/3 and training). Done by the adapters' existing resize. Native `(H, W)` goes into `meta` so masks can be mapped back. |
| 8 | Chunk size for per-frame backbones (DINOv2, Swin per-frame, Swin clip)? | **Configurable kwarg, default 16.** VideoMAE is fixed at 16 (position embeddings). |
| 9 | Chunk overlap? | **Configurable `overlap` kwarg, default 0** (no overlap). Not fixed now; to be chosen by experiments (possibly as a hyperparameter / ablation). Chunk-boundary indices go in `meta` so Exp 2 can report the score with and without boundary pairs. _(Changed: was "no overlap", see Log.)_ |
| 10 | Videos shorter than one chunk? | **VideoMAE: raise an error when T < 16.** Per-frame backbones accept any T ≥ 1 as one short chunk (no padding). |
| 11 | How padding shows up in the output? | **Trimmed + meta counts.** Padded outputs are already dropped; `meta` adds `chunk_id`, global `frame_index`, `n_pad`, `is_last`, `n_frames_total`, `native_size`. |
| 12 | AnyUp3D temporal context across chunk boundaries: whose job? | **Decided in T02.** |
| 13 | AnyUp3D chunk policy? | **1 backbone chunk = 1 AnyUp3D call**, no context across boundaries. VideoMAE → 8 feature frames per call (matches training, `train_3d.py:218-219`). Boundaries logged in `meta`. T03 implements the call. |
| 14 | Per-frame backbones give AnyUp3D 16 feature frames per call (training used 8). OK? | **Keep 16.** More temporal context; out of training distribution and more memory (note for Phase B / ablations). |
| 15 | With overlap > 0, which outputs are kept for overlapped frames? | **Earlier chunk wins.** Each later chunk drops its first `overlap` input frames' outputs (= `overlap / temporal_stride` feature frames). The overlap only gives left context to the later chunk's backbone pass. |
| 16 | Does the overlap also apply to AnyUp3D? | **Yes, same chunks** (one `overlap` value; keeps the 1:1 policy of Q13). |
| 17 | Odd overlap with tubelet 2? | **Must be a multiple of `temporal_stride`**, else `ValueError` (VideoMAE, Swin clip: even only). Also `0 ≤ overlap < chunk_size`. |
| 18 | Where are the overlap outputs dropped? | **In T02, before AnyUp3D.** Only the backbone sees the context; AnyUp3D gets the trimmed chunk, so its calls have no context across boundaries, and with overlap > 0 non-first chunks give AnyUp3D `(chunk_size − overlap) / temporal_stride` feature frames (VideoMAE: fewer than training's 8). |
| 19 | How to run the pytest (no torch on the local machine)? | **On Colab.** You run `python -m pytest tests/test_video_reader.py -v` and paste the output. |
| 20 | DAVIS source for the Colab notebook? | **Always download from ETH** (`DAVIS_URL` in the notebook), not the Drive zip. |
| 21 | Video file for the decord vs torchvision check? | **Encode DAVIS `bear` (82 frames) to mp4 with ffmpeg** in the notebook (x264, crf 18, 24 fps). No UCF-101 needed. Codec loss vs the JPEGs is reported. |
| 22 | Pass criterion for the chunk-invariance checks (DINOv2 failed abs 1e-3 with cos 0.999999)? | **Relative error + cosine:** pass if `max_abs / max|feature| < 1e-4` (`REL_TOL`) **and** every frame's cosine > 0.99999; notebook prints `max|feature|`. Applies to all chunk-invariance checks. |
| 23 | Video-file fallback now that `torchvision.io.read_video` is gone? | **decord only.** No fallback; if `import decord` fails, `VideoFileSource` raises `ImportError("... pip install eva-decord")`. |
| 24 | Training reader `anyup/data/video_dataset.py` has the same dead fallback? | **Note it for P2** in `plan.md`. No training-code change in T02. |

## Design

### Files

- New: `anyup/video.py` (sources + chunk runner).
- New: `tests/test_video_reader.py` (local pytest; uses a fake adapter, no downloads).
- New: `tests/T02_video_reader_check.ipynb` (Colab; real backbones on DAVIS `bear` + an mp4 encoded from it).
- No change to `anyup/backbones.py`.

### API

```python
from anyup.backbones import BackboneLoader
from anyup.video import open_video, iter_backbone_chunks

src = open_video(".../JPEGImages/480p/bear")   # folder | video file path | (T,3,H,W) tensor
len(src), src.native_size, src.fps             # n frames, (H, W), fps or None for folders/tensors

bb = BackboneLoader.load("videomae", device="cuda")
for out in iter_backbone_chunks(bb, src, chunk_size=None, overlap=0):   # None → backbone default (16); overlap in input frames
    out.features   # (t_out, C, h, w)    padded outputs already dropped
    out.guide      # (t_out, 3, gH, gW)
    out.meta       # T01 keys + chunk keys (below)
```

### Sources (`anyup/video.py`)

Common interface: `__len__()`, `native_size -> (H, W)`, `fps -> float | None`, `get(indices: list[int]) -> (n, 3, H, W) float32 [0,1]` (CPU).

| Class | Input | Reading |
|---|---|---|
| `FrameFolderSource` | dir of `*.jpg` (configurable ext) | `sorted(glob)`, PIL `convert("RGB")` → tensor / 255. Same ordering as Exp 2/3. |
| `VideoFileSource` | mp4/avi/... | `decord.VideoReader(...).get_batch(indices)` per chunk. No fallback; without decord → `ImportError` naming `eva-decord`. |
| `TensorSource` | `(T, 3, H, W)` float [0,1] or uint8 | slicing; uint8 → /255. |

`open_video(x)` dispatches: tensor → `TensorSource`; dir → `FrameFolderSource`; file → `VideoFileSource`.

### Chunk runner

```
iter_backbone_chunks(bb, src, chunk_size=None, overlap=0):
  N = len(src)
  chunk_size = bb.NUM_FRAMES if VideoMAE (must equal 16 if given) else (chunk_size or 16)
  check 0 <= overlap < chunk_size and overlap % bb.temporal_stride == 0, else ValueError
  if VideoMAE and N < 16: raise ValueError
  step = chunk_size - overlap
  starts = [0, step, 2*step, ...] while the previous chunk's end < N   # no chunk made only of context frames
  for chunk_id, start in enumerate(starts):
      idx = list(range(start, min(start + chunk_size, N)))          # global indices, n_valid = len(idx)
      n_ctx = 0 if chunk_id == 0 else overlap                       # leading frames already output by the previous chunk
      n_pad = (16 - n_valid) if VideoMAE else 0
      frames = src.get(idx + [idx[-1]] * n_pad)                     # (n_valid + n_pad, 3, H, W)
      out = bb(frames)
      keep = [k for k, f in enumerate(out.meta["frame_index"])
              if n_ctx <= f < n_valid]                              # drop context (earlier chunk wins) and padded outputs
      out.features, out.guide = out.features[keep], out.guide[keep]
      meta += chunk_id, chunk_start=start, overlap, n_ctx, n_valid, n_pad, is_last,
              frame_index = [start + f for kept f]   (global),
              local_frame_index (original T01 value, kept),
              t_out = len(keep), n_frames_total=N, native_size, fps, source (path or "tensor")
      yield out
```

- VideoMAE padded tubelets: a tubelet is kept if its first frame is a real frame (e.g. 15 valid → tubelet (14, 15-pad) is kept; its guide is frame 14).
- Swin `clip` mode: odd last chunk is handled natively by the adapter (`ceil(T/2)`), no padding.
- Overlap: `n_ctx` is a multiple of `temporal_stride`, so context tubelets align with the previous chunk's tubelets and every frame is output exactly once (earlier chunk wins). The context only affects the later chunk's backbone pass; AnyUp3D (T03) sees the trimmed chunk (Q18).
- Chunk boundaries: each chunk's `meta["chunk_start"]` gives them; a helper `chunk_boundaries(metas) -> list[int]` returns the global feature-frame indices where a new chunk starts (for Exp 2 with/without boundary pairs).
- AnyUp3D (T03): called once per yielded chunk on `(features, guide)` exactly as yielded (context already dropped); no cross-chunk context for AnyUp3D.

### Compute-cost lines

- `chunk_size` (default 16) sets backbone batch size and the AnyUp3D T per call → memory. Dependent: VideoMAE must stay 16 (`VideoMAEAdapter.NUM_FRAMES`); T03 `q_chunk_size` must shrink if chunk_size grows; `overlap` must stay < chunk_size.
- `overlap` (default 0) sets backbone compute: about `chunk_size / (chunk_size − overlap)` × the no-overlap cost. Dependent: must be < `chunk_size` and a multiple of `temporal_stride`; reduces AnyUp3D T per non-first chunk.

## Steps

1. Implement sources + `open_video` in `anyup/video.py`.
2. Implement `iter_backbone_chunks` + `chunk_boundaries`.
3. Pytest `tests/test_video_reader.py` with a fake adapter (stride 1 and stride 2, fixed-T 16) on synthetic tensors; run on Colab (no torch locally), you paste the output.
4. Colab notebook `tests/T02_video_reader_check.ipynb` (clones `experiments-task02`, fetch + checkout). You run it and paste back the output; I log it.
5. Update `plan.md` Task Index and this file. Close out.

## Tests / Regression Checks

pytest (fake adapter, CPU; run on Colab, Q19):
- Every frame covered exactly once: concatenated global `frame_index` = `range(N)` (stride 1) or `range(0, …)` per chunk with stride 2, for N ∈ {1, 15, 16, 17, 33, 48}.
- VideoMAE-like fixed-16: N=33 → chunks of 16, 16, 1+15 pad; last chunk `t_out=1`, `n_pad=15`; N<16 raises.
- Overlap: for overlap ∈ {0, 2, 4, 8} (stride 1 and 2), every frame is output exactly once, outputs of frame f come from the earliest chunk containing it, `n_ctx` correct; odd overlap with stride 2 and overlap ≥ chunk_size raise.
- `TensorSource` / `FrameFolderSource` (JPEGs written to a tmp dir) give identical frames (PNG for lossless check, or JPEG vs PIL-decoded reference, diff = 0).
- `VideoFileSource` without decord raises `ImportError` naming `eva-decord`.

Colab (real backbones):
- **Chunked = direct (exact):** for a 16-frame DAVIS clip, `iter_backbone_chunks` output equals `bb(frames)` (max abs diff = 0) for all three backbones.
- **Per-frame chunk invariance:** DINOv2 / Swin per-frame, chunk 16 vs chunk 4 on the same clip: relative error `max|a−b| / max|a|` < 1e-4 and every frame's cosine > 0.99999 (Q22).
- **Overlap = 0 is the default path:** `overlap=0` output equals the no-overlap output exactly. With VideoMAE overlap 4 on a full clip: shapes, global indices, and cosine to the overlap-0 features reported (expected < 1, not a pass/fail check).
- **Folder reader vs Exp 2/3 loader:** same JPEG list and same [0,1] pixels as the old `sorted(glob("*.jpg"))` + PIL path (diff = 0).
- **Chunked = direct** covers `videomae`, `swin_t` per_frame, `swin_t` clip, `dinov2` (`torch.equal`).
- **Per-frame chunk invariance** also compares chunk 16 against one call on all 82 frames.
- **Video file:** DAVIS `bear` encoded to mp4 (Q21). decord reader: frame count, size, fps; mp4 vs JPEG codec loss reported (INFO); hiding `decord` (`sys.modules["decord"] = None`) gives an `ImportError` naming `eva-decord` (Q23); runner on the mp4 with DINOv2 (shapes, `fps`, `source` in meta).
- Full DAVIS clip (`bear`, 82 frames) with VideoMAE: chunk starts 0/16/…/80, last chunk n_valid 2 / n_pad 14 / t_out 1, global indices = `range(0, 82, 2)`, peak GPU memory per chunk within 10% (`MEM_TOL`), explicit `overlap=0` = default (`torch.equal`), overlap 4 indices + per-frame cosine to overlap 0 (INFO).

## Done When

- `anyup/video.py` exists; `open_video` handles folders, video files and tensors.
- `iter_backbone_chunks` processes every frame once (for any valid `overlap`), pads/drops VideoMAE's last chunk, drops overlap context (earlier chunk wins), and puts global indices + chunk meta on every output.
- Local pytest passes; Colab check passes (chunked = direct exactly, reader = old loader exactly).

## Implementation Log

_Updated live during implementation. Per step: what changed, files touched, test results (actual numbers/shapes). Also log mid-implementation questions and answers, changed decisions (old → new, why), and anything added or removed. Sections above are edited in place to match, so the finished file is a report of what was actually done._

- **Q&A change (before implementation):** Q9 changed from "no overlap" → "configurable `overlap` kwarg, default 0". Why: the overlap should be decided by experiments, not fixed now (possibly a hyperparameter). Added Q15–Q18 (merge rule, AnyUp3D, stride constraint, drop point); Scope, Design, compute-cost lines, Tests and Done When updated to match.
- **Steps 1–2 (code):** new `anyup/video.py`.
  - Sources: `VideoSource` base; `TensorSource` (float [0,1] or uint8 → /255); `FrameFolderSource` (`sorted(glob(*ext))`, PIL `.convert("RGB")` → `pil_to_tensor` / 255, same as Exp 2/3); `VideoFileSource` (decord `get_batch` per chunk, `fps = get_avg_fps()`; on any decord error, warns and falls back to `torchvision.io.read_video(output_format="TCHW")`, whole file in memory). `open_video` dispatches tensor / dir / file (warns on unknown extension).
  - Runner: `iter_backbone_chunks(bb, src, chunk_size=None, overlap=0)`. Fixed-T backbones detected by `bb.NUM_FRAMES` (VideoMAE), no backbone-name checks. Validation errors: fixed-T with other chunk_size, fixed-T with N < 16, `chunk_size < 1`, overlap outside `[0, chunk_size)`, overlap not a multiple of `temporal_stride`. Kept outputs: `n_ctx ≤ local frame_index < n_valid`. Yields a new `BackboneOutput`; T01 meta kept, `frame_index` replaced by global indices, `t_out` updated, added `local_frame_index, chunk_id, chunk_start, chunk_size, overlap, n_ctx, n_valid, n_pad, is_last, n_chunks, n_frames_total, native_size, fps, source`.
  - Helpers: `chunk_starts(n, chunk_size, overlap)`; `chunk_boundaries(metas)` → positions in the concatenated output where a new chunk starts (excluding 0).
  - Added beyond the design: `n_chunks`, `chunk_size`, `source` meta keys; `chunk_starts` exposed for testing; `DEFAULT_CHUNK_SIZE = 16` module constant (with compute-cost comment).
  - Not tested locally: no torch on this machine. `python3 -m py_compile` passes for both files.
- **Step 3 (tests):** new `tests/test_video_reader.py`, fake adapters (`FakeAdapter` stride 1/2, `FakeFixed16` VideoMAE-like). Synthetic frame k = k/1000 everywhere, so each output's feature value identifies its source frame. Covers: coverage for N ∈ {1, 15, 16, 17, 33, 48} × stride {1, 2} × overlap {0, 2, 4, 8}; fixed-16 coverage for N ∈ {16, 17, 33, 48, 82}; earlier-chunk-wins (N=20, chunk 8, overlap 4 → starts 0/4/8/12); overlap 0 = plain chunking; padding (N=33 → n_pad 15, t_out 8/8/1; N=31 → tubelet (30, pad) kept); all error paths; default chunk size and short video; `chunk_boundaries` (N=33 → [8, 16]); meta values; sources (uint8/float tensor, PNG folder = tensor with diff 0, sorted order, open_video errors). Video-file decoding is not in pytest (needs decord + a real file); it's in the Colab notebook (step 4). **Result (Colab, Python 3.13.16, pytest 8.4.2, branch `experiments-task02`): 84 passed, 0 failed, 3.87 s.** (48 variable-T coverage + 20 fixed-16 coverage + 16 other.)
- **Step 4 (Colab notebook):** new `tests/T02_video_reader_check.ipynb` (16 cells, generated by a script; every code cell compiles locally, not run). Clones `experiments-task02` (fetch + `checkout -B` for an existing clone), installs `eva-decord` (as `requirements.txt`; on failure the fallback path is used and the "decord backend used" check fails visibly). DAVIS from ETH (Q20), clip `bear`. Checks as in Tests (Colab) above; Q20/Q21 added; UCF-101 replaced by the DAVIS mp4 (Q21). Config constants in Cell 2 (`CHUNK=16`, `SMALL_CHUNK=4`, `OVERLAP=4`, `VIDEO_FPS=24`, `VIDEO_CRF=18`, `EXACT_MAX_ABS=1e-3`, `EXACT_MIN_COS=0.99999`, `MEM_TOL=0.10`). **Result (Colab run 1, Tesla T4, torch 2.11.0+cu130, torchvision 0.26.0, transformers 5.21.0, timm 1.0.30, decord 0.6.0): 17 checks, 2 failed.**
  - PASS: folder reader = Exp 2/3 loader (82 frames, (480, 854), max_abs 0); chunked = direct for videomae (8, 768, 14, 14), swin_t per_frame (16, 768, 7, 7), swin_t clip (8, 768, 7, 7), dinov2 (16, 768, 16, 16), all max_abs 0; swin_t per-frame chunk invariance (16 vs 4: max_abs 7.56e-05, min_cos 1.000000; 16 vs whole: 1.93e-05); VideoMAE full clip (41 feature frames, 6 chunks, last n_valid 2 / n_pad 14, boundaries [8, 16, 24, 32, 40]); peak memory per chunk [488, 497, 497, 497, 497, 497] MiB (spread 1.9%); overlap=0 = default (max_abs 0); overlap 4 indices (7 chunks, starts 0/12/…/72, t_out [8, 6, 6, 6, 6, 6, 3]); decord backend, 82 frames, (480, 854), 24.0 fps; runner on mp4 (82, 768, 16, 16).
  - INFO: VideoMAE overlap 4 vs 0 per-frame cosine mean 0.9643, min 0.9165 (first chunk identical) — VideoMAE features depend noticeably on the chunk's temporal context. mp4 (crf 18) vs JPEG: max_abs 0.2314, mean_abs 0.01174.
  - **FAIL 1, dinov2 chunk invariance:** 16 vs 4 max_abs 1.16e-02 (tol 1e-3) but min_cos 0.999999; 16 vs whole max_abs 6.57e-04. Indices correct. Looks like batch-size-dependent GPU kernels on large-magnitude raw features (abs tolerance too strict), not a chunking bug. Open, see question.
  - **FAIL 2, torchvision fallback:** `ImportError: cannot import name 'read_video' from 'torchvision.io'` — torchvision 0.26 removed `read_video`, so the Q6 fallback is dead code on current Colab. The same fallback exists in the training reader `anyup/data/video_dataset.py:40` (P2 issue). Open, see question.
- **Fixes after Colab run 1** (Q22–Q24 answered):
  - Q22 (FAIL 1): notebook `tests/T02_video_reader_check.ipynb` Cell 2 `EXACT_MAX_ABS = 1e-3` → `REL_TOL = 1e-4` + new `compare_rel` (returns max_abs, max|a|, relative error, worst-frame cosine); Cell 6 passes on `rel < REL_TOL and min_cos > EXACT_MIN_COS` for both 16 vs 4 and 16 vs whole, and prints `max|f|`. Old absolute criterion → relative + cosine: absolute error depends on the feature scale, which differs between backbones. No code change in `anyup/video.py` for this.
  - Q6/Q23 (FAIL 2): `anyup/video.py` `VideoFileSource` → decord only: torchvision branch, `_video` and the warning removed; `backend = "decord"` is now a class attribute; missing decord → `ImportError("VideoFileSource needs decord: pip install eva-decord")`. Module docstring no longer mentions the fallback. Design table and Q6 updated (old → new logged in Q6).
  - Notebook Cell 8: fallback check and decord-vs-torchvision pixel diff removed; added "no decord → clear ImportError" check. Patched with a script (the generator script from step 4 was lost with the scratchpad); every code cell compiles, not run.
  - pytest: added `test_video_file_without_decord_raises` (monkeypatches `sys.modules["decord"] = None`). Compiles; not run (no torch locally).
  - Q24: training-reader fallback logged in `plan.md` as a P2 issue.
  - **Result: pending; you rerun pytest and the notebook on Colab.**
