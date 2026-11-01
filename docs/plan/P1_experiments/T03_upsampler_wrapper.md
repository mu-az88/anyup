# T03 — Unified Upsampler Wrapper

| Field | Value |
|---|---|
| Task ID | T03 |
| Plan | P1 Experiments (`plan.md`), Phase A.2 |
| Branch | `experiments-task02` (continued from T02, your choice) |
| Status | in progress |
| Depends on | T01 (`BackboneOutput`), T02 (chunk policy, `iter_backbone_chunks`) |
| Blocks | Notebook refactor (Exp 1–3 on shared backbones + upsamplers) |

## Goal

One entry point for every upsampler (AnyUp3D, AnyUp2D per-frame, bilinear), taking one T02 chunk and returning upsampled features, so every experiment can run any backbone × upsampler combination with identical loading and calling code.

## Scope

- In scope: registry + loader; AnyUp3D (safe checkpoint loading incl. the pickled `TStage`, strict keys, architecture args); AnyUp2D (published weights, per-frame loop); bilinear; the T02 AnyUp3D policy (one chunk = one call); output meta.
- Out of scope: other published upsamplers and extra baselines (Phase C, one class each); the canonical checkpoint decision (open question); metrics; fixing `train_3d.py` (P2).

## Q&A (decisions agreed before implementation)

| # | Question | Answer |
|---|---|---|
| 1 | Which upsamplers? | **AnyUp3D, AnyUp2D, bilinear.** Registry, so more are one class each later. |
| 2 | Input? | **A `BackboneOutput` chunk:** `up(out)` reads `features` + `guide` and copies `meta`. One call per T02 chunk (T02 Q13). |
| 3 | AnyUp3D checkpoint? | **Path required (no default), `strict=True`.** step6000 vs step6500 stays open. Path + step recorded in meta. |
| 4 | Pickled `TStage` in the checkpoint? | **`torch.load(weights_only=True)` with a stub class named `__main__.TStage` allowlisted via `torch.serialization.safe_globals`.** The file can't run code on load. If the checkpoint holds other non-tensor objects, loading fails with a message naming them; we then allowlist those or revisit. Notebooks no longer define `TStage`. _(Changed: was "custom unpickler with `weights_only=False`"; a security hook flagged `weights_only=False`, see Log.)_ |
| 5 | AnyUp2D weights? | **`anyup_paper.pth` from the AnyUp release URL** (as Exp 1/2), via the torch.hub cache. Path override allowed. |
| 6 | Default output size? | **Guide size** (as Exp 2/3); explicit `output_size` overrides. |
| 7 | AnyUp3D `q_chunk_size`? | **Kwarg, default `None`** (no chunking, as in all old experiments). |
| 8 | AnyUp2D per-frame: batch or loop? | **Loop, one frame per call** (as Exp 2). |
| 9 | AnyUp3D architecture args? | **Fixed training defaults, overridable** (`window_t=4, t_k=3, kernel_size_lfu=5, qk_dim=128, num_heads=4, window_ratio=0.1, kernel_size=1, input_dim=3`); all values recorded in meta. |
| 10 | Output device? | **Same device as the input.** The caller moves/saves. |
| 11 | Regression-test checkpoint? | **step6500 on Drive** (`/content/drive/MyDrive/results/anyup3d_step6500.pth`, as Exp 2/3). The test only checks new path = old path. |

## Design

### Files

- New: `anyup/upsamplers.py`.
- New: `tests/test_upsamplers.py` (pytest, CPU, random-init models, no downloads).
- New: `tests/T03_upsampler_check.ipynb` (Colab; real checkpoints, regression vs Exp 2/3 code).
- No change to `anyup/model.py`, `anyup/model_2d.py`, `anyup/backbones.py`, `anyup/video.py`.

### API

```python
from anyup.backbones import BackboneLoader
from anyup.video import iter_backbone_chunks
from anyup.upsamplers import UpsamplerLoader

bb = BackboneLoader.load("videomae", device="cuda")
up = UpsamplerLoader.load("anyup3d", ckpt_path=".../anyup3d_step6500.pth", device="cuda",
                          q_chunk_size=None)               # arch kwargs overridable, e.g. window_t=4
for chunk in iter_backbone_chunks(bb, src):
    res = up(chunk, output_size=None)                      # None → guide size
    res.features   # (t_out, C, H, W) on the input device
    res.meta       # chunk meta + upsampler keys (below)
UpsamplerLoader.available()   # -> ["anyup2d", "anyup3d", "bilinear"]
```

- Registry: `@register_upsampler("name")`, `UpsamplerLoader.load(name, **kw)` (mirrors `BackboneLoader`).
- Output: `UpsampleOutput(features, meta)` dataclass. `meta` = copy of the chunk's meta + `upsampler`, `output_size`, and upsampler-specific keys.
- All upsamplers run under `torch.no_grad()`, models frozen and in `eval()`.
- Inputs used as given: raw features (AnyUp normalizes internally) and the ImageNet-normalized guide from `BackboneOutput` (fixes Exp 3's raw [0,1] guide by construction).

### Upsamplers

| Name | Call | Shapes | Extra meta |
|---|---|---|---|
| `anyup3d` | `model(guide5d, feats5d, output_size=(H, W), q_chunk_size=q)`, guide first | features `(T, C, h, w)` → `(1, C, T, h, w)`; guide `(T, 3, gH, gW)` → `(1, 3, T, gH, gW)`; out `(1, C, T, H, W)` → `(T, C, H, W)` | `ckpt_path`, `ckpt_step` (`global_step` from the checkpoint if present), arch args, `q_chunk_size` |
| `anyup2d` | for each t: `model(guide[t:t+1], feats[t:t+1], output_size=(H, W))` | per frame `(1, C, h, w)` → `(1, C, H, W)`; stacked `(T, C, H, W)` | `weights` (URL or path) |
| `bilinear` | `F.interpolate(feats, size=(H, W), mode="bilinear", align_corners=False)` (as Exp 3); guide unused | `(T, C, h, w)` → `(T, C, H, W)` | — |

### Checkpoint loading (AnyUp3D)

```
class TStage: stub, __module__ = "__main__", fields t, start_step, batch_size (as in Exp 2/3)
with torch.serialization.safe_globals([TStage]):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)   # safe loader; no code runs from the file
sd = ckpt["anyup"]                         # KeyError listing the available keys if missing
model.load_state_dict(sd, strict=True)     # raises on any missing/unexpected key
```

### Compute-cost lines

- `q_chunk_size` (default None): AnyUp3D cross-attention query chunk → peak memory. Dependent: T02 `chunk_size` (AnyUp3D T per call) and `output_size`; reduce `q_chunk_size` when either grows.
- `output_size` (default guide size): output resolution → memory and time of every upsampler. Dependent: backbone `guide_size` (T01), `q_chunk_size`.

## Steps

1. Implement `anyup/upsamplers.py`: output dataclass, registry, safe checkpoint loader (`TStage` stub), three upsamplers.
2. pytest `tests/test_upsamplers.py` (CPU, random init); run on Colab (no torch locally), you paste the output.
3. Colab notebook `tests/T03_upsampler_check.ipynb` (clones `experiments-task02`, fetch + checkout); you run it and paste the summary.
4. Update `plan.md` and this file. Close out.

## Tests / Regression Checks

pytest (CPU, random-init models, tiny sizes):
- Shapes for all three: `(T, C, h, w)` + guide `(T, 3, gH, gW)` → `(T, C, gH, gW)` by default, explicit `output_size` respected; meta = chunk meta + upsampler keys.
- Bilinear equals `F.interpolate(..., align_corners=False)` exactly.
- AnyUp3D: `q_chunk_size` set vs None (relative error + cosine, as T02 Q22); one call per chunk (T dimension kept).
- AnyUp2D: loop output equals calling the model per frame by hand (exact).
- Checkpoint loading: a checkpoint saved with a `__main__.TStage` object loads with `weights_only=True`; a checkpoint with another non-allowlisted object fails; `"anyup"` key missing → clear error; extra/missing state-dict key → error (strict).

Colab (real weights, DAVIS `bear`, VideoMAE, 16 frames):
- **AnyUp3D vs Exp 2 `run_anyup3d` (exact):** same features and guides (Exp 2's PIL-resized guides), step6500 → max_abs ≈ 0. Load reports 0 missing / 0 unexpected.
- **AnyUp2D vs Exp 2 `run_anyup2d` (exact):** same inputs → max_abs ≈ 0.
- **Bilinear vs Exp 3 `upsample_bilinear` (exact):** Swin-T per-frame features at `PROP_SIZE=56`.
- **Full path (INFO):** `iter_backbone_chunks` + `up` on all 82 frames for each upsampler: shapes, frame indices, time and peak memory per chunk.
- **q_chunk_size:** AnyUp3D with e.g. 4096 vs None: relative error + cosine, and peak memory (INFO).

## Done When

- `anyup/upsamplers.py` exists; `UpsamplerLoader.load` builds all three; each takes a `BackboneOutput` chunk.
- AnyUp3D loads checkpoints strictly without notebook-side `TStage`.
- pytest passes; Colab: AnyUp3D and AnyUp2D match the Exp 2 code, bilinear matches the Exp 3 code.

## Implementation Log

_Updated live during implementation. Per step: what changed, files touched, test results (actual numbers/shapes). Also log mid-implementation questions and answers, changed decisions (old → new, why), and anything added or removed. Sections above are edited in place to match, so the finished file is a report of what was actually done._

- **Q&A change (before implementation):** Q4 changed from "custom unpickler with `weights_only=False`" → "`weights_only=True` + allowlisted `__main__.TStage` stub". Why: a security hook flagged that `weights_only=False` lets the file run arbitrary code on load; the safe loader is about as simple. Risk: the real checkpoint may contain other non-tensor objects; the Colab check will show it. Design and Tests updated.
- **OK received; implementation started.** Status Q&A → in progress.
- **Pre-check (Q4 risk):** `Train_3D_full.ipynb` `save_checkpoint` stores `anyup`, `optimizer` (AdamW), `scheduler` (CosineAnnealingLR), `global_step`, `data_epoch`, `cfg = vars(TrainConfig)`. `cfg` fields are str/int/float/bool/None plus `t_schedule: List[TStage]` (a `@dataclass`). So the only non-primitive global should be `__main__.TStage`; confirmed on the real file only in the Colab check (step 3).
- **Step 1 (code):** new `anyup/upsamplers.py`.
  - `ANYUP3D_ARCH` dict (training defaults, Q9) with compute-cost comments; `ANYUP2D_URL`.
  - `UpsampleOutput(features, meta)`; registry `register_upsampler` / `UpsamplerLoader.load` / `available()` (mirrors `BackboneLoader`).
  - `TStage` stub (`__module__ = "__main__"`) + `load_anyup3d_checkpoint(path)`: `torch.load(weights_only=True)` inside `safe_globals([TStage])`; `KeyError` listing the keys if `"anyup"` is missing.
  - Base `Upsampler(nn.Module)`: `forward(chunk, output_size=None)` under `no_grad`; checks shapes and equal T; moves inputs to `self.device`, output back to the input device (Q10); meta = copy of chunk meta + `upsampler`, `output_size`, extras.
  - `anyup3d`: `strict=True` load, frozen + eval; `(T,C,h,w)→(1,C,T,h,w)`, guide first, `q_chunk_size` passed through; meta `ckpt_path`, `ckpt_step` (= `global_step`), `q_chunk_size`, `arch`. Unknown arch kwargs → `TypeError` (added: catches typos such as `window=4`, which `AnyUp(**kwargs)` would silently swallow).
  - `anyup2d`: URL via `torch.hub.load_state_dict_from_url(..., weights_only=True)` or local `weights=` path; `strict=True` (Exp 2 default); one frame per call; meta `weights`.
  - `bilinear`: `F.interpolate(bilinear, align_corners=False)`; guide unused.
  - Added beyond the design: `device` kwarg on every upsampler (bilinear included, default `"cuda"`); `weights_only=True` also for AnyUp2D weights.
  - `py_compile` passes; not run (no torch locally).
- **Step 2 (tests):** new `tests/test_upsamplers.py`, 20 test functions (24 cases with parametrization). Tiny CPU sizes (T=4, C=16, 4×4 grid, 16×16 guide, AnyUp3D `qk_dim=32`). The TStage fixture pickles a dataclass under the real global name `__main__.TStage` and removes it from `__main__` before loading. Covers: registry; shapes/meta/device for all three (default and explicit `output_size`); meta is a copy; mismatched T raises; bilinear = `F.interpolate` (`torch.equal`); AnyUp3D = direct 5D model call (`torch.equal`); one call uses temporal context (changing frame 0 changes frame 1's output); `q_chunk_size=100` vs None (rel < 1e-4, cos > 0.99999); AnyUp2D = manual per-frame loop (`torch.equal`); models frozen; TStage loads as the stub with the right fields; non-allowlisted object → `UnpicklingError`; missing `"anyup"` → `KeyError` naming the keys; strict missing/unexpected key and arch size mismatch → `RuntimeError`; unknown arch kwarg → `TypeError`. `py_compile` passes; waiting for the Colab run.
- **pytest run 1 (Colab, Python 3.13.16, pytest 8.4.2): 24 passed, 8 errors, 12.20 s.** All 24 test bodies passed; the 8 errors are teardown-only, in every test using the `ckpt_with_tstage` fixture: `TypeError: cannot delete '__qualname__' attribute of immutable type '_TrainTStage'` (`monkeypatch` can't undo a class `__qualname__` change). Test-harness bug, not a wrapper bug. Fix: the fixture now builds a fresh dataclass per call (`_make_train_tstage`, `__module__="__main__"`, `__qualname__="TStage"` set directly); `monkeypatch` is only used for `__main__.TStage`. No change to `anyup/upsamplers.py`.
