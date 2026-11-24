# T04_training — Compute the Image Half Once per Training Step (T=8 Fits on a T4)

| Field | Value |
|---|---|
| Task ID | T04_training |
| Plan | P1_training Training pipeline (`plan.md`) |
| Branch | `fix/shared-image-encoding`, branched from `main` (`9e43ba2`: teacher fix D-026 and T05 monitor D-028 merged) |
| Status | done (D-031; not merged yet) |
| Depends on | Teacher 16-frame fix (merged, D-026): its T=8 debug check moved here (D-022, S14). T05_training monitor (merged, D-028), used to measure before/after |
| Blocks | run_03 on a T4 reaching the T=8 stage (`start_step=15000`) |

## Goal

Make the T=8 training stage fit in a T4's 14.6 GiB by computing AnyUp's image half once per step and reusing it for both student passes (main + self-consistency). Same layers, weights, checkpoints and outputs: **not an architecture change**, only how the step's computation is organised.

## Scope

- In scope:
  - `anyup/model.py`: split `AnyUp.forward` into an image half and a feature half (Design), so the image half can be computed once and reused.
  - `Train_3D_full.ipynb` cell 13: the main pass and the self-consistency pass share one image encoding.
  - `Train_3D_full.ipynb` cells 2 / 3, for the Step 4 run only (added in Step 4, as T05 did): `REPO_BRANCH` = this branch until the merge, and a commented debug-run block.
  - New `tests/test_shared_image_encoding.py`.
  - A T4 debug run measuring peak memory before vs after at T=2/4/8. This is also the teacher fix's deferred T=8 check (S14).
- Out of scope:
  - Changing what the model computes, its layers or sizes, or any loss.
  - Other memory measures (activation checkpointing, `q_chunk_size`, lower resolution, dropping self-consistency at T=8): only if this isn't enough, see Q3.
  - The validation loop: it has no gradients and one pass per clip, so nothing to share.

## Findings

- **A. Where it runs out.** D-021: the T4 debug run reaches 10.1–10.2 GiB reserved at T=2 (batch 2) and T=4 (batch 1), both 4 frames per step. At T=8 (batch 1, 8 frames) it runs out of memory on the first step, inside the self-consistency forward `pred_aug = anyup(crop_high_5d, gt_aug_low, …)` (`query_encoder` Conv3d), with 14.28 GiB allocated.
- **B. The two passes share the image.** Cell 13 calls `anyup(crop_high_5d, guidance_feats_low, out_size)` and then `anyup(crop_high_5d, gt_aug_low, out_size)`: same frames, same output size, and features of the same shape (only their values differ: clean vs noisy).
- **C. Which parts depend only on the image** (`anyup/model.py:82-123`):
  - `image_encoder(image)` → `rope` (coordinates from the encoded `t, h, w`);
  - `query_encoder(enc)` pooled to `(t, *out_size)`;
  - `key_encoder(enc)` pooled to `(t, h, w)`, where `t, h, w` is the feature grid, the same in both passes (B).
  
  The rest uses the features: `key_features_encoder(normalize(feats))`, `aggregation(cat(k_img, ·))`, and `cross_decode(q, k, v=feats)`.
- **D. So both passes compute the image half identically,** and autograd keeps both sets of its activations until `backward()`. The image half runs at full 224² for every frame with `qk_dim=128` channels, so it is likely the largest memory item. This is inferred and will be measured in Step 4.
- **E. Gradients are unchanged in math.** One tensor used in two places gets the sum of both gradients, exactly as two identical copies do. Only the order of the floating-point additions changes, so values can differ at rounding level.

## Q&A (decisions agreed before implementation)

| # | Question | Answer |
|---|---|---|
| 1 | API: (a) add `encode_image(image, feats_shape, output_size)` and `decode(encoded, feats, …)`, and rewrite `forward` as `decode(encode_image(…))`, so there's one code path, proven equal by the test; (b) add them but leave `forward`'s code untouched, as a second path that could drift. | **(a)**, one code path (Muaaz, 2026-11-17). |
| 2 | `anyup/model.py` is also used by Munther's experiments (through `forward`). OK to change it there, given `forward`'s outputs stay equal (test), or should he review first? | **Proceed and tell him** (Muaaz, 2026-11-17): the exact-equality test + his tests in Step 3 are the proof; Munther gets a short message after the merge (Step 5). |
| 3 | If T=8 still doesn't fit after this: decide the next measure now, or after the measurement? | **After measuring** (Muaaz, 2026-11-17). |
| 4 | Test tolerance: outputs of old vs new `forward` **exactly equal** (`torch.equal`); gradients equal within `rtol=1e-5, atol=1e-7` (Finding E). | **OK as stated** (Muaaz, 2026-11-17). **Amended in Step 2** (Muaaz, 2026-11-17: "Accept, fp32 is the check"): the tolerance applies in fp32 (pytest + dry run with autocast off). Under bf16 autocast, losses must be bit-equal; gradients differ by one bf16 rounding of the summed image-half gradient (measured ≤ 1.0e-2 per-tensor relative L2), accepted. |
| 5 | Safety margin: what peak reserved memory at T=8 counts as "fits"? The real run will also see larger or odd videos and other variation. | **≤ 13.5 GiB peak reserved at T=8** (about 1 GiB headroom; Muaaz, 2026-11-17). |

## Design

`anyup/model.py`, sketch (Q1 (a)):

```python
def encode_image(self, image, feats_shape, output_size):
    """Image half: depends only on the frames (+ feature grid / output size). Reusable across passes."""
    _, _, t, h, w = feats_shape
    enc = ...                                   # image_encoder + rope, as in forward today
    q     = F.adaptive_avg_pool3d(self.query_encoder(enc), output_size=(t, *output_size))
    k_img = F.adaptive_avg_pool3d(self.key_encoder(enc),   output_size=(t, h, w))
    return q, k_img

def decode(self, encoded, feats, vis_attn=False, q_chunk_size=None):
    """Feature half: everything that depends on the features."""
    q, k_img = encoded
    k = self.aggregation(torch.cat([k_img, self.key_features_encoder(F.normalize(feats, dim=1))], dim=1))
    # vis_attn: same warning + reset to False as today's upsample()
    return self.cross_decode(q, k, feats, store_attn=vis_attn, q_chunk_size=q_chunk_size)

def forward(self, image, features, output_size=None, vis_attn=False, q_chunk_size=None):
    output_size = output_size if output_size is not None else image.shape[-2:]
    return self.decode(self.encode_image(image, features.shape, output_size), features,
                       vis_attn=vis_attn, q_chunk_size=q_chunk_size)
```

- `upsample()` is removed: its only caller is `forward` (checked 2026-11-17: no other `.upsample(` call on the 3D model in `anyup/`, `tests/`, `experiments/` or the notebook; `model_2d.py` has its own). Its body moves into `encode_image` / `decode` unchanged, in the same order. Methods are not in the state dict, so checkpoints are unaffected.
- Call signature `model(guide, features, output_size=..., q_chunk_size=...)` unchanged, so `anyup/upsamplers.py` (Munther's Exp 0–3), `eval_linear_probe.py` and validation in cell 13 need no change.

Cell 13:

```python
out_size   = (TOKEN_H, TOKEN_W)                                                     # 14×14 (existing line)
encoded    = anyup.encode_image(crop_high_5d, guidance_feats_low.shape, out_size)   # image half once per step
pred_feats = anyup.decode(encoded, guidance_feats_low)
...
if cfg.lambda_self_consistency > 0:                                                 # existing condition
    ...
    pred_aug = anyup.decode(encoded, gt_aug_low)                                    # reuses it (Finding B)
```

- `gt_aug_low` has the same shape as `guidance_feats_low` (same `tok0:tok0 + n_tok` slice, same `F.interpolate` size), which is what makes the reuse valid; an `assert` on the shapes guards it (no GPU cost).
- Validation (`val_pred = anyup(...)`) keeps calling `forward`: one pass, no gradients, nothing to share.

## Steps

0. Muaaz creates `fix/shared-image-encoding` from `main` (`9e43ba2`).
1. `tests/test_shared_image_encoding.py` against the **current** `forward` first (record the reference outputs/gradients), then the `model.py` change; run locally.
2. Cell 13 change; clear outputs. Local CPU dry run of one step with a tiny teacher, like D-006's.
3. Run Munther's and the existing model tests: `test_upsamplers.py`, `test_weight_loading.py`, `test_forward3d.py`, `test_teacher_16frame.py`, `test_losses.py`, `test_train_monitor.py`.
4. Colab T4 debug run through the T05_training monitor (`TrainMonitor`, D-028; the old `scripts/debug/teacher_debug_cells.py` is superseded). Peak memory at T=2/4/8 vs D-021 (the monitor's T4 run 1 reproduced D-021's T=2/4 memory within 0.04 GiB, so the two are comparable); finite losses at all three. **Before the run:** the run settings + a memory estimate and OOM likelihood go to Muaaz for an OK (`CLAUDE.md` "Warn before OOM risk"; T=8 reaching OOM is the point of the run).
5. After the merge: a short message to Munther (Q2): `model.py`'s `forward` now goes through `encode_image` / `decode`, outputs proven equal, his tests pass.

## Tests / Regression Checks

`tests/test_shared_image_encoding.py` (CPU, small random `AnyUp`, fixed seed):
- New `forward` == old `forward` (kept in the test as a reference copy), `torch.equal`, for T_tok = 1, 2, 4.
- `decode(encode_image(x), f)` == `forward(x, f)`.
- **Two passes, shared vs separate:** loss = L(pass1) + L(pass2). Parameter gradients with a shared `encoded` equal those from two separate `forward` calls (Q4 tolerance).
- `vis_attn` / `q_chunk_size` keep their current behaviour.

Debug run: D-021's numbers as the "before"; the "after" must show T=2/4 peak no higher, T=8 finishing its steps with finite losses within Q5's margin.

## Done When

- All tests pass (new + the ones in Step 3); `forward` outputs are unchanged.
- T4 debug run: T=8 steps finish with finite losses; peak reserved at T=8 is within the Q5 margin; before/after numbers are recorded here.
- S14 closed (teacher fix's T=8 check). `docs/DECISIONS.md` entry; log and Task Index up to date.
- Munther informed (message drafted for Muaaz, Q2).

## Implementation Log

_Updated live during implementation. Per step: what changed, files touched, test results (actual numbers/shapes). Also log mid-implementation questions and answers, changed decisions (old → new, why), and anything added or removed. Sections above are edited in place to match, so the finished file is a report of what was actually done._

- **2026-11-15: task file written** (D-022), from D-021's numbers and reading `anyup/model.py` / cell 13. Nothing implemented or measured yet. Status `Q&A`: Q1–Q5 are open.
- **2026-11-17: Q&A round 1.** Q3 → after measuring; Q5 → ≤ 13.5 GiB peak reserved (both as proposed). Muaaz asked for simple explanations of Q1, Q2, Q4; still open.
- **2026-11-17: Q&A round 2.** Q1 → (a); Q2 → proceed and tell Munther; Q4 → OK as stated. All questions answered. Checked against the code before finalising (Design updated in place):
  - `upsample()` has no caller besides `forward` → removed, body split into `encode_image` / `decode`; `vis_attn` passed through to keep its warning.
  - Cell 13: `out_size = (TOKEN_H, TOKEN_W)` (14×14); the self-consistency pass sits under `if cfg.lambda_self_consistency > 0`; validation keeps `forward`.
  - Added: Step 0 (branch), `test_forward3d.py` + `test_train_monitor.py` in Step 3, the OOM estimate before Step 4, Step 5 (message to Munther).
  - Side finding, not in T04's scope: notebook cell 27 (dummy sanity check) calls `anyup(dummy_low_5d, dummy_high_5d, …)`, i.e. low-res **pixels** as the guide and high-res **pixels** as the features; `out` would have 3 channels and the `assert out.shape == dummy_gt.shape` (768) would fail if the cell runs.
  - Task file waiting for Muaaz's OK.
- **2026-11-17: task file approved ("go ahead with step 0"); Step 0 done.** Branch `fix/shared-image-encoding` created from `main` (`9e43ba2`); the uncommitted task-file edits moved with it. Status Q&A → in progress. Cell 27 side finding → **S16** (Muaaz: "log cell 27 as a skip"; not fixed in T04).
- **2026-11-17: Step 1 done.** `anyup/model.py`: `upsample()` removed; its body plus the image-encoder/rope part of `forward` moved, unchanged and in the same order, into `encode_image(image, feats_shape, output_size=None)` → `(q, k_img)` and `decode(encoded, feats, vis_attn, q_chunk_size)`. `forward` = `decode(encode_image(…))`, same signature. Added (not in the Design sketch): `decode` raises a `ValueError` naming `encode_image` when the features' (B, t, h, w) differ from the grid it was encoded for (shape compare only, no GPU cost).
  - New `tests/test_shared_image_encoding.py`: a verbatim copy of the old `forward` + `upsample` (main `9e43ba2`) as the reference; small `AnyUp` (qk_dim 32, 4 heads, `kernel_size_lfu=5`, `t_k=3`, `window_t=3`), 24-ch features, 32² images, 4² features, 8² output. Tests: new `forward` == old, `torch.equal`, for (T, T_tok) = (1,1), (2,1), (4,2), (8,4); batch 2 + default output size + `q_chunk_size=7`; `vis_attn` warns, same output; `decode(encode_image)` == `forward` (4 cases); mismatched features rejected; two-pass gradients shared vs separate at (4,2), (8,4).
  - **Local run (Python 3.12.10, torch 2.11.0+cpu): 13 passed, 6.03 s.**
  - Gradients, measured separately (scratchpad script, same setup as the test): 71 gradient tensors; 40 not bit-equal; max_abs 1.25e-9 (T=4) / 7.97e-10 (T=8); max relative to the tensor's max 6.8e-7 / 1.0e-6. Rounding level, as Finding E expects; outputs and losses bit-equal.
- **2026-11-17: Step 2 (code + CPU dry run).** `Train_3D_full.ipynb` cell 13 (notebook cell index 25): `pred_feats = anyup(...)` → `encoded = anyup.encode_image(crop_high_5d, guidance_feats_low.shape, out_size)` + `pred_feats = anyup.decode(encoded, guidance_feats_low)`; in the self-consistency branch, `assert gt_aug_low.shape == guidance_feats_low.shape` (message names cell 13) + `pred_aug = anyup.decode(encoded, gt_aug_low)`. +6 / −2 lines, applied as exact one-occurrence replacements on the raw JSON (CRLF kept); valid JSON, cell compiles, outputs still cleared. Checked first: `anyup` is a plain `AnyUp` (no DDP / `torch.compile`), and the model has no dropout or other randomness in the forward (`dropout=0.0`; `randn` only at init), so one shared encoding is valid. Validation keeps `forward`.
  - Dry run (scratchpad `dryrun_cell13.py`, not in the repo): executes the notebook's own step lines (`video = batch[...]` … `scheduler.step()`) of cell 13 from `git HEAD` (old) and from the working tree (new), with cells 3 / 8 / 12 exec'd from the same version; real `extract_teacher_features` with a fake VideoMAE-shaped teacher (fixed-seed Conv3d, 16 frames → (B, 1568, 768)); small AnyUp (qk_dim 32, as cell 7 otherwise); same init and seeds; step 6000 so every loss term is active; `GradScaler` disabled (CPU).
  - Shapes: q (B, 32, T_tok, 14, 14), k_img (B, 32, T_tok, 7, 7) at T=2 (B 2) / 4 / 8 (T_tok 1 / 2 / 4).
  - **Losses bit-equal old vs new at T=2/4/8, all finite; `grad_norm` equal to 7 digits.** CPU time per step old → new: 11.2 → 6.0 s (T=2), 9.7 → 5.2 s (T=4), 15.9 → 9.1 s (T=8): the image half now runs once (CPU timing only, not a GPU estimate).
  - **Unexpected: under the notebook's bf16 autocast the gradients differ at bf16 rounding level, not fp32.** 40/73 gradient tensors (the image-half parameters) are not bit-equal; per-tensor relative L2 difference max 7.5e-3 / 6.0e-3 / 1.0e-2 (median 2.6e-3 / 1.2e-3 / 1.0e-3) at T=2/4/8; max_abs ≤ 8.4e-8. After one AdamW step the weights differ by up to 1.1e-4 (AdamW's first step ≈ lr·sign(g), lr 2e-4, amplifies tiny differences on near-zero gradients).
  - Same dry run with autocast off (fp32): per-tensor relative L2 max 8.4e-6 / 1.4e-5 / 1.9e-5, max_abs ≤ 3.5e-8 (inside Q4's atol 1e-7), weights after the step ≤ 4.3e-6. So the bf16 gap is rounding, not a logic difference: with sharing, the two passes' gradients w.r.t. `q` / `k_img` are added in bf16 before going back through the image encoder (old: each pass went back separately and the sums were done in fp32 in `.grad`). Bit-identical gradients would need two backward passes through the encoder, i.e. exactly the memory T04 removes.
  - Side observation, not T04 (old and new agree): on CPU, the bf16-autocast `grad_norm` is 157 / 609 / 182 vs 0.15 / 0.24 / 0.15 with autocast off, same model and inputs. With `grad_clip_max_norm = 1.0` this would clip almost every bf16 step. CPU autocast differs from CUDA autocast, so this says nothing yet about the Colab runs.
  - **Asked Muaaz:** Q4's gradient tolerance (rtol 1e-5, atol 1e-7) holds in fp32 (pytest + this dry run) but cannot hold under bf16 autocast. **Answer: accept, fp32 is the check** (Q4 amended in place: old "gradients within rtol 1e-5 / atol 1e-7" → that tolerance in fp32; under bf16, losses bit-equal and the bf16-rounding gap accepted). The `grad_norm` side observation → **S17** (Muaaz: "log as S17"). Step 2 done.
- **2026-11-17: Step 3 done. Local CPU, 121 passed, 0 failed (143 s):** `test_shared_image_encoding` 13, `test_upsamplers` 34, `test_weight_loading` 7, `test_forward3d` 10, `test_teacher_16frame` 20, `test_losses` 19, `test_train_monitor` 18.
- **2026-11-17: Step 4 memory estimate (CPU measurement, scratchpad `saved_bytes.py`; no GPU used).** Bytes autograd keeps for backward per pass, real model config (qk_dim 128, 4 heads, window_ratio 0.1, lfu 5, t_k 3), batch 1, 224² guide, 7² → 14², unique storages:

  | | image half | feature half |
  |---|---|---|
  | T=4 (T_tok 2), bf16 | 3.11 GiB | 0.04 GiB |
  | T=8 (T_tok 4), bf16 | 6.18 GiB | 0.06 GiB |
  | T=8, fp32 (upper bracket) | 11.59 GiB | 0.12 GiB |

  - Finding D confirmed: the image half is ≈ 99 % of what each pass keeps. CPU autocast policies can differ from CUDA, but the bf16 numbers fit D-021: 10.1 GiB at T=4 holds two passes, so it can't be fp32 (2 × 5.8 GiB).
  - Estimate (reserved): **T=2/4: 10.1 → ≈ 7.0 GiB** (−3.1 GiB, one image half fewer). **T=8: old ≈ 3.8 + 2 × 6.24 ≈ 16.3 GiB** (fits the OOM), **new ≈ 10.1–13.4 GiB**. "Base" (teacher, weights, optimizer, allocator cache, backward working set) is 10.1 − 2 × 3.15 ≈ 3.8 GiB at T=4. The low end assumes it doesn't grow with T; the high end assumes its backward working set (≤ 3.3 GiB at T=4) doubles at T=8. Not measured; Step 4 measures it.
  - OOM likelihood at T=8: **medium** (high end at the Q5 margin of 13.5 GiB, below the T4's 14.6 GiB). T=2/4: low (less than the 10.1 GiB already run).
- **2026-11-17: Step 4 run settings approved by Muaaz** ("OK as proposed", with the estimate above) and **set up as T05 did** ("same as T05"). Cell 2: `REPO_BRANCH = "fix/shared-image-encoding"` (back to `"main"` after the merge). Cell 3: a commented "T04_training debug run" block after the T05 block: `experiment_name = "t04_shared_enc"` (+ its `checkpoint_dir` / `log_dir`), `max_steps = 90`, `val_every_n_steps = 10_000` (no validation), schedule T=2 batch 2 from step 0, T=4 batch 1 from 30, T=8 batch 1 from 60. Notebook +22 / −3 lines in total with Step 2; valid JSON, cells compile, and the block compiles when uncommented. `docs/DECISIONS.md` **D-030** added. Waiting for the commit + push and the Colab run.
- **2026-11-24: Step 4 done. Colab Tesla T4 (14.6 GiB), code `293d667`, PyTorch 2.14.1+cu130.** The T04 block in cell 3 was uncommented for the run (not saved). 90 steps in 3 min 38 s, **no OOM**. Monitor stage summaries vs D-021 (the "before"):

  | stage | steps | median s/step (it/s) | first step | peak alloc / reserved | D-021: reserved, it/s |
  |---|---|---|---|---|---|
  | T=2, bs 2 | 0–29 | 2.267 (0.44) | 5.8 s | 5.70 / **5.79 GiB** (40 %) | 10.06 GiB, 0.35 |
  | T=4, bs 1 | 30–59 | 1.882 (0.53) | 2.1 s | 5.40 / **5.79 GiB** (40 %) | 10.16 GiB, 0.41 |
  | T=8, bs 1 | 60–89 | 2.851 (0.35) | 2.8 s | 10.05 / **10.26 GiB** (70 %) | OOM on its first step (14.28 GiB allocated) |

  - **Q5 margin met:** 10.26 ≤ 13.5 GiB at T=8 (3.2 GiB below the margin, 4.3 GiB below the T4's 14.6). T=2/4 peak reserved −4.27 / −4.37 GiB vs D-021, so "no higher" holds.
  - Vs the Step 4 estimate: T=2/4 came in 1.2 GiB below it (≈ 7.0 estimated), T=8 0.2 GiB above its low end (10.1–13.4). The estimate came from CPU autograd byte counts; the cause of the difference was not separated.
  - Speed vs D-021: T=2 ×1.27, T=4 ×1.30 (vs D-028's monitor run, 0.33 / 0.40 it/s: ×1.33 each). The image half now runs once per step.
  - **Losses finite at every step of all three stages**, shown by the monitor's per-step check (`monitor_nan_every_step = True`) and its check before the final save: a NaN/inf would have stopped the run with `NonFiniteLossError`. Loss values are written only at logging points (`log_every_n_steps = 50`, not changed in the T04 block), so `runs/t04_shared_enc/monitor_history.csv` has rows for step 0 (T=2) and step 50 (T=4) only, none at T=8; the progress bar's final `loss=0.1043, T=4` is from step 50. **Muaaz chose not to re-run with a shorter logging interval** (Done When asks for finite losses, not values).
  - 0 skipped steps (GradScaler); monitor overhead 0.05 s = 0.021 % of step time. Debug checkpoint `checkpoints/t04_shared_enc/anyup3d_step90_final.pth` (90 steps, not for use).
  - Not measured: validation (off in this run, as approved).
  - `monitor_history.csv` (pasted by Muaaz, 2026-11-24), the 2 logging points: step 0 (T=2, bs 2) loss_total 0.0819 (rec 0.0804, inp 0.00082, aug 0.00076, temporal 0), grad_norm 0.0905, grad_scale 65536, 5.72 GiB reserved; step 50 (T=4, bs 1) loss_total 0.1043 (rec 0.1011, inp 0.0024, aug 0.00081, temporal 1.15e-5), grad_norm 0.0691, grad_scale 65536, 5.79 GiB. **S17:** the CUDA bf16 grad_norm is ≈ 0.07–0.09, not the CPU dry run's 157–609, so the S17 effect doesn't show on the GPU at these 2 steps (S17 stays open, row updated). **S15:** no change of scale, 0 skips (row updated). `loss/temporal` is 0 at T=2 and non-zero at T=4, as in T05 run 1 (T01_training note).
  - **S14 checked:** the teacher fix trains at T=8 on a T4 (finite losses, 10.26 GiB). Q3 (next memory measure) is not needed.
- **2026-11-24: Close-out.** All Done When items met: tests (Step 3, 121 passed); T4 run (Step 4); S14 → `checked`; `docs/DECISIONS.md` **D-031**, and D-030's stale "not yet committed" Branch line corrected (→ `293d667`, pushed); Task Index → done. Status → done (not merged yet).
  - **Step 5, message to Munther (drafted for Muaaz, to send after the merge):**
    > Heads-up on `anyup/model.py` (T04_training, now in `main`): `AnyUp.forward` now runs through two new methods, `encode_image(image, feats_shape, output_size)` (image half) and `decode(encoded, feats, vis_attn, q_chunk_size)` (feature half). `upsample()` was removed; its only caller was `forward`. The call `model(guide, features, output_size=..., q_chunk_size=...)` is unchanged, and the outputs are bit-identical to the old code (`torch.equal`, `tests/test_shared_image_encoding.py`). No weights, layers or checkpoint keys changed. Your `test_upsamplers.py` (34) and `test_weight_loading.py` (7) pass, so Exp 0–3 need no change. Only code that called `.upsample(` directly would break. Why: training now reuses the image half for the self-consistency pass, so the T=8 stage fits on a T4 (10.3 GiB peak, was OOM).
  - `REPO_BRANCH` in cell 2 back to `"main"` (main's original line, `9e43ba2`) **before the merge, on this branch** (Muaaz, 2026-11-24; old plan: after the merge, D-030), so the merge brings it in and `main` needs no follow-up commit. The T04 block in cell 3 stays commented, as T05's does. Left for after the merge: the message to Munther.
