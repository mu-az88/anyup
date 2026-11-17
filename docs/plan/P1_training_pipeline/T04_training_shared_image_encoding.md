# T04_training — Compute the Image Half Once per Training Step (T=8 Fits on a T4)

| Field | Value |
|---|---|
| Task ID | T04_training |
| Plan | P1_training Training pipeline (`plan.md`) |
| Branch | `fix/shared-image-encoding`, branched from `main` after the teacher fix is merged into `main` (D-008) |
| Status | Q&A |
| Depends on | Teacher 16-frame fix (`teacher_16frame_plan.md`): its T=8 debug check moved here (D-022, S14). T05_training (monitor), used to measure before/after (D-025) |
| Blocks | run_03 on a T4 reaching the T=8 stage (`start_step=15000`) |

## Goal

Make the T=8 training stage fit in a T4's 14.6 GiB by computing AnyUp's image half once per step and reusing it for both student passes (main + self-consistency). Same layers, weights, checkpoints and outputs: **not an architecture change**, only how the step's computation is organised.

## Scope

- In scope:
  - `anyup/model.py`: split `AnyUp.forward` into an image half and a feature half (Design), so the image half can be computed once and reused.
  - `Train_3D_full.ipynb` cell 13: the main pass and the self-consistency pass share one image encoding.
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
| 1 | API: (a) add `encode_image(image, feats_shape, output_size)` and `decode(encoded, feats, …)`, and rewrite `forward` as `decode(encode_image(…))`, so there's one code path, proven equal by the test; (b) add them but leave `forward`'s code untouched, as a second path that could drift. | _open_. Proposal: (a) |
| 2 | `anyup/model.py` is also used by Munther's experiments (through `forward`). OK to change it there, given `forward`'s outputs stay equal (test), or should he review first? | _open_ |
| 3 | If T=8 still doesn't fit after this: decide the next measure now, or after the measurement? | _open_. Proposal: after, with the numbers |
| 4 | Test tolerance: outputs of old vs new `forward` **exactly equal** (`torch.equal`); gradients equal within `rtol=1e-5, atol=1e-7` (Finding E). | _open_ |
| 5 | Safety margin: what peak reserved memory at T=8 counts as "fits"? The real run will also see larger or odd videos and other variation. | _open_. Proposal: ≤ 13.5 GiB reserved (about 1 GiB headroom) |

## Design

`anyup/model.py`, sketch (final names per Q1):

```python
def encode_image(self, image, feats_shape, output_size):
    """Image half: depends only on the frames (+ feature grid / output size). Reusable across passes."""
    _, _, t, h, w = feats_shape
    enc = ...                                   # image_encoder + rope, as in forward today
    q     = F.adaptive_avg_pool3d(self.query_encoder(enc), output_size=(t, *output_size))
    k_img = F.adaptive_avg_pool3d(self.key_encoder(enc),   output_size=(t, h, w))
    return q, k_img

def decode(self, encoded, feats, q_chunk_size=None):
    """Feature half: everything that depends on the features."""
    q, k_img = encoded
    k = self.aggregation(torch.cat([k_img, self.key_features_encoder(F.normalize(feats, dim=1))], dim=1))
    return self.cross_decode(q, k, feats, q_chunk_size=q_chunk_size)

def forward(self, image, features, output_size=None, vis_attn=False, q_chunk_size=None):
    output_size = output_size if output_size is not None else image.shape[-2:]
    return self.decode(self.encode_image(image, features.shape, output_size), features, q_chunk_size)
```

Cell 13:

```python
encoded    = anyup.encode_image(crop_high_5d, guidance_feats_low.shape, out_size)   # image half once per step
pred_feats = anyup.decode(encoded, guidance_feats_low)
...
pred_aug   = anyup.decode(encoded, gt_aug_low)                                     # reuses it (Finding B)
```

## Steps

1. `tests/test_shared_image_encoding.py` against the **current** `forward` first (record the reference outputs/gradients), then the `model.py` change; run locally.
2. Cell 13 change; clear outputs. Local CPU dry run of one step with a tiny teacher, like D-006's.
3. Run Munther's and the existing model tests: `test_upsamplers.py`, `test_weight_loading.py`, `test_teacher_16frame.py`, `test_losses.py`.
4. Colab T4 debug run through the T05_training monitor (`TrainMonitor`, D-028; the old `scripts/debug/teacher_debug_cells.py` is superseded). Peak memory at T=2/4/8 vs D-021 (the monitor's T4 run 1 reproduced D-021's T=2/4 memory within 0.04 GiB, so the two are comparable); finite losses at all three.

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

## Implementation Log

_Updated live during implementation. Per step: what changed, files touched, test results (actual numbers/shapes). Also log mid-implementation questions and answers, changed decisions (old → new, why), and anything added or removed. Sections above are edited in place to match, so the finished file is a report of what was actually done._

- **2026-11-15: task file written** (D-022), from D-021's numbers and reading `anyup/model.py` / cell 13. Nothing implemented or measured yet. Status `Q&A`: Q1–Q5 are open.
