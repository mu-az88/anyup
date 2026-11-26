# AnyUp3D Training Pipeline: Issues and Fixes

- **A. Issues that limit results** (1–6), ordered by expected impact
- **B. Bugs and correctness** (B1–B6)
- **C. Efficiency** (C1–C3)

**Suggested order:** do A1–A3 first (small code changes, direct effect on the training signal). Then fix validation (B5) so you can measure the effect, then A6 and A5, and finally A4, which is the largest change. The B and C items are small and can be done alongside. A1 and A2 change what the model learns, so retrain from the 2D initialization rather than resuming old checkpoints.

---

# A. Issues that limit results

---

## 1. The teacher sees clips it wasn't trained on, so the ground truth is degraded

### Problem
VideoMAE-base was pretrained only on **16-frame** clips (8 tubelets × 196 patches = 1568 tokens). Its position table has exactly 1568 entries. The curriculum (T = 2 → 4 → 8) feeds the teacher the same short clip as the student, which causes two problems:

- **Out-of-distribution input.** The teacher behaves differently on 2–8 frame clips, so its features get worse. Those features are the student's ground truth, so AnyUp3D learns to imitate a degraded target.
- **A broken patch.** To make the short clip fit, `extract_teacher_features` stretches the position table along time by bilinear interpolation. VideoMAE's table is one 1D sinusoid over the flattened (t, h, w) sequence, not separate time and space embeddings. Blending its entries produces vectors that correspond to no real position.

There's also a **train/eval mismatch**: the Exp 2/3 evaluations take features from full 16-frame clips, while training used short-clip features.

### Fix
**Always run the teacher on 16 frames.** Give the student a T-frame window and take the matching slice of tubelets from the teacher's output.

```
16 frames ──► teacher ──► [t0 t1 t2 t3 t4 t5 t6 t7]
                                  └──┘ window
frames f0..f0+T ──► student ──► predicts features for that window
```

1. Check `encoder.embeddings.position_embeddings`: expect a plain tensor of shape `(1, 1568, 768)`.
2. Set `TEACHER_T = encoder.config.num_frames` (16) and pass it to every `build_loader` call.
3. Simplify `extract_teacher_features`: require 16 frames and delete the position-table interpolation.
4. In the step, pick a random tubelet-aligned window (`f0 = 2·tok0`) and slice the frames, GT and input features the same way.
5. Self-consistency: add noise to the full clip, then slice. Validation: use a fixed window.

### Notes
- The curriculum's purposes (limiting student VRAM, a gentle start from 2D weights) are preserved, because the student's input size is unchanged.
- Teacher VRAM rises only modestly (no_grad, bf16). Data loading gets slower, since 16 frames are decoded per sample.
- Spatial accuracy is unaffected: slicing happens along time only.
- Detailed implementation: `teacher_16frame_plan.md`.

---

## 2. The temporal loss pushes features toward being static

### Problem
```python
feat_diff = pred_feats[:, :, 1:] - pred_feats[:, :, :-1]
loss_temp = (feat_diff.pow(2) * gate).mean() * t_lam
```
- **It rewards freezing.** It penalizes *any* change between tubelets, so its best answer is features that never change. The teacher's features change legitimately over time, so this loss fights the reconstruction loss and causes oversmoothing. That's exactly the "it's just smoothing" objection a reviewer would raise, and it inflates the Exp 2 adjacent-frame cosine. The RGB gate only switches the loss off where pixels move a lot.
- **Its scale is arbitrary.** It squares raw 768-d differences, so it measures *absolute* change while the cosine term measures *relative* change (direction). Its strength scales with feature norm² (for the same 5% change, a norm-50 vector gives about 2,500× the loss of a norm-1 vector), so λ = 0.2 has no stable meaning.

### Fix
**Match the teacher's change instead of suppressing change**, on unit-normalized features:

```python
def temporal_consistency_loss(pred, gt, gate=None):
    if pred.shape[2] < 2:                                   # T_tok = 1 → no pairs
        return pred.new_zeros((), dtype=torch.float32)
    p = F.normalize(pred.float(), dim=1)                    # direction only → scale-free
    g = F.normalize(gt.float(), dim=1)
    err = ((p[:, :, 1:] - p[:, :, :-1]) - (g[:, :, 1:] - g[:, :, :-1])).pow(2).sum(1, keepdim=True)
    return (err * gate if gate is not None else err).mean()
```

With error `e_t = p_t − g_t`, this equals `‖e_{t+1} − e_t‖²`, the *change in error*. A stable error costs nothing; flicker and freezing are both penalized. Reconstruction penalizes the *size* of the error on each frame; this loss decides *where the residual error goes* (into a stable offset rather than flicker).

### Notes
- The RGB gate is no longer needed for correctness. Drop it, or keep it as an ablation row.
- **λ needs one retuning pass.** Aim for the temporal term at roughly 10–30% of the reconstruction term early in the T=4 stage.
- Direction-only: feature magnitude is still constrained by the reconstruction loss.
- Exp 2's adjacent-frame cosine may drop slightly (the inflated part disappears). Add a **dynamics error** metric (the same formula at evaluation) to show consistency without oversmoothing.
- Include it in the ablation (with vs. without).
- Detailed implementation: `temporal_dynamics_loss_plan.md`.

---

## 3. The temporal warmup is wasted

### Problem
The warmup ramps λ from 0 to full over steps 0 → 5000. But in the T=2 stage each clip has one tubelet (T_tok = 1), so the temporal loss is 0 the whole time. The ramp completes while the loss is off. At step 5000, T=4 starts and the loss appears at full strength, just as the clips double in length.

```
step:      0 ─────────── 5000 ─────────── 15000
T_tok:     1             │ 2
λ now:     0 → 0.2       │ 0.2 instantly
λ fixed:   0             │ 0 → 0.2 → 0.2
```

### Fix
Start the ramp where T_tok > 1 first happens, derived from the schedule:

```python
TEMP_START = next((s.start_step for s in cfg.t_schedule if s.t // TUBELET_SIZE > 1), None)

def get_temporal_lambda(step):
    if TEMP_START is None or step < TEMP_START:
        return 0.0
    w = cfg.temporal_lambda_warmup_steps
    return cfg.lambda_temporal_consistency * (1.0 if w <= 0 else min(1.0, (step - TEMP_START) / w))
```

### Notes
- Consider a shorter ramp (about 1000–2000 steps) so most of the T=4 stage runs at full strength.
- Verify by printing λ at steps 0, 4999, 5000, 6000 and 15000, and log it to TensorBoard.
- Resuming works, since λ depends only on the step number. Tune λ for issue 2 with this fix in place.

---

## 4. The training task doesn't match the inference task

### Problem
- **Different ratio.** Training always upsamples 7×7 → 14×14 (2×). Exp 3 upsamples 14×14 → 56×56 (4×), and figures at 224×224 are 16×. The model never practices the ratios it's evaluated on and is never supervised finer than 14×14.
- **Different input.** At inference, the input is real teacher features of a sharp frame at native 14×14. In training, it's features of a blurry image (112 px upscaled to 224), bilinearly shrunk to 7×7.

### Fix
Use the **crop-as-target** setup. The input is real 14×14 teacher features of the full frame. The target is teacher features of a zoomed-in crop. Upsample the full frame and compare only the crop region. The crop size sets the ratio r; sample r ∈ {2, 4}.

```
full frame ─► teacher ─► 14×14 ─► AnyUp ─► (14r)×(14r) ─► cut crop region (14×14)
zoomed crop ─► teacher ─► 14×14 target ─────────────────────┘ compare
```

1. Check the original AnyUp training code in the repo and match its scheme.
2. **Use higher-resolution video.** The crop is sharp only if the source is at least 224·r px. UCF-101 (320×240) can't support this; YouTube-VOS (720p) can. This ties into issue 5.
3. Add a pair sampler that returns `full`, `zoom` and the crop offset in output cells.
4. Training step: `pred_full = anyup(full, teacher(full), output_size=(14r, 14r))`, then `pred_feats = pred_full[..., oy:oy+14, ox:ox+14]` vs. `teacher(zoom)`.
5. Input consistency on `pred_full`; RGB gate (if kept) on `zoom`; self-consistency off at first.
6. Validate at a fixed r = 4 (matching Exp 3) with a center crop.
7. Checks: at r = 1 the task becomes identity (loss → ~0); compare PCA of prediction vs. target for alignment; measure VRAM at r = 4, T = 8.

### Notes
- This is the largest change: a new dataset, a new sampler, and up to 16× more query pixels at r = 4. Do it after the 2D-vs-3D baseline experiment and fixes 1–3.
- It's worth an ablation row: fixed synthetic 2× vs. crop-as-target {2, 4}.
- Known trade-off: the teacher sees less context in a zoomed crop, so the target isn't a perfect high-res version. This is the accepted compromise in this line of work.

---

## 5. The data is narrow and weakly augmented

### Problem
- Ten UCF-101 classes is a small, homogeneous domain.
- `Resize((256, 256))` stretches 320×240 frames, and the `CenterCrop(256)` after it does nothing.
- The only augmentation is a 32-px crop jitter.
- Frames are consecutive (about 25 fps), so a T=8 clip covers about 0.3 s with little motion. Exp 3 samples at about 2 fps, so the temporal attention is tested on far larger frame-to-frame changes than it was trained on. This may explain the losses on the fast-panning drift clips.
- Failures are silent: `except Exception: frames = []` produces all-zero clips, and `stream.frames == 0` falls back to `t*4`, which can produce duplicated (motionless) clips.

### Fix
- Expand the data: full UCF-101, a Kinetics subset, or YouTube-VOS (which also solves issue 4's resolution constraint).
- Resize the short side to the target size, then crop. Preserve aspect ratio.
- Add horizontal flip and color jitter, applied **identically across all frames** of a clip.
- Sample a random temporal stride (1–4) per clip so the model sees real motion.
- Count and log failed or short decodes instead of silently padding.

---

## 6. Batch size 1 gives very noisy gradients

### Problem
From T=4 onward, each optimizer step sees one clip. Gradients are noisy, which slows convergence and makes results seed-sensitive.

### Fix
Gradient accumulation to an effective batch of about 8:

```python
ACCUM = 8   # effective batch = batch_size × ACCUM; no extra memory, only slower optimizer steps
(total_loss / ACCUM).backward()
if (step + 1) % ACCUM == 0:
    torch.nn.utils.clip_grad_norm_(anyup.parameters(), cfg.grad_clip_max_norm)
    optimizer.step(); optimizer.zero_grad(); scheduler.step()
```

### Notes
- `max_steps`, save/validation intervals, the curriculum `start_step` values and the scheduler's `T_max` must then count **optimizer** steps, not micro-steps. Pick one convention and apply it everywhere.
- Remove the GradScaler when doing this. It's unnecessary with bf16 (see B2).

---

# B. Bugs and correctness

## B1. The smoke test (cell 14) calls AnyUp with the wrong arguments

### Problem
```python
out = anyup(dummy_low_5d, dummy_high_5d, output_size=(TOKEN_H, TOKEN_W))
```
The signature is `anyup(guide, features, output_size=...)`: guide image first, then features. The test passes the **low-res pixels as the guide** and the **high-res pixels as features**, and never uses the dummy features it creates. It can pass while testing nothing like the real call path.

### Fix
```python
dummy_feats_low = torch.randn(1, ENCODER_DIM, 2, TOKEN_H_LOW, TOKEN_W_LOW, device=device)
out = anyup(dummy_high_5d, dummy_feats_low, output_size=(TOKEN_H, TOKEN_W))   # guide = HR pixels, then features
```

---

## B2. GradScaler isn't needed with bf16

### Problem
Loss scaling exists to stop fp16 gradients from underflowing. bf16 has the same exponent range as fp32, so the scaler does nothing useful. Its state also isn't saved in checkpoints, so on resume it restarts at scale 65536 and can skip the first optimizer steps.

### Fix
Remove the scaler and step directly:
```python
optimizer.zero_grad()
total_loss.backward()
torch.nn.utils.clip_grad_norm_(anyup.parameters(), cfg.grad_clip_max_norm)
optimizer.step()
scheduler.step()
```

---

## B3. The config is pickled into checkpoints

### Problem
`"cfg": vars(cfg)` stores `TStage` dataclass objects. That's why loading needs `weights_only=False`, and checkpoints break if `TStage` is renamed or moved.

### Fix
```python
from dataclasses import asdict
"cfg": asdict(cfg),   # plain dicts/lists → loadable with weights_only=True
```
Old checkpoints still need `weights_only=False` once; new ones won't.

---

## B4. The resume checkpoint is loaded twice

### Problem
Cell 7 loads the checkpoint for the model weights; cell 10 loads the same file again for the optimizer and scheduler. That wastes time and Drive I/O, and the two loads can drift apart if one path changes.

### Fix
Load once in cell 7 into `resume_ckpt`, then in cell 10 use `resume_ckpt["optimizer"]` and `resume_ckpt["scheduler"]`, and `del resume_ckpt` afterwards to free memory.

---

## B5. Validation is noisy and not comparable over time

### Problem
- The spatial crop and the temporal start are random, so the same checkpoint gives different scores.
- Validation runs at the *current* curriculum T, so the curve jumps at every stage change.
- It runs over the whole test split every 1000 steps, which is slow.
- It logs only reconstruction loss, not temporal consistency, which is the paper's main claim.

### Fix
- Use a fixed seed, center crop, fixed temporal start and fixed T (e.g. T=4) for every validation run.
- Use a fixed subset of about 200 clips.
- Log both `val/rec` and a temporal metric: adjacent-frame cosine plus the dynamics error from issue 2.

---

## B6. The self-consistency loss is weak and expensive

### Problem
Gaussian noise with σ = 0.05 on normalized pixels barely changes the input, so the "augmented" features are nearly identical to the original. Meanwhile it costs an extra teacher pass **and** an extra student pass with gradients, roughly doubling student memory and compute.

### Fix
Either use augmentations that actually change the input (color jitter, blur, small spatial shifts applied consistently across frames), or set `lambda_self_consistency = 0` and spend the saved compute on batch size or longer clips. Decide with a short ablation.

---

# C. Efficiency: more steps in the same Colab time

## C1. Video decoding is a bottleneck

### Problem
For each sample, `av` decodes every frame from 0 up to the random start, then discards them. With only 2 workers, the GPU often waits on data. This gets worse with the 16-frame teacher (issue 1).

### Fix
Pre-extract frames once to JPEG or NPY on local disk, then load only the needed indices. A lighter alternative is `container.seek(...)` to jump near the start frame before decoding.

---

## C2. Teacher features can be cached

### Problem
The teacher is frozen, yet it runs twice per step (and a third time for self-consistency) on data that repeats every epoch.

### Fix
Precompute teacher features for a few fixed crops per video and store them on disk. Training then loads features instead of running VideoMAE. This is especially worthwhile once the teacher always runs on 16 frames. It's incompatible with fully random crops, so either fix a small set of crops per clip or keep the teacher online only for the augmentations you need.

---

## C3. Checkpoints fill Drive

### Problem
A full checkpoint (model, optimizer, scheduler) every 250 steps accumulates quickly on Drive.

### Fix
Keep only the last N checkpoints (e.g. 3) plus the best one by validation score:
```python
ckpts = sorted(ckpt_dir.glob("anyup3d_step*.pth"), key=lambda p: p.stat().st_mtime)
for old in ckpts[:-KEEP_LAST]:          # KEEP_LAST = 3; exclude the "best" file from this list
    old.unlink()
```
