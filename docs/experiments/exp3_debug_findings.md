# Exp 3 (DAVIS label propagation): what is wrong and how to fix it

Summary of the T11 diagnostics (`experiments/exp3_debug.ipynb`, `exp3_debug_E.ipynb`, `exp3_anyup3d_mem.ipynb`). All J&F values are official DAVIS-2017 J&F. Runs were on a Colab T4.

## TL;DR

1. **Exp 3 scores are low for every method because the input resolution is too low.** This is not a bug in the propagation or the scoring. Swin-T on 224×224 squashed frames gives a 7×7 token grid, and the 56² propagation grid caps J&F at 0.71. Small and thin objects vanish. At those settings every upsampler fails, so comparing them (the thesis 0.266 → 0.347, T05 0.127 → 0.231) means little.
2. **At a usable resolution the pipeline works.** With DINOv2-B at 448×798 (aspect kept) and a 112×200 propagation grid, bilinear reaches 0.66 and AnyUp2D 0.79, and no object vanishes. Our implementation of the published DINO protocol gives 0.58 (the paper reports 0.62), so the code is sound.
3. **At usable resolutions AnyUp3D is the weak one:** 0.56–0.60, below bilinear and AnyUp2D, and it still loses 11–13 % of object-frames. Its cause is not known yet. Only hypotheses, listed below.
4. **AnyUp3D memory limits how far we can test it.** The attention mask is built in full. On a T4 it only fits with 4–8 frames per call at high resolution, while Exp 3 used 16.

## Problem 1: low absolute J&F for all methods

### Evidence

| Test | What changed | Result |
|---|---|---|
| A: ceiling | GT masks resized to the propagation grid and back, no model | Ceilings: 28² **0.50**, 56² **0.71** (current), 112² **0.87**, 56×100 (aspect kept) 0.82. Even a perfect model scores at most 0.71 at 56². |
| B: local window | DINO-style spatial window in propagation (r = 6, 12) | **No effect** (±0.001). Not the bottleneck. |
| C: backbone | Swin-T 224 (7×7 tokens) → DINOv2-B 224 (16×16 tokens), everything else the same | bilinear 0.127 → **0.355**, AnyUp3D 0.231 → **0.392** (29/30 clips better) |
| D: Swin at 448 | Swin-T 448² (14×14 tokens) | bilinear 0.212, AnyUp3D 0.329. Helps, but stays low. |
| E: resolution ladder (5 clips) | DINOv2-B input and propagation grid raised together | see table below |
| E: DINO reference | Published DINO protocol (476×854 input, native 34×61 grid, no upsampler) | **0.581** (paper, ViT-B/16, 30 clips: 0.623) |

Test E, DINOv2-B, 5 clips (blackswan, dog, bmx-trees, drift-chicane, parkour). The figures are J&F, with lost object-frames in brackets:

| input → propagation grid | ceiling | bilinear | AnyUp2D | AnyUp3D |
|---|---|---|---|---|
| 224² → 56² | 0.71 | 0.311 (23 %) | 0.398 (28 %) | 0.326 (48 %) |
| 448² → 112² | 0.86 | 0.590 (0 %) | **0.763** (0 %) | 0.562 (11 %) |
| 448×798 → 112×200 | 0.89 | 0.663 (0 %) | **0.791** (0 %) | 0.595 (13 %) |

"Lost object-frames" are object-frames where the object is in the GT but the prediction has 0 pixels of it.

### Why it happens

- **Coarse backbone tokens (main cause).**
  - Swin-T has stride 32, so 224² gives 7×7 tokens, and frames are squashed from 480×854 to 224².
  - An upsampler can sharpen edges, but it cannot bring back an object that is smaller than one token.
  - The DINO reference uses a coarser propagation grid than ours (34×61 vs 56²) and still scores +0.27. So the token grid, not the propagation grid, is what limits the result.
- **Coarse propagation grid (second cause).**
  - 56² caps J&F at 0.71, and squashing the aspect ratio costs more.
  - At 56² small objects disappear in the first frame already (e.g. lab-coat), and propagation can never recover them.
- **Not the cause:**
  - The spatial window (Test B).
  - The propagation and scoring code: the DINO reference matches the paper's order of magnitude, the equivalence checks PASS, and T05 is reproduced.

**Caveat:** in Test E the input resolution and the propagation grid were raised together, so their effects are not separated. The "isolation" runs (448 → 56², 224 → 112²) were not run.

### Ways to fix it (pick one)

| Option | What | Pros | Cons |
|---|---|---|---|
| **1a. DINOv2-B at 448×798 + 112×200 grid** | Standard image-backbone setup, close to the DINO protocol | Measured: bilinear 0.66 / AnyUp2D 0.79. Comparable to the literature. Fits on a T4. | AnyUp3D currently loses here (Problem 2). Not a video backbone. ~45 min / 5 clips on a T4 for all three upsamplers. |
| **1b. Keep Swin-T, at higher input (448+, aspect kept) + bigger grid** | Keeps the "zero-shot video backbone" story | Same backbone as the thesis | Measured at 448²: only 0.21–0.33. Stride 32 stays the limit. Probably still not publishable. |
| **1c. VideoMAE (AnyUp3D's training backbone)** | In-distribution for AnyUp3D | Tests AnyUp3D where it was trained | **Not tested.** VideoMAE also has coarse tokens (14×14 at 224, tubelet 2). Same resolution problem likely. |
| **1d. Keep the current setup, report the ceiling as a "lower bound"** | No change, only reporting | Zero work | Objects vanish for every method, and the reader sees J&F 0.13–0.23. Weak for a paper. |

Whatever the backbone: use a propagation grid of at least 112 on the short side, keep the aspect ratio, and drop the window.

## Problem 2: AnyUp3D underperforms where the setup works

### Evidence

- At 448 input, AnyUp3D is 0.17–0.20 below AnyUp2D and 0.03–0.07 below bilinear.
- It is the only method that still loses objects (11–13 % of object-frames).
- parkour collapses already at 224: AnyUp3D 0.086 vs bilinear 0.328 (and 0.064 in run 1 on 30 clips). It also collapsed on drift-straight on Swin-T in T05.
- Earlier (Swin-T 224, DINOv2 224) AnyUp3D did beat bilinear. But that was in the failing regime of Problem 1, where every method is near zero.
- Only 5 clips, 2 of them hard by choice. AnyUp3D has not been run on all 30 clips at 448.

### Why it happens (hypotheses, none tested)

1. **Zero-shot backbone shift.** AnyUp3D was trained on VideoMAE features only, and DINOv2 is new to it. AnyUp2D's released checkpoint was trained on more varied features.
2. **Temporal attention smears moving objects.** It attends to neighbouring frames (`window_t=4`). Fast motion (parkour, drift) could pull features to where the object *was*. This fits the "lost objects" pattern.
3. **Shorter temporal context than in training.** We used 8 frames per call, and 4 at 112×200, because of memory. Exp 3 used 16. It is not clear whether this matters, since 112² (8 frames) is already bad.
4. **Scale mismatch.** At 448 input the low-res grid is 32×32 → 112², a different token grid and output size than in training. Not checked.

### How to find out (cheap tests, T4, 5 clips, 448² → 112²)

| Test | Answers |
|---|---|
| AnyUp3D with **1 frame per call** (no temporal partner) | If this ≈ AnyUp2D, the temporal part is what hurts (hypothesis 2). If it is still bad, it is the spatial part / domain (hypothesis 1 or 4). |
| AnyUp3D with 4 vs 8 frames per call (16 needs an A100) | Effect of the temporal context (hypothesis 3) |
| Same ladder with **VideoMAE** as the backbone | Whether AnyUp3D works on its own training backbone at a usable resolution (hypothesis 1) |
| Per-frame cosine of AnyUp3D vs AnyUp2D features on parkour | Where and when the features drift |

### Ways to fix it (after the tests say which cause)

- **If it is the temporal part:** change the temporal attention, e.g. a motion-aware window, a gated temporal path, or a smaller `window_t`. This needs retraining (P2).
- **If it is the backbone shift:** train AnyUp3D on several backbones and resolutions (DINOv2, Swin, VideoMAE). This is the P3 "scaling data and backbones" plan, moved earlier.
- **If it only works on VideoMAE:** run Exp 3 on VideoMAE (option 1c) and state honestly that AnyUp3D does not generalise across backbones yet.

## Problem 3: AnyUp3D memory

### Evidence (T4 14.6 GiB, one call, `q_chunk_size=1024`)

| propagation grid | 4 frames | 8 frames | 16 frames |
|---|---|---|---|
| 56² | 1.2 GiB | 1.9 GiB | 3.5 GiB |
| 112² | 3.5 GiB | 6.5 GiB | 12.7 GiB |
| 112×200 | 5.9 GiB | 11.3 GiB | not run (mask alone 9.7 GiB) |

Without `q_chunk_size`, every 112² and 112×200 case runs out of memory.

### Why it happens

`CrossAttentionBlock3D` builds the **full** boolean window mask, (T·H·W) × (T·h·w), on the CPU, and copies it to the GPU on every call. `q_chunk_size` only chunks the attention, not the mask. So memory grows with frames² × pixels, even though each query looks at only a small window.

### Ways to fix it

| Option | What | Cost |
|---|---|---|
| **3a. Build the mask per query chunk** | Compute only the rows of the current query chunk, inside the chunk loop | Small code change. Outputs must stay exactly the same (regression test). |
| **3b. Real windowed / sparse attention** | e.g. NATTEN or PyTorch FlexAttention with a block mask | Bigger change, also faster. Outputs must match, or the model must be retrained. |
| **3c. Live with it** | 4–8 frames per call on a T4, 16 on an A100 | No code. Temporal context differs between settings, and this must be reported. |

## Suggested order

1. **Decide the Exp 3 setup (Problem 1).** Recommended: 1a (DINOv2-B 448×798 → 112×200). It is the only measured setup that is publishable. Optionally run the isolation runs to separate the input resolution from the grid.
2. **Run the cheap AnyUp3D diagnostics (Problem 2)**: 1 frame vs 4 vs 8, and VideoMAE. This decides whether the fix is in evaluation or in training.
3. **Fix the mask (3a)** before any big run, so the 16-frame context fits and run time drops.
4. **Then run all 30 clips** in the chosen setup with bilinear / AnyUp2D / AnyUp3D, and add the paired statistics.
