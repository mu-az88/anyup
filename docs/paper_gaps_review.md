# AnyUp3D: Where the Paper Is Weak, and How to Fix It

*Review written on 2026-11-24 from the thesis text, the T05 results, the T11 debug run and `skipped_and_future_concerns.md`.*

## The short version

We currently show that AnyUp3D makes video features **more stable** and that this **helps label propagation when the backbone is very coarse**. We do not yet show that the **3D architecture** causes the gain, that the extra stability is **correct** and not just blur, or that the method is **universal** in a useful way. Two cheap experiments decide most of the story: a fairly fine-tuned AnyUp2D, and AnyUp2D with simple temporal smoothing. Run those first.

---

## 1. The claims that go further than the evidence

**"The gain comes from the 3D temporal modelling."** This is the biggest problem. AnyUp3D is AnyUp2D *plus* 6,000 steps of fine-tuning on UCF-101 with a VideoMAE teacher and a temporal loss. The AnyUp2D we compare against is the original checkpoint with no fine-tuning. So the gain could come from the fine-tuning (new domain, new teacher, temporal loss) and not from the 3D convolutions or 3D attention. Right now nothing separates the two.

**"AnyUp3D adds temporal consistency."** Our evidence is adjacent-frame cosine similarity (AnyUp2D 0.919 → AnyUp3D 0.969). This metric rewards blur over time: a feature map that never changes scores a perfect 1.0. AnyUp3D is even more stable than the raw VideoMAE features it starts from (0.880). That might be good, or it might be over-smoothing. We cannot tell yet.

**"Universal and backbone-agnostic."** The model was trained on one backbone. On Swin-T with a 7×7 grid it gains +0.104 J&F over bilinear. With better input features (DINOv2, 16×16 grid, T11 Test C) the gain drops to +0.037. At 448² it doesn't run at all on a T4. The gain shrinks as the backbone gets better, and a reviewer will notice.

**"Spatial quality is preserved."** There is no valid Exp 1 number yet (S02). In the old step500 run, AnyUp3D scored slightly *below* AnyUp2D (41.93 vs 42.36 mIoU).

**Wording to delete.** Chapter 5 says temporal modelling is "necessary and sufficient", and Chapter 3 calls it "state-of-the-art spatiotemporal upsampling". Neither is supported. "The first temporally consistent universal upsampler" needs a fresh literature check (2025–26) before we keep it.

**Inconsistent numbers in the thesis.** The abstract pairs the *weighted* J&F (0.266 → 0.347) with the *unweighted* clip count (27/30). Chapter 5 quotes the unweighted mean (0.269 → 0.342). All of these are old-protocol numbers, and T05 replaces them. The sign-test p-value belongs to neither (T04).

---

## 2. What an angry reviewer will say

> **"Your best J&F is 0.39. DINO ViT-B/16 gets 0.62 on DAVIS with plain features and no upsampler."**

Our Exp 3 protocol is not the standard one: squashed 224² input, a 56² propagation grid, non-standard settings. T11 Test A shows the 56² grid alone caps J&F at 0.71. So the gain is measured inside a weak setup. **Fix:** run the standard DINO protocol (T11 Test E), with AnyUp3D included.

> **"It doesn't scale."**

AnyUp3D uses about 10.5× the memory of AnyUp2D and cannot handle DINOv2 at 448² on a T4. Full attention over all frames and pixels is the core of the design, so this is a limit of the architecture, not of our code. Exp 0 has only been measured on a T4 (S07).

> **"Your consistency metric can be gamed."**

See Section 1. Stability alone proves nothing without a matching accuracy number.

> **"Where are the obvious baselines?"**

There are four:
- AnyUp2D followed by simple temporal smoothing (a moving average of features over frames).
- AnyUp2D fine-tuned with exactly our recipe.
- At least one other published upsampler (FeatUp, LoftUp, JAFAR).
- SAM 2, which we cite as a key baseline but never ran.

> **"Where are the ablations?"**

Nothing separates the 3D convolutions from the 3D attention, and nothing varies the temporal kernel, the temporal window, the temporal loss or the RGB-difference gate. The thesis admits this itself.

> **"The training is tiny and there is only one run."**

Ten UCF-101 classes (1,170 clips), 6,000 steps, a curriculum that stopped at T=4, one seed. Which checkpoint is canonical is still open (S06).

> **"It fails on fast motion."**

`drift-straight` collapses to J&F 0.033, and `parkour` with DINOv2 drops to 0.064 vs bilinear's 0.328. At chunk boundaries AnyUp3D loses 0.032 in temporal cosine, against 0.009 for AnyUp2D. This looks like a real weakness, not noise.

> **"One dataset, 30 clips."**

Everything is on the 30 DAVIS-2017 validation clips, with a single downstream task.

---

## 3. What would actually convince people

### Must have (without these, expect a rejection)

1. **A fair AnyUp2D.** Fine-tune AnyUp2D on the same data, teacher, steps and losses, with and without the temporal loss. If AnyUp3D still wins, the architecture claim holds. This is the most important experiment in the paper.
2. **Trivial smoothing baselines.** Run AnyUp2D, then average features over neighbouring frames (a moving average or a 3-frame mean). If AnyUp3D doesn't beat this, the method has no reason to exist.
3. **A consistency metric that checks correctness.** Estimate optical flow on the RGB frames (e.g. with RAFT), warp frame t+1's features back to frame t, and measure the agreement. Always report it next to per-frame accuracy, ideally as a stability-vs-accuracy plot.
4. **The standard propagation protocol** (T11 Test E) with AnyUp3D in it. This needs windowed or chunked attention so it fits in memory.
5. **Exp 1 on an A100** (S02), to settle the "no spatial regression" claim.
6. **Core ablations:** 3D conv only, 3D attention only, temporal kernel size, temporal window size, temporal loss on/off.

### Strong additions

- **More correspondence tasks** in the standard propagation literature (DINO, CRW): JHMDB keypoint propagation, VIP body-part propagation, and above all **TAP-Vid-DAVIS point tracking** by feature matching. Point tracking asks directly whether high-resolution features are *correct* over time, not just smooth.
- **A dense video task with a trained head:** a linear probe on VSPW video semantic segmentation (mIoU plus its temporal-consistency score), or video depth with a linear head (accuracy plus flicker).
- **One other published upsampler** (LoftUp or JAFAR), applied frame by frame.
- **Long videos:** overlapping upsampler chunks, to remove the boundary dip.

### Nice to have

- A second training seed.
- Training on more than one backbone, to back up "universal" properly.

---

## 4. Suggested order

1. Fair AnyUp2D fine-tune + smoothing baselines (they decide the story).
2. Flow-warped consistency metric (cheap: it reuses Exp 2).
3. Standard DINO propagation protocol with AnyUp3D (needs the memory fix).
4. Ablations.
5. Exp 1 and Exp 0 on an A100.
6. Extra tasks (TAP-Vid, JHMDB, VSPW) as time allows.

*Housekeeping:* the thesis PDF title page says **B.Sc.**, but `CLAUDE.md` and `project_context.md` say **M.Sc.** Fix whichever is wrong.
