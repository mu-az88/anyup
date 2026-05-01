# Muaz Notes — Chapter 3 Review

## What is NOT Implemented from the Friend's Notes

| # | Note | Status |
|---|---|---|
| 1 | AnyUp `\cite{anyup}` on every mention | Not done — citation only at line 358 |
| 3 | Add AnyUp vs JAFAR training strategy in Possible Scenarios | Not done — section missing entirely |
| 4 | Remove Flow Warping Error from functional requirements (conflict with what was actually measured) | Not done — still at line 47 |
| 5 | Functional and non-functional requirements rewrite | Not done |
| 6 | Section 3.3.1 opening needs more explanation | Not done |
| 7 | Option B + decision rephrase in 3.3.1 | Not done |
| 8 | Stage 1 (3.4.1) first paragraph rephrase | Not done |
| 9 | Forward-reference implementation sections from 3.4 | Not done |
| 10 | Remove Gaussian derivatives claim from LFU in Stage 2 | Not done — still at line 298 |
| 11 | "Large-scale training" UCF-101 phrasing | Not done — still at line 333 |
| 12 | DAVIS enabling temporal consistency measurement | Not done — still at lines 349–353 |
| 13 | GPU mention removed from 3.6 intro | Not done — still at line 358 |

Already done: note 2 (architecture overview wording is already correct).

---

## Suggestions for Functional Requirements

The requirements should match what was actually built and evaluated. Suggested rewrite:

> **FR1.** The system must accept a sequence of $T$ consecutive video frames alongside their corresponding low-resolution feature maps as input, and produce spatially and temporally coherent high-resolution feature maps as output.
>
> **FR2.** The system must support arbitrary upsampling scales, preserving the resolution-agnostic behavior of the original AnyUp model.
>
> **FR3.** The system must reduce frame-to-frame feature inconsistency compared to the frame-by-frame AnyUp baseline, measured by the mean cosine similarity between adjacent predicted frames on the DAVIS-2017 validation split.
>
> **FR4.** The system must maintain competitive spatial accuracy relative to the original AnyUp baseline, measured by the J&F score on the DAVIS-2017 video object segmentation benchmark.

**Key changes from current version:**
- Flow Warping Error removed (was never computed)
- TC metric replaced with cosine similarity (the metric actually reported in Chapter 4)
- "Demonstrably reduce" softened to "reduce"

---

## Suggestions for Non-Functional Requirements

Keep the current three and add one:

> **NFR1. Backbone Agnosticism:** The temporal modules must integrate into the AnyUp pipeline without making any assumptions about the feature extractor used.
>
> **NFR2. Trainability Under Hardware Constraints:** The model must be trainable on a single academic GPU (NVIDIA T4, 15 GB VRAM), with sequence length $T$ and batch size serving as adjustable parameters to accommodate memory limits across curriculum stages.
>
> **NFR3. Temporal Generalization:** The model must learn a temporal receptive field that transfers from short training clips to longer inference sequences without re-training or architectural change.
>
> **NFR4. Scope Constraints:** Real-time processing is out of scope. The primary focus is spatial fidelity and temporal stability.

**Changes from current version:**
- NFR2 (was "Scalability") renamed and made hardware constraint explicit
- NFR3 is new — captures the curriculum generalization goal
- NFR4 keeps the scope constraint unchanged

---

## On the AnyUp vs JAFAR Possible Scenario (Note 3)

Based on the AnyUp paper excerpt, the two options for the training scenario are:

**Option A — JAFAR strategy:** Train only at low resolutions — both teacher features and student inputs are computed from scaled-down images (e.g., upsample 16×16 → 32×32 where both are extracted from small inputs). Simple and efficient but less powerful.

**Option B — AnyUp crop-based strategy:** Sample a high-resolution crop, feed it to the frozen teacher to get the target features, then downsample the same crop to create the low-resolution guidance input. The upsampler learns to recover the teacher's features from the downsampled guidance.

**Decision: Option B adopted.** The crop-based strategy provides a true high-resolution supervision signal, forcing the model to learn meaningful upsampling rather than trivial low-resolution interpolation. In video, this matters more because temporal artifacts are more visible at higher resolutions.

**Open question before writing:** Should the scenario also explain the temporal extension — that the same spatial crop must be applied consistently across all $T$ frames (a "spatial tube crop")? This is the video-specific addition to the AnyUp crop strategy.

---

## Status of thesis_fixes_instructions.md (Chapter 3 items)

All previously tracked fixes are done:

| # | Fix | Status |
|---|---|---|
| 5 | Future tense → present in Evaluation Plan | Done — line 574 |
| 8 | Implementation callout about 3D windowed vs column-wise temporal attention | Done — lines 423–425 |
| 9 | Chapter Summary: 3-stage pipeline, no standalone temporal module | Done — lines 587–597 |
| 11 | T=1 mechanism explanation | Done — lines 470–471 |
| 14 | Figure label renamed to `fig:anyup3d_framework` | Done — line 275 |
| 15 | Missing space before "such as" | Done — line 282 |
| Ref B | `\cite{videomae}` instead of V2 | Done — lines 358, 453 |
