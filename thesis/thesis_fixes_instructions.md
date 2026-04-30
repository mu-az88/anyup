# AnyUp3D — Thesis Fix Instructions

Issues are grouped by the same severity tiers as the original review.
Each item states what to do and where. Work through **Critical** items first.

---

## Critical — Factual Errors / Cross-Chapter Contradictions

### 1. Training dataset: replace YouTube-VOS with UCF-101

**Where:** Abstract, Chapter 1 (Objectives + Methodology sections).

Replace every occurrence of "YouTube-VOS" with "UCF-101". YouTube-VOS is never
used in the actual work; UCF-101 is the dataset used throughout Chapters 3 and 4.

---

### 2. T-curriculum: fix Chapter 4 to match Chapter 3

**Decision:** Chapter 3's curriculum table is correct. Fix Chapter 4.

**Where:** Chapter 4, wherever the curriculum stages are described.

- Change the stated stages from "T=1 → T=4 → T=8 → T=16" to "T=2 → T=4 → T=8".
- Add a sentence clarifying that training ended at 6,000 steps, so the T=8 stage
  (triggered at step 15,000) was never reached in practice.
- Ensure the language is consistent: only the T=2 and T=4 stages were actually
  completed during the 6,000-step run.

---

### 3. DINOv2, CLIP, and SigLIP: correct the framing in Chapter 2

**Decision:** Fix Chapter 2 — these are supported architectures, not tested ones.

**Where:** Chapter 2, the paragraph that currently says DINOv2 is "one of the three
primary backbones used in the AnyUp3D experimental pipeline, alongside CLIP and SigLIP."

Rewrite to make clear that DINOv2, CLIP, and SigLIP are examples of architectures
the backbone-agnostic design is compatible with, not backbones evaluated in the
experiments. The only backbone used in experiments is VideoMAE.

Example rewrite: *"The backbone-agnostic design is compatible with a range of
pretrained feature extractors, including DINOv2, CLIP, and SigLIP; the experiments
in Chapter 4 use VideoMAE as the primary extractor."*

---

### 4. Remove "flow warping error" from the Abstract

**Decision:** Remove the metric — report only what was computed.

**Where:** Abstract.

Delete the reference to "flow warping error" as a temporal stability metric. The
only temporal consistency metric reported in Chapter 4 is cosine similarity between
adjacent frames. The Abstract must only claim metrics that appear in the results.

---

### 5. Chapter 3 Evaluation Plan: change future tense to present

**Where:** Chapter 3, "Planned Downstream Evaluation" subsection (end of chapter).

The subsection currently describes the DAVIS-2017 J&F evaluation as something that
"will be conducted following training completion." Those results are already
presented in Chapter 4.

Rewrite the subsection so it says the downstream evaluation "is presented in
Chapter 4" rather than framing it as future work.

---

## Unclear — Confusing or Misleading Framing

### 6. Abstract: change future tense to past tense for completed results

**Where:** Abstract.

Change "We anticipate that AnyUp3D will demonstrate significant improvements..." to
"AnyUp3D demonstrated significant improvements..." (or equivalent past tense).
The Abstract must report what was found, not what was anticipated.

---

### 7. "Temporal Attention Module": clarify it is not a standalone component

**Decision:** Clarify that temporal attention is integrated into the 3D
cross-attention block, not a separate module.

**Where:** Chapter 1 (introduction/contributions), Chapter 2 (architecture
overview), Chapter 3 chapter summary.

Replace all references to a distinct "Temporal Attention Module" with language that
makes clear temporal attention is realized inside the 3D windowed cross-attention
block (CrossAttention3D) — via 3D masking and 3D convolutions in the encoder.
Do not imply it is a separable component alongside the 3D Convolution block.

Suggested terminology: "temporally-extended cross-attention" or "3D cross-attention
with temporal masking."

---

### 8. Chapter 2 temporal attention definition: add implementation callout in Chapter 3

**Decision:** Keep Chapter 2 as theoretical context; add an explicit callout in
Chapter 3.

**Where:** Chapter 3, at or near the CrossAttention3D implementation description.

Chapter 2 defines temporal attention as a query at (x, y) attending to keys at the
same (x, y) across all T frames — a pure temporal self-attention. The actual
implementation is a 3D windowed cross-attention (HR query tokens attending to LR
key tokens within a spatiotemporal 3D window), which is a different mechanism.

Add a short implementation note in Chapter 3 — one to two sentences — explaining
that the theoretical framing in Chapter 2 is a conceptual simplification, and that
the realized mechanism is 3D windowed cross-attention rather than pure column-wise
temporal self-attention. Briefly state why (efficiency, joint spatiotemporal
context).

---

### 9. Chapter 3 summary: rewrite to match the actual three-stage pipeline

**Where:** Chapter 3, "Chapter Summary" section.

The current summary describes a "four-stage pipeline" with a standalone 3D
Convolution Module and a separate Temporal Attention Module. Neither matches the
actual System Framework (Section 3.3), which has three stages and no standalone
temporal module.

Rewrite the summary to accurately describe the three stages:
1. Input Volume Construction
2. Spatiotemporal Feature Encoding (which includes ResBlock3D — the 3D convolution
   is inside this stage, not a standalone stage)
3. Spatiotemporal Cross-Attention and Upsampling

Do not refer to a separate "Temporal Attention stage."

---

### 10. Novelty claim: add a footnote distinguishing AnyUp3D from prior work

**Decision:** Keep the strong claim; add a footnote for qualification.

**Where:** Abstract, Chapter 1, Chapter 2, Chapter 3 — wherever the phrase "first
task-agnostic, temporally consistent feature upsampler" (or equivalent) appears.

Add a footnote at the first occurrence explaining what specifically distinguishes
AnyUp3D from the closest prior methods (e.g., FeatUp operates on single frames;
LIIF is a spatial-only continuous upsampler; video super-resolution methods are
task-specific and pixel-domain). The footnote does not need to be long — two to
three sentences that an examiner would find satisfying.

---

### 11. T=1 curriculum: fix Chapter 3 to explain the T=1 mechanism

**Decision:** T=1 was actually used — fix Chapter 3 to explain how.

**Where:** Chapter 3, the section that states "The minimum of T=2 is imposed by
VideoMAE's tubelet embedding."

VideoMAE's tubelet size 2 normally requires T≥2. If T=1 was used in the warm-up
stage, Chapter 3 must explain the mechanism — for example, whether a different
feature extractor was used for the T=1 stage, whether frames were duplicated to
satisfy the tubelet constraint, or whether a spatial-only forward pass was applied.

Update the minimum-T explanation to accurately describe what was done, so the
Chapter 4 description of "beginning with single-frame inputs (T=1)" is consistent
with and supported by the Chapter 3 implementation description.

---

### 12. Chapter 5 (Conclusion) is missing

**Status:** Deferred — will be written separately.

Chapter 1 states "Chapter Five concludes the work and offers recommendations for
future research." No chapter5.tex exists in the project. This must be written and
included before submission.

---

## Minor — Polish and Small Fixes

### 13. Unclosed backtick quotes in LaTeX (Chapter 1)

**Where:** Chapter 1 — any line where a word or phrase opens with ` `` ` but does
not close with ` '' `.

Find all unclosed opening backtick pairs and add the matching closing `''`. In
LaTeX, `''` produces the correct closing double quotation mark; an unclosed ` `` `
renders as a second opening quote.

---

### 14. Figure label: rename from `video_anyup_framework`

**Where:** Wherever `\label{fig:video_anyup_framework}` appears in Chapter 3.

Change to `\label{fig:anyup3d_framework}`. Search the entire project for
`\ref{fig:video_anyup_framework}` and update every occurrence to
`\ref{fig:anyup3d_framework}`.

---

### 15. Missing space before "such as" (Chapter 3, line ~282)

**Where:** Chapter 3, approximately line 282.

Change `"video processing backbonesuch as VideoMAE"` to
`"video processing backbone such as VideoMAE"`.

---

### 16. Table 2 (J&F): subset display — no change needed

**Decision:** Leave as is.

The table caption already states that per-clip scores are shown for a representative
subset and that the mean is computed over all 30 clips. No further action required.

---

### 17. Abstract: unclosed backtick quotes

**Where:** Abstract.

Same issue as #13. Find any ` `` ` that are not closed with ` '' ` and add the
missing closing pairs. Check "universal feature upsampling" and "AnyUp3D"
specifically.

---

## References

### A. Missing citations — add bib entries and `\cite{}` in text

Add the following bib entries and insert `\cite{}` at every point in the text where
these works are mentioned without a citation:

| Work | Citation key (suggested) | Reference |
|---|---|---|
| CLIP | `radford2021clip` | Radford et al., "Learning Transferable Visual Models From Natural Language Supervision," ICML 2021 |
| DINO | `caron2021dino` | Caron et al., "Emerging Properties in Self-Supervised Vision Transformers," ICCV 2021 |
| SigLIP | `zhai2023siglip` | Zhai et al., "Sigmoid Loss for Language Image Pre-Training," ICCV 2023 |
| UCF-101 | `soomro2012ucf101` | Soomro et al., "UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild," 2012 |
| Kinetics-400 | `kay2017kinetics` | Kay et al., "The Kinetics Human Action Video Dataset," 2017 |

CLIP is mentioned in the Abstract, Chapter 1, Chapter 2, and Chapter 3. DINO is
mentioned in Chapter 2. SigLIP appears in Chapter 1 and Chapter 2. UCF-101 is
written as prose in the Chapter 3 dataset section with no `\cite{}`. Kinetics-400
is mentioned in Chapter 4 in "pretrained on Kinetics-400."

---

### B. Wrong VideoMAE citation — switch from V2 to V1

The model used is `MCG-NJU/videomae-base` (ViT-B, tubelet size 2, patch size 16),
which is VideoMAE V1 (Tong et al., NeurIPS 2022), not VideoMAE V2.

- Replace all `\cite{wang2023videomae}` with `\cite{videomae}` throughout
  Chapters 3 and 4.
- Remove the `wang2023videomae` bib entry unless VideoMAE V2 is explicitly
  discussed somewhere (it is not).

---

### C. Bib entry quality fixes

Make the following corrections to existing bib entries:

**`upscale_video`** — two problems:
- `author = {Zhou, et al.}` is invalid BibTeX. Replace with real authors:
  Zheyuan Chen, Yabo Zhang, Xiaopeng Sun, et al.
- `journal = {CVPR}` is wrong. Change to `@inproceedings` with
  `booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition}`.

**`cutie`** — same author problem:
- `author = {Cheng, et al.}` — replace with real authors:
  Ho Kei Cheng, Seoung Wug Oh, Brian Price, Joon-Young Lee, Alexander Schwing.

**`carafe`** — conference listed as journal:
- `journal = {ICCV}` — change to `@inproceedings` with
  `booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision}`.

**`liif`** — arXiv citation for a published conference paper:
- Change to `@inproceedings`, set `booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition}`, `year = {2021}`.

**`featup`** — wrong first author and missing venue:
- `author = {Fu, Daniel Y. and Zhang, Qihang and others}` — the actual first
  author is Mark Hamilton, not Daniel Fu. Fix the author field.
- FeatUp was published at ICLR 2024. Change from arXiv preprint to
  `@inproceedings` with `booktitle = {International Conference on Learning Representations}`, `year = {2024}`.

---

### D. Remove unused bib entries

Delete the following entries from the `.bib` file — they are never cited and will
remain unused after the other fixes above are applied:

- `stereo_video` — never cited anywhere; leftover from an earlier draft.
- `youtubevos` — never cited; will remain unused after the YouTube-VOS → UCF-101
  fix in Issue 1.
- `wang2023videomae` — becomes unused after the V1/V2 switch in Refs B above.

---

## Priority Order Summary

| Priority | Issues |
|---|---|
| Fix immediately | 1, 2, 3, 4, 5 |
| Fix before submission | 6, 7, 8, 9, 11, 12 (write Ch. 5) |
| Fix for polish | 10, 13, 14, 15, 17 |
| No change needed | 16 |
| References | A, B, C, D — fix alongside the issues that introduce each citation |