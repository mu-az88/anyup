# Thesis Review Notes — AnyUp3D

Issues are grouped by severity. **Critical** = factually wrong or directly contradictory between chapters. **Unclear** = confusing, vague, or inconsistent framing that a reader or examiner will likely push back on. **Minor** = small fixes that improve polish.

---

## CRITICAL — Factual Errors / Cross-Chapter Contradictions

### 1. Training dataset is wrong in the Abstract and Chapter 1
- **Abstract** and **Chapter 1 (Objectives + Methodology)** say the model is trained on **YouTube-VOS**.
- **Chapter 3 and Chapter 4** clearly state training was done on **UCF-101**.
- YouTube-VOS is never used anywhere in the actual work. This is a major factual error that will be caught immediately by any reader.
- **Fix:** Replace every mention of YouTube-VOS in the Abstract and Chapter 1 with UCF-101.

### 2. T-curriculum numbers conflict between Chapter 3 and Chapter 4
- **Chapter 3 table** shows three stages: T=2 at step 0, T=4 at step 5,000, T=8 at step 15,000.
- **Chapter 4** describes stages as: T=1 → T=4 → T=8 → T=16.
- Also, **training ran for only 6,000 steps** (stated in Chapter 4), which means the step 15,000 trigger for T=8 was never reached. So either the curriculum table in Chapter 3 is wrong, or Chapter 4's description of the curriculum is wrong.
- This needs to be reconciled — one of them reflects what actually happened.

### 3. DINOv2, CLIP, and SigLIP are described as actual experimental backbones but were never used
- **Chapter 2** explicitly says DINOv2 is "one of the three primary backbones used in the AnyUp3D experimental pipeline, alongside CLIP and SigLIP."
- **Chapter 4** uses only VideoMAE. DINOv2, CLIP, and SigLIP appear nowhere in the experiments.
- This is a false statement of fact. Either the experiments need to include those backbones, or Chapter 2 must be corrected to say these are *example* backbones the architecture *supports*, not ones it was tested with.

### 4. "Flow warping error" metric promised but never computed
- **Abstract**: states results will include "flow warping error" as a temporal stability metric.
- **Chapter 4**: only reports cosine similarity for temporal consistency — no flow warping error anywhere.
- Either compute and report it, or remove it from the Abstract.

### 5. Chapter 3's Evaluation Plan describes Chapter 4's results as "planned future work"
- The Evaluation Plan section (end of Chapter 3) describes the DAVIS-2017 downstream J&F evaluation as something that "will be conducted following training completion."
- Chapter 4 actually presents those results. The language in Chapter 3 is still written in future tense as if the results don't exist yet.
- **Fix:** Rewrite the Planned Downstream Evaluation subsection in Chapter 3 to say "is presented in Chapter 4" rather than describing it as planned.

---

## UNCLEAR — Confusing or Misleading Framing

### 6. Abstract uses future tense ("We anticipate...") for completed work
- "We anticipate that AnyUp3D will demonstrate significant improvements..." — the results are already in Chapter 4.
- The abstract should report what was found, not what was anticipated.
- **Fix:** Change to past tense: "AnyUp3D demonstrated significant improvements..."

### 7. "Temporal Attention module" is described as a separate component, but it does not exist as a standalone module
- Chapters 1, 2, and 3's summary all describe the architecture as having two distinct novel modules: a **3D Convolution Module** and a **Temporal Attention Module**.
- But looking at the actual implementation in Chapter 3, there is no separate "Temporal Attention" module. The temporal extension is baked into the **3D windowed cross-attention** (3D mask + 3D convolutions in the encoder). The word "Temporal Attention" refers to a theoretical concept in Chapter 2 (attention along the time axis) but that exact module is not what was implemented.
- This is confusing because a reader will look for a "Temporal Attention Module" in the architecture and not find it as a discrete component.
- **Fix:** Either (a) clarify that temporal attention is *integrated* into the 3D cross-attention block rather than being a separate module, or (b) clearly map the theoretical "Temporal Attention" from Chapter 2 to the specific implementation component in Chapter 3.

### 8. Chapter 2's description of Temporal Attention does not match what was implemented
- **Chapter 2** defines temporal attention as: a query at position (x,y) in frame t attending to keys at the same (x,y) position across all T frames. This is a pure temporal self-attention with no spatial cross-attention.
- **What was actually implemented**: a 3D windowed *cross*-attention where each HR query token attends to LR key tokens within a spatiotemporal 3D window — a very different mechanism.
- The theoretical grounding in Chapter 2 does not match the engineering in Chapter 3.

### 9. Chapter 3 "Chapter Summary" describes a different architecture than what was built
- The summary says the system is a "four-stage pipeline: frozen backbone feature extraction, local spatio-temporal modeling via a 3D Convolution module, arbitrary-scale spatial upsampling via the unchanged AnyUp Window Attention core, and cross-frame consistency refinement via a Temporal Attention module."
- But the actual System Framework (Section 3.3) only has **three stages**: Input Volume Construction, Spatiotemporal Feature Encoding, and Spatiotemporal Cross-Attention and Upsampling.
- Also, the 3D Convolution is not a standalone stage — it is inside the encoder (ResBlock3D). And there is no separate "Temporal Attention" stage.

### 10. "First task-agnostic, temporally consistent feature upsampler" claim is unsupported
- This claim appears in the Abstract, Chapter 1, Chapter 2, and Chapter 3. It is a very strong novelty claim.
- No evidence or citation is provided to confirm that no prior work has done this. If an examiner finds a counterexample, the thesis is weakened.
- **Fix:** Soften to "one of the first" or add a footnote explaining what specifically distinguishes this from prior methods that might qualify.

### 11. Chapter 4 says T=1 is the starting curriculum point, but Chapter 3 says T=2 is the minimum
- **Chapter 3**: "The minimum of T=2 is imposed by VideoMAE's tubelet embedding, which requires at least one complete temporal tube."
- **Chapter 4**: "beginning with single-frame inputs (T=1) to warm up the spatial upsampling pathway."
- These directly contradict each other. If T=1 is truly impossible with VideoMAE, Chapter 4 is wrong. If T=1 was used, Chapter 3's explanation of the minimum is wrong.

### 12. Missing Chapter 5
- **Chapter 1** states: "Chapter Five concludes the work and offers recommendations for future research."
- There is no `chapter5.tex` file in the project. Chapter 5 (Conclusion) is missing entirely from the thesis source.

---

## MINOR — Polish and Small Fixes

### 13. Unclosed backtick quotes in LaTeX (Chapter 1)
- Lines like `` ``AnyUp3D system`` and `` ``universal efficiency`` open with ` `` ` but never close with `''`. In LaTeX this renders incorrectly — the closing quote will look like opening quotes.
- **Fix:** Replace all `` ``word `` patterns with `` ``word'' ``.

### 14. Figure label still says `video_anyup_framework` (Chapter 3)
- `\label{fig:video_anyup_framework}` — the old project name is in the label. While this doesn't break compilation, it's inconsistent.
- **Fix:** Change to `\label{fig:anyup3d_framework}` and update any `\ref{}` that points to it.

### 15. Missing space before "such as" in Chapter 3 (line ~282)
- "video processing backbone**such as** VideoMAE" — missing space before "such as."

### 16. Chapter 4: only 9 of 30 clips shown in Table 2 (J&F), but the mean covers all 30
- The table caption says "per-clip scores shown for a representative subset; the mean is computed over all 30 clips." This is fine, but it would be stronger to include the full table in an appendix so the claim is fully verifiable.
- No big fix needed, but worth acknowledging.

### 17. Abstract: backtick quotes not closed
- Same issue as #13 — `` ``universal feature upsampling`` and `` ``AnyUp3D `` are opened but not closed with `''`.

---

---

## REFERENCES REVIEW

### A. Missing citations — mentioned in text with no `\cite{}`

| What | Where | Who to cite |
|------|-------|-------------|
| **CLIP** | Abstract, Ch.1, Ch.2, Ch.3 | Radford et al. (2021), "Learning Transferable Visual Models From Natural Language Supervision", ICML 2021 |
| **DINO** (original) | Ch.2 §2.1.5 — "DINO is a self-supervised learning method…" | Caron et al. (2021), "Emerging Properties in Self-Supervised Vision Transformers", ICCV 2021 |
| **SigLIP** | Ch.1, Ch.2 | Zhai et al. (2023), "Sigmoid Loss for Language Image Pre-Training", ICCV 2023 |
| **UCF-101** | Ch.3 dataset section — "Soomro et al. (2012)" is written out as prose but there is no `\cite{}` and no bib entry | Soomro et al. (2012), "UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild" |
| **Kinetics-400** | Ch.4 — "pretrained on Kinetics-400" | Kay et al. (2017), "The Kinetics Human Action Video Dataset" |

These five are the most obvious gaps — they're all proper scientific works or datasets being relied on.

---

### B. Wrong citation — VideoMAE V1 vs V2

This is a subtle but real error. The bib file has **two separate VideoMAE entries**:

- `videomae` → VideoMAE **V1** by Tong et al., NeurIPS 2022 ("VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training")
- `wang2023videomae` → VideoMAE **V2** by Wang et al., CVPR 2023 ("VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking")

The thesis consistently cites `wang2023videomae` (V2) throughout Ch.3 and Ch.4. But the actual model used is `MCG-NJU/videomae-base` (ViT-B, tubelet size 2, patch size 16), which is the **V1** model from the `videomae` entry. VideoMAE V2 is a different, larger-scale paper.

**Fix:** Replace all `\cite{wang2023videomae}` with `\cite{videomae}`, and remove or keep `wang2023videomae` only if V2 is specifically discussed somewhere.

---

### C. Bib entry quality problems

**1. `upscale_video` — two problems:**
- `author = {Zhou, et al.}` — "et al." is not valid BibTeX syntax. It needs real names. The actual authors are Zheyuan Chen, Yabo Zhang, Xiaopeng Sun et al.
- `journal = {CVPR}` — CVPR is a conference, not a journal. Should be `@inproceedings` with `booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition}`.

**2. `cutie` — same "et al." author problem:**
- `author = {Cheng, et al.}` — needs real authors. The Cutie paper is by Ho Kei Cheng, Seoung Wug Oh, Brian Price, Joon-Young Lee, Alexander Schwing.

**3. `carafe` — conference listed as journal:**
- `journal = {ICCV}` — should be `@inproceedings` with `booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision}`.

**4. `liif` — arXiv citation for a published conference paper:**
- Listed as `arXiv preprint arXiv:2012.09161, year={2020}`, but LIIF was published at **CVPR 2021**. Using the arXiv date and not the publication venue is technically incorrect for a published work.
- **Fix:** Change to `@inproceedings` with `booktitle = {CVPR}`, `year = {2021}`.

**5. `featup` — wrong first author + arXiv only:**
- `author = {Fu, Daniel Y. and Zhang, Qihang and others}` — the actual first author of FeatUp is **Mark Hamilton**, not Daniel Fu. This is a significant error.
- FeatUp was published at **ICLR 2024**, not just arXiv. Should reflect the venue.

---

### D. Unused bib entries (should be removed)

- `stereo_video` — never cited anywhere in the thesis. Looks like a leftover from an earlier draft. Remove it.
- `youtubevos` — never cited, and once the YouTube-VOS → UCF-101 fix from the main review is applied, it will remain unused. Remove it, or cite it where YouTube-VOS is mentioned if that text is intentionally kept.
- `wang2023videomae` — once corrected to cite `videomae` instead (see point B above), this entry becomes unused and should be removed to avoid confusion.

---

### References Quick Priority

| Priority | Fix |
|----------|-----|
| Fix now | Add bib entries for CLIP, DINO, SigLIP, UCF-101, Kinetics-400 and add `\cite{}` in text |
| Fix now | Switch all `\cite{wang2023videomae}` → `\cite{videomae}` (V1 is what was used) |
| Before submission | Fix `featup` author (Hamilton, not Fu) |
| Before submission | Fix `upscale_video` and `carafe` to use `@inproceedings` with proper `booktitle` |
| Before submission | Fix `upscale_video` and `cutie` author fields (remove "et al.") |
| Before submission | Update `liif` to reflect CVPR 2021 publication |
| Cleanup | Remove `stereo_video`, `youtubevos`, `wang2023videomae` unused entries |

---

## Summary Priority Order

| Priority | Items |
|----------|-------|
| Fix immediately (factually wrong) | 1 (YouTube-VOS → UCF-101), 2 (curriculum mismatch), 3 (DINOv2/CLIP/SigLIP), 4 (flow warping error), 5 (evaluation plan tense) |
| Fix before final submission (confusing to examiner) | 6 (future tense abstract), 7 & 8 (Temporal Attention mismatch), 9 (chapter summary wrong), 11 (T=1 vs T=2), 12 (missing Ch. 5) |
| Fix for polish | 10 (soften novelty claim), 13 & 17 (backtick quotes), 14 (label name), 15 (missing space), 16 (full table) |
