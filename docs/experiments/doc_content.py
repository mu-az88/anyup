"""Hand-written text for docs/experiments/experiments.html (T09). Read by build_docs.py.

Everything generated (code, config values, helper signatures, links, results) comes from the notebooks
and the source. This file holds only what the code cannot say: purpose, data flow with shapes, a short
description per notebook cell, calls the parser cannot see (`implicit`), protocol changes vs the thesis,
thesis reference values, and known issues. Facts here were checked against the code at T09 (see the task log).

Inline markup: `code` → <code>; nothing else.
"""

TITLE = "AnyUp3D experiments"
SUBTITLE = ("Documentation of the four evaluation notebooks (Exp 0–3): what each measures, how it runs, "
            "and the shared code it calls.")

OVERVIEW = [
    ("All experiments share one pipeline: a frozen backbone produces low-resolution features and an "
     "ImageNet-normalized guide image (`anyup.backbones`); an upsampler maps the features to the guide "
     "resolution (`anyup.upsamplers`); a metric scores the result (`anyup.metrics`). Every comparison uses the "
     "same backbone output for all upsamplers, so the upsampler is the only variable."),
    ("Three upsamplers are compared throughout: bilinear interpolation, the published AnyUp (2D), applied "
     "frame by frame, and AnyUp3D, applied to a chunk of frames in one call. AnyUp3D was trained on VideoMAE "
     "features only; results on DINOv2 (Exp 1) and Video Swin-T (Exp 3) are therefore zero-shot."),
]

OVERVIEW_TABLE = {
    "columns": ["", "Question", "Backbone", "Data", "Metric", "Conditions"],
    "rows": [
        ["exp0", "Inference cost", "none (synthetic VideoMAE-shaped inputs)", "—",
         "time (ms), peak GPU memory (MiB), parameters", "bilinear, AnyUp2D, AnyUp3D"],
        ["exp1", "Spatial quality on single images (no regression)", "DINOv2-base (timm), T = 1",
         "ADE20K, 20,210 train / 2,000 val", "linear-probe mIoU, 150 classes", "bilinear, AnyUp2D, AnyUp3D"],
        ["exp2", "Temporal consistency", "VideoMAE-base", "DAVIS-2017 val, 30 clips, all frames",
         "adjacent-frame cosine similarity", "raw, bilinear, AnyUp2D, AnyUp3D"],
        ["exp3", "Downstream video task, no training", "Video Swin-T (Kinetics-400), per frame",
         "DAVIS-2017 val, 30 clips, all frames", "official J&F (label propagation)", "bilinear, AnyUp2D, AnyUp3D"],
    ],
}

REPRODUCIBILITY = [
    ("Each notebook clones the repository at a fixed branch on Colab (Cell 1), sets "
     "`CUBLAS_WORKSPACE_CONFIG` before CUDA starts, and seeds Python, NumPy and PyTorch with deterministic "
     "kernels (`set_seed`). Exp 0 also measures a non-deterministic `fast` mode."),
    ("Results are written by `save_results`: a per-item CSV (clip, image or measurement) and a summary JSON "
     "holding the summary, the full config and the environment (git commit and dirty flag, GPU, CUDA, package "
     "versions, checkpoint SHA-256)."),
]

EXPERIMENTS = {
    # ── Exp 0 ────────────────────────────────────────────────────────────────
    "exp0": {
        "title": "Exp 0 · Efficiency",
        "nav": "Exp 0 · Efficiency",
        "provenance": "T08 (renamed from Exp 4)",
        "purpose": [
            ("Exp 0 measures the inference cost of each upsampler: parameter count, weight size, run time and "
             "peak GPU memory of one forward pass under `no_grad`. The cost does not depend on image content, so "
             "the inputs are synthetic tensors of the shapes used in Exp 2, and no dataset or backbone is loaded."),
            ("The main row is the Exp 2 setting (224² output, 8 feature frames, AnyUp3D query chunk 4096), "
             "measured once with non-deterministic kernels (`fast`) and once with the deterministic seeding of "
             "Exp 1–3. A sweep over output size, frames and query chunk shows how the cost scales; rows that run "
             "out of memory are recorded as OOM."),
        ],
        "flow": [
            ("synthetic features", "(T, 768, 14, 14)"),
            ("synthetic guide", "(T, 3, S, S)"),
            ("upsampler", "one call per chunk"),
            ("output", "(T, 768, S, S)"),
            ("CUDA events + peak memory", "median, IQR, MiB"),
        ],
        "flow_note": "Main row: T = 8, S = 224, q = 4096. Sweep: S ∈ {56, 112, 224}, T ∈ {4, 8, 16}, AnyUp3D q ∈ {None, 1024, 4096}.",
        "steps": {
            1: "Sets `CUBLAS_WORKSPACE_CONFIG`, mounts Drive, clones or switches `/content/anyup` to `REPO_BRANCH` and installs the package. Identical in all four notebooks.",
            2: "GPU profile (warm-up and repeat counts), the main setting, the sweep grid and the thesis reference values.",
            3: "Loads the three upsamplers, counts parameters (fp32 size) and checks the AnyUp3D count against the thesis value of 1,540,736. `set_mode` switches between `fast` and `deterministic` seeding.",
            4: "Builds synthetic inputs from a fixed seed and times one call: warm-up calls, then repeated calls timed with CUDA events. Peak memory is read after resetting the peak counter; the extra peak is the peak minus the memory in use before the calls. An out-of-memory error is recorded as `OOM`.",
            5: "Runs the main row (`fast`), the sweep from small to large inputs, and the main row again (`deterministic`). Checks that every output has shape (T, 768, S, S) and that the extra memory of bilinear is within 10 % of its output tensor.",
            6: "Computes the AnyUp3D / AnyUp2D ratios (time, extra and total peak memory, parameters) per mode, prints them next to the thesis claims, and saves `exp0_per_item.csv` and `exp0_summary.json`.",
            7: "Optional plot of per-frame time against output size, one panel per frame count.",
        },
        "implicit": {
            4: ["Upsampler.forward", "BilinearUpsampler._upsample", "AnyUp2DUpsampler._upsample",
                "AnyUp3DUpsampler._upsample", "AnyUp2D.forward", "AnyUp.forward"],
        },
        "protocol": [
            "The thesis settings for the time and memory ratios (GPU, resolution, frames, query chunk) were not recorded; Exp 0 re-measures them under documented settings, so the thesis ratios are shown for reference only.",
            "Timing uses CUDA events with warm-up, reported as median and interquartile range. Memory is reported as total peak and as extra peak over the memory in use before the call; ratios use the extra peak.",
        ],
        "thesis": [
            ("AnyUp3D parameters", "1,540,736 (5.88 MB)"),
            ("Parameters vs AnyUp2D", "+75.5 %"),
            ("Run time vs AnyUp2D", "≈ 2.7×"),
            ("Memory vs AnyUp2D", "≈ 10.5×"),
        ],
        "issues": [
            ("S07", "Only the T4 run is planned; the A100 run (the GPU of the full Exp 1 run) is skipped for now."),
            ("S06", "The checkpoint is step6500; the choice of the paper's canonical checkpoint is open."),
        ],
        "results_files": ["exp0_summary.json", "exp0_per_item.csv"],
        "drive_dir": "results/T08/exp0/<RUN_NAME>/  (e.g. T4_main_s224_f8_q4096_sweep)",
    },
    # ── Exp 1 ────────────────────────────────────────────────────────────────
    "exp1": {
        "title": "Exp 1 · ADE20K linear probe",
        "nav": "Exp 1 · ADE20K probe",
        "provenance": "T06 refactor, T07 clean-up",
        "purpose": [
            ("Exp 1 checks that AnyUp3D does not lose spatial quality on single images. DINOv2-base features of "
             "ADE20K images (T = 1) are upsampled to 224², and a linear 1×1 convolution is trained on them for each "
             "upsampler; the metric is mIoU over the 150 classes at 224²."),
            ("The probe is deliberately weak, so differences in mIoU reflect the upsampled features rather than "
             "the head. Each seed gives every upsampler the same head initialization and the same data order. "
             "AnyUp3D never saw DINOv2 features or still images in training, so this test is zero-shot."),
        ],
        "flow": [
            ("image", "resized to 224² (squashed)"),
            ("DINOv2-base", "(B, 768, 16, 16), fp16 cache"),
            ("upsampler.forward_images", "(B, 768, 224, 224)"),
            ("1×1 conv head", "(B, 150, 224, 224)"),
            ("mIoU", "label 0 ignored"),
        ],
        "flow_note": "Masks are resized with nearest neighbour to 224². `forward_images` treats the batch as unrelated images (each a one-frame clip), never as consecutive frames.",
        "steps": {
            1: "Same setup cell as the other notebooks.",
            2: "GPU profile (A100 = full protocol; T4 = a small test profile whose numbers are not for reporting), data subset, image size and optimizer settings.",
            3: "Extracts ADE20K from the Drive zip and lists image–mask pairs. DINOv2-base features are extracted once into an fp16 cache, built on local disk, mirrored to Drive and reused across sessions; the backbone is then freed and the three upsamplers are loaded.",
            4: "Defines the probe: a 1×1 convolution from 768 channels to 150 classes, trained with AdamW and a step learning-rate schedule. Each (condition, seed) run evaluates on the validation set after every epoch, checkpoints the head to Drive so a disconnected run resumes, and records per-image mIoU in the last epoch.",
            5: "Loops over conditions and seeds; finished runs are skipped when the cell is re-run.",
            6: "Per condition: mIoU per seed, mean ± std and mean per-image mIoU. Paired statistics of AnyUp3D against each baseline over images (per-image mIoU averaged over seeds; images without labelled pixels dropped) and over seeds. Saves `exp1_per_item.csv` and `exp1_summary.json`.",
            7: "Optional figure of the trained heads' predictions on a few validation images.",
        },
        "implicit": {
            3: ["BackboneAdapter.forward", "DINOv2Adapter._extract"],
            4: ["BilinearUpsampler._upsample", "AnyUp2DUpsampler._upsample_images",
                "AnyUp3DUpsampler._upsample_images", "AnyUp2D.forward", "AnyUp.forward"],
        },
        "protocol": [
            "The thesis run used the step500 checkpoint; Exp 1 now uses step6500.",
            "DINOv2 is loaded through the repository's timm wrapper instead of Hugging Face, and the training data order is seeded.",
            "The thesis run set `window_t=3` (training used 4); this has no effect at T = 1, since `window_t` only enters the temporal attention mask (verified in T06).",
            "Three seeds of three epochs each, with paired statistics over images and over seeds.",
        ],
        "thesis": [
            ("mIoU, bilinear", "39.54"),
            ("mIoU, AnyUp2D", "42.36"),
            ("mIoU, AnyUp3D (step500)", "41.93"),
        ],
        "issues": [
            ("S02", "The full A100 run of the current protocol has not been done yet."),
            ("S05", "The integrity of the ADE20K zip is not checked against a pinned hash."),
            ("S04", "The regression checks against the old code ran before the deterministic seeding of T07 and were not repeated after it."),
            ("S06", "The checkpoint is step6500; the choice of the paper's canonical checkpoint is open."),
        ],
        "results_files": ["exp1_summary.json"],
        "drive_dir": "results/T06/exp1/<RUN_NAME>/  (e.g. A100_step6500_s224_b16_q4096_trall_vaall_e3)",
    },
    # ── Exp 2 ────────────────────────────────────────────────────────────────
    "exp2": {
        "title": "Exp 2 · Temporal consistency",
        "nav": "Exp 2 · Temporal consistency",
        "provenance": "T05 refactor, T07 clean-up",
        "purpose": [
            ("Exp 2 measures temporal consistency: for each pair of adjacent feature frames, the cosine "
             "similarity between the two feature vectors at each pixel, averaged over pixels and then over the "
             "pairs of a clip. A temporally stable upsampler should not add flicker to the features it "
             "upsamples. The raw low-resolution VideoMAE features are reported as a reference."),
            ("VideoMAE accepts exactly 16 frames, so long clips are processed in chunks. Pairs that cross a "
             "chunk boundary are reported separately (`cross`) from pairs inside a chunk (`within`), since "
             "AnyUp3D sees each chunk independently."),
        ],
        "flow": [
            ("clip, all N frames", "(N, 3, 480, W)"),
            ("chunks of 16, overlap 8", "backbone context only"),
            ("VideoMAE-base", "(t, 768, 14, 14) + guide (t, 3, 224, 224)"),
            ("upsampler", "(t, 768, 224, 224)"),
            ("TemporalCosineStream", "float64, per pixel"),
            ("per clip", "all / within / cross"),
        ],
        "flow_note": "t ≤ 8 feature frames per chunk (VideoMAE tubelet 2). Overlapping outputs are kept from the earlier chunk, so a clip of N frames gives ⌈N/2⌉ feature frames, frames 0, 2, 4, … (checked per clip).",
        "steps": {
            1: "Same setup cell as the other notebooks.",
            2: "Clips, checkpoint, conditions, guide size, chunking and AnyUp3D query chunk.",
            3: "Seeds the run, downloads DAVIS-2017, lists the 30 validation clips, and loads VideoMAE (guide at `GUIDE_SIZE`) and the upsamplers.",
            4: "For each clip the backbone runs once: `iter_backbone_chunks` splits all frames into chunks of 16 with 8 frames of overlap. Each condition then streams over the chunks into a `TemporalCosineStream`, which marks the first pair of every later chunk as a cross-boundary pair, so no full-clip upsampled tensor is held in memory. A check confirms the kept feature frames are exactly frames 0, 2, 4, … of the clip.",
            5: "Mean ± std over clips of `all`, `within` and `cross` per condition, and paired statistics of AnyUp3D against each other condition on per-clip `all`. Saves `exp2_per_item.csv` and `exp2_summary.json`.",
            6: "Optional figure: the first three PCA components of the upsampled features of one chunk, one row per upsampler.",
        },
        "implicit": {
            4: ["BackboneAdapter.forward", "VideoMAEAdapter._extract", "Upsampler.forward",
                "BilinearUpsampler._upsample", "AnyUp2DUpsampler._upsample", "AnyUp3DUpsampler._upsample",
                "AnyUp2D.forward", "AnyUp.forward", "TemporalCosineStream.update", "TemporalCosineStream.result"],
        },
        "protocol": [
            "All frames of each clip are used; the thesis sampled 16 frames per clip (`linspace`).",
            "Clips are processed in VideoMAE chunks of 16 with 8 frames of backbone-only overlap, and the boundary pairs are reported separately.",
            "Frames are resized as tensors (the thesis used PIL; feature cosine ≈ 0.993 between the two), and the cosine is computed per pixel in float64.",
            "On the old protocol, the refactored code reproduced the old Exp 2 code exactly (T05 regression check), so the differences to the thesis come from the protocol changes above rather than from the implementation.",
        ],
        "thesis": [
            ("AnyUp3D − AnyUp2D, temporal cosine", "+0.081 (30 clips)"),
        ],
        "issues": [
            ("Phase B", "AnyUp3D's similarity drops at chunk boundaries (mean within − cross ≈ 0.032 in the T05 run) because each chunk is upsampled without context from the previous one. Whether to give AnyUp3D an upsampler-side overlap is open."),
            ("S04", "The regression checks against the old code ran before the deterministic seeding of T07 and were not repeated after it."),
            ("S06", "The checkpoint is step6500; the choice of the paper's canonical checkpoint is open."),
        ],
        "results_files": ["exp2_summary.json"],
        "drive_dir": "results/T05/exp2/<RUN_NAME>/  (e.g. g224_c16_o8_q4096_n30)",
    },
    # ── Exp 3 ────────────────────────────────────────────────────────────────
    "exp3": {
        "title": "Exp 3 · Label propagation (video object segmentation)",
        "nav": "Exp 3 · Label propagation",
        "provenance": "T05 refactor, T07 clean-up",
        "purpose": [
            ("Exp 3 tests whether more consistent features help a downstream video task without any training. "
             "In semi-supervised video object segmentation by non-parametric label propagation (Jabri et al. "
             "2020; DINO protocol), the ground-truth mask of frame 0 is carried to every later frame through "
             "feature affinities with frame 0 and the six preceding frames. The predictions are scored with the "
             "official DAVIS-2017 J&F (region similarity J and boundary accuracy F)."),
            ("Video Swin-T, run frame by frame, was never seen in AnyUp3D training, and the propagation has no "
             "learned component. The absolute J&F is therefore a lower bound; the comparison between upsamplers "
             "on identical backbone features is the result."),
        ],
        "flow": [
            ("clip, all T frames", "(T, 3, 480, W)"),
            ("Swin-T per frame, 224", "(t, 768, 7, 7) + guide (t, 3, 56, 56)"),
            ("upsampler", "(t, 768, 56, 56)"),
            ("concatenate chunks", "(T, 768, 56, 56)"),
            ("propagate_labels", "τ 0.07, top-k 7, frame 0 + last 6"),
            ("PNG at native size", "(T, 480, W)"),
            ("eval_clip_jf → summarize_jf", "J, F, J&F"),
        ],
        "flow_note": "Chunks of 16 frames without overlap: Swin-T runs per frame (each frame duplicated into a two-frame clip), so overlap would not change any feature. The first and last frames are not scored (official protocol).",
        "steps": {
            1: "Same setup cell as the other notebooks.",
            2: "Clips, checkpoint, conditions, Swin-T input normalization, propagation size, chunking and propagation settings.",
            3: "Seeds the run, downloads DAVIS-2017, loads Swin-T in per-frame mode with Kinetics normalization (guide at `PROP_SIZE`) and the upsamplers, and clears this run's mask folder.",
            4: "For each clip the backbone runs once in chunks of 16. Each condition upsamples every chunk to 56², concatenates them and propagates the frame-0 labels with `propagate_labels`. Predictions are written as palette PNGs at native resolution in the official results layout and read back, so the score is computed on exactly what the DAVIS toolkit would read.",
            5: "`summarize_jf` aggregates over objects as the official toolkit does; paired statistics of AnyUp3D against each baseline use per-clip J&F. Saves `exp3_per_item.csv`, `exp3_summary.json` and the zipped masks.",
            6: "Optional stills and a side-by-side comparison video for a few clips.",
        },
        "implicit": {
            4: ["BackboneAdapter.forward", "SwinTAdapter._extract", "Upsampler.forward",
                "BilinearUpsampler._upsample", "AnyUp2DUpsampler._upsample", "AnyUp3DUpsampler._upsample",
                "AnyUp2D.forward", "AnyUp.forward"],
        },
        "protocol": [
            "All frames of each clip are used; the thesis used every 12th frame.",
            "AnyUp3D receives the ImageNet-normalized guide it was trained with; the thesis passed raw [0, 1] frames.",
            "Swin-T inputs use the Kinetics statistics of its weights; the thesis used ImageNet statistics.",
            "Scoring follows the official DAVIS toolkit: first and last frame not scored, per-object aggregation, toolkit boundaries. On the same masks the thesis metric gave about +0.02 J&F more.",
            "Propagation runs on float32 features; the thesis propagated float16-cached features.",
            "On the old protocol, the refactored code reproduced the old Exp 3 upsampled features and masks exactly (T05 regression check).",
        ],
        "thesis": [
            ("J&F, bilinear → AnyUp3D", "0.266 → 0.347 (+0.081)"),
            ("Clips improved", "27 / 30"),
        ],
        "issues": [
            ("Phase B", "On `drift-straight`, propagation with AnyUp3D features collapses (J&F 0.033 vs 0.297 for bilinear in the T05 run); the masks are kept for inspection."),
            ("S04", "The regression checks against the old code ran before the deterministic seeding of T07 and were not repeated after it."),
            ("S06", "The checkpoint is step6500; the choice of the paper's canonical checkpoint is open."),
        ],
        "results_files": ["exp3_summary.json"],
        "drive_dir": "results/T05/exp3/<RUN_NAME>/  (e.g. p56_c16_o0_qNone_kinetics_n30)",
    },
}
