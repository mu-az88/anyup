# AnyUp3D: Project Context (always loaded)

## What This Project Is

AnyUp3D extends **AnyUp** (resolution-agnostic 2D feature upsampling) to **video**, adding temporal consistency. It uses 3D/temporal components: a temporal kernel `t_k=3` and windowed 3D cross-attention. The goal is upsampled video features that are both spatially faithful and temporally stable.

## Status

- The M.Sc. thesis "AnyUp3D" is accepted and the exam passed (Oct 2026). Thesis work is closed.
- The current goal is the **strongest possible paper**, following the supervisor's suggestions. Venue and scope (one paper or several) are still to be decided.
- The plan is about 6 weeks of technical work, then 2 weeks of writing and review. It is tiered: every milestone leaves the project publishable.
- Order of work: **experiments** (current) → training-pipeline fixes → scaling data and backbones.

## Environment

- Google Colab (T4 / A100); Google Drive for checkpoints.
- Repo: `https://github.com/mu-az88/anyup.git`, branch `3Dconv`, cloned to `/content/anyup`.
- Checkpoints: `run_02/anyup3d_step6500.pth` (or step6000, **unresolved**), `run_01/anyup3d_step10000.pth`.
- Datasets: UCF-101 (training), DAVIS-2017 (evaluation), ADE20K (spatial benchmark).
- Backbones: VideoMAE (`MCG-NJU/videomae-base`, the training backbone), Video Swin-T (torchvision, Kinetics-400), DINOv2 (timm via `anyup/backbone/vit_wrapper.py`).
- Experiments: Exp 1 = ADE20K linear-probe mIoU (DINOv2, T=1); Exp 2 = DAVIS temporal cosine (VideoMAE); Exp 3 = DAVIS label-propagation J&F (Swin-T per-frame).

## Implementation Gotchas (do not break these)

- AnyUp call signature: `model(guide, features, output_size=(H, W), q_chunk_size=...)`. **Guide comes first.**
- Checkpoint key is `"anyup"`, not `"model"`. `torch.load(..., weights_only=False)` is required (pickled `TStage` object).
- `kernel_size_lfu=5` is mandatory; `t_k=3` is the temporal kernel size throughout.
- `window_t=4` (as in training). It's not stored in the checkpoint, so a wrong value loads silently.
- Guides are **ImageNet-normalized** (as in training), for both AnyUp2D and AnyUp3D.
- Import AnyUp2D from `anyup.model_2d`, **never via `torch.hub`** (the hub loads the 3D class).
- Pass **raw unnormalized features** to both AnyUp2D and AnyUp3D; they normalize internally.
- AnyUp3D cross-attention can OOM; pass `q_chunk_size` to process queries in chunks.
- natten is **not** used in the forward pass. `CrossAttention3D` uses `nn.MultiheadAttention` with a hand-built boolean window mask (`compute_attention_mask_3d`).
- VideoMAE `last_hidden_state` (1, 1568, 768) → reshape (8, 14, 14, 768) → permute (8, 768, 14, 14).
- Losses and augmentations live in `anyup.data.training` (not `anyup.training`).
- PyTorch `Conv3d` has no `padding_mode="reflect"`; use the `SpatialReflectConv3d` wrapper (reflect in H/W, replicate in T).
- DAVIS: download from `https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip`. `data.davis-challenge.ch` is unreachable from Colab.

## Thesis Results (baseline to reproduce)

- Cosine fidelity: 0.9899 (step 250) → 0.9913 (step 6,000) (source notebook unconfirmed).
- Exp 1: ADE20K mIoU (ran with `window_t=3`, step500; needs a re-run).
- Exp 2: temporal cosine similarity +0.081 across 30 DAVIS-2017 clips.
- Exp 3: weighted J&F 0.266 → 0.347 (+0.081); 27/30 clips improved; zero-shot on Video Swin-T. Settings: `TAU=0.07`, `TOP_K=7`, `N_LAST_FRAMES=6`; exclude frame 0; official DAVIS boundary metric (`bound_th=0.008`).
- Exp 4: 1,540,736 params (5.88 MB), +75.5% vs AnyUp2D; about 2.7× slower and about 10.5× more memory.

## Open Questions

- Which checkpoint is canonical: step6000 or step6500?
- The sign-test p ≈ 9.3×10⁻¹⁰ is quoted for a 30/30 result, but Exp 3 reports 27/30. Which result does it belong to?
- SAM 2 is cited as a key baseline but was never evaluated. Evaluate it or remove it.
- Which notebook produced the cosine fidelity numbers? Where is the Exp 4 notebook?
