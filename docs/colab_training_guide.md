# Running AnyUp 3D Training in Google Colab

This guide walks you through cloning the repo, installing dependencies, configuring training, and launching `train_3d.ipynb` inside Google Colab — from scratch to your first checkpoint.

---

## Table of Contents

1. [Runtime Setup](#1-runtime-setup)
2. [Clone the Repo](#2-clone-the-repo)
3. [Install Dependencies](#3-install-dependencies)
4. [Mount Google Drive (checkpoints & data)](#4-mount-google-drive-checkpoints--data)
5. [Open `train_3d.ipynb`](#5-open-train_3dipynb)
6. [Cell-by-Cell Configuration Guide](#6-cell-by-cell-configuration-guide)
   - [Cell 1 — Imports & Paths](#cell-1--imports--paths)
   - [Cell 2 — TrainConfig (main config block)](#cell-2--trainconfig-main-config-block)
   - [Cell 3 — Reproducibility & Device](#cell-3--reproducibility--device)
   - [Cell 4 — Video Encoder (Teacher)](#cell-4--video-encoder-teacher)
   - [Cell 5 — AnyUp 3D Model](#cell-5--anyup-3d-model)
   - [Cell 6 — Loss Functions](#cell-6--loss-functions)
   - [Cell 7 — Dataset & DataLoader](#cell-7--dataset--dataloader)
   - [Cell 8 — Optimizer](#cell-8--optimizer)
   - [Cell 9 — Checkpoint Helpers](#cell-9--checkpoint-helpers)
   - [Cell 10 — Training Loop](#cell-10--training-loop)
   - [Cell 11 — Quick Validation](#cell-11--quick-validation)
7. [Monitoring Training](#7-monitoring-training)
8. [Resuming a Run](#8-resuming-a-run)
9. [Colab-Specific Tips](#9-colab-specific-tips)

---

## 1. Runtime Setup

Before opening the notebook, make sure you have a GPU runtime:

1. In Colab, go to **Runtime → Change runtime type**
2. Set **Hardware accelerator** to **GPU** (T4 is free; A100 is available on Colab Pro)
3. Click **Save**

To verify after the runtime starts, the first notebook cell will print the GPU name. You should see something like:

```
GPU: Tesla T4
```

If it prints `cpu`, the runtime is not configured correctly — check step 1 again.

---

## 2. Clone the Repo

Run this in a Colab code cell before opening the notebook:

```python
!git clone https://github.com/YOUR_USERNAME/anyup.git
%cd anyup
```

> Replace `YOUR_USERNAME/anyup` with the actual repo path.  
> If the repo is private, use a personal access token:
> ```python
> !git clone https://YOUR_TOKEN@github.com/YOUR_USERNAME/anyup.git
> ```

To switch to the 3D training branch:

```python
!git checkout 3Dconv
```

---

## 3. Install Dependencies

```python
!pip install -r requirements.txt
```

This installs everything listed in `requirements.txt`:

| Package | Purpose |
|---|---|
| `torch`, `torchvision` | Core deep learning |
| `transformers` | VideoMAE teacher encoder |
| `tensorboard` | Training loss visualisation |
| `timm` | ViT backbone utilities |
| `eva-decord` | Fast video decoding |
| `hydra-core`, `rich` | Used by the CLI `train.py` (not the notebook) |

If you are on a Colab T4 and want to reduce memory usage, you can skip `hydra-core` and `rich` — they are not needed by the notebook.

---

## 4. Mount Google Drive (checkpoints & data)

Colab's local storage is wiped between sessions. Mount Drive to persist checkpoints and load your video dataset:

```python
from google.colab import drive
drive.mount('/content/drive')
```

After mounting, you can point the config to Drive paths:

```python
cfg.checkpoint_dir = "/content/drive/MyDrive/anyup3d/checkpoints"
cfg.log_dir        = "/content/drive/MyDrive/anyup3d/runs"
cfg.dataset_root   = "/content/drive/MyDrive/datasets/videos"
```

See [Cell 2](#cell-2--trainconfig-main-config-block) for where exactly to set these.

---

## 5. Open `train_3d.ipynb`

After cloning, open the notebook inside Colab:

1. In Colab, go to **File → Open notebook**
2. Select the **Google Drive** tab (if you cloned into Drive) or **Upload** to upload it
3. Alternatively, run this to open it directly from the repo directory:

```python
# This opens a file browser in the left panel — navigate to train_3d.ipynb
from google.colab import files
```

Or simply navigate to it in the Colab file browser on the left sidebar.

---

## 6. Cell-by-Cell Configuration Guide

The entire training run is controlled by **Cell 2** (`TrainConfig`). All other cells just use values from that config — you rarely need to edit them. The sections below describe what each cell does and what you might want to change.

---

### Cell 1 — Imports & Paths

```python
REPO_ROOT = Path(".").resolve()
```

**What it does:** Adds the repo root to `sys.path` so `import anyup` works.  
**What to change:** Nothing, unless you cloned the repo to a custom path — then set `REPO_ROOT` explicitly:

```python
REPO_ROOT = Path("/content/drive/MyDrive/anyup")
```

---

### Cell 2 — TrainConfig (main config block)

This is the **single source of truth** for all training parameters. Every option is listed below.

```python
cfg = TrainConfig()
```

#### Paths

| Field | Default | What it does |
|---|---|---|
| `model_ckpt_2d` | `"checkpoints/anyup_2d.pth"` | Path to pretrained 2D AnyUp weights. Set to `None` to train from random init. |
| `checkpoint_dir` | `"checkpoints/anyup3d"` | Where periodic checkpoints are saved. Point to Drive to persist across sessions. |
| `resume` | `None` | Path to a 3D checkpoint to resume from. Leave `None` for a fresh run. |
| `log_dir` | `"runs/anyup3d"` | TensorBoard log directory. |

Example override for Colab:
```python
cfg.model_ckpt_2d  = "/content/drive/MyDrive/anyup3d/anyup_2d.pth"
cfg.checkpoint_dir = "/content/drive/MyDrive/anyup3d/checkpoints"
cfg.log_dir        = "/content/drive/MyDrive/anyup3d/runs"
```

---

#### Video Encoder (Teacher)

| Field | Default | What it does |
|---|---|---|
| `encoder` | `"videomae"` | Which teacher architecture to use. Only `"videomae"` is implemented. |
| `encoder_model` | `"MCG-NJU/videomae-base"` | HuggingFace model ID. Switch to `"MCG-NJU/videomae-large"` for a stronger teacher at higher memory cost. |

---

#### Model Architecture

| Field | Default | What it does |
|---|---|---|
| `qk_dim` | `128` | Query/key projection dimension inside cross-attention. Lower = faster, less expressive. Minimum recommended: `64`. |
| `num_heads` | `4` | Number of attention heads. Must divide `qk_dim` evenly. |
| `window_ratio` | `0.1` | Fraction of spatial tokens inside each attention window. Higher = more context, more memory. |
| `kernel_size_lfu` | `5` | Spatial kernel size for the feature unification module. |
| `t_k` | `1` | Temporal kernel size for 3D convolutions. `1` = no temporal convolution (good for T=1 warmup). |
| `input_dim` | `3` | Input RGB channels. Do not change. |

---

#### Spatial Resolution

| Field | Default | What it does |
|---|---|---|
| `img_size` | `224` | Spatial size of the input video frames fed to AnyUp. Reduce to `112` to halve memory usage. |
| `crop_size` | `448` | High-res crop size used when extracting GT features from the teacher. Should be `2 × img_size`. |

---

#### T-Curriculum

Controls how the number of frames increases during training. Each stage activates at a given global step.

```python
cfg.t_schedule = [
    TStage(t=1,  start_step=0,     batch_size=8),
    TStage(t=4,  start_step=5000,  batch_size=4),
    TStage(t=8,  start_step=15000, batch_size=2),
    TStage(t=16, start_step=30000, batch_size=1),
]
```

| Field | What it does |
|---|---|
| `t` | Number of video frames for this stage. |
| `start_step` | Global training step at which this stage activates. |
| `batch_size` | Batch size for this stage. Halve it each time `t` doubles to keep memory roughly constant. |

**Memory tip for Colab T4 (15 GB):** Use smaller batch sizes or remove the last stage (`t=16`). A safe starting point:

```python
cfg.t_schedule = [
    TStage(t=1, start_step=0,     batch_size=4),
    TStage(t=4, start_step=5000,  batch_size=2),
    TStage(t=8, start_step=15000, batch_size=1),
]
```

---

#### Optimizer

| Field | Default | What it does |
|---|---|---|
| `lr` | `2e-4` | Peak learning rate for AdamW. Reduce to `5e-5` if training is unstable. |
| `weight_decay` | `1e-4` | AdamW weight decay. |
| `grad_clip_max_norm` | `1.0` | Maximum gradient norm. Prevents exploding gradients. Reduce if you see `NaN` loss. |

---

#### Loss Weights (λ values)

| Field | Default | What it does |
|---|---|---|
| `lambda_cos_mse` | `1.0` | Weight on the primary reconstruction loss (Cosine + MSE). Do not reduce below `0.5`. |
| `lambda_input_consistency` | `0.1` | Weight on input-space consistency. Prevents hallucinated structure. |
| `lambda_self_consistency` | `0.1` | Weight on augmentation self-consistency. Set to `0` to disable. |
| `lambda_temporal_consistency` | `0.2` | Weight on temporal smoothness between adjacent frames. Only active when `T > 1`. |
| `temporal_lambda_warmup_steps` | `5000` | Steps over which `lambda_temporal_consistency` linearly ramps from `0` to its full value. Set to `0` to skip warmup. |

---

#### Training Duration

| Field | Default | What it does |
|---|---|---|
| `max_steps` | `50000` | Total number of gradient steps. For a Colab smoke test, reduce to `1000`. |
| `save_every_n_steps` | `500` | How often to write a checkpoint. On Colab, make sure `checkpoint_dir` points to Drive or these are lost on session disconnect. |
| `val_every_n_steps` | `1000` | How often to run the validation loop. |
| `log_every_n_steps` | `50` | How often to log loss to TensorBoard and update the progress bar. |

---

#### Data

| Field | Default | What it does |
|---|---|---|
| `dataset_root` | `"data/videos"` | Root directory passed to `VideoDataset`. Set to your Drive path if using Drive. |
| `num_workers` | `4` | DataLoader worker processes. Set to `0` on Colab if you hit multiprocessing errors. |
| `pin_memory` | `True` | Pins CPU tensors to enable faster GPU transfer. Leave `True` when using a GPU runtime. |

---

#### Debug Mode

Set `cfg.debug = True` to run only 10 steps with `T=1` and `batch_size=1`. Use this to verify the full pipeline works before committing to a long run:

```python
cfg.debug = True        # activates debug mode
cfg.debug_steps = 10    # number of steps to run (default 10)
```

---

### Cell 3 — Reproducibility & Device

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**What to change:** Nothing. Device is set automatically.  
Change `cfg.seed` in Cell 2 if you want a different random seed.

---

### Cell 4 — Video Encoder (Teacher)

Loads the frozen VideoMAE teacher from HuggingFace. The model is downloaded on first run and cached in `~/.cache/huggingface/`.

**What to change:** Nothing directly — the `encoder_model` field in `TrainConfig` (Cell 2) controls which model is loaded.

If you want a different encoder architecture entirely (e.g. InternVideo, S3D), replace the body of this cell with your own loading code. The only requirement is that your encoder:
- Takes `pixel_values` shaped `(B, C, T, H, W)`, ImageNet-normalised
- Returns a `last_hidden_state` shaped `(B, N_tokens, C_enc)` where `N_tokens = T * H_tok * W_tok`

---

### Cell 5 — AnyUp 3D Model

Builds `AnyUp` from `anyup/model.py` using the architecture parameters from `TrainConfig`.

**Checkpoint loading order:**
1. If `cfg.resume` is set → load the full 3D checkpoint (model + optimizer + step). This takes priority.
2. Else if `cfg.model_ckpt_2d` is set and exists → load 2D pretrained weights with `strict=False`. Keys that do not match are silently skipped.
3. Else → random init.

**What to change:**
- Point `cfg.model_ckpt_2d` and/or `cfg.resume` to the right paths in Cell 2.
- If you see many `Missing keys` in the output after loading 2D weights, that is expected — the 3D model has new temporal modules that the 2D checkpoint does not have.

---

### Cell 6 — Loss Functions

Defines the loss functions and the temporal lambda ramp. No configuration needed here — all weights are read from `TrainConfig`.

If you want to **disable a loss term entirely**, set its lambda to `0.0` in Cell 2:

```python
cfg.lambda_self_consistency      = 0.0  # turn off augmentation consistency
cfg.lambda_temporal_consistency  = 0.0  # turn off temporal smoothness
```

---

### Cell 7 — Dataset & DataLoader

**This is the cell you are most likely to need to edit.**

The cell currently uses `StubVideoDataset`, which generates random tensors. This is fine for a smoke test (`cfg.debug = True`) but produces meaningless features during real training.

#### Replacing the stub with a real dataset

Replace the `StubVideoDataset` class and the `build_loader` function body with your actual dataset:

```python
# Example — replace with your real dataset class
from your_dataset_module import VideoDataset

def build_loader(t: int, batch_size: int, split: str = "train") -> DataLoader:
    dataset = VideoDataset(
        root=cfg.dataset_root,
        split=split,
        num_frames=t,
        img_size=cfg.img_size,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=cfg.num_workers if not cfg.debug else 0,
        pin_memory=cfg.pin_memory and device.type == "cuda",
        drop_last=True,
    )
```

**Requirements for `__getitem__`:**

The dataset must return a dict with at least:

| Key | Shape | dtype | Description |
|---|---|---|---|
| `video` | `(T, C, H, W)` | `float32` | ImageNet-normalised frames |
| `augmented_video` | `(T, C, H, W)` | `float32` | (Optional) augmented version of the same clip, for self-consistency loss |

If `augmented_video` is missing, the self-consistency loss is automatically skipped — no code change needed.

**DataLoader `num_workers` note:** Colab sometimes crashes with `num_workers > 0` due to shared memory limits. If you see `RuntimeError: DataLoader worker (pid ...) is killed by signal: Bus error`, set:

```python
cfg.num_workers = 0
```

---

### Cell 8 — Optimizer

Builds AdamW with a cosine learning rate decay schedule. All parameters are from `TrainConfig`.

**What to change:** Nothing directly. Use `cfg.lr`, `cfg.weight_decay`, and `cfg.grad_clip_max_norm` in Cell 2.

If you want a different schedule (e.g. linear warmup + cosine), replace the scheduler line:

```python
# Replace the CosineAnnealingLR with warmup + cosine:
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1000,
    num_training_steps=cfg.max_steps,
)
```

---

### Cell 9 — Checkpoint Helpers

Saves `anyup` model weights, optimizer state, scheduler state, and `global_step` to `cfg.checkpoint_dir`.

**What to change:** Nothing. Point `cfg.checkpoint_dir` to Drive in Cell 2 to ensure checkpoints survive session disconnects.

Checkpoints are saved as:
```
checkpoints/anyup3d/anyup3d_step5000.pth
checkpoints/anyup3d/anyup3d_step50000_final.pth
```

---

### Cell 10 — Training Loop

This is the main loop. It:
1. Checks whether to advance the curriculum stage at each step
2. Pulls a batch from the DataLoader (resetting the iterator when it is exhausted)
3. Extracts GT features from the frozen teacher
4. Runs AnyUp forward pass
5. Computes all four loss components
6. Backpropagates, clips gradients, steps optimizer and scheduler
7. Logs to TensorBoard every `log_every_n_steps` steps
8. Saves a checkpoint every `save_every_n_steps` steps

**What to change:**  
- All loop behaviour is driven by `TrainConfig` — edit Cell 2, not this cell.  
- If your encoder does not return a `last_hidden_state` attribute, update the `extract_teacher_features` helper (the function defined just before this cell).

---

### Cell 11 — Quick Validation / Smoke Test

Runs a single forward pass with a dummy batch to verify shapes are correct. Run this **before** launching Cell 10 to catch shape mismatches early.

```
Input  features : torch.Size([1, 768, 4, 14, 14])
Output features : torch.Size([1, 768, 4, 14, 14])
Validation passed.
```

If this cell throws an error, check that `ENCODER_DIM`, `TOKEN_H`, and `TOKEN_W` are consistent with your encoder's actual output shape.

---

## 7. Monitoring Training

### TensorBoard in Colab

Run this in a separate cell to launch TensorBoard inside the notebook:

```python
%load_ext tensorboard
%tensorboard --logdir {cfg.log_dir}
```

The following metrics are logged at every `log_every_n_steps` steps:

| Metric | Description |
|---|---|
| `loss/total` | Sum of all weighted loss terms |
| `loss/rec` | Primary reconstruction loss (Cosine + MSE) |
| `loss/aug` | Augmentation self-consistency loss |
| `loss/temporal` | Temporal smoothness loss |
| `lr` | Current learning rate |
| `curriculum/T` | Current number of frames |
| `curriculum/bs` | Current batch size |

### Progress bar

The `tqdm` progress bar in Cell 10 shows a live summary:

```
Training: 12%|██▌         | 6000/50000 [14:32<1:49:05, loss=0.0821, rec=0.0714, T=4, lr=1.94e-04]
```

---

## 8. Resuming a Run

Colab sessions disconnect after ~90 minutes of inactivity (or ~12 hours regardless). To resume:

1. Make sure `checkpoint_dir` points to Drive (set in Cell 2) — otherwise the checkpoint is lost.
2. After reconnecting and re-running Cells 1–3, set:

```python
cfg.resume = "/content/drive/MyDrive/anyup3d/checkpoints/anyup3d_step5000.pth"
```

3. Run Cell 5 (model) and Cell 8 (optimizer) — they both load state from `cfg.resume` automatically.
4. Run Cell 10. The loop starts from `global_step` (loaded from the checkpoint), not from 0.

---

## 9. Colab-Specific Tips

### Out of memory errors

If you hit `CUDA out of memory`:
- Reduce `batch_size` in the earliest `t_schedule` stage.
- Reduce `img_size` from `224` to `112` (and `crop_size` from `448` to `224`).
- Reduce `qk_dim` from `128` to `64`.
- Set `num_workers = 0` to free shared memory.

### Slow first batch

The first batch of each `T`-stage is slow because the DataLoader workers are being spawned and the teacher model runs JIT compilation. This is normal — subsequent batches will be faster.

### Keeping the session alive

Colab disconnects on inactivity. To prevent this during long runs, you can use a browser console trick or set `save_every_n_steps` low (e.g. `100`) so you always have a recent checkpoint to resume from.

### Free Colab vs Colab Pro

| Concern | Free T4 | Colab Pro A100 |
|---|---|---|
| GPU memory | 15 GB | 40 GB |
| Max practical `T` | 8 | 16 |
| Max practical `batch_size` at `T=1` | 4 | 16 |
| Session limit | ~90 min idle / 12 hr total | ~24 hr |
| Recommended `img_size` | 112–224 | 224 |
