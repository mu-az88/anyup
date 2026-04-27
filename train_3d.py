"""
train.py — AnyUp3D training entry point.

Usage:
    python train.py --config configs/anyup3d_train.yaml
    python train.py --config configs/anyup3d_train.yaml --debug

Depends on:
    - anyup/model.py          (AnyUp3D model)
    - anyup/data/training/losses.py         (combined_loss, all λ-weighted components)
    - scripts/load_2d_weights.py        (load_2d_weights_into_3d)
    - Person B: anyup/data/video_dataset.py + get_video_dataloaders()
"""

import argparse
import os
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from transformers import VideoMAEModel, AutoImageProcessor

# ── Project imports ────────────────────────────────────────────────────────────
from anyup.model import AnyUp
from anyup.data.training.losses import combined_loss
from anyup.utils.seed import set_seed
from scripts.load_2d_weights import load_2d_weights_into_3d

# Person B's data interface — import will fail until their module is ready.
# Stub is provided below for smoke-testing without Person B's code.
try:
    from anyup.data.video_dataset import get_video_dataloaders
    _HAS_PERSON_B_DATA = True
except ImportError:
    _HAS_PERSON_B_DATA = False


# ==============================================================================
# 5.3 — T-Curriculum Scheduler
# ==============================================================================

class TCurriculumScheduler:
    """
    Tracks the current T-curriculum stage based on global step count.

    Stages are defined in config under `t_schedule` as a list of dicts:
        [{t: 1, start_step: 0, batch_size: 8},
         {t: 4, start_step: 5000, batch_size: 4}, ...]

    The scheduler is step-monotone: once a stage is entered it is never rolled back.
    """

    def __init__(self, t_schedule: list):
        # Sort by start_step ascending so stage lookup is O(n) scan from the end
        self.stages = sorted(t_schedule, key=lambda s: s["start_step"])
        self._current_stage_idx = 0

    def step(self, global_step: int) -> dict:
        """
        Call once per training step. Returns the active stage dict.
        Updates internal stage index if a new stage threshold is crossed.
        """
        # Walk forward through stages — only advances, never retreats
        for i in range(self._current_stage_idx, len(self.stages)):
            if global_step >= self.stages[i]["start_step"]:
                self._current_stage_idx = i
            else:
                break
        return self.stages[self._current_stage_idx]

    @property
    def current_t(self) -> int:
        return self.stages[self._current_stage_idx]["t"]

    @property
    def current_batch_size(self) -> int:
        return self.stages[self._current_stage_idx]["batch_size"]

    def stage_changed(self, global_step: int) -> bool:
        """True on the exact step when a new T stage is first entered."""
        for stage in self.stages:
            if global_step == stage["start_step"] and global_step > 0:
                return True
        return False


# ==============================================================================
# 5.4 — Checkpointing
# ==============================================================================

def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    t_stage: int,
    val_metric: float | None = None,
):
    """Save model + optimizer state + training metadata to `path`."""
    state = {
        "step": step,
        "t_stage": t_stage,                    # current T-curriculum stage (number of frames)
        "val_metric": val_metric,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    tmp_path = path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)                 # atomic write — no half-written checkpoints


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device = torch.device("cuda"),
) -> tuple[int, int]:
    """
    Load checkpoint into model (and optionally optimizer).
    Returns (step, t_stage) so the training loop can resume correctly.
    """
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    step = state.get("step", 0)
    t_stage = state.get("t_stage", 1)
    return step, t_stage


class BestCheckpointTracker:
    """
    Tracks the best validation metric seen so far and saves a dedicated
    best-model checkpoint whenever it improves.
    Lower metric = better (e.g. MSE, RMSE). Set `higher_is_better=True` for mIoU.
    """

    def __init__(self, save_path: str, higher_is_better: bool = False):
        self.save_path = save_path
        self.higher_is_better = higher_is_better
        self.best_metric = -math.inf if higher_is_better else math.inf

    def update(
        self,
        val_metric: float,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        t_stage: int,
    ) -> bool:
        """
        Compare `val_metric` to best seen. If improved, save checkpoint.
        Returns True if this was a new best.
        """
        improved = (
            val_metric > self.best_metric if self.higher_is_better
            else val_metric < self.best_metric
        )
        if improved:
            self.best_metric = val_metric
            save_checkpoint(self.save_path, model, optimizer, step, t_stage, val_metric)
            return True
        return False


# ==============================================================================
# GT Feature Extraction
# ==============================================================================

def build_gt_extractor(cfg, device: torch.device):
    """
    Returns a callable that maps a video tensor (B, C, T, H, W) → GT features
    (B, D, T', H', W') where T', H', W' are the patch-grid dimensions.

    Currently supports: videomae.
    The extractor runs under torch.no_grad() — it is never trained.
    """
    if cfg.encoder == "videomae":
        processor = AutoImageProcessor.from_pretrained(cfg.encoder_model)
        backbone = VideoMAEModel.from_pretrained(cfg.encoder_model).to(device).eval()
        for p in backbone.parameters():
            p.requires_grad_(False)

        t_tubelet = 2                           # VideoMAE temporal patch size (tubelet size = 2)
        patch_size = 16                         # spatial patch size for ViT-B
        embed_dim = 768                         # ViT-B hidden dim
        model_img_size = backbone.config.image_size           # 224 for videomae-base
        model_num_frames = backbone.config.num_frames         # 16 for videomae-base
        model_T_tok = model_num_frames // t_tubelet           # 8

        def extract(video: torch.Tensor) -> torch.Tensor:
            # video: (B, C, T, H, W) ∈ [0, 1]
            B, C, T, H, W = video.shape

            # ── 1. Ensure T ≥ t_tubelet (VideoMAE needs ≥ 2 frames) ──────────
            if T < t_tubelet:
                reps = (t_tubelet + T - 1) // T
                video = video.repeat(1, 1, reps, 1, 1)[:, :, :t_tubelet]
                T = t_tubelet

            T_tok = T // t_tubelet

            # VideoMAE expects (B, T, C, H, W)
            pixel_values = video.permute(0, 2, 1, 3, 4)

            # ── 2. Resize spatial dims to model's native image_size ───────────
            if H != model_img_size or W != model_img_size:
                pixel_values = torch.nn.functional.interpolate(
                    pixel_values.reshape(B * T, C, H, W),
                    size=(model_img_size, model_img_size),
                    mode="bilinear",
                    align_corners=False,
                ).view(B, T, C, model_img_size, model_img_size)

            H_tok = model_img_size // patch_size
            W_tok = model_img_size // patch_size

            # ── 3. Interpolate position embeddings for variable T ─────────────
            # videomae-base position_embeddings are fixed for num_frames=16
            # (shape 1 × 1568 × 768).  Temporarily swap in an interpolated copy
            # when the input T differs from the model's native num_frames.
            orig_pos_data = None
            if T_tok != model_T_tok:
                orig_pos = backbone.embeddings.position_embeddings.data  # (1, model_T_tok*S, D)
                S = H_tok * W_tok
                p = orig_pos.view(1, model_T_tok, S, embed_dim).permute(0, 3, 1, 2)  # (1, D, Tt, S)
                p_interp = torch.nn.functional.interpolate(
                    p.float(),
                    size=(T_tok, S),
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1).reshape(1, T_tok * S, embed_dim).to(orig_pos.dtype)
                orig_pos_data = orig_pos.clone()
                backbone.embeddings.position_embeddings.data = p_interp

            try:
                with torch.no_grad():
                    out = backbone(pixel_values=pixel_values)
            finally:
                if orig_pos_data is not None:
                    backbone.embeddings.position_embeddings.data = orig_pos_data

            # last_hidden_state: (B, T_tok*H_tok*W_tok, embed_dim)
            tokens = out.last_hidden_state
            feats = (
                tokens
                .reshape(B, T_tok, H_tok, W_tok, embed_dim)
                .permute(0, 4, 1, 2, 3)        # → (B, D, T_tok, H_tok, W_tok)
                .contiguous()
            )
            return feats

        return extract
    else:
        raise ValueError(f"Unknown encoder: {cfg.encoder!r}. Supported: ['videomae']")


# ==============================================================================
# Stub DataLoader (used when Person B's module is unavailable)
# ==============================================================================

def _stub_dataloader(cfg, t: int, batch_size: int, device: torch.device):
    """
    Synthetic DataLoader stub for smoke-testing without Person B's data module.
    Yields batches of shape consistent with what the real loader should return.
    Remove once Person B's get_video_dataloaders() is available.
    """
    H = cfg.img_size                            # ↓ reduce to save memory — controls spatial resolution
    W = cfg.img_size                            # ↓ reduce to save memory — linked to H above
    C = 3
    while True:
        video = torch.rand(batch_size, C, t, H, W, device=device)   # (B, C, T, H, W) ∈ [0,1]
        video_crop = torch.rand(batch_size, C, t, H * 2, W * 2, device=device)  # high-res crop
        patch_size = 14
        yield {"video": video, "video_crop": video_crop, "patch_size": patch_size}


def get_loader(cfg, t: int, batch_size: int, device: torch.device):
    """
    Return a data iterator for the given (T, batch_size) combination.
    Uses Person B's get_video_dataloaders() when available, otherwise falls
    back to _stub_dataloader.
    """
    if _HAS_PERSON_B_DATA:
        train_loader, _ = get_video_dataloaders(cfg, t=t, batch_size=batch_size)
        return iter(train_loader)
    else:
        print("[train] WARNING: using stub DataLoader — replace with Person B's module")
        return _stub_dataloader(cfg, t=t, batch_size=batch_size, device=device)


# ==============================================================================
# Temporal λ Warmup
# ==============================================================================

def temporal_lambda(cfg, global_step: int) -> float:
    """
    Linearly ramps lambda_temporal_consistency from 0 → its full config value
    over the first `temporal_lambda_warmup_steps` steps.
    Returns the full value immediately if warmup_steps == 0.
    """
    warmup = cfg.temporal_lambda_warmup_steps
    if warmup == 0 or global_step >= warmup:
        return cfg.lambda_temporal_consistency
    return cfg.lambda_temporal_consistency * (global_step / warmup)


# ==============================================================================
# Validation stub
# ==============================================================================

def run_validation(model: nn.Module, cfg, device: torch.device, global_step: int) -> float:
    """
    Placeholder validation loop.
    Replace with real eval harness (Phase 6) — returns a dummy metric until then.
    Should return the primary scalar metric used for best-checkpoint tracking.
    """
    # TODO (Phase 6): implement per-frame mIoU + temporal coherence evaluation
    print(f"  [val] step={global_step}  (validation not yet implemented — returning inf)")
    return float("inf")


# ==============================================================================
# Main Training Loop  (5.2 + 5.3 + 5.4)
# ==============================================================================

def main():
    # ── Args ──────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Train AnyUp3D")
    parser.add_argument("--config", default="configs/anyup3d_train.yaml")
    parser.add_argument("--debug", action="store_true",
                        help="Override config debug flag for quick smoke test")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.debug:
        cfg.debug = True

    # ── Debug overrides ───────────────────────────────────────────────────────
    if cfg.debug:
        print("[debug] Overriding config for smoke test: 10 steps, T=1, batch=1")
        cfg.max_steps = cfg.debug_steps
        cfg.t_schedule = [{"t": 1, "start_step": 0, "batch_size": 1}]
        cfg.log_every_n_steps = 1
        cfg.save_every_n_steps = cfg.debug_steps
        cfg.val_every_n_steps = cfg.debug_steps

    # ── Seeding ───────────────────────────────────────────────────────────────
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}  seed={cfg.seed}")

    # ── Logging ───────────────────────────────────────────────────────────────
    log_dir = Path(cfg.checkpoint_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    # ── Model ─────────────────────────────────────────────────────────────────
    if cfg.model_ckpt_2d and Path(cfg.model_ckpt_2d).exists():
        model = load_2d_weights_into_3d(cfg.model_ckpt_2d).to(device)
        print(f"[train] Loaded 2D weights from {cfg.model_ckpt_2d}")
    else:
        model = AnyUp().to(device)
        print(f"[train] WARNING: 2D checkpoint not found at {cfg.model_ckpt_2d!r} — training from scratch")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    global_step = 0
    current_t_stage = 1
    if cfg.resume and Path(cfg.resume).exists():
        global_step, current_t_stage = load_checkpoint(
            cfg.resume, model, optimizer, device=device
        )
        print(f"[train] Resumed from {cfg.resume}  step={global_step}  T={current_t_stage}")

    # ── Curriculum + Checkpointing setup ──────────────────────────────────────
    scheduler = TCurriculumScheduler(OmegaConf.to_container(cfg.t_schedule, resolve=True))

    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_tracker = BestCheckpointTracker(
        save_path=str(ckpt_dir / "best.pth"),
        higher_is_better=False,                # tracking loss / MSE — lower = better
    )

    # ── GT extractor (frozen video encoder) ───────────────────────────────────
    extract_gt = build_gt_extractor(cfg, device)

    # ── Initial curriculum state ───────────────────────────────────────────────
    stage = scheduler.step(global_step)
    current_t = stage["t"]
    current_batch_size = stage["batch_size"]   # ↓ reduce any stage's batch_size in config to save memory
    print(f"[train] Starting at T={current_t}  batch_size={current_batch_size}")

    # ── DataLoader ────────────────────────────────────────────────────────────
    data_iter = get_loader(cfg, current_t, current_batch_size, device)

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    t0 = time.time()

    while global_step < cfg.max_steps:

        # -- 5.3: T-curriculum stage check ------------------------------------
        stage = scheduler.step(global_step)
        new_t = stage["t"]
        new_bs = stage["batch_size"]           # ↓ controlled by t_schedule in config

        if new_t != current_t:
            print(f"\n[curriculum] Step {global_step}: T {current_t} → {new_t}, "
                  f"batch_size {current_batch_size} → {new_bs}")
            current_t = new_t
            current_batch_size = new_bs
            # Rebuild DataLoader for the new (T, batch_size) combination
            data_iter = get_loader(cfg, current_t, current_batch_size, device)
            model.train()

        # -- Data loading -----------------------------------------------------
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = get_loader(cfg, current_t, current_batch_size, device)
            batch = next(data_iter)

        # batch keys (Person B's contract):
        #   "video"      : (B, C, T, H, W)   — low-res input clip, values ∈ [0, 1]
        #   "video_crop" : (B, C, T, Hc, Wc) — high-res crop for GT extraction
        #   "patch_size" : int                — patch size for this batch
        video = batch["video"].to(device, non_blocking=True)            # (B, C, T, H, W)
        video_crop = batch["video_crop"].to(device, non_blocking=True)  # (B, C, T, Hc, Wc)
        patch_size = batch["patch_size"]
        if isinstance(patch_size, torch.Tensor):
            patch_size = patch_size[0].item()

        # -- GT feature extraction (frozen encoder, no grad) ------------------
        with torch.no_grad():
            gt_features = extract_gt(video_crop)    # (B, D, T', H', W')

        # -- Forward pass -----------------------------------------------------
        # AnyUp.forward(image, features, output_size) — upsample gt_features
        # guided by the low-res video, targeting the GT token-grid resolution.
        _, _, T_tok, H_tok, W_tok = gt_features.shape
        pred_features = model(video, gt_features, output_size=(H_tok, W_tok))

        # -- Loss -------------------------------------------------------------
        lam_t = temporal_lambda(cfg, global_step)   # ramped temporal λ

        # combined_loss expects channels-last (B, T, H, W, C) tensors and a
        # model callable with signature model(p, V) where p is features-first.
        # AnyUp.forward is (image, features, output_size) / channels-first, so
        # we convert layouts and provide a thin wrapper for self_consistency_loss.
        pred_cl = pred_features.permute(0, 2, 3, 4, 1).contiguous()   # (B,T,H',W',C)
        gt_cl   = gt_features.permute(0, 2, 3, 4, 1).contiguous()     # (B,T,H',W',C)
        video_cl = video.permute(0, 2, 3, 4, 1).contiguous()          # (B,T,H,W,3)

        def _model_cl(p_cl, V_cl):
            """Adapter: channels-last (p,V) → channels-first AnyUp → channels-last."""
            p_cf  = p_cl.permute(0, 4, 1, 2, 3)
            V_cf  = V_cl.permute(0, 4, 1, 2, 3)
            out_h, out_w = p_cl.shape[2], p_cl.shape[3]
            out = model(V_cf, p_cf, output_size=(out_h, out_w))
            return out.permute(0, 2, 3, 4, 1)

        loss_dict = combined_loss(
            q_pred=pred_cl,
            q_target=gt_cl,
            p=gt_cl,                                  # coarse features (GT used as placeholder)
            V=video_cl,
            model=_model_cl,
            lambda1=cfg.lambda_input_consistency,
            lambda2=cfg.lambda_self_consistency,
            lambda3_max=lam_t,
            step=global_step,
        )
        loss = loss_dict["total"]
        loss_components = loss_dict

        # -- Backward + optimizer step ----------------------------------------
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_max_norm)
        optimizer.step()

        global_step += 1

        # -- Logging ----------------------------------------------------------
        if global_step % cfg.log_every_n_steps == 0:
            elapsed = time.time() - t0
            steps_per_sec = cfg.log_every_n_steps / elapsed
            t0 = time.time()

            print(
                f"[step {global_step:06d}] "
                f"T={current_t}  "
                f"loss={loss.item():.4f}  "
                f"L_recon={loss_components['reconstruction']:.4f}  "
                f"L_inp={loss_components['input']:.4f}  "
                f"L_self={loss_components['self']:.4f}  "
                f"L_temp={loss_components['temporal']:.4f}  "
                f"λ_temp={lam_t:.3f}  "
                f"{steps_per_sec:.1f} it/s"
            )

            writer.add_scalar("loss/total", loss.item(), global_step)
            writer.add_scalar("loss/reconstruction", loss_components["reconstruction"], global_step)
            writer.add_scalar("loss/input_consistency", loss_components["input"], global_step)
            writer.add_scalar("loss/self_consistency", loss_components["self"], global_step)
            writer.add_scalar("loss/temporal_consistency", loss_components["temporal"], global_step)
            writer.add_scalar("curriculum/T", current_t, global_step)
            writer.add_scalar("curriculum/batch_size", current_batch_size, global_step)
            writer.add_scalar("lambda/temporal", lam_t, global_step)
            writer.add_scalar("perf/steps_per_sec", steps_per_sec, global_step)

            if torch.cuda.is_available():
                mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
                writer.add_scalar("memory/peak_gpu_gb", mem_gb, global_step)
                torch.cuda.reset_peak_memory_stats(device)

        # -- Periodic checkpoint (5.4) ----------------------------------------
        if global_step % cfg.save_every_n_steps == 0:
            ckpt_path = str(ckpt_dir / f"step_{global_step:06d}.pth")
            save_checkpoint(ckpt_path, model, optimizer,
                            step=global_step, t_stage=current_t)
            print(f"[ckpt] Saved periodic checkpoint → {ckpt_path}")

        # -- Validation + best checkpoint (5.4) --------------------------------
        if global_step % cfg.val_every_n_steps == 0:
            model.eval()
            val_metric = run_validation(model, cfg, device, global_step)
            model.train()

            writer.add_scalar("val/metric", val_metric, global_step)

            is_best = best_tracker.update(
                val_metric, model, optimizer,
                step=global_step, t_stage=current_t
            )
            if is_best:
                print(f"[ckpt] New best metric={val_metric:.4f} → {best_tracker.save_path}")

    # ── End of training ───────────────────────────────────────────────────────
    final_path = str(ckpt_dir / "final.pth")
    save_checkpoint(final_path, model, optimizer,
                    step=global_step, t_stage=current_t)
    print(f"\n[train] Training complete. Final checkpoint → {final_path}")
    writer.close()


if __name__ == "__main__":
    main()