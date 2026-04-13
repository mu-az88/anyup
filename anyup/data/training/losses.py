
import torch
import torch.nn.functional as F
from anyup.data.training.augmentations import augment_guidance_video

def cos_mse_loss(
    pred: torch.Tensor,   # (B, T, h, w, C)  — model output q'
    target: torch.Tensor, # (B, T, h, w, C)  — GT encoder features q̂
) -> torch.Tensor:
    """
    Primary reconstruction loss: L_cos-mse = (1 - cosine_sim) + MSE
    Averaged over all spatiotemporal positions (T, h, w) and the batch.

    Args:
        pred:   (B, T, h, w, C)  predicted feature volume
        target: (B, T, h, w, C)  ground-truth feature volume

    Returns:
        Scalar loss tensor.
    """
    B, T, h, w, C = pred.shape  # T=number of frames, h/w=spatial resolution tokens, C=feature dim
                                 # ↑ if you change T (curriculum) this automatically handles it

    # --- flatten spatiotemporal positions for vectorised ops ---
    # (B, T*h*w, C) — T*h*w is the total token count per sample
    # ↓ total_tokens = T * h * w — grows fast; reduce T or h/w to save memory
    pred_flat   = pred.reshape(B, T * h * w, C)   # ↓ depends on T (5.3 curriculum scheduler)
    target_flat = target.reshape(B, T * h * w, C) # must match pred shape exactly

    # --- cosine similarity ---
    # F.cosine_similarity on dim=-1 → (B, T*h*w), values in [-1, 1]
    cos_sim = F.cosine_similarity(pred_flat, target_flat, dim=-1)  # (B, T*h*w)
    cos_loss = (1.0 - cos_sim).mean()  # scalar; 0 when pred and target are identical

    # --- MSE ---
    mse_loss = F.mse_loss(pred_flat, target_flat, reduction="mean")  # scalar

    return cos_loss + mse_loss




def input_consistency_loss(
    q_pred: torch.Tensor,  # (B, T, H, W, C) — high-res model output q'
    p: torch.Tensor,       # (B, T, h, w, C) — coarse input features (low-res)
) -> torch.Tensor:
    """
    L_input-consistency: downsample q' spatially to match p's resolution,
    then compute cos_mse_loss(q'_downsampled, p).

    Downsampling is spatial-only (per frame) — the T dimension is never touched.

    Args:
        q_pred: (B, T, H, W, C)  high-resolution predicted features
        p:      (B, T, h, w, C)  coarse input feature volume

    Returns:
        Scalar loss tensor.
    """
    B, T, H, W, C = q_pred.shape   # ↑ H, W = high-res spatial dims (e.g. 56x56)
    _, _, h, w, _ = p.shape         # ↑ h, w = low-res spatial dims (e.g. 14x14)
                                    #   ratio H/h must match the model's upsampling factor

    # --- move C before spatial dims for F.interpolate ---
    # F.interpolate expects (N, C, H, W); we have (B, T, H, W, C)
    # Merge B and T into a single batch axis → (B*T, C, H, W)
    # ↓ B*T is the effective batch size here; large T multiplies memory linearly
    q_flat = q_pred.permute(0, 1, 4, 2, 3)         # (B, T, C, H, W)
    q_flat = q_flat.reshape(B * T, C, H, W)         # (B*T, C, H, W) — ↑ depends on T (curriculum)

    # --- spatial downsampling, per frame, no temporal touching ---
    # mode="area" = proper anti-aliased average pooling when shrinking
    # ↓ target size (h, w) must match p's spatial dims exactly
    q_down = F.interpolate(
        q_flat,
        size=(h, w),        # ↓ low-res target; if you change patch size, update here
        mode="area",        # correct for downsampling; switch to "bilinear" only if upsampling
    )                       # → (B*T, C, h, w)

    # --- restore (B, T, h, w, C) to match p's layout ---
    q_down = q_down.reshape(B, T, C, h, w)          # (B, T, C, h, w)
    q_down = q_down.permute(0, 1, 3, 4, 2)          # (B, T, h, w, C)

    return cos_mse_loss(q_down, p)


def self_consistency_loss(
    model,                          # AnyUp3D — callable: (p, V) → q' of shape (B, T, H, W, C)
    p: torch.Tensor,                # (B, T, h, w, C) — coarse feature input; shared across both branches
    V: torch.Tensor,                # (B, T, H, W, 3) — clean guidance video, float in [0, 1]
    brightness_range: tuple = (0.85, 1.15),  # ↑ passed to augment_guidance_video
    contrast_range:   tuple = (0.85, 1.15),  # ↑ passed to augment_guidance_video
    noise_std: float = 0.02,                  # ↑ passed to augment_guidance_video
) -> torch.Tensor:
    """
    L_self-consistency = L_cos-mse(f(p, V), f(p, V_aug))

    Gradients flow only through the clean branch f(p, V).
    The augmented branch runs under torch.no_grad() — it is the pseudo-target.

    Args:
        model:  AnyUp3D model (must be in train mode for the clean branch)
        p:      (B, T, h, w, C)  coarse input features
        V:      (B, T, H, W, 3)  clean RGB guidance frames

    Returns:
        Scalar loss tensor (with grad attached via clean branch).
    """
    # --- augmented branch: no gradients, treated as fixed target ---
    # ↓ this forward pass costs the same memory as the clean pass but no backward graph
    # ↓ if OOM: move augmented branch to after clean branch so activations don't overlap
    V_aug = augment_guidance_video(V, brightness_range, contrast_range, noise_std)
    with torch.no_grad():
        q_aug = model(p, V_aug)     # (B, T, H, W, C) — pseudo-target, detached

    # --- clean branch: gradients flow normally ---
    # ↓ this is the expensive pass — activations retained for backward
    # ↓ if OOM at this step: enable gradient checkpointing on CrossAttentionBlock3D (task 7.2)
    q_clean = model(p, V)           # (B, T, H, W, C) — prediction with grad

    # cos_mse_loss averages over (B, T, H, W) — scalar
    return cos_mse_loss(q_clean, q_aug.detach())  # .detach() is redundant but explicit

# Add to anyup/data/training/losses.py

def _cos_mse_per_pair(
    a: torch.Tensor,  # (B, T-1, H, W, C)
    b: torch.Tensor,  # (B, T-1, H, W, C)
) -> torch.Tensor:
    """
    Per-pair cos-mse loss without global reduction.
    Returns (B, T-1) — one scalar per adjacent frame pair per sample.
    Needed so the temporal gate can zero out specific pairs before averaging.
    """
    B, Tm1, H, W, C = a.shape  # ↑ Tm1 = T-1 pairs; grows with T (curriculum)

    # flatten spatial for cosine_similarity — (B*Tm1, H*W, C)
    # ↓ H*W*C floats per pair — reduce H,W or T if OOM
    a_flat = a.reshape(B * Tm1, H * W, C)   # ↑ depends on H,W (model output resolution)
    b_flat = b.reshape(B * Tm1, H * W, C)

    # cosine similarity → (B*Tm1, H*W), mean over spatial → (B*Tm1,)
    cos_sim  = F.cosine_similarity(a_flat, b_flat, dim=-1)  # (B*Tm1, H*W)
    cos_loss = (1.0 - cos_sim).mean(dim=-1)                 # (B*Tm1,)

    # MSE per pair — mean over H*W*C → (B*Tm1,)
    mse_loss = ((a_flat - b_flat) ** 2).mean(dim=(-1, -2))  # (B*Tm1,)

    # per-pair loss → (B, T-1)
    return (cos_loss + mse_loss).reshape(B, Tm1)


def temporal_consistency_loss(
    q_pred: torch.Tensor,       # (B, T, H, W, C) — model output q'
    V: torch.Tensor,            # (B, T, H, W, 3) — RGB guidance video, float in [0,1]
    rgb_diff_threshold: float = 0.05,  # ↑ gate threshold; lower = stricter (fewer pairs used)
                                       #   0.05 ≈ normal motion; raise to 0.10 for action video
) -> torch.Tensor:
    """
    L_temporal-consistency: penalise feature flicker between adjacent frames,
    gated by how much the RGB actually changed.

    Gate logic: if mean(|V_t - V_{t-1}|) < threshold → apply loss on this pair.
    Pairs with large RGB change (motion/cuts) are zeroed out before averaging.

    Returns:
        Scalar loss. Returns 0.0 if ALL pairs are gated out (e.g. pure action clip).
    """
    B, T, H, W, C = q_pred.shape   # ↑ T set by curriculum scheduler (task 5.3)
                                    # ↑ H,W = high-res output spatial dims

    if T < 2:
        # T=1 warmup stage — no adjacent pairs exist, loss is zero
        return q_pred.new_tensor(0.0)

    # --- adjacent feature pairs ---
    q_curr = q_pred[:, 1:,  ...]   # (B, T-1, H, W, C) — frames 1..T-1
    q_prev = q_pred[:, :-1, ...]   # (B, T-1, H, W, C) — frames 0..T-2

    # --- RGB gate: mean absolute diff per adjacent frame pair ---
    # V diff → (B, T-1, H, W, 3)
    v_diff = (V[:, 1:, ...] - V[:, :-1, ...]).abs()            # (B, T-1, H, W, 3)
    # ↓ mean over H, W, C → (B, T-1) scalar per pair
    v_diff_mean = v_diff.mean(dim=(2, 3, 4))                   # (B, T-1)
                                                                # ↑ cheap — no learned params

    # gate = 1 where scene is stable, 0 where there's large motion/cut
    # ↓ rgb_diff_threshold is the key tuning knob — see docstring
    gate = (v_diff_mean < rgb_diff_threshold).float()          # (B, T-1)

    # --- per-pair loss ---
    pair_loss = _cos_mse_per_pair(q_curr, q_prev)              # (B, T-1)

    # --- apply gate before averaging ---
    gated_loss = pair_loss * gate                               # (B, T-1), zeroed on fast pairs

    # normalise by number of active pairs to avoid scale collapse when many pairs are gated
    n_active = gate.sum().clamp(min=1.0)   # ↑ clamp avoids div-by-zero on all-action clips
    return gated_loss.sum() / n_active

# Add to anyup/data/training/losses.py


def get_lambda3(
    step: int,              # ↑ current global training step
    lambda3_max: float,     # ↑ target λ3 value after warmup (e.g. 0.01)
    warmup_steps: int,      # ↑ number of steps to ramp from 0 → lambda3_max
                            #   set in config (task 5.1); increase if temporal loss destabilises early training
) -> float:
    """
    Linear warmup schedule for λ3 (temporal consistency weight).
    Returns 0.0 for step < warmup_steps, lambda3_max at step >= warmup_steps.
    """
    if warmup_steps <= 0:
        return lambda3_max                              # no warmup — use max immediately
    return min(lambda3_max, lambda3_max * step / warmup_steps)  # linear ramp


def combined_loss(
    # --- predictions ---
    q_pred: torch.Tensor,       # (B, T, H, W, C) — high-res model output q'
    q_target: torch.Tensor,     # (B, T, h_gt, w_gt, C) — GT encoder features q̂
    p: torch.Tensor,            # (B, T, h, w, C) — coarse input features
    V: torch.Tensor,            # (B, T, H, W, 3) — RGB guidance video, float in [0,1]
    model,                      # AnyUp3D — needed for L_self forward passes

    # --- fixed weights ---
    lambda1: float = 0.5,       # ↑ weight for L_input-consistency
    lambda2: float = 0.5,       # ↑ weight for L_self-consistency
                                #   if self-consistency dominates, reduce lambda2 first

    # --- temporal schedule ---
    lambda3_max: float = 0.01,  # ↑ max weight for L_temporal — keep small; temporal loss is supplementary
    warmup_steps: int  = 1000,  # ↑ steps before lambda3 reaches max; increase for unstable early training
                                #   depends on T-curriculum (task 5.3): set >= steps before T>1 is reached
    step: int = 0,              # ↑ current global step — passed in from training loop (task 5.2)

    # --- augmentation params forwarded to L_self ---
    brightness_range: tuple = (0.85, 1.15),  # ↑ see augmentations.py
    contrast_range:   tuple = (0.85, 1.15),  # ↑ see augmentations.py
    noise_std: float = 0.02,                  # ↑ see augmentations.py

    # --- gate param forwarded to L_temporal ---
    rgb_diff_threshold: float = 0.05,         # ↑ see temporal_consistency_loss

) -> dict:
    """
    Full AnyUp3D training objective.

    Returns a dict with keys:
        'total'        — scalar to call .backward() on
        'reconstruction' — L_cos-mse (unweighted)
        'input'        — L_input-consistency (unweighted)
        'self'         — L_self-consistency (unweighted)
        'temporal'     — L_temporal-consistency (unweighted)
        'lambda3'      — current λ3 value (for logging)

    Returning individual components lets the training loop log each separately
    (task 5.5) without recomputing them.
    """
    # --- L_reconstruction ---
    l_recon = cos_mse_loss(q_pred, q_target)            # scalar

    # --- L_input-consistency ---
    l_input = input_consistency_loss(q_pred, p)         # scalar

    # --- L_self-consistency ---
    # ↓ runs model twice — most expensive loss; disable first if OOM
    l_self  = self_consistency_loss(
        model, p, V,
        brightness_range=brightness_range,
        contrast_range=contrast_range,
        noise_std=noise_std,
    )                                                   # scalar

    # --- L_temporal-consistency ---
    # ↓ self-disables (returns 0.0) when T=1 — safe to call unconditionally
    l_temporal = temporal_consistency_loss(
        q_pred, V,
        rgb_diff_threshold=rgb_diff_threshold,
    )                                                   # scalar

    # --- current λ3 ---
    lam3 = get_lambda3(step, lambda3_max, warmup_steps)

    # --- combine ---
    # ↓ total is the only tensor .backward() is called on in the training loop
    total = (
        l_recon
        + lambda1 * l_input
        + lambda2 * l_self
        + lam3    * l_temporal
    )

    return {
        "total":          total,
        "reconstruction": l_recon.detach(),
        "input":          l_input.detach(),
        "self":           l_self.detach(),
        "temporal":       l_temporal.detach(),
        "lambda3":        lam3,
    }