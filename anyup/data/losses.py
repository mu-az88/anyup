
import torch
import torch.nn.functional as F


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