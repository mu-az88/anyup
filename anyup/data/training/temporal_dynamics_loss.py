"""Temporal loss that matches the teacher's change over time (T01_training).

Replaces the old inline loss in `Train_3D_full.ipynb` cell 13 (`feat_diff.pow(2) * gate`), which
penalized any change over time and so rewarded frozen features (T01 Findings A–C).

    loss = mean over (B, pairs, H, W) of ‖Δ norm(pred) − Δ norm(gt)‖²     (summed over channels)

with Δx = x[:, :, 1:] − x[:, :, :-1] along the tubelet axis and norm = unit length per location.
Stable errors cost nothing; flicker and freezing are both penalized. Scale-free: values lie in
[0, 16] per location. Computed in float32, also under bf16 autocast.
"""
import torch
import torch.nn.functional as F


def temporal_dynamics_loss(pred, gt, gate=None):
    """pred, gt: (B, C, T_tok, H, W). Optional gate: (B, 1, T_tok-1, H, W), weights per frame pair."""
    if pred.shape != gt.shape:
        raise ValueError(f"temporal_dynamics_loss: shape mismatch pred {tuple(pred.shape)} vs gt {tuple(gt.shape)}")
    if pred.dim() != 5:
        raise ValueError(f"temporal_dynamics_loss: expected (B, C, T_tok, H, W), got {tuple(pred.shape)}")
    B, _, T_tok, H, W = pred.shape
    if gate is not None and tuple(gate.shape) != (B, 1, T_tok - 1, H, W):
        raise ValueError(f"temporal_dynamics_loss: gate {tuple(gate.shape)} must be (B, 1, T_tok-1, H, W) = "
                         f"{(B, 1, T_tok - 1, H, W)}")
    if T_tok < 2:                                  # T_tok=1: no pairs (avoids NaN from an empty mean)
        return pred.new_zeros((), dtype=torch.float32)
    p = F.normalize(pred.float(), dim=1)           # (B, C, T_tok, H, W), unit per location → scale-free; float32
    g = F.normalize(gt.float(), dim=1)
    err = ((p[:, :, 1:] - p[:, :, :-1]) - (g[:, :, 1:] - g[:, :, :-1])).pow(2).sum(1, keepdim=True)   # (B, 1, T_tok-1, H, W)
    return (err * gate.float() if gate is not None else err).mean()
