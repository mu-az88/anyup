import random
import torch
import torch.nn.functional as F


@torch.no_grad()
def extract_teacher_features(encoder, video, device=None):
    """video (B, 16, 3, H, W) -> (B, D, 8, 14, 14)"""
    c = encoder.config
    B, T, C, H, W = video.shape
    if T != c.num_frames:                                  # 16 — teacher native; loader must load this many
        raise ValueError(f"teacher needs {c.num_frames} frames, got {T}")
    pv = video.to(device or next(encoder.parameters()).device)
    if (H, W) != (c.image_size, c.image_size):             # 224; crop_low (112) is upscaled here
        pv = F.interpolate(pv.reshape(B * T, C, H, W), size=(c.image_size, c.image_size),
                           mode="bilinear", align_corners=False).view(B, T, C, c.image_size, c.image_size)
    g = c.image_size // c.patch_size                       # 14 tokens per side
    out = encoder(pixel_values=pv).last_hidden_state       # (B, 1568, D) — position table untouched
    return out.reshape(B, T // c.tubelet_size, g, g, c.hidden_size).permute(0, 4, 1, 2, 3).contiguous()


def sample_clip_indices(total, n_frames, stride, rng=random):
    """n_frames indices at `stride`; falls back to the largest stride that fits, then clamps (pads) at the end."""
    s = max(1, min(stride, (total - 1) // (n_frames - 1)))  # stride 4 → span 61 frames; short clips shrink s
    start = rng.randint(0, max(0, total - ((n_frames - 1) * s + 1)))
    return [min(start + i * s, total - 1) for i in range(n_frames)]


def sample_student_window(t_student, teacher_t, tubelet, rng=random):
    """Random tubelet-aligned window. Returns (f0, tok0, n_tok)."""
    if t_student % tubelet or not tubelet <= t_student <= teacher_t:
        raise ValueError(f"invalid t_student={t_student}")
    n_tok = t_student // tubelet                           # T=2→1, 4→2, 8→4 tubelets
    tok0 = rng.randint(0, teacher_t // tubelet - n_tok)
    return tok0 * tubelet, tok0, n_tok


def take_window(frames, feats, f0, tok0, n_tok, tubelet):
    """frames (B, 16, ...) and feats (B, D, 8, h, w) -> matching window of each."""
    return frames[:, f0:f0 + n_tok * tubelet], feats[:, :, tok0:tok0 + n_tok].contiguous()


def assemble_clip(got, idxs):
    """got {frame index: frame} as decoded -> len(idxs) frames in idxs order; [] if nothing decoded.
    Indices never decoded (video shorter than its header says) repeat the last decoded frame."""
    frames = [got[i] for i in idxs if i in got]            # never got[i] directly: a KeyError → all-zero clip
    return frames + [frames[-1]] * (len(idxs) - len(frames)) if frames else []
