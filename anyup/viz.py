"""
T10: shared helpers for the presentation-grade visual cells of Exp 1–3. No model code.

    pca = FeaturePCA(seed=0).fit({"bilinear": s_b, "anyup3d": s_3})   # samples (N, C) per condition
    rgb = pca.rgb(feats, "anyup3d")                                    # (T, C, H, W) → uint8 (T, H, W, 3)
    fl  = flicker_map(feats)                                           # (T, C, H, W) → (T-1, H, W) = 1 − cos(t, t+1)
    write_mp4(frames, "x.mp4", fps=8); write_gif(frames, "x.gif", fps=8)

PCA colours use a FIXED scale per condition (p1/p99 of the fit sample, set once in fit). Normalizing
each frame by its own min/max would make perfectly stable features flicker in a video.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from anyup.metrics import _unit_frame

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Features → colours
# ---------------------------------------------------------------------------

class FeaturePCA:
    """
    3-component PCA, basis fitted jointly on all conditions (a colour = the same feature direction in
    every panel), colour scale fixed per condition. Exact: covariance + eigh in float64 (deterministic).
    """

    def __init__(self, seed: int = 0, q_low: float = 0.01, q_high: float = 0.99):
        self.seed, self.q_low, self.q_high = seed, q_low, q_high
        self.mean: Optional[torch.Tensor] = None        # (C,) float64, CPU
        self.components: Optional[torch.Tensor] = None  # (C, 3) float64, CPU
        self.scale: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}   # key → (lo (3,), hi (3,))

    @staticmethod
    def sample_tokens(feats: torch.Tensor, n: int, seed: int = 0) -> torch.Tensor:
        """(T, C, H, W) → (min(n, T·H·W), C) random tokens, float64 on CPU."""
        T, C, H, W = feats.shape
        flat = feats.permute(0, 2, 3, 1).reshape(-1, C)                    # (T·H·W, C)
        idx = np.random.default_rng(seed).choice(flat.shape[0], min(n, flat.shape[0]), replace=False)
        return flat[torch.as_tensor(np.sort(idx), device=flat.device)].double().cpu()

    def fit(self, samples: Dict[str, torch.Tensor]) -> "FeaturePCA":
        """samples: condition → (N_k, C) tokens. Basis on all of them; colour scale per condition."""
        X = torch.cat([s.double().cpu() for s in samples.values()])         # (ΣN, C)
        self.mean = X.mean(dim=0)
        Xc = X - self.mean
        cov = Xc.T @ Xc / max(X.shape[0] - 1, 1)                            # (C, C)
        evals, evecs = torch.linalg.eigh(cov)                               # ascending
        comps = evecs[:, -3:].flip(dims=[1])                                # (C, 3), largest first
        # fix the eigenvector sign (eigh's sign is arbitrary): largest-|.| entry positive
        sign = torch.sign(comps.gather(0, comps.abs().argmax(dim=0, keepdim=True)))
        self.components = comps * sign
        for k, s in samples.items():
            p = self._project(s.double().cpu())                            # (N_k, 3)
            self.scale[k] = (torch.quantile(p, self.q_low, dim=0), torch.quantile(p, self.q_high, dim=0))
        return self

    def _project(self, tokens: torch.Tensor) -> torch.Tensor:
        """(N, C) → (N, 3), same dtype / device as tokens."""
        m = self.mean.to(tokens.device, tokens.dtype)
        c = self.components.to(tokens.device, tokens.dtype)
        return (tokens - m) @ c

    @torch.no_grad()
    def rgb(self, feats: torch.Tensor, key: str) -> np.ndarray:
        """(T, C, H, W) on any device → uint8 (T, H, W, 3), with the fixed colour scale of `key`."""
        if self.components is None:
            raise RuntimeError("call fit() first")
        lo, hi = (v.to(feats.device, torch.float32) for v in self.scale[key])
        out = []
        for t in range(feats.shape[0]):                                     # one frame at a time (memory)
            tok = feats[t].float().permute(1, 2, 0)                         # (H, W, C)
            p = self._project(tok.reshape(-1, tok.shape[-1])).reshape(*tok.shape[:2], 3)   # (H, W, 3)
            p = ((p - lo) / (hi - lo).clamp_min(1e-8)).clamp(0, 1)
            out.append((p * 255).round().byte().cpu().numpy())
        return np.stack(out)


@torch.no_grad()
def flicker_map(feats: torch.Tensor) -> np.ndarray:
    """
    (T, C, H, W) → float64 (T-1, H, W): 1 − per-pixel cosine between frames t and t+1.
    Same float64 math as metrics.temporal_cosine, so map[t].mean() = 1 − its per_pair[t].
    0 = stable, 2 = flipped. For a chunk boundary, prepend the previous chunk's last frame.
    """
    if feats.ndim != 4 or feats.shape[0] < 2:
        raise ValueError(f"feats must be (T>=2, C, H, W), got {tuple(feats.shape)}")
    out = []
    prev = _unit_frame(feats[0])
    for t in range(1, feats.shape[0]):
        cur = _unit_frame(feats[t])
        out.append((1.0 - (prev * cur).sum(dim=0)).cpu().numpy())          # (H, W)
        prev = cur
    return np.stack(out)


def colorize(values: np.ndarray, vmin: float, vmax: float, cmap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
    """(…, H, W) float → uint8 (…, H, W, 3) RGB with a FIXED range (dark = low)."""
    v = np.clip((np.asarray(values, np.float64) - vmin) / max(vmax - vmin, 1e-12), 0, 1)
    u8 = (v * 255).round().astype(np.uint8)
    flat = u8.reshape(-1, u8.shape[-2], u8.shape[-1])
    rgb = np.stack([cv2.cvtColor(cv2.applyColorMap(f, cmap), cv2.COLOR_BGR2RGB) for f in flat])
    return rgb.reshape(*u8.shape, 3)


def denorm_guide(guide: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    """Normalized (T, 3, H, W) → uint8 (T, H, W, 3)."""
    m = torch.tensor(mean, device=guide.device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=guide.device).view(1, 3, 1, 1)
    x = (guide.float() * s + m).clamp(0, 1)
    return (x.permute(0, 2, 3, 1) * 255).round().byte().cpu().numpy()


# ---------------------------------------------------------------------------
# Masks, labels, layout
# ---------------------------------------------------------------------------

def davis_palette(n: int = 256) -> np.ndarray:
    """(n, 3) uint8: the PASCAL VOC / DAVIS palette (0 = black, 1 = red, 2 = green, ...)."""
    pal = np.zeros((n, 3), np.uint8)
    for i in range(n):
        c, r, g, b = i, 0, 0, 0
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        pal[i] = (r, g, b)
    return pal


_PALETTE = davis_palette()


def overlay(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.55, contour: bool = True) -> np.ndarray:
    """uint8 (H, W, 3) + id map (H, W) → uint8 (H, W, 3). Background (id 0, 255 = void) unchanged."""
    mask = np.asarray(mask).astype(np.int64)
    fg = (mask > 0) & (mask < 255)
    out = rgb.astype(np.float32).copy()
    col = _PALETTE[np.clip(mask, 0, 255)].astype(np.float32)
    out[fg] = out[fg] * (1 - alpha) + col[fg] * alpha
    out = np.clip(out, 0, 255).astype(np.uint8)
    if contour:
        for obj in np.unique(mask[fg]):
            cs, _ = cv2.findContours((mask == obj).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(out, cs, -1, tuple(int(v) for v in _PALETTE[obj]), 1)
    return out


def label_strip(img: np.ndarray, text: str, height: int = 22, font_scale: float = 0.5) -> np.ndarray:
    """uint8 (H, W, 3) → uint8 (height + H, W, 3), white text on a black strip above the image."""
    strip = np.zeros((height, img.shape[1], 3), np.uint8)
    cv2.putText(strip, text, (4, height - 7), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([strip, img], axis=0)


def grid(rows: List[List[np.ndarray]], pad: int = 4) -> np.ndarray:
    """2-D list of uint8 (h, w, 3) panels → one uint8 image; panels are top-left aligned in equal cells."""
    ch = max(p.shape[0] for r in rows for p in r)
    cw = max(p.shape[1] for r in rows for p in r)
    ncol = max(len(r) for r in rows)
    out = np.full((len(rows) * (ch + pad) - pad, ncol * (cw + pad) - pad, 3), 255, np.uint8)
    for i, r in enumerate(rows):
        for j, p in enumerate(r):
            y, x = i * (ch + pad), j * (cw + pad)
            out[y:y + p.shape[0], x:x + p.shape[1]] = p
    return out


def resize_rgb(img: np.ndarray, size: Tuple[int, int], nearest: bool = False) -> np.ndarray:
    """uint8 (H, W, 3) → (h, w, 3); size = (h, w)."""
    return cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Video output
# ---------------------------------------------------------------------------

def _even(img: np.ndarray) -> np.ndarray:
    """Pad to even height / width (H.264 yuv420p needs it)."""
    h, w = img.shape[:2]
    return np.pad(img, ((0, h % 2), (0, w % 2), (0, 0)), constant_values=255)


def write_mp4(frames: List[np.ndarray], path: PathLike, fps: float) -> Path:
    """uint8 (H, W, 3) RGB frames → H.264 / yuv420p MP4 (plays in browsers and PowerPoint). Needs imageio-ffmpeg."""
    import imageio.v2 as imageio
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264", pixelformat="yuv420p",
                            macro_block_size=1, quality=8) as w:
        for f in frames:
            w.append_data(_even(f))
    return path


def write_gif(frames: List[np.ndarray], path: PathLike, fps: float, max_width: int = 960) -> Path:
    """uint8 (H, W, 3) RGB frames → looping GIF, downscaled to max_width."""
    from PIL import Image
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    s = min(1.0, max_width / w)
    size = (int(round(w * s)), int(round(h * s)))
    imgs = [Image.fromarray(f).resize(size, Image.LANCZOS) if s < 1 else Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(round(1000 / fps)), loop=0)
    return path
