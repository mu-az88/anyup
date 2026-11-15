"""
Shared DAVIS-2017 utilities for AnyUp3D experiments (task T04).

    root  = download_davis("/content")                 # → /content/DAVIS (skips if present)
    clips = list_clips(root, "val")                     # 30 clips
    src   = open_clip(root, clip)                       # T02 FrameFolderSource, frames (N, 3, H, W) [0, 1]
    gt    = load_annotations(root, clip)                # (N, H, W) uint8 object ids
    pred  = propagate_labels(feats, gt[0])              # Exp 3 label propagation on upsampled features
    save_mask_png(pred[t], ann_paths(root, clip)[0], out)   # native-res palette PNG, official results layout

Scoring (official J&F) is in anyup.metrics (eval_clip_jf / summarize_jf).
"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .video import FrameFolderSource

PathLike = Union[str, Path]

DAVIS_URL = "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"   # data.davis-challenge.ch is unreachable from Colab
RES = "480p"

# Exp 3 label propagation hyperparameters (Exp 3 Cell 2)
TAU = 0.07          # softmax temperature, Jabri et al. NeurIPS 2020
TOP_K = 7           # neighbours kept per query, DINO eval_video_segmentation.py
N_LAST_FRAMES = 6   # ↓ context frames besides frame 0, DINO default; affinity memory ∝ (1 + N_LAST_FRAMES)·(H·W)². Dependent: upsampler output_size


# ---------------------------------------------------------------------------
# Download + paths
# ---------------------------------------------------------------------------

def download_davis(dest: PathLike = "/content", url: str = DAVIS_URL) -> Path:
    """
    Download + extract DAVIS-2017 trainval 480p into <dest>/DAVIS. Skips if ImageSets/2017/val.txt exists.
    The zip goes to <dest>/davis_zip/ (its own folder) and is kept for reruns.
    """
    dest = Path(dest)
    root = dest / "DAVIS"
    if (root / "ImageSets" / "2017" / "val.txt").exists():
        return root
    zdir = dest / "davis_zip"
    zdir.mkdir(parents=True, exist_ok=True)
    zpath = zdir / url.rsplit("/", 1)[-1]
    if not zpath.exists():
        tmp = zpath.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(dest)                      # archive's top folder is DAVIS/
    if not (root / "ImageSets" / "2017" / "val.txt").exists():
        raise FileNotFoundError(f"{root}/ImageSets/2017/val.txt not found after extracting {zpath}")
    return root


def list_clips(root: PathLike, split: str = "val") -> List[str]:
    """Clip names from ImageSets/2017/<split>.txt, in file order (as davis2017 DAVIS(sequences='all'))."""
    path = Path(root) / "ImageSets" / "2017" / f"{split}.txt"
    if not path.exists():
        raise FileNotFoundError(f"split file not found: {path}")
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


def frames_dir(root: PathLike, clip: str, res: str = RES) -> Path:
    return Path(root) / "JPEGImages" / res / clip


def ann_dir(root: PathLike, clip: str, res: str = RES) -> Path:
    return Path(root) / "Annotations" / res / clip


def frame_paths(root: PathLike, clip: str, res: str = RES) -> List[Path]:
    return sorted(frames_dir(root, clip, res).glob("*.jpg"))


def ann_paths(root: PathLike, clip: str, res: str = RES) -> List[Path]:
    return sorted(ann_dir(root, clip, res).glob("*.png"))


def open_clip(root: PathLike, clip: str, res: str = RES) -> FrameFolderSource:
    """T02 frame reader for the clip: frames (N, 3, H, W) float [0, 1], sorted like the annotations."""
    return FrameFolderSource(frames_dir(root, clip, res), ext=".jpg")


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------

def load_annotation(path: PathLike) -> np.ndarray:
    """Palette PNG → (H, W) uint8 object ids (0 = bg, 255 = void)."""
    return np.array(Image.open(path))


def load_annotations(root: PathLike, clip: str, res: str = RES) -> np.ndarray:
    """(N, H, W) uint8 GT for every frame. Raises if the annotation and frame counts differ (train/val have one per frame)."""
    paths = ann_paths(root, clip, res)
    n_frames = len(frame_paths(root, clip, res))
    if len(paths) != n_frames:
        raise ValueError(f"{clip}: {len(paths)} annotations for {n_frames} frames")
    return np.stack([load_annotation(p) for p in paths])


def num_objects(ann0: np.ndarray) -> int:
    """Objects in the first-frame GT: max id, void (255) ignored (davis2017 get_all_masks; Exp 3 used ann0.max())."""
    a = np.asarray(ann0)
    return int(a[a != 255].max()) if (a != 255).any() else 0


def one_hot_objects(masks: np.ndarray, n_obj: int) -> np.ndarray:
    """(N, H, W) ids → (n_obj, N, H, W) bool; channel k = object k+1 (official layout)."""
    masks = np.asarray(masks)
    return masks[None] == np.arange(1, n_obj + 1)[:, None, None, None]


def resize_labels(mask: np.ndarray, size: Union[int, Tuple[int, int]]) -> np.ndarray:
    """Nearest-neighbour resize of an id map (H, W) to size (h, w), as Exp 3 (cv2.INTER_NEAREST)."""
    h, w = (size, size) if isinstance(size, int) else size
    return cv2.resize(np.asarray(mask), (w, h), interpolation=cv2.INTER_NEAREST)


def save_mask_png(mask, ref_ann_path: PathLike, out_path: PathLike) -> None:
    """
    Save a predicted id map as a palette PNG at the reference annotation's resolution (Exp 3 save_mask_png).
    mask: (h, w) int tensor or array, ids 0..n_objects. Layout for the official evaluator:
    <results>/<clip>/<frame stem>.png
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ref_img = Image.open(ref_ann_path)
    H_orig, W_orig = np.array(ref_img).shape[:2]
    mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
    if mask_np.max() > 254:
        raise ValueError(f"object id {mask_np.max()} does not fit a palette PNG (<= 254; 255 = void)")
    mask_np = mask_np.astype(np.uint8)
    mask_np = cv2.resize(mask_np, (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)
    out_img = Image.fromarray(mask_np, mode="P")
    if ref_img.mode == "P":
        out_img.putpalette(ref_img.getpalette())
    out_img.save(out_path)


# ---------------------------------------------------------------------------
# Label propagation (Exp 3 Cell 7 propagate_clip, constants as kwargs)
# ---------------------------------------------------------------------------

def _label_maps_frame0(ann0_HW: np.ndarray, n_objects: int, H: int, W: int) -> torch.Tensor:
    """Exp 3 build_label_maps_frame0: frame-0 GT → one-hot (n_objects+1, H*W) float32 at the feature grid."""
    ann_resized = cv2.resize(
        ann0_HW, (W, H), interpolation=cv2.INTER_NEAREST
    ).astype(np.int64)                      # (H, W)

    n_ch    = n_objects + 1                 # background channel + one per object
    N       = H * W
    label   = np.zeros((n_ch, N), dtype=np.float32)
    ann_flat = ann_resized.flatten()        # (N,)

    for obj_id in range(n_ch):
        label[obj_id, ann_flat == obj_id] = 1.0

    return torch.from_numpy(label)          # (n_ch, N)


@torch.no_grad()
def propagate_labels(feats: torch.Tensor, ann0: np.ndarray, tau: float = TAU, top_k: int = TOP_K,
                     n_last: int = N_LAST_FRAMES) -> torch.Tensor:
    """
    Non-parametric label propagation (Jabri et al. 2020 / DINO protocol), Exp 3 propagate_clip.

    feats : (T, C, H, W) upsampled features, frame 0 = annotated frame; computed on feats.device
    ann0  : (H0, W0) uint8 GT of frame 0 at any resolution (resized to (H, W) with nearest)
    Returns: (T, H, W) int64 predicted ids on CPU. pred[0] is the GT itself, so never score frame 0.

    Memory: affinity per frame is (H·W) × ((1 + n_last)·H·W) float32 (56²: ≈ 275 MB).
    Dependent: upsampler output_size (H, W) and n_last.
    """
    T, C, H, W = feats.shape   # H, W = propagation grid (upsampler output size)
    N = H * W                  # ↓ spatial locations per frame; affinity ∝ N²

    feats_flat = feats.reshape(T, C, N).permute(0, 2, 1)       # (T, N, C)
    feats_flat = F.normalize(feats_flat, dim=-1)               # (T, N, C) unit vectors

    ann0 = np.asarray(ann0)
    if (ann0 == 255).any():
        # Exp 3 used ann0.max() as the object count, which a void label would turn into 255 objects
        raise ValueError("ann0 contains void (255) pixels; Exp 3 propagation does not handle them")
    n_objects = int(ann0.max())                                # Exp 3: DAVIS ids are contiguous from 1
    n_ch = n_objects + 1                                       # +1 for background

    label_maps = torch.zeros(T, n_ch, N, device=feats.device)  # (T, n_ch, N)
    label_maps[0] = _label_maps_frame0(ann0, n_objects, H, W).to(feats.device)

    for t in range(1, T):
        past      = list(range(max(1, t - n_last), t))
        context_t = [0] + past                                 # frame 0 always first

        ctx_feats  = feats_flat[context_t].reshape(-1, C)                      # (Nctx*N, C)
        ctx_labels = label_maps[context_t].permute(1, 0, 2).reshape(n_ch, -1)  # (n_ch, Nctx*N)

        q = feats_flat[t]                                      # (N, C)
        A = torch.mm(q, ctx_feats.T) / tau                     # (N, Nctx*N)

        topk_vals, topk_idx = A.topk(top_k, dim=-1)            # (N, top_k)
        A_sparse = torch.full_like(A, float("-inf"))
        A_sparse.scatter_(1, topk_idx, topk_vals)

        A_soft = F.softmax(A_sparse, dim=-1)                   # (N, Nctx*N)
        new_labels = torch.mm(A_soft, ctx_labels.T)            # (N, n_ch)
        label_maps[t] = new_labels.T                           # (n_ch, N)

    pred = label_maps.argmax(dim=1)                            # (T, N)
    return pred.view(T, H, W).cpu()                            # (T, H, W) int64
