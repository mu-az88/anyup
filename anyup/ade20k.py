"""
Shared ADE20K utilities for AnyUp3D experiments (task T06).

    root  = prepare_ade20k("/content", drive_zip=".../ADEChallengeData2016.zip")
    pairs = list_pairs(root, "train")                     # [(image path, mask path)], 20,210 (val: 2,000)
    build_feature_cache(bb, pairs, cache_dir, "train", batch_size=16)   # fp16 LR features + uint8 masks
    cache_meta(cache_dir, "train")["stems_sha1"]                         # data identity for the results config
    ds    = CachedFeats(cache_dir, "train", pairs, size=224)            # → guide, feats, mask

Protocol (old Exp 1): images squashed to size², masks nearest-resized to size², ids 1..150, 0 = ignore.
Scoring is in anyup.metrics (update_conf_mat / compute_miou / image_miou).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .backbones import IMAGENET_MEAN, IMAGENET_STD, normalize_frames, resize_frames

PathLike = Union[str, Path]
Pair = Tuple[Path, Path]

ADE_URL = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
ADE_DIRNAME = "ADEChallengeData2016"
NUM_CLASSES = 150
IGNORE_INDEX = 0    # 0 = unlabeled, excluded from mIoU
_SPLIT_DIR = {"train": "training", "val": "validation"}


# ---------------------------------------------------------------------------
# Download + paths
# ---------------------------------------------------------------------------

def prepare_ade20k(dest: PathLike = "/content", drive_zip: Optional[PathLike] = None, url: str = ADE_URL) -> Path:
    """
    Extract ADE20K into <dest>/ADEChallengeData2016. Skips if already extracted.
    Uses drive_zip if it exists; otherwise downloads to <dest>/ade20k_zip/ and copies the zip to drive_zip (if given).
    """
    dest = Path(dest)
    root = dest / ADE_DIRNAME
    if (root / "images" / "validation").is_dir():
        return root
    if drive_zip is not None and Path(drive_zip).exists():
        zpath = Path(drive_zip)
    else:
        zdir = dest / "ade20k_zip"
        zdir.mkdir(parents=True, exist_ok=True)
        zpath = zdir / url.rsplit("/", 1)[-1]
        if not zpath.exists():
            tmp = zpath.with_suffix(".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(zpath)
        if drive_zip is not None:
            Path(drive_zip).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(zpath, drive_zip)
    with zipfile.ZipFile(zpath) as z:
        _check_members(z.namelist(), ADE_DIRNAME)
        z.extractall(dest)
    return root


def _check_members(names: List[str], top: str) -> None:
    """Refuse archives with members outside <top>/ (absolute paths, '..', other folders): dest also holds the repo checkout."""
    for n in names:
        parts = Path(n).parts
        if not parts or parts[0] != top or Path(n).is_absolute() or ".." in parts:
            raise ValueError(f"unexpected member {n!r} in the ADE20K zip (all members must be under {top}/)")


def list_pairs(root: PathLike, split: str) -> List[Pair]:
    """(image, mask) paths sorted by stem, split "train" / "val". Raises unless the stems match one to one."""
    if split not in _SPLIT_DIR:
        raise ValueError(f"split must be one of {list(_SPLIT_DIR)}, got {split!r}")
    root = Path(root)
    imgs = {p.stem: p for p in (root / "images" / _SPLIT_DIR[split]).glob("*.jpg")}
    masks = {p.stem: p for p in (root / "annotations" / _SPLIT_DIR[split]).glob("*.png")}
    if not imgs or imgs.keys() != masks.keys():
        raise ValueError(f"ADE20K {split}: {len(imgs)} images and {len(masks)} masks don't pair up "
                         f"({len(imgs.keys() ^ masks.keys())} unmatched stems)")
    return [(imgs[s], masks[s]) for s in sorted(imgs)]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_image(path: PathLike, size: int) -> torch.Tensor:
    """→ (3, size, size) float [0, 1]; same resize as the backbone adapters (T01)."""
    arr = np.array(Image.open(path).convert("RGB"))                         # (H, W, 3) uint8
    frame = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)      # (3, H, W)
    return resize_frames(frame[None], size)[0]


def load_mask(path: PathLike, size: int) -> torch.Tensor:
    """→ (size, size) long class ids; nearest, as old Exp 1 Cell 4."""
    mask = transforms.functional.resize(Image.open(path), (size, size),
                                        interpolation=transforms.InterpolationMode.NEAREST)
    return torch.from_numpy(np.array(mask)).long()


def guide_from_image(frame: torch.Tensor) -> torch.Tensor:
    """(3, S, S) [0, 1] → ImageNet-normalized guide, as BackboneAdapter.forward."""
    return normalize_frames(frame[None], IMAGENET_MEAN, IMAGENET_STD)[0]


class ADE20KImages(Dataset):
    """→ frame (3, S, S) [0, 1], mask (S, S) long."""

    def __init__(self, pairs: List[Pair], size: int):
        self.pairs, self.size = pairs, size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img, mask = self.pairs[idx]
        return load_image(img, self.size), load_mask(mask, self.size)


# ---------------------------------------------------------------------------
# Feature cache: <split>_feats.f16 (N, C, h, w), <split>_masks.u8 (N, S, S), <split>_meta.json (written last)
# ---------------------------------------------------------------------------

def _cache_paths(cache_dir: PathLike, split: str) -> Tuple[Path, Path, Path]:
    d = Path(cache_dir)
    return d / f"{split}_feats.f16", d / f"{split}_masks.u8", d / f"{split}_meta.json"


def _stems_sha1(pairs: List[Pair]) -> str:
    return hashlib.sha1("\n".join(p[0].stem for p in pairs).encode()).hexdigest()


def _backbone_id(backbone) -> str:
    return getattr(backbone, "timm_name", None) or backbone.name


def _read_complete_meta(cache_dir: PathLike, split: str) -> Optional[Dict]:
    """The meta dict if meta and both data files exist with the expected byte sizes, else None."""
    feats, masks, meta_path = _cache_paths(cache_dir, split)
    if not (meta_path.exists() and feats.exists() and masks.exists()):
        return None
    meta = json.loads(meta_path.read_text())
    if feats.stat().st_size != meta["n"] * int(np.prod(meta["feat_shape"])) * 2:    # fp16
        return None
    if masks.stat().st_size != meta["n"] * meta["size"] ** 2:                       # uint8
        return None
    return meta


def cache_meta(cache_dir: PathLike, split: str) -> Optional[Dict]:
    """The cache's meta (n, size, feat_shape, backbone, stems_sha1), or None if the cache is incomplete."""
    return _read_complete_meta(cache_dir, split)


def cache_complete(cache_dir: PathLike, split: str) -> bool:
    return _read_complete_meta(cache_dir, split) is not None


def _check_meta(meta: Dict, pairs: List[Pair], size: int, backbone_id: Optional[str] = None) -> None:
    got = dict(n=len(pairs), size=size, stems_sha1=_stems_sha1(pairs))
    if backbone_id is not None:
        got["backbone"] = backbone_id
    bad = {k: (meta[k], v) for k, v in got.items() if meta[k] != v}
    if bad:
        raise ValueError(f"feature cache doesn't match this run (cached, wanted): {bad}")


@torch.no_grad()
def build_feature_cache(backbone, pairs: List[Pair], cache_dir: PathLike, split: str,
                        batch_size: int, num_workers: int = 2) -> Path:
    """
    Run the backbone once over pairs; store fp16 features and uint8 masks. Skips if a complete, matching cache exists.
    Data goes to *.part files, renamed when finished, then the meta is written: an interrupted run never looks complete.
    """
    size = backbone.input_size[0]
    if backbone.input_size[0] != backbone.input_size[1]:
        raise ValueError(f"square input expected, got {backbone.input_size}")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta = _read_complete_meta(cache_dir, split)
    if meta is not None:
        _check_meta(meta, pairs, size, _backbone_id(backbone))
        return cache_dir

    feats_path, masks_path, meta_path = _cache_paths(cache_dir, split)
    feats_part, masks_part = Path(f"{feats_path}.part"), Path(f"{masks_path}.part")
    meta_path.unlink(missing_ok=True)
    n = len(pairs)
    masks_mm = np.memmap(masks_part, dtype="uint8", mode="w+", shape=(n, size, size))
    feats_mm = None
    loader = DataLoader(ADE20KImages(pairs, size), batch_size=batch_size,   # ↓ images per backbone call → memory. Independent of training
                        shuffle=False, num_workers=num_workers)
    i = 0
    for frames, masks in loader:                                            # (B, 3, S, S), (B, S, S)
        feats = backbone(frames).features.half().cpu().numpy()              # (B, C, h, w) fp16
        if not np.isfinite(feats).all():
            raise ValueError(f"non-finite fp16 features in batch starting at item {i}")
        if feats_mm is None:
            feat_shape = list(feats.shape[1:])
            feats_mm = np.memmap(feats_part, dtype="float16", mode="w+", shape=(n, *feat_shape))
        feats_mm[i:i + len(feats)] = feats
        masks_mm[i:i + len(feats)] = masks.numpy().astype("uint8")
        i += len(feats)
    feats_mm.flush()
    masks_mm.flush()
    del feats_mm, masks_mm
    os.replace(feats_part, feats_path)
    os.replace(masks_part, masks_path)
    meta = dict(split=split, n=n, size=size, feat_shape=feat_shape,
                backbone=_backbone_id(backbone), stems_sha1=_stems_sha1(pairs))
    tmp = Path(f"{meta_path}.part")
    tmp.write_text(json.dumps(meta, indent=2))
    os.replace(tmp, meta_path)
    return cache_dir


def copy_cache(src: PathLike, dst: PathLike, split: str) -> None:
    """Copy one split's cache from src to dst, meta last. Raises if src is incomplete."""
    if not cache_complete(src, split):
        raise ValueError(f"no complete {split} cache in {src}")
    Path(dst).mkdir(parents=True, exist_ok=True)
    for s, d in zip(_cache_paths(src, split), _cache_paths(dst, split)):    # feats, masks, meta
        shutil.copyfile(s, d)


class CachedFeats(Dataset):
    """→ guide (3, S, S) ImageNet-normalized, feats (C, h, w) fp32, mask (S, S) long."""

    def __init__(self, cache_dir: PathLike, split: str, pairs: List[Pair], size: int,
                 backbone_id: Optional[str] = None):
        meta = _read_complete_meta(cache_dir, split)
        if meta is None:
            raise ValueError(f"no complete {split} cache in {cache_dir}")
        _check_meta(meta, pairs, size, backbone_id)
        feats_path, masks_path, _ = _cache_paths(cache_dir, split)
        n = meta["n"]
        self.pairs, self.size = pairs, size
        self.feats = np.memmap(feats_path, dtype="float16", mode="r", shape=(n, *meta["feat_shape"]))
        self.masks = np.memmap(masks_path, dtype="uint8", mode="r", shape=(n, size, size))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        guide = guide_from_image(load_image(self.pairs[idx][0], self.size))
        feat = torch.from_numpy(self.feats[idx].astype("float32"))
        mask = torch.from_numpy(self.masks[idx].astype("int64"))
        return guide, feat, mask
