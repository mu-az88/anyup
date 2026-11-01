"""
Shared metrics for AnyUp3D experiments (task T04). Each score is defined once here.

    DAVIS J&F (official semi-supervised protocol, vendored code below):
        res = eval_clip_jf(gt, pred)            # gt, pred: (N, H, W) uint8 object ids, native 480p
        summary = summarize_jf({clip: res, ...})
    Feature metrics (float64):
        temporal_cosine(feats, boundaries)      # Exp 2
        gap_cosine_fidelity(lr, hr)             # cosine fidelity (VideoMAE viz notebook)
    ADE20K mIoU (Exp 1):  update_conf_mat / compute_miou
    Paired statistics:    paired_stats(ours, base)
    Results I/O:          save_results(out_dir, rows, summary, config)
"""

from __future__ import annotations

import csv
import json
import math
import platform
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ===========================================================================
# Vendored: official DAVIS 2017 evaluation code
#   https://github.com/davisvideochallenge/davis2017-evaluation
#   commit ac7c43fca936f9722837b7fbd337d284ba37004b
#   db_eval_iou, db_eval_boundary, f_measure, _seg2bmap: davis2017/metrics.py
#   db_statistics:                                     davis2017/utils.py
#   Copied verbatim (only the `__main__` demo block is left out).
#
#   BSD 3-Clause License
#   Copyright (c) 2020, DAVIS: Densely Annotated VIdeo Segmentation
#   All rights reserved.
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are met:
#   1. Redistributions of source code must retain the above copyright notice, this
#      list of conditions and the following disclaimer.
#   2. Redistributions in binary form must reproduce the above copyright notice,
#      this list of conditions and the following disclaimer in the documentation
#      and/or other materials provided with the distribution.
#   3. Neither the name of the copyright holder nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#   FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#   DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#   CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#   OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ===========================================================================

def db_eval_iou(annotation, segmentation, void_pixels=None):
    """ Compute region similarity as the Jaccard Index.
    Arguments:
        annotation   (ndarray): binary annotation   map.
        segmentation (ndarray): binary segmentation map.
        void_pixels  (ndarray): optional mask with void pixels

    Return:
        jaccard (float): region similarity
    """
    assert annotation.shape == segmentation.shape, \
        f'Annotation({annotation.shape}) and segmentation:{segmentation.shape} dimensions do not match.'
    annotation = annotation.astype(bool)
    segmentation = segmentation.astype(bool)

    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape, \
            f'Annotation({annotation.shape}) and void pixels:{void_pixels.shape} dimensions do not match.'
        void_pixels = void_pixels.astype(bool)
    else:
        void_pixels = np.zeros_like(segmentation)

    # Intersection between all sets
    inters = np.sum((segmentation & annotation) & np.logical_not(void_pixels), axis=(-2, -1))
    union = np.sum((segmentation | annotation) & np.logical_not(void_pixels), axis=(-2, -1))

    j = inters / union
    if j.ndim == 0:
        j = 1 if np.isclose(union, 0) else j
    else:
        j[np.isclose(union, 0)] = 1
    return j


def db_eval_boundary(annotation, segmentation, void_pixels=None, bound_th=0.008):
    assert annotation.shape == segmentation.shape
    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape
    if annotation.ndim == 3:
        n_frames = annotation.shape[0]
        f_res = np.zeros(n_frames)
        for frame_id in range(n_frames):
            void_pixels_frame = None if void_pixels is None else void_pixels[frame_id, :, :, ]
            f_res[frame_id] = f_measure(segmentation[frame_id, :, :, ], annotation[frame_id, :, :], void_pixels_frame, bound_th=bound_th)
    elif annotation.ndim == 2:
        f_res = f_measure(segmentation, annotation, void_pixels, bound_th=bound_th)
    else:
        raise ValueError(f'db_eval_boundary does not support tensors with {annotation.ndim} dimensions')
    return f_res


def f_measure(foreground_mask, gt_mask, void_pixels=None, bound_th=0.008):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.

    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.
        void_pixels     (ndarray): optional mask with void pixels

    Returns:
        F (float): boundaries F-measure
    """
    assert np.atleast_3d(foreground_mask).shape[2] == 1
    if void_pixels is not None:
        void_pixels = void_pixels.astype(bool)
    else:
        void_pixels = np.zeros_like(foreground_mask).astype(bool)

    bound_pix = bound_th if bound_th >= 1 else \
        np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))

    # Get the pixel boundaries of both masks
    fg_boundary = _seg2bmap(foreground_mask * np.logical_not(void_pixels))
    gt_boundary = _seg2bmap(gt_mask * np.logical_not(void_pixels))

    from skimage.morphology import disk

    # fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
    fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))
    # gt_dil = binary_dilation(gt_boundary, disk(bound_pix))
    gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # % Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall)

    return F


def _seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.
    Arguments:
        seg     : Segments labeled from 1..k.
        width	  :	Width of desired bmap  <= seg.shape[1]
        height  :	Height of desired bmap <= seg.shape[0]
    Returns:
        bmap (ndarray):	Binary boundary map.
     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
    """

    seg = seg.astype(bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (
        width > w | height > h | abs(ar1 - ar2) > 0.01
    ), "Can" "t convert %dx%d seg to %dx%d bmap." % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap


def db_statistics(per_frame_values):
    """ Compute mean,recall and decay from per-frame evaluation.
    Arguments:
        per_frame_values (ndarray): per-frame evaluation

    Returns:
        M,O,D (float,float,float):
            return evaluation statistics: mean,recall,decay.
    """

    # strip off nan values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        M = np.nanmean(per_frame_values)
        O = np.nanmean(per_frame_values > 0.5)

    N_bins = 4
    ids = np.round(np.linspace(1, len(per_frame_values), N_bins + 1) + 1e-10) - 1
    ids = ids.astype(np.uint8)

    D_bins = [per_frame_values[ids[i]:ids[i + 1] + 1] for i in range(0, 4)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        D = np.nanmean(D_bins[0]) - np.nanmean(D_bins[3])

    return M, O, D

# ===========================================================================
# End of vendored code
# ===========================================================================


# ---------------------------------------------------------------------------
# DAVIS J&F wrappers (replicate davis2017/evaluation.py, semi-supervised task)
# ---------------------------------------------------------------------------

DAVIS_VOID_LABEL = 255   # davis2017/davis.py DAVIS.VOID_LABEL


def _one_hot(masks: np.ndarray, n_obj: int) -> np.ndarray:
    """(N, H, W) ids → (n_obj, N, H, W) bool, object k+1 in channel k (davis2017 get_all_masks / read_masks)."""
    tmp = np.ones((n_obj, *masks.shape))
    tmp = tmp * np.arange(1, n_obj + 1)[:, None, None, None]
    return (tmp == masks[None, ...]) > 0


def eval_clip_jf(gt: np.ndarray, pred: np.ndarray, bound_th: float = 0.008) -> Dict:
    """
    Official DAVIS 2017 semi-supervised J and F for one clip.

    gt   : (N, H, W) uint8 GT object ids for ALL frames (0 = bg, 255 = void), native resolution
    pred : (N, H, W) predicted object ids for the same frames, same resolution
    Same steps as DAVISEvaluation.evaluate: void → 0; n_objects = max id of GT frame 0; frames 1:-1 scored;
    predicted objects one-hot over the scored frames, zero-padded to n_objects (more → error).

    Returns: n_objects, n_scored, J / F (n_objects, n_scored) per object per frame,
             J_M/J_R/J_D, F_M/F_R/F_D lists (one value per object, from db_statistics).
    """
    gt = np.asarray(gt)
    pred = np.asarray(pred)
    if gt.ndim != 3 or gt.shape != pred.shape:
        raise ValueError(f"gt and pred must both be (N, H, W) with equal shapes, got {gt.shape} and {pred.shape}")
    if gt.shape[0] < 3:
        raise ValueError(f"need >= 3 frames (first and last are not scored), got {gt.shape[0]}")
    if gt.shape[0] - 2 > 255:
        # db_statistics casts frame ids to uint8 (vendored as is); longer clips would overflow silently
        raise ValueError(f"db_statistics supports <= 255 scored frames, got {gt.shape[0] - 2}")

    gt = gt.astype(np.float64)                     # davis2017 reads masks into float arrays
    gt[gt == DAVIS_VOID_LABEL] = 0                 # void → bg (semi-supervised passes no void mask)
    n_obj = int(np.max(gt[0]))
    gt_oh = _one_hot(gt, n_obj)[:, 1:-1]           # (n_obj, N-2, H, W): drop first and last frame

    pred_s = pred[1:-1].astype(np.float64)         # (N-2, H, W)
    n_pred = int(np.max(pred_s))
    if n_pred > n_obj:
        raise ValueError(f"prediction has object id {n_pred} > number of GT objects {n_obj}")
    pred_oh = _one_hot(pred_s, n_pred)
    if n_pred < n_obj:
        zero_padding = np.zeros((n_obj - n_pred, *pred_oh.shape[1:]))
        pred_oh = np.concatenate([pred_oh, zero_padding], axis=0)

    n_scored = gt_oh.shape[1]
    J = np.zeros((n_obj, n_scored))
    Fb = np.zeros((n_obj, n_scored))
    for ii in range(n_obj):
        J[ii, :] = db_eval_iou(gt_oh[ii, ...], pred_oh[ii, ...], None)
        Fb[ii, :] = db_eval_boundary(gt_oh[ii, ...], pred_oh[ii, ...], None, bound_th=bound_th)

    out = {"n_objects": n_obj, "n_scored": n_scored, "J": J, "F": Fb,
           "J_M": [], "J_R": [], "J_D": [], "F_M": [], "F_R": [], "F_D": []}
    for ii in range(n_obj):
        for key, arr in (("J", J), ("F", Fb)):
            m, r, d = db_statistics(arr[ii])
            out[f"{key}_M"].append(float(m))
            out[f"{key}_R"].append(float(r))
            out[f"{key}_D"].append(float(d))
    return out


def summarize_jf(per_clip: Dict[str, Dict]) -> Dict:
    """
    Aggregate eval_clip_jf results as evaluation_method.py does: every statistic is the mean over
    ALL objects of all clips (clip order = dict order); J&F-Mean = (J-Mean + F-Mean) / 2.

    Returns: J&F-Mean, J-Mean, J-Recall, J-Decay, F-Mean, F-Recall, F-Decay, n_clips, n_objects,
             per_object {"<clip>_<k>": {J, F, JF}}, per_clip {clip: {J, F, JF, n_objects}} (mean over the clip's objects).
    """
    if not per_clip:
        raise ValueError("per_clip is empty")
    stats = {k: [] for k in ("J_M", "J_R", "J_D", "F_M", "F_R", "F_D")}
    per_object, clips = {}, {}
    for clip, res in per_clip.items():
        for k in stats:
            stats[k].extend(res[k])
        for ii in range(res["n_objects"]):
            j, f = res["J_M"][ii], res["F_M"][ii]
            per_object[f"{clip}_{ii + 1}"] = {"J": j, "F": f, "JF": (j + f) / 2}
        cj, cf = float(np.mean(res["J_M"])), float(np.mean(res["F_M"]))
        clips[clip] = {"J": cj, "F": cf, "JF": (cj + cf) / 2, "n_objects": res["n_objects"]}
    m = {k: float(np.mean(v)) for k, v in stats.items()}
    return {
        "J&F-Mean": (m["J_M"] + m["F_M"]) / 2.0,
        "J-Mean": m["J_M"], "J-Recall": m["J_R"], "J-Decay": m["J_D"],
        "F-Mean": m["F_M"], "F-Recall": m["F_R"], "F-Decay": m["F_D"],
        "n_clips": len(per_clip), "n_objects": len(per_object),
        "per_object": per_object, "per_clip": clips,
    }


# ---------------------------------------------------------------------------
# Feature metrics (float64)
# ---------------------------------------------------------------------------

def temporal_cosine(feats: torch.Tensor, boundaries: Optional[Sequence[int]] = None) -> Dict:
    """
    Exp 2 adjacent-frame cosine: per pixel, cos(f_t, f_{t+1}) along C; mean over pixels, then over pairs.

    feats      : (T, C, H, W) upsampled features (any device / dtype; computed in float64)
    boundaries : positions p where a new chunk starts (anyup.video.chunk_boundaries); pair (p-1, p) is "cross"
    Returns: all (headline), within, cross (None if no such pair), per_pair (T-1,) list, n_pairs, n_cross.
    One pair at a time, so peak extra memory ≈ 2 frames in float64 (2·C·H·W·8 bytes).
    """
    if feats.ndim != 4 or feats.shape[0] < 2:
        raise ValueError(f"feats must be (T>=2, C, H, W), got {tuple(feats.shape)}")
    T = feats.shape[0]
    cross = set(boundaries or [])
    bad = [p for p in cross if not 0 < p < T]
    if bad:
        raise ValueError(f"boundaries must satisfy 0 < p < T={T}, got {bad}")

    per_pair = []
    prev = F.normalize(feats[0].double(), dim=0)                 # (C, H, W) unit vectors per pixel (Exp 2: dim=1 of (T, C, H, W))
    for t in range(1, T):
        cur = F.normalize(feats[t].double(), dim=0)              # (C, H, W)
        per_pair.append((prev * cur).sum(dim=0).mean().item())  # (H, W) → mean over pixels
        prev = cur
    within = [s for p, s in enumerate(per_pair, start=1) if p not in cross]
    crossv = [s for p, s in enumerate(per_pair, start=1) if p in cross]
    return {
        "all": float(np.mean(per_pair)),
        "within": float(np.mean(within)) if within else None,
        "cross": float(np.mean(crossv)) if crossv else None,
        "per_pair": per_pair, "n_pairs": len(per_pair), "n_cross": len(crossv),
    }


def gap_cosine_fidelity(lr: torch.Tensor, hr: torch.Tensor) -> Dict:
    """
    Cosine fidelity (VideoMAE viz notebook): per frame, cos(GAP(lr_t), GAP(hr_t)), in float64.

    lr : (T, C, h, w) backbone features;  hr : (T, C, H, W) upsampled features of the same frames
    Returns: per_frame (T,) list, mean.
    """
    if lr.ndim != 4 or hr.ndim != 4 or lr.shape[:2] != hr.shape[:2]:
        raise ValueError(f"lr (T, C, h, w) and hr (T, C, H, W) must share T and C, got {tuple(lr.shape)}, {tuple(hr.shape)}")
    lr_avg = lr.double().mean(dim=(-2, -1))                      # (T, C)
    hr_avg = hr.double().to(lr_avg.device).mean(dim=(-2, -1))    # (T, C)
    cos = F.cosine_similarity(lr_avg, hr_avg, dim=1)             # (T,)
    per_frame = cos.tolist()
    return {"per_frame": per_frame, "mean": float(np.mean(per_frame))}


# ---------------------------------------------------------------------------
# ADE20K mIoU (Exp 1 Cell 7, verbatim apart from the constants becoming kwargs)
# ---------------------------------------------------------------------------

def update_conf_mat(conf_mat, logits, targets, num_classes: int = 150, ignore_index: int = 0):
    """conf_mat (K, K) long CPU, updated in place; logits (B, K, H, W); targets (B, H, W) ids 1..K, 0 = ignore."""
    preds    = logits.argmax(dim=1)
    mask     = (targets != ignore_index)
    tgt_flat = (targets[mask] - 1).cpu()    # shift 1–150 → 0–149
    prd_flat = preds[mask].cpu()
    valid    = (tgt_flat >= 0) & (tgt_flat < num_classes) & \
               (prd_flat >= 0) & (prd_flat < num_classes)
    idx      = tgt_flat[valid] * num_classes + prd_flat[valid]
    conf_mat += torch.bincount(idx, minlength=num_classes * num_classes) \
                     .reshape(num_classes, num_classes)


def compute_miou(conf_mat) -> float:
    """mIoU in %, over classes present in the GT (rows with > 0 pixels)."""
    tp    = conf_mat.diag().float()
    fp    = (conf_mat.sum(0) - conf_mat.diag()).float()
    fn    = (conf_mat.sum(1) - conf_mat.diag()).float()
    iou   = tp / (tp + fp + fn + 1e-10)
    valid = conf_mat.sum(1) > 0
    return iou[valid].mean().item() * 100.0


# ---------------------------------------------------------------------------
# Paired statistics
# ---------------------------------------------------------------------------

def paired_stats(ours: Sequence[float], base: Sequence[float]) -> Dict:
    """
    Paired comparison of per-item scores (clips, objects or seeds), higher = better.

    Sign test: exact binomial on non-tied pairs (ties dropped), p=0.5.
    Wilcoxon signed-rank: scipy default (zero differences dropped).
    p values: two-sided (default report) and one-sided (H1: ours > base). Paired t: two-sided.
    A test that can't be computed (e.g. all pairs tied, n < 2) gives None.
    """
    from scipy import stats

    a = np.asarray(ours, dtype=np.float64)
    b = np.asarray(base, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or a.size == 0:
        raise ValueError(f"ours and base must be equal-length 1-D, got {a.shape} and {b.shape}")
    d = a - b
    n_imp, n_worse = int((d > 0).sum()), int((d < 0).sum())
    n_nz = n_imp + n_worse

    def _try(fn):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p = float(fn())
            return None if math.isnan(p) else p
        except ValueError:
            return None

    return {
        "n": int(a.size), "n_improved": n_imp, "n_worse": n_worse, "n_tied": int(a.size) - n_nz,
        "mean_ours": float(a.mean()), "mean_base": float(b.mean()),
        "mean_delta": float(d.mean()), "median_delta": float(np.median(d)),
        "sign_p_two_sided": stats.binomtest(n_imp, n_nz, 0.5, alternative="two-sided").pvalue if n_nz else None,
        "sign_p_one_sided": stats.binomtest(n_imp, n_nz, 0.5, alternative="greater").pvalue if n_nz else None,
        "wilcoxon_p_two_sided": _try(lambda: stats.wilcoxon(a, b, alternative="two-sided").pvalue) if n_nz else None,
        "wilcoxon_p_one_sided": _try(lambda: stats.wilcoxon(a, b, alternative="greater").pvalue) if n_nz else None,
        "ttest_p_two_sided": _try(lambda: stats.ttest_rel(a, b).pvalue) if a.size >= 2 else None,
    }


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------

def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, torch.Tensor):
        return x.tolist()
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    raise TypeError(f"not JSON-serializable: {type(x).__name__}")


def environment() -> Dict:
    """Versions, GPU and git commit of the anyup repo, for the results JSON."""
    repo = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": commit,
    }


def save_results(out_dir: Union[str, Path], rows: List[Dict], summary: Dict, config: Dict,
                 name: str = "results"):
    """
    Write <out_dir>/<name>_per_item.csv (one row per clip / object / seed; columns = union of row keys,
    in first-seen order) and <out_dir>/<name>_summary.json ({summary, config, env}).
    Returns (csv_path, json_path).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}_per_item.csv"
    json_path = out_dir / f"{name}_summary.json"

    fields: List[str] = []
    for r in rows:
        fields.extend(k for k in r if k not in fields)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(_jsonable(r))
    with open(json_path, "w") as f:
        json.dump(_jsonable({"summary": summary, "config": config, "env": environment()}), f, indent=2)
    return csv_path, json_path
