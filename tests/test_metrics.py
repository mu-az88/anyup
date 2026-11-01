"""
T04 pytest: shared metrics (anyup/metrics.py).

CPU only, synthetic data, no downloads. Run on Colab:

    python -m pytest tests/test_metrics.py -v
"""

import csv
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from scipy import stats

from anyup.metrics import (_one_hot, compute_miou, db_statistics, eval_clip_jf, gap_cosine_fidelity,
                           paired_stats, save_results, summarize_jf, temporal_cosine, update_conf_mat)

SEED = 0
N, H, W = 5, 32, 48   # ↓ tiny clip: 5 frames (3 scored), 32×48 masks; F tolerance = ceil(0.008·diag) = 1 px


# ─── helpers ────────────────────────────────────────────────────────────────

def square(mask, obj, y, x, s):
    mask[..., y:y + s, x:x + s] = obj
    return mask


def two_object_gt():
    gt = np.zeros((N, H, W), np.uint8)
    square(gt, 1, 4, 4, 8)      # object 1: 8×8
    square(gt, 2, 16, 24, 4)    # object 2: 4×4 = 16 px
    return gt


# ─── DAVIS J&F ──────────────────────────────────────────────────────────────

def test_identical_masks_score_one():
    gt = two_object_gt()
    r = eval_clip_jf(gt, gt.copy())
    assert r["n_objects"] == 2 and r["n_scored"] == N - 2
    assert r["J"].shape == (2, N - 2) and r["F"].shape == (2, N - 2)
    assert np.all(r["J"] == 1) and np.all(r["F"] == 1)


def test_empty_prediction_scores_zero():
    gt = two_object_gt()
    r = eval_clip_jf(gt, np.zeros_like(gt))
    assert np.all(r["J"] == 0) and np.all(r["F"] == 0)


def test_empty_gt_and_empty_pred_is_one():
    gt = two_object_gt()
    gt[1:, gt[0] == 2] = 0             # object 2 disappears after frame 0
    pred = gt.copy()
    r = eval_clip_jf(gt, pred)
    assert np.all(r["J"][1] == 1) and np.all(r["F"][1] == 1)   # official convention: empty vs empty = 1


def test_first_and_last_frame_not_scored():
    gt = two_object_gt()
    pred = gt.copy()
    pred[0] = 0
    pred[-1] = 0                       # wrong on frames 0 and N-1 only
    r = eval_clip_jf(gt, pred)
    assert np.all(r["J"] == 1) and np.all(r["F"] == 1)
    pred[2] = 0                        # wrong on a scored frame
    assert eval_clip_jf(gt, pred)["J"][0, 1] == 0


def test_per_object_aggregation_hand_computed():
    gt = two_object_gt()
    pred = gt.copy()
    pred[:, 16:20, 26:28] = 0          # object 2: keep 2 of 4 columns → J = 8/16 per frame
    a = eval_clip_jf(gt, pred)
    assert a["J_M"] == [1.0, 0.5]

    gt_b = np.zeros((N, H, W), np.uint8)
    square(gt_b, 1, 2, 2, 6)
    b = eval_clip_jf(gt_b, np.zeros_like(gt_b))   # one object, J = 0
    s = summarize_jf({"a": a, "b": b})
    assert s["n_clips"] == 2 and s["n_objects"] == 3
    assert s["J-Mean"] == pytest.approx((1.0 + 0.5 + 0.0) / 3)          # official: mean over objects
    assert s["per_clip"]["a"]["J"] == pytest.approx(0.75)                # per clip: mean over its objects
    assert s["per_clip"]["b"]["J"] == 0.0
    assert set(s["per_object"]) == {"a_1", "a_2", "b_1"}
    assert s["J&F-Mean"] == pytest.approx((s["J-Mean"] + s["F-Mean"]) / 2)


def test_void_is_background_and_pred_padding():
    gt = two_object_gt()
    gt[:, 28:, :] = 255                # void band
    pred = gt.copy()
    pred[pred == 255] = 0
    pred[pred == 2] = 0                # object 2 never predicted → n_pred = 1, zero-padded
    r = eval_clip_jf(gt, pred)
    assert r["n_objects"] == 2
    assert np.all(r["J"][0] == 1) and np.all(r["J"][1] == 0)


def test_pred_id_above_gt_objects_raises():
    gt = two_object_gt()
    pred = gt.copy()
    pred[2, 0, 0] = 3
    with pytest.raises(ValueError, match="object id 3"):
        eval_clip_jf(gt, pred)


def test_jf_input_errors():
    gt = two_object_gt()
    with pytest.raises(ValueError):
        eval_clip_jf(gt, gt[:, :, :-1])
    with pytest.raises(ValueError, match=">= 3 frames"):
        eval_clip_jf(gt[:2], gt[:2])


def test_one_hot_layout():
    m = np.array([[[0, 1], [2, 1]]])                     # (1, 2, 2)
    oh = _one_hot(m.astype(np.float64), 2)               # (2, 1, 2, 2)
    assert oh.dtype == bool and oh.shape == (2, 1, 2, 2)
    assert oh[0, 0].tolist() == [[False, True], [False, True]]
    assert oh[1, 0].tolist() == [[False, False], [True, False]]


def test_db_statistics_hand_computed():
    m, r, d = db_statistics(np.array([1.0, 1.0, 0.0, 0.0, 0.6]))
    assert m == pytest.approx(0.52)
    assert r == pytest.approx(0.6)
    assert d == pytest.approx(1.0 - 0.3)                 # bins [0:2] mean 1.0, [3:5] mean 0.3


# ─── temporal cosine ────────────────────────────────────────────────────────

def test_temporal_cosine_constant_video():
    f = torch.randn(1, 8, 6, 6).repeat(4, 1, 1, 1)
    r = temporal_cosine(f)
    assert r["n_pairs"] == 3 and r["all"] == pytest.approx(1.0, abs=1e-12)
    assert r["cross"] is None and r["n_cross"] == 0


def test_temporal_cosine_boundary_split():
    f = torch.randn(8, 6, 6)
    feats = torch.stack([f, f, -f, -f])                  # pairs: +1, -1, +1
    r = temporal_cosine(feats, boundaries=[2])           # pair (1, 2) crosses a chunk boundary
    assert r["per_pair"] == pytest.approx([1.0, -1.0, 1.0], abs=1e-12)
    assert r["all"] == pytest.approx(1 / 3, abs=1e-12)
    assert r["within"] == pytest.approx(1.0, abs=1e-12)
    assert r["cross"] == pytest.approx(-1.0, abs=1e-12) and r["n_cross"] == 1


def test_temporal_cosine_matches_exp2_definition():
    def adjacent_cosine_sim(feats):                      # Exp 2 Cell 6, verbatim
        feats_norm = F.normalize(feats, dim=1)
        sim        = (feats_norm[:-1] * feats_norm[1:]).sum(dim=1)
        return sim.mean().item()
    g = torch.Generator().manual_seed(SEED)
    feats = torch.randn(6, 16, 8, 8, generator=g)
    assert temporal_cosine(feats)["all"] == pytest.approx(adjacent_cosine_sim(feats), abs=1e-6)


def test_temporal_cosine_large_identical_not_above_one():
    f = torch.randn(1, 768, 64, 64).repeat(2, 1, 1, 1)   # float32 one-shot cosine of this size can exceed 1
    v = temporal_cosine(f)["all"]
    assert abs(v - 1.0) < 1e-12


def test_temporal_cosine_errors():
    with pytest.raises(ValueError):
        temporal_cosine(torch.randn(1, 4, 2, 2))
    with pytest.raises(ValueError, match="boundaries"):
        temporal_cosine(torch.randn(3, 4, 2, 2), boundaries=[3])


# ─── GAP cosine fidelity ────────────────────────────────────────────────────

def test_gap_fidelity_values():
    lr = torch.randn(3, 8, 4, 4)
    hr = lr.repeat_interleave(4, dim=2).repeat_interleave(4, dim=3)   # nearest ×4 → same GAP
    r = gap_cosine_fidelity(lr, hr)
    assert r["per_frame"] == pytest.approx([1.0] * 3, abs=1e-12)
    assert gap_cosine_fidelity(lr, -hr)["mean"] == pytest.approx(-1.0, abs=1e-12)
    with pytest.raises(ValueError):
        gap_cosine_fidelity(lr, hr[:2])


# ─── ADE20K mIoU ────────────────────────────────────────────────────────────

NUM_CLASSES, IGNORE_INDEX = 150, 0

def _exp1_update_conf_mat(conf_mat, logits, targets):     # Exp 1 Cell 7, verbatim
    preds    = logits.argmax(dim=1)
    mask     = (targets != IGNORE_INDEX)
    tgt_flat = (targets[mask] - 1).cpu()
    prd_flat = preds[mask].cpu()
    valid    = (tgt_flat >= 0) & (tgt_flat < NUM_CLASSES) & \
               (prd_flat >= 0) & (prd_flat < NUM_CLASSES)
    idx      = tgt_flat[valid] * NUM_CLASSES + prd_flat[valid]
    conf_mat += torch.bincount(idx, minlength=NUM_CLASSES * NUM_CLASSES) \
                     .reshape(NUM_CLASSES, NUM_CLASSES)

def _exp1_compute_miou(conf_mat):                         # Exp 1 Cell 7, verbatim
    tp    = conf_mat.diag().float()
    fp    = (conf_mat.sum(0) - conf_mat.diag()).float()
    fn    = (conf_mat.sum(1) - conf_mat.diag()).float()
    iou   = tp / (tp + fp + fn + 1e-10)
    valid = conf_mat.sum(1) > 0
    return iou[valid].mean().item() * 100.0


def test_miou_matches_exp1():
    g = torch.Generator().manual_seed(SEED)
    a = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    b = torch.zeros_like(a)
    for _ in range(3):
        logits = torch.randn(2, NUM_CLASSES, 16, 16, generator=g)
        targets = torch.randint(0, NUM_CLASSES + 1, (2, 16, 16), generator=g)
        update_conf_mat(a, logits, targets)
        _exp1_update_conf_mat(b, logits, targets)
    assert torch.equal(a, b)
    assert compute_miou(a) == _exp1_compute_miou(b)


def test_miou_perfect():
    targets = torch.randint(1, 5, (1, 8, 8))
    logits = F.one_hot(targets - 1, NUM_CLASSES).permute(0, 3, 1, 2).float()
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    update_conf_mat(cm, logits, targets)
    assert compute_miou(cm) == pytest.approx(100.0)


# ─── paired statistics ──────────────────────────────────────────────────────

def test_sign_test_thesis_values():
    base = np.linspace(0.2, 0.5, 30)
    r = paired_stats(base + 0.1, base)
    assert r["n_improved"] == 30 and r["n_worse"] == 0
    assert r["sign_p_one_sided"] == pytest.approx(0.5 ** 30, rel=1e-9)        # 9.31e-10 (thesis p)
    ours = base + np.r_[np.full(27, 0.1), np.full(3, -0.1)]
    r27 = paired_stats(ours, base)
    assert r27["sign_p_one_sided"] == pytest.approx(4526 / 2 ** 30, rel=1e-9)  # 4.22e-6
    assert r27["sign_p_two_sided"] == pytest.approx(2 * 4526 / 2 ** 30, rel=1e-9)


def test_paired_stats_match_scipy_and_ties():
    g = np.random.default_rng(SEED)
    base = g.random(20)
    ours = base + g.normal(0.05, 0.1, 20)
    ours[:2] = base[:2]                                  # two ties
    r = paired_stats(ours, base)
    assert r["n_tied"] == 2 and r["n_improved"] + r["n_worse"] == 18
    assert r["sign_p_two_sided"] == pytest.approx(stats.binomtest(r["n_improved"], 18).pvalue)
    assert r["wilcoxon_p_two_sided"] == pytest.approx(stats.wilcoxon(ours, base).pvalue)
    assert r["wilcoxon_p_one_sided"] == pytest.approx(stats.wilcoxon(ours, base, alternative="greater").pvalue)
    assert r["ttest_p_two_sided"] == pytest.approx(stats.ttest_rel(ours, base).pvalue)
    assert r["mean_delta"] == pytest.approx(float(np.mean(ours - base)))


def test_paired_stats_all_tied_and_errors():
    r = paired_stats([0.5, 0.5], [0.5, 0.5])
    assert r["sign_p_two_sided"] is None and r["wilcoxon_p_two_sided"] is None
    with pytest.raises(ValueError):
        paired_stats([1.0, 2.0], [1.0])


# ─── results I/O ────────────────────────────────────────────────────────────

def test_save_results_round_trip(tmp_path):
    rows = [{"clip": "a", "J": np.float64(0.5), "n": np.int64(3)},
            {"clip": "b", "J": 0.25, "F": 0.1}]
    summary = {"J-Mean": np.float32(0.375), "per_clip": {"a": [1, 2]}}
    config = {"tau": 0.07, "path": tmp_path, "size": (56, 56)}
    csv_path, json_path = save_results(tmp_path / "out", rows, summary, config, name="exp3")
    with open(csv_path) as f:
        got = list(csv.DictReader(f))
    assert list(got[0]) == ["clip", "J", "n", "F"]
    assert got[0]["J"] == "0.5" and got[1]["F"] == "0.1" and got[0]["F"] == ""
    data = json.loads(json_path.read_text())
    assert data["summary"]["J-Mean"] == pytest.approx(0.375)
    assert data["config"]["size"] == [56, 56] and data["config"]["path"] == str(tmp_path)
    assert {"torch", "python", "git_commit", "time_utc"} <= set(data["env"])


def test_save_results_rejects_unknown_types(tmp_path):
    with pytest.raises(TypeError):
        save_results(tmp_path, [], {"x": object()}, {})
