"""
T04 pytest: shared DAVIS utilities (anyup/davis.py).

CPU only, a tiny fake DAVIS tree in tmp_path, no downloads. Run on Colab:

    python -m pytest tests/test_davis.py -v
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from anyup import davis
from anyup.metrics import _one_hot

SEED = 0
N, H, W = 5, 32, 48          # ↓ fake clip: 5 frames of 32×48 (native size)
TAU, TOP_K, N_LAST_FRAMES = 0.07, 7, 6   # Exp 3 Cell 2 values (used by the copied Exp 3 code)


# ─── fake DAVIS tree ────────────────────────────────────────────────────────

def _palette():
    pal = np.zeros((256, 3), np.uint8)
    pal[1], pal[2] = (128, 0, 0), (0, 128, 0)
    return pal.flatten().tolist()


def _gt(t):
    m = np.zeros((H, W), np.uint8)
    m[4 + t:12 + t, 4:12] = 1          # object 1 moves down 1 px per frame
    m[16:24, 30:38] = 2                # object 2 static
    return m


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "DAVIS"
    (r / "ImageSets" / "2017").mkdir(parents=True)
    (r / "ImageSets" / "2017" / "val.txt").write_text("clipB\nclipA\n")
    g = np.random.default_rng(SEED)
    for clip in ["clipA", "clipB"]:
        fd, ad = davis.frames_dir(r, clip), davis.ann_dir(r, clip)
        fd.mkdir(parents=True)
        ad.mkdir(parents=True)
        for t in range(N):
            Image.fromarray(g.integers(0, 256, (H, W, 3), dtype=np.uint8)).save(fd / f"{t:05d}.jpg")
            a = Image.fromarray(_gt(t), mode="P")
            a.putpalette(_palette())
            a.save(ad / f"{t:05d}.png")
    return r


# ─── paths + clips ──────────────────────────────────────────────────────────

def test_list_clips_file_order(root):
    assert davis.list_clips(root, "val") == ["clipB", "clipA"]
    with pytest.raises(FileNotFoundError):
        davis.list_clips(root, "train")


def test_paths_sorted_and_open_clip(root):
    fp, ap = davis.frame_paths(root, "clipA"), davis.ann_paths(root, "clipA")
    assert [p.stem for p in fp] == [f"{t:05d}" for t in range(N)] == [p.stem for p in ap]
    src = davis.open_clip(root, "clipA")
    assert len(src) == N and tuple(src.native_size) == (H, W)
    frames = src.get(range(N))
    assert frames.shape == (N, 3, H, W) and frames.dtype == torch.float32


def test_download_skips_when_present(root):
    assert davis.download_davis(root.parent, url="http://invalid.invalid/none.zip") == root


# ─── annotations ────────────────────────────────────────────────────────────

def test_load_annotations(root):
    gt = davis.load_annotations(root, "clipA")
    assert gt.shape == (N, H, W) and gt.dtype == np.uint8
    assert all(np.array_equal(gt[t], _gt(t)) for t in range(N))


def test_load_annotations_count_mismatch(root):
    (davis.ann_dir(root, "clipA") / f"{N - 1:05d}.png").unlink()
    with pytest.raises(ValueError, match="annotations"):
        davis.load_annotations(root, "clipA")


def test_num_objects_ignores_void():
    assert davis.num_objects(np.array([[0, 1], [2, 255]], np.uint8)) == 2
    assert davis.num_objects(np.zeros((2, 2), np.uint8)) == 0


def test_one_hot_matches_metrics_layout():
    m = np.stack([_gt(t) for t in range(N)])
    assert np.array_equal(davis.one_hot_objects(m, 2), _one_hot(m.astype(np.float64), 2))


def test_resize_labels_nearest():
    m = _gt(0)
    r = davis.resize_labels(m, (8, 12))
    assert r.shape == (8, 12) and set(np.unique(r)) <= {0, 1, 2}
    assert np.array_equal(r, cv2.resize(m, (12, 8), interpolation=cv2.INTER_NEAREST))
    assert davis.resize_labels(m, 16).shape == (16, 16)


def test_save_mask_png_round_trip(root, tmp_path):
    ref = davis.ann_paths(root, "clipA")[0]
    small = torch.from_numpy(davis.resize_labels(_gt(2), (8, 12))).long()   # (8, 12) int64 like propagate_labels
    out = tmp_path / "res" / "clipA" / "00002.png"
    davis.save_mask_png(small, ref, out)
    img = Image.open(out)
    assert img.mode == "P" and img.getpalette() == Image.open(ref).getpalette()
    got = np.array(img)
    assert got.shape == (H, W)
    assert np.array_equal(got, cv2.resize(small.numpy().astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST))
    with pytest.raises(ValueError, match="palette"):
        davis.save_mask_png(np.full((8, 12), 300), ref, out)


# ─── label propagation vs Exp 3 ─────────────────────────────────────────────

def _exp3_reference(root, clip):
    """Exp 3 Cell 7 propagate_clip + build_label_maps_frame0 + load_ann, verbatim (comments trimmed)."""
    def load_ann(ann_path):
        return np.array(Image.open(ann_path))

    def build_label_maps_frame0(ann0_HW, n_objects, H, W):
        ann_resized = cv2.resize(
            ann0_HW, (W, H), interpolation=cv2.INTER_NEAREST
        ).astype(np.int64)
        n_ch    = n_objects + 1
        N       = H * W
        label   = np.zeros((n_ch, N), dtype=np.float32)
        ann_flat = ann_resized.flatten()
        for obj_id in range(n_ch):
            label[obj_id, ann_flat == obj_id] = 1.0
        return torch.from_numpy(label)

    @torch.no_grad()
    def propagate_clip(feats_TCHW, ann_paths, sampled_idxs):
        T, C, H, W = feats_TCHW.shape
        N = H * W
        feats_flat = feats_TCHW.view(T, C, N).permute(0, 2, 1)
        feats_flat = F.normalize(feats_flat, dim=-1)
        ann0 = load_ann(ann_paths[sampled_idxs[0]])
        n_objects = int(ann0.max())
        n_ch = n_objects + 1
        label_maps = torch.zeros(T, n_ch, N)
        label_maps[0] = build_label_maps_frame0(ann0, n_objects, H, W)
        for t in range(1, T):
            past      = list(range(max(1, t - N_LAST_FRAMES), t))
            context_t = [0] + past
            ctx_feats  = feats_flat[context_t].reshape(-1, C)
            ctx_labels = label_maps[context_t].permute(1, 0, 2).reshape(n_ch, -1)
            q = feats_flat[t]
            A = torch.mm(q, ctx_feats.T) / TAU
            topk_vals, topk_idx = A.topk(TOP_K, dim=-1)
            A_sparse = torch.full_like(A, float("-inf"))
            A_sparse.scatter_(1, topk_idx, topk_vals)
            A_soft = F.softmax(A_sparse, dim=-1)
            new_labels = torch.mm(A_soft, ctx_labels.T)
            label_maps[t] = new_labels.T
        pred = label_maps.argmax(dim=1)
        return pred.view(T, H, W)

    return propagate_clip, davis.ann_paths(root, clip)


@pytest.mark.parametrize("T", [2, 9])          # 9 > 1 + N_LAST_FRAMES: context window slides
def test_propagation_matches_exp3(root, T):
    g = torch.Generator().manual_seed(SEED)
    feats = torch.randn(T, 16, 8, 12, generator=g)      # (T, C, H, W) upsampled features
    ref_fn, ap = _exp3_reference(root, "clipA")
    ref = ref_fn(feats, ap, list(range(T)))
    got = davis.propagate_labels(feats, davis.load_annotation(ap[0]))
    assert got.dtype == torch.int64 and got.shape == (T, 8, 12)
    assert torch.equal(got, ref)
    assert torch.equal(got[0], torch.from_numpy(davis.resize_labels(_gt(0), (8, 12))).long())   # frame 0 = GT


def test_propagation_kwargs_change_result(root):
    g = torch.Generator().manual_seed(SEED)
    feats = torch.randn(6, 16, 8, 12, generator=g)
    ann0 = _gt(0)
    a = davis.propagate_labels(feats, ann0)
    b = davis.propagate_labels(feats, ann0, top_k=1)
    assert a.shape == b.shape and not torch.equal(a, b)


def test_propagation_rejects_void():
    ann0 = _gt(0)
    ann0[0, 0] = 255
    with pytest.raises(ValueError, match="void"):
        davis.propagate_labels(torch.randn(2, 4, 8, 12), ann0)
