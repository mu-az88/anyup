

import torch
from anyup.data.crop_sampler import SpatiotemporalCropSampler, SpatiotemporalCrop


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_video_and_features(
    T=16, H=224, W=224, h_p=28, w_p=28, C=64
):
    """Make a dummy video and coarse feature volume."""
    video = torch.rand(T, H, W, 3)    # (T, H, W, 3)
    p     = torch.rand(T, h_p, w_p, C)  # (T, h_p, w_p, C)
    return video, p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_output_shapes():
    video, p = _make_video_and_features()  # spatial_ratio = 224 // 28 = 8
    sampler = SpatiotemporalCropSampler(n_frames=4, crop_size=(7, 7), temporal_stride=1)
    crop = sampler.sample(video, p)

    assert crop.video_full.shape == (16, 224, 224, 3), crop.video_full.shape
    assert crop.p_crop.shape     == (4, 7, 7, 64),     crop.p_crop.shape
    assert crop.video_crop.shape == (4, 56, 56, 3),    crop.video_crop.shape  # 7 * 8 = 56
    print("PASS test_output_shapes")


def test_coordinate_alignment():
    """
    The critical invariant: the pixel crop in video space must correspond
    exactly to the feature crop in feature space.
    We inject known values and verify the slices match.
    """
    T, H, W, h_p, w_p, C = 8, 64, 64, 8, 8, 4
    r = H // h_p   # = 8

    # Build a feature volume where p[t, fh, fw, :] = fh * 100 + fw
    p = torch.zeros(T, h_p, w_p, C)
    for fh in range(h_p):
        for fw in range(w_p):
            p[:, fh, fw, :] = fh * 100 + fw

    # Build a video where video[t, vh, vw, :] = (vh // r) * 100 + (vw // r)
    # → same value as the feature patch that covers this pixel
    video = torch.zeros(T, H, W, 3)
    for vh in range(H):
        for vw in range(W):
            video[:, vh, vw, :] = (vh // r) * 100 + (vw // r)

    sampler = SpatiotemporalCropSampler(n_frames=4, crop_size=(3, 3), temporal_stride=1)

    for _ in range(20):   # multiple random seeds
        crop = sampler.sample(video, p)
        c = crop.coords

        # Every feature value in p_crop should equal the video values in video_crop
        # (both encode fh * 100 + fw for the patch they cover)
        p_vals    = crop.p_crop[..., 0]                              # (T', 3, 3)
        video_vals = crop.video_crop[:, ::r, ::r, 0]                 # (T', 3, 3) — top-left pixel of each patch

        assert torch.allclose(p_vals, video_vals), (
            f"Alignment failure at coords fh0={c['fh0']}, fw0={c['fw0']}\n"
            f"p_vals:\n{p_vals[0]}\nvideo_vals:\n{video_vals[0]}"
        )

    print("PASS test_coordinate_alignment — pixel/feature alignment verified over 20 random crops")


def test_video_full_is_unmodified():
    """video_full must be the original tensor, not a copy of the crop."""
    video, p = _make_video_and_features()
    sampler = SpatiotemporalCropSampler(n_frames=4, crop_size=(7, 7))
    crop = sampler.sample(video, p)
    assert crop.video_full.data_ptr() == video.data_ptr(), \
        "video_full should be the same tensor (no copy)"
    print("PASS test_video_full_is_unmodified")


def test_temporal_stride():
    """With stride=2, frame_indices should be spaced by 2."""
    video, p = _make_video_and_features(T=16)
    sampler = SpatiotemporalCropSampler(n_frames=4, crop_size=(7, 7), temporal_stride=2)
    crop = sampler.sample(video, p)
    indices = crop.coords["frame_indices"]
    diffs = [indices[i+1] - indices[i] for i in range(len(indices) - 1)]
    assert all(d == 2 for d in diffs), f"Expected stride=2, got diffs={diffs}"
    print("PASS test_temporal_stride — diffs:", diffs)


def test_coords_consistency():
    """vh0 must equal fh0 * r_h, etc."""
    video, p = _make_video_and_features()
    sampler = SpatiotemporalCropSampler(n_frames=4, crop_size=(7, 7))
    crop = sampler.sample(video, p)
    c = crop.coords
    r_h, r_w = c["spatial_ratio_h"], c["spatial_ratio_w"]

    assert c["vh0"] == c["fh0"] * r_h
    assert c["vh1"] == c["fh1"] * r_h
    assert c["vw0"] == c["fw0"] * r_w
    assert c["vw1"] == c["fw1"] * r_w
    print("PASS test_coords_consistency — video/feature coords are consistent")


def test_raises_on_bad_spatial_ratio():
    """Non-integer spatial ratio should raise cleanly."""
    video = torch.rand(8, 225, 224, 3)   # 225 not divisible by 28
    p     = torch.rand(8, 28,  28,  64)
    sampler = SpatiotemporalCropSampler(n_frames=4, crop_size=(7, 7))
    try:
        sampler.sample(video, p)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print("PASS test_raises_on_bad_spatial_ratio:", e)


def test_raises_on_temporal_mismatch():
    video = torch.rand(16, 224, 224, 3)
    p     = torch.rand(8, 28, 28, 64)   # T mismatch
    sampler = SpatiotemporalCropSampler(n_frames=4, crop_size=(7, 7))
    try:
        sampler.sample(video, p)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print("PASS test_raises_on_temporal_mismatch:", e)


if __name__ == "__main__":
    test_output_shapes()
    test_coordinate_alignment()
    test_video_full_is_unmodified()
    test_temporal_stride()
    test_coords_consistency()
    test_raises_on_bad_spatial_ratio()
    test_raises_on_temporal_mismatch()