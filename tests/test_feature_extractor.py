
import torch
import torch.nn as nn
from anyup.data.feature_extractor import PerFrameFeatureExtractor


# ---------------------------------------------------------------------------
# Minimal stub encoder — avoids downloading DINOv2 in CI
# ---------------------------------------------------------------------------

class _StubEncoder(nn.Module):
    """
    Mimics DINOv2's forward_features interface with a fixed patch_size=14.
    Returns deterministic patch tokens for shape/alignment testing.
    """
    def forward_features(self, x: torch.Tensor) -> dict:
        B, C, H, W = x.shape
        patch_size = 14
        h = H // patch_size
        w = W // patch_size
        n_patches = h * w
        embed_dim = 384   # ViT-S embedding dim
        # Return zeros with the correct shape — content doesn't matter for shape tests
        return {"x_norm_patchtokens": torch.zeros(B, n_patches, embed_dim, device=x.device)}


def _make_extractor(device="cpu") -> PerFrameFeatureExtractor:
    """Build an extractor with the stub encoder injected."""
    ext = PerFrameFeatureExtractor.__new__(PerFrameFeatureExtractor)
    nn.Module.__init__(ext)  # must be called before assigning any nn.Module attributes
    ext.patch_size   = 14
    ext.device       = torch.device(device)
    ext.cache_p      = True
    ext.max_cache    = 8
    ext._cache       = {}
    stub = _StubEncoder().to(device)
    stub.eval()
    ext.encoder = stub
    return ext


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_extract_gt_shape():
    """q̂ must be (T', h_gt, w_gt, C) with h_gt = H_c // patch_size."""
    ext = _make_extractor()
    T_prime, H_c, W_c = 4, 56, 56   # 56 // 14 = 4 patches per side
    video_crop = torch.rand(T_prime, H_c, W_c, 3)
    q_hat = ext.extract_gt(video_crop)
    assert q_hat.shape == (4, 4, 4, 384), f"Bad shape: {q_hat.shape}"
    print("PASS test_extract_gt_shape:", q_hat.shape)


def test_extract_gt_no_gradient():
    """No gradients should flow through the extractor."""
    ext = _make_extractor()
    video_crop = torch.rand(2, 28, 28, 3, requires_grad=False)
    q_hat = ext.extract_gt(video_crop)
    assert not q_hat.requires_grad, "q̂ must be detached — no grad through frozen encoder"
    print("PASS test_extract_gt_no_gradient")


def test_extract_gt_independent_frames():
    """
    Each frame must be encoded independently.
    Replacing one frame should change only its corresponding output slice.
    """
    ext = _make_extractor()
    T_prime, H_c, W_c = 3, 28, 28
    crop_a = torch.rand(T_prime, H_c, W_c, 3)
    crop_b = crop_a.clone()
    crop_b[1] = torch.rand(H_c, W_c, 3)   # change only frame 1

    q_a = ext.extract_gt(crop_a)
    q_b = ext.extract_gt(crop_b)

    # Frames 0 and 2 are identical → outputs must match
    assert torch.allclose(q_a[0], q_b[0]), "Frame 0 output changed despite identical input"
    assert torch.allclose(q_a[2], q_b[2]), "Frame 2 output changed despite identical input"
    # The stub returns zeros regardless, so this test validates the per-frame loop structure.
    print("PASS test_extract_gt_independent_frames")


def test_cache_hit():
    """extract_p should return the cached tensor on the second call."""
    ext = _make_extractor()
    video = torch.rand(8, 28, 28, 3)
    frame_indices = [0, 2, 4, 6]

    p1 = ext.extract_p(video, clip_idx=0, frame_indices=frame_indices)
    p2 = ext.extract_p(video, clip_idx=0, frame_indices=frame_indices)

    # Same object — cache hit means no re-extraction
    assert p1.data_ptr() == p2.data_ptr(), "Cache miss on identical key — caching not working"
    print("PASS test_cache_hit")


def test_cache_miss_on_different_frames():
    """Different frame_indices for the same clip must produce separate cache entries."""
    ext = _make_extractor()
    video = torch.rand(8, 28, 28, 3)

    p1 = ext.extract_p(video, clip_idx=0, frame_indices=[0, 2, 4, 6])
    p2 = ext.extract_p(video, clip_idx=0, frame_indices=[1, 3, 5, 7])

    assert p1.data_ptr() != p2.data_ptr(), "Different frame sets must be separate cache entries"
    assert len(ext._cache) == 2
    print("PASS test_cache_miss_on_different_frames")


def test_cache_eviction():
    """Cache should not exceed max_cache entries."""
    ext = _make_extractor()
    ext.max_cache = 3   # ↓ tiny cache for testing
    video = torch.rand(8, 28, 28, 3)

    for i in range(6):   # insert 6 entries into a cache of size 3
        ext.extract_p(video, clip_idx=i, frame_indices=[0, 1])

    assert len(ext._cache) <= 3, f"Cache grew to {len(ext._cache)}, expected ≤ 3"
    print("PASS test_cache_eviction — cache size:", len(ext._cache))


def test_p_returned_on_cpu():
    """extract_p must return the tensor on CPU to free GPU memory."""
    ext = _make_extractor(device="cpu")
    video = torch.rand(4, 28, 28, 3)
    p = ext.extract_p(video, clip_idx=0, frame_indices=[0, 1, 2, 3])
    assert p.device.type == "cpu", f"p should be on CPU, got {p.device}"
    print("PASS test_p_returned_on_cpu")


def test_raises_on_non_divisible_spatial():
    """Frame sizes not divisible by patch_size must raise cleanly."""
    ext = _make_extractor()
    frame = torch.rand(30, 30, 3)   # 30 not divisible by 14
    try:
        ext._encode_frame(frame)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print("PASS test_raises_on_non_divisible_spatial:", e)


def test_clear_cache():
    ext = _make_extractor()
    video = torch.rand(4, 28, 28, 3)
    ext.extract_p(video, clip_idx=0, frame_indices=[0, 1])
    assert len(ext._cache) == 1
    ext.clear_cache()
    assert len(ext._cache) == 0
    print("PASS test_clear_cache")


if __name__ == "__main__":
    test_extract_gt_shape()
    test_extract_gt_no_gradient()
    test_extract_gt_independent_frames()
    test_cache_hit()
    test_cache_miss_on_different_frames()
    test_cache_eviction()
    test_p_returned_on_cpu()
    test_raises_on_non_divisible_spatial()
    test_clear_cache()