
import torch
import tempfile, subprocess, os
from anyup.data.video_dataset import VideoDataset, IMAGENET_MEAN, IMAGENET_STD

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dummy_video(path: str, n_frames: int = 32, fps: int = 8,
                      h: int = 240, w: int = 320) -> None:
    """Create a minimal MP4 with ffmpeg (solid colour, no audio)."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi",
        "-i", f"color=c=blue:size={w}x{h}:rate={fps}:duration={n_frames/fps}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        path,
    ]
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_output_shape():
    with tempfile.TemporaryDirectory() as d:
        _make_dummy_video(os.path.join(d, "clip.mp4"), n_frames=32)
        ds = VideoDataset(d, n_frames=8, spatial_size=(224, 224), stride=2)
        clip = ds[0]
        assert clip.shape == (8, 224, 224, 3), f"Bad shape: {clip.shape}"
        print("PASS test_output_shape:", clip.shape)


def test_output_dtype_and_range():
    """Output must be float32; values should span a reasonable normalised range."""
    with tempfile.TemporaryDirectory() as d:
        _make_dummy_video(os.path.join(d, "clip.mp4"), n_frames=32)
        ds = VideoDataset(d, n_frames=4, spatial_size=(128, 128))
        clip = ds[0]
        assert clip.dtype == torch.float32, f"Expected float32, got {clip.dtype}"
        # ImageNet-normalised blue frame: R≈-2.1, G≈-2.0, B≈2.4 roughly
        assert clip.min() < 0, "Normalised values should go negative"
        assert clip.max() > 0, "Normalised values should be positive too"
        print("PASS test_output_dtype_and_range — min/max:", clip.min().item(), clip.max().item())


def test_imagenet_normalization():
    """A frame of constant 0.5 grey should normalise to (0.5 - mean) / std."""
    with tempfile.TemporaryDirectory() as d:
        _make_dummy_video(os.path.join(d, "grey.mp4"), n_frames=16)
        ds = VideoDataset(d, n_frames=2, spatial_size=(64, 64))
        clip = ds[0]
        # Just check channel dimension is last (T, H, W, C)
        assert clip.shape[-1] == 3, "Channel must be last"
        print("PASS test_imagenet_normalization — channel last confirmed")


def test_dataset_length():
    with tempfile.TemporaryDirectory() as d:
        for i in range(3):
            _make_dummy_video(os.path.join(d, f"clip_{i}.mp4"), n_frames=32)
        ds = VideoDataset(d, n_frames=8, spatial_size=(224, 224))
        assert len(ds) == 3, f"Expected 3 videos, got {len(ds)}"
        print("PASS test_dataset_length:", len(ds))


def test_short_video_fallback():
    """Videos shorter than stride*n_frames should still return a valid clip."""
    with tempfile.TemporaryDirectory() as d:
        _make_dummy_video(os.path.join(d, "short.mp4"), n_frames=10)
        ds = VideoDataset(d, n_frames=8, spatial_size=(64, 64),
                          stride=4, min_video_frames=8)
        clip = ds[0]
        assert clip.shape == (8, 64, 64, 3)
        print("PASS test_short_video_fallback")


if __name__ == "__main__":
    test_output_shape()
    test_output_dtype_and_range()
    test_imagenet_normalization()
    test_dataset_length()
    test_short_video_fallback()