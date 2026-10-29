"""
Video reader + chunked backbone runner for AnyUp3D experiments (task T02).

    src = open_video(".../JPEGImages/480p/bear")    # frame folder | video file | (T, 3, H, W) tensor
    bb  = BackboneLoader.load("videomae", device="cuda")
    for out in iter_backbone_chunks(bb, src, chunk_size=None, overlap=0):
        out.features   # (t_out, C, h, w)   padded and overlap-context outputs already dropped
        out.guide      # (t_out, 3, gH, gW)
        out.meta       # T01 keys + chunk keys; meta["frame_index"] is GLOBAL

Every frame is used (no skipping) and output exactly once. Chunks are loaded one
at a time, so memory stays constant in the video length (except for the
torchvision fallback of VideoFileSource, which decodes the whole file).

AnyUp3D policy (T02 Q13/Q18): one yielded chunk = one AnyUp3D call (T03), with no
temporal context across chunk boundaries for AnyUp3D.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch
from PIL import Image
import torchvision.transforms.functional as TF

from anyup.backbones import BackboneAdapter, BackboneOutput

DEFAULT_CHUNK_SIZE = 16   # ↓ frames per backbone call for variable-T backbones; sets backbone batch size and AnyUp3D T per call (memory). Dependent: overlap < chunk_size; T03 q_chunk_size

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

class VideoSource:
    """
    Common interface:
      len(src)          number of frames
      src.native_size   (H, W) of the stored frames
      src.fps           float, or None when unknown (frame folders, tensors)
      src.get(indices)  (n, 3, H, W) float32 in [0, 1], CPU
      src.name          path string, or "tensor"
    """

    name: str = ""
    fps: Optional[float] = None

    def __len__(self) -> int:
        raise NotImplementedError

    @property
    def native_size(self) -> Tuple[int, int]:
        raise NotImplementedError

    def get(self, indices: Sequence[int]) -> torch.Tensor:
        raise NotImplementedError


class TensorSource(VideoSource):
    """In-memory (T, 3, H, W) tensor, float in [0, 1] or uint8."""

    def __init__(self, frames: torch.Tensor):
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError(f"frames must be (T, 3, H, W), got {tuple(frames.shape)}")
        if frames.dtype == torch.uint8:
            frames = frames.float() / 255.0
        elif not frames.is_floating_point():
            raise ValueError(f"frames must be float in [0, 1] or uint8, got dtype {frames.dtype}")
        self.frames = frames.float()
        self.name = "tensor"

    def __len__(self):
        return self.frames.shape[0]

    @property
    def native_size(self):
        return tuple(self.frames.shape[-2:])

    def get(self, indices):
        return self.frames[list(indices)].cpu()                     # (n, 3, H, W)


class FrameFolderSource(VideoSource):
    """
    Folder of image frames, e.g. DAVIS JPEGImages/480p/<clip>.
    Ordering = sorted(glob), decoding = PIL .convert("RGB") / 255 — same as Exp 2/3.
    """

    def __init__(self, path: Union[str, Path], ext: str = ".jpg"):
        self.path = Path(path)
        self.files = sorted(self.path.glob(f"*{ext}"))
        if not self.files:
            raise ValueError(f"No *{ext} frames in {self.path}")
        self.name = str(self.path)
        with Image.open(self.files[0]) as im:
            self._size = (im.height, im.width)

    def __len__(self):
        return len(self.files)

    @property
    def native_size(self):
        return self._size

    def get(self, indices):
        frames = []
        for i in indices:
            with Image.open(self.files[i]) as im:
                frames.append(TF.pil_to_tensor(im.convert("RGB")))    # (3, H, W) uint8
        return torch.stack(frames).float() / 255.0                  # (n, 3, H, W)


class VideoFileSource(VideoSource):
    """
    Video file (mp4, avi, ...). decord with per-chunk get_batch (constant memory);
    falls back to torchvision.io.read_video, which decodes the WHOLE file into memory.
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.name = str(self.path)
        self._vr = None
        self._video = None
        try:
            import decord
            decord.bridge.set_bridge("torch")
            self._vr = decord.VideoReader(str(self.path), ctx=decord.cpu(0))
            self.backend = "decord"
            self.fps = float(self._vr.get_avg_fps())
            h, w = self._vr[0].shape[:2]
            self._size = (int(h), int(w))
        except Exception as e:
            warnings.warn(f"decord failed on {self.path} ({e!r}); falling back to torchvision.io "
                          f"(whole video decoded into memory)")
            from torchvision.io import read_video
            video, _, info = read_video(str(self.path), output_format="TCHW", pts_unit="sec")  # (T, 3, H, W) uint8
            self._video = video
            self.backend = "torchvision"
            self.fps = float(info["video_fps"]) if "video_fps" in info else None
            self._size = tuple(video.shape[-2:])

    def __len__(self):
        return len(self._vr) if self._vr is not None else self._video.shape[0]

    @property
    def native_size(self):
        return self._size

    def get(self, indices):
        indices = list(indices)
        if self._vr is not None:
            frames = self._vr.get_batch(indices).permute(0, 3, 1, 2)   # (n, H, W, 3) → (n, 3, H, W) uint8
        else:
            frames = self._video[indices]                              # (n, 3, H, W) uint8
        return frames.float() / 255.0


def open_video(x: Union[str, Path, torch.Tensor, VideoSource], ext: str = ".jpg") -> VideoSource:
    """Dispatch: VideoSource → as is; tensor → TensorSource; dir → FrameFolderSource; file → VideoFileSource."""
    if isinstance(x, VideoSource):
        return x
    if isinstance(x, torch.Tensor):
        return TensorSource(x)
    p = Path(x)
    if p.is_dir():
        return FrameFolderSource(p, ext=ext)
    if p.is_file():
        if p.suffix.lower() not in VIDEO_EXTS:
            warnings.warn(f"Unrecognized video extension {p.suffix!r}; trying to decode anyway")
        return VideoFileSource(p)
    raise FileNotFoundError(f"{p} is neither a directory nor a file")


# ---------------------------------------------------------------------------
# Chunk runner
# ---------------------------------------------------------------------------

def chunk_starts(n_frames: int, chunk_size: int, overlap: int) -> List[int]:
    """Start index of every chunk. Stops once a chunk reaches the end, so no chunk is only context."""
    step = chunk_size - overlap
    starts = [0]
    while starts[-1] + chunk_size < n_frames:
        starts.append(starts[-1] + step)
    return starts


def iter_backbone_chunks(
    bb: BackboneAdapter,
    src: Union[str, Path, torch.Tensor, VideoSource],
    chunk_size: Optional[int] = None,
    overlap: int = 0,
) -> Iterator[BackboneOutput]:
    """
    Run `bb` on every frame of `src`, one chunk at a time.

    chunk_size : frames per backbone call. Fixed-T backbones (VideoMAE, bb.NUM_FRAMES=16)
                 must use their NUM_FRAMES; others default to DEFAULT_CHUNK_SIZE.
    overlap    : input frames shared with the previous chunk (default 0). Must be a multiple
                 of bb.temporal_stride and < chunk_size. Overlapped outputs are kept from the
                 EARLIER chunk; the later chunk drops its first `overlap` frames' outputs, so the
                 overlap is backbone-only context. Cost ≈ chunk_size / (chunk_size - overlap) ×.

    Last chunk of a fixed-T backbone is padded with its last frame; padded outputs are dropped.
    """
    src = open_video(src)
    n_total = len(src)
    fixed_t = getattr(bb, "NUM_FRAMES", None)   # VideoMAE: 16 (fixed position embeddings)

    if fixed_t is not None:
        if chunk_size is not None and chunk_size != fixed_t:
            raise ValueError(f"{bb.name} needs chunk_size={fixed_t}, got {chunk_size}")
        if n_total < fixed_t:
            raise ValueError(f"{bb.name} needs at least {fixed_t} frames, video {src.name} has {n_total}")
        chunk_size = fixed_t
    elif chunk_size is None:
        chunk_size = DEFAULT_CHUNK_SIZE
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if not 0 <= overlap < chunk_size:
        raise ValueError(f"overlap must satisfy 0 <= overlap < chunk_size={chunk_size}, got {overlap}")
    if overlap % bb.temporal_stride:
        raise ValueError(f"overlap must be a multiple of temporal_stride={bb.temporal_stride}, got {overlap}")
    if n_total < 1:
        raise ValueError(f"video {src.name} has no frames")

    starts = chunk_starts(n_total, chunk_size, overlap)
    for chunk_id, start in enumerate(starts):
        idx = list(range(start, min(start + chunk_size, n_total)))   # global indices of real frames
        n_valid = len(idx)
        n_ctx = 0 if chunk_id == 0 else overlap                      # leading frames already output by the previous chunk
        n_pad = fixed_t - n_valid if fixed_t is not None else 0

        frames = src.get(idx + [idx[-1]] * n_pad)                    # (n_valid + n_pad, 3, H, W) [0, 1]
        out = bb(frames)

        local = out.meta["frame_index"]                              # input-frame index (in chunk) per output
        keep = [k for k, f in enumerate(local) if n_ctx <= f < n_valid]   # drop context and padded outputs
        meta = dict(out.meta)
        meta.update(
            local_frame_index=[local[k] for k in keep],
            frame_index=[start + local[k] for k in keep],            # GLOBAL frame index per kept output
            t_out=len(keep),
            chunk_id=chunk_id,
            chunk_start=start,
            chunk_size=chunk_size,
            overlap=overlap,
            n_ctx=n_ctx,
            n_valid=n_valid,
            n_pad=n_pad,
            is_last=chunk_id == len(starts) - 1,
            n_chunks=len(starts),
            n_frames_total=n_total,
            native_size=tuple(src.native_size),
            fps=src.fps,
            source=src.name,
        )
        yield BackboneOutput(features=out.features[keep], guide=out.guide[keep], meta=meta)


def chunk_boundaries(metas: Sequence[Dict]) -> List[int]:
    """
    Positions in the concatenated output (torch.cat of features over chunks, in order) where a
    new chunk starts, excluding 0. A pair (p-1, p) for p in the result crosses a chunk boundary.
    """
    bounds, pos = [], 0
    for m in metas:
        if m["chunk_id"] > 0:
            bounds.append(pos)
        pos += m["t_out"]
    return bounds
