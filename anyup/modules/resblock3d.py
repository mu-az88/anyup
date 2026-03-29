import torch
from torch import nn
import torch.nn.functional as F

from .conv3d_reflect import SpatialReflectConv3d


def _make_conv3d(in_ch, out_ch, kernel_size, pad_mode, t_k=1):
    """
    Returns a Conv3d-based layer that respects pad_mode.
    When pad_mode='reflect', uses SpatialReflectConv3d (manual padding).
    Otherwise uses plain Conv3d with built-in padding.
    """
    if pad_mode == "reflect":
        # SpatialReflectConv3d handles padding manually — no built-in padding
        return SpatialReflectConv3d(in_ch, out_ch, k=kernel_size, t_k=t_k)
    else:
        p = kernel_size // 2
        return nn.Conv3d(
            in_ch, out_ch, (t_k, kernel_size, kernel_size),
            padding=(t_k // 2, p, p),
            padding_mode=pad_mode,
            bias=False
        )


class ResBlock3D(nn.Module):
    """
    3D extension of ResBlock.

    Operates on (B, C, T, H, W) tensors.
    All Conv2d layers replaced with Conv3d equivalents.
    GroupNorm and SiLU are shape-agnostic and unchanged.
    Reflect padding is handled via SpatialReflectConv3d from Unit 1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        t_kernel_size: int = 1,
        num_groups: int = 8,
        pad_mode: str = "zeros",
        norm_fn=None,
        activation_fn=nn.SiLU,
        use_conv_shortcut: bool = False,
    ):
        super().__init__()
        self.t_kernel_size = t_kernel_size

        N = (lambda c: norm_fn(num_groups, c)) if norm_fn else (lambda c: nn.Identity())

        self.block = nn.Sequential(
            N(in_channels),
            activation_fn(),
            _make_conv3d(in_channels, out_channels, kernel_size, pad_mode, t_kernel_size),
            N(out_channels),
            activation_fn(),
            _make_conv3d(out_channels, out_channels, kernel_size, pad_mode, t_kernel_size),
        )

        # shortcut: always kernel_size=1 so padding=0 — plain Conv3d is safe regardless of pad_mode
        self.shortcut = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
            if use_conv_shortcut or in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        return self.block(x) + self.shortcut(x)