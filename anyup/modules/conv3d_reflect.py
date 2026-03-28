import torch
from torch import nn
import torch.nn.functional as F


class SpatialReflectConv3d(nn.Module):
    """
    Drop-in 3D replacement for:
        nn.Conv2d(in_ch, out_ch, k, padding=k//2, padding_mode="reflect", bias=False)

    Pads H and W with reflect, T with replicate, then applies Conv3d with no built-in padding.
    """
    def __init__(self, in_ch: int, out_ch: int, k: int, t_k: int = 1):
        super().__init__()
        self.pad_hw = k // 2
        self.pad_t  = t_k // 2         # 0 when t_k=1, 1 when t_k=3
        self.conv   = nn.Conv3d(in_ch, out_ch, (t_k, k, k), padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        p, pt = self.pad_hw, self.pad_t

        # reflect in H and W
        x = F.pad(x, (p, p, p, p, 0, 0), mode="reflect")

        # replicate in T (only has effect when t_k > 1)
        if pt > 0:
            x = F.pad(x, (0, 0, 0, 0, pt, pt), mode="replicate")

        return self.conv(x)
