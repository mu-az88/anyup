import torch
from torch import nn


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RoPE3D(nn.Module):
    def __init__(
        self,
        dim: int,
        theta: int = 100,
    ):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.freqs = nn.Parameter(torch.empty(3, self.dim))  # 2 → 3 axes

    def _device_weight_init(self):
        block = self.dim // 3
        last_block = self.dim - 2 * block   # = block + (dim % 3), always >= block

        # canonical frequency sequence for one block
        freqs_1d = self.theta ** torch.linspace(0, -1, block // 2)
        freqs_1d = torch.cat([freqs_1d, freqs_1d])   # length = block (even blocks only)

        # cycle freqs_1d to fill any target size — same repeat logic, just longer
        def fill(size):
            reps = (size + len(freqs_1d) - 1) // len(freqs_1d)   # ceiling division
            return freqs_1d.repeat(reps)[:size]

        freqs_3d = torch.zeros(3, self.dim)
        freqs_3d[0,      :block   ] = fill(block)       # z
        freqs_3d[1,  block:block*2] = fill(block)       # x
        freqs_3d[2, block*2:      ] = fill(last_block)  # y + overflow cycles back
        self.freqs.data.copy_(freqs_3d * 2 * torch.pi)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        # coords: (1, t*h*w, 3)  ←  was (1, h*w, 2)
        angle = coords @ self.freqs                                      # (B, t*h*w, dim)
        return x * angle.cos() + rotate_half(x) * angle.sin()