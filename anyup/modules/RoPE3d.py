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
        freqs_1d = self.theta ** torch.linspace(0, -1, self.dim // 6)  # dim//4 → dim//6
        freqs_1d = torch.cat([freqs_1d, freqs_1d])
        freqs_3d = torch.zeros(3, self.dim)                             # freqs_2d → freqs_3d
        freqs_3d[0,  :self.dim // 3]              = freqs_1d            # z (temporal)
        freqs_3d[1,  self.dim // 3: -self.dim // 3] = freqs_1d         # x (height)
        freqs_3d[2, -self.dim // 3:]              = freqs_1d            # y (width)
        self.freqs.data.copy_(freqs_3d * 2 * torch.pi)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        # coords: (1, t*h*w, 3)  ←  was (1, h*w, 2)
        angle = coords @ self.freqs                                      # (B, t*h*w, dim)
        return x * angle.cos() + rotate_half(x) * angle.sin()