import torch
from torch import nn
import torch.nn.functional as F


class LearnedFeatureUnification3D(nn.Module):
    """
    3D extension of LearnedFeatureUnification.

    Operates on (B, C, T, H, W) tensors.
    Basis shape: (out_channels, 1, t_k, k, k)
    Each input channel is convolved with every basis filter (depthwise conv3d).
    Softmax is taken across basis filters (dim=1), then averaged across channels (dim=2).
    Output: (B, out_channels, T, H, W)

    Gaussian derivative initialization is stubbed — to be implemented in Unit 5.
    """

    def __init__(
        self,
        out_channels: int,
        kernel_size: int = 3,
        t_kernel_size: int = 1,
        init_gaussian_derivatives: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.t_kernel_size = t_kernel_size

        if init_gaussian_derivatives:
            raise NotImplementedError(
                "3D Gaussian derivative initialization is not yet implemented. "
                "This will be added in Unit 5. Use init_gaussian_derivatives=False for now."
            )

        self.basis = nn.Parameter(
            torch.randn(out_channels, 1, t_kernel_size, kernel_size, kernel_size)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (B, C, T, H, W)
        b, c, t, h, w = features.shape

        # depthwise conv3d → (B, out_channels * C, T, H, W)
        x = self._depthwise_conv(features, self.basis, self.kernel_size, self.t_kernel_size)

        # separate basis and channel dims → (B, out_channels, C, T, H, W)
        x = x.view(b, self.out_channels, c, t, h, w)

        # softmax across basis filters: which filter best describes each spatiotemporal patch?
        attn = F.softmax(x, dim=1)

        # average across input channels: output is invariant to input dimensionality
        # → (B, out_channels, T, H, W)
        return attn.mean(dim=2)

    @staticmethod
    def _depthwise_conv(
        feats: torch.Tensor,
        basis: torch.Tensor,
        k: int,
        t_k: int,
    ) -> torch.Tensor:
        b, c, t, h, w = feats.shape
        p  = k   // 2      # spatial padding
        pt = t_k // 2      # temporal padding (0 when t_k=1)

        # pad: F.pad order for 5D is (W_left, W_right, H_top, H_bot, T_front, T_back)
        x = F.pad(feats, (p, p, p, p, pt, pt), value=0)

        # basis: (out_channels, 1, t_k, k, k)
        # repeat across C so each channel gets its own copy of all basis filters
        # → (out_channels * C, 1, t_k, k, k)
        weight = basis.repeat(c, 1, 1, 1, 1)

        # depthwise conv3d: groups=c keeps channels independent
        # → (B, out_channels * C, T, H, W)
        x = F.conv3d(x, weight, groups=c)

        # denominator: counts how many kernel cells contributed to each position
        # corrects for zero-padded borders so boundary values aren't artificially smaller
        mask = torch.ones(1, 1, t, h, w, dtype=feats.dtype, device=feats.device)
        mask_padded = F.pad(mask, (p, p, p, p, pt, pt), value=0)
        ones_kernel = torch.ones(1, 1, t_k, k, k, dtype=feats.dtype, device=feats.device)
        denom = F.conv3d(mask_padded, ones_kernel)

        return x / denom  # (B, out_channels * C, T, H, W)