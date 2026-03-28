import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from anyup.modules import SpatialReflectConv3d


def test_output_shape():
    B, C, T, H, W = 2, 4, 5, 16, 16
    out_ch = 8

    for k in [1, 3, 5]:
        for t_k in [1, 3]:
            layer = SpatialReflectConv3d(in_ch=C, out_ch=out_ch, k=k, t_k=t_k)
            x = torch.randn(B, C, T, H, W)
            out = layer(x)
            assert out.shape == (B, out_ch, T, H, W), \
                f"Failed for k={k}, t_k={t_k}: got {out.shape}"


def test_reflect_padding_values():
    layer = SpatialReflectConv3d(in_ch=1, out_ch=1, k=3, t_k=1)
    B, C, T, H, W = 1, 1, 1, 4, 4
    x = torch.randn(B, C, T, H, W)

    # manually apply reflect padding in H and W only
    expected_padded = F.pad(x, (1, 1, 1, 1, 0, 0), mode="reflect")

    # hook to capture the tensor just before conv
    captured = {}
    def hook(module, input, output):
        captured["pre_conv"] = input[0]
    layer.conv.register_forward_hook(hook)

    _ = layer(x)

    assert torch.allclose(captured["pre_conv"], expected_padded), \
        "Padding applied to conv input does not match expected reflect padding"


def test_single_frame_matches_conv2d():
    C, out_ch, k = 4, 8, 3
    layer3d = SpatialReflectConv3d(in_ch=C, out_ch=out_ch, k=k, t_k=1)

    # copy weights into an equivalent Conv2d
    layer2d = nn.Conv2d(C, out_ch, k, padding=k // 2, padding_mode="reflect", bias=False)
    with torch.no_grad():
        # Conv3d weight shape: (out_ch, in_ch, 1, k, k) → squeeze to (out_ch, in_ch, k, k)
        layer2d.weight.copy_(layer3d.conv.weight.squeeze(2))

    x2d = torch.randn(2, C, 16, 16)
    x3d = x2d.unsqueeze(2)           # (B, C, T=1, H, W)

    out2d = layer2d(x2d)
    out3d = layer3d(x3d).squeeze(2)  # back to (B, out_ch, H, W)

    assert torch.allclose(out2d, out3d, atol=1e-5), \
        "3D single-frame output does not match Conv2d reference"
