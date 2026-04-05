from torch import nn
import torch.nn.functional as F
import torch

from anyup.modules.RoPE3d import RoPE3D


from .layers import ResBlock
from .layers import LearnedFeatureUnification
from .layers import setup_cross_attention_block
from .layers import RoPE
from .layers.attention import CrossAttentionBlock
from .modules import SpatialReflectConv3d, LearnedFeatureUnification3D, ResBlock3D
from .utils.img import create_coordinate, create_coordinates_3d   


class AnyUp(nn.Module):
    def __init__(
            self,
            input_dim=3,
            qk_dim=128,
            kernel_size=1,
            kernel_size_lfu=5,
            window_ratio=0.1,
            num_heads=4,
            init_gaussian_derivatives=False,
            use_natten=False,
            lfu_dim=None,
            t_k=1,
            **kwargs,
    ):
        super().__init__()
        self.qk_dim = qk_dim
        self.t_k = t_k
        self.lfu_dim = lfu_dim if lfu_dim is not None else qk_dim
        self.window_ratio = window_ratio
        self._rb_args = dict(kernel_size=1, num_groups=8, pad_mode="reflect", norm_fn=nn.GroupNorm,
                             activation_fn=nn.SiLU)

        # Encoders
        self.image_encoder = self._make_encoder(input_dim, kernel_size)
        self.key_encoder = self._make_encoder(qk_dim, 1)
        self.query_encoder = self._make_encoder(qk_dim, 1)
        self.key_features_encoder = self._make_encoder(None, 1, first_layer_k=kernel_size_lfu,
                                                       init_gaussian_derivatives=init_gaussian_derivatives)

        # Cross-attention
        self.cross_decode = setup_cross_attention_block(
            use_natten=use_natten,
            qk_dim=qk_dim,
            num_heads=num_heads,
            window_ratio=window_ratio
        )
        self.aggregation = self._make_encoder(2 * qk_dim, 3)

        # RoPE for (H*W, C)
        self.rope = RoPE3D(qk_dim)
        self.rope._device_weight_init()

    def _make_encoder(self, in_ch, k, layers=2, first_layer_k=0, init_gaussian_derivatives=False):
        pre = (
            SpatialReflectConv3d(in_ch, self.qk_dim, k, t_k=self.t_k)
            if first_layer_k == 0 else
            LearnedFeatureUnification3D(
                self.lfu_dim,
                first_layer_k,
                t_kernel_size=self.t_k,
                init_gaussian_derivatives=init_gaussian_derivatives
            )
        )
        blocks = [
            ResBlock3D(
                self.qk_dim if first_layer_k == 0 or i != 0 else self.lfu_dim,
                self.qk_dim,
                **self._rb_args
            )
            for i in range(layers)
        ]
        return nn.Sequential(pre, *blocks)

    def upsample(self, enc_img, feats, out_size, vis_attn=False, q_chunk_size=None):
        b, c, t, h, w = feats.shape                                          # 2D: b, c, h, w

        # Q
        q = F.adaptive_avg_pool3d(                                            # 2D: adaptive_avg_pool2d
            self.query_encoder(enc_img),
            output_size=(t, *out_size)                                        # 2D: out_size
        )

        # K
        k = F.adaptive_avg_pool3d(                                            # 2D: adaptive_avg_pool2d
            self.key_encoder(enc_img),
            output_size=(t, h, w)                                             # 2D: (h, w)
        )
        k = torch.cat([k, self.key_features_encoder(F.normalize(feats, dim=1))], dim=1)
        k = self.aggregation(k)

        # V
        v = feats

        if not isinstance(self.cross_decode, CrossAttentionBlock) and vis_attn:
            import warnings
            warnings.warn("Visualization of attention maps is not supported for NATTEN-based cross-attention.")
            vis_attn = False

        return self.cross_decode(q, k, v, vis_attn=vis_attn, q_chunk_size=q_chunk_size)


    def forward(self, image, features, output_size=None, vis_attn=False, q_chunk_size=None):
        # output_size is still spatial-only (H, W); image.shape[-2:] works for both 4D and 5D
        output_size = output_size if output_size is not None else image.shape[-2:]

        enc = self.image_encoder(image)               # (b, c, t, h, w)
        t, h, w = enc.shape[-3], enc.shape[-2], enc.shape[-1]

        coords = create_coordinates_3d(t, h, w, device=enc.device, dtype=enc.dtype)  # (1, t*h*w, 3)

        # Flatten spatiotemporal tokens: (b, c, t, h, w) → (b, t*h*w, c)
        enc = enc.permute(0, 2, 3, 4, 1).reshape(enc.shape[0], -1, enc.shape[1])
        enc = self.rope(enc, coords)                  # (b, t*h*w, c)

        # Restore 5D layout: (b, t*h*w, c) → (b, c, t, h, w)
        enc = enc.view(enc.shape[0], t, h, w, enc.shape[-1]).permute(0, 4, 1, 2, 3)

        return self.upsample(enc, features, output_size, vis_attn=vis_attn, q_chunk_size=q_chunk_size)
