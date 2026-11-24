import warnings

import torch
import torch.nn.functional as F
from torch import nn

from .modules import (
    SpatialReflectConv3d,
    LearnedFeatureUnification3D,
    ResBlock3D,
    RoPE3D,
    create_coordinates_3d,
)
from .modules.cross_attention3d import CrossAttentionBlock3D


class AnyUp(nn.Module):
    def __init__(
            self,
            input_dim=3,
            qk_dim=128,
            kernel_size=1,
            kernel_size_lfu=5,
            window_ratio=0.1,
            window_t=3,
            num_heads=4,
            init_gaussian_derivatives=False,
            lfu_dim=None,
            t_k=3,
            **kwargs,
    ):
        super().__init__()
        self.qk_dim = qk_dim
        self.t_k = t_k
        self.lfu_dim = lfu_dim if lfu_dim is not None else qk_dim
        self.window_ratio = window_ratio
        self.window_t = window_t
        self._rb_args = dict(kernel_size=1, num_groups=8, pad_mode="reflect", norm_fn=nn.GroupNorm,
                             activation_fn=nn.SiLU)

        # Encoders
        self.image_encoder = self._make_encoder(input_dim, kernel_size)
        self.key_encoder = self._make_encoder(qk_dim, 1)
        self.query_encoder = self._make_encoder(qk_dim, 1)
        self.key_features_encoder = self._make_encoder(None, 1, first_layer_k=kernel_size_lfu,
                                                       init_gaussian_derivatives=init_gaussian_derivatives)

        # Cross-attention
        self.cross_decode = CrossAttentionBlock3D(
            qk_dim=qk_dim,
            num_heads=num_heads,
            window_ratio=window_ratio,
            window_t=window_t,
        )
        self.aggregation = self._make_encoder(2 * qk_dim, 3)

        # RoPE for (T*H*W, C)
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
                init_gaussian_derivatives=init_gaussian_derivatives,
            )
        )
        blocks = [
            ResBlock3D(
                self.qk_dim if first_layer_k == 0 or i != 0 else self.lfu_dim,
                self.qk_dim,
                **self._rb_args,
            )
            for i in range(layers)
        ]
        return nn.Sequential(pre, *blocks)

    def encode_image(self, image, feats_shape, output_size=None):
        """Image half (T04_training): everything that depends only on the frames.

        image:       (B, 3, T, H, W) guide frames (ImageNet-normalized).
        feats_shape: shape of the features that will be decoded, (B, C, t, h, w); only (t, h, w) is used.
        output_size: (H_out, W_out); None → image size, as in forward.
        Returns (q, k_img): q (B, qk, t, H_out, W_out), k_img (B, qk, t, h, w). Reusable for any features
        on the same (t, h, w) grid, e.g. the main and self-consistency passes of one training step.
        """
        output_size = output_size if output_size is not None else image.shape[-2:]
        _, _, t_f, h_f, w_f = feats_shape

        enc = self.image_encoder(image)               # (B, C, T, H, W)
        t, h, w = enc.shape[-3], enc.shape[-2], enc.shape[-1]

        coords = create_coordinates_3d(t, h, w, device=enc.device, dtype=enc.dtype)

        # Flatten spatiotemporal tokens: (B, C, T, H, W) -> (B, T*H*W, C)
        enc = enc.permute(0, 2, 3, 4, 1).reshape(enc.shape[0], -1, enc.shape[1])
        enc = self.rope(enc, coords)

        # Restore 5D layout: (B, T*H*W, C) -> (B, C, T, H, W)
        enc = enc.view(enc.shape[0], t, h, w, enc.shape[-1]).permute(0, 4, 1, 2, 3)

        # Q
        q = F.adaptive_avg_pool3d(
            self.query_encoder(enc),
            output_size=(t_f, *output_size),
        )                                             # (B, qk, t_f, H_out, W_out)

        # K (image part)
        k_img = F.adaptive_avg_pool3d(
            self.key_encoder(enc),
            output_size=(t_f, h_f, w_f),
        )                                             # (B, qk, t_f, h_f, w_f)
        return q, k_img

    def decode(self, encoded, feats, vis_attn=False, q_chunk_size=None):
        """Feature half (T04_training): everything that depends on the features.

        encoded: (q, k_img) from encode_image, built for features on the same (t, h, w) grid.
        feats:   (B, C, t, h, w) raw low-res features (normalized inside, as before).
        """
        q, k_img = encoded
        if k_img.shape[0] != feats.shape[0] or k_img.shape[-3:] != feats.shape[-3:]:
            raise ValueError(
                f"AnyUp.decode: features (B, t, h, w) = {(feats.shape[0], *feats.shape[-3:])} don't match "
                f"the grid encode_image was called with {(k_img.shape[0], *k_img.shape[-3:])}; "
                f"call encode_image with these features' shape")

        # K
        k = torch.cat([k_img, self.key_features_encoder(F.normalize(feats, dim=1))], dim=1)
        k = self.aggregation(k)

        # V
        v = feats

        if vis_attn:
            warnings.warn("Visualisation of attention maps is not supported for 3D cross-attention.")
            vis_attn = False

        return self.cross_decode(q, k, v, store_attn=vis_attn, q_chunk_size=q_chunk_size)

    def forward(self, image, features, output_size=None, vis_attn=False, q_chunk_size=None):
        encoded = self.encode_image(image, features.shape, output_size)
        return self.decode(encoded, features, vis_attn=vis_attn, q_chunk_size=q_chunk_size)