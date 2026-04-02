import torch
import torch.nn as nn
from torch import einsum
from typing import Optional
from anyup.modules.attention_masking3d import compute_attention_mask_3d


class CrossAttention3D(nn.Module):
    def __init__(self, qk_dim, num_heads,
                 q_chunk_size: Optional[int] = None,
                 store_attn: bool = False):
        super().__init__()
        self.norm_q = nn.RMSNorm(qk_dim)
        self.norm_k = nn.RMSNorm(qk_dim)
        self.q_chunk_size = q_chunk_size
        self.store_attn = store_attn
        self.attention = nn.MultiheadAttention(
            embed_dim=qk_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )

    @torch.no_grad()
    def _slice_mask(self, mask, start, end):
        if mask is None:
            return None
        if mask.dim() == 2:
            return mask[start:end, :]
        elif mask.dim() == 3:
            return mask[:, start:end, :]
        else:
            raise ValueError("attn_mask must be 2D or 3D")

    def forward(self, query, key, value, mask=None,
                q_chunk_size: Optional[int] = None,
                store_attn: Optional[bool] = None):
        q_chunk_size = self.q_chunk_size if q_chunk_size is None else q_chunk_size
        store_attn = self.store_attn if store_attn is None else store_attn

        val = key

        query = self.norm_q(query)
        key = self.norm_k(key)

        if q_chunk_size is None or query.size(1) <= q_chunk_size:
            _, attn = self.attention(query, key, val,
                                     average_attn_weights=True,
                                     attn_mask=mask)
            features = einsum("b i j, b j d -> b i d", attn, value)
            return features, (attn if store_attn else None)

        B, Q, _ = query.shape
        outputs = []
        attns = [] if store_attn else None

        for start in range(0, Q, q_chunk_size):
            end = min(start + q_chunk_size, Q)
            q_chunk = query[:, start:end, :]
            mask_chunk = self._slice_mask(mask, start, end)

            _, attn_chunk = self.attention(q_chunk, key, val,
                                           average_attn_weights=True,
                                           attn_mask=mask_chunk)
            out_chunk = einsum("b i j, b j d -> b i d", attn_chunk, value)
            outputs.append(out_chunk)
            if store_attn:
                attns.append(attn_chunk)

        features = torch.cat(outputs, dim=1)
        attn_scores = torch.cat(attns, dim=1) if store_attn else None
        return features, attn_scores


class CrossAttentionBlock3D(nn.Module):
    def __init__(self, qk_dim, num_heads, window_ratio: float = 0.1,
                 window_t=None, q_chunk_size: Optional[int] = None, **kwargs):
        super().__init__()
        self.cross_attn = CrossAttention3D(qk_dim, num_heads, q_chunk_size=q_chunk_size)
        self.window_ratio = window_ratio
        self.window_t = window_t
        self.conv3d = nn.Conv3d(qk_dim, qk_dim, kernel_size=(1, 3, 3), stride=1,
                                padding=(0, 1, 1), bias=False)

    def forward(self, q, k, v, q_chunk_size: Optional[int] = None,
                store_attn: Optional[bool] = None, **kwargs):
        b, _, t, h, w = q.shape
        _, _, t_k, h_k, w_k = k.shape
        c = v.shape[1]

        q = self.conv3d(q)

        if self.window_ratio > 0:
            attn_mask = compute_attention_mask_3d(
                t, h, w, t_k, h_k, w_k,
                spatial_ratio=self.window_ratio,
                window_t=self.window_t
            ).to(q.device)
        else:
            attn_mask = None

        q = q.permute(0, 2, 3, 4, 1).reshape(b, t * h * w, -1)
        k = k.permute(0, 2, 3, 4, 1).reshape(b, t_k * h_k * w_k, -1)
        v = v.permute(0, 2, 3, 4, 1).reshape(b, t_k * h_k * w_k, -1)

        features, attn = self.cross_attn(q, k, v, mask=attn_mask,
                                          q_chunk_size=q_chunk_size,
                                          store_attn=store_attn)

        features = features.reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3)
        return features
