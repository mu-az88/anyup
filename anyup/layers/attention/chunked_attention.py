import torch
import torch.nn as nn
from torch import einsum
from typing import Optional
from .attention_masking import compute_attention_mask


class CrossAttention(nn.Module):
    '''
    CrossAttention: 
    → Takes flat sequences of Q, K, V 
    → normalises Q and K 
    → runs multi-head dot-product similarity 
    → throws away the internal output 
    → uses only the attention weights to blend the raw V values.
    '''
    def __init__(self, qk_dim, num_heads,
                 q_chunk_size: Optional[int] = None,
                 store_attn: bool = False):
        '''
        The constructor takes: qk_dim = how wide each Q/K vector is (e.g. 128 numbers per position), 
        num_heads = how many parallel "attention heads" to run, 
        q_chunk_size = optional memory-saving parameter, 
        store_attn = whether to save the attention map for visualisation.
        '''
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
        '''
        PyTorch's standard multi-head attention module. 
        batch_first=True means tensors come in as (batch, sequence, channels) rather than (sequence, batch, channels). 
        num_heads means it runs num_heads independent dot-product attentions in parallel, 
        each looking at different "subspaces" — like having 4 people each reading the same document looking for different things.
        '''

    @torch.no_grad()
    def _slice_mask(self, mask, start, end):
        if mask is None:
            return None
        # 2D: (tgt_len, src_len), 3D: (B*num_heads or B, tgt_len, src_len)
        if mask.dim() == 2:
            return mask[start:end, :]
        elif mask.dim() == 3:
            return mask[:, start:end, :]
        else:
            raise ValueError("attn_mask must be 2D or 3D")

    def forward(self, query, key, value, mask=None,
                q_chunk_size: Optional[int] = None,
                store_attn: Optional[bool] = None):
        '''
        Inputs arrive as (B, sequence_length, channels) tensors. 
        Here B is batch size, 
        sequence_length is the number of positions (e.g. 224×224 = 50176 pixels flattened), 
        channels is qk_dim.
        '''
        q_chunk_size = self.q_chunk_size if q_chunk_size is None else q_chunk_size
        store_attn = self.store_attn if store_attn is None else store_attn
        '''
        These two lines just resolve which value to use — the stored default or the one passed in this specific call.
        '''
        val = key

        '''
        Wait — val = key? This looks odd. 
        It's a placeholder: PyTorch's MHA needs something as V to compute internally, so it uses K. 
        But we don't care about MHA's output anyway — we'll override it with the real raw value below via einsum. 
        This is essentially a dummy V for the internal MHA call.

        Notice the unusual design: PyTorch's built-in MultiheadAttention internally projects V and outputs a blended result — 
        but the code throws that output away and only keeps the attention weight map. 
        Then it manually blends the raw, original V using einsum. 
        This is intentional: it preserves the original feature values without distortion from learned projections.

        But why project Q, K, and V at all?
        Here's the key question. If attention is just "find similar patches and blend them", 
        why not compute the dot product directly on the raw inputs?
        The answer is: the raw vectors might not be in a good "shape" for computing similarity. 
        Imagine two feature vectors that are semantically very similar (both represent "a dog's ear") but their raw numbers happen to be very different due to how the backbone was trained. 
        A direct dot product would give a low similarity score — wrong answer.
        The Wq and Wk projections rotate and rescale the vectors into a new space where semantically similar things end up having high dot products. 
        They're essentially asking: "What aspect of this vector is relevant for the question of similarity?"
        Think of it like converting temperature units before comparing — if one number is in Celsius and another in Fahrenheit, 
        comparing them raw is meaningless. The projection is the conversion step.

        Why AnyUp is fine skipping Wv
        Now it all connects. The Wv projection exists in standard transformers to let the model decide "which aspects of V to emphasise when blending". 
        For a language model that's useful — 
        when translating a French word, you might want to emphasise the grammatical aspect of some tokens and the semantic aspect of others.
        But AnyUp's V is a DINOv2 or CLIP feature vector. 
        Every dimension already means something carefully learned by a large pretrained model. 
        Passing it through Wv would mix those dimensions together, producing a new vector that no longer lives in DINOv2's feature space. 
        The whole point of the paper — and the thing that makes AnyUp work with any backbone without retraining — 
        is that the output stays in the same space as the input features. 
        Removing Wv is what makes that possible.
        '''
        query = self.norm_q(query)
        key = self.norm_k(key)

        # Fast path: no chunking
        if q_chunk_size is None or query.size(1) <= q_chunk_size:
            _, attn = self.attention(query, key, val,   # val = key here!
                                     average_attn_weights=True,
                                     attn_mask=mask)
            features = einsum("b i j, b j d -> b i d", attn, value)   # real V here
            return features, (attn if store_attn else None)

        # Chunked over the query length (tgt_len)
        B, Q, _ = query.shape
        outputs = []
        attns = [] if store_attn else None

        for start in range(0, Q, q_chunk_size):
            end = min(start + q_chunk_size, Q)
            q_chunk = query[:, start:end, :]
            mask_chunk = self._slice_mask(mask, start, end)

            # We ignore the MHA output as in JAFAR:
            # use the averaged attention to weight the unprojected V.
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


class CrossAttentionBlock(nn.Module): 
    '''
    CrossAttentionBlock: 
    → Receives 2D image feature maps → applies a 3×3 conv to Q for local context 
    → computes a window mask so each pixel only attends to nearby patches 
    → flattens everything to sequences 
    → calls CrossAttention 
    → reshapes the output back to a 2D feature map.
    '''
    def __init__(self, qk_dim, num_heads, window_ratio: float = 0.1,
                 q_chunk_size: Optional[int] = None, **kwargs):
        super().__init__()
        self.cross_attn = CrossAttention(
            qk_dim, num_heads,
            q_chunk_size=q_chunk_size
        )
        self.window_ratio = window_ratio
        self.conv2d = nn.Conv2d(qk_dim, qk_dim, kernel_size=3, stride=1, padding=1, bias=False)
        '''
        The conv2d here acts as a local context aggregator for the query before cross-attention. Here's why:

        - q represents the feature map of the upsampled/decoded image (spatial query tokens). Before attending to the
        encoder's key/value features, it's useful to let each query token gather information from its immediate spatial       
        neighbors.
        - A 3×3 conv with padding=1 does exactly that — it aggregates each spatial position with its 8 neighbors, giving each 
        query token a richer, spatially-aware representation.
        - Only q goes through this, not k or v, because the goal is to smooth/contextualize the query-side features before the
        cross-attention lookup. This is a common pattern in image-to-image attention (e.g., in super-resolution or
        segmentation decoders) to reduce noise in individual pixel/patch queries.

        In short: it's a lightweight spatial smoothing step on the query features to improve cross-attention quality before   
        flattening into tokens.

        claude clarification:
        what you mean is that we get info from neighboring queries so we can attend better to the similar area in the keys                                                                                                                         
        Exactly. The conv2d lets each query position "see" its local neighborhood before attending to the keys, so spatially  
        coherent regions in q produce coherent attention patterns into k/v. Instead of each pixel querying independently,
        nearby queries agree on what they're looking for — which naturally concentrates attention on the corresponding local region in the key/value feature map.
        '''

    def forward(self, q, k, v, q_chunk_size: Optional[int] = None, store_attn: Optional[bool] = None, vis_attn=False,
                **kwargs):
        store_attn = store_attn or vis_attn
        q = self.conv2d(q)
        '''
        Apply the 3×3 conv to Q. Shape stays (B, 128, H_out, W_out) — 
        same spatial size, same channels, but each position now has context from its neighbours.
        '''
        if self.window_ratio > 0:
            attn_mask = compute_attention_mask(
                *q.shape[-2:], *k.shape[-2:], window_size_ratio=self.window_ratio
            ).to(q.device)
        else:
            attn_mask = None
        b, _, h, w = q.shape
        _, _, h_k, w_k = k.shape
        c = v.shape[1]
        q = q.permute(0, 2, 3, 1).view(b, h * w, -1)
        k = k.permute(0, 2, 3, 1).view(b, h_k * w_k, -1)
        v = v.permute(0, 2, 3, 1).view(b, h_k * w_k, -1)

        features, attn = self.cross_attn(q, k, v, mask=attn_mask,
                                         q_chunk_size=q_chunk_size,
                                         store_attn=store_attn)
        features = features.view(b, h, w, c).permute(0, 3, 1, 2)
        if vis_attn:
            from anyup.utils.visualization import visualize_attention_oklab
            import matplotlib.pyplot as plt

            ref, out = visualize_attention_oklab(attn[0], h, w, h_k, w_k)

            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(ref.cpu().numpy())
            ax[0].set_title("Reference (Values)")
            ax[0].set_xticks([-.5, w_k - .5], labels=[0, w_k])
            ax[0].set_yticks([-.5, h_k - .5], labels=[0, h_k])

            ax[1].imshow(out.cpu().numpy())
            ax[1].set_title("Attention Output")
            ax[1].set_xticks([-.5, w - .5], labels=[0, w])
            ax[1].set_yticks([-.5, h - .5], labels=[0, h])
            plt.show()

        return features
