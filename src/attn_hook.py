from typing import Optional, Tuple
import math
import torch
from torch import nn
import torch.nn.functional as F

from src.ARC_ViT import ARCTransformerEncoderLayer as DefaultLayer, MultiHeadSelfAttention

class MultiHeadSelfAttentionWithAttn(MultiHeadSelfAttention):
    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.rotary(q)
        k = self.rotary(k)

        attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: (B, L)
            # attn_mask: (B, 1, 1, L)
            attn_mask = key_padding_mask.view(batch_size, 1, 1, seq_len).expand(-1, self.num_heads, -1, -1)
            # 1 -> ignore, 0 -> keep.
        
        # Manual Attention
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if attn_mask is not None:
             attn = attn.masked_fill(attn_mask, float("-inf"))

        attn_weights = attn.softmax(dim=-1)
        
        # Dropout
        if self.training:
            attn_dropped = self.attn_dropout(attn_weights)
        else:
            attn_dropped = attn_weights

        x_out = (attn_dropped @ v)
        x_out = x_out.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        x_out = self.proj(x_out)
        x_out = self.proj_dropout(x_out)

        if return_attn:
            return x_out, attn_weights
        return x_out, None


class ARCTransformerEncoderLayerWithAttn(DefaultLayer):
    """
    Wraps the standard ARC ViT layer but replaces self_attn with the hooking version.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace self_attn
        old_attn = self.self_attn
        new_attn = MultiHeadSelfAttentionWithAttn(
            embed_dim=old_attn.embed_dim,
            num_heads=old_attn.num_heads,
            dropout=old_attn.attn_dropout.p, # Approximate
            max_seq_len=kwargs.get('max_seq_len', 1024), # Need from args
            no_rope=kwargs.get('no_rope', 1),
        )
        self.self_attn = new_attn

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False
    ) -> torch.Tensor:
        # Match ARCTransformerEncoderLayer.forward structure (Post-Norm)
        residual = x
        x, attn_weights = self.self_attn(x, key_padding_mask=key_padding_mask, return_attn=return_attn)
        x = residual + self.dropout1(x)
        x = self.norm1(x)

        residual = x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        x = residual + self.dropout3(x)
        x = self.norm2(x)
        
        if return_attn:
            return x, attn_weights, attn_weights 
        return x