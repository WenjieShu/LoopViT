
from typing import Optional, Tuple
import math
import torch
from torch import nn
import torch.nn.functional as F

from src.ARC_ViT import ARCTransformerEncoderLayer as DefaultLayer, MultiHeadSelfAttention
from src.ARC_ViT1 import MultiHeadSelfAttention as MultiHeadSelfAttention_v1, RMSNorm, ConvolutionalGLU

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

        # Use rotary embeddings from parent if available, or just ignore if it was built-in logic
        # src.ARC_ViT has self.rotary
        q = self.rotary(q)
        k = self.rotary(k)

        attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: (B, L)
            # attn_mask: (B, 1, 1, L)
            attn_mask = key_padding_mask.view(batch_size, 1, 1, seq_len).expand(-1, self.num_heads, -1, -1)
            # 1 -> ignore, 0 -> keep.
            # SDPA expects boolean mask where True indicates elements to ATTEND to? No.
            # Documentation: "Binary mask of shape (batch, src_len) where 1 indicates valid elements" 
            # OR "Boolean mask where True indicates values to participate in attention".
            # BUT standard PyTorch `masked_fill` uses True for padding.
            # Let's stick to standard manual attention computation for extraction to be safe and clear.
        
        # Manual Attention
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if attn_mask is not None:
             # attn_mask is True for padding in simple implementations often
             # src.ARC_ViT helper prepares pad_mask where True = padding
             # We need to mask out padding with -inf
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
        # Use simple creation or copy weights?
        # Ideally we construct it same way.
        # Let's rely on standard construction args usually passed.
        # But wait, we need to match initialization.
        # Creating a fresh one is fine if we load state dict later (which we do).
        
        # We need to ensure we used correct init arguments. 
        # Inspecting `ARCTransformerEncoderLayer.__init__` args:
        # embed_dim, num_heads, mlp_dim, dropout, max_seq_len, no_rope
        
        # We have to re-instantiate because we can't easily swap the class of an existing instance 
        # while keeping method overrides clean if `forward` logic differs.
        
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


# --- NEW: V1 Support (ConvGLU + RMSNorm) ---

class MultiHeadSelfAttentionWithAttn_v1(MultiHeadSelfAttention_v1):
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
             # key_padding_mask: True means ignore (padding)
             # We want standard attention mask logic: -inf where mask is True
             attn_mask = key_padding_mask.view(batch_size, 1, 1, seq_len).expand(-1, self.num_heads, -1, -1)

        # Manual Attention for Hook
        # (B, H, L, D) @ (B, H, D, L) -> (B, H, L, L)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        if attn_mask is not None:
             # Assuming key_padding_mask is boolean (True = Pad)
             attn = attn.masked_fill(attn_mask, float("-inf"))

        attn_weights = attn.softmax(dim=-1)

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


class ARCTransformerEncoderLayerWithAttn_v1(nn.Module):
    # Mimics ARCTransformerEncoderLayer from ARC_ViT1
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        max_seq_len: int,
        no_rope: int = 1,
        grid_size: int = 0
    ) -> None:
        super().__init__()
        self.num_task_tokens = no_rope
        
        self.self_attn = MultiHeadSelfAttentionWithAttn_v1(
            embed_dim=embed_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            no_rope=no_rope,
        )
        self.dropout1 = nn.Dropout(dropout)
        
        # RMSNorm
        self.norm1 = RMSNorm(embed_dim)
        
        # ConvolutionalGLU
        self.mlp = ConvolutionalGLU(
            in_features=embed_dim,
            hidden_features=mlp_dim,
            act_layer=nn.GELU,
            drop=dropout,
            num_task_tokens=no_rope
        )
        
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False
    ) -> torch.Tensor:
        # 1. Attention Block
        residual = x
        x = self.norm1(x)
        x, attn_weights = self.self_attn(x, key_padding_mask=key_padding_mask, return_attn=return_attn)
        x = residual + self.dropout1(x)

        # 2. Hybrid MLP Block
        residual = x
        x = self.norm2(x)
        
        # Calc grid size
        seq_len = x.shape[1]
        num_img_tokens = seq_len - self.num_task_tokens
        
        # Safe sqrt
        if num_img_tokens > 0:
             grid_size = int(math.sqrt(num_img_tokens))
        else:
             grid_size = 0 # Should not happen in valid flow
             
        x = self.mlp(x, H=grid_size, W=grid_size)
        x = residual + self.dropout2(x)
        
        if return_attn:
            return x, attn_weights, attn_weights
        return x