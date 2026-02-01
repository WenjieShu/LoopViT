from typing import Optional, Tuple
import math
import torch
from torch import nn
from timm.models.vision_transformer import PatchEmbed
from utils.pos_embed import VisionRotaryEmbeddingFast

# 复用 ViT1 的组件
from src.ARC_ViT1 import RMSNorm, ConvolutionalGLU

class RelativePositionBias(nn.Module):
    """
    2D Relative Positional Bias (Simplified for ARC).
    Learnable table of biases added to attention scores based on relative grid distance.
    Supports dynamic grid sizes via index re-generation and clamping.
    """
    def __init__(self, num_heads, h, w):
        super().__init__()
        self.num_heads = num_heads
        self.h = h
        self.w = w
        # (2H-1) * (2W-1) possible relative positions
        self.num_relative_distance = (2 * h - 1) * (2 * w - 1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads)
        )
        
        # Generate relative position index for the default size
        self.register_buffer("relative_position_index", self._generate_relative_position_index(h, w))

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def _generate_relative_position_index(self, h, w):
        coords_h = torch.arange(h)
        coords_w = torch.arange(w)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2, H, W
        coords_flatten = torch.flatten(coords, 1)  # 2, H*W
        
        # relative_coords: [2, H*W, H*W]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # H*W, H*W, 2
        
        # Shift to start from 0, based on the INITIALIZED size (self.h, self.w)
        # This aligns the center of the new grid with the center of the learned table
        relative_coords[:, :, 0] += self.h - 1
        relative_coords[:, :, 1] += self.w - 1
        
        # Clamp values to ensure they fall within the learned table range
        # This handles cases where the input grid is larger than the initialization
        relative_coords[:, :, 0] = relative_coords[:, :, 0].clamp(0, 2 * self.h - 2)
        relative_coords[:, :, 1] = relative_coords[:, :, 1].clamp(0, 2 * self.w - 2)
        
        relative_coords[:, :, 0] *= 2 * self.w - 1
        
        return relative_coords.sum(-1)  # H*W, H*W

    def forward(self, h, w):
        """
        Args:
            h, w: Current grid height and width.
        """
        # If dimensions match the cached ones, use the buffer (fast path)
        if h == self.h and w == self.w:
            index = self.relative_position_index
        else:
            # Dynamic generation for variable resolution (slow path)
            # We generate indices on the fly on the correct device
            device = self.relative_position_bias_table.device
            
            # Re-implement generation logic inline to ensure device correctness
            coords_h = torch.arange(h, device=device)
            coords_w = torch.arange(w, device=device)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
            coords_flatten = torch.flatten(coords, 1)
            
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            
            relative_coords[:, :, 0] += self.h - 1
            relative_coords[:, :, 1] += self.w - 1
            
            relative_coords[:, :, 0] = relative_coords[:, :, 0].clamp(0, 2 * self.h - 2)
            relative_coords[:, :, 1] = relative_coords[:, :, 1].clamp(0, 2 * self.w - 2)
            
            relative_coords[:, :, 0] *= 2 * self.w - 1
            index = relative_coords.sum(-1)

        # [H*W*H*W, nHead]
        relative_position_bias = self.relative_position_bias_table[index.view(-1)]
        # [H*W, H*W, nHead] -> [1, nHead, H*W, H*W]
        relative_position_bias = relative_position_bias.view(
            h * w, h * w, -1
        ).permute(2, 0, 1).contiguous().unsqueeze(0)
        return relative_position_bias

class MultiHeadSelfAttentionWithBias(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
        no_rope: int = 1,
        grid_size: int = 15, 
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.no_rope = no_rope

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        half_head_dim = embed_dim // num_heads // 2
        
        # RoPE setup
        img_seq_len = max_seq_len - no_rope
        rope_grid_size = int(img_seq_len ** 0.5)

        self.rotary = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=rope_grid_size,
            no_rope=no_rope,
        )
        
        # Add Relative Bias
        self.rel_pos_bias = RelativePositionBias(num_heads, grid_size, grid_size)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.rotary(q)
        k = self.rotary(k)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # --- Inject Relative Position Bias ---
        # Calculate current grid size dynamically from input sequence length
        n_task = self.no_rope
        num_img_tokens = seq_len - n_task
        current_grid_size = int(math.sqrt(num_img_tokens))
        
        # Pass current grid size to bias module
        bias = self.rel_pos_bias(current_grid_size, current_grid_size)
        
        attn_scores[:, :, n_task:, n_task:] = attn_scores[:, :, n_task:, n_task:] + bias

        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :].to(dtype=torch.bool)
            attn_scores = attn_scores.masked_fill(mask, torch.finfo(attn_scores.dtype).min)

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        context = self.proj(context)
        context = self.proj_dropout(context)
        return context

class ARCTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        max_seq_len: int,
        no_rope: int = 1,
        grid_size: int = 15,
    ) -> None:
        super().__init__()
        self.num_task_tokens = no_rope
        
        self.self_attn = MultiHeadSelfAttentionWithBias(
            embed_dim=embed_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            no_rope=no_rope,
            grid_size=grid_size
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = RMSNorm(embed_dim)
        
        self.mlp = ConvolutionalGLU(
            in_features=embed_dim,
            hidden_features=mlp_dim,
            act_layer=nn.GELU,
            drop=dropout,
            num_task_tokens=no_rope
        )
        
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(embed_dim)

    def forward(self, x, key_padding_mask=None):
        # Standard Pre-Norm
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, key_padding_mask=key_padding_mask)
        x = residual + self.dropout1(x)

        residual = x
        x = self.norm2(x)
        
        # Calculate grid size for ConvGLU
        seq_len = x.shape[1]
        num_img_tokens = seq_len - self.num_task_tokens
        grid_size = int(math.sqrt(num_img_tokens))
        
        x = self.mlp(x, H=grid_size, W=grid_size)
        x = residual + self.dropout2(x)
        return x

class ARCTransformerEncoder(nn.Module):
    def __init__(self, *, depth, embed_dim, num_heads, mlp_dim, dropout, max_seq_len, no_rope=0, grid_size=15):
        super().__init__()
        self.layers = nn.ModuleList([
            ARCTransformerEncoderLayer(
                embed_dim, num_heads, mlp_dim, dropout, max_seq_len, no_rope, grid_size
            ) for _ in range(depth)
        ])

    def forward(self, x, key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return x

class ARCViT2(nn.Module):
    """
    ARCViT2: Hybrid ViT + Relative Positional Bias (TransNeXt style).
    """
    def __init__(
        self,
        num_tasks: int,
        image_size: int = 30,
        num_colors: int = 10,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_dim: int = 512,
        dropout: float = 0.1,
        num_task_tokens: int = 1,
        patch_size: int = 2
    ) -> None:
        super().__init__()
        # ... (Basic init same as ViT1) ...
        if image_size <= 0: raise ValueError("image_size > 0")
        if num_colors <= 0: raise ValueError("num_colors > 0")
        if num_tasks <= 0: raise ValueError("num_tasks > 0")

        self.image_size = image_size
        self.num_colors = num_colors
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        
        if patch_size is None:
            self.seq_length = image_size * image_size
            grid_size = image_size
        else:
            self.seq_length = (image_size//patch_size)**2
            grid_size = image_size // patch_size
            
        print(f"ARCViT2 - Patch: {patch_size}, Grid: {grid_size}x{grid_size}")
        
        self.num_task_tokens = num_task_tokens
        self.color_embed = nn.Embedding(num_colors, embed_dim)
        self.task_token_embed = nn.Embedding(num_tasks, embed_dim * self.num_task_tokens)
        self.patch_embed = PatchEmbed(image_size, patch_size, embed_dim, embed_dim, bias=True)

        total_seq_len = self.num_task_tokens + self.seq_length
        self.positional_embed = nn.Parameter(torch.zeros(1, self.seq_length, embed_dim))
        
        self.encoder = ARCTransformerEncoder(
            depth=depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            max_seq_len=total_seq_len,
            no_rope=num_task_tokens,
            grid_size=grid_size # Pass grid size
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = RMSNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_colors * (1 if patch_size is None else patch_size)**2)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.positional_embed, std=0.02)
        nn.init.trunc_normal_(self.task_token_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.color_embed.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, pixel_values, task_ids, attention_mask=None):
        # ... (Same forward logic as ViT1) ...
        if pixel_values.dim() != 3: raise ValueError("pixel_values must be (B, H, W)")
        batch_size = pixel_values.size(0)
        device = pixel_values.device

        tokens = self.color_embed(pixel_values.long())
        tokens = self.patch_embed(tokens.permute((0, 3, 1, 2)))
        tokens = tokens + self.positional_embed[:, : tokens.size(1), :]

        task_tokens = self.task_token_embed(task_ids.long())
        task_tokens = task_tokens.reshape(batch_size, self.num_task_tokens, -1)
        hidden_states = torch.cat([task_tokens, tokens], dim=1)
        hidden_states = self.dropout(hidden_states)

        key_padding_mask = None
        if attention_mask is not None:
            if self.patch_size is not None:
                attention_mask = attention_mask.reshape(batch_size, self.image_size//self.patch_size, self.patch_size, self.image_size//self.patch_size, self.patch_size)
                attention_mask = torch.max(torch.max(attention_mask, dim=2)[0], dim=3)[0]
            flat_mask = attention_mask.view(batch_size, self.seq_length)
            pad_mask = ~flat_mask.bool()
            pad_mask = torch.cat(
                [torch.zeros(batch_size, self.num_task_tokens, device=device, dtype=torch.bool), pad_mask],
                dim=1,
            )
            key_padding_mask = pad_mask

        encoded = self.encoder(hidden_states, key_padding_mask=key_padding_mask)
        encoded = self.norm(encoded)
        pixel_states = encoded[:, self.num_task_tokens:, :]

        logits = self.head(pixel_states)
        logits = logits.reshape((-1, self.image_size//self.patch_size, self.image_size//self.patch_size, self.patch_size, self.patch_size, self.num_colors))
        logits = logits.permute((0, 1, 3, 2, 4, 5))
        logits = logits.reshape(batch_size, self.image_size, self.image_size, self.num_colors)
        logits = logits.permute(0, 3, 1, 2)
        return logits