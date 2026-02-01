from typing import Optional, Tuple
import math

from utils.pos_embed import VisionRotaryEmbeddingFast
import torch
from torch import nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    More stable than LayerNorm for deep/recurrent architectures (URM style).
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (Batch, Seq, Dim)
        norm_x = torch.mean(x ** 2, dim=-1, keepdim=True)
        return self.scale * x * torch.rsqrt(norm_x + self.eps)


class ConvolutionalGLU(nn.Module):
    """
    Hybrid FFN: Gated Linear Unit with Depthwise Convolution (TransNeXt style).
    Handles hybrid sequences (Task Tokens + Image Tokens).
    """
    def __init__(
        self, 
        in_features, 
        hidden_features=None, 
        out_features=None, 
        act_layer=nn.GELU, 
        drop=0., 
        num_task_tokens=1
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        # TransNeXt heuristic: Reduce hidden dim by 2/3 to keep params similar to standard MLP
        # because GLU has 2 projections in the first layer.
        hidden_features = int(2 * hidden_features / 3)
        
        self.num_task_tokens = num_task_tokens
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        
        # Depthwise Convolution: groups = in_channels
        self.dwconv = nn.Conv2d(
            hidden_features, 
            hidden_features, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            groups=hidden_features
        )
        
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        # x: [B, N_total, C_in]
        # H, W: Spatial dimensions of the image part
        
        # 1. Global Linear Projection (Task + Image)
        # Output: [B, N_total, 2 * C_hidden] -> Split into Gate and Value
        x_gate, x_val = self.fc1(x).chunk(2, dim=-1) 
        
        # 2. Split Task and Image tokens
        # We only apply Conv to the Image part to preserve spatial locality.
        # Task tokens are abstract and shouldn't be convolved spatially.
        task_gate = x_gate[:, :self.num_task_tokens, :]
        img_gate = x_gate[:, self.num_task_tokens:, :]
        
        # 3. Apply Depthwise Conv to Image part
        B, N_img, C = img_gate.shape
        
        # Check shape consistency
        if N_img != H * W:
             # Fallback or error if shapes don't match (e.g. during inference with different sizes if not careful)
             # For ARC, we assume consistent H*W per batch usually.
             pass

        # [B, N_img, C] -> [B, C, H, W]
        img_gate = img_gate.transpose(1, 2).reshape(B, C, H, W)
        img_gate = self.dwconv(img_gate)
        # [B, C, H, W] -> [B, N_img, C]
        img_gate = img_gate.flatten(2).transpose(1, 2)
        
        # 4. Recombine (Task tokens just pass through, effectively Identity conv)
        x_gate = torch.cat([task_gate, img_gate], dim=1)
        
        # 5. Gating & Output Projection
        x = self.act(x_gate) * x_val
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
        no_rope: int = 1,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        if self.head_dim % 2 != 0:
            raise ValueError("Rotary embeddings require the head dimension to be even")

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        half_head_dim = embed_dim // num_heads // 2
        
        # [FIX] 回归原始逻辑，但更严谨：先减去 task tokens 再开方
        # 这样能保证 grid_size 严格等于 image_size // patch_size
        img_seq_len = max_seq_len - no_rope
        grid_size = int(img_seq_len ** 0.5)
        
        self.rotary = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=grid_size, # 这里是 32
            no_rope=no_rope,
        )

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

        # Use scaled_dot_product_attention for memory efficiency
        attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: True means ignore (padding)
            # SDPA boolean mask: True means attend (keep)
            attn_mask = ~key_padding_mask[:, None, None, :].to(dtype=torch.bool)

        context = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=False
        )

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
    ) -> None:
        super().__init__()
        self.num_task_tokens = no_rope
        
        self.self_attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            no_rope=no_rope,
        )
        self.dropout1 = nn.Dropout(dropout)
        # Upgrade: LayerNorm -> RMSNorm
        self.norm1 = RMSNorm(embed_dim)
        
        # Upgrade: Standard MLP -> ConvolutionalGLU
        self.mlp = ConvolutionalGLU(
            in_features=embed_dim,
            hidden_features=mlp_dim,
            act_layer=nn.GELU,
            drop=dropout,
            num_task_tokens=no_rope
        )
        
        self.dropout2 = nn.Dropout(dropout) # Path dropout
        self.norm2 = RMSNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-Norm Architecture
        
        # 1. Attention Block
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, key_padding_mask=key_padding_mask)
        x = residual + self.dropout1(x)

        # 2. Hybrid MLP Block
        residual = x
        x = self.norm2(x)
        
        # Calculate spatial dimensions for ConvGLU
        # x shape: [Batch, SeqLen, Dim]
        # SeqLen = NumTaskTokens + H*W
        seq_len = x.shape[1]
        num_img_tokens = seq_len - self.num_task_tokens
        
        # Assume square grid for ARC
        grid_size = int(math.sqrt(num_img_tokens))
        
        x = self.mlp(x, H=grid_size, W=grid_size)
        x = residual + self.dropout2(x)
        
        return x


class ARCTransformerEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth: int,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        max_seq_len: int,
        no_rope: int = 0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ARCTransformerEncoderLayer(
                    embed_dim,
                    num_heads,
                    mlp_dim,
                    dropout,
                    max_seq_len=max_seq_len,
                    no_rope=no_rope,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return x


class ARCViT1(nn.Module):
    """
    Hybrid Vision Transformer tailored for ARC tasks (Version 1).
    Features:
    - RMSNorm instead of LayerNorm
    - ConvolutionalGLU (Hybrid FFN) instead of standard MLP
    - Task-aware processing
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

        if image_size <= 0:
            raise ValueError(" must be > 0.")
        if num_colors <= 0:
            raise ValueError(" must be > 0.")
        if num_tasks <= 0:
            raise ValueError(" must be > 0.")

        self.image_size = image_size
        self.num_colors = num_colors
        self.embed_dim = embed_dim
        if patch_size is None:
            self.seq_length = image_size * image_size
        else:
            self.seq_length = (image_size//patch_size)**2
        self.patch_size = patch_size
        print(f"ARCViT1 - Patch size: {self.patch_size}, sequence length: {self.seq_length}")
        self.num_task_tokens = num_task_tokens
        self.color_embed = nn.Embedding(num_colors, embed_dim)
        self.task_token_embed = nn.Embedding(num_tasks, embed_dim * self.num_task_tokens)
        self.patch_embed = PatchEmbed(image_size, patch_size, embed_dim, embed_dim, bias=True)

        total_seq_len = self.num_task_tokens + self.seq_length
        self.positional_embed = nn.Parameter(torch.zeros(1, self.seq_length, embed_dim))
        
        # Use the new Encoder
        self.encoder = ARCTransformerEncoder(
            depth=depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            max_seq_len=total_seq_len,
            no_rope=num_task_tokens,
            )

        self.dropout = nn.Dropout(dropout)
        # Upgrade: Final Norm is also RMSNorm
        self.norm = RMSNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_colors * (1 if patch_size is None else patch_size)**2)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.positional_embed, std=0.02)
        nn.init.trunc_normal_(self.task_token_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.color_embed.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        pixel_values: torch.Tensor,
        task_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:

        if pixel_values.dim() != 3:
            raise ValueError(" must be (batch, height, width).")
        if pixel_values.size(1) != self.image_size or pixel_values.size(2) != self.image_size:
            raise ValueError(
                " height/width must match configured image_size="
                f"{self.image_size}. Received {pixel_values.shape[1:]}"
            )

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
            if attention_mask.shape != (batch_size, self.image_size, self.image_size):
                raise ValueError(
                    " must match pixel grid size."
                )
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