# src/attn_hook_loop.py
from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Any

import torch
from torch import nn
from timm.models.vision_transformer import PatchEmbed

from utils.pos_embed import VisionRotaryEmbeddingFast
from src.ARC_LoopViT import LoopARCViT  # 直接复用已有的 Loop 模型结构


class MultiHeadSelfAttentionWithAttn(nn.Module):
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
        self.rotary = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=int(max_seq_len ** 0.5),
            no_rope=no_rope,
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.rotary(q)
        k = self.rotary(k)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :].to(dtype=torch.bool)
            attn_scores = attn_scores.masked_fill(
                mask,
                torch.finfo(attn_scores.dtype).min,
            )

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        context = self.proj(context)
        context = self.proj_dropout(context)

        if return_attn:
            return context, attn_scores, attn_weights
        return context


class ARCTransformerEncoderLayerWithAttn(nn.Module):
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
        self.self_attn = MultiHeadSelfAttentionWithAttn(
            embed_dim=embed_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            no_rope=no_rope,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, mlp_dim)
        self.activation = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = x
        if return_attn:
            x_sa, scores, attn = self.self_attn(
                x, key_padding_mask=key_padding_mask, return_attn=True
            )
        else:
            x_sa = self.self_attn(x, key_padding_mask=key_padding_mask)
            scores = None
            attn = None

        x = residual + self.dropout1(x_sa)
        x = self.norm1(x)

        residual = x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        x = residual + self.dropout3(x)
        x = self.norm2(x)

        if return_attn:
            return x, scores, attn
        return x


class LoopARCViTWithAttn(LoopARCViT):
    """
    在 LoopARCViT 基础上扩展：在 forward 时可返回所有 loop step、每层的 attention。
    注意：我们复用 LoopARCViT 的 _encode_inputs / _prepare_attention_mask / head 逻辑，
    只重写 forward 主体，在 loop 里用带 attn 的 ARCTransformerEncoderLayerWithAttn。
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # 替换 core_layers 为带 attn 的版本
        total_seq_len = self.num_task_tokens + self.seq_length
        new_core_layers = nn.ModuleList(
            [
                ARCTransformerEncoderLayerWithAttn(
                    embed_dim=self.embed_dim,
                    num_heads=kwargs.get("num_heads", 8),
                    mlp_dim=kwargs.get("mlp_dim", 512),
                    dropout=kwargs.get("dropout", 0.1),
                    max_seq_len=total_seq_len,
                    no_rope=self.num_task_tokens,
                )
                for _ in range(self.loop_core_depth)
            ]
        )
        # 由于 LoopARCViT.__init__ 里已经建过 core_layers，这里直接覆盖
        self.core_layers = new_core_layers

    def forward(
        self,
        pixel_values: torch.Tensor,
        task_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        dynamic_exit: Optional[bool] = None,
        gate_threshold: Optional[float] = None,
        return_attn: bool = False,
    ):
        hidden_states, batch_size = self._encode_inputs(pixel_values, task_ids)
        device = hidden_states.device
        key_padding_mask = self._prepare_attention_mask(attention_mask, batch_size, device)

        use_dynamic_exit = bool(dynamic_exit) and self.use_exit_gate
        threshold = gate_threshold if gate_threshold is not None else self.default_gate_threshold
        running_hidden = hidden_states
        if use_dynamic_exit:
            finished_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
            cached_final = torch.zeros_like(running_hidden)
            exit_steps = torch.full((batch_size,), self.max_loop_steps, dtype=torch.long, device=device)
        else:
            cached_final = None
            exit_steps = torch.full((batch_size,), self.max_loop_steps, dtype=torch.long, device=device)

        gate_probs: List[torch.Tensor] = []
        all_scores: List[torch.Tensor] = []
        all_attn: List[torch.Tensor] = []

        for step in range(self.max_loop_steps):
            if self.step_embed is not None:
                running_hidden = running_hidden + self.step_embed.weight[step].view(1, 1, -1)

            for layer in self.core_layers:
                if return_attn:
                    running_hidden, scores, attn = layer(
                        running_hidden, key_padding_mask=key_padding_mask, return_attn=True
                    )
                    all_scores.append(scores)  # (B,H,N,N)
                    all_attn.append(attn)
                else:
                    running_hidden = layer(running_hidden, key_padding_mask=key_padding_mask)

            if self.use_exit_gate:
                gate_logit = self.exit_gate(running_hidden[:, 0, :]).squeeze(-1)
                gate_prob = torch.sigmoid(gate_logit)
                gate_probs.append(gate_prob)
            else:
                gate_prob = None

            if use_dynamic_exit and gate_prob is not None:
                eligible = (step + 1) >= self.min_loop_steps
                exit_now = (gate_prob >= threshold) & eligible & (~finished_mask)
                if exit_now.any():
                    cached_final[exit_now] = running_hidden[exit_now]
                    exit_steps[exit_now] = step + 1
                    finished_mask = finished_mask | exit_now
                running_hidden = torch.where(
                    finished_mask.view(batch_size, 1, 1),
                    cached_final,
                    running_hidden,
                )
                if finished_mask.all():
                    break

        final_states = running_hidden if cached_final is None else torch.where(
            finished_mask.view(batch_size, 1, 1),
            cached_final,
            running_hidden,
        )
        final_states = self.norm(final_states)
        pixel_states = final_states[:, self.num_task_tokens :, :]
        logits = self.head(pixel_states)
        logits = logits.reshape(
            (
                -1,
                self.image_size // (self.patch_size or 1),
                self.image_size // (self.patch_size or 1),
                (self.patch_size or 1),
                (self.patch_size or 1),
                self.num_colors,
            )
        )
        logits = logits.permute((0, 1, 3, 2, 4, 5))
        logits = logits.reshape(pixel_values.size(0), self.image_size, self.image_size, self.num_colors)
        logits = logits.permute(0, 3, 1, 2)

        metadata = {
            "gate_probs": gate_probs,
            "exit_steps": exit_steps,
            "max_steps": self.max_loop_steps,
        }

        if return_attn:
            return logits, (all_scores, all_attn, metadata)
        return logits, (None, None, metadata)


def load_loop_vit_with_attn(
    ckpt_path: str,
    num_tasks: int,
    image_size: int = 64,
    num_colors: int = 12,
    embed_dim: int = 512,
    loop_core_depth: int = 2,
    max_loop_steps: int = 6,
    min_loop_steps: int = 1,
    num_heads: int = 8,
    mlp_dim: int = 512,
    dropout: float = 0.1,
    num_task_tokens: int = 1,
    patch_size: int = 2,
    device: str = "cuda",
) -> LoopARCViTWithAttn:
    """加载 LoopARCViT 的 checkpoint，并映射到 LoopARCViTWithAttn 上。"""
    ckpt = torch.load(ckpt_path, map_location=device)
    state: Dict[str, Any] = ckpt["model_state"]

    # 1) 去掉前缀 "_orig_mod."
    new_state: Dict[str, Any] = {}
    for k, v in state.items():
        if k.startswith("_orig_mod."):
            nk = k[len("_orig_mod.") :]
        else:
            nk = k
        new_state[nk] = v

    # 2) 构造新模型
    model = LoopARCViTWithAttn(
        num_tasks=num_tasks,
        image_size=image_size,
        num_colors=num_colors,
        embed_dim=embed_dim,
        loop_core_depth=loop_core_depth,
        max_loop_steps=max_loop_steps,
        min_loop_steps=min_loop_steps,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        dropout=dropout,
        num_task_tokens=num_task_tokens,
        patch_size=patch_size,
        use_exit_gate=True,
        gate_threshold=0.6,
        add_step_embeddings=True,
    ).to(device)

    # 3) 加载权重
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)

    model.eval()
    return model