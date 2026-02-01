from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from timm.models.vision_transformer import PatchEmbed

from src.ARC_ViT import ARCTransformerEncoderLayer


@dataclass
class LoopForwardMetadata:
    """Metadata describing loop behaviour for :class:`LoopARCViTVariable`."""

    gate_probs: List[torch.Tensor]
    exit_steps: torch.Tensor
    max_steps: int


class LoopARCViTVariable(nn.Module):
    """Vision Transformer variant that supports per-sample loop depth control.

    This class mirrors :class:`src.ARC_LoopViT.LoopARCViT` but extends the
    forward pass with an optional ``forced_steps`` argument. When provided, the
    loop will execute until either the gate triggers (if dynamic exit is
    enabled) or the forced step budget for each sample is exhausted, whichever
    happens first.
    """

    def __init__(
        self,
        num_tasks: int,
        image_size: int = 30,
        num_colors: int = 10,
        embed_dim: int = 256,
        loop_core_depth: int = 2,
        max_loop_steps: int = 8,
        min_loop_steps: int = 1,
        num_heads: int = 8,
        mlp_dim: int = 512,
        dropout: float = 0.1,
        num_task_tokens: int = 1,
        patch_size: Optional[int] = 2,
        use_exit_gate: bool = True,
        gate_threshold: float = 0.6,
        add_step_embeddings: bool = True,
    ) -> None:
        super().__init__()

        if image_size <= 0:
            raise ValueError("`image_size` must be > 0.")
        if num_colors <= 0:
            raise ValueError("`num_colors` must be > 0.")
        if num_tasks <= 0:
            raise ValueError("`num_tasks` must be > 0.")
        if loop_core_depth <= 0:
            raise ValueError("`loop_core_depth` must be > 0.")
        if max_loop_steps <= 0:
            raise ValueError("`max_loop_steps` must be > 0.")
        if min_loop_steps <= 0 or min_loop_steps > max_loop_steps:
            raise ValueError("`min_loop_steps` must be in [1, max_loop_steps].")

        self.image_size = image_size
        self.num_colors = num_colors
        self.embed_dim = embed_dim
        self.loop_core_depth = loop_core_depth
        self.max_loop_steps = max_loop_steps
        self.min_loop_steps = min_loop_steps
        self.use_exit_gate = use_exit_gate
        self.default_gate_threshold = gate_threshold
        self.add_step_embeddings = add_step_embeddings

        self.patch_size = patch_size
        if patch_size is None:
            self.seq_length = image_size * image_size
        else:
            self.seq_length = (image_size // patch_size) ** 2

        self.num_task_tokens = num_task_tokens
        self.color_embed = nn.Embedding(num_colors, embed_dim)
        self.task_token_embed = nn.Embedding(num_tasks, embed_dim * num_task_tokens)
        self.patch_embed = PatchEmbed(image_size, patch_size, embed_dim, embed_dim, bias=True)
        self.positional_embed = nn.Parameter(torch.zeros(1, self.seq_length, embed_dim))

        total_seq_len = self.num_task_tokens + self.seq_length
        self.core_layers = nn.ModuleList(
            [
                ARCTransformerEncoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    max_seq_len=total_seq_len,
                    no_rope=num_task_tokens,
                )
                for _ in range(loop_core_depth)
            ]
        )

        if add_step_embeddings:
            self.step_embed = nn.Embedding(max_loop_steps, embed_dim)
        else:
            self.step_embed = None

        if use_exit_gate:
            self.exit_gate = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Linear(embed_dim // 2, 1),
            )
        else:
            self.exit_gate = None

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        output_dim = num_colors * (1 if patch_size is None else patch_size) ** 2
        self.head = nn.Linear(embed_dim, output_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.positional_embed, std=0.02)
        nn.init.trunc_normal_(self.task_token_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.color_embed.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        if self.step_embed is not None:
            nn.init.trunc_normal_(self.step_embed.weight, std=0.02)
        if self.exit_gate is not None:
            for module in self.exit_gate:
                if isinstance(module, nn.Linear):
                    nn.init.trunc_normal_(module.weight, std=0.02)
                    nn.init.zeros_(module.bias)

    def _encode_inputs(
        self,
        pixel_values: torch.Tensor,
        task_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if pixel_values.dim() != 3:
            raise ValueError("`pixel_values` must be (batch, height, width).")
        if (
            pixel_values.size(1) != self.image_size
            or pixel_values.size(2) != self.image_size
        ):
            raise ValueError(
                "`pixel_values` height/width must match configured image_size="
                f"{self.image_size}. Received {pixel_values.shape[1:]}"
            )

        batch_size = pixel_values.size(0)
        tokens = self.color_embed(pixel_values.long())
        tokens = self.patch_embed(tokens.permute((0, 3, 1, 2)))
        tokens = tokens + self.positional_embed[:, : tokens.size(1), :]

        task_tokens = self.task_token_embed(task_ids.long())
        task_tokens = task_tokens.reshape(batch_size, self.num_task_tokens, -1)
        hidden_states = torch.cat([task_tokens, tokens], dim=1)
        hidden_states = self.dropout(hidden_states)
        return hidden_states, batch_size

    def _prepare_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        if attention_mask.shape != (batch_size, self.image_size, self.image_size):
            raise ValueError("`attention_mask` must match pixel grid size.")
        if self.patch_size is not None:
            mask = attention_mask.reshape(
                batch_size,
                self.image_size // self.patch_size,
                self.patch_size,
                self.image_size // self.patch_size,
                self.patch_size,
            )
            mask = torch.max(torch.max(mask, dim=2)[0], dim=3)[0]
        else:
            mask = attention_mask
        flat_mask = mask.view(batch_size, self.seq_length)
        pad_mask = ~flat_mask.bool()
        pad_mask = torch.cat(
            [torch.zeros(batch_size, self.num_task_tokens, device=device, dtype=torch.bool), pad_mask],
            dim=1,
        )
        return pad_mask

    def forward(
        self,
        pixel_values: torch.Tensor,
        task_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        dynamic_exit: Optional[bool] = None,
        gate_threshold: Optional[float] = None,
        forced_steps: Optional[Union[int, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, LoopForwardMetadata]:
        hidden_states, batch_size = self._encode_inputs(pixel_values, task_ids)
        device = hidden_states.device
        key_padding_mask = self._prepare_attention_mask(attention_mask, batch_size, device)

        use_dynamic_exit = bool(dynamic_exit) and self.use_exit_gate
        threshold = gate_threshold if gate_threshold is not None else self.default_gate_threshold

        forced_steps_tensor: Optional[torch.Tensor]
        if forced_steps is None:
            forced_steps_tensor = None
        elif isinstance(forced_steps, int):
            forced_steps_tensor = torch.full((batch_size,), forced_steps, dtype=torch.long, device=device)
        elif isinstance(forced_steps, torch.Tensor):
            forced_steps_tensor = forced_steps.to(device=device, dtype=torch.long)
        else:
            raise TypeError("`forced_steps` must be an int or Tensor.")

        if forced_steps_tensor is not None:
            forced_steps_tensor = torch.clamp(
                forced_steps_tensor,
                min=max(1, self.min_loop_steps),
                max=self.max_loop_steps,
            )

        use_forced_exit = forced_steps_tensor is not None
        needs_exit_tracking = use_dynamic_exit or use_forced_exit

        running_hidden = hidden_states
        if needs_exit_tracking:
            finished_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
            cached_final = torch.zeros_like(running_hidden)
            exit_steps = torch.full((batch_size,), self.max_loop_steps, dtype=torch.long, device=device)
        else:
            cached_final = None
            exit_steps = torch.full((batch_size,), self.max_loop_steps, dtype=torch.long, device=device)

        gate_probs: List[torch.Tensor] = []

        for step in range(self.max_loop_steps):
            if self.step_embed is not None:
                running_hidden = running_hidden + self.step_embed.weight[step].view(1, 1, -1)

            for layer in self.core_layers:
                running_hidden = layer(running_hidden, key_padding_mask=key_padding_mask)

            gate_prob: Optional[torch.Tensor] = None
            if self.use_exit_gate:
                gate_logit = self.exit_gate(running_hidden[:, 0, :]).squeeze(-1)
                gate_prob = torch.sigmoid(gate_logit)
                gate_probs.append(gate_prob)

            exit_now: Optional[torch.Tensor] = None
            if use_dynamic_exit and gate_prob is not None:
                eligible = (step + 1) >= self.min_loop_steps
                exit_now = (gate_prob >= threshold) & eligible
            if use_forced_exit:
                forced_exit = (step + 1) >= forced_steps_tensor
                exit_now = forced_exit if exit_now is None else (exit_now | forced_exit)

            if needs_exit_tracking and exit_now is not None:
                exit_now = exit_now & (~finished_mask)
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

        final_states = running_hidden if not needs_exit_tracking else torch.where(
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

        metadata = LoopForwardMetadata(
            gate_probs=gate_probs,
            exit_steps=exit_steps,
            max_steps=self.max_loop_steps,
        )
        return logits, metadata
