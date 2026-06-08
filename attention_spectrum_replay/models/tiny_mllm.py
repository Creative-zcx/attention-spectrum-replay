from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MLLMOutput:
    logits: torch.Tensor
    cross_attentions: torch.Tensor  # [B,L,H,T,Gh,Gw]
    image_embeds: torch.Tensor
    text_embeds: torch.Tensor


class CrossAttentionBlock(nn.Module):
    """Small explicit multi-head cross-attention block.

    This avoids heavyweight attention kernels so local CPU runs stay quick.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.ln_q = nn.LayerNorm(hidden_dim)
        self.ln_out = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.ln_ff = nn.LayerNorm(hidden_dim)

    def forward(self, text: torch.Tensor, visual: torch.Tensor):
        bsz, tq, dim = text.shape
        pv = visual.size(1)
        q_in = self.ln_q(text)
        q = self.q_proj(q_in).view(bsz, tq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(visual).view(bsz, pv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(visual).view(bsz, pv, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(self.dropout(attn), v).transpose(1, 2).contiguous().view(bsz, tq, dim)
        text = self.ln_out(text + self.o_proj(ctx))
        text = self.ln_ff(text + self.ff(text))
        return text, attn


class TinyMLLM(nn.Module):
    """A compact MLLM-like model with explicit cross-attention maps.

    It is intended for deterministic CPU development. The
    interface mirrors the fields used by ASRTrainer.
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        hidden_dim: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        image_size: int = 32,
        patch_size: int = 8,
        max_seq_len: int = 32,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.vocab_size = int(vocab_size)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.grid_size = image_size // patch_size
        self.token_emb = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        self.patch_proj = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.visual_pos = nn.Parameter(torch.zeros(1, self.grid_size * self.grid_size, hidden_dim))
        nn.init.normal_(self.visual_pos, std=0.02)
        self.blocks = nn.ModuleList([CrossAttentionBlock(hidden_dim, num_heads) for _ in range(num_layers)])
        self.text_pool = nn.LayerNorm(hidden_dim)
        self.image_pool = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, num_classes))

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        functional_token_mask: Optional[torch.Tensor] = None,
        **_: object,
    ) -> MLLMOutput:
        bsz = images.shape[0]
        visual_grid = self.patch_proj(images)  # [B,D,Gh,Gw]
        gh, gw = visual_grid.shape[-2:]
        visual = visual_grid.flatten(2).transpose(1, 2) + self.visual_pos[:, : gh * gw]
        seq = input_ids.size(1)
        pos = torch.arange(seq, device=input_ids.device).unsqueeze(0).expand(bsz, seq)
        text = self.token_emb(input_ids) + self.pos_emb(pos.clamp_max(self.pos_emb.num_embeddings - 1))
        if attention_mask is not None:
            text = text * attention_mask.unsqueeze(-1).to(text.dtype)
        attn_layers: List[torch.Tensor] = []
        for block in self.blocks:
            text, attn = block(text, visual)
            # attn [B, heads, T, P]
            attn_layers.append(attn.reshape(bsz, self.num_heads, seq, gh, gw))
        if attention_mask is None:
            mask = (input_ids != 0).float()
        else:
            mask = attention_mask.float()
        pooled_text = (text * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled_text = self.text_pool(pooled_text)
        pooled_image = self.image_pool(visual.mean(dim=1))
        logits = self.classifier(pooled_text)
        cross_attentions = torch.stack(attn_layers, dim=1)
        return MLLMOutput(
            logits=logits,
            cross_attentions=cross_attentions,
            image_embeds=pooled_image,
            text_embeds=pooled_text,
        )
