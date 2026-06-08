from __future__ import annotations

import math
from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 16, alpha: int = 32, dropout: float = 0.05):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x)
        lora = F.linear(F.linear(self.dropout(x), self.lora_a), self.lora_b) * self.scaling
        return base + lora


def replace_linear_with_lora(module: nn.Module, target_name_fragments: Iterable[str], rank: int, alpha: int, dropout: float) -> List[str]:
    """Replace selected nn.Linear submodules with LoRA wrappers.

    This small utility is used by local adapters. For Hugging Face/PEFT runs,
    prefer peft.LoraConfig because it handles tied weights and saving format.
    """
    replaced: List[str] = []
    targets = tuple(target_name_fragments)
    for name, child in list(module.named_children()):
        full_match = any(t in name for t in targets)
        if isinstance(child, nn.Linear) and full_match:
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
            replaced.append(name)
        else:
            for sub in replace_linear_with_lora(child, targets, rank, alpha, dropout):
                replaced.append(f"{name}.{sub}")
    return replaced
