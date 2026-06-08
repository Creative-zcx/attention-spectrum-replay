from __future__ import annotations

from copy import deepcopy
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def rowwise_symmetric_kl_from_similarity(sim: torch.Tensor, ref_sim: torch.Tensor) -> torch.Tensor:
    p = F.softmax(sim, dim=-1).clamp_min(1e-8)
    q = F.softmax(ref_sim, dim=-1).clamp_min(1e-8)
    return 0.5 * ((p * (p.log() - q.log())).sum(dim=-1) + (q * (q.log() - p.log())).sum(dim=-1)).mean()


def geometry_regularizer(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    ref_image_embeds: torch.Tensor,
    ref_text_embeds: torch.Tensor,
) -> torch.Tensor:
    image_embeds = F.normalize(image_embeds.float(), dim=-1)
    text_embeds = F.normalize(text_embeds.float(), dim=-1)
    ref_image_embeds = F.normalize(ref_image_embeds.float(), dim=-1)
    ref_text_embeds = F.normalize(ref_text_embeds.float(), dim=-1)
    sim = image_embeds @ text_embeds.t()
    ref_sim = ref_image_embeds @ ref_text_embeds.t()
    return rowwise_symmetric_kl_from_similarity(sim, ref_sim)


class EWC:
    def __init__(self, lambda_ewc: float = 1.0e4):
        self.lambda_ewc = lambda_ewc
        self.prev_params: Dict[str, torch.Tensor] = {}
        self.fisher: Dict[str, torch.Tensor] = {}

    def estimate(self, model: nn.Module, dataloader, loss_fn, device: torch.device) -> None:
        model.eval()
        fisher = {name: torch.zeros_like(p, device=device) for name, p in model.named_parameters() if p.requires_grad}
        n_batches = 0
        for batch in dataloader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            model.zero_grad(set_to_none=True)
            out = model(**model_batch_kwargs(batch))
            loss = loss_fn(out.logits, batch["labels"])
            loss.backward()
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[name] += p.grad.detach().square()
            n_batches += 1
        for name in fisher:
            fisher[name] /= max(1, n_batches)
        self.fisher = {k: v.detach().cpu() for k, v in fisher.items()}
        self.prev_params = {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}
        model.train()

    def loss(self, model: nn.Module) -> torch.Tensor:
        if not self.prev_params:
            return next(model.parameters()).new_zeros(())
        total = None
        for name, p in model.named_parameters():
            if name in self.prev_params:
                prev = self.prev_params[name].to(p.device)
                fish = self.fisher[name].to(p.device)
                val = (fish * (p - prev).square()).sum()
                total = val if total is None else total + val
        if total is None:
            return next(model.parameters()).new_zeros(())
        return self.lambda_ewc * total


class LwF:
    def __init__(self, temperature: float = 2.0, weight: float = 1.0):
        self.temperature = temperature
        self.weight = weight
        self.teacher: Optional[nn.Module] = None

    def set_teacher(self, model: nn.Module) -> None:
        self.teacher = deepcopy(model).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    def loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        t = self.temperature
        return self.weight * (t * t) * F.kl_div(
            F.log_softmax(student_logits / t, dim=-1),
            F.softmax(teacher_logits / t, dim=-1),
            reduction="batchmean",
        )


def model_batch_kwargs(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    keys = ["images", "input_ids", "attention_mask", "functional_token_mask"]
    return {k: batch[k] for k in keys if k in batch}
