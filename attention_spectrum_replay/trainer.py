from __future__ import annotations

import copy
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .config import ExperimentConfig
from .data import make_dataloader
from .losses import geometry_regularizer, model_batch_kwargs
from .memory import PrototypeMemory
from .metrics import accuracy_from_logits, compute_ap_af, compute_last_avg
from .spectral import SpectralEncoder
from .utils import ensure_dir, save_json, to_device


class ASRTrainer:
    def __init__(
        self,
        cfg: ExperimentConfig,
        model: nn.Module,
        spectral_encoder: SpectralEncoder,
        memory: PrototypeMemory,
        train_sets: Sequence[torch.utils.data.Dataset],
        val_sets: Optional[Sequence[torch.utils.data.Dataset]] = None,
    ):
        self.cfg = cfg
        self.model = model
        self.spectral_encoder = spectral_encoder
        self.memory = memory
        self.train_sets = list(train_sets)
        self.val_sets = list(val_sets) if val_sets is not None else None
        self.device = torch.device(cfg.runtime.device)
        self.model.to(self.device)
        self.ref_model = copy.deepcopy(model).to(self.device).eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.out_dir = ensure_dir(cfg.runtime.output_dir)
        self.global_step = 0
        self.history: Dict[str, object] = {"stages": [], "score_matrix": None}

    def fit(self) -> Dict[str, object]:
        n_stages = len(self.train_sets)
        score_matrix = np.full((n_stages, n_stages), np.nan, dtype=float)
        for stage_idx, dataset in enumerate(self.train_sets):
            stage_info = self.train_stage(stage_idx, dataset)
            self.update_memory(stage_idx, dataset)
            if self.val_sets is not None:
                for eval_idx in range(stage_idx + 1):
                    score = self.evaluate(self.val_sets[eval_idx])
                    score_matrix[stage_idx, eval_idx] = score["accuracy"]
            stage_info["eval_seen"] = score_matrix[stage_idx, : stage_idx + 1].tolist() if self.val_sets is not None else []
            self.history["stages"].append(stage_info)
            self.history["score_matrix"] = score_matrix.tolist()
            self._save_stage(stage_idx)
        metrics = {}
        if self.val_sets is not None:
            metrics.update(compute_ap_af(np.nan_to_num(score_matrix, nan=0.0)))
            metrics.update(compute_last_avg(np.nan_to_num(score_matrix, nan=0.0)))
        self.history["final_metrics"] = metrics
        save_json(self.history, self.out_dir / "metrics.json")
        self.memory.save(self.out_dir / "memory.pt")
        torch.save({"model": self.model.state_dict(), "config": self.cfg.to_dict()}, self.out_dir / "checkpoint.pt")
        return self.history

    def train_stage(self, stage_idx: int, dataset: torch.utils.data.Dataset) -> Dict[str, object]:
        self.model.train()
        loader = make_dataloader(
            dataset,
            batch_size=self.cfg.optim.batch_size,
            shuffle=True,
            num_workers=self.cfg.runtime.num_workers,
        )
        opt = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.cfg.optim.lr,
            betas=tuple(self.cfg.optim.betas),
            weight_decay=self.cfg.optim.weight_decay,
        )
        total_steps = max(1, len(loader) * self.cfg.optim.epochs_per_stage)
        scheduler = CosineAnnealingLR(opt, T_max=total_steps) if self.cfg.optim.cosine_decay else None
        losses: List[Dict[str, float]] = []
        beta = 0.0 if stage_idx == 0 or self.cfg.method == "vanilla" else self.cfg.asr.beta
        for epoch in range(self.cfg.optim.epochs_per_stage):
            pbar = tqdm(loader, desc=f"stage {stage_idx+1}/{len(self.train_sets)} epoch {epoch+1}", leave=False)
            for batch in pbar:
                batch = to_device(batch, self.device)
                opt.zero_grad(set_to_none=True)
                out = self.model(**model_batch_kwargs(batch))
                task_loss = F.cross_entropy(out.logits, batch["labels"])
                spec_loss = out.logits.new_zeros(())
                if self.cfg.method == "asr" and beta > 0 and self.memory.is_initialized():
                    spec = self.spectral_encoder(out.cross_attentions, batch.get("functional_token_mask"))
                    spec_loss = self.memory.spectral_loss(spec.descriptor, spec.angular, batch["skill_probs"])
                geo_loss = out.logits.new_zeros(())
                if self.cfg.asr.gamma > 0:
                    with torch.no_grad():
                        ref_out = self.ref_model(**model_batch_kwargs(batch))
                    geo_loss = geometry_regularizer(out.image_embeds, out.text_embeds, ref_out.image_embeds, ref_out.text_embeds)
                loss = task_loss + beta * spec_loss + self.cfg.asr.gamma * geo_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at stage {stage_idx}, step {self.global_step}: {loss.item()}")
                loss.backward()
                if self.cfg.optim.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.max_grad_norm)
                opt.step()
                if scheduler is not None:
                    scheduler.step()
                self.global_step += 1
                row = {
                    "task": float(task_loss.detach().cpu()),
                    "spec": float(spec_loss.detach().cpu()),
                    "geo": float(geo_loss.detach().cpu()),
                    "total": float(loss.detach().cpu()),
                    "lr": float(opt.param_groups[0]["lr"]),
                }
                losses.append(row)
                if self.global_step % self.cfg.runtime.log_every == 0:
                    pbar.set_postfix(row)
        summary = {
            "stage": stage_idx,
            "beta": beta,
            "num_batches": len(losses),
            "loss_mean": {k: float(np.mean([x[k] for x in losses])) for k in losses[0].keys()} if losses else {},
        }
        return summary

    @torch.no_grad()
    def update_memory(self, stage_idx: int, dataset: torch.utils.data.Dataset) -> Dict[str, int]:
        self.model.eval()
        loader = make_dataloader(
            dataset,
            batch_size=self.cfg.optim.batch_size,
            shuffle=False,
            num_workers=self.cfg.runtime.num_workers,
        )
        descriptors = []
        angular = []
        skill_probs = []
        for batch in loader:
            batch = to_device(batch, self.device)
            out = self.model(**model_batch_kwargs(batch))
            spec = self.spectral_encoder(out.cross_attentions, batch.get("functional_token_mask"))
            descriptors.append(spec.descriptor.detach().cpu())
            angular.append(spec.angular.detach().cpu())
            skill_probs.append(batch["skill_probs"].detach().cpu())
        counts = self.memory.update(torch.cat(descriptors), torch.cat(angular), torch.cat(skill_probs))
        save_json(counts, self.out_dir / f"stage_{stage_idx:02d}_prototype_counts.json")
        self.model.train()
        return counts

    @torch.no_grad()
    def evaluate(self, dataset: torch.utils.data.Dataset) -> Dict[str, float]:
        self.model.eval()
        loader = make_dataloader(
            dataset,
            batch_size=self.cfg.optim.batch_size,
            shuffle=False,
            num_workers=self.cfg.runtime.num_workers,
        )
        total_correct = 0
        total = 0
        total_loss = 0.0
        for batch in loader:
            batch = to_device(batch, self.device)
            out = self.model(**model_batch_kwargs(batch))
            loss = F.cross_entropy(out.logits, batch["labels"], reduction="sum")
            pred = out.logits.argmax(dim=-1)
            total_correct += int((pred == batch["labels"]).sum().item())
            total += int(batch["labels"].numel())
            total_loss += float(loss.detach().cpu())
        self.model.train()
        return {"accuracy": total_correct / max(1, total), "loss": total_loss / max(1, total)}

    def _save_stage(self, stage_idx: int) -> None:
        if not self.cfg.runtime.save_every_stage:
            return
        torch.save(
            {
                "stage": stage_idx,
                "model": self.model.state_dict(),
                "memory": self.memory.state_dict(),
                "config": self.cfg.to_dict(),
            },
            self.out_dir / f"stage_{stage_idx:02d}.pt",
        )
