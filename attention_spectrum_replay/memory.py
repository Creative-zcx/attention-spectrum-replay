from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from .config import ASRConfig
from .spectral import symmetric_kl
from .utils import ensure_dir, safe_torch_load


@dataclass
class Prototype:
    mean: torch.Tensor
    var: torch.Tensor
    angular: torch.Tensor
    count: int
    initialized: bool = True


class PrototypeMemory:
    """Skill-conditioned Gaussian spectral prototype memory.

    Stores (mu_s, diagonal Sigma_s, angular prototype d_hat_s) for every skill.
    No raw image/question/answer data is stored.
    """

    def __init__(self, skills: List[str], descriptor_dim: int, angular_bins: int, cfg: ASRConfig):
        self.skills = list(skills)
        self.skill_to_id = {s: i for i, s in enumerate(self.skills)}
        self.descriptor_dim = int(descriptor_dim)
        self.angular_bins = int(angular_bins)
        self.cfg = cfg
        self.prototypes: Dict[str, Prototype] = {}

    def __contains__(self, skill: str) -> bool:
        return skill in self.prototypes and self.prototypes[skill].initialized

    def initialized_skills(self) -> List[str]:
        return [s for s in self.skills if s in self]

    def is_initialized(self) -> bool:
        return any(s in self for s in self.skills)

    def update(self, descriptors: torch.Tensor, angular: torch.Tensor, skill_probs: torch.Tensor) -> Dict[str, int]:
        if descriptors.ndim != 2 or descriptors.size(-1) != self.descriptor_dim:
            raise ValueError(f"descriptors must be [N,{self.descriptor_dim}], got {tuple(descriptors.shape)}")
        if angular.ndim != 2 or angular.size(-1) != self.angular_bins:
            raise ValueError(f"angular must be [N,{self.angular_bins}], got {tuple(angular.shape)}")
        if skill_probs.ndim != 2 or skill_probs.size(-1) != len(self.skills):
            raise ValueError(f"skill_probs must be [N,{len(self.skills)}], got {tuple(skill_probs.shape)}")
        descriptors = descriptors.detach().cpu().float()
        angular = angular.detach().cpu().float()
        skill_probs = skill_probs.detach().cpu().float()
        counts: Dict[str, int] = {}
        for sid, skill in enumerate(self.skills):
            mask = skill_probs[:, sid] > self.cfg.tau_skill
            n = int(mask.sum().item())
            counts[skill] = n
            if n == 0:
                continue
            x = descriptors[mask]
            a = angular[mask]
            mean = x.mean(dim=0)
            if n > 1:
                var = x.var(dim=0, unbiased=True)
            else:
                var = torch.full_like(mean, self.cfg.cov_floor)
            var = var.clamp_min(self.cfg.cov_floor)
            ang = a.mean(dim=0)
            ang = ang / ang.sum().clamp_min(1e-8)
            if skill not in self.prototypes or not self.prototypes[skill].initialized:
                self.prototypes[skill] = Prototype(mean=mean, var=var, angular=ang, count=n)
            else:
                old = self.prototypes[skill]
                alpha = self.cfg.alpha
                new_mean = alpha * old.mean + (1.0 - alpha) * mean
                new_var = (alpha * old.var + (1.0 - alpha) * var).clamp_min(self.cfg.cov_floor)
                new_ang = alpha * old.angular + (1.0 - alpha) * ang
                new_ang = new_ang / new_ang.sum().clamp_min(1e-8)
                self.prototypes[skill] = Prototype(
                    mean=new_mean,
                    var=new_var,
                    angular=new_ang,
                    count=old.count + n,
                )
        return counts

    def tensors(self, device: torch.device | str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        means = []
        vars_ = []
        angular = []
        mask = []
        for skill in self.skills:
            proto = self.prototypes.get(skill)
            if proto is None or not proto.initialized:
                means.append(torch.zeros(self.descriptor_dim))
                vars_.append(torch.ones(self.descriptor_dim))
                angular.append(torch.ones(self.angular_bins) / self.angular_bins)
                mask.append(0.0)
            else:
                means.append(proto.mean)
                vars_.append(proto.var.clamp_min(self.cfg.cov_floor))
                angular.append(proto.angular)
                mask.append(1.0)
        return (
            torch.stack(means).to(device),
            torch.stack(vars_).to(device),
            torch.stack(angular).to(device),
            torch.tensor(mask, device=device, dtype=torch.float32),
        )

    def spectral_loss(self, descriptors: torch.Tensor, angular: torch.Tensor, skill_probs: torch.Tensor) -> torch.Tensor:
        if not self.is_initialized():
            return descriptors.new_zeros(())
        means, vars_, proto_ang, init_mask = self.tensors(descriptors.device)
        diff = descriptors[:, None, :] - means[None, :, :]
        d2 = (diff.square() / vars_[None, :, :].clamp_min(self.cfg.cov_floor)).sum(dim=-1)
        if self.cfg.normalize_mahalanobis_by_dim:
            d2 = d2 / max(1, self.descriptor_dim)
        active_probs = skill_probs.to(descriptors.device, descriptors.dtype) * init_mask[None, :]
        prob_sum = active_probs.sum(dim=-1, keepdim=True)
        active_probs = torch.where(prob_sum > 0, active_probs / prob_sum.clamp_min(1e-8), active_probs)
        mah = (active_probs * d2).sum(dim=-1)

        p = angular[:, None, :].expand(-1, len(self.skills), -1)
        q = proto_ang[None, :, :].expand(angular.size(0), -1, -1)
        kl = symmetric_kl(p.reshape(-1, self.angular_bins), q.reshape(-1, self.angular_bins), eps=1e-8).reshape(
            angular.size(0), len(self.skills)
        )
        ang = (active_probs * kl).sum(dim=-1)
        masked_d2 = torch.where(init_mask[None, :] > 0, d2, torch.full_like(d2, 1e9))
        dmin = masked_d2.min(dim=-1).values
        weight = torch.exp(-dmin / max(self.cfg.tau_mah, 1e-8))
        has_active = (prob_sum.squeeze(-1) > 0).to(descriptors.dtype)
        return (has_active * weight * (mah + self.cfg.lambda_ang * ang)).sum() / has_active.sum().clamp_min(1.0)

    def state_dict(self) -> Dict[str, object]:
        return {
            "skills": self.skills,
            "descriptor_dim": self.descriptor_dim,
            "angular_bins": self.angular_bins,
            "cfg": self.cfg.__dict__,
            "prototypes": {
                k: {
                    "mean": v.mean,
                    "var": v.var,
                    "angular": v.angular,
                    "count": v.count,
                    "initialized": v.initialized,
                }
                for k, v in self.prototypes.items()
            },
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, object]) -> "PrototypeMemory":
        cfg = ASRConfig(**state["cfg"])
        mem = cls(state["skills"], int(state["descriptor_dim"]), int(state["angular_bins"]), cfg)
        for k, v in state["prototypes"].items():
            mem.prototypes[k] = Prototype(
                mean=v["mean"].float(),
                var=v["var"].float(),
                angular=v["angular"].float(),
                count=int(v["count"]),
                initialized=bool(v.get("initialized", True)),
            )
        return mem

    def save(self, path: str | Path) -> None:
        path = Path(path)
        ensure_dir(path.parent)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "PrototypeMemory":
        return cls.from_state_dict(safe_torch_load(path, map_location=map_location))
