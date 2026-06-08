from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SpectralConfig


@dataclass
class SpectralBatch:
    descriptor: torch.Tensor       # [B, D]
    angular: torch.Tensor          # [B, M]
    radial: torch.Tensor           # [B, K]
    per_head_descriptor: torch.Tensor  # [B, J, D0]


class SpectralEncoder(nn.Module):
    """ASR spectral encoder for cross-attention maps.

    Expected attention tensor layouts:
      * [B, L, Hh, Tq, Gh, Gw]
      * [B, L, Hh, Tq, P] plus grid_shape=(Gh, Gw)
      * list/tuple of L tensors shaped [B, Hh, Tq, Gh, Gw] or [B, Hh, Tq, P]

    It implements the paper descriptor: radial spectrum, angular spectrum,
    angular anisotropy, dominant spectral radius, dominant spectral angle and
    dominant power per selected layer-head, followed by layer-head mean and
    variance/std aggregation.
    """

    def __init__(self, cfg: SpectralConfig):
        super().__init__()
        if cfg.radial_bins <= 0 or cfg.angular_bins <= 0:
            raise ValueError("radial_bins and angular_bins must be positive")
        if cfg.frequency_mode not in {"fftfreq", "raw_paper"}:
            raise ValueError("frequency_mode must be 'fftfreq' or 'raw_paper'")
        if cfg.fft_norm not in {"ortho", "backward"}:
            raise ValueError("fft_norm must be 'ortho' or 'backward'")
        if cfg.head_stat not in {"variance", "std"}:
            raise ValueError("head_stat must be 'variance' or 'std'")
        self.cfg = cfg

    @property
    def descriptor_dim_per_head(self) -> int:
        return self.cfg.radial_bins + 2 * self.cfg.angular_bins + 3

    @property
    def descriptor_dim(self) -> int:
        return 2 * self.descriptor_dim_per_head

    def forward(
        self,
        attentions: torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, ...],
        functional_token_mask: Optional[torch.Tensor] = None,
        grid_shape: Optional[Tuple[int, int]] = None,
    ) -> SpectralBatch:
        attn = self._standardize_attention(attentions, grid_shape=grid_shape)
        # attn: [B, L, Hh, Tq, Gh, Gw]
        if self.cfg.selected_last_layers > 0 and attn.size(1) > self.cfg.selected_last_layers:
            attn = attn[:, -self.cfg.selected_last_layers :]

        maps = self._aggregate_tokens(attn, functional_token_mask)
        # maps: [B, J, Gh, Gw]
        bsz, num_lh, gh, gw = maps.shape
        flat_maps = maps.reshape(bsz * num_lh, gh, gw)
        per = self._encode_maps(flat_maps).reshape(bsz, num_lh, -1)

        mean = per.mean(dim=1)
        centered = per - mean.unsqueeze(1)
        variance = (centered * centered).mean(dim=1)
        second = torch.sqrt(variance + self.cfg.eps) if self.cfg.head_stat == "std" else variance
        descriptor = torch.cat([mean, second], dim=-1)

        k = self.cfg.radial_bins
        m = self.cfg.angular_bins
        radial = per[..., :k].mean(dim=1)
        angular = per[..., k : k + m].mean(dim=1)
        radial = _normalize_distribution(radial, self.cfg.eps)
        angular = _normalize_distribution(angular, self.cfg.eps)
        return SpectralBatch(
            descriptor=descriptor,
            angular=angular,
            radial=radial,
            per_head_descriptor=per,
        )

    def _standardize_attention(
        self,
        attentions: torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, ...],
        grid_shape: Optional[Tuple[int, int]],
    ) -> torch.Tensor:
        if isinstance(attentions, (list, tuple)):
            if len(attentions) == 0:
                raise ValueError("attentions list is empty")
            layers = []
            for a in attentions:
                if a.dim() == 4:
                    if grid_shape is None:
                        p = a.size(-1)
                        side = int(p ** 0.5)
                        if side * side != p:
                            raise ValueError("grid_shape required for non-square visual-token count")
                        gh, gw = side, side
                    else:
                        gh, gw = grid_shape
                    a = a.reshape(a.size(0), a.size(1), a.size(2), gh, gw)
                elif a.dim() != 5:
                    raise ValueError(f"layer attention must be 4D or 5D, got shape {tuple(a.shape)}")
                layers.append(a)
            return torch.stack(layers, dim=1)

        attn = attentions
        if attn.dim() == 5:
            if grid_shape is None:
                p = attn.size(-1)
                side = int(p ** 0.5)
                if side * side != p:
                    raise ValueError("grid_shape required for non-square visual-token count")
                gh, gw = side, side
            else:
                gh, gw = grid_shape
            return attn.reshape(attn.size(0), attn.size(1), attn.size(2), attn.size(3), gh, gw)
        if attn.dim() == 6:
            return attn
        raise ValueError("attentions must be 5D/6D tensor or list of layer tensors")

    def _aggregate_tokens(self, attn: torch.Tensor, functional_token_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # attn [B, L, Hh, Tq, Gh, Gw]
        bsz, layers, heads, tq, gh, gw = attn.shape
        attn = attn.clamp_min(0)
        if functional_token_mask is None:
            token_weights = torch.ones(bsz, tq, device=attn.device, dtype=attn.dtype) / max(tq, 1)
        else:
            token_weights = functional_token_mask.to(device=attn.device, dtype=attn.dtype)
            if token_weights.dim() != 2 or token_weights.shape != (bsz, tq):
                raise ValueError(
                    f"functional_token_mask must have shape {(bsz, tq)}, got {tuple(token_weights.shape)}"
                )
            denom = token_weights.sum(dim=1, keepdim=True)
            fallback = torch.ones_like(token_weights) / max(tq, 1)
            token_weights = torch.where(denom > 0, token_weights / denom.clamp_min(self.cfg.eps), fallback)
        maps = (attn * token_weights[:, None, None, :, None, None]).sum(dim=3)
        maps = maps.reshape(bsz, layers * heads, gh, gw)
        maps = maps / maps.sum(dim=(-1, -2), keepdim=True).clamp_min(self.cfg.eps)
        return maps

    def _encode_maps(self, maps: torch.Tensor) -> torch.Tensor:
        # maps [N, Gh, Gw]
        n, gh, gw = maps.shape
        fft = torch.fft.fft2(maps, dim=(-2, -1), norm=self.cfg.fft_norm)
        power = (fft.real.square() + fft.imag.square()).clamp_min(0)
        power_sum = power.sum(dim=(-1, -2), keepdim=True).clamp_min(self.cfg.eps)
        power_norm = power / power_sum
        power_for_bins = power_norm

        coords = self._coords(gh, gw, maps.device, maps.dtype)
        radial_masks = coords["radial_masks"]  # [K, Gh*Gw]
        angular_masks = coords["angular_masks"]  # [M, Gh*Gw]
        rho_flat = coords["rho_flat"]
        theta_flat = coords["theta_flat"]

        flat = power_for_bins.reshape(n, gh * gw)
        radial_energy = flat @ radial_masks.t()
        angular_energy = flat @ angular_masks.t()
        radial = _normalize_distribution(radial_energy, self.cfg.eps)
        angular = _normalize_distribution(angular_energy, self.cfg.eps)
        anisotropy = angular / angular.mean(dim=-1, keepdim=True).clamp_min(self.cfg.eps)

        peak_source = power_norm if self.cfg.normalize_peak_power else power
        peak_flat = peak_source.reshape(n, gh * gw)
        peak_idx = torch.argmax(peak_flat, dim=-1)
        peak_power = peak_flat.gather(1, peak_idx[:, None]).squeeze(1)
        rho_star = rho_flat.gather(0, peak_idx).to(maps.dtype)
        theta_star = theta_flat.gather(0, peak_idx).to(maps.dtype)

        return torch.cat(
            [radial, angular, anisotropy, rho_star[:, None], theta_star[:, None], peak_power[:, None]],
            dim=-1,
        )

    def _coords(self, gh: int, gw: int, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        # Not registered as buffers because H/W/device can vary across backbones.
        if self.cfg.frequency_mode == "fftfreq":
            fy = torch.fft.fftfreq(gh, d=1.0, device=device).to(dtype)
            fx = torch.fft.fftfreq(gw, d=1.0, device=device).to(dtype)
        else:
            fy = (torch.arange(gh, device=device, dtype=dtype) / max(gh, 1)).to(dtype)
            fx = (torch.arange(gw, device=device, dtype=dtype) / max(gw, 1)).to(dtype)
        yy, xx = torch.meshgrid(fy, fx, indexing="ij")
        rho = torch.sqrt(yy.square() + xx.square())
        rho = rho / rho.max().clamp_min(self.cfg.eps)
        theta = torch.atan2(xx, yy)  # matches paper convention atan2(v/W, u/H)

        rho_flat = rho.reshape(-1)
        theta_flat = theta.reshape(-1)

        radial_edges = torch.linspace(0.0, 1.0 + self.cfg.eps, self.cfg.radial_bins + 1, device=device, dtype=dtype)
        angular_edges = torch.linspace(-torch.pi, torch.pi + self.cfg.eps, self.cfg.angular_bins + 1, device=device, dtype=dtype)
        radial_bin = torch.bucketize(rho_flat, radial_edges, right=False) - 1
        radial_bin = radial_bin.clamp(0, self.cfg.radial_bins - 1)
        angular_bin = torch.bucketize(theta_flat, angular_edges, right=False) - 1
        angular_bin = angular_bin.clamp(0, self.cfg.angular_bins - 1)
        radial_masks = F.one_hot(radial_bin, num_classes=self.cfg.radial_bins).to(dtype).t().contiguous()
        angular_masks = F.one_hot(angular_bin, num_classes=self.cfg.angular_bins).to(dtype).t().contiguous()
        return {
            "rho_flat": rho_flat,
            "theta_flat": theta_flat,
            "radial_masks": radial_masks,
            "angular_masks": angular_masks,
        }


def _normalize_distribution(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x / x.sum(dim=-1, keepdim=True).clamp_min(eps)


def symmetric_kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = _normalize_distribution(p.clamp_min(eps), eps)
    q = _normalize_distribution(q.clamp_min(eps), eps)
    kl_pq = (p * (p.log() - q.log())).sum(dim=-1)
    kl_qp = (q * (q.log() - p.log())).sum(dim=-1)
    return 0.5 * (kl_pq + kl_qp)


def jensen_shannon_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = _normalize_distribution(p.clamp_min(eps), eps)
    q = _normalize_distribution(q.clamp_min(eps), eps)
    m = 0.5 * (p + q)
    return 0.5 * ((p * (p.log() - m.log())).sum(dim=-1) + (q * (q.log() - m.log())).sum(dim=-1))


def high_frequency_loss(prev_radial: torch.Tensor, cur_radial: torch.Tensor, rho0_bin: int) -> torch.Tensor:
    if rho0_bin < 0 or rho0_bin >= prev_radial.size(-1):
        raise ValueError("rho0_bin must index a valid radial bin")
    return (prev_radial[..., rho0_bin + 1 :] - cur_radial[..., rho0_bin + 1 :]).sum(dim=-1)
