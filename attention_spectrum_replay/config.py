from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class SpectralConfig:
    radial_bins: int = 8
    angular_bins: int = 8
    eps: float = 1.0e-8
    selected_last_layers: int = 4
    frequency_mode: str = "fftfreq"  # fftfreq or raw_paper
    fft_norm: str = "ortho"          # ortho or backward
    head_stat: str = "variance"      # variance or std
    normalize_peak_power: bool = True


@dataclass
class ASRConfig:
    beta: float = 0.5
    lambda_ang: float = 0.1
    tau_mah: float = 1.0
    alpha: float = 0.9
    tau_skill: float = 0.2
    cov_floor: float = 1.0e-4
    normalize_mahalanobis_by_dim: bool = True
    gamma: float = 0.05


@dataclass
class OptimConfig:
    lr: float = 2.0e-4
    weight_decay: float = 0.01
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    epochs_per_stage: int = 1
    batch_size: int = 16
    max_grad_norm: float = 1.0
    cosine_decay: bool = True
    warmup_steps: int = 0


@dataclass
class ModelConfig:
    kind: str = "tiny"
    hidden_dim: int = 64
    num_layers: int = 4
    num_heads: int = 4
    image_size: int = 32
    patch_size: int = 8
    vocab_size: int = 4096
    num_classes: int = 8
    max_seq_len: int = 32
    freeze_backbone: bool = False
    hf_model_name_or_path: Optional[str] = None
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


@dataclass
class DataConfig:
    kind: str = "synthetic"
    num_stages: int = 3
    train_samples_per_stage: int = 24
    val_samples_per_stage: int = 12
    num_classes: int = 8
    seed: int = 123
    manifest_dir: Optional[str] = None
    image_root: Optional[str] = None
    max_seq_len: int = 32
    task_order: List[str] = field(default_factory=lambda: [
        "count", "color", "locate", "relation", "read", "object", "activity", "attribute", "existence", "other"
    ])
    answer_vocab: Optional[str] = None


@dataclass
class RuntimeConfig:
    device: str = "cpu"
    seed: int = 7
    num_workers: int = 0
    output_dir: str = "runs/asr"
    log_every: int = 10
    save_every_stage: bool = True
    use_amp: bool = False


@dataclass
class ExperimentConfig:
    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    method: str = "asr"  # asr, vanilla, ewc, lwf, er
    skills: List[str] = field(default_factory=lambda: [
        "count", "color", "locate", "relation", "read", "object", "activity", "attribute", "existence", "other"
    ])

    @staticmethod
    def from_yaml(path: str | Path) -> "ExperimentConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return ExperimentConfig.from_dict(raw)

    @staticmethod
    def from_dict(raw: Dict[str, Any]) -> "ExperimentConfig":
        cfg = ExperimentConfig()
        for section_name in ["spectral", "asr", "optim", "model", "data", "runtime"]:
            if section_name in raw and raw[section_name] is not None:
                section_obj = getattr(cfg, section_name)
                for key, value in raw[section_name].items():
                    if not hasattr(section_obj, key):
                        raise KeyError(f"Unknown config key {section_name}.{key}")
                    setattr(section_obj, key, value)
        for key in ["method", "skills"]:
            if key in raw:
                setattr(cfg, key, raw[key])
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
