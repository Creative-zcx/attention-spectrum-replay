from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch

torch.set_num_threads(1)

from .config import ExperimentConfig
from .data import JsonlMultimodalDataset, SimpleTokenizer, build_synthetic_stream, load_answer_vocab
from .memory import PrototypeMemory
from .models.tiny_mllm import TinyMLLM
from .models.llava_adapter import HFMLLMAdapter
from .spectral import SpectralEncoder
from .trainer import ASRTrainer
from .utils import ensure_dir, set_seed


def build_model_and_data(cfg: ExperimentConfig):
    skills = cfg.skills
    if cfg.data.kind == "synthetic":
        train_sets, val_sets, tokenizer = build_synthetic_stream(cfg, skills)
        cfg.model.vocab_size = max(cfg.model.vocab_size, len(tokenizer.id_to_token))
        cfg.model.num_classes = cfg.data.num_classes
        model = TinyMLLM(
            vocab_size=cfg.model.vocab_size,
            num_classes=cfg.model.num_classes,
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            num_heads=cfg.model.num_heads,
            image_size=cfg.model.image_size,
            patch_size=cfg.model.patch_size,
            max_seq_len=cfg.model.max_seq_len,
        )
        return model, train_sets, val_sets, tokenizer

    if cfg.data.kind == "jsonl":
        if cfg.data.manifest_dir is None or cfg.data.image_root is None:
            raise ValueError("jsonl data requires data.manifest_dir and data.image_root")
        manifest_dir = Path(cfg.data.manifest_dir)
        train_files = sorted(manifest_dir.glob("stage_*_train.jsonl"))
        val_files = sorted(manifest_dir.glob("stage_*_val.jsonl"))
        if not train_files:
            raise ValueError(f"No stage_*_train.jsonl files found in {manifest_dir}")
        tokenizer_path = manifest_dir / "tokenizer.json"
        tokenizer = SimpleTokenizer(max_len=cfg.data.max_seq_len)
        if tokenizer_path.exists():
            tokenizer = SimpleTokenizer.load(tokenizer_path)
        else:
            texts = []
            for f in train_files + val_files:
                for line in f.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        texts.append(json.loads(line)["question"])
            tokenizer.fit(texts)
            tokenizer.save(tokenizer_path)
        answer_to_id = load_answer_vocab(cfg.data.answer_vocab)
        if answer_to_id:
            cfg.model.num_classes = max(cfg.model.num_classes, max(answer_to_id.values()) + 1)
        train_sets = [
            JsonlMultimodalDataset(f, cfg.data.image_root, tokenizer, skills, answer_to_id, cfg.model.image_size)
            for f in train_files
        ]
        val_sets = [
            JsonlMultimodalDataset(f, cfg.data.image_root, tokenizer, skills, answer_to_id, cfg.model.image_size)
            for f in val_files
        ] if val_files else None
        cfg.model.vocab_size = max(cfg.model.vocab_size, len(tokenizer.id_to_token))
        if cfg.model.kind == "tiny":
            model = TinyMLLM(
                vocab_size=cfg.model.vocab_size,
                num_classes=cfg.model.num_classes,
                hidden_dim=cfg.model.hidden_dim,
                num_layers=cfg.model.num_layers,
                num_heads=cfg.model.num_heads,
                image_size=cfg.model.image_size,
                patch_size=cfg.model.patch_size,
                max_seq_len=cfg.model.max_seq_len,
            )
        elif cfg.model.kind in {"llava", "hf"}:
            if cfg.model.hf_model_name_or_path is None:
                raise ValueError("HF model requires model.hf_model_name_or_path")
            model = HFMLLMAdapter(
                model_name_or_path=cfg.model.hf_model_name_or_path,
                num_classes=cfg.model.num_classes,
                lora_rank=cfg.model.lora_rank,
                lora_alpha=cfg.model.lora_alpha,
                lora_dropout=cfg.model.lora_dropout,
            )
        else:
            raise ValueError(f"Unknown model kind: {cfg.model.kind}")
        return model, train_sets, val_sets, tokenizer
    raise ValueError(f"Unknown data kind: {cfg.data.kind}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Attention-Spectrum Replay")
    parser.add_argument("--config", required=True, help="YAML experiment config")
    parser.add_argument("--output-dir", default=None, help="Override runtime.output_dir")
    args = parser.parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    if args.output_dir:
        cfg.runtime.output_dir = args.output_dir
    set_seed(cfg.runtime.seed)
    ensure_dir(cfg.runtime.output_dir)
    model, train_sets, val_sets, tokenizer = build_model_and_data(cfg)
    encoder = SpectralEncoder(cfg.spectral)
    memory = PrototypeMemory(cfg.skills, encoder.descriptor_dim, cfg.spectral.angular_bins, cfg.asr)
    cfg.save_yaml(Path(cfg.runtime.output_dir) / "resolved_config.yaml")
    trainer = ASRTrainer(cfg, model, encoder, memory, train_sets, val_sets)
    history = trainer.fit()
    print(json.dumps(history.get("final_metrics", {}), indent=2))


if __name__ == "__main__":
    main()
