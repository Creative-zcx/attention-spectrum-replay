from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

from .skill_parser import HeuristicSkillParser, WordVocab


FUNCTION_WORDS = {
    "what", "where", "which", "who", "how", "many", "color", "word", "text", "sign", "read",
    "count", "number", "left", "right", "front", "behind", "next", "object", "doing", "is", "are",
    "there", "sitting", "printed", "closest", "relation", "shape", "size"
}


class SimpleTokenizer:
    def __init__(self, max_len: int = 32):
        self.max_len = int(max_len)
        self.token_to_id = {"<pad>": 0, "<unk>": 1, "<image>": 2}
        self.id_to_token = ["<pad>", "<unk>", "<image>"]

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r"<image>|[a-zA-Z0-9']+", text.lower())

    def fit(self, texts: Iterable[str]) -> None:
        for text in texts:
            for tok in self.tokenize(text):
                if tok not in self.token_to_id:
                    self.token_to_id[tok] = len(self.id_to_token)
                    self.id_to_token.append(tok)

    def encode(self, text: str) -> Dict[str, torch.Tensor]:
        toks = self.tokenize(text)[: self.max_len]
        ids = [self.token_to_id.get(tok, 1) for tok in toks]
        mask = [1] * len(ids)
        func = [1 if tok in FUNCTION_WORDS else 0 for tok in toks]
        while len(ids) < self.max_len:
            ids.append(0)
            mask.append(0)
            func.append(0)
        if sum(func) == 0:
            # Use all non-padding tokens when no explicit functional tokens are detected.
            func = [m for m in mask]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "functional_token_mask": torch.tensor(func, dtype=torch.float32),
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"id_to_token": self.id_to_token, "max_len": self.max_len}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SimpleTokenizer":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls(raw.get("max_len", 32))
        tok.id_to_token = list(raw["id_to_token"])
        tok.token_to_id = {t: i for i, t in enumerate(tok.id_to_token)}
        return tok


class SyntheticContinualDataset(Dataset):
    """Small deterministic multimodal stream for CPU development.

    The images contain simple colored squares at locations. Questions encode
    one dominant skill, and labels are low-cardinality classification targets.
    """

    def __init__(
        self,
        stage: int,
        split: str,
        tokenizer: SimpleTokenizer,
        skills: Sequence[str],
        n_samples: int,
        num_classes: int = 8,
        image_size: int = 32,
        seed: int = 0,
    ):
        self.stage = int(stage)
        self.split = split
        self.tokenizer = tokenizer
        self.skills = list(skills)
        self.n_samples = int(n_samples)
        self.num_classes = int(num_classes)
        self.image_size = int(image_size)
        self.seed = int(seed) + stage * 1009 + (0 if split == "train" else 7919)
        self.parser = HeuristicSkillParser(self.skills)
        self.records = [self._make_record(i) for i in range(self.n_samples)]

    def __len__(self) -> int:
        return len(self.records)

    def _make_record(self, idx: int) -> Dict[str, object]:
        rng = random.Random(self.seed + idx)
        # Cycle through a subset of skills for each stage, with overlap to test soft memory.
        primary_skill = self.skills[(self.stage + idx) % min(len(self.skills), 5)]
        color_names = ["red", "green", "blue", "yellow"]
        color_idx = rng.randrange(4)
        count = 1 + rng.randrange(4)
        loc_idx = rng.randrange(4)
        if primary_skill == "count":
            q = "USER: <image> QUESTION: How many colored squares are visible? ASSISTANT:"
            label = count % self.num_classes
        elif primary_skill == "color":
            q = "USER: <image> QUESTION: What color is the largest square? ASSISTANT:"
            label = color_idx % self.num_classes
        elif primary_skill == "locate":
            q = "USER: <image> QUESTION: Where is the bright square sitting? ASSISTANT:"
            label = loc_idx % self.num_classes
        elif primary_skill == "relation":
            q = "USER: <image> QUESTION: Is the blue square left of the red square? ASSISTANT:"
            label = rng.randrange(2)
        elif primary_skill == "read":
            q = "USER: <image> QUESTION: What word is printed on the sign? ASSISTANT:"
            label = (idx + self.stage) % self.num_classes
        else:
            q = "USER: <image> QUESTION: What object is shown? ASSISTANT:"
            label = (idx + color_idx) % self.num_classes
        return {
            "question": q,
            "label": label,
            "skill": primary_skill,
            "color_idx": color_idx,
            "count": count,
            "loc_idx": loc_idx,
            "idx": idx,
        }

    def _draw_image(self, rec: Dict[str, object]) -> torch.Tensor:
        size = self.image_size
        img = torch.zeros(3, size, size, dtype=torch.float32)
        colors = torch.tensor(
            [[1.0, 0.15, 0.15], [0.15, 1.0, 0.15], [0.15, 0.25, 1.0], [1.0, 0.9, 0.1]],
            dtype=torch.float32,
        )
        positions = [(2, 2), (size // 2, 2), (2, size // 2), (size // 2, size // 2)]
        count = int(rec["count"])
        color_idx = int(rec["color_idx"])
        for k in range(count):
            y, x = positions[(int(rec["loc_idx"]) + k) % len(positions)]
            c = colors[(color_idx + k) % 4]
            img[:, y : y + size // 3, x : x + size // 3] = c[:, None, None]
        # Stage-specific weak background to ensure non-stationarity.
        img += 0.02 * self.stage
        return img.clamp(0, 1)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        rec = self.records[idx]
        enc = self.tokenizer.encode(rec["question"])
        probs = self.parser([rec["question"]]).squeeze(0)
        return {
            "images": self._draw_image(rec),
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "functional_token_mask": enc["functional_token_mask"],
            "labels": torch.tensor(int(rec["label"]), dtype=torch.long),
            "skill_probs": probs,
            "question_text": rec["question"],
            "stage": torch.tensor(self.stage, dtype=torch.long),
        }


def build_synthetic_stream(cfg, skills: Sequence[str]) -> Tuple[List[Dataset], List[Dataset], SimpleTokenizer]:
    tokenizer = SimpleTokenizer(max_len=cfg.data.max_seq_len)
    texts = []
    for stage in range(cfg.data.num_stages):
        tmp = SyntheticContinualDataset(stage, "train", tokenizer, skills, cfg.data.train_samples_per_stage, cfg.data.num_classes, cfg.model.image_size, cfg.data.seed)
        texts.extend([r["question"] for r in tmp.records])
    tokenizer.fit(texts)
    train_sets = [
        SyntheticContinualDataset(
            stage,
            "train",
            tokenizer,
            skills,
            cfg.data.train_samples_per_stage,
            cfg.data.num_classes,
            cfg.model.image_size,
            cfg.data.seed,
        )
        for stage in range(cfg.data.num_stages)
    ]
    val_sets = [
        SyntheticContinualDataset(
            stage,
            "val",
            tokenizer,
            skills,
            cfg.data.val_samples_per_stage,
            cfg.data.num_classes,
            cfg.model.image_size,
            cfg.data.seed,
        )
        for stage in range(cfg.data.num_stages)
    ]
    return train_sets, val_sets, tokenizer


class JsonlMultimodalDataset(Dataset):
    """Generic JSONL dataset for VQA/CoIN/UCIT manifests.

    Required fields per line:
      image: path relative to image_root or absolute path
      question: prompt/question/instruction string
      label: integer class label, or answer string if answer_vocab is provided
    Optional fields:
      skill_probs: list of floats over skills
      skill: skill name
      functional_token_mask: list of 0/1 values aligned to tokenized text
    """

    def __init__(
        self,
        manifest: str | Path,
        image_root: str | Path,
        tokenizer: SimpleTokenizer,
        skills: Sequence[str],
        answer_to_id: Optional[Dict[str, int]] = None,
        image_size: int = 336,
    ):
        self.manifest = Path(manifest)
        self.image_root = Path(image_root)
        self.tokenizer = tokenizer
        self.skills = list(skills)
        self.skill_to_id = {s: i for i, s in enumerate(self.skills)}
        self.answer_to_id = answer_to_id
        self.image_size = int(image_size)
        self.parser = HeuristicSkillParser(self.skills)
        self.records = [json.loads(line) for line in self.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.records:
            raise ValueError(f"Empty manifest: {self.manifest}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        rec = self.records[idx]
        q = rec["question"]
        enc = self.tokenizer.encode(q)
        label = rec.get("label", rec.get("answer"))
        if isinstance(label, str):
            if self.answer_to_id is None:
                raise ValueError("String labels require answer_vocab/answer_to_id")
            label = self.answer_to_id[label]
        skill_probs = rec.get("skill_probs")
        if skill_probs is None:
            skill = rec.get("skill")
            if skill in self.skill_to_id:
                probs = torch.full((len(self.skills),), 0.05 / max(1, len(self.skills) - 1), dtype=torch.float32)
                probs[self.skill_to_id[skill]] = 0.95
            else:
                probs = self.parser([q]).squeeze(0)
        else:
            probs = torch.tensor(skill_probs, dtype=torch.float32)
            probs = probs / probs.sum().clamp_min(1e-8)
        return {
            "images": preprocess_image(self._image_path(rec["image"]), self.image_size),
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "functional_token_mask": enc["functional_token_mask"],
            "labels": torch.tensor(int(label), dtype=torch.long),
            "skill_probs": probs,
            "question_text": q,
            "stage": torch.tensor(int(rec.get("stage", 0)), dtype=torch.long),
        }

    def _image_path(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else self.image_root / path


def preprocess_image(path: str | Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image.thumbnail((image_size, image_size), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (image_size, image_size), (0, 0, 0))
    x = (image_size - image.width) // 2
    y = (image_size - image.height) // 2
    canvas.paste(image, (x, y))
    arr = np.asarray(canvas).astype("float32") / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])[:, None, None]
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711])[:, None, None]
    return (tensor - mean) / std


def collate_samples(items: List[Dict[str, object]]) -> Dict[str, object]:
    tensor_keys = ["images", "input_ids", "attention_mask", "functional_token_mask", "labels", "skill_probs", "stage"]
    batch: Dict[str, object] = {}
    for key in tensor_keys:
        if key in items[0]:
            batch[key] = torch.stack([x[key] for x in items])
    batch["question_text"] = [str(x.get("question_text", "")) for x in items]
    return batch


def make_dataloader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int = 0) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_samples)


def load_answer_vocab(path: Optional[str | Path]) -> Optional[Dict[str, int]]:
    if path is None:
        return None
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        return {str(k): int(v) for k, v in raw.items()}
    return {str(v): i for i, v in enumerate(raw)}
