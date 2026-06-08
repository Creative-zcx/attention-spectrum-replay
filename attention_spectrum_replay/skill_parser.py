from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeuristicSkillParser:
    """Lightweight deterministic skill parser used for CPU and fallback runs.

    The paper trains a Transformer classifier offline. For a full run, use
    TransformerSkillParser and train_skill_parser.py. This parser is included
    for deterministic CPU tests and as a transparent template-based baseline.
    """

    DEFAULT_PATTERNS: Dict[str, List[str]] = {
        "count": [r"\bhow many\b", r"\bnumber of\b", r"\bcount\b"],
        "color": [r"\bwhat colou?r\b", r"\bcolor\b", r"\bred\b|\bblue\b|\bgreen\b|\byellow\b"],
        "locate": [r"\bwhere\b", r"\blocation\b", r"\bsitting\b", r"\bon top\b"],
        "relation": [r"\bleft of\b", r"\bright of\b", r"\bbehind\b", r"\bin front\b", r"\bnext to\b"],
        "read": [r"\bread\b", r"\bword\b", r"\btext\b", r"\bsign\b", r"\bprinted\b", r"\bocr\b"],
        "object": [r"\bwhat is\b", r"\bwhich object\b", r"\bobject\b"],
        "activity": [r"\bdoing\b", r"\bactivity\b", r"\bplaying\b", r"\briding\b"],
        "attribute": [r"\bshape\b", r"\bsize\b", r"\bkind\b", r"\battribute\b"],
        "existence": [r"\bis there\b", r"\bare there\b", r"\bdoes .* have\b", r"\bvisible\b"],
        "other": [r".*"],
    }

    def __init__(self, skills: Sequence[str], confidence: float = 0.85):
        self.skills = list(skills)
        self.skill_to_id = {s: i for i, s in enumerate(self.skills)}
        self.confidence = float(confidence)
        self.patterns = {
            s: [re.compile(p, re.IGNORECASE) for p in self.DEFAULT_PATTERNS.get(s, [])]
            for s in self.skills
        }

    def __call__(self, questions: Sequence[str], device: Optional[torch.device] = None) -> torch.Tensor:
        rows = []
        for q in questions:
            q = q or ""
            scores = torch.ones(len(self.skills), dtype=torch.float32) * ((1.0 - self.confidence) / max(1, len(self.skills) - 1))
            matched = None
            for skill in self.skills:
                if skill == "other":
                    continue
                if any(p.search(q) for p in self.patterns.get(skill, [])):
                    matched = skill
                    break
            if matched is None:
                matched = "other" if "other" in self.skill_to_id else self.skills[-1]
            scores[:] = (1.0 - self.confidence) / max(1, len(self.skills) - 1)
            scores[self.skill_to_id[matched]] = self.confidence
            rows.append(scores)
        probs = torch.stack(rows, dim=0)
        return probs.to(device) if device is not None else probs


class TransformerSkillParser(nn.Module):
    """Transformer text classifier matching the paper-level parser design.

    It uses a small internal word vocabulary so it can be trained without a
    specific LLM tokenizer. Defaults correspond to the paper: 4 layers,
    8 heads, hidden size 512.
    """

    def __init__(
        self,
        vocab_size: int,
        skills: Sequence[str],
        max_len: int = 64,
        hidden_size: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.skills = list(skills)
        self.max_len = int(max_len)
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, hidden_size)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=4 * hidden_size,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, len(self.skills))

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seq = input_ids.shape
        pos = torch.arange(seq, device=input_ids.device).unsqueeze(0).expand(bsz, seq)
        x = self.embedding(input_ids) + self.pos_embedding(pos.clamp_max(self.max_len - 1))
        key_padding_mask = attention_mask == 0 if attention_mask is not None else input_ids == 0
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        if attention_mask is None:
            mask = (input_ids != 0).float()
        else:
            mask = attention_mask.float()
        pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return self.classifier(self.norm(pooled))

    @torch.no_grad()
    def predict_proba(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return F.softmax(self.forward(input_ids, attention_mask), dim=-1)


class WordVocab:
    def __init__(self, min_freq: int = 1):
        self.min_freq = int(min_freq)
        self.token_to_id = {"<pad>": 0, "<unk>": 1}
        self.id_to_token = ["<pad>", "<unk>"]

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9']+|<image>", text.lower())

    def fit(self, texts: Iterable[str]) -> None:
        counts: Dict[str, int] = {}
        for text in texts:
            for tok in self.tokenize(text):
                counts[tok] = counts.get(tok, 0) + 1
        for tok, cnt in sorted(counts.items()):
            if cnt >= self.min_freq and tok not in self.token_to_id:
                self.token_to_id[tok] = len(self.id_to_token)
                self.id_to_token.append(tok)

    def encode(self, text: str, max_len: int) -> Dict[str, torch.Tensor]:
        toks = self.tokenize(text)[:max_len]
        ids = [self.token_to_id.get(t, 1) for t in toks]
        mask = [1] * len(ids)
        while len(ids) < max_len:
            ids.append(0)
            mask.append(0)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"id_to_token": self.id_to_token}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "WordVocab":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        vocab = cls()
        vocab.id_to_token = list(raw["id_to_token"])
        vocab.token_to_id = {t: i for i, t in enumerate(vocab.id_to_token)}
        return vocab
