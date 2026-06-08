from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from .skill_parser import TransformerSkillParser, WordVocab
from .utils import ensure_dir, set_seed


class SkillJsonlDataset(Dataset):
    def __init__(self, path: str | Path, vocab: WordVocab, skills: List[str], max_len: int):
        self.records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        self.vocab = vocab
        self.skills = skills
        self.skill_to_id = {s: i for i, s in enumerate(skills)}
        self.max_len = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        text = r.get("question", r.get("text", ""))
        skill = r.get("skill")
        if skill not in self.skill_to_id:
            raise ValueError(f"Unknown skill {skill}")
        enc = self.vocab.encode(text, self.max_len)
        return enc["input_ids"], enc["attention_mask"], torch.tensor(self.skill_to_id[skill], dtype=torch.long)


def collate(items):
    return {
        "input_ids": torch.stack([x[0] for x in items]),
        "attention_mask": torch.stack([x[1] for x in items]),
        "labels": torch.stack([x[2] for x in items]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ASR Transformer skill parser")
    parser.add_argument("--train-jsonl", required=True, help="JSONL with fields question/text and skill")
    parser.add_argument("--val-jsonl", default=None)
    parser.add_argument("--skills", required=True, help="Comma-separated skill names")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    set_seed(args.seed)
    out = ensure_dir(args.output_dir)
    skills = [s.strip() for s in args.skills.split(",") if s.strip()]
    train_records = [json.loads(line) for line in Path(args.train_jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]
    vocab = WordVocab(min_freq=1)
    vocab.fit([r.get("question", r.get("text", "")) for r in train_records])
    vocab.save(out / "skill_vocab.json")
    train_ds = SkillJsonlDataset(args.train_jsonl, vocab, skills, args.max_len)
    val_ds = SkillJsonlDataset(args.val_jsonl, vocab, skills, args.max_len) if args.val_jsonl else None
    model = TransformerSkillParser(
        vocab_size=len(vocab.id_to_token),
        skills=skills,
        max_len=args.max_len,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    ).to(args.device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for batch in train_loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = F.cross_entropy(logits, batch["labels"])
            loss.backward()
            opt.step()
            total_loss += float(loss.detach().cpu()) * batch["labels"].numel()
            pred = logits.argmax(-1)
            correct += int((pred == batch["labels"]).sum().item())
            total += int(batch["labels"].numel())
        print(json.dumps({"epoch": epoch + 1, "train_loss": total_loss / max(1, total), "train_acc": correct / max(1, total)}))
    torch.save({"model": model.state_dict(), "skills": skills, "vocab_size": len(vocab.id_to_token)}, out / "skill_parser.pt")


if __name__ == "__main__":
    main()
