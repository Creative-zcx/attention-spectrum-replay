#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

os.environ.setdefault("HF_HOME", str(Path("data/hf_cache").resolve()))
os.environ.setdefault("HF_DATASETS_CACHE", str(Path("data/hf_cache/datasets").resolve()))
os.environ.setdefault("USE_NUMEXPR", "0")

from datasets import load_dataset


TYPE_TO_STAGE = {
    0: ("object", "object"),
    1: ("count", "count"),
    2: ("color", "color"),
    3: ("locate", "locate"),
}


def _prompt(question: str) -> str:
    return f"USER: <image> QUESTION: {question.strip()} ASSISTANT:"


def _write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a tiny real VQA subset from ThucPD/coco-qa-vi.")
    parser.add_argument("--output-root", default="data/hf_cocoqa_tiny")
    parser.add_argument("--train-per-stage", type=int, default=16)
    parser.add_argument("--val-per-stage", type=int, default=8)
    parser.add_argument("--max-scan", type=int, default=20000)
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    manifest_dir = output_root / "manifests"
    image_root = output_root / "images"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)

    quotas = {
        stage: {"train": args.train_per_stage, "val": args.val_per_stage}
        for stage in TYPE_TO_STAGE
    }
    rows: Dict[int, Dict[str, List[Dict[str, object]]]] = {
        stage: {"train": [], "val": []}
        for stage in TYPE_TO_STAGE
    }
    answer_vocab: Dict[str, int] = {}

    ds = load_dataset("ThucPD/coco-qa-vi", split="train", streaming=True)
    scanned = 0
    for sample in ds:
        scanned += 1
        if scanned > args.max_scan:
            break
        stage_type = int(sample.get("type", -1))
        if stage_type not in TYPE_TO_STAGE:
            continue
        stage = stage_type
        split = "train" if len(rows[stage]["train"]) < quotas[stage]["train"] else "val"
        if len(rows[stage][split]) >= quotas[stage][split]:
            continue

        question = str(sample.get("question", "")).strip()
        answer = str(sample.get("answer", "")).strip().lower()
        image = sample["image"].convert("RGB")
        image.thumbnail((args.image_size, args.image_size))

        sample_idx = len(rows[stage]["train"]) + len(rows[stage]["val"])
        image_name = f"stage_{stage:02d}_{sample_idx:04d}.jpg"
        image.save(image_root / image_name, quality=90)

        if answer not in answer_vocab:
            answer_vocab[answer] = len(answer_vocab)
        skill, _ = TYPE_TO_STAGE[stage]
        rows[stage][split].append(
            {
                "image": image_name,
                "question": _prompt(question),
                "answer": answer,
                "skill": skill,
                "stage": stage,
                "source_dataset": "ThucPD/coco-qa-vi",
                "source_image_id": str(sample.get("image_id", "")),
                "source_question_id": int(sample.get("question_id", -1)),
            }
        )

        if all(len(rows[s]["train"]) >= quotas[s]["train"] and len(rows[s]["val"]) >= quotas[s]["val"] for s in rows):
            break

    missing = {
        TYPE_TO_STAGE[s][0]: {
            "train": quotas[s]["train"] - len(rows[s]["train"]),
            "val": quotas[s]["val"] - len(rows[s]["val"]),
        }
        for s in rows
        if len(rows[s]["train"]) < quotas[s]["train"] or len(rows[s]["val"]) < quotas[s]["val"]
    }
    if missing:
        raise RuntimeError(f"Could not fill requested quotas after scanning {scanned} samples: {missing}")

    for stage in sorted(rows):
        _write_jsonl(manifest_dir / f"stage_{stage:02d}_train.jsonl", rows[stage]["train"])
        _write_jsonl(manifest_dir / f"stage_{stage:02d}_val.jsonl", rows[stage]["val"])

    (manifest_dir / "answer_vocab.json").write_text(
        json.dumps(answer_vocab, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary = {
        "source_dataset": "ThucPD/coco-qa-vi",
        "scanned": scanned,
        "num_answers": len(answer_vocab),
        "stages": {
            str(stage): {
                "skill": TYPE_TO_STAGE[stage][0],
                "train": len(rows[stage]["train"]),
                "val": len(rows[stage]["val"]),
            }
            for stage in sorted(rows)
        },
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
