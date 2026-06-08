from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .utils import ensure_dir, save_json


DEFAULT_TASK_ORDER = [
    "count", "color", "locate", "comparison", "attribute", "activity", "existence", "read", "object", "other"
]


def image_name(split: str, image_id: int) -> str:
    prefix = "COCO_train2014" if "train" in split.lower() else "COCO_val2014"
    return f"{prefix}_{image_id:012d}.jpg"


def normalize_answer(ans: str) -> str:
    return str(ans).strip().lower()


def map_question_type(question: str, question_type: str) -> str:
    q = (question or "").lower()
    qt = (question_type or "").lower()
    blob = f"{qt} {q}"
    if re.search(r"\bhow many\b|\bnumber of\b", blob):
        return "count"
    if "color" in blob or "colour" in blob:
        return "color"
    if re.search(r"\bwhere\b|\blocation\b|\bplace\b|\bsitting\b|\bstanding\b", blob):
        return "locate"
    if re.search(r"\bmore\b|\bless\b|\bsame\b|\bdifferent\b|\bcompare\b|\btaller\b|\bsmaller\b", blob):
        return "comparison"
    if re.search(r"\bshape\b|\bsize\b|\bkind\b|\btype\b|\battribute\b", blob):
        return "attribute"
    if re.search(r"\bdoing\b|\bplaying\b|\briding\b|\bactivity\b", blob):
        return "activity"
    if re.search(r"\bis there\b|\bare there\b|\bdoes\b|\bdo you see\b|\bvisible\b", blob):
        return "existence"
    if re.search(r"\bread\b|\bword\b|\btext\b|\bsign\b|\bprinted\b|\bletter\b", blob):
        return "read"
    if re.search(r"\bwhat is\b|\bwhat are\b|\bwho\b|\bobject\b|\banimal\b", blob):
        return "object"
    return "other"


def load_vqa(questions_json: Path, annotations_json: Path) -> List[Dict[str, object]]:
    qs = json.loads(questions_json.read_text(encoding="utf-8"))["questions"]
    anns = json.loads(annotations_json.read_text(encoding="utf-8"))["annotations"]
    q_by_id = {int(q["question_id"]): q for q in qs}
    records = []
    for ann in anns:
        q = q_by_id[int(ann["question_id"])]
        answer = normalize_answer(ann.get("multiple_choice_answer", ann.get("answer", "")))
        records.append(
            {
                "question_id": int(ann["question_id"]),
                "image_id": int(ann["image_id"]),
                "question": q["question"],
                "answer": answer,
                "question_type": ann.get("question_type", ""),
            }
        )
    return records


def build_answer_vocab(train_records: List[Dict[str, object]], top_k: int) -> Dict[str, int]:
    counts = Counter(r["answer"] for r in train_records)
    answers = [a for a, _ in counts.most_common(top_k)]
    return {a: i for i, a in enumerate(answers)}


def write_split(
    records: List[Dict[str, object]],
    out_dir: Path,
    split: str,
    answer_vocab: Dict[str, int],
    task_order: List[str],
    image_subdir: str,
) -> None:
    by_task: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for rec in records:
        task = map_question_type(str(rec["question"]), str(rec.get("question_type", "")))
        if rec["answer"] not in answer_vocab:
            continue
        by_task[task].append(rec)
    for stage, task in enumerate(task_order):
        path = out_dir / f"stage_{stage:02d}_{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for rec in by_task.get(task, []):
                row = {
                    "image": str(Path(image_subdir) / image_name(split, int(rec["image_id"]))),
                    "question": f"USER: <image> QUESTION: {rec['question']} ASSISTANT:",
                    "answer": rec["answer"],
                    "label": int(answer_vocab[rec["answer"]]),
                    "skill": task,
                    "stage": stage,
                    "question_id": rec["question_id"],
                    "image_id": rec["image_id"],
                    "question_type": rec.get("question_type", ""),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare VQA v2 question-type incremental JSONL manifests")
    parser.add_argument("--train-questions", required=True)
    parser.add_argument("--train-annotations", required=True)
    parser.add_argument("--val-questions", required=True)
    parser.add_argument("--val-annotations", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-answers", type=int, default=3129)
    parser.add_argument("--task-order", default=",".join(DEFAULT_TASK_ORDER))
    parser.add_argument("--train-image-subdir", default="train2014")
    parser.add_argument("--val-image-subdir", default="val2014")
    args = parser.parse_args()
    out = ensure_dir(args.output_dir)
    order = [x.strip() for x in args.task_order.split(",") if x.strip()]
    train = load_vqa(Path(args.train_questions), Path(args.train_annotations))
    val = load_vqa(Path(args.val_questions), Path(args.val_annotations))
    vocab = build_answer_vocab(train, args.top_answers)
    save_json(vocab, out / "answer_vocab.json")
    write_split(train, out, "train", vocab, order, args.train_image_subdir)
    write_split(val, out, "val", vocab, order, args.val_image_subdir)
    save_json({"task_order": order, "num_answers": len(vocab)}, out / "manifest_meta.json")
    print(f"Wrote manifests to {out}")


if __name__ == "__main__":
    main()
