#!/usr/bin/env bash
set -euo pipefail
# Edit these paths to point to the official VQA v2 annotations.
python -m attention_spectrum_replay.prepare_vqav2 \
  --train-questions data/vqa/v2_OpenEnded_mscoco_train2014_questions.json \
  --train-annotations data/vqa/v2_mscoco_train2014_annotations.json \
  --val-questions data/vqa/v2_OpenEnded_mscoco_val2014_questions.json \
  --val-annotations data/vqa/v2_mscoco_val2014_annotations.json \
  --output-dir data/vqav2_manifests \
  --top-answers 3129
