# 🌊 Attention-Spectrum Replay

Attention-Spectrum Replay (ASR) is a continual learning framework for multimodal language models. It regularizes cross-modal attention in the frequency domain, preserving task-relevant image-text alignment without storing raw examples from previous tasks.

## ✨ Highlights

- 🧠 **Data-free replay for multimodal continual learning**: preserves previous skills through compact spectral prototypes instead of image/question replay.
- 📡 **Frequency-domain attention descriptors**: extracts radial and angular FFT statistics from image-token cross-attention maps.
- 🗂️ **Skill-conditioned memory**: maintains per-skill Gaussian prototypes and angular distributions with EMA updates.
- 🧭 **ASR regularization objective**: combines Mahalanobis spectral matching, angular KL alignment, confidence-adaptive weighting, and geometry preservation.

## 🛠️ Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For Hugging Face model adapters and LoRA-based MLLM runs:

```bash
pip install -e '.[hf]'
```

## 🚀 Quickstart: COCO-QA Continual VQA

Prepare a real-image VQA continual learning dataset from Hugging Face:

```bash
python scripts/prepare_hf_cocoqa_subset.py \
  --output-root data/hf_cocoqa_scaled \
  --train-per-stage 128 \
  --val-per-stage 32 \
  --image-size 256
```

This creates four continual stages from `ThucPD/coco-qa-vi`:

```text
data/hf_cocoqa_scaled/
  images/
  manifests/
    answer_vocab.json
    stage_00_train.jsonl
    stage_00_val.jsonl
    stage_01_train.jsonl
    stage_01_val.jsonl
    ...
```

Train ASR on the prepared dataset:

```bash
python -m attention_spectrum_replay.train --config configs/hf_cocoqa_scaled.yaml
```

Outputs are written to the configured run directory:

```text
runs/hf_cocoqa_scaled/
  resolved_config.yaml
  metrics.json
  memory.pt
  checkpoint.pt
  stage_00.pt
  stage_00_prototype_counts.json
  ...
```

## ⚙️ Training Configurations

The project uses YAML experiment files under `configs/`.

- 🖼️ `hf_cocoqa_scaled.yaml`: real COCO-QA continual VQA run for local development.
- ⚡ `hf_cocoqa_tiny.yaml`: smaller real COCO-QA run for quick iteration.
- 🧪 `local_synthetic.yaml`: synthetic multimodal stream for CPU-only development.
- 🦙 `vqav2_llava15_7b.yaml`: VQA v2 + LLaVA-style large-model configuration.
- 📊 `vqacl_skill_concept.yaml`, `coin.yaml`, `ucit.yaml`: benchmark-oriented configuration templates.

Each config is divided into `spectral`, `asr`, `optim`, `model`, `data`, and `runtime` sections. The resolved configuration is saved with every run for traceability.

## 🧾 Dataset Format

The generic VQA loader expects one train and validation JSONL file per continual stage:

```text
stage_00_train.jsonl
stage_00_val.jsonl
stage_01_train.jsonl
stage_01_val.jsonl
...
```

Each row can use an integer class label:

```json
{
  "image": "relative/or/absolute/path.jpg",
  "question": "USER: <image> QUESTION: What color is the car? ASSISTANT:",
  "label": 17,
  "skill": "color",
  "stage": 0
}
```

Or a string answer with an answer vocabulary:

```json
{
  "image": "000123.jpg",
  "question": "USER: <image> QUESTION: How many people are visible? ASSISTANT:",
  "answer": "2",
  "skill": "count",
  "stage": 1
}
```

When using string answers, set `data.answer_vocab` to a JSON dictionary mapping answer strings to integer IDs. If `skill_probs` are provided, they are used directly; otherwise the loader derives skill probabilities from `skill` or the heuristic parser.

## 🦙 VQA v2 With LLaVA

For a larger VQA v2 run:

1. Download the official VQA v2 annotations and COCO images.
2. Build continual manifests:

```bash
bash scripts/run_vqav2_prepare.sh
```

3. Edit `configs/vqav2_llava15_7b.yaml`:

```yaml
data:
  manifest_dir: data/vqav2_manifests
  image_root: data/vqav2_images
  answer_vocab: data/vqav2_manifests/answer_vocab.json
model:
  hf_model_name_or_path: liuhaotian/llava-v1.5-7b
```

4. Start training:

```bash
bash scripts/run_vqav2_asr.sh
```

Large-model benchmark results depend on hardware, precision, checkpoint family, preprocessing templates, and how the selected MLLM exposes cross-modal attention maps.

## 🧩 Skill Parser

ASR can consume explicit `skill_probs` from the dataset, derive near-one-hot skill probabilities from a `skill` field, or use the included parser fallback.

To train the Transformer skill parser:

```bash
python -m attention_spectrum_replay.train_skill_parser \
  --train-jsonl data/skill_train.jsonl \
  --val-jsonl data/skill_val.jsonl \
  --skills count,color,locate,relation,read,object,activity,attribute,existence,other \
  --output-dir runs/skill_parser \
  --epochs 10 \
  --lr 1e-4
```

