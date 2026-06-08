#!/usr/bin/env bash
set -euo pipefail
python -m attention_spectrum_replay.train --config configs/vqav2_llava15_7b.yaml
