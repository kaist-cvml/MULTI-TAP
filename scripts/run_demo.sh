#!/bin/bash

set -euo pipefail

export HF_HUB_ENABLE_HF_TRANSFER=1

python run_demo.py \
  --model brl-xfact/Multi-TAP-Qwen2-VL-2B \
  --dataset brl-xfact/polaris_imagereward \
  --device cuda



