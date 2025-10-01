#!/bin/bash

set -euo pipefail

export HF_HUB_ENABLE_HF_TRANSFER=1

python multi_objective_train.py --model_name Qwen/Qwen2-VL-2B-Instruct \
                    --checkpoint_dir  \
                    --dataset  \
                    --save_dir  \
                    --device cuda:0 \