#!/bin/bash

export HF_HUB_ENABLE_HF_TRANSFER=1


ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch \
    --config_file accelerate_config/ds3.yaml \
    single_objective_train.py \
    recipes/samples/qwen.yaml