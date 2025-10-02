# MULTI-TAP: Multi-Objective Task-Aware Predictor for Image-Text Alignment

## Authors
Eunki Kim∗, Na Min An∗, James Thorne, Hyunjung Shim  
∗ equal contribution  
Affiliations: KAIST AI (Eunki Kim, Na Min An, Hyunjung Shim)

Contacts: eunkikim@kaist.ac.kr, naminan@kaist.ac.kr

Project Page: https://kaist-cvml.github.io/multitap  
Hugging Face Org: https://huggingface.co/brl-xfact

## Abstract
Evaluating image-text alignment while reflecting human preferences across multiple aspects is a significant issue for the development of reliable vision-language applications. It becomes especially crucial in real-world scenarios where multiple valid descriptions exist depending on contexts or user needs. However, research progress is hindered by the lack of comprehensive benchmarks and existing evaluation predictors lacking at least one of these key properties: (1) Alignment with human judgments, (2) Long-sequence processing, (3) Inference efficiency, and (4) Applicability to multi-objective scoring. To address these challenges, we propose a plug-and-play architecture to build a robust predictor, MULTI-TAP (Multi-Objective Task-Aware Predictor), capable of both multi and single-objective scoring. MULTI-TAP can produce a single overall score, utilizing a reward head built on top of a large vision-language model (LVLMs). We show that MULTI-TAP is robust in terms of application to different LVLM architectures, achieving significantly higher performance than existing metrics (e.g., +42.3 Kendall’s τ_c compared to IXCREW-S on FlickrExp) and even on par with the GPT-4o-based predictor, G-VEval, with a smaller size (7–8B). By training a lightweight ridge regression layer on the frozen hidden states of a pre-trained LVLM, MULTI-TAP can produce fine-grained scores for multiple human-interpretable objectives. MULTI-TAP performs better than VisionREWARD, a high-performing multi-objective reward model, in both performance and efficiency on multi-objective benchmarks and our newly released text-image-to-text dataset, EYE4ALL. Our new dataset, consisting of chosen/rejected human preferences (EYE4ALLPref) and human-annotated fine-grained scores across seven dimensions (EYE4ALLMulti), can serve as a foundation for developing more accessible AI systems by capturing the underlying preferences of users, including blind and low-vision (BLV) individuals. Our contributions can guide future research for developing human-aligned predictors.

ArXiv: coming soon

## Checkpoints
We release MULTI-TAP checkpoints for multiple LVLM backbones (see Hugging Face org `brl-xfact`):

- brl-xfact/Multi-TAP-Qwen2-VL-2B
- brl-xfact/Multi-TAP-Qwen2-VL-7B
- brl-xfact/Multi-TAP-Llama3.2-11B-Vision
- brl-xfact/Multi-TAP-internlm-xcomposer2d5-7b

Visit the organization page for activity and assets: https://huggingface.co/brl-xfact

## Datasets
- brl-xfact/Eye4AllMulti  
- brl-xfact/Eye4AllPreference

## Installation
```bash
# Python 3.10+ recommended
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt 
```

Login to Hugging Face (optional, recommended for gated models):
```bash
huggingface-cli login
```

## Quickstart: Demo Inference
You can quickly score an image-text pair using the demo script.
```bash
python run_demo.py \
  --model_name Qwen/Qwen2-VL-2B-Instruct \
  --ckpt_path brl-xfact/Multi-TAP-Qwen2-VL-2B \
  --dataset brl-xfact/polaris_imagereward \
  --device cuda:0
```

## Training
We provide scripts for single-objective training (end-to-end reward head) and for the multi-objective pipeline (feature extraction + ridge regression).

### Single-Objective Training
Script: `scripts/train_single.sh`

This launches distributed training via `accelerate` using config `accelerate_config/ds3.yaml` and the entry `single_objective_train.py` with a sample recipe.

```bash
bash scripts/train_single.sh
```

The script expands to:
```bash
ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info \
accelerate launch \
  --config_file accelerate_config/ds3.yaml \
  single_objective_train.py \
  recipes/samples/qwen.yaml
```

- `recipes/samples/qwen.yaml` contains `ModelArguments`, `DataArguments`, and `RewardArguments` used by `single_objective_train.py` via the `H4ArgumentParser`.
- You can switch to `recipes/samples/llama.yaml` or `recipes/samples/internlm.yaml` depending on your backbone.

Entry: `single_objective_train.py`
- Parses `(ModelArguments, DataArguments, RewardArguments)` from the recipe YAML.
- Initializes `VLMRewardModel` and `AutoProcessor`.
- Trains the reward head with MSE loss against human scores, logging to Weights & Biases if configured.
- Saves periodic checkpoints and a final checkpoint under `output_dir`.

Common knobs (set in your recipe YAML):
- `model_name_or_path`: backbone model (e.g., `Qwen/Qwen2-VL-2B-Instruct`)
- `dataset_name`, `dataset_split`: HF dataset ID and splits
- `per_device_train_batch_size`, `learning_rate`, `num_train_epochs`, `max_length`
- `output_dir`, `run_name`, `report_to`/`project_name`/`entity_name`

### Multi-Objective Training + Inference
Script: `scripts/mutli_train_inference.sh`

This script extracts features from an LVLM and trains a lightweight Ridge regression for multi-objective scoring using `multi_objective_train.py`.

Edit the placeholders in the script before running:
```bash
python multi_objective_train.py --model_name Qwen/Qwen2-VL-2B-Instruct \
  --checkpoint_dir <PATH_OR_HF_ID_TO_SINGLE_OBJECTIVE_CHECKPOINT> \
  --dataset <vision_reward|text-image-to-text|text-to-image> \
  --save_dir <OUTPUT_FEATURE_DIR> \
  --device cuda:0
```

Arguments for `multi_objective_train.py`:
- `--model_name`: backbone (e.g., `Qwen/Qwen2-VL-2B-Instruct`)
- `--checkpoint_dir`: path or HF repo of a trained single-objective checkpoint (weights loaded into reward head)
- `--dataset`: one of `vision_reward`, `text-image-to-text`, `text-to-image`
- `--save_dir`: where intermediate feature chunks are written
- `--device`: device string (e.g., `cuda:0` or `cpu`)

The pipeline:
1. Load reward model and processor, put the model into eval BF16.
2. Process dataset in chunks, save hidden-state features to `save_dir`.
3. Train Ridge regression with cross-validated alpha on train split.
4. Evaluate on test split and report MSE and per-dimension accuracy.

## Project Structure
- `models/reward_model.py`: Implementation of `VLMRewardModel` and config
- `modules/`: argument dataclasses, utils, chat templates, data processing
- `single_objective_train.py`: end-to-end training for reward head
- `multi_objective_train.py`: feature extraction + ridge regression pipeline
- `run_demo.py`: quick demo script for single-objective rewards
- `recipes/samples/*.yaml`: sample configs for different backbones
- `scripts/`: convenience scripts

## Acknowledgements
We thank the maintainers and contributors of Hugging Face Transformers, Accelerate, and associated LVLM backbones.

## Citation
ArXiv preprint coming soon.

```bibtex
@misc{kim2025multiobjectivetaskawarepredictorimagetext,
      title={Multi-Objective Task-Aware Predictor for Image-Text Alignment}, 
      author={Eunki Kim and Na Min An and James Thorne and Hyunjung Shim},
      year={2025},
      eprint={2510.00766},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.00766}, 
}
```

## License
This repository is released for research purposes. See LICENSE (if provided) for terms.
