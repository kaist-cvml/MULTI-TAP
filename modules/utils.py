import os
import torch
import random
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import Any, List, Literal, Dict, Callable
from torch.utils.data.dataloader import default_collate
import datasets
import json
from PIL import Image
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset

import random
from collections import defaultdict
import io

from safetensors.torch import load_model
from safetensors import safe_open
import json
from torchvision import transforms
from torchvision.transforms.functional import to_tensor as totensor
from PIL.Image import Image as PILImage

try:
    from Github.LLaVA.llava import LlavaLlamaForCausalLM
    from Github.LLaVA.llava.conversation import conv_templates
    from Github.LLaVA.llava.mm_utils import tokenizer_image_token
    from Github.LLaVA.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
except:
    pass


LLAVA_CHAT_TEMPLATE = "{% for message in messages %}{% if message['role'] != 'system' %}{{ message['role'].upper() + ': '}}{% endif %}{# Render all images first #}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>\n' }}{% endfor %}{# Render all text next #}{% if message['role'] != 'assistant' %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ content['text'] + ' '}}{% endfor %}{% else %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{% generation %}{{ content['text'] + ' '}}{% endgeneration %}{% endfor %}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"

QWEN_CHAT_TEMPLATE = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"

INTERNLM_CHAT_TEMPLATE = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"


def process_data(batch, processor, training_args, data_args, model_type, tokenizer=None):
    
    if isinstance(batch["img"], str):
        assert os.path.isfile(batch["img"])
        image = Image.open(batch["img"]).convert("RGB").resize((560, 560))
    elif isinstance(batch["img"], dict):
        image = Image.open(io.BytesIO(batch["img"]['bytes'])).convert("RGB").resize((560, 560))
    elif isinstance(batch["img"], list):
        batch['img'] = torch.Tensor(batch['img'])            
        # (n,560, 560) -> (3, 560, 560)
        if batch["img"].shape[0] == 1:
            image = Image.fromarray((batch["img"].repeat(3, 1, 1).numpy()*255).astype(np.uint8).transpose(1, 2, 0)).convert("RGB").resize((560, 560))
        elif batch["img"].shape[0] == 2:
            image = Image.fromarray((batch["img"][:1].repeat(3, 1, 1).numpy()*255).astype(np.uint8).transpose(1, 2, 0)).convert("RGB").resize((560, 560))
        elif batch["img"].shape[0] > 3:
            image = Image.fromarray((batch["img"][:3].numpy()*255).astype(np.uint8).transpose(1, 2, 0)).convert("RGB").resize((560, 560))
        else:
            image = Image.fromarray((batch["img"].numpy()*255).astype(np.uint8).transpose(1, 2, 0)).convert("RGB").resize((560, 560))
    else:
        image = batch['img']

    if 'internlm' in model_type.lower():
            
        return {
            "samples" : {
                "text_input": "Describe this image. <ImageHere>;" + batch["cand"],
                # "text_input": "<ImageHere>;" + batch["cand"],
                "image": totensor(image.convert("RGB").resize((560,560))).unsqueeze(0).bfloat16()
            },
            "human_score": torch.tensor(batch["human_score"])
        }
    
    else:                       
        prompt = processor.apply_chat_template(eval("[{'role': 'user','content': [{'type': 'text', 'text': 'Describe this image.'}]},{'role': 'assistant','content': [{'type': 'image'},{'type': 'text', 'text' : \"" + batch["cand"].strip().replace('"', " ") + "\"}],}]"), tokenize=False)
        # prompt = processor.apply_chat_template(eval("[{'role': 'user','content': [{'type': 'image'},{'type': 'text', 'text' : \"" + batch["cand"].strip().replace('"', " ") + "\"}],}]"), tokenize=False)
        
        inputs = processor(text = prompt, images = image, return_tensors="pt", padding="max_length", truncation=True, max_length=training_args.max_length)
        
        return_processed_batch = {
            **inputs,
        }
        
        if "human_score" in batch:
            temp = torch.tensor(batch["human_score"])
            if temp.dim() >= 1:
                for i in range(temp.dim()):
                    temp = temp.squeeze(0)
            return_processed_batch["human_score"] = temp
            
            
        return return_processed_batch
        

def balance_dataset_by_score(dataset, score_key="human_score", seed=42):
    """
    Balances the Hugging Face Dataset by the number of samples for each unique human score.

    Args:
        dataset: Hugging Face Dataset object containing `score_key`.
        score_key: The column name in the dataset representing the human score.
        seed: Random seed for reproducibility.

    Returns:
        A balanced Hugging Face Dataset object.
    """

    # Convert to a pandas DataFrame for easier grouping and manipulation
    df = dataset.to_pandas()

    # Group by the score_key and find the minimum group size
    grouped = df.groupby(score_key)
    min_count = grouped.size().min()

    # Sample `min_count` rows randomly from each group
    balanced_df = grouped.apply(lambda x: x.sample(n=min_count, random_state=seed)).reset_index(drop=True)

    # Convert back to a Hugging Face Dataset
    balanced_dataset = HFDataset.from_pandas(balanced_df)

    return balanced_dataset

class IdentityDataCollator:
    def __call__(self, batch):
        batch_dict = {key: default_collate([item[key] for item in batch]) for key in batch[0]}
        return batch_dict

class BTDataset(Dataset):
    def __init__(self, data_path: str):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        super().__init__()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def map(self, function: Callable, batched: bool = False, num_proc: int = 1, desc: str = "", fn_kwargs: Dict[str, Any] = {}, writer_batch_size: int = 1000):
        
        new_data = []
        for idx in range(len(self)):
            item = self[idx]
            processed_item = function(item, **fn_kwargs)  # Pass additional kwargs to the function
            new_data.append(processed_item)
        self.data = new_data
        return self
        
def initialize_reward_model_head(model: AutoModel):
    
    hidden_size = getattr(
        model.config, 
        "hidden_size", 
        getattr(
            model.config, 
            "d_model", 
            getattr(
                model.config, 
                "max_position_embeddings", 
                4096
            )
        )
    )
    
    print(">>> Classification head initialized to with normal distribution.: ", model.score.weight.size())
    nn.init.normal_(model.score.weight, mean=0.0, std=1/np.sqrt(hidden_size+1))
    # print(">>> Classification head initialized to with normal distribution.: ", model.score.weight.size())
    # nn.init.normal_(model.score.weight, mean=0.0, std=1/np.sqrt(hidden_size+1))

    return model

def load_model_with_index(model, checkpoint_dir):
    """Load model weights from checkpoint directory or HuggingFace Hub"""
    
    # Check if checkpoint_dir is a HuggingFace Hub model ID
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory not found locally: {checkpoint_dir}")
        print("Attempting to load from HuggingFace Hub...")
        
        try:
            from huggingface_hub import snapshot_download
            import tempfile
            
            # Download the model files from HuggingFace Hub
            with tempfile.TemporaryDirectory() as temp_dir:
                hf_token = os.getenv("HF_TOKEN")
                if hf_token:
                    downloaded_path = snapshot_download(
                        repo_id=checkpoint_dir,
                        cache_dir=temp_dir,
                        token=hf_token
                    )
                else:
                    print("Warning: HF_TOKEN not found. Downloading without authentication.")
                    downloaded_path = snapshot_download(
                        repo_id=checkpoint_dir,
                        cache_dir=temp_dir
                    )
                
                # Recursively call with the downloaded path
                return load_model_with_index(model, downloaded_path)
                
        except Exception as e:
            print(f"Failed to download from HuggingFace Hub: {e}")
            raise FileNotFoundError(f"Could not load model from {checkpoint_dir}")
    
    # Local loading logic
    index_file = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    if not os.path.exists(index_file):
        if os.path.exists(os.path.join(checkpoint_dir, "model.safetensors")):
            print("Loading from single safetensors file...")
            return load_model(model, os.path.join(checkpoint_dir, "model.safetensors"))
        raise FileNotFoundError(f"Index file not found: {index_file}")
    
    # Load the index file
    with open(index_file, "r") as f:
        index_data = json.load(f)
    
    weight_map = index_data.get("weight_map", {})
    if not weight_map:
        raise ValueError("Invalid index file: missing weight_map.")
    
    # Load tensors from respective safetensor files
    state_dict = {}
    for weight_name, file_name in weight_map.items():
        file_path = os.path.join(checkpoint_dir, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")
        
        with safe_open(file_path, framework="pt") as f:
            if weight_name in f.keys():
                state_dict[weight_name] = f.get_tensor(weight_name)
    
    print(f"Loading {len(state_dict)} weight tensors...")
    model.load_state_dict(state_dict, strict=False)

class ImageSplittingCollator:
    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def __call__(self, batch):
        full_batch = {}
        image_free_batch = {}

        for key in batch[0]:
            values = [item[key] for item in batch]

            if isinstance(values[0], PILImage):
                # Keep original images in full_batch
                full_batch[key] = values
                # Exclude images from image_free_batch
                continue

            try:
                collated = default_collate(values)
            except TypeError:
                collated = values  # fallback

            full_batch[key] = collated
            image_free_batch[key] = collated

        return full_batch, image_free_batch