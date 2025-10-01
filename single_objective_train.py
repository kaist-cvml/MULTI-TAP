import os
import logging
import sys
import datetime
from typing import Any, Dict, List
import io
from io import BytesIO
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, AutoTokenizer, set_seed
from datasets import load_dataset
from models.reward_model import VLMRewardConfig, VLMRewardModel
from accelerate import Accelerator
from accelerate import InitProcessGroupKwargs
from datetime import timedelta
from tqdm import tqdm

from modules import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    RewardArguments,
    LLAVA_CHAT_TEMPLATE,
    QWEN_CHAT_TEMPLATE,
    INTERNLM_CHAT_TEMPLATE,
    process_data,
    balance_dataset_by_score,
    ImageSplittingCollator,
    BTDataset,
)
from liger_kernel.transformers import apply_liger_kernel_to_qwen2
import wandb
from torchvision.transforms.functional import to_tensor as totensor

def main(model_args, data_args, training_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)
    
    # Initialize accelerator
    handler = InitProcessGroupKwargs(timeout = timedelta(seconds=10800))
    accelerator = Accelerator(kwargs_handlers=[handler])
    
    # Initialize logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO if torch.distributed.get_rank() == 0 else logging.ERROR)

    # Load model and processor
    config = VLMRewardConfig.from_pretrained(model_args.model_name_or_path)
    model = VLMRewardModel(model_args, config)
    
    if 'qwen' in model_args.model_name_or_path.lower():
        apply_liger_kernel_to_qwen2()
    
    nn.init.normal_(model.reward_head.weight, mean=0.0, std=1/np.sqrt(model.hidden_size+1))
    model.reward_head.requires_grad_(True)
    
    # Training loop
    model.train()
    
    # reward head train only
    
    # for param in model.parameters():
    #     param.requires_grad = False
    # model.reward_head.requires_grad_(True)
    
    if accelerator.is_main_process:
        print("Model loaded and ready for training.")
    
    if 'internlm' in model_args.model_name_or_path.lower():
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True,
        )
        tokenizer = None
        # InternLM processors often use eos as pad
        processor.pad_token = getattr(processor, "eos_token", None)
        if getattr(processor, "eos_token_id", None) is not None:
            model.config.pad_token_id = processor.eos_token_id
    else:
        processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
        if getattr(processor, "tokenizer", None) is not None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
            model.config.pad_token_id = processor.tokenizer.eos_token_id
        tokenizer = None
    
    if processor is None:
        if "qwen" in model_args.model_name_or_path.lower():
            processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
        elif "llama" in model_args.model_name_or_path.lower():
            processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct")
        elif "internlm" in model_args.model_name_or_path.lower():
            processor = AutoProcessor.from_pretrained("internlm/internlm-xcomposer2d5-7b")
        else:
            pass

    if getattr(processor, "chat_template", None) is None:
        if "qwen" in model_args.model_name_or_path.lower():
            processor.chat_template = QWEN_CHAT_TEMPLATE
        elif "llava" in model_args.model_name_or_path.lower():
            processor.chat_template = LLAVA_CHAT_TEMPLATE
        elif "internlm" in model_args.model_name_or_path.lower():
            processor.chat_template = INTERNLM_CHAT_TEMPLATE
        else:
            pass
    
    # Load dataset
    print("Loading dataset...")
    
    train_dataset = None
    val_dataset = None
    test_dataset = None

    def load_split(split_name: str):
        if data_args.dataset_name and data_args.dataset_name.endswith(".json"):
            if split_name == "train":
                return BTDataset(data_args.dataset_name)
            else:
                return None
        return load_dataset(data_args.dataset_name, split=split_name)

    if 'train' in data_args.dataset_split:
        train_dataset = load_split("train")
    if 'validation' in data_args.dataset_split:
        val_dataset = load_split("validation")
    if 'test' in data_args.dataset_split:
        test_dataset = load_split("test")

    accelerator.wait_for_everyone()

    # DataLoader for distributed training
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=training_args.per_device_train_batch_size, 
        collate_fn=ImageSplittingCollator(),
        shuffle=True,
    )
    
    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=None,
        )
    
    if test_dataset is not None:
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=None,
        )
        
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, training_args.num_train_epochs)
    
    model, optimizer, scheduler, train_dataloader = accelerator.prepare(
        model, optimizer, scheduler, train_dataloader
    )
    if val_dataset is not None:
        val_dataloader = accelerator.prepare(val_dataloader)
    if test_dataset is not None:
        test_dataloader = accelerator.prepare(test_dataloader)
    
    num_training_steps = len(train_dataloader) * training_args.num_train_epochs
    global_step = 0
    
    run_name = training_args.run_name
    if training_args.output_dir is not None:
        training_args.output_dir = os.path.join(training_args.output_dir, model_args.model_name_or_path.split("/")[-1] + "-" + run_name)
        os.makedirs(training_args.output_dir, exist_ok=True)
    
    if training_args.report_to is not None and accelerator.is_main_process:
            wandb.init(project=training_args.project_name, entity=training_args.entity_name, name=run_name, config={
                'model_name': model_args.model_name_or_path,
                'dataset_split': data_args.dataset_split,
                'learning_rate': training_args.learning_rate,
                'num_train_epochs': training_args.num_train_epochs,
                'per_device_train_batch_size': training_args.per_device_train_batch_size,
                'per_device_eval_batch_size': training_args.per_device_eval_batch_size,
                'seed': training_args.seed,
            })
    
    for epoch in range(training_args.num_train_epochs):
        with accelerator.accumulate(model):
            for batch, _ in tqdm(train_dataloader, desc=f"Epoch {epoch+1}", disable = not accelerator.is_local_main_process):                
                if isinstance(batch["img"], str):
                    assert os.path.isfile(batch["img"])
                    image = Image.open(batch["img"]).convert("RGB").resize((560, 560))
                elif isinstance(batch["img"], dict):
                    image = Image.open(io.BytesIO(batch["img"]['bytes'])).convert("RGB").resize((560, 560))

                else:
                    image = batch['img']
                    
                if 'internlm' in model_args.model_name_or_path.lower():
                    inputs = {
                        "samples": {
                            "text_input": "<ImageHere>;" + batch["cand"],
                            "image": totensor(image.convert("RGB").resize((560,560))).unsqueeze(0).bfloat16()
                        },
                        "human_score": torch.tensor(batch["human_score"])
                    }
                
                else:
                    prompt = processor.apply_chat_template(eval("[{'role': 'user','content': [{'type': 'text', 'text': 'Describe this image.'}]},{'role': 'assistant','content': [{'type': 'image'},{'type': 'text', 'text' : \"" + batch["cand"][0].strip().replace('"', " ") + "\"}],}]"), tokenize=False)
                    
                    inputs = processor(text = prompt, images = image[0], return_tensors="pt", padding="max_length", truncation=True, max_length=training_args.max_length)
                    
                                    
                scores = model(**inputs.to(model.device)).rewards
                
                batch['human_score'] = torch.tensor(batch['human_score'][0]).to(torch.bfloat16).to(scores.device)
                
                loss = F.mse_loss(scores, batch['human_score'].to(torch.bfloat16).to(scores.device)) / training_args.gradient_accumulation_steps
                
                # Backward pass
                accelerator.backward(loss)
                if global_step % training_args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                global_step += 1
                if accelerator.is_local_main_process and global_step % 10 == 0:
                    logger.info(f"Step {global_step}/{num_training_steps}, Loss: {loss.item()}")
                if training_args.report_to is not None and accelerator.is_local_main_process:
                        wandb.log({"Loss": loss.item(),
                                    "Learning Rate": scheduler.get_last_lr()[0],
                                    "Global Step": global_step})
                        
                # save model checkpoint every 1000 steps
                
                if global_step % 10000 == 0:
                    unwrapped_model = accelerator.unwrap_model(model)
                    if training_args.output_dir is not None:                        
                        if val_dataset is not None:
                            model.eval()
                            with torch.no_grad():
                                for val_batch in tqdm(val_dataloader, desc="Validation", disable = not accelerator.is_local_main_process):
                                    if isinstance(val_batch["img"], str):
                                        assert os.path.isfile(val_batch["img"])
                                        image = Image.open(val_batch["img"]).convert("RGB").resize((560, 560))
                                    elif isinstance(val_batch["img"], dict):
                                        image = Image.open(io.BytesIO(val_batch["img"]['bytes'])).convert("RGB").resize((560, 560))
                                    else:
                                        image = val_batch['img']
                                        
                                    if 'internlm' in model_args.model_name_or_path.lower():
                                        inputs = {
                                            "samples": {
                                                "text_input": "<ImageHere>;" + val_batch["cand"],
                                                "image": totensor(image.convert("RGB").resize((560,560))).unsqueeze(0).bfloat16()
                                            },
                                            "human_score": torch.tensor(val_batch["human_score"])
                                        }
                                    else:
                                        prompt = processor.apply_chat_template(eval("[{'role': 'user','content': [{'type': 'text', 'text': 'Describe this image.'}]},{'role': 'assistant','content': [{'type': 'image'},{'type': 'text', 'text' : \"" + val_batch["cand"][0].strip().replace('"', " ") + "\"}],}]"), tokenize=False)
                                        
                                        inputs = processor(text = prompt, images = image[0], return_tensors="pt", padding="max_length", truncation=True, max_length=training_args.max_length)
                                    scores = model(**inputs.to(model.device)).rewards
                                    
                                    val_batch['human_score'] = torch.tensor(val_batch['human_score'][0]).to(torch.bfloat16).to(scores.device)
                                    loss = F.mse_loss(scores, val_batch['human_score'].to(torch.bfloat16).to(scores.device))
                                                                        
                                    logger.info(f"Validation Loss: {loss.item()}")
                                    if training_args.report_to is not None and accelerator.is_local_main_process:
                                        wandb.log({"Val Loss": loss.item()})
                            
                            accelerator.wait_for_everyone()                        
                            
                            save_dir = os.path.join(training_args.output_dir, "step-{}".format(global_step)).replace(".", "_")
                            os.makedirs(save_dir, exist_ok=True)
                            unwrapped_model.save_pretrained(save_dir,
                                                            is_main_process=accelerator.is_main_process,
                                                            save_function=accelerator.save,
                                                            state_dict=accelerator.get_state_dict(model)
                                                            )
                            del unwrapped_model
                            
                            accelerator.wait_for_everyone() 
                                                
                    model.train()
                    
                    # reward head train only
                    # for param in model.parameters():
                    #     param.requires_grad = False
                    # model.reward_head.requires_grad_(True)
                                    
        
        accelerator.wait_for_everyone()
                

    logger.info("Training completed successfully!")
    
    accelerator.wait_for_everyone()
    
    if test_dataset is not None:
        model.eval()
        with torch.no_grad():
            for test_batch in tqdm(test_dataloader, desc="Testing", disable = not accelerator.is_local_main_process):
                if isinstance(test_batch["img"], str):
                    assert os.path.isfile(test_batch["img"])
                    image = Image.open(test_batch["img"]).convert("RGB").resize((560, 560))
                elif isinstance(test_batch["img"], dict):
                    image = Image.open(io.BytesIO(test_batch["img"]['bytes'])).convert("RGB").resize((560, 560))
                else:
                    image = test_batch['img']
                if 'internlm' in model_args.model_name_or_path.lower():
                    inputs = {
                        "samples": {
                            "text_input": "<ImageHere>;" + test_batch["cand"],
                            "image": totensor(image.convert("RGB").resize((560,560))).unsqueeze(0).bfloat16()
                        },
                        "human_score": torch.tensor(test_batch["human_score"])
                    }
                else:
                    prompt = processor.apply_chat_template(eval("[{'role': 'user','content': [{'type': 'text', 'text': 'Describe this image.'}]},{'role': 'assistant','content': [{'type': 'image'},{'type': 'text', 'text' : \"" + test_batch["cand"][0].strip().replace('"', " ") + "\"}],}]"), tokenize=False)
                    
                    inputs = processor(text = prompt, images = image[0], return_tensors="pt", padding="max_length", truncation=True, max_length=training_args.max_length)
                scores = model(**inputs.to(model.device)).rewards
                test_batch['human_score'] = torch.tensor(test_batch['human_score'][0]).to(torch.bfloat16).to(scores.device)
                loss = F.mse_loss(scores, test_batch['human_score'].to(torch.bfloat16).to(scores.device))
                logger.info(f"Test Loss: {loss.item()}")
                                
                if training_args.report_to is not None and accelerator.is_local_main_process:
                    wandb.log({"Test Loss": loss.item()})
    
    logger.info("Testing completed successfully!")
    
    accelerator.wait_for_everyone()
    
    unwrapped_model = accelerator.unwrap_model(model)
    if training_args.output_dir is not None:
        save_dir = os.path.join(training_args.output_dir, "final_checkpoint")
        os.makedirs(save_dir, exist_ok=True)
        unwrapped_model.save_pretrained(
            save_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
            max_shard_size="2GB",
            safe_serialization=True
        )
    
    # push model to huggingface hub
    if accelerator.is_main_process and training_args.push_to_hub:
    # if training_args.push_to_hub:
        try:
            from huggingface_hub import HfApi
            
            # Initialize the Hugging Face API client
            api = HfApi(token=os.getenv("HF_TOKEN"))
            repo_id = save_dir.split("/")
            
            # Create or ensure the repo exists
            api.create_repo(repo_id=repo_id, private=False, exist_ok=True)
            
            # Upload all files from the save directory
            api.upload_folder(
                folder_path=save_dir,
                repo_id=repo_id,
                repo_type="model",
            )
            
            logger.info(f"Model pushed to hub successfully at {repo_id}!")
        except Exception as e:
            logger.info("Model push to hub failed!")
            logger.info(e)
    
    logger.info("Model saved successfully!")
    wandb.finish()
        

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Parse arguments (update this to match your argument parser)
    parser = H4ArgumentParser((ModelArguments, DataArguments, RewardArguments))
    model_args, data_args, training_args = parser.parse()
    
    main(model_args, data_args, training_args)

