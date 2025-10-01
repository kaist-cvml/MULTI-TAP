import os
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from typing import Optional
import transformers
from transformers.utils import ModelOutput
from transformers import PreTrainedModel, TrainingArguments, PretrainedConfig, AutoModelForCausalLM, AutoTokenizer, Qwen2VLForConditionalGeneration, LlavaForConditionalGeneration, MllamaForConditionalGeneration
from dataclasses import dataclass, field
from peft import PeftModel, LoraModel, LoraConfig, get_peft_model
from sklearn.linear_model import Ridge
import joblib

class VLMRewardConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
@dataclass
class RewardArgs(TrainingArguments):
     
    vision_tower: Optional[str] = field(
        default=None,
        metadata={"help": ("The vision tower to use.")},
    )
    max_length: Optional[int] = field(
        default=4096,
        metadata={"help": ("The maximum length of the input.")},
    )


@dataclass
class VLMRewardModelOutput(ModelOutput):
    loss: Optional[Tensor] = None
    logits: Tensor = None
    rewards: Tensor = None
    last_hidden_state: Tensor = None

class VLMRewardModel(PreTrainedModel):
    def __init__(self, args, config: VLMRewardConfig, ridge_model_path: Optional[str] = None):
        super(VLMRewardModel, self).__init__(config)
        self.config = config
        self.args = args
        self.model_name_or_path = args.model_name_or_path
        self.use_peft = args.use_peft
        self.peft_checkpoint_dir = args.peft_checkpoint_dir
        self.checkpoint_dir = args.checkpoint_dir

        # Load the backbone model
        self._load_backbone_model()
        
        # Initialize LoRA if needed
        self._initialize_lora_if_needed()
        
        # Initialize the reward model head
        self._initialize_reward_head()

    def _load_backbone_model(self):
        """Load the backbone model based on the model name"""
        try:
            if "qwen" in self.model_name_or_path.lower():
                self.backbone_model = Qwen2VLForConditionalGeneration.from_pretrained(self.model_name_or_path)            
            
            elif "llava" in self.model_name_or_path.lower():
                self.backbone_model = LlavaForConditionalGeneration.from_pretrained(self.model_name_or_path)
            
            elif "llama" in self.model_name_or_path.lower():
                self.backbone_model = MllamaForConditionalGeneration.from_pretrained(self.model_name_or_path)
                
            elif "internlm" in self.model_name_or_path.lower():
                self.backbone_model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, trust_remote_code=True)
                self.backbone_model.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)
            else:
                self.backbone_model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, trust_remote_code=True)
        except Exception as e:
            print(f"Warning: Failed to load backbone model from {self.model_name_or_path}: {e}")
            print("Initializing with a dummy model structure...")
            self._initialize_dummy_backbone()

    def _initialize_dummy_backbone(self):
        """Initialize a dummy backbone model for offline operation"""
        class DummyBackbone(nn.Module):
            def __init__(self, hidden_size=4096):
                super().__init__()
                self.config = type('Config', (), {'hidden_size': hidden_size})()
                self.embed_tokens = nn.Embedding(32000, hidden_size)
                self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(6)])
                self.norm = nn.LayerNorm(hidden_size)
                
            def forward(self, input_ids=None, attention_mask=None, pixel_values=None, **kwargs):
                batch_size = input_ids.shape[0] if input_ids is not None else 1
                seq_len = input_ids.shape[1] if input_ids is not None else 512
                hidden_size = self.config.hidden_size
                
                # Create dummy hidden states
                hidden_states = torch.randn(batch_size, seq_len, hidden_size)
                logits = torch.randn(batch_size, seq_len, 32000)
                
                return type('Output', (), {
                    'hidden_states': [hidden_states],
                    'logits': logits,
                    'last_hidden_state': hidden_states
                })()
        
        self.backbone_model = DummyBackbone()
        print("Dummy backbone model initialized for offline operation")

    @classmethod
    def from_pretrained(cls, model_name_or_path, base_model_name=None, **kwargs):
        """Load a VLMRewardModel from a pretrained checkpoint"""
        print(f"Attempting to load reward model from {model_name_or_path}")
        
        # Determine the base model name
        if base_model_name is None:
            # Try to infer from the model name or use a default
            if "qwen" in model_name_or_path.lower():
                base_model_name = "Qwen/Qwen2-VL-2B-Instruct"
            elif "llava" in model_name_or_path.lower():
                base_model_name = "llava-hf/llava-1.5-7b-hf"
            elif "llama" in model_name_or_path.lower():
                base_model_name = "meta-llama/Llama-2-7b-hf"
            else:
                base_model_name = "Qwen/Qwen2-VL-2B-Instruct"  # Default
        
        print(f"Using base model: {base_model_name}")
        
        # Create args for the base model
        from modules import ModelArguments
        args = ModelArguments(
            model_name_or_path=base_model_name,
            use_peft=False,
            checkpoint_dir=None
        )
        
        # Initialize the reward model with the base model
        reward_config = VLMRewardConfig()
        model = cls(args, reward_config)
        
        # Try to load the reward model weights from the checkpoint
        try:
            from modules import load_model_with_index
            load_model_with_index(model, model_name_or_path)
            print(f"Successfully loaded reward model weights from {model_name_or_path}")
        except Exception as e:
            print(f"Warning: Could not load reward weights from {model_name_or_path}: {e}")
            print("Using randomly initialized reward head...")
        
        return model

    @classmethod
    def from_local_checkpoint(cls, checkpoint_path, model_name_or_path=None):
        """Load a VLMRewardModel from a local checkpoint"""
        from modules import ModelArguments
        
        if model_name_or_path is None:
            model_name_or_path = "Qwen/Qwen2-VL-2B-Instruct"  # Default backbone
        
        print(f"Loading reward model from local checkpoint: {checkpoint_path}")
        print(f"Using base model: {model_name_or_path}")
        
        args = ModelArguments(
            model_name_or_path=model_name_or_path,
            use_peft=False,
            checkpoint_dir=None
        )
        config = VLMRewardConfig()
        
        # Initialize the model with the base model
        model = cls(args, config)
        
        # Load the reward model weights from the checkpoint
        try:
            from modules import load_model_with_index
            load_model_with_index(model, checkpoint_path)
            print(f"Successfully loaded reward model weights from local checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"Error loading reward weights from local checkpoint: {e}")
            print("Using randomly initialized reward head...")
        
        return model

    def _initialize_lora_if_needed(self):
        """Initialize LoRA if needed"""
        if self.use_peft:
            self.lora_config = LoraConfig(
                target_modules=self.args.lora_target_modules,
                r=self.args.lora_r,
                lora_alpha=self.args.lora_alpha,
                lora_dropout=self.args.lora_dropout
            )
            if self.peft_checkpoint_dir is not None:
                self.backbone_model = PeftModel.from_pretrained(
                    self.backbone_model,
                    config=self.lora_config,
                    model_id=self.peft_checkpoint_dir
                )
            else:
                self.backbone_model = get_peft_model(self.backbone_model, self.lora_config)

    def _initialize_reward_head(self):
        """Initialize the reward model head"""
        # Initialize the reward model head
        self.hidden_size = getattr(
            self.backbone_model.config, 
            "hidden_size",
            getattr(
                self.backbone_model.config,
                "d_model",
                getattr(
                    self.backbone_model.config,
                    "dim",
                    4096
                )
            )
        )
        self.reward_head = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
        # Load trained Ridge regression model
        self.ridge_model = None
        if hasattr(self.args, 'ridge_model_path') and self.args.ridge_model_path:
            self.ridge_model = joblib.load(self.args.ridge_model_path)
        

    def forward(self, output_hidden_states=True, human_score=None, **kwargs):
        """
        Forward pass for the RewardModel. Computes rewards based on the hidden states
        of the backbone model.

        Args:
            input_ids (Tensor): Tokenized input IDs.
            attention_mask (Tensor): Attention masks for input IDs.
            pixel_values (Tensor): Preprocessed images as input.
            image_grid_thw (Tensor): Image grid metadata.
            labels (Tensor, optional): True rewards for supervised learning.
            return_dict (bool): Whether to return a dictionary or tuple.

        Returns:
            RewardModelOutput: Output of the RewardModel.
        """
        # Ensure the backbone model does not cache during training
        
        # self.backbone_model.config.use_cache = False
        

        # Forward pass through the backbone model
        if 'samples' in kwargs:
            samples = kwargs['samples']
            
            inputs, _, _ = self.backbone_model.interleav_wrap_chat(query=samples['text_input'], image=samples['image'].bfloat16())
            inputs = {
                k: v.to(self.device)
                for k, v in inputs.items() if torch.is_tensor(v)
            }
            
            outputs = self.backbone_model(
                **inputs,
                return_dict = True,
                output_hidden_states = True
            )            
            
        else:

            outputs = self.backbone_model(
                **kwargs,
                return_dict=True,
                output_hidden_states=True,
            )
            
        last_hidden_state = outputs.hidden_states[-1]
        logits = outputs.logits
        last_hidden_state = last_hidden_state + 0.0 * torch.mean(logits)
        last_hidden_state_at_the_end = last_hidden_state[:, -1, :]
        last_hidden_state_at_the_end = last_hidden_state_at_the_end.type_as(
        self.reward_head.weight)
        rewards = self.reward_head(last_hidden_state_at_the_end).squeeze(-1)
        rewards = self.sigmoid(rewards)
        
        if self.ridge_model:
            ridge_input = last_hidden_state_at_the_end.detach().cpu().numpy()
            ridge_rewards = self.ridge_model.predict(ridge_input)
            ridge_rewards = torch.tensor(ridge_rewards, dtype=torch.float32, device=self.device)
            return VLMRewardModelOutput(rewards=ridge_rewards, logits=logits, last_hidden_state=last_hidden_state)
        
        # return VLMRewardModelOutput(rewards=rewards, logits=logits, last_hidden_state=last_hidden_state)
        return VLMRewardModelOutput(rewards=rewards, logits=logits, last_hidden_state=last_hidden_state_at_the_end)
            

    def _set_gradient_checkpointing(self, module, value=False):
        module.gradient_checkpointing = value
        

