import os
import gc
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from datasets import load_dataset
from transformers import AutoProcessor
from torchvision.transforms.functional import to_tensor as totensor

from models.reward_model import VLMRewardModel, VLMRewardConfig
from modules import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    LLAVA_CHAT_TEMPLATE,
    QWEN_CHAT_TEMPLATE,
    INTERNLM_CHAT_TEMPLATE,
    load_model_with_index
)

def prepare_input(text, image, processor, model_name, device, response=None):
    if 'qwen' in model_name.lower() or 'llama' in model_name.lower():
        
        # not cookbook
        if response is not None:
            prompt = [{"role": "assistant", "content": [{"type": "text", "text": text},
                                                        {"type": "image"}, {"type": "text", "text": response}]}]
        
        else:
            prompt = [{"role": "user", "content": [{"type": "text", "text": "Describe this image."}]},
                    {"role": "assistant", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        
        # cookbook
        # prompt = [{'role': 'user','content': [{'type': 'image'},{'type': 'text', 'text' : text}]}]
        
        prompt = processor.apply_chat_template(prompt, tokenize=False)
        image = image.convert("RGB").resize((560, 560))
        inputs = processor(text=prompt, images=image, return_tensors="pt", padding="max_length", max_length=1024, truncation=True)
        return {k: v.to(device) for k, v in inputs.items()}
    
    elif 'internlm' in model_name.lower():
        
        if response is not None:
            return{
                "samples": {
                    "text_input": text + "<ImageHere>;" + response,
                    "image": totensor(image.convert("RGB").resize((560, 560))).unsqueeze(0).unsqueeze(0).bfloat16()
                }
            }
            
        else:
            return {
                "samples": {
                    # not cookbook
                    "text_input": "Describe this image. <ImageHere>;" + text,
                    # cookbook
                    "text_input": "<ImageHere>;" + text,
                    "image": totensor(image.convert("RGB").resize((560, 560))).unsqueeze(0).unsqueeze(0).bfloat16()
                }
            }
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def get_dual_inputs(entry, dataset_type):
    if dataset_type == "text-image-to-text":
        return [
            {"text": entry["question"], "response": entry["response_1"], "image": entry["image"]},
            {"text": entry["question"], "response": entry["response_2"], "image": entry["image"]}
            
        ]
    elif dataset_type == "text-to-image":
        return [
            {"text": entry["prompt"], "image": entry["image_1"]},
            {"text": entry["prompt"], "image": entry["image_2"]}
        ]
    else:
        return [{"text": entry["prompt"], "image": entry["image"]}]

def process_batch(dataset, model, processor, device="cuda", model_name="Qwen/Qwen2-VL-2B-Instruct", save_dir="temp_outputs", dataset_type="vision_reward"):
    os.makedirs(save_dir, exist_ok=True)
    batch_size = 4
    chunk_size = 1000
    all_outputs = []
    dataset = list(dataset)
    for chunk_start in tqdm(range(0, len(dataset), chunk_size)):
        chunk_scores = []
        chunk_end = min(chunk_start + chunk_size, len(dataset))
        for start_idx in range(chunk_start, chunk_end, batch_size):
            batch = dataset[start_idx:start_idx + batch_size]
            for entry in batch:
                dual_inputs = get_dual_inputs(entry, dataset_type)
                for input_obj in dual_inputs:
                    response = input_obj.get("response", None)
                    inputs = prepare_input(input_obj["text"], input_obj["image"], processor, model_name, device, response)
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            rewards = model(**inputs).last_hidden_state.cpu().float().numpy()
                            chunk_scores.append([rewards])
                    del inputs
                    torch.cuda.empty_cache()
        chunk_scores = np.array(chunk_scores)
        chunk_file = os.path.join(save_dir, f"chunk_{chunk_start}_{chunk_end}.npy")
        np.save(chunk_file, chunk_scores)
        del chunk_scores
        gc.collect()

    print("Combining chunks...")
    all_chunks = [np.load(os.path.join(save_dir, f)) for f in sorted(os.listdir(save_dir)) if f.startswith("chunk_")]
    for f in sorted(os.listdir(save_dir)):
        if f.startswith("chunk_"): os.remove(os.path.join(save_dir, f))

    return np.vstack(all_chunks)

def load_dataset_by_name(name):
    if name == "vision_reward":
        dataset = load_dataset("THUDM/VisionRewardDB-Image", split="train")
        dataset = dataset.train_test_split(test_size=0.025, shuffle=False)
        train_dataset = dataset['train']
        test_dataset = dataset['test']
        def get_targets(dataset): return np.array([x["meta_result"] for x in dataset])
    elif name.startswith("text"):
        dataset = load_dataset("PKU-Alignment/align-anything", name=name)
        train_dataset = dataset["train"]
        test_dataset = dataset["val"]
        if name == "text-image-to-text":
            def get_targets(dataset):
                return np.array([
                    [x[k]>=2 for k in [
                        "prompt_following_rate_1", "objective_rules_rate_1", "clarity_rate_1", "information_richness_rate_1", "safety_rate_1"
                    ]] +
                    [x[k]>=2 for k in [
                        "prompt_following_rate_2", "objective_rules_rate_2", "clarity_rate_2", "information_richness_rate_2", "safety_rate_2"
                    ]]
                    for x in dataset
                ])
        elif name == "text-to-image":
            def get_targets(dataset):
                return np.array([
                    [x[k]>=2 for k in [
                        "prompt_following_rate_1", "objective_rules_rate_1", "aesthetics_rate_1", "information_richness_rate_1", "safety_rate_1"
                    ]] +
                    [x[k]>=2 for k in [
                        "prompt_following_rate_2", "objective_rules_rate_2", "aesthetics_rate_2", "information_richness_rate_2", "safety_rate_2"
                    ]]
                    for x in dataset
                ])
        else:
            raise ValueError("Unknown text dataset type.")
    else:
        raise ValueError(f"Unsupported dataset: {name}")
    return train_dataset, test_dataset, get_targets

def train_and_evaluate(args):
    model_args = ModelArguments(
        model_name_or_path=args.model_name,
        use_peft=False,
        checkpoint_dir=args.checkpoint_dir,
    )
    config = VLMRewardConfig()
    model = VLMRewardModel(model_args, config)
    load_model_with_index(model, args.checkpoint_dir)

    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    if 'internlm' in args.model_name.lower():
        model.backbone_model.tokenizer = processor

    model = model.to(args.device).eval().bfloat16()
    
    print("model:", args.model_name)
    print("checkpoint_dir:", args.checkpoint_dir)

    train_dataset, test_dataset, get_targets = load_dataset_by_name(args.dataset)
    
    print(f"Loaded {args.dataset} dataset with {len(train_dataset)} training samples and {len(test_dataset)} test samples.")

    print("Processing training data...")
    train_outputs = process_batch(train_dataset, model, processor, args.device, args.model_name, args.save_dir, args.dataset)
    # train_outputs = np.array(train_outputs).reshape(len(train_outputs), -1)
    train_targets = get_targets(train_dataset)
    # train_targets = train_targets.reshape(-1, 2, train_targets.shape[-1] // 2)
    # train_targets = train_targets.reshape(-1, train_targets.shape[-1])

    best_alpha, best_mse = None, float("inf")
    for alpha in [0.001, 0.01, 0.1, 1, 10, 100]:
        reg = Ridge(alpha=alpha)
        reg.fit(train_outputs, train_targets)
        mse = mean_squared_error(reg.predict(train_outputs), train_targets)
        print(f"alpha: {alpha}, mse: {mse:.4f}")
        if mse < best_mse:
            best_alpha, best_mse = alpha, mse

    print(f"Training final model with best alpha: {best_alpha}")
    reg = Ridge(alpha=best_alpha)
    reg.fit(train_outputs, train_targets)

    del train_outputs, train_targets
    gc.collect()
    torch.cuda.empty_cache()

    print("Processing test data...")
    test_outputs = process_batch(test_dataset, model, processor, args.device, args.model_name, args.save_dir, args.dataset)
    # test_outputs = np.array(test_outputs).reshape(len(test_outputs), -1)
    test_targets = get_targets(test_dataset)
    # test_targets = test_targets.reshape(-1, 2, test_targets.shape[-1] // 2)
    # test_targets = test_targets.reshape(-1, test_targets.shape[-1])

    preds = reg.predict(test_outputs)
    mse = mean_squared_error(preds, test_targets)
    print(f"Test MSE: {mse:.4f}")

    threshold = 0.5
    accuracies = []
    for i in range(test_targets.shape[1]):
        acc = np.mean(np.abs(preds[:, i] - test_targets[:, i]) <= threshold) * 100
        accuracies.append(acc)

    print(f"Average accuracy: {np.mean(accuracies):.2f}%")
    for i, acc in enumerate(accuracies):
        print(f"Dimension {i}: {acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="temp_outputs_qwen_2")
    parser.add_argument("--dataset", type=str, choices=["vision_reward", "text-image-to-text", "text-to-image"], required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    train_and_evaluate(args)
