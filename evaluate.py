import os
import gc
import torch
import json
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from collections import defaultdict
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from datasets import load_dataset
from transformers import AutoProcessor
from torchvision.transforms.functional import to_tensor as totensor

from models.reward_model_tuned import VLMRewardModel, VLMRewardConfig
from modules import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    LLAVA_CHAT_TEMPLATE,
    QWEN_CHAT_TEMPLATE,
    INTERNLM_CHAT_TEMPLATE,
    load_model_with_index
)

from Github.Polos.polos.metrics.regression_metrics import RegressionReport

def cal_acc(score_sample, target_sample):
    
    tol_cnt = 0.
    true_cnt = 0.
    for idx in range(len(score_sample)):
        item_base = score_sample[idx] #["ranking"]
        item = target_sample[idx]["rewards"]
        for i in range(len(item_base)):
            for j in range(i+1, len(item_base)):
                if item_base[i] > item_base[j]:
                    if item[i] >= item[j]:
                        tol_cnt += 1
                    elif item[i] < item[j]:
                        tol_cnt += 1
                        true_cnt += 1
                elif item_base[i] < item_base[j]:
                    if item[i] > item[j]:
                        tol_cnt += 1
                        true_cnt += 1
                    elif item[i] <= item[j]:
                        tol_cnt += 1
    
    return true_cnt / tol_cnt

def load_dataset_by_name(dataset_name, hf_token):
    if dataset_name == "filtered-polaris":
        test_dataset = load_dataset("yuwd/Polaris", split="validation")
        test_dataset = test_dataset.filter(lambda x: x['human_score'] <= 0.5)
        test_dataset = test_dataset.rename_column("human_score", "score")
        test_dataset = test_dataset.rename_column("cand", "text")
        test_dataset = test_dataset.rename_column("img", "image")

        for i in range(len(test_dataset)):
            test_dataset[i]["refs"] = test_dataset[i]["refs"][0]

    elif dataset_name == "filtered-oid":
        test_dataset = load_dataset("filtered_oid", split="test", token=hf_token)
        test_dataset = test_dataset.rename_column("chosen_caption", "refs")
        test_dataset = test_dataset.rename_column("rejected_caption", "text")

    elif dataset_name == "pascal":
        test_dataset = load_dataset("pascal50s", split="test", token=hf_token)
        test_dataset = test_dataset.rename_column("a", "refs")
        test_dataset = test_dataset.rename_column("b", "text")

    elif dataset_name == "foil":
        test_dataset = load_dataset("foil", split="test", token=hf_token)
        test_dataset = test_dataset.rename_column("refs", "orig_refs")
        test_dataset = test_dataset.rename_column("orig_mt", "refs")
        test_dataset = test_dataset.rename_column("foil_mt", "text")
        test_dataset = test_dataset.rename_column("imgid", "image")

    elif dataset_name == "eye4b-pref":
        test_dataset = load_dataset("Eye4BPreference", split="test", token=hf_token)
        test_dataset = test_dataset.rename_column("request", "meta")
        test_dataset = test_dataset.rename_column("chosen", "refs")
        test_dataset = test_dataset.rename_column("rejected", "text")
        test_dataset = test_dataset.rename_column("overall_score", "score")

        for i in range(len(test_dataset)):
            test_dataset[i]["refs"] = test_dataset[i]["refs"][0]

    elif dataset_name == "eye4b-o":
        test_dataset = load_dataset("blv_eye4b", split="test", token=hf_token)
        test_dataset = test_dataset.rename_column("request", "meta")
        test_dataset = test_dataset.rename_column("response", "text")
        test_dataset = test_dataset.rename_column("overall_score", "score")

    elif dataset_name == "eye4b-a":
        test_dataset = load_dataset("blv_eye4b", split="test", token=hf_token)
        test_dataset = test_dataset.rename_column("request", "meta")
        test_dataset = test_dataset.rename_column("response", "text")
        test_dataset = test_dataset.rename_column("avg_score", "score")

    elif dataset_name == "flickrexp":
        test_dataset = load_dataset("flickrexp", split="test", token=hf_token)
        test_dataset = test_dataset.rename_column("references", "refs")
        test_dataset = test_dataset.rename_column("candidate", "text")

        for i in range(len(test_dataset)):
            test_dataset[i]["refs"] = test_dataset[i]["refs"][0]

    elif dataset_name == "flickrcf":
        test_dataset = load_dataset("flickrcf", split="test", token=hf_token)
        test_dataset = test_dataset.rename_column("references", "refs")
        test_dataset = test_dataset.rename_column("candidate", "text")

        for i in range(len(test_dataset)):
            test_dataset[i]["refs"] = test_dataset[i]["refs"][0]

    elif dataset_name == "polaris":
        test_dataset = load_dataset("polaris_test", split="test", token=hf_token)
        test_dataset = test_dataset.rename_column("mt", "text")

        for i in range(len(test_dataset)):
            test_dataset[i]["refs"] = test_dataset[i]["refs"][0]

    elif dataset_name == "imgrew":
        test_dataset = load_dataset("imgrew_test", split="test", token=hf_token)
        test_dataset = test_dataset.rename_column("prompt", "text")
        
    else:
        raise Exception("Not implemented dataset")
    
    return test_dataset

def evaluate(args):
    
    # model load
    model_args = ModelArguments(
        model_name_or_path= args.model_name,
        use_peft=False,
        checkpoint_dir=None,
    )
    config = VLMRewardConfig()
    model = VLMRewardModel(model_args, config)
    load_model_with_index(model, args.ckpt_path)
    
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True,
                                              token=os.getenv("HF_TOKEN"))
    
    if 'internlm' in args.model_name:
        model.backbone_model.tokenizer = processor
    
    
    model = model.to(args.device).eval().bfloat16()
    
    test_dataset = load_dataset_by_name(args.dataset_name, args.token)
    
    if args.dataset_name in ["filtered-polaris", "filtered-oid", "pascal", "foil", "eye4b-pref", "imgrew"]:
        pairwise = True
    elif args.dataset_name in ["eye4b-o", "eye4b-a", "flickrexp", "flickrcf", "polaris"]:
        pairwise = False
        rep_b = RegressionReport(kendall_type='b')
        rep_c = RegressionReport(kendall_type='c')
        gt_scores = []
        sys_scores = []

    if args.dataset_name == "pascal":
        total_score = defaultdict(list)
    else:
        total_score = []

    if args.dataset_name in ["imgrew"]:
        target_sample = []
        score_sample = []
    
    with torch.no_grad():
        i = 0
        for example in tqdm(test_dataset): 
            if args.dataset_name in ["imgrew"]:
                ranking = example["ranking"]
                score_sample.append(ranking)
                prmpt = example["text"]
                rewards = []
                for img in [example[f"image_{i+1}"] for i in range(len(ranking))]:
                    if isinstance(img, str):
                        image = Image.open(img).convert("RGB").resize((560, 560))
                    else:
                        image = img.convert("RGB").resize((560, 560))
                    prompt = [{'role': 'assistant','content': [{'type': 'text', 'text': 'Describe this image.'},
                                                                {'type': 'image'},
                                                                {'type': 'text', 'text' : prmpt}]}]
                    prompt = processor.apply_chat_template(prompt, tokenize=False)
            
                    inputs = processor(text=prompt, images=image, return_tensors="pt", padding="max_length", max_length=2048, truncation=True)
                    inputs = {k: v.to(args.device) for k, v in inputs.items()}
                    
                    score = model(**inputs).rewards
                    score = score.cpu().float().numpy()
                    
                    rewards.append(float(score))
                
                target_item = {
                    "id": example['image_id'],
                    "prompt": prmpt,
                    "rewards": rewards
                }
                target_sample.append(target_item)
                print(score_sample)
                print(target_sample)
                
            else:
                # if path convert to image
                if isinstance(example["image"], str):
                    image = Image.open(example["image"]).convert("RGB").resize((560, 560))
                else:
                    image = example["image"].convert("RGB").resize((560, 560))
    
                if args.dataset_name in ["eye4b-pref"]:
                    
                    prompt = [{'role': 'assistant','content': [
                                {'type': 'text', 'text': example["meta"]},
                                {'type': 'image'},
                                {'type': 'text', 'text' : example["refs"]}]}]
                    
                else:
                    prompt = [{'role': 'assistant','content': [{'type': 'text', 'text': 'Describe this image.'},
                                                                {'type': 'image'},
                                                                {'type': 'text', 'text' : example["text"]}]}]
                
                prompt = processor.apply_chat_template(prompt, tokenize=False)
                
                inputs = processor(text=prompt, images=image, return_tensors="pt", padding="max_length", max_length=2048, truncation=True)
                inputs = {k: v.to(args.device) for k, v in inputs.items()}
                
                score = model(**inputs).rewards
                score = score.cpu().float().numpy()

                if pairwise:
                    if args.dataset_name in ["eye4b-pref"]:
                    
                        
                        prompt = [{'role': 'assistant','content': [
                                    {'type': 'text', 'text': example["meta"]},
                                    {'type': 'image'},
                                    {'type': 'text', 'text' : example["refs"]}]}]                        
                    else:
                        prompt = [
                            {'role': 'assistant','content': [{'type': 'text', 'text': 'Describe this image.'},{'type': 'image'},{'type': 'text', 'text' : example["refs"]}]}]
    
                    prompt = processor.apply_chat_template(prompt, tokenize=False)
                    
                    if isinstance(example["image"], str):
                        image = Image.open(example["image"]).convert("RGB").resize((560, 560))
                    else:
                        image = example["image"].convert("RGB").resize((560, 560))
                    
                    inputs = processor(text=prompt, images=image, return_tensors="pt", padding="max_length", max_length=2048, truncation=True)
                    inputs = {k: v.to(args.device) for k, v in inputs.items()}
                    pair_score = model(**inputs).rewards
                    pair_score = pair_score.cpu().float().numpy()
    
                    if args.dataset_name == "pascal":
                        score = 0 if score < pair_score else 1
                        if score == int(example['label']):
                            total_score[example['category_str']].append(1)
                        else:
                            total_score[example['category_str']].append(0)
                    else:
                        if score < pair_score:
                            total_score.append(1)
                        else:
                            total_score.append(0)
                else:
                    sys_scores.append(float(score[0]))
                    gt_scores.append(example["score"])
                
            # i += 1
            # if i == 10:
            #     break

    if args.dataset_name == "pascal":
        total_score = {k:float(np.mean(v)) for k,v in total_score.items()}
        print(total_score) # {'HI': 0.499, 'HC': 0.35, 'HM': 0.513, 'MM': 0.595}
    elif args.dataset_name in ["imgrew"]:
        test_acc = cal_acc(score_sample, target_sample)
        print("Accuracy: ", test_acc)
    else:
        total_score = np.array(total_score)
        
        if pairwise:
            print("Accuracy: ", np.mean(total_score))
        else:
            coef_tensor = rep_b.compute(sys_scores, gt_scores)
            coefs = {k : round(float(v.numpy() if not isinstance(v,float) else v),4) for k, v in coef_tensor.items()}
            print("Kendall tau_b: ", coefs)
            coef_tensor = rep_c.compute(sys_scores, gt_scores)
            coefs = {k : round(float(v.numpy() if not isinstance(v,float) else v),4) for k, v in coef_tensor.items()}
            print("Kendall tau_c: ", coefs)
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="filtered-polaris", choices=["filtered-polaris", "filtered-oid", "pascal", "foil", "eye4b-pref",
                                                                                        "eye4b-o", "eye4b-a", "flickrexp", "flickrcf", "polaris", "imgrew"])
    parser.add_argument("--token", type=str, default=os.getenv("HF_TOKEN"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--ckpt_path", type=str, default="")
    
    
    args = parser.parse_args()
    
    evaluate(args)

if __name__ == "__main__":
    main()