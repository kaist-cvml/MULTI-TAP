import os
import argparse
import torch
from datasets import load_dataset
from transformers import AutoProcessor

from models.reward_model import VLMRewardModel, VLMRewardConfig

def main():
    parser = argparse.ArgumentParser(description="Run demo for Multi-TAP reward model")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-2B-Instruct", 
                       help="Base model name for the backbone")
    parser.add_argument("--ckpt_path", type=str, default="brl-xfact/Multi-TAP-Qwen2-VL-2B",
                       help="Path to checkpoint (local directory or HuggingFace model ID)")
    parser.add_argument("--dataset", type=str, default="brl-xfact/polaris_imagereward",
                       help="Dataset to use for demo")
    parser.add_argument("--device", type=str, default="cuda:4",
                       help="Device to run on (cuda/cpu)")
    args = parser.parse_args()

    # Normalize/validate device
    requested_device = args.device
    if requested_device.startswith("cuda"):
        if torch.cuda.is_available():
            # Validate index if provided
            if ":" in requested_device:
                try:
                    cuda_index = int(requested_device.split(":", 1)[1])
                    if cuda_index < 0 or cuda_index >= torch.cuda.device_count():
                        print(f"Warning: Requested {requested_device} not available. Falling back to cuda:0.")
                        args.device = "cuda:0"
                except ValueError:
                    print(f"Warning: Malformed CUDA device '{requested_device}'. Falling back to cuda:0.")
                    args.device = "cuda:0"
            else:
                args.device = "cuda:0"
        else:
            print("Warning: CUDA not available. Falling back to CPU.")
            args.device = "cpu"
    else:
        args.device = "cpu"

    # Check if ckpt_path is a local directory or a remote model
    model = None
    if os.path.exists(args.ckpt_path) and os.path.isdir(args.ckpt_path):
        # Local directory - use from_local_checkpoint
        print(f"Loading model from local directory: {args.ckpt_path}")
        try:
            model = VLMRewardModel.from_local_checkpoint(args.ckpt_path, args.model_name)
            print("Successfully loaded model from local directory")
        except Exception as e:
            print(f"Error loading model from local directory: {e}")
            raise
    else:
        # Remote model - use from_pretrained
        print(f"Loading model from HuggingFace Hub: {args.ckpt_path}")
        try:
            model = VLMRewardModel.from_pretrained(args.ckpt_path, base_model_name=args.model_name)
            print("Successfully loaded model from HuggingFace Hub")
        except Exception as e:
            print(f"Failed to load model from HuggingFace Hub: {e}")
            print("Falling back to loading backbone model only...")
            # If loading the full reward model fails, use the backbone model with a randomly initialized reward head
            try:
                model = VLMRewardModel.from_pretrained(args.model_name, base_model_name=args.model_name)
                print("Successfully initialized backbone-only reward model")
            except Exception as e2:
                print(f"Error initializing backbone-only model: {e2}")
                raise

    # Load processor with proper token handling
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        processor = AutoProcessor.from_pretrained(args.model_name, token=hf_token)
    else:
        print("Warning: HF_TOKEN not found. Loading processor without authentication.")
        processor = AutoProcessor.from_pretrained(args.model_name)

    # Move model to device and set to evaluation mode
    model = model.to(args.device).eval()
    
    # On CUDA, disable Flash/Memory-Efficient SDPA and force math backend to avoid cuDNN engine errors
    if args.device.startswith("cuda"):
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass
        # Use float16 for attention kernels stability
        model = model.half()

    # Pull one sample
    print(f"Loading dataset: {args.dataset}")
    try:
        try:
            ds = load_dataset(args.dataset, split="test")
        except Exception:
            ds = load_dataset(args.dataset, split="train[:1]")
        example = ds[0]
        print("Successfully loaded dataset sample")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

    # Process image
    try:
        if isinstance(example.get("img"), str):
            from PIL import Image
            image = Image.open(example["img"]).convert("RGB").resize((560, 560))
        else:
            image = example["img"].convert("RGB").resize((560, 560))
        print("Successfully processed image")
    except Exception as e:
        print(f"Error processing image: {e}")
        raise

    text = example.get("cand") or example.get("text") or example.get("prompt") or "Describe this image."
    print(f"Using text: {text[:100]}...")

    # Create prompt and process
    try:
        prompt = [{"role": "assistant", "content": [
            {"type": "text", "text": "Describe this image."},
            {"type": "image"},
            {"type": "text", "text": text}
        ]}]
        prompt = processor.apply_chat_template(prompt, tokenize=False)
        print("Successfully created prompt")

        inputs = processor(text=prompt, images=image, return_tensors="pt", padding="max_length", max_length=1024, truncation=True)
        inputs = {k: v.to(args.device) for k, v in inputs.items()}
        print("Successfully processed inputs")
    except Exception as e:
        print(f"Error processing inputs: {e}")
        raise

    # Run inference
    try:
        with torch.no_grad():
            if args.device.startswith("cuda"):
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    score = model(**inputs).rewards
            else:
                score = model(**inputs).rewards
            print("Reward:", float(score.detach().cpu().float().item()))
    except Exception as e:
        print(f"Error during inference: {e}")
        raise


if __name__ == "__main__":
    main()


