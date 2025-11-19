import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import pandas as pd
from tqdm import tqdm
import json
import argparse
from utils.similarity import compute_consine_similarity
from PIL import Image
from transformers import AutoModelForCausalLM,AutoProcessor 
from utils.masker import PhiAttentionMasker as AttentionMasker
import warnings
warnings.filterwarnings("ignore")
def get_phi_model(model_id, device='cuda:0'):

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="cuda",
        trust_remote_code=True
    )
    
    return model, processor
    

def process_single_sample(model, processor, 
                          image_path, prompt, target_layer,mask_cfg=None, device="cuda"):
    """Process a single sample and get the hidden state of the last layer"""
    device = torch.device(device)
    result = []
    
    with torch.no_grad():
        masker = None
        if mask_cfg and "head_mask" in mask_cfg:
            masker = AttentionMasker(
                model=model,
                head_mask=mask_cfg["head_mask"],
                mask_type=mask_cfg.get("mask_type", "scale_mask"),
                scale_factor=mask_cfg.get("scale_factor", 1e-5)
            )
            masker.apply_hooks()
        
        try:
            image = Image.open(image_path).convert("RGB")
            messages = [
                {"role": "user", "content": f"<|image_1|>\n"+ prompt }
            ]
            prompt = processor.tokenizer.apply_chat_template(messages,
                                          tokenize=False,
                                          add_generation_prompt=True,)
            inputs  = processor(prompt,image,return_tensors="pt").to(device)

            outputs = model(
                **inputs,
                output_hidden_states=True,
            )
            
            hidden_state = outputs.hidden_states[target_layer][0, -1, :]
            result.append(hidden_state)
            result = torch.cat(result, dim=0)
            result = result.to(torch.float32).cpu()
        except Exception as e:
            print(f"Error during processing: {e}")
            import traceback
            traceback.print_exc()
            result = None
            
        finally:
            if masker:
                masker.remove_hooks()
    
    return result


def defense_pipeline(args):
    """Main defense pipeline - process each prompt separately"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    threshold = args.threshold
    safe_heads = args.safe_heads
    hidden_layer = args.hidden_layer
    
    # Create save directory
    csv_path = args.csv_path
    model_name = args.model_path
    save_path = args.save_path
    model_name_without_ext = os.path.basename(model_name)
    filename_without_ext = os.path.splitext(os.path.basename(csv_path))[0]
    save_dir = os.path.join(save_path, model_name_without_ext, f"{filename_without_ext}_layer{hidden_layer}")
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Loading model: {model_name}")
    model, processor = get_phi_model(model_name, device=device)
    
    # Load data
    try:
        data = pd.read_csv(csv_path)
        # Add new columns for results
        data["IsPotentialHarmful"] = False
        data["Deflection_Angle"] = 0.0  # Add a column to record difference scores
    except FileNotFoundError:
        print(f"Data file not found: {csv_path}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Configure safe head masks
    mask_cfg = {"head_mask": {head: ["q"] for head in safe_heads}}
    
    # Prepare image paths and query texts
    batch_query_text = data["jailbreak_query"]
    batch_image_path = [os.path.join(args.image_path, path) for path in data["image_path"]]
    sample_count = min(len(batch_image_path), len(batch_query_text))

    # Process each prompt individually
    print("Starting prompt analysis...")
    for idx, (image_path, prompt) in enumerate(tqdm(zip(batch_image_path, batch_query_text), total=len(batch_image_path), desc="Processing samples")):
        if idx >= len(data):
            print(f"Warning: Index {idx} exceeds data range, stopping")
            break
            
        # Get baseline hidden state (no mask)
        base_hs = process_single_sample(
            model, processor,
            image_path, prompt, hidden_layer,mask_cfg=None, device=device
        )
        
        if base_hs is None:
            print(f"Warning: Sample {idx} baseline hidden state retrieval failed, skipping")
            continue
            
        # Get hidden state after masking safe heads
        masked_hs = process_single_sample(
            model, processor,
            image_path, prompt, hidden_layer,mask_cfg=mask_cfg, device=device
        )
        
        if masked_hs is None:
            print(f"Warning: Sample {idx} masked hidden state retrieval failed, skipping")
            continue
        # result = assign_by_principal_angle(masked_hs.cuda(), safet_hs, jail_hs)
        # Calculate difference metric
        diff = compute_consine_similarity(base_hs, masked_hs)
        
        data.at[idx, "Deflection_Angle"] = float(diff)
        data.at[idx, "IsPotentialHarmful"] = (diff > threshold)
        
        # Save intermediate results periodically
        if (idx + 1) % 100 == 0:
            save_file = os.path.join(save_dir, "defense_results.csv")
            data.to_csv(save_file, index=False)
            print(f"Processed {idx+1}/{sample_count} samples")

    # Save final results
    save_file = os.path.join(save_dir, "defense_results.csv")
    data.to_csv(save_file, index=False)
    
    # Output statistics
    harmful_count = data["IsPotentialHarmful"].sum()
    total_count = len(data)
    print(f"Analysis completed: {harmful_count}/{total_count} prompts marked as potentially harmful")
    print(f"Results saved to {save_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM Defense Mechanism Parameters")
    parser.add_argument("--model_path", type=str, default="/path/to/your/model",
                        help="Path to the model")
    parser.add_argument("--image_path", type=str, default="/path/to/your/images",
                        help="Directory for image files")
    parser.add_argument("--csv_path", type=str, default="/path/to/your/dataset.csv",
                        help="Path to the dataset CSV file")
    parser.add_argument("--save_path", default="./results/detect",
                         help="Result save path")
    parser.add_argument("--hidden_layer", type=int, default=-1,
                         help="Hidden layer number")
    parser.add_argument("--safe_heads", type=json.loads, required=True)
    parser.add_argument("--threshold", type=float, default=3.5,
                         help="Decision threshold")
    
    args = parser.parse_args()
    args.safe_heads = [tuple(h) for h in args.safe_heads]  
    
    defense_pipeline(args)