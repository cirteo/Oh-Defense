import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import pandas as pd
import os
import json
from tqdm import tqdm
from copy import deepcopy
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from utils.similarity import compute_consine_similarity
from utils.masker import Qwen2AttentionMasker as AttentionMasker
import argparse


def get_qwen2_model(model_path, device='cuda:0'):
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
    )
    
    if device != "auto" and not isinstance(model.device, torch.device):
        model = model.to(device)
    
    return processor, model


def preprocess_data(processor, data, root, device='cuda:0'):
    """Preprocess data, cache image tensors and input_ids"""
    preprocessed_data = []
    for i, r in tqdm(data.iterrows(), total=len(data), desc="Preprocessing data"):
        # Prepare conversation template
        prompt = r['jailbreak_query']
        image_path = r['image_path']
        image_path = os.path.join(root, image_path)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        for key in inputs:
            if hasattr(inputs[key], 'to'):
                inputs[key] = inputs[key].to(device)
        # Store preprocessed data
        preprocessed_data.append({
            'inputs': inputs,
        })
    
    print(f"Successfully preprocessed {len(preprocessed_data)} samples")
    return preprocessed_data


def get_last_hidden_state(model, preprocessed_data, mask_cfg=None, device='cuda:0'):
    """Get hidden states from specified layer of the model, using preprocessed data"""
    with torch.no_grad():
        last_hidden_states = []
        
        # Set up attention masker
        masker = None
        if mask_cfg is not None and 'head_mask' in mask_cfg:          
            masker = AttentionMasker(
                model=model,
                head_mask=mask_cfg.get('head_mask', {}),
                mask_type=mask_cfg.get('mask_type', 'scale_mask'),
                scale_factor=mask_cfg.get('scale_factor', 1e-5)
            )
            masker.apply_hooks()
        
        try:
            # Process preprocessed data
            for sample in tqdm(preprocessed_data, desc="Processing samples"):
                inputs = sample['inputs']
                
                for key in inputs:
                    if hasattr(inputs[key], 'to') and inputs[key].device.type != device.split(':')[0]:
                        inputs[key] = inputs[key].to(device)
                
                # Get model outputs
                outputs = model(**inputs, output_hidden_states=True)
                last_hs = outputs.hidden_states[-1]


                last_hidden_states.append(last_hs[:, -1, :].to(torch.float32))
            
            # Combine results from all samples
            if last_hidden_states:
                combined_states = torch.cat(last_hidden_states, dim=0)
                return combined_states
            else:
                print("Warning: No hidden states collected")
                return torch.tensor([])
            
        finally:
            # Ensure masker is removed
            if masker is not None:
                masker.remove_hooks()

def update_mask_cfg(mask_cfg, layer, head):
    """Update mask configuration"""
    now_mask_key = (layer, head)
    new_mask_cfg = deepcopy(mask_cfg)
    if 'head_mask' not in new_mask_cfg:
        new_mask_cfg['head_mask'] = {}
    new_mask_cfg['head_mask'][now_mask_key] = mask_cfg['mask_qkv']
    return new_mask_cfg



def safety_head_attribution(model_path, data_path, root, storage_path=None, 
                          search_cfg=None, device='cuda:0'):
    """Analyze and identify safety attention heads"""
    # 默认配置
    if search_cfg is None:
        search_cfg = {
            "mask_qkv": ['q'],
            "scale_factor": 1e-5,
            "mask_type": "scale_mask"
        }
    
    # Load model and data
    print("加载模型和数据...")
    processor, model = get_qwen2_model(model_path, device)
    data = pd.read_csv(data_path)
    
    # Get model configuration
    layer_nums = 28  # Qwen2-VL-7B has 28 layers
    head_nums = 28   # Qwen2-VL-7B has 28 heads per layer
    
    # Preprocess data
    print("Preprocessing data...")
    preprocessed_data = preprocess_data(processor, data, root, device)
    
    if len(preprocessed_data) == 0:
        print("Error: No data successfully preprocessed, please check data path and image path")
        return {}
    
    # Get baseline hidden states (last layer)
    print("Computing baseline hidden states...")
    base_lhs = get_last_hidden_state(model, preprocessed_data, None, device)
    
    
    angle_dict = {}    
    # Iterate through all layers and heads
    for layer in tqdm(range(layer_nums), desc="Processing layers"):
        for head in range(head_nums):
            if 'head_mask' in search_cfg and (layer, head) in search_cfg['head_mask']:
                continue
                
            # Calculate impact of current head
            now_mask_cfg = update_mask_cfg(search_cfg, layer, head)
            last_hs = get_last_hidden_state(model, preprocessed_data, now_mask_cfg, device)
            
            # Calculate subspace similarity
            angle = compute_consine_similarity(base_lhs, last_hs)
            angle_dict[(layer, head)] = angle
            print(f"Layer {layer}, Head {head}: Deflection Angle = {angle:.2f}")

    if storage_path:
        model_name = os.path.basename(args.model_path)
        storage_path = os.path.join(storage_path, model_name)
        os.makedirs(storage_path, exist_ok=True)
        csv_basename = os.path.basename(args.csv_path)
        csv_name = os.path.splitext(csv_basename)[0]
        simple_result_file = os.path.join(storage_path, f"{csv_name}.json")
        with open(simple_result_file, 'w') as f:
            json.dump(
                {f"{k[0]}-{k[1]}": v for k, v in angle_dict.items()}, 
                f, 
                indent=2
            )
        print(f"Saved simplified results to {simple_result_file}")
    
    return angle_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Qwen2-VL Safety Head Attribution Analysis")
    parser.add_argument("--model_path", type=str, default="/path/to/your/model",
                        help="Path to the model")
    parser.add_argument("--image_path", type=str, default="/path/to/your/images",
                        help="Directory for image files")
    parser.add_argument("--csv_path", type=str, default="/path/to/your/dataset.csv",
                        help="Path to the dataset CSV file")
    parser.add_argument("--storage_path", type=str, default="angle/",
                        help="Path to save the results")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to run the model on (e.g., cuda:0, cpu)")
    args = parser.parse_args()
    # Configuration
    search_config = {          
        "mask_qkv": ['q'],         
        "scale_factor": 1e-5,      
        "mask_type": "scale_mask"  
    }
    
    # Run safety head analysis
    results = safety_head_attribution(
        model_path=args.model_path,
        csv_path=args.csv_path,
        storage_path=args.storage_path,
        search_cfg=search_config,
        device=args.device
    )