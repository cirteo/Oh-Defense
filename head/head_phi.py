import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import pandas as pd
import os
import json
from tqdm import tqdm
from copy import deepcopy
from transformers import AutoModelForCausalLM,AutoProcessor 
from utils.similarity import compute_consine_similarity
from utils.masker import PhiAttentionMasker as AttentionMasker
from PIL import Image
import argparse

def get_phi_model(model_id, device='cuda:0'):

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="cuda",
        trust_remote_code=True
    )
    
    return model, processor


def preprocess_data(processor, data, root, device='cuda:0'):
    """预处理数据，缓存输入格式化的结果"""
    preprocessed_data = []
    
    for i, r in tqdm(data.iterrows(), total=len(data), desc="Preprocessing data"):
        # Prepare input
        prompt = r['jailbreak_query']
        image_path = r['image_path']
        image_path = os.path.join(root, image_path)
        
        image = Image.open(image_path)
        messages = [
            {"role": "user", "content": f"<|image_1|>\n"+ prompt }
        ]
        prompt = processor.tokenizer.apply_chat_template(messages,
                                          tokenize=False,
                                          add_generation_prompt=True,)
        inputs  = processor(prompt,image,return_tensors="pt").to(device)
        
        # Store preprocessed data
        preprocessed_data.append({
            'inputs': inputs,
        })
    
    print(f"Successfully preprocessed {len(preprocessed_data)} samples")
    return preprocessed_data


def get_last_hidden_state(model,preprocessed_data, mask_cfg=None):
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
                
                # Get model outputs
                outputs = model(**inputs, 
                               output_hidden_states=True)
                

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





def safety_head_attribution(model_id, data_path, root, storage_path=None, 
                          search_cfg=None, device='cuda:0'):
    """Analyze and identify safety attention heads"""
    # Default configuration
    if search_cfg is None:
        search_cfg = {
            "mask_qkv": ['q'],
            "scale_factor": 1e-5,
            "mask_type": "scale_mask"
        }
    
    # Load model and data
    print("Loading model and data...")
    model, processor = get_phi_model(model_id, device)
    data = pd.read_csv(data_path)
    
    # Get model configuration
    layer_nums = 32  
    head_nums = 32   
    
    # Preprocess data
    print("Preprocessing data...")
    preprocessed_data = preprocess_data(processor, data, root, device)
    
    if len(preprocessed_data) == 0:
        print("Error: No data successfully preprocessed, please check data path and image path")
        return {}
    
    # Get baseline hidden states (last layer)
    print("Computing baseline hidden states...")
    base_lhs = get_last_hidden_state(model,preprocessed_data, None, device)
    

    
    
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
    parser = argparse.ArgumentParser(description="PhiVision Safety Head Attribution Analysis")
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