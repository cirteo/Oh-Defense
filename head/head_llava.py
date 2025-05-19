import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import pandas as pd
import json
from tqdm import tqdm
from copy import deepcopy
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.conversation import conv_templates
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
import argparse
from utils.similarity import compute_consine_similarity
from utils.masker import AttentionMasker 
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)



def get_llava_model(model_path, device='cuda:0'):
    """Load LLaVA model"""
    model_name = "llava_v1"
    model_base = None
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, model_base, model_name
    )

    model = model.to(device)
    conversation_template = conv_templates[model_name].copy()
    return tokenizer, model, image_processor, conversation_template


def preprocess_data(model, tokenizer, image_processor, conversation_template, data, device='cuda:0'):
    """Preprocess data, cache image tensors and input_ids"""
    preprocessed_data = []
    
    for i, r in tqdm(data.iterrows(), total=len(data), desc="Preprocessing data"):
        # Prepare conversation template
        prompt = r['jailbreak_query']
        if hasattr(model.config, 'mm_use_im_start_end') and model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        conv = conversation_template.copy()
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        query = conv.get_prompt()
        
        # Process image
        image_path = os.path.join(args.image_path, r['image_path'])
        
        image = Image.open(image_path)
        image_tensor = image_processor(image, return_tensors="pt")['pixel_values'].unsqueeze(0).half().to(device)
        input_ids = tokenizer_image_token(query, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        
        # Store preprocessed data
        preprocessed_data.append({
            'image_tensor': image_tensor,
            'input_ids': input_ids
        })
        
    
    print(f"Successfully preprocessed {len(preprocessed_data)} samples")
    return preprocessed_data


def get_last_hidden_state(model, preprocessed_data, mask_cfg=None):
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
                image_tensor = sample['image_tensor']
                input_ids = sample['input_ids']
                
                with torch.inference_mode():
                    # Get model outputs
                    outputs = model(
                        input_ids=input_ids,
                        images=image_tensor,
                        output_hidden_states=True,
                    )
                    hidden_state = outputs.hidden_states[-1]      

                last_hidden_states.append(hidden_state[:, -1, :].to(torch.float32))
            
            # Combine results from all samples
            if last_hidden_states:
                last_hidden_states = torch.cat(last_hidden_states, dim=0)
            else:
                print("Warning: No hidden states collected")
                last_hidden_states = torch.tensor([])
            
            return last_hidden_states
            
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


def safety_head_attribution(model_path, csv_path, storage_path=None, 
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
    tokenizer, model, image_processor, conversation_template = get_llava_model(model_path, device)
    data = pd.read_csv(csv_path)
    
    # Get model configuration
    layer_nums = len(model.model.layers)
    head_nums = model.config.num_attention_heads
    
    # Preprocess data
    print("Preprocessing data...")
    preprocessed_data = preprocess_data(model, tokenizer, image_processor, conversation_template, data, device)
    
    if len(preprocessed_data) == 0:
        print("Error: No data successfully preprocessed, please check data path and image path")
        return {}
    
    # Get baseline hidden states (last layer)
    print("Computing baseline hidden states...")
    base_lhs = get_last_hidden_state(
        model, 
        preprocessed_data, 
        None,  # No mask applied
    )


    angle_dict = {}

    
    # Iterate through all layers and heads
    for layer in tqdm(range(layer_nums), desc="Processing layers"):
        for head in range(head_nums):
            if 'head_mask' in search_cfg and (layer, head) in search_cfg['head_mask']:
                    continue
            # Calculate impact of current head
            now_mask_cfg = update_mask_cfg(search_cfg, layer, head)
            last_hs = get_last_hidden_state(
                model, 
                preprocessed_data, 
                now_mask_cfg, 
            )
            
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
    parser = argparse.ArgumentParser(description="LLaVA Safety Head Attribution Analysis")
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