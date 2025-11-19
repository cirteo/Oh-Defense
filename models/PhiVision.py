import torch
import transformers
from transformers import AutoModelForCausalLM, AutoProcessor 

from PIL import Image
import warnings
import models.attack_model as attack_model

# disable some warnings
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
warnings.filterwarnings('ignore')


class PhiVision(attack_model.attack_model):
    def __init__(self, device, model_path):
        self.device = device  # or cpu
        torch.set_default_device(self.device)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map='auto',trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True)

    def inference(self, image_path, prompt):
        if image_path is None:
            messages = [
                {"role": "user", "content": f"<|image_1|>\n"+ prompt }
            ]
            prompt = self.processor.tokenizer.apply_chat_template(messages,
                                            tokenize=False,
                                            add_generation_prompt=True,)
            inputs  = self.processor(prompt,None,return_tensors="pt").to(self.device)
        else:
            image = Image.open(image_path)
            messages = [
                {"role": "user", "content": f"<|image_1|>\n"+ prompt }
            ]
            prompt = self.processor.tokenizer.apply_chat_template(messages,
                                            tokenize=False,
                                            add_generation_prompt=True,)
            inputs  = self.processor(prompt,image,return_tensors="pt").to(self.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=200,
            eos_token_id=self.processor.tokenizer.eos_token_id, 
        )
        response = self.processor.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        return response
