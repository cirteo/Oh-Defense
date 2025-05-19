import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from PIL import Image
import warnings
import models.attack_model as attack_model
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.conversation import conv_templates
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

warnings.filterwarnings('ignore')

class LLaVA(attack_model.attack_model):
    def __init__(self, device,model_path):
        self.device = device
        torch.set_default_device(self.device)
        self.model_name = "llava_v1"
        model_base = None
        self.model_path = model_path
        self.tokenizer,self.model,self.image_processor,_ = load_pretrained_model(model_path,model_base,self.model_name)
        self.conversation_template = conv_templates[self.model_name].copy()

    def inference(self, image_path, prompt):
        image = Image.open(image_path)
        if self.model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        conv = self.conversation_template.copy()
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        query = conv.get_prompt()

        image_tensor = self.image_processor(image, return_tensors="pt")['pixel_values']
        image_tensor = image_tensor.unsqueeze(0).half().cuda()
        input_ids = tokenizer_image_token(query,self.tokenizer,IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        with torch.inference_mode():
            output = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    max_new_tokens=200,
                    use_cache=True)
        outputs = self.tokenizer.decode(output[0][len(input_ids)-1:], skip_special_tokens=True)
        
        return outputs
