
from transformers import AutoTokenizer,AutoProcessor,Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
processor = AutoProcessor.from_pretrained("/home/wmk/code/model_weight/Qwen2.5-VL-7B-Instruct")
data_0 = "five"
data_1 = "four"
data_2 = "three"
data_3 = "two"
data_4 = "one"
#data = [3040,52670, 34024, 27856, 19789, 603]
tokenizer = AutoTokenizer.from_pretrained("/home/wmk/code/model_weight/Qwen2.5-VL-7B-Instruct")
#print(tokenizer.decode(data))
print(tokenizer.encode(data_0))
print(tokenizer.encode(data_1))
print(tokenizer.encode(data_2))
print(tokenizer.encode(data_3))
print(tokenizer.encode(data_4))
# data=[785, 4271, 315, 279, 2168, 374, 3040, 13, 151645, 198]
# print(tokenizer.decode(data))
#model = Qwen2_5_VLForConditionalGeneration.from_pretrained("/home/wmk/code/model_weight/Qwen2.5-VL-7B-Instruct")
# messages = [{
#             "role": "user",
#             "content": [
#                 {"type": "image", "image": "/home/lrz/Q-Align-main/playground/DIQA-5000_phase1/val/res/val_res_00001.jpg"},
#                 {"type": "text", "text": "You are an image quality expert.Rate the score of the image."},
#             ]
#         }]
# text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# print(text)
# image_inputs, video_inputs = process_vision_info(messages)
# inputs = processor(text=[text],images=image_inputs,videos=video_inputs,padding=True,return_tensors="pt",)
# outputs = model(input_ids = inputs.input_ids,pixel_values = inputs.pixel_values,image_grid_thw = inputs.image_grid_thw)
# print(inputs)
# print(outputs.logits)
#print(processor.tokenizer.encode("five")[-1])
import argparse
import torch
import json
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig
)
from peft import PeftModel
import time
from concurrent.futures import ThreadPoolExecutor

def wa5(logits):
    keys = ["five", "four", "three", "two", "one"]
    logprobs = np.array([logits[k] for k in keys])
    max_log = np.max(logprobs)
    probs = np.exp(logprobs - max_log) / np.sum(np.exp(logprobs - max_log))  # 数值稳定
    return np.inner(probs, np.array([1, 0.75, 0.5, 0.25, 0.])) * 5

def load_model(args):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    ) if args.quant else None

    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model if args.use_peft else args.model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        quantization_config=quant_config,
        attn_implementation="flash_attention_2"  # 启用Flash Attention
    )
    
    if args.use_peft:
        model = PeftModel.from_pretrained(base_model, args.model_path,torch_dtype=torch.bfloat16)
        model = model.merge_and_unload()  # 合并LoRA权重提升推理速度
    return model.eval()


def main(args):
    # 初始化
    start_time = time.time()
    processor = AutoProcessor.from_pretrained(
        args.base_model if args.use_peft else args.model_path,
        trust_remote_code=True
    )
    model = load_model(args)
    print(model.config)
    # 加载数据
    with open("/home/lrz/Q-Align-main/playground/data/converted_dataset_val.json") as f:
        data = json.load(f)
    data = data[:16]
    # 提示词模板
    prompt_template = "You are an image quality expert. Could you evaluate the quality of this image? Only use five, four, three, two, one to evaluate." 
    messages = [
            {
            "role": "user",
            "content": [
                {"type": "image", 
                 "image": f"/home/lrz/Q-Align-main/playground/DIQA-5000_phase1/val/res/{data[0]['image']}",
                 },
                {"type": "text", "text": prompt_template},
            ],
        },
        {
            "role": "assistant",
            "content": "The quality of the image is"
        },
        ]      
    text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
    image_inputs, video_inputs = process_vision_info(messages)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
        inputs = inputs.to("cuda")
        outputs = model(input_ids=inputs.input_ids,pixel_values=inputs.pixel_values,image_grid_thw=inputs.image_grid_thw)
        print(torch.argmax(outputs.logits[:,-1]))
        id = torch.argmax(outputs.logits[:,-1])
    response = processor.decode([id], skip_special_tokens=True)
    print(response.split("assistant")[-1].strip())               
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/lrz/Qwen/output/model/test_fft_lora_overall/checkpoint-219",
                     )
    parser.add_argument("--base-model", type=str, 
                      default="/home/wmk/code/model_weight/Qwen2.5-VL-7B-Instruct",
                      )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use-peft", default=True, 
                      )
    parser.add_argument("--quant", action="store_true", 
                     )
    args = parser.parse_args()
    
    main(args)