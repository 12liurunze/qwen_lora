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
    print(logits)
    keys = [" five", " four", " three", " two", " one"]
    logprobs = np.array([logits[k] for k in keys])
    print(f"logprobs:{logprobs}")
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
        device_map="cuda:1",
        trust_remote_code=True,
        torch_dtype = torch.bfloat16,
        quantization_config=quant_config,
        attn_implementation="flash_attention_2"  # 启用Flash Attention
    )
    
    if args.use_peft:
        model = PeftModel.from_pretrained(base_model, args.model_path,torch_dtype=torch.bfloat16)
        model = model.merge_and_unload()  # 合并LoRA权重提升推理速度
    return model.eval()

def preprocess_item(item, processor, prompt_template):
    """并行预处理单个样本"""
    try:
        messages = [
            {
            "role": "user",
            "content": [
                {"type": "image", 
                 "image": f"/home/lrz/Q-Align-main/playground/DIQA-5000_phase1/train/res/{item['image']}",
                 },
                {"type": "text", "text": prompt_template},
            ],
        },
        {
            "role": "assistant",
            "content": "The quality of the image is"
        },
        ]
        text = processor.apply_chat_template(messages, tokenize=False,add_generation_prompt=False)
        if text.endswith('<|im_end|>\n'):
            cleaned_output = text[:-11]
        else:
            cleaned_output = text
        image = Image.open(messages[0]['content'][0]['image'])

        return cleaned_output, image, item['image']
    except Exception as e:
        print(f"Error processing {item['image']}: {str(e)}")
        return None

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
    with open("/home/lrz/Qwen/data/train/train_diqa_overall_quality.json") as f:
        data = json.load(f)
    data = data[:20]
    # 提示词模板
    prompt_template = "You are an image quality expert. Could you evaluate the quality of this image?" 
                     
                     
    
    preprocessed_data = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(preprocess_item, item, processor, prompt_template) for item in data]
        
        for future in tqdm(futures, desc="Preprocessing"):
            result = future.result()
            if result: preprocessed_data.append(result)
    
    
    # 批量推理
    results = []
    batch_size = 1  
    token_ids = {k: processor.tokenizer.encode(k)[-1] for k in [" five", " four", " three", " two", " one"]}
    print(token_ids)
    for i in tqdm(range(0, len(preprocessed_data), batch_size), desc="Evaluating"):
        batch = preprocessed_data[i:i+batch_size]
        texts, images, img_names = zip(*batch)
        
        try:
            # 批量处理
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                inputs = processor(
                    text=list(texts),
                    images=list(images),
                    padding=True,
                    return_tensors="pt",
                    truncation=True
                ).to(args.device)
                with torch.no_grad():
                    outputs = model(input_ids = inputs.input_ids,pixel_values = inputs.pixel_values,image_grid_thw = inputs.image_grid_thw)
                    logits = outputs.logits[:, -1]
                    print(logits)  # 取最后一个token
                    # 批量计算分数
                    for j in range(len(batch)):
                        scores = {k: logits[j, token_ids[k]].item() for k in token_ids}
                        results.append({
                            "image": img_names[j],
                            "overall": wa5(scores),
                            
                        })
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            batch_size = max(batch_size // 2, 1)
            print(f"OOM detected, reducing batch_size to {batch_size}")
            continue
    
    # 保存结果
    with open("/home/lrz/Qwen/output/data/overall_train.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation completed in {(time.time()-start_time)/60:.2f} minutes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/lrz/Qwen/output/model/test_fft_lora_overall/checkpoint-219")
    parser.add_argument("--base-model", type=str, default="/home/wmk/code/model_weight/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--use-peft", default=True)
    parser.add_argument("--quant", action="store_true")
    args = parser.parse_args()
    main(args)