# 注：该测试使用的是训练好的lora权重+原模型，而非合并好的模型
# 测试时需使用val.json中的数据

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import threading
import logging

logging.basicConfig(level=logging.ERROR)

# 配置
MODEL_PATH = "./models/fine_tuned/qwen3-4b-ft" 
SYSTEM_PROMPT = "你是一个擅长处理诗歌与典故的专家模型。"

MAX_NEW_TOKENS = 1024        
TEMPERATURE = 0.7
TOP_P = 0.9
DO_SAMPLE = True

# 加载模型
print("正在加载模型，请稍候...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
model.eval()
print("模型加载完成！\n")

# 交互循环
print("=" * 60)
print("模型测试".center(60))
print("=" * 60)
print("输入您的问题，输入 /quit 退出。")
print("-" * 60)

while True:
    print("\n用户输入：")
    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        if line.strip() == "/quit":
            print("退出程序。")
            exit(0)
        lines.append(line)
    user_input = "\n".join(lines).strip()
    if not user_input:
        continue

    system_part = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"
    input_text = system_part + user_input + f"<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,          
        skip_special_tokens=True,  
    )

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=1.1,    
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    print("\n模型输出：", end="", flush=True)
    for new_text in streamer:
        print(new_text, end="", flush=True)
    print("\n")  

    thread.join()  
    print("-" * 60)