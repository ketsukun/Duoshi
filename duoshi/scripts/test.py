# 注：该测试使用的是合并好的模型，集成了rag，可直接交互测试

import os
import sys
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import threading
import logging
from rag_retriever import get_retriever, clean_query, get_allusion_confidence

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.basicConfig(level=logging.ERROR)

# 配置
MODEL_PATH = "./models/fine_tuned/merged_qwen3-4b"   
SYSTEM_PROMPT = "你是一个擅长处理诗歌与典故的专家模型。"
MAX_NEW_TOKENS_SELECT = 16      
MAX_NEW_TOKENS_EXPLAIN = 768
TEMPERATURE = 0.7
TOP_P = 0.9
MIN_CONFIDENCE = 0.5
MIN_SCORE_MARGIN = 0.05
TOP_K_CANDIDATES = 5

retriever = get_retriever()

print("正在加载模型（FP16，GPU+CPU 混合运行）...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

EOS_TOKEN_IDS = [
    tokenizer.convert_tokens_to_ids("<|im_end|>"),
    tokenizer.eos_token_id,
]

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    max_memory={0: "5GiB", "cpu": "10GiB"}, 
    trust_remote_code=True,
)
model.eval()
print("模型加载完成！\n")

# 辅助函数（选择典故）
def generate_non_stream(prompt, max_new_tokens=MAX_NEW_TOKENS_SELECT):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,           
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=EOS_TOKEN_IDS,
        )
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()

# 主交互循环
print("=" * 60)
print("RAG 典故解析系统".center(60))
print("=" * 60)
print("输入诗句（多行，空行提交），输入 /quit 退出")
print("-" * 60)

while True:
    print("\n用户输入：")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        trimmed = line.strip()
        if trimmed == "":
            break
        if trimmed == "/quit":
            print("退出。")
            sys.exit(0)
        lines.append(line)
    user_input = "\n".join(lines).strip()
    if not user_input:
        continue

    # 第1步：清理提问套语并进行 RAG 检索
    query_text = clean_query(user_input)
    candidates = retriever.retrieve(query_text)[:TOP_K_CANDIDATES]
    if not candidates:
        print("未找到匹配的典故（相似度低于阈值），请尝试其他诗句。")
        continue

    top_score, score_margin = get_allusion_confidence(candidates)
    if top_score < MIN_CONFIDENCE:
        print(f"暂无法确定典故（最高相似度 {top_score:.3f} 低于最低置信度 {MIN_CONFIDENCE:.3f}）。")
        continue
    if len(candidates) > 1 and score_margin < MIN_SCORE_MARGIN:
        print(f"暂无法确定典故（与其他典故分差 {score_margin:.3f} 低于最低分差 {MIN_SCORE_MARGIN:.3f}）。")
        continue

    # 构造候选列表
    candidate_list = ""
    for i, cand in enumerate(candidates, 1):
        name = cand['allusion']['allusion_name']
        score = cand['score']          
        matched_poem = cand['matched_poem']
        source_preview = cand['allusion']['source_text'][:50] + "..." if len(cand['allusion']['source_text']) > 50 else cand['allusion']['source_text']
        candidate_list += f"{i}. {name}（相似度：{score:.3f}）\n   匹配诗例：{matched_poem}\n   原文：{source_preview}\n"

    # 第2步：模型选择典故
    select_prompt = (
        f"用户诗句：\n{query_text}\n\n"
        f"检索到以下候选典故：\n{candidate_list}\n"
        "请首先判断用户诗句最可能使用了上述哪个典故，只需输出对应的编号（如“1”），如果上述典故都不匹配，请输出“0”。"
    )

    select_full = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{select_prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    print("模型正在选择最匹配的典故...", end="", flush=True)
    select_response = generate_non_stream(select_full, max_new_tokens=MAX_NEW_TOKENS_SELECT)
    print("完成。")

    match = re.fullmatch(r'\s*(\d+)\s*', select_response)
    if not match:
        print("无法确定典故。")
        continue
    else:
        chosen_num = int(match.group())
        if chosen_num == 0:
            print("无法确定典故。")
            continue  
        elif 1 <= chosen_num <= len(candidates):
            chosen_idx = chosen_num - 1
        else:
            print(f"编号 {chosen_num} 超出范围，暂无法确定典故。")
            continue

    chosen = candidates[chosen_idx]
    chosen_score, chosen_margin = get_allusion_confidence(candidates, chosen_idx)
    if chosen_score < MIN_CONFIDENCE or chosen_margin < MIN_SCORE_MARGIN:
        print("无法确定典故。")
        continue
    print(f"模型选择了：{chosen['allusion']['allusion_name']} (相似度：{chosen['score']:.3f})")

    # 第3步：构造标准提示词并生成解析
    rag_info = (
        f"allusion_name：{chosen['allusion']['allusion_name']}\n"
        f"source_text：{chosen['allusion']['source_text']}\n"
        f"allusion_mean：{chosen['allusion']['allusion_mean']}"
    )

    rag_output = (
        f"使用典故：{chosen['allusion']['allusion_name']}\n"
        f"原文：{chosen['allusion']['source_text']}\n"
        f"释义：{chosen['allusion']['allusion_mean']}\n"
        "解析："
    )
    explain_user_content = (
        f"RAG 检索到的典故信息：\n{rag_info}\n\n"
        f"用户诗句：\n{query_text}\n\n"
        "回答要求：：指出用户诗句使用了/化用了/反用了什么典故，并解释该典故在诗句中的用法。"
        "不要添加“解析：”标题。严格基于 RAG 检索结果，不得添加外部知识。"
        "注意保持语言流畅自然，具有古典诗词赏析的韵味。"
    )
    explain_full = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{explain_user_content}<|im_end|>\n<|im_start|>assistant\n{rag_output}"

    inputs = tokenizer(explain_full, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS_EXPLAIN,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=EOS_TOKEN_IDS,
        streamer=streamer,
    )
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    print(f"\n模型解析：{rag_output}", end="", flush=True)
    for new_text in streamer:
        print(new_text, end="", flush=True)
    print("\n") 
    thread.join()

    print("-" * 60)
