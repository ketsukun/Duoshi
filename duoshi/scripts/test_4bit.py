import os
import sys
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
import threading
import logging
from rag_retriever import get_retriever, clean_query, get_allusion_confidence

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.basicConfig(level=logging.ERROR)

# 配置
MODEL_PATH = "./models/fine_tuned/merged_qwen3-4b"
SYSTEM_PROMPT = "你是一个擅长处理诗歌与典故的专家模型。"
MAX_NEW_TOKENS_EXPLAIN = 768
TEMPERATURE = 0.7
TOP_P = 0.9
MIN_CONFIDENCE = 0.5
MIN_SCORE_MARGIN = 0.05
TOP_K_CANDIDATES = 5

retriever = get_retriever()

print("正在加载模型（4-bit NF4 量化，GPU+CPU 混合运行）...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    fix_mistral_regex=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Qwen 对话以 <|im_end|> 结束，同时保留 <|endoftext|> 作为兼容终止符。
EOS_TOKEN_IDS = [
    tokenizer.convert_tokens_to_ids("<|im_end|>"),
    tokenizer.eos_token_id,
]

# 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    max_memory={0: "5GiB", "cpu": "10GiB"},
    trust_remote_code=True,
)
model.eval()
print("模型加载完成！\n")

# 辅助函数（约束模型只在合法编号中选择）
def select_candidate(prompt, candidate_count):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    option_token_ids = {}
    for option in range(candidate_count + 1):
        token_ids = tokenizer.encode(str(option), add_special_tokens=False)
        if not token_ids:
            raise ValueError(f"候选编号 {option} 编码结果为空。")
        option_token_ids[option] = token_ids

    # 先计算共同提示词，再逐 token 计算各合法编号的平均对数概率。
    # 这样既不会生成越界编号，也兼容“10”一类被拆成多个 token 的编号。
    with torch.no_grad():
        prompt_outputs = model(**inputs, use_cache=True)
        first_log_probs = F.log_softmax(prompt_outputs.logits[0, -1].float(), dim=-1)
        option_scores = {}

        for option, token_ids in option_token_ids.items():
            token_scores = [first_log_probs[token_ids[0]].item()]
            past_key_values = prompt_outputs.past_key_values

            for previous_token, current_token in zip(token_ids, token_ids[1:]):
                token_input = torch.tensor([[previous_token]], device=model.device)
                outputs = model(
                    input_ids=token_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                log_probs = F.log_softmax(outputs.logits[0, -1].float(), dim=-1)
                token_scores.append(log_probs[current_token].item())
                past_key_values = outputs.past_key_values

            option_scores[option] = sum(token_scores) / len(token_scores)

    return max(option_scores, key=option_scores.get)

# 主交互循环
print("=" * 60)
print("RAG 典故解析系统（4-bit 量化版）".center(60))
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
    # FAISS 仍召回诗例，模型只接收得分最高的 5 个候选进行编号选择。
    candidates = retriever.retrieve(query_text)[:TOP_K_CANDIDATES]
    if not candidates:
        print("未找到匹配的典故（相似度低于阈值），请尝试其他诗句。")
        continue

    # 置信度和候选分差需根据验证集继续调优
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

    # 第2步：模型在合法编号中选择典故
    select_prompt = (
        f"用户诗句：\n{query_text}\n\n"
        f"检索到以下候选典故：\n{candidate_list}\n"
        "请判断用户诗句最可能使用了上述哪个典故。选择对应编号；如果都不匹配，选择0。只考虑编号，不要解释。"
    )
    select_full = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{select_prompt}<|im_end|>\n<|im_start|>assistant\n"
    print("模型正在选择最匹配的典故...", end="", flush=True)
    try:
        chosen_num = select_candidate(select_full, len(candidates))
    except ValueError as error:
        print(f"失败：{error}")
        continue
    print("完成。")

    if chosen_num == 0:
        print("无法确定典故。")
        continue
    chosen_idx = chosen_num - 1
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
    # 原文和释义由程序直接写入，避免生成模型对 RAG 字段做任何改写。
    rag_output = (
        f"使用典故：{chosen['allusion']['allusion_name']}\n"
        f"原文：{chosen['allusion']['source_text']}\n"
        f"释义：{chosen['allusion']['allusion_mean']}\n"
        "解析："
    )
    explain_user_content = (
        f"RAG 检索到的典故信息：\n{rag_info}\n\n"
        f"用户诗句：\n{query_text}\n\n"
        "回答要求：只解释该典故在用户诗句中的具体用法，不要再次输出典故名称、原文或释义，"
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
        repetition_penalty=1.15,
        no_repeat_ngram_size=6,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=EOS_TOKEN_IDS,
        streamer=streamer,
    )
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    print(f"\n模型解析：{rag_output}", end="", flush=True)
    for new_text in streamer:
        if not new_text:
            continue
        print(new_text, end="", flush=True)
    print("\n")
    thread.join()

    print("-" * 60)
