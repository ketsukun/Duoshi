import os
import sys
import threading
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = ROOT_DIR / "duoshi"
SCRIPTS_DIR = PROJECT_DIR / "scripts"
MODEL_PATH = PROJECT_DIR / "models" / "fine_tuned" / "merged_qwen3-4b"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from rag_retriever import clean_query, get_allusion_confidence, get_retriever


SYSTEM_PROMPT = "你是一个擅长处理诗歌与典故的专家模型。"
MAX_NEW_TOKENS_EXPLAIN = 768
TEMPERATURE = 0.7
TOP_P = 0.9
MIN_CONFIDENCE = 0.5
MIN_SCORE_MARGIN = 0.05
TOP_K_CANDIDATES = 5
OUTPUT_CHARACTERS_PER_SECOND = 16
OUTPUT_STREAM_INTERVAL = 1 / OUTPUT_CHARACTERS_PER_SECOND


class AllusionEngine:
    def __init__(self, mode):
        if mode not in {"fp16", "4bit"}:
            raise ValueError("模型精度必须是 fp16 或 4bit。")

        self.mode = mode
        self.mode_label = "FP16 半精度" if mode == "fp16" else "4-bit NF4 量化"
        self._inference_lock = threading.Lock()

        self.retriever = self._load_retriever()
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_PATH),
            trust_remote_code=True,
            fix_mistral_regex=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.eos_token_ids = [
            self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            self.tokenizer.eos_token_id,
        ]
        self.model = self._load_model()
        self.model.eval()

    def _load_retriever(self):
        previous_dir = Path.cwd()
        try:
            os.chdir(PROJECT_DIR)
            return get_retriever()
        finally:
            os.chdir(previous_dir)

    def _load_model(self):
        common_options = {
            "device_map": "auto",
            "max_memory": {0: "5GiB", "cpu": "10GiB"},
            "trust_remote_code": True,
        }
        if self.mode == "4bit":
            common_options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            common_options["torch_dtype"] = torch.float16

        return AutoModelForCausalLM.from_pretrained(str(MODEL_PATH), **common_options)

    def _select_candidate(self, prompt, candidate_count):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        option_token_ids = {}
        for option in range(candidate_count + 1):
            token_ids = self.tokenizer.encode(str(option), add_special_tokens=False)
            if not token_ids:
                raise ValueError(f"候选编号 {option} 编码结果为空。")
            option_token_ids[option] = token_ids

        with torch.no_grad():
            prompt_outputs = self.model(**inputs, use_cache=True)
            first_log_probs = F.log_softmax(prompt_outputs.logits[0, -1].float(), dim=-1)
            option_scores = {}

            for option, token_ids in option_token_ids.items():
                token_scores = [first_log_probs[token_ids[0]].item()]
                past_key_values = prompt_outputs.past_key_values

                for previous_token, current_token in zip(token_ids, token_ids[1:]):
                    token_input = torch.tensor([[previous_token]], device=self.model.device)
                    outputs = self.model(
                        input_ids=token_input,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    log_probs = F.log_softmax(outputs.logits[0, -1].float(), dim=-1)
                    token_scores.append(log_probs[current_token].item())
                    past_key_values = outputs.past_key_values

                option_scores[option] = sum(token_scores) / len(token_scores)

        return max(option_scores, key=option_scores.get)

    @staticmethod
    def _build_candidate_list(candidates):
        candidate_list = ""
        for index, candidate in enumerate(candidates, 1):
            allusion = candidate["allusion"]
            source_text = allusion["source_text"]
            source_preview = source_text[:50] + "..." if len(source_text) > 50 else source_text
            candidate_list += (
                f"{index}. {allusion['allusion_name']}（相似度：{candidate['score']:.3f}）\n"
                f"   匹配诗例：{candidate['matched_poem']}\n"
                f"   原文：{source_preview}\n"
            )
        return candidate_list

    def _build_selection_prompt(self, query_text, candidates):
        candidate_list = self._build_candidate_list(candidates)
        selection = (
            f"用户诗句：\n{query_text}\n\n"
            f"检索到以下候选典故：\n{candidate_list}\n"
            "请判断用户诗句最可能使用了上述哪个典故。选择对应编号；"
            "如果都不匹配，选择0。只考虑编号，不要解释。"
        )
        return (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{selection}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    @staticmethod
    def _build_explanation_prompt(query_text, chosen):
        allusion = chosen["allusion"]
        rag_output = AllusionEngine._build_rag_output(chosen)
        rag_info = (
            f"allusion_name：{allusion['allusion_name']}\n"
            f"source_text：{allusion['source_text']}\n"
            f"allusion_mean：{allusion['allusion_mean']}"
        )
        content = (
            f"RAG 检索到的典故信息：\n{rag_info}\n\n"
            f"用户诗句：\n{query_text}\n\n"
            "回答要求：：指出用户诗句使用了/化用了/反用了什么典故，并解释该典故在诗句中的用法。"
            "不要添加“解析：”标题。严格基于 RAG 检索结果，不得添加外部知识。"
            "注意保持语言流畅自然，具有古典诗词赏析的韵味。"
        )
        return (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{content}<|im_end|>\n"
            f"<|im_start|>assistant\n{rag_output}"
        )

    @staticmethod
    def _build_rag_output(chosen):

        allusion = chosen["allusion"]
        return (
            f"使用典故：{allusion['allusion_name']}\n"
            f"原文：{allusion['source_text']}\n"
            f"释义：{allusion['allusion_mean']}\n"
            "解析："
        )

    @staticmethod
    def _stream_characters(text):

        for character in text:
            yield {"type": "delta", "content": character}
            time.sleep(OUTPUT_STREAM_INTERVAL)

    def stream_analysis(self, user_input):
        with self._inference_lock:
            query_text = clean_query(user_input)
            yield {"type": "status", "message": "正在检索典故诗例…"}

            candidates = self.retriever.retrieve(query_text)[:TOP_K_CANDIDATES]
            if not candidates:
                yield {
                    "type": "refusal",
                    "message": "未找到足够匹配的典故，请换一句诗再试。",
                }
                return

            top_score, score_margin = get_allusion_confidence(candidates)
            if top_score < MIN_CONFIDENCE or (
                len(candidates) > 1 and score_margin < MIN_SCORE_MARGIN
            ):
                yield {
                    "type": "refusal",
                    "message": "候选典故的置信度或区分度不足，暂时无法确定。",
                }
                return

            yield {"type": "status", "message": "正在判断最匹配的典故…"}
            selection_prompt = self._build_selection_prompt(query_text, candidates)
            chosen_number = self._select_candidate(selection_prompt, len(candidates))
            if chosen_number == 0:
                yield {"type": "refusal", "message": "这些候选均不匹配，暂时无法确定典故。"}
                return

            chosen_index = chosen_number - 1
            chosen = candidates[chosen_index]
            chosen_score, chosen_margin = get_allusion_confidence(candidates, chosen_index)
            if chosen_score < MIN_CONFIDENCE or chosen_margin < MIN_SCORE_MARGIN:
                yield {"type": "refusal", "message": "模型选择的候选证据不足，暂时无法确定典故。"}
                return

            yield {
                "type": "selection",
                "allusion_name": chosen["allusion"]["allusion_name"],
                "score": round(chosen["score"], 4),
                "matched_poem": chosen["matched_poem"],
            }

            explanation_prompt = self._build_explanation_prompt(query_text, chosen)
            inputs = self.tokenizer(
                explanation_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            )
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )
            generation_options = {
                **inputs,
                "max_new_tokens": MAX_NEW_TOKENS_EXPLAIN,
                "do_sample": True,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "repetition_penalty": 1.15,
                "no_repeat_ngram_size": 6,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.eos_token_ids,
                "streamer": streamer,
            }
            generation_thread = threading.Thread(
                target=self.model.generate,
                kwargs=generation_options,
                daemon=True,
            )
            generation_thread.start()

            try:
                rag_output = self._build_rag_output(chosen)
                yield from self._stream_characters(rag_output)

                for text in streamer:
                    if not text:
                        continue
                    yield from self._stream_characters(text)
            finally:
                generation_thread.join()
            yield {"type": "done"}
