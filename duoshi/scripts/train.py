import json
import os
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

MODEL_NAME = "./models/qwen3-4b"
TRAIN_FILE = os.path.join(ROOT_DIR, "data/processed/train.json")
VAL_FILE = os.path.join(ROOT_DIR, "data/processed/val.json")
OUTPUT_DIR = os.path.join(ROOT_DIR, "models/fine_tuned/qwen3-4b-ft")  

# 超参数
MAX_SEQ_LEN = 2048
SYSTEM_PROMPT = "你是一个擅长处理诗歌与典故的专家模型。"

USE_LORA = True
USE_4BIT = True
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

EPOCHS = 5
BATCH_SIZE = 4
GRAD_ACCUM = 4
LEARNING_RATE = 2e-4
WARMUP_STEPS = 50
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 10
SAVE_STEPS = 400          
EVAL_STEPS = 200
SAVE_TOTAL_LIMIT = 2
EARLY_STOP_PATIENCE = 3
FP16 = True

# 数据集（手动构建 labels）
class InstructDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length, system_prompt):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.system_part = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n"
        self.assistant_start = "<|im_start|>assistant\n"
        self.end_token = "<|im_end|>"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        user_content = item['instruction'] + "\n" + item['input']
        assistant_content = item['output']

        prompt = self.system_part + user_content + f"<|im_end|>\n{self.assistant_start}"
        response = assistant_content + self.end_token

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False, truncation=False)
        response_ids = self.tokenizer.encode(response, add_special_tokens=False, truncation=False)

        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids

        if len(input_ids) > self.max_length:
            excess = len(input_ids) - self.max_length
            if excess < len(prompt_ids):
                prompt_ids = prompt_ids[excess:]
                input_ids = prompt_ids + response_ids
                labels = [-100] * len(prompt_ids) + response_ids
            else:
                input_ids = input_ids[-self.max_length:]
                labels = labels[-self.max_length:]

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

# 数据整理器（解决变长序列问题）
def collate_fn(batch, tokenizer):

    max_len = max(len(item["input_ids"]) for item in batch)
    padded_batch = {"input_ids": [], "attention_mask": [], "labels": []}

    for item in batch:
        pad_len = max_len - len(item["input_ids"])

        padded_batch["input_ids"].append(item["input_ids"] + [tokenizer.pad_token_id] * pad_len)

        padded_batch["attention_mask"].append(item["attention_mask"] + [0] * pad_len)

        padded_batch["labels"].append(item["labels"] + [-100] * pad_len)

    return {k: torch.tensor(v) for k, v in padded_batch.items()}

# 主训练函数
def main():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16 if FP16 else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

    if USE_LORA:
        if USE_4BIT:
            model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    train_dataset = InstructDataset(TRAIN_FILE, tokenizer, MAX_SEQ_LEN, SYSTEM_PROMPT)
    eval_dataset = None
    if os.path.exists(VAL_FILE):
        eval_dataset = InstructDataset(VAL_FILE, tokenizer, MAX_SEQ_LEN, SYSTEM_PROMPT)

    data_collator = lambda batch: collate_fn(batch, tokenizer)

    # 训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=FP16,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        eval_strategy="steps" if eval_dataset else "no",
        save_strategy="steps",
        logging_strategy="steps",
        prediction_loss_only=True,
    )

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=EARLY_STOP_PATIENCE,
        early_stopping_threshold=0.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[early_stopping] if eval_dataset else None,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info(f"训练完成，模型已保存至 {OUTPUT_DIR}")

if __name__ == "__main__":
    main()