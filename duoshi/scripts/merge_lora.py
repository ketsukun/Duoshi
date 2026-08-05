import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

# 路径配置
BASE_MODEL_PATH = os.path.join(ROOT_DIR, "models/qwen3-4b")
ADAPTER_PATH = os.path.join(ROOT_DIR, "models/fine_tuned/qwen3-4b-ft")
MERGED_OUTPUT_DIR = os.path.join(ROOT_DIR, "models/fine_tuned/merged_qwen3-4b")

# 检查路径
if not os.path.isdir(BASE_MODEL_PATH):
    logger.error(f"Base model not found at {BASE_MODEL_PATH}.")
    logger.info("Please download the original model using:")
    logger.info("  huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 --local-dir ./models/qwen3-4b-base --local-dir-use-symlinks False")
    sys.exit(1)

if not os.path.isdir(ADAPTER_PATH):
    logger.error(f"Adapter not found at {ADAPTER_PATH}.")
    sys.exit(1)

# 加载基座
logger.info(f"Loading base model from {BASE_MODEL_PATH}...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载 LoRA 适配器
logger.info(f"Loading adapter from {ADAPTER_PATH}...")
peft_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

logger.info("Merging LoRA weights...")
merged_model = peft_model.merge_and_unload()  

# 保存合并后的模型
logger.info(f"Saving merged model to {MERGED_OUTPUT_DIR}...")
os.makedirs(MERGED_OUTPUT_DIR, exist_ok=True)
merged_model.save_pretrained(MERGED_OUTPUT_DIR, safe_serialization=True)
tokenizer.save_pretrained(MERGED_OUTPUT_DIR)

logger.info("Merge completed successfully!")
logger.info(f"Merged model saved to {MERGED_OUTPUT_DIR}")