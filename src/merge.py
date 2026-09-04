import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_ID = "Qwen/Qwen3-4B"

ADAPTER_PATH = "outputs/v1/checkpoint-633"  # 수정하기

MERGED_MODEL_PATH = "outputs/v1/merged-qwen3-4b-star"


# Tokenizer 로드
tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_PATH,
)

# Base Model 로드
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# LoRA Adapter 적용
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
)

# LoRA 가중치를 Base Model에 병합
merged_model = model.merge_and_unload()

# 병합된 모델 저장
merged_model.save_pretrained(
    MERGED_MODEL_PATH,
    safe_serialization=True,
)

# Tokenizer 저장
tokenizer.save_pretrained(
    MERGED_MODEL_PATH,
)

print("Merge completed!")
print(f"Merged model saved to: {MERGED_MODEL_PATH}")