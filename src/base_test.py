import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.config.config import settings

# =========================
# 설정
# =========================
MODEL_PATH = settings.MODEL_NAME

PROMPT = """
면접 질문과 면접 답변을 함께 분석하여 STAR 구조와 답변의 강점 및 개선점을 작성한다.

## STAR 분석

실제 경험:
- situation: 답변에서 확인되는 경험의 배경과 상황
- task: 맡은 역할, 해결해야 할 문제 또는 목표
- action: 답변자가 실제 수행한 행동과 과정
- result: 행동 이후 확인되는 결과, 변화 또는 영향

가상 상황:
- situation: 질문에서 제시한 가정 상황
- task: 해당 상황에서 수행해야 할 목표 또는 역할
- action: 답변자가 제시한 대응 방법과 행동
- result: 답변에서 확인되는 결과만 작성

## 작성 규칙
- 질문과 답변에서 확인되는 내용만 작성한다.
- 답변에 없는 사실, 경험, 수치, 성과, 행동, 결과를 생성하거나 추론하지 않는다.
- situation, task, action, result 중 근거가 없는 요소는 반드시 "답변에서 확인되지 않음"으로 작성한다.
- 일부 근거만 확인되면 확인되는 내용만 작성한다.
- situation과 task는 간결한 명사형으로 작성한다.
- action과 result는 문장 형태의 함체로 작성하고 "~함"으로 끝낸다.
- strengths와 improvements는 최소 1개씩 작성하고, 구체적인 문장으로 작성하며 "~함"으로 끝낸다.
- 모든 내용은 간결하게 작성한다.
- JSON 외의 설명이나 마크다운은 출력하지 않는다.

## 출력 형식
{
  "star": {
    "situation": "",
    "task": "",
    "action": "",
    "result": ""
  },
  "strengths": [],
  "improvements": []
}

### 면접 질문:

### 면접 답변:
"""

MAX_NEW_TOKENS = 512


# =========================
# 모델 로드
# =========================
print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map={"": 0},
    trust_remote_code=True,
)

model.eval()

print("Model loaded.")
print(f"Device: {next(model.parameters()).device}")


# =========================
# 입력 준비
# =========================
messages = [
    {
        "role": "user",
        "content": PROMPT,
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)

inputs = tokenizer(
    text,
    return_tensors="pt",
).to(model.device)


# =========================
# Warm-up
# =========================
print("\nWarm-up inference...")

with torch.inference_mode():
    _ = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
    )

if torch.cuda.is_available():
    torch.cuda.synchronize()


# =========================
# 실제 추론 시간 측정
# =========================
print("\nMeasuring inference time...")

if torch.cuda.is_available():
    torch.cuda.synchronize()

start_time = time.perf_counter()

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )

if torch.cuda.is_available():
    torch.cuda.synchronize()

elapsed = time.perf_counter() - start_time


# =========================
# 결과 출력
# =========================
input_length = inputs["input_ids"].shape[1]
output_length = outputs.shape[1]
generated_tokens = output_length - input_length

print("\n" + "=" * 60)
print("BASE MODEL INFERENCE RESULT")
print("=" * 60)

print(f"Input tokens      : {input_length}")
print(f"Generated tokens  : {generated_tokens}")
print(f"Inference time    : {elapsed:.3f} sec")

if generated_tokens > 0:
    print(f"Generation speed  : {generated_tokens / elapsed:.2f} tokens/sec")

print("\n--- Output ---")

result = tokenizer.decode(
    outputs[0][input_length:],
    skip_special_tokens=True,
)

print(result)

print("=" * 60)