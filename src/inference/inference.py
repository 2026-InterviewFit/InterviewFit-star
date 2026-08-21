import json
import time

import pandas as pd

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.inference.prompts.inference_prompt import INFERENCE_SYSTEM_PROMPT, create_inference_prompt
from src.config.config import settings


_model = None
_tokenizer = None

NUM_SAMPLES = 5


def clean_json(response: str):
    response = response.strip()

    if response.startswith("```json"):
        response = response[len("```json"):]
    if response.endswith("```"):
        response = response[:-3]

    return response.strip()


def load_model(model_path):
    global _model, _tokenizer

    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    _tokenizer = AutoTokenizer.from_pretrained(model_path)

    nb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        settings.MODEL_NAME,
        quantization_config=nb_config,
        device_map="cuda",
    )

    # Inference: KV Cache ON
    base_model.config.use_cache = True

    _model = PeftModel.from_pretrained(
        base_model,
        model_path,
    )

    _model.eval()

    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM total:", torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, "GB")
    print("VRAM allocated:", torch.cuda.memory_allocated() / 1024 ** 3, "GB")
    print("VRAM reserved:", torch.cuda.memory_reserved() / 1024 ** 3, "GB")
    print(_model.hf_device_map)

    return _model, _tokenizer


def build_messages(question: str, answer: str,):
    return [
        {
            "role": "system",
            "content": INFERENCE_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": create_inference_prompt(
                question,
                answer
            ),
        },
    ]


def generate(
    model,
    tokenizer,
    messages,
    max_new_tokens: int = 2048,
):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated = outputs[0][
        inputs["input_ids"].shape[-1]:
    ]

    return tokenizer.decode(
        generated,
        skip_special_tokens=True,
    )


def predict(
    model,
    tokenizer,
    question: str,
    answer: str
):
    messages = build_messages(
        question,
        answer,
    )

    start_time = time.perf_counter()

    response = generate(
        model,
        tokenizer,
        messages,
        settings.MAX_NEW_TOKENS
    )

    inference_time = time.perf_counter() - start_time

    print(f"\n추론 시간: {inference_time:.3f}초")

    try:
        cleaned_response = clean_json(response)
        return json.loads(cleaned_response)
    except json.JSONDecodeError:
        return {
            "error": "INVALID_JSON",
            "raw_response": response,
        }


if __name__ == "__main__":
    df = pd.read_csv(settings.REVIEWED_VALIDATION_MERGED_PATH)
    samples = df.head(NUM_SAMPLES)

    for epoch in range(1, 6):
        print(f"\n{'=' * 30}")
        print(f"Epoch {epoch}")
        print(f"{'=' * 30}")

        # Epoch별 모델을 한 번만 로드
        model, tokenizer = load_model(settings.SAVE_MODEL_PATH / f"epoch-{epoch}")

        for index, row in samples.iterrows():
            question = row["question"]
            answer = row["answer"]

            print(f"\n--- Sample {index} ---")
            print(f"질문: {question}")
            print(f"답변: {answer}")

            result = predict(
                model=model,
                tokenizer=tokenizer,
                question=question,
                answer=answer,
            )

            print(
                json.dumps(
                    result,
                    ensure_ascii=False,
                    indent=2,
                )
            )

        # 다음 Epoch 모델을 위해 현재 모델 정리
        del model
        del tokenizer
        torch.cuda.empty_cache()