import json
import time

import pandas as pd

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.inference.prompts.inference_prompt import INFERENCE_SYSTEM_PROMPT, create_inference_prompt
from src.config.config import settings



NUM_SAMPLES = 1


def clean_json(response: str):
    response = response.strip()

    if response.startswith("```json"):
        response = response[len("```json"):]
    if response.endswith("```"):
        response = response[:-3]

    return response.strip()


def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

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
        attn_implementation="sdpa",
    )

    # Inference: KV Cache ON
    base_model.config.use_cache = True

    model = PeftModel.from_pretrained(
        base_model,
        model_path,
    )

    model.eval()

    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM total:", torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, "GB")
    print("VRAM allocated:", torch.cuda.memory_allocated() / 1024 ** 3, "GB")
    print("VRAM reserved:", torch.cuda.memory_reserved() / 1024 ** 3, "GB")
    print(model.hf_device_map)

    return model, tokenizer


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

    print("Input tokens:", inputs["input_ids"].shape[-1])
    print("Max new tokens:", max_new_tokens)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated = outputs[0][
        inputs["input_ids"].shape[-1]:
    ]

    print("Generated tokens:", outputs.shape[1] - inputs["input_ids"].shape[1])

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
    checkpoint_path = settings.OUTPUT_DIR / "checkpoint-12" # 실제 best checkpoint 사용하기

    model, tokenizer = load_model(checkpoint_path)

    df = pd.read_csv(settings.FILTERED_VALIDATED_VALIDATION_PATH)
    samples = df.head(NUM_SAMPLES)

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