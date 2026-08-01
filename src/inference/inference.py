import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.inference.prompts.inference_prompt import INFERENCE_SYSTEM_PROMPT, create_inference_prompt
from src.config.config import settings


_model = None
_tokenizer = None


def clean_json(response: str):
    response = response.strip()

    if response.startswith("```json"):
        response = response[len("```json"):]

    if response.endswith("```"):
        response = response[:-3]

    return response.strip()


def load_model():
    global _model, _tokenizer

    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    _tokenizer = AutoTokenizer.from_pretrained(
        settings.SAVE_MODEL_PATH
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        settings.MODEL_NAME,
        device_map="auto",
        torch_dtype="auto",
    )

    _model = PeftModel.from_pretrained(
        base_model,
        settings.SAVE_MODEL_PATH,
    )

    _model.eval()

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
    max_new_tokens: int = 1024,
):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
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


def predict(question: str, answer: str,):
    model, tokenizer = load_model()

    messages = build_messages(
        question,
        answer,
    )

    response = generate(
        model,
        tokenizer,
        messages,
    )

    try:
        cleaned_response = clean_json(response)

        return json.loads(cleaned_response)
    except json.JSONDecodeError:
        return {
            "error": "INVALID_JSON",
            "raw_response": response,
        }


if __name__ == "__main__":
    result = predict(
        question="협업 경험을 말씀해주세요.",
        answer=(
            "캡스톤 프로젝트에서 "
            "팀원들과 API를 개발했습니다."
        ),
    )

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )