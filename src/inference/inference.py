import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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


def predict(question: str, answer: str,):
    model, tokenizer = load_model(
        settings.SAVE_MODEL_PATH / "epoch-1"
    )

    messages = build_messages(
        question,
        answer,
    )

    response = generate(
        model,
        tokenizer,
        messages,
        settings.MAX_NEW_TOKENS
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
        question="위기 관리 경험이 있나요.",
        answer="위기 관리 경험이 있나요 어 라고 말씀을 주셨는데요. 어 이거 이 질문은 어 직장에서 인건지 아니면 제가 살아오는 내 인생에 있어서인건지 어떤 쪽으로 제가 답변을 드려야 될지는 조금 더 구체적인 질문이 필요로 할 것 같고요. 일단은 내 삶 속에서 내 삶에서 어 위기 관리 경험이 있냐 라고 생각하고 말씀을 드린다면 제 삶 속에서 투자를 한 번 잘못해서 어 많은 돈을 어 날린 적이 있습니다. 그때 저희 가정이 어 파탄나지 않을까 또 제가 저의 그 생명에 위험하지 않을까 즉 나의 자존심이 허락지 않아서 제가 정말로 자살을 하고 싶을 정도의 어떤 그런 좌절한 적이 좌절감이 있었는데 그래도 저의 그 긍정적이고 음 어떤 그 지혜로운 어떤 그 생각이 저를 다시 살게했고 그런 에너지들이 저에게 다시 기회를 주셨고 그래서 지금 이 자리에 있게 된 거 이런 경험이 있습니다. 모든 것에 감사합니다. 이상입니다.",
    )

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )