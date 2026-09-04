from huggingface_hub import create_repo, upload_folder


repo_id = ""


create_repo(
    repo_id,
    repo_type="model",
    exist_ok=True,
)


# LoRA Adapter
upload_folder(
    folder_path="outputs/v1/checkpoint-633", # 수정하기
    repo_id=repo_id,
    repo_type="model",
    allow_patterns=[
        # 필수: LoRA 어댑터
        "adapter_model.safetensors",
        "adapter_config.json",
        # 추천: Tokenizer
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        # 있으면 추천: 생성/채팅 설정
        "generation_config.json",
        "chat_template.jinja",
        # 문서
        "README.md",
    ],
)


# model
upload_folder(
    folder_path="outputs/v1/merged-qwen3-4b-star", # 수정하기
    repo_id=repo_id,
    repo_type="model",
)