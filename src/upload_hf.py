from huggingface_hub import create_repo, upload_folder


create_repo(
    "repo_name",
    repo_type="model",
    exist_ok=True,
)


upload_folder(
    folder_path="outputs/checkpoint-633",
    repo_id="your_repo_id",
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