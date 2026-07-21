from pathlib import Path
from typing import ClassVar

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # =====================
    # Project Root
    # =====================
    BASE_DIR: ClassVar[Path] = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    # =====================
    # Dataset
    # =====================
    TRAIN_INPUT_PATH: Path = (
            BASE_DIR /
            "data/raw/train.csv"
    )
    VALIDATION_INPUT_PATH: Path = (
            BASE_DIR /
            "data/raw/validation.csv"
    )
    GENERATED_TRAIN_PATH: Path = (
            BASE_DIR /
            "data/generated/train_star_dataset.csv"
    )
    GENERATED_VALIDATION_PATH: Path = (
            BASE_DIR /
            "data/generated/validation_star_dataset.csv"
    )
    VALIDATED_TRAIN_PATH: Path = (
            BASE_DIR /
            "data/validated/train_validated.csv"
    )
    VALIDATED_VALIDATION_PATH: Path = (
            BASE_DIR /
            "data/validated/validation_validated.csv"
    )
    VALIDATED_DATA_REPORT_DIR: Path = (
            BASE_DIR /
            "reports/validated_reports"
    )

    # =====================
    # LLM API
    # =====================
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4.1-mini"
    UPSTAGE_API_KEY: str = ""
    UPSTAGE_MODEL: str = "solar-pro3"
    TEMPERATURE: float = 0.2

    # =====================
    # Preprocess
    # =====================
    TRAIN_PATH: Path = (
        BASE_DIR /
        "data/processed/train.jsonl"
    )
    VALID_PATH: Path = (
        BASE_DIR /
        "data/processed/valid.jsonl"
    )
    TEST_PATH: Path = (
        BASE_DIR /
        "data/processed/test.jsonl"
    )

    # =====================
    # Training
    # =====================
    MODEL_NAME: str = "Qwen/Qwen3-4B" # "Qwen/Qwen3-8B"
    OUTPUT_DIR: Path = (
        BASE_DIR /
        "outputs"
    )
    SAVE_MODEL_PATH: Path = (
        BASE_DIR /
        "models/qwen3-lora"
    )
    NUM_EPOCHS: int = 3
    BATCH_SIZE: int = 2
    GRAD_ACCUMULATION: int = 4
    LEARNING_RATE: float = 2e-4

    # =====================
    # Evaluation
    # =====================
    REPORT_DIR: Path = (
        BASE_DIR /
        "reports/evaluation"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )


settings = Settings()