from src.config.config import settings
from pathlib import Path


def create_directories():
    paths = [
        # generated
        settings.GENERATED_TRAIN_PATH.parent,
        settings.GENERATED_VALIDATION_PATH.parent,

        # validated
        settings.VALIDATED_TRAIN_PATH.parent,
        settings.VALIDATED_VALIDATION_PATH.parent,
        settings.VALIDATED_DATA_REPORT_DIR,

        # split
        settings.SPLIT_TRAIN_DIR,
        settings.SPLIT_VALIDATION_DIR,

        # reviewed
        settings.REVIEWED_TRAIN_DIR,
        settings.REVIEWED_VALIDATION_DIR,

        # processed
        settings.TRAIN_PATH.parent,
        settings.VALID_PATH.parent,
        settings.TEST_PATH.parent,

        # training outputs
        settings.OUTPUT_DIR,
        settings.SAVE_MODEL_PATH,

        # evaluation
        settings.REPORT_DIR,
    ]

    for path in paths:
        Path(path).mkdir(
            parents=True,
            exist_ok=True
        )