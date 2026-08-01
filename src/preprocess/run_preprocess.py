from sklearn.model_selection import train_test_split

from src.config.config import settings

from src.preprocess.convert_dataset import (
    convert_csv_to_dataset,
    save_jsonl
)


def run_preprocess_pipeline():
    # train 데이터 그대로 사용
    train_dataset = convert_csv_to_dataset(
        settings.VALIDATED_TRAIN_PATH
    )
    # validation → valid/test 분리
    validation_dataset = convert_csv_to_dataset(
        settings.VALIDATED_VALIDATION_PATH
    )

    valid_dataset, test_dataset = train_test_split(
        validation_dataset,
        test_size=0.5,
        random_state=42,
        shuffle=True
    )

    if not train_dataset:
        raise ValueError("Train dataset is empty")
    if not valid_dataset:
        raise ValueError("Validation dataset is empty")
    if not test_dataset:
        raise ValueError("Test dataset is empty")

    save_jsonl(
        train_dataset,
        settings.TRAIN_PATH
    )
    save_jsonl(
        valid_dataset,
        settings.VALID_PATH
    )
    save_jsonl(
        test_dataset,
        settings.TEST_PATH
    )

    print("=" * 50)
    print(f"Train : {len(train_dataset)}")
    print(f"Valid : {len(valid_dataset)}")
    print(f"Test  : {len(test_dataset)}")
    print("=" * 50)


if __name__ == "__main__":
    run_preprocess_pipeline()