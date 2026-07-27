from src.config.directory import create_directories
from src.config.config import settings

from src.generation.generate_dataset import generate_dataset
from src.validation.validate_dataset import validate_dataset
from src.validation.statistics import analyze_dataset


# 테스트용
TEST_SAMPLE_SIZE = 2
def run_test_generation_pipeline():
    create_directories()
    generate_dataset(
        settings.TRAIN_INPUT_PATH,
        settings.GENERATED_TRAIN_PATH,
        sample_size=TEST_SAMPLE_SIZE,
        save_interval=1
    )
    validate_dataset(
        settings.GENERATED_TRAIN_PATH,
        settings.VALIDATED_TRAIN_PATH,
    )
    analyze_dataset(
        settings.VALIDATED_TRAIN_PATH,
        "train",
        settings.VALIDATED_DATA_REPORT_DIR / "train_analysis.json"
    )


def run_generation_pipeline():
    """
    데이터 생성 파이프라인

    1. 디렉토리 생성
    2. Train 데이터 생성
    3. Validation 데이터 생성
    4. Train 데이터 검증
    5. Validation 데이터 검증
    6. 데이터 분석 리포트 생성
    """

    print("=" * 60)
    print("1. Create directories")
    print("=" * 60)

    create_directories()

    print("=" * 60)
    print("2. Generate train dataset")
    print("=" * 60)

    generate_dataset(
        settings.TRAIN_INPUT_PATH,
        settings.GENERATED_TRAIN_PATH
    )

    print("=" * 60)
    print("3. Generate validation dataset")
    print("=" * 60)

    generate_dataset(
        settings.VALIDATION_INPUT_PATH,
        settings.GENERATED_VALIDATION_PATH
    )

    print("=" * 60)
    print("4. Validate train dataset")
    print("=" * 60)

    validate_dataset(
        settings.GENERATED_TRAIN_PATH,
        settings.VALIDATED_TRAIN_PATH,
    )

    print("=" * 60)
    print("5. Validate validation dataset")
    print("=" * 60)

    validate_dataset(
        settings.GENERATED_VALIDATION_PATH,
        settings.VALIDATED_VALIDATION_PATH,
    )

    print("=" * 60)
    print("6. Analyze dataset")
    print("=" * 60)

    analyze_dataset(
        settings.VALIDATED_TRAIN_PATH,
        "train",
        settings.VALIDATED_DATA_REPORT_DIR / "train_analysis.json"
    )

    analyze_dataset(
        settings.VALIDATED_VALIDATION_PATH,
        "validation",
        settings.VALIDATED_DATA_REPORT_DIR / "validation_analysis.json"
    )

    print("=" * 60)
    print("Pipeline Finished")
    print("=" * 60)


def run_generation_pipeline_missing():
    print("=" * 60)
    print("2. Generate train dataset")
    print("=" * 60)

    generate_dataset(
        settings.TRAIN_INPUT_PATH,
        settings.GENERATED_TRAIN_PATH,
        processed_path=settings.VALIDATED_TRAIN_PATH,
        sample_size=TEST_SAMPLE_SIZE,
        save_interval=1
    )

    print("=" * 60)
    print("3. Generate validation dataset")
    print("=" * 60)

    generate_dataset(
        settings.VALIDATION_INPUT_PATH,
        settings.GENERATED_VALIDATION_PATH,
        processed_path=settings.VALIDATED_VALIDATION_PATH
    )

    print("=" * 60)
    print("4. Validate train dataset")
    print("=" * 60)

    validate_dataset(
        settings.GENERATED_TRAIN_PATH,
        settings.VALIDATED_TRAIN_PATH,
    )

    print("=" * 60)
    print("5. Validate validation dataset")
    print("=" * 60)

    validate_dataset(
        settings.GENERATED_VALIDATION_PATH,
        settings.VALIDATED_VALIDATION_PATH,
    )

    print("=" * 60)
    print("6. Analyze dataset")
    print("=" * 60)

    analyze_dataset(
        settings.VALIDATED_TRAIN_PATH,
        "train",
        settings.VALIDATED_DATA_REPORT_DIR / "train_analysis.json"
    )

    analyze_dataset(
        settings.VALIDATED_VALIDATION_PATH,
        "validation",
        settings.VALIDATED_DATA_REPORT_DIR / "validation_analysis.json"
    )

    print("=" * 60)
    print("Pipeline Finished")
    print("=" * 60)


if __name__ == "__main__":
    # run_test_generation_pipeline()
    # run_generation_pipeline()
    run_generation_pipeline_missing()