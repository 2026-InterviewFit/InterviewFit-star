from pathlib import Path
import json

import pandas as pd
from pydantic import ValidationError
from tqdm import tqdm

from src.schema.schema import InterviewAnalysis
from src.config.config import settings


def is_valid_output(output: dict) -> tuple[bool, str]:
    """
    생성된 STAR 분석 결과가 학습에 사용 가능한지 검증한다.
    """

    try:
        InterviewAnalysis(**output)
    except ValidationError as e:
        print(e.errors())
        return False, "schema_error"

    return True, "valid"


def validate_dataset(input_path: Path, output_path: Path):
    """
    생성된 STAR dataset을 검증하고
    통과한 데이터만 저장한다.
    """

    df = pd.read_csv(input_path)

    validated = []

    error_count = {
        "json_error": 0,
        "schema_error": 0
    }

    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            output = json.loads(
                row["star_analysis"]
            )
        except json.JSONDecodeError:
            error_count["json_error"] += 1
            continue

        valid, reason = is_valid_output(output)

        if not valid:
            error_count[reason] += 1
            continue

        validated.append(row.to_dict())

    validated_df = pd.DataFrame(validated)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    validated_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("=" * 50)
    print(f"전체 데이터 : {len(df)}")
    print(f"정상 데이터 : {len(validated_df)}")
    print(f"제외 데이터 : {len(df) - len(validated_df)}")
    print("=" * 50)

    print("\n오류 통계")

    for key, value in error_count.items():
        if value > 0:
            print(f"{key:25}: {value}")


if __name__ == "__main__":
    validate_dataset(
        settings.GENERATED_TRAIN_PATH,
        settings.VALIDATED_TRAIN_PATH
    )

    validate_dataset(
        settings.GENERATED_VALIDATION_PATH,
        settings.VALIDATED_VALIDATION_PATH
    )