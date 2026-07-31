import json
from pathlib import Path

import pandas as pd

from src.config.config import settings


def average_length(series):
    """
    문자열 길이 평균 계산
    """

    return series.fillna("").astype(str).str.len().mean()


def safe_average(values):
    """
    리스트 평균 계산 (빈 리스트 방지)
    """

    if not values:
        return 0

    return sum(values) / len(values)


def analyze_dataset(dataset_path: Path, dataset_name: str, report_path: Path):
    """
    validated dataset 통계 분석 및 report 저장
    """

    df = pd.read_csv(dataset_path)

    situation_lengths = []
    task_lengths = []
    action_lengths = []
    result_lengths = []
    strengths_count = []
    improvements_count = []
    json_error_count = 0

    for output in df["star_analysis"]:
        try:
            data = json.loads(output)
        except (json.JSONDecodeError, TypeError):
            json_error_count += 1
            continue

        star = data.get("star", {})
        situation_lengths.append(
            len(star.get("situation", ""))
        )
        task_lengths.append(
            len(star.get("task", ""))
        )
        action_lengths.append(
            len(star.get("action", ""))
        )
        result_lengths.append(
            len(star.get("result", ""))
        )
        strengths_count.append(
            len(data.get("strengths", []))
        )
        improvements_count.append(
            len(data.get("improvements", []))
        )

    report = {
        "dataset": dataset_name,
        "total_count": len(df),

        "occupation_distribution": (
            df["occupation"]
            .value_counts()
            .to_dict()
        ),

        "experience_distribution": (
            df["experience"]
            .value_counts()
            .to_dict()
        ),

        "average_length": {
            "question": average_length(df["question"]),
            "answer": average_length(df["answer"]),
            "summary": average_length(df["summary"])
        },

        "star_average_length": {
            "situation": safe_average(situation_lengths),
            "task": safe_average(task_lengths),
            "action": safe_average(action_lengths),
            "result": safe_average(result_lengths)
        },

        "average_evaluation_count": {
            "strengths": safe_average(strengths_count),
            "improvements": safe_average(improvements_count)
        },

        "json_error_count": json_error_count
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=4
        )

    print("=" * 60)
    print(f"{dataset_name} 분석 완료")
    print(f"Report 저장: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
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