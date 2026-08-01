from collections import Counter

from src.schema.schema import EvaluationScore


SCORE_FIELDS = [
    field_name
    for field_name in EvaluationScore.model_fields
    if field_name.endswith("_score")
]


def calculate_metrics(results: list[dict]) -> dict[str, int | float]:
    """
    Judge 결과를 기반으로
    STAR 평가 점수 평균과 accuracy를 계산한다.
    """

    total = len(results)

    if total == 0:
        return {}

    summary = {
        "samples": total,
    }

    # 평균 점수 계산
    for field in SCORE_FIELDS:
        values = [
            result[field]
            for result in results
            if field in result
        ]

        summary[f"avg_{field}"] = round(
            sum(values) / len(values),
            3,
        ) if values else 0

    # is_correct 계산
    correct_count = sum(
        int(result.get("is_correct", 0))
        for result in results
    )

    summary["correct_count"] = correct_count
    summary["incorrect_count"] = (
            total - correct_count
    )
    summary["accuracy"] = round(
        correct_count / total,
        4,
    )

    return summary


def print_summary(summary: dict):
    print("=" * 50)
    print("Evaluation Summary")
    print("=" * 50)
    print(
        f"Samples       : {summary['samples']}"
    )
    print(
        f"Correct Count : {summary['correct_count']}"
    )
    print(
        f"Incorrect     : {summary['incorrect_count']}"
    )
    print(
        f"Correct Rate  : "
        f"{summary['accuracy']:.2%}"
    )
    print()

    for field in SCORE_FIELDS:
        print(
            f"{field:<25}: "
            f"{summary[f'avg_{field}']:.3f}"
        )


def collect_feedback(results: list[dict], top_k: int = 10,):
    """
    Judge feedback 빈도 확인
    """

    feedbacks = [
        result["feedback"]
        for result in results
        if result.get("feedback")
    ]

    return Counter(
        feedbacks
    ).most_common(top_k)


if __name__ == "__main__":
    results = [
        {
            "situation_score": 5,
            "task_score": 4,
            "action_score": 5,
            "result_score": 4,
            "strengths_score": 5,
            "improvements_score": 4,
            "overall_score": 5,
            "is_correct": 1,
            "feedback": "우수한 답변입니다."
        },
        {
            "situation_score": 4,
            "task_score": 4,
            "action_score": 4,
            "result_score": 3,
            "strengths_score": 4,
            "improvements_score": 4,
            "overall_score": 4,
            "is_correct": 1,
            "feedback": "Result가 조금 부족합니다."
        }
    ]
    summary = calculate_metrics(
        results
    )
    print_summary(
        summary
    )

    print("\nSummary Dict")
    print(summary)


    print("\nTop Feedback")

    for feedback, count in collect_feedback(results):
        print(
            f"{count}회 - {feedback}"
        )