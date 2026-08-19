from collections import Counter


def calculate_metrics(results: list[dict]) -> dict[str, int | float]:
    """
    Judge 결과를 기반으로
    STAR 평가 점수 평균과 accuracy를 계산
    """

    total = len(results)

    if total == 0:
        return {
            "samples": 0,
            "average_score": 0,
            "accuracy": 0,
        }

    scores = [
        result["score"]
        for result in results
        if "score" in result
    ]

    average_score = (
        round(
            sum(scores) / len(scores),
            3,
        )
        if scores
        else 0
    )

    # 4점 이상을 정확한 분석으로 정의
    correct_count = sum(
        1
        for score in scores
        if score >= 4
    )

    incorrect_count = (
            len(scores) - correct_count
    )

    accuracy = (
        round(
            correct_count / len(scores),
            4,
        )
        if scores
        else 0
    )

    return {
        "samples": total,
        "evaluated_samples": len(scores),
        "average_score": average_score,
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "accuracy": accuracy,
    }


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