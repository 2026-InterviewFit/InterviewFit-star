import json

import pandas as pd
from tqdm import tqdm

from src.config.config import settings
from src.inference.inference import predict
from src.evaluation.judge import judge
from src.evaluation.metrics import (
    calculate_metrics,
    print_summary,
)


RESULT_PATH = settings.REPORT_DIR / "result.csv"
SUMMARY_PATH = settings.REPORT_DIR / "summary.json"


def load_test_dataset():
    with open(
        settings.TEST_PATH,
        encoding="utf-8",
    ) as f:

        return [
            json.loads(line)
            for line in f
        ]


def parse_messages(messages):
    user_content = None
    assistant_content = None

    for message in messages:
        if message["role"] == "user":
            user_content = message["content"]

        elif message["role"] == "assistant":
            assistant_content = message["content"]

    return user_content, assistant_content


def parse_user_content(content):
    question_marker = "### 질문:"
    answer_marker = "### 답변:"

    question_start = content.index(question_marker)
    answer_start = content.index(answer_marker)

    question = (
        content[
            question_start + len(question_marker):
            answer_start
        ]
        .strip()
    )

    answer = (
        content[
            answer_start + len(answer_marker):
        ]
        .strip()
    )

    return question, answer


def run_evaluation():
    dataset = load_test_dataset()

    results = []
    judge_results = []

    for sample in tqdm(
            dataset,
            desc="Evaluating"
    ):
        user_content, assistant_content = parse_messages(
            sample["messages"]
        )

        if user_content is None or assistant_content is None:
            continue

        try:
            ground_truth = json.loads(
                assistant_content
            )
        except json.JSONDecodeError:
            continue

        question, answer = parse_user_content(
            user_content
        )
        prediction = predict(
            question,
            answer,
        )

        if "error" in prediction:
            score = {
                "situation_score": 0,
                "task_score": 0,
                "action_score": 0,
                "result_score": 0,
                "strengths_score": 0,
                "improvements_score": 0,
                "overall_score": 0,

                "is_correct": 0,

                "feedback": "Invalid JSON output"
            }
        else:
            score = judge(
                ground_truth,
                prediction,
            )

        judge_results.append(
            score
        )

        results.append(
            {
                "question":question,
                "answer":answer,

                "ground_truth":json.dumps(
                    ground_truth,
                    ensure_ascii=False,
                ),

                "prediction":json.dumps(
                    prediction,
                    ensure_ascii=False,
                ),

                **score,
            }
        )

    df = pd.DataFrame(results)

    df.to_csv(
        RESULT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    summary = calculate_metrics(
        judge_results
    )

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=4,
        )

    print_summary(summary)


if __name__ == "__main__":
    run_evaluation()