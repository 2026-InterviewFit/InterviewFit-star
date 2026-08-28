import json

import pandas as pd
from tqdm import tqdm

from src.inference.inference import load_model
from src.config.config import settings
from src.inference.inference import predict
from src.evaluation.judge import judge
from src.evaluation.metrics import calculate_metrics


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


def run_evaluation(checkpoint_path):
    dataset = load_test_dataset()

    print("=" * 50)
    print("Evaluating Best Model")
    print("=" * 50)

    model, tokenizer = load_model(checkpoint_path)

    results = []
    judge_results = []

    for index, sample in enumerate(tqdm(dataset, desc="Evaluating Best Model"), start=1):
        user_content, assistant_content = parse_messages(sample["messages"])

        if user_content is None or assistant_content is None:
            continue

        try:
            ground_truth = json.loads(assistant_content)
        except json.JSONDecodeError:
            continue

        try:
            question, answer = parse_user_content(user_content)
        except ValueError:
            continue

        prediction = predict(
            model,
            tokenizer,
            question,
            answer,
        )

        if "error" in prediction:
            judge_score = None
            judge_feedback = "Prediction failed."
            status = "prediction_error"
        else:
            try:
                judge_result = judge(
                    ground_truth,
                    prediction,
                )
                judge_score = judge_result["score"]
                judge_feedback = judge_result["feedback"]
                judge_results.append(
                    judge_result
                )
                status = "success"
            except Exception as e:
                judge_score = None
                judge_feedback = str(e)
                status = "judge_error"

        results.append(
            {
                "index": index,
                "question": question,
                "answer": answer,
                "ground_truth": json.dumps(
                    ground_truth,
                    ensure_ascii=False,
                ),
                "prediction": json.dumps(
                    prediction,
                    ensure_ascii=False,
                ),
                "judge_score": judge_score,
                "judge_feedback": judge_feedback,
                "status": status,
            }
        )

        if index % 20 == 0:
            df = pd.DataFrame(results)
            df.to_csv(
                settings.REPORT_DIR / "result.csv",
                index=False,
                encoding="utf-8-sig",
            )

    # 상세 평가 결과 저장
    df = pd.DataFrame(results)
    df.to_csv(
        settings.REPORT_DIR / f"result.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 전체 평가 지표 계산
    summary = calculate_metrics(judge_results)

    with open(
            settings.REPORT_DIR / f"summary.json",
            "w",
            encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=4,
        )


if __name__ == "__main__":
    checkpoint_path = settings.OUTPUT_DIR / "checkpoint-xxxx" # 실제 best checkpoint 사용하기
    run_evaluation(checkpoint_path)