import json
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from src.preprocess.prompts.system import SYSTEM_PROMPT
from src.schema.schema import InterviewAnalysis


def build_sample(row):
    try:
        output = json.loads(
            row["star_analysis"]
        )

        if not validate_analysis(output):
            return None
    except Exception as e:
        print(f"Failed to build sample: {e}")
        return None

    user_prompt = (
        f"### 질문:\n{row['question']}\n\n"
        f"### 답변:\n{row['answer']}"
    )

    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    output,
                    ensure_ascii=False
                )
            }
        ]
    }


def convert_csv_to_dataset(csv_path):
    df = pd.read_csv(csv_path)

    dataset = []

    for _, row in df.iterrows():
        sample = build_sample(row)

        if sample:
            dataset.append(sample)

    return dataset


def save_jsonl(data, path):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False
                )
                + "\n"
            )


def validate_analysis(output):
    try:
        InterviewAnalysis.model_validate(output)
        return True
    except ValidationError:
        return False