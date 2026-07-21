import json
import time

import os
import pandas as pd
from tqdm import tqdm

from src.generation.prompts.interview import create_prompt
from src.generation.llm_client import generate
from src.config.config import settings


def generate_dataset(input_path, output_path, sample_size=None, save_interval=100):
    df = pd.read_csv(input_path)

    if sample_size:
        df = df.head(sample_size)

    results = []

    # 기존 파일 있으면 이어쓰기 가능
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        processed_answers = set(existing_df["answer"])
    else:
        processed_answers = set()

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # 이미 처리한 데이터는 skip
        if row["answer"] in processed_answers:
            continue

        prompt = create_prompt(row["answer"])
        output = generate(prompt)

        time.sleep(2)

        if output is None:
            continue

        output_json = parse_analysis(output)

        if output_json is None:
            continue

        results.append(
            create_result_row(row, output_json)
        )

        processed_answers.add(row["answer"])

        # 일정 개수마다 저장
        if len(results) >= save_interval:
            save_results(results,output_path)

            print(f"\n\n{idx}번째까지 저장 완료")

            results.clear()

    if results:
        save_results(results,output_path)


def parse_analysis(output: str):
    try:
        output_json = json.loads(output)
        return output_json
    except json.JSONDecodeError as e:
        print("JSON parsing failed:", e)

    return None


def create_result_row(row, output_json):
    star_result = {
        "star": output_json["star"],
        "strengths": output_json["strengths"],
        "improvements": output_json["improvements"]
    }
    return {
        "occupation": row["occupation"],
        "occupation_code": row["occupation_code"],
        "experience": row["experience"],
        "question": row["question"],
        "answer": row["answer"],
        "summary": row["summary"],
        "is_star_applicable": output_json["is_star_applicable"],
        "star_analysis": json.dumps(
            star_result,
            ensure_ascii=False
        ),
        "prompt_version": "v1",
        "model": settings.UPSTAGE_MODEL
    }


def save_results(results, output_path):
    save_df = pd.DataFrame(results)

    save_df.to_csv(
        output_path,
        mode="a",
        header=not os.path.exists(output_path),
        index=False,
        encoding="utf-8-sig"
    )


if __name__ == "__main__":
    generate_dataset(
        settings.TRAIN_INPUT_PATH,
        settings.GENERATED_TRAIN_PATH,
        3,
        1
    )

    generate_dataset(
        settings.VALIDATION_INPUT_PATH,
        settings.GENERATED_VALIDATION_PATH,
        3,
        1
    )