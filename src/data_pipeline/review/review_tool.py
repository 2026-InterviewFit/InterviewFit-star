import os
import json
from pathlib import Path

import pandas as pd

from src.config.config import settings


TRAIN_INPUT_DIR = settings.SPLIT_TRAIN_DIR
VALIDATION_INPUT_DIR = settings.SPLIT_VALIDATION_DIR

TRAIN_OUTPUT_DIR = settings.REVIEWED_TRAIN_DIR
VALIDATION_OUTPUT_DIR = settings.REVIEWED_VALIDATION_DIR


def print_star(star_text):
    print("\n[STAR Analysis]")
    print("-" * 60)

    try:
        data = json.loads(star_text)

        # {"star": {...}} 또는 {...} 둘 다 지원
        star = data.get("star", data)

        print("\nSituation:")
        print(star.get("situation", ""))

        print("\nTask:")
        print(star.get("task", ""))

        print("\nAction:")
        print(star.get("action", ""))

        print("\nResult:")
        print(star.get("result", ""))

        strengths = data.get("strengths", [])
        print("\nStrengths:")
        if strengths:
            for strength in strengths:
                print(f"- {strength}")
        else:
            print("")

        improvements = data.get("improvements", [])
        print("\nImprovements:")
        if improvements:
            for improvement in improvements:
                print(f"- {improvement}")
        else:
            print("")
    except Exception:
        print(star_text)


def select_file(input_dir: Path):
    files = sorted(
        input_dir.glob("*.csv")
    )

    if not files:
        raise Exception(
            f"{input_dir}에 csv 파일이 없습니다."
        )

    print("\n검수할 직무 선택")
    print("-" * 50)

    for idx, file in enumerate(files):
        print(
            f"[{idx}] {file.stem}"
        )

    while True:
        try:
            choice = int(
                input("\n번호 입력: ")
            )

            if 0 <= choice < len(files):
                return files[choice]

        except ValueError:
            pass

        print("잘못된 입력입니다.")


def select_dataset_type():
    print(
        """
검수 데이터 선택

[1] Train
[2] Validation
"""
    )

    while True:
        choice = input("> ")

        if choice == "1":
            return (
                TRAIN_INPUT_DIR,
                TRAIN_OUTPUT_DIR
            )

        elif choice == "2":
            return (
                VALIDATION_INPUT_DIR,
                VALIDATION_OUTPUT_DIR
            )

        print("잘못된 입력입니다.")


def get_output_file(
    input_file: Path,
    output_dir: Path
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    return output_dir / (
        input_file.stem + "_reviewed.csv"
    )


def load_review_data(
    input_file: Path,
    output_file: Path
):
    df = pd.read_csv(
        input_file,
        encoding="utf-8-sig"
    )

    if output_file.exists():

        reviewed_df = pd.read_csv(
            output_file,
            encoding="utf-8-sig"
        )

        reviewed_count = len(reviewed_df)

        print(
            f"\n기존 검수 완료: {reviewed_count}개"
        )

        # 검수 완료 개수만큼 건너뛰기
        return df.iloc[reviewed_count:]

    print(
        "\n새 검수 시작"
    )

    return df


def save_row(
    row,
    status,
    output_file: Path
):
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result = row.to_frame().T.copy()
    result["review_status"] = status

    try:
        result.to_csv(
            output_file,
            mode="a",
            header=not output_file.exists(),
            index=False,
            encoding="utf-8-sig"
        )

        print("exists:", output_file.exists())

        if output_file.exists():
            print("size:", output_file.stat().st_size)

    except Exception as e:
        print("저장 실패:", repr(e))


def print_progress(
    input_file: Path,
    output_file: Path
):
    total = len(
        pd.read_csv(
            input_file,
            encoding="utf-8-sig"
        )
    )

    done = 0

    if output_file.exists():
        done = len(
            pd.read_csv(
                output_file,
                encoding="utf-8-sig"
            )
        )

    print(
        f"\n진행률: {done}/{total} "
        f"({done / total * 100:.2f}%)"
    )


def review(
    input_file: Path,
    output_file: Path
):
    df = load_review_data(
        input_file,
        output_file
    )

    total = len(
        pd.read_csv(
            input_file,
            encoding="utf-8-sig"
        )
    )

    print_progress(
        input_file,
        output_file
    )

    for idx, row in df.iterrows():
        os.system(
            "cls"
            if os.name == "nt"
            else "clear"
        )

        current = idx + 1

        print("=" * 80)
        print(
            f"{idx + 1} / {total}"
        )
        print("=" * 80)

        print("\n[Question]")
        print(row["question"])

        print("\n[Answer]")
        print(row["answer"])

        print_star(
            row["star_analysis"]
        )

        print("\n")
        print("=" * 80)

        print(
            """
선택:
[Enter] KEEP
[n] REMOVE
[h] HOLD
[q] QUIT
"""
        )

        while True:
            choice = input("> ").lower()

            if choice == "":
                save_row(
                    row,
                    "KEEP",
                    output_file
                )
                break
            elif choice == "n":
                save_row(
                    row,
                    "REMOVE",
                    output_file
                )
                break
            elif choice == "h":
                save_row(
                    row,
                    "HOLD",
                    output_file
                )
                break
            elif choice == "q":
                print(
                    "\n저장 완료:",
                    output_file
                )
                return
            else:
                print(
                    "잘못된 입력입니다."
                )


if __name__ == "__main__":
    input_dir, output_dir = select_dataset_type()

    input_file = select_file(
        input_dir
    )

    output_file = get_output_file(
        input_file,
        output_dir
    )

    print(
        "\n입력:",
        input_file
    )

    print(
        "출력:",
        output_file
    )

    review(
        input_file,
        output_file
    )