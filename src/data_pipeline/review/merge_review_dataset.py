import pandas as pd
from pathlib import Path

from src.config.config import settings


REMOVE_COLUMNS = [
    "review_status",
    "prompt_version",
    "model"
]


def merge_reviewed_dataset(
    input_dir: Path,
    output_path: Path
):
    files = sorted(input_dir.glob("*.csv"))

    if not files:
        print(f"No files found in {input_dir}")
        return

    dfs = []

    for file in files:
        df = pd.read_csv(
            file,
            encoding="utf-8-sig"
        )

        # KEEP 데이터만 유지
        df = df[df["review_status"] == "KEEP"]

        # 검수/생성 메타데이터 제거
        df = df.drop(
            columns=REMOVE_COLUMNS,
            errors="ignore"
        )

        dfs.append(df)

        print(f"{file.name}: {len(df)} rows kept")

    merged_df = pd.concat(
        dfs,
        ignore_index=True
    )

    merged_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nMerged {len(files)} files")
    print(f"Total rows: {len(merged_df)}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    merge_reviewed_dataset(
        input_dir=settings.REVIEWED_TRAIN_DIR,
        output_path=settings.REVIEWED_TRAIN_MERGED_PATH
    )

    merge_reviewed_dataset(
        input_dir=settings.REVIEWED_VALIDATION_DIR,
        output_path=settings.REVIEWED_VALIDATION_MERGED_PATH
    )