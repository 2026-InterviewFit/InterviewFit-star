import pandas as pd
from pathlib import Path

from src.config.directory import create_directories
from src.config.config import settings


FILES = [
    "train_validated",
    "validation_validated",
    "train_remain_validated",
    "validation_remain_validated"
]

OCCUPATIONS = [
    "ICT",
    "RND",
    "Management",
    "SalesMarketing",
    "PublicService",
    "Design",
    "ProductionManufacturing"
]


def split_by_occupation(
    input_dir: Path,
    output_dir: Path,
    files: list[str],
    occupations: list[str],
    split_type: str
):
    create_directories()

    for occupation in occupations:
        dfs = []

        for file_name in files:
            df = pd.read_csv(
                input_dir / f"{file_name}.csv",
                encoding="utf-8-sig"
            )

            split_df = df[
                df["occupation"] == occupation
            ]

            if split_df.empty:
                continue

            dfs.append(split_df)


        if not dfs:
            continue


        merged = pd.concat(
            dfs,
            ignore_index=True
        )

        merged.to_csv(
            output_dir / f"{split_type}_{occupation}.csv",
            index=False,
            encoding="utf-8-sig"
        )

        print(
            split_type,
            occupation,
            len(merged)
        )


if __name__ == "__main__":
    split_by_occupation(
        input_dir=settings.VALIDATED_TRAIN_PATH.parent,
        output_dir=settings.SPLIT_TRAIN_DIR,
        files=[
            "train_validated",
            "train_remain_validated"
        ],
        occupations=OCCUPATIONS,
        split_type="train"
    )

    split_by_occupation(
        input_dir=settings.VALIDATED_VALIDATION_PATH.parent,
        output_dir=settings.SPLIT_VALIDATION_DIR,
        files=[
            "validation_validated",
            "validation_remain_validated"
        ],
        occupations=OCCUPATIONS,
        split_type="validation"
    )