import pandas as pd
from pathlib import Path

from src.config.config import settings


def print_validated_dataset(input_path: str):
    df = pd.read_csv(
        input_path,
        encoding="utf-8-sig"
    )

    # 선별 전 STAR 적용 가능 여부 비율 확인
    star_counts = df["is_star_applicable"].value_counts()
    star_ratios = df["is_star_applicable"].value_counts(normalize=True) * 100

    print("\n[Before Sampling]")
    print(f"Total: {len(df)}")
    print(f"True: {star_counts.get(True, 0)} ({star_ratios.get(True, 0):.2f}%)")
    print(f"False: {star_counts.get(False, 0)} ({star_ratios.get(False, 0):.2f}%)")


def sample_validated_dataset(
    input_path: Path,
    output_path: Path,
    true_sample_ratio: float = 1.0,
    false_sample_ratio: float = 0.2,
    random_state: int = 42
):
    df = pd.read_csv(
        input_path,
        encoding="utf-8-sig"
    )

    star_applicable = df[df["is_star_applicable"] == True]

    star_not_applicable = df[df["is_star_applicable"] == False]

    true_sample_size = int(
        len(star_applicable) * true_sample_ratio
    )

    false_sample_size = int(
        len(star_not_applicable) * false_sample_ratio
    )

    star_applicable_sampled = star_applicable.sample(
        n=true_sample_size,
        random_state=random_state
    )

    star_not_applicable_sampled = star_not_applicable.sample(
        n=false_sample_size,
        random_state=random_state
    )

    # 합치기
    sampled_df = pd.concat(
        [
            star_applicable_sampled,
            star_not_applicable_sampled
        ],
        ignore_index=True
    )

    # 순서 섞기
    sampled_df = sampled_df.sample(
        frac=1,
        random_state=random_state
    ).reset_index(drop=True)

    sampled_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\n\nInput: {len(df)}")
    print(f"STAR applicable: {len(star_applicable)}")
    print(f"STAR not applicable: {len(star_not_applicable)}")
    print(f"Sampled not applicable: {len(star_not_applicable_sampled)}")
    print(f"Final: {len(sampled_df)}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    print_validated_dataset(settings.VALIDATED_TRAIN_PATH)
    print_validated_dataset(settings.VALIDATED_VALIDATION_PATH)

    sample_validated_dataset(
        input_path=settings.VALIDATED_TRAIN_PATH,
        output_path=settings.FILTERED_VALIDATED_TRAIN_PATH,
        false_sample_ratio=0.1
    )
    sample_validated_dataset(
        input_path=settings.VALIDATED_VALIDATION_PATH,
        output_path=settings.FILTERED_VALIDATED_VALIDATION_PATH,
        true_sample_ratio=0.1,
        false_sample_ratio = 0.05
    )