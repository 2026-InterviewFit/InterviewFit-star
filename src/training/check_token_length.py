from transformers import AutoTokenizer

from src.config.config import settings
from src.training.common.dataset import load_datasets


MAX_SEQ_LENGTH = 2048


def check_lengths(dataset, tokenizer, name):
    lengths = []

    for example in dataset:
        text = example["text"]

        tokens = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
        )

        lengths.append(len(tokens["input_ids"]))

    over_limit = sum(
        length > MAX_SEQ_LENGTH
        for length in lengths
    )

    sorted_lengths = sorted(lengths)

    def percentile(p):
        index = int(len(sorted_lengths) * p) - 1
        index = max(0, min(index, len(sorted_lengths) - 1))
        return sorted_lengths[index]

    print(f"\n===== {name} =====")
    print(f"데이터 수: {len(lengths)}")
    print(f"최대 토큰 길이: {max(lengths)}")
    print(f"평균 토큰 길이: {sum(lengths) / len(lengths):.1f}")
    print(f"P90: {percentile(0.90)}")
    print(f"P95: {percentile(0.95)}")
    print(f"P99: {percentile(0.99)}")
    print(f"{MAX_SEQ_LENGTH} 초과: {over_limit}개")
    print(
        f"{MAX_SEQ_LENGTH} 초과 비율: "
        f"{over_limit / len(lengths) * 100:.2f}%\n\n"
    )


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        settings.MODEL_NAME
    )

    datasets = load_datasets(
        tokenizer,
        settings.TRAIN_PATH,
        settings.VALID_PATH,
    )

    check_lengths(
        datasets["train"],
        tokenizer,
        "TRAIN",
    )

    check_lengths(
        datasets["validation"],
        tokenizer,
        "VALIDATION",
    )


if __name__ == "__main__":
    main()