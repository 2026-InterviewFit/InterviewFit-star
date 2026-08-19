from datasets import load_dataset
from transformers import AutoTokenizer

from src.config.config import settings


def apply_chat_template(dataset, tokenizer):
    def formatting(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )

        text = text.replace(
            "<think>\n\n</think>\n\n",
            ""
        )

        return {
            "text": text
        }

    return dataset.map(
        formatting,
        remove_columns=dataset.column_names
    )


def load_datasets(tokenizer, train_path, valid_path):
    datasets = load_dataset(
        "json",
        data_files={
            "train": str(train_path),
            "validation": str(valid_path),
        }
    )

    datasets["train"] = apply_chat_template(
        datasets["train"],
        tokenizer,
    )

    datasets["validation"] = apply_chat_template(
        datasets["validation"],
        tokenizer,
    )

    return datasets


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(
        settings.MODEL_NAME
    )

    datasets = load_dataset(
        "json",
        data_files={
            "train": str(settings.TRAIN_PATH),
            "validation": str(settings.VALID_PATH),
        }
    )

    datasets["validation"] = apply_chat_template(
        datasets["validation"],
        tokenizer,
    )

    print("=" * 80)
    print(datasets["validation"][0]["text"])
    print("=" * 80)