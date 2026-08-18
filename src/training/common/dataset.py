from datasets import load_dataset


def apply_chat_template(dataset, tokenizer):
    def formatting(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
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