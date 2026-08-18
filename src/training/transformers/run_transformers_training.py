from src.training.transformers.trainer import (
    load_transformers_model,
    apply_peft_lora,
    create_transformers_trainer,
)


def run_transformers_training_pipeline():
    model, tokenizer = load_transformers_model()

    model = apply_peft_lora(model)

    trainer = create_transformers_trainer(model, tokenizer)

    trainer.train()


if __name__ == "__main__":
    run_transformers_training_pipeline()