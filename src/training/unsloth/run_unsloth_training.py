from src.training.unsloth.trainer import (
    load_unsloth_model,
    apply_unsloth_lora,
    create_unsloth_trainer,
)


def run_unsloth_training_pipeline():
    model, tokenizer = load_unsloth_model()

    model = apply_unsloth_lora(model)

    trainer = create_unsloth_trainer(model, tokenizer)

    trainer.train()


if __name__ == "__main__":
    run_unsloth_training_pipeline()