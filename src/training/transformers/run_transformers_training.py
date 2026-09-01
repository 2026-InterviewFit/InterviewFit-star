import os
import wandb

import gc
import torch

from src.config.config import settings
from src.training.transformers.trainer import (
    load_transformers_model,
    apply_peft_lora,
    create_transformers_trainer,
)


os.environ["WANDB_WATCH"] = "false"
os.environ["WANDB_LOG_MODEL"] = "false"


def run_transformers_training_pipeline():
    wandb.init(
        project="interview-star-finetuning",
        name=settings.WANDB_RUN_NAME,
        config={
            "base_model": "Qwen/Qwen3-4B",
            "dataset": "dataset-v1-refine",
            "finetuning_method": "QLoRA",
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
    )

    try:
        model, tokenizer = load_transformers_model()

        model = apply_peft_lora(model)

        trainer = create_transformers_trainer(model, tokenizer)

        trainer.train()

        best_checkpoint = trainer.state.best_model_checkpoint

        print("=" * 50)
        print("Training completed")
        print(f"Best checkpoint: {best_checkpoint}")
        print(f"Best eval loss: {trainer.state.best_metric}")
        print("=" * 50)

        return best_checkpoint
    finally:
        del trainer
        del model
        del tokenizer

        gc.collect()
        torch.cuda.empty_cache()

        wandb.finish()


if __name__ == "__main__":
    run_transformers_training_pipeline()