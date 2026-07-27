from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)

from peft import get_peft_model

from trl import SFTTrainer

from src.config.config import settings

from src.training.common.dataset import load_datasets
from src.training.common.lora_config import get_peft_lora_config


MAX_SEQ_LENGTH = 4096


def load_transformers_model():
    tokenizer = AutoTokenizer.from_pretrained(
        settings.MODEL_NAME
    )

    model = AutoModelForCausalLM.from_pretrained(
        settings.MODEL_NAME,
        device_map="auto",
        torch_dtype="auto",
    )

    return model, tokenizer


def apply_peft_lora(model):
    model = get_peft_model(
        model,
        get_peft_lora_config()
    )

    return model


def create_transformers_trainer(model, tokenizer):
    datasets = load_datasets(
        tokenizer,
        settings.TRAIN_PATH,
        settings.VALID_PATH,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
        args=TrainingArguments(
            output_dir=settings.OUTPUT_DIR,
            num_train_epochs=settings.NUM_EPOCHS,
            per_device_train_batch_size=settings.BATCH_SIZE,
            gradient_accumulation_steps=settings.GRAD_ACCUMULATION,
            learning_rate=settings.LEARNING_RATE,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            bf16=True,
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            warmup_ratio=0.03,
        ),
    )

    return trainer


def save_transformers_model(model, tokenizer):
    model.save_pretrained(
        settings.SAVE_MODEL_PATH
    )

    tokenizer.save_pretrained(
        settings.SAVE_MODEL_PATH
    )