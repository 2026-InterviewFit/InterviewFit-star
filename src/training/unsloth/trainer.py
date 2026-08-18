import os

os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"
os.environ["UNSLOTH_DISABLE_XFORMERS"] = "1"

from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported

from trl import SFTTrainer
from transformers import TrainingArguments

from src.config.config import settings

from src.training.common.dataset import load_datasets
from src.training.common.lora_config import get_unsloth_lora_config
from src.training.common.save_model import SaveEpochCallback


MAX_SEQ_LENGTH = 2048


def load_unsloth_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=settings.MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
        attn_implementation="sdpa",
    )

    # print(tokenizer.chat_template)
    print("{% generation %}" in tokenizer.chat_template)

    return model, tokenizer


def apply_unsloth_lora(model):
    lora = get_unsloth_lora_config()

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora["r"],
        target_modules=lora["target_modules"],
        lora_alpha=lora["lora_alpha"],
        lora_dropout=lora["lora_dropout"],
        bias=lora["bias"],
        use_gradient_checkpointing="unsloth",
    )

    model.print_trainable_parameters()

    return model


def create_unsloth_trainer(model, tokenizer):
    datasets = load_datasets(
        tokenizer,
        settings.TRAIN_PATH,
        settings.VALID_PATH,
    )

    training_args = TrainingArguments(
        output_dir=settings.OUTPUT_DIR,
        num_train_epochs=settings.NUM_EPOCHS,
        per_device_train_batch_size=settings.BATCH_SIZE,
        gradient_accumulation_steps=settings.GRAD_ACCUMULATION,
        learning_rate=settings.LEARNING_RATE,
        logging_steps=10,
        # save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=3,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        seed=42,
        report_to="wandb",
    )

    trainer  = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        # tokenizer=tokenizer,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
        assistant_only_loss=True,  # 답변 부분만 loss를 적용 (넣을지 말지 고민해보기)
        args=training_args,
    )

    trainer.add_callback(
        SaveEpochCallback(
            tokenizer,
            settings.SAVE_MODEL_PATH
        )
    )

    return trainer
