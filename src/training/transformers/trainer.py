import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model
from trl import SFTTrainer, SFTConfig

from src.config.config import settings

from src.training.common.dataset import load_datasets
from src.training.common.lora_config import get_peft_lora_config
from src.training.common.save_model import SaveEpochCallback


def load_transformers_model():
    tokenizer = AutoTokenizer.from_pretrained(
        settings.MODEL_NAME
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        settings.MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": 0},
    )

    return model, tokenizer


def apply_peft_lora(model):
    model = get_peft_model(
        model,
        get_peft_lora_config()
    )

    model.gradient_checkpointing_enable()

    model.config.use_cache = False
    model.enable_input_require_grads()

    model.print_trainable_parameters()

    return model


def create_transformers_trainer(model, tokenizer):
    datasets = load_datasets(
        tokenizer,
        settings.TRAIN_PATH,
        settings.VALID_PATH,
    )

    training_args = SFTConfig(
        output_dir=settings.OUTPUT_DIR,
        # label_names=["labels"],
        run_name=settings.WANDB_RUN_NAME,
        num_train_epochs=settings.NUM_EPOCHS,
        per_device_train_batch_size=settings.BATCH_SIZE,
        gradient_accumulation_steps=settings.GRAD_ACCUMULATION,
        learning_rate=settings.LEARNING_RATE,
        logging_steps=50,
        # eval_strategy="steps",
        # eval_steps=211,
        # save_strategy="steps",
        # save_steps=211,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # save_total_limit=3,
        bf16=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        warmup_ratio=0.03,
        max_length=settings.MAX_SEQ_LENGTH,
        packing=False,
        report_to="wandb",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        args=training_args,
    )

    # trainer.add_callback(
    #     SaveEpochCallback(
    #         tokenizer,
    #         settings.OUTPUT_DIR,
    #     )
    # )

    return trainer