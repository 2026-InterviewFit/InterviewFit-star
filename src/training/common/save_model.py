from transformers import TrainerCallback
from pathlib import Path


class SaveEpochCallback(TrainerCallback):
    def __init__(self, tokenizer, save_dir):
        self.tokenizer = tokenizer
        self.save_dir = Path(save_dir)

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs["model"]

        epoch = int(state.epoch)

        save_path = self.save_dir / f"epoch-{epoch}"
        save_path.mkdir(
            parents=True,
            exist_ok=True
        )

        model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        print(f"Saved: {save_path}")