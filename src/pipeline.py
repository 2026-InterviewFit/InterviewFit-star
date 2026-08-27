from src.preprocess.run_preprocess import run_preprocess_pipeline
from src.training.transformers.run_transformers_training import run_transformers_training_pipeline
from src.evaluation.evaluate import run_evaluation

from src.config.config import settings


def run_full_pipeline():
    print("=" * 50)
    print("1. Preprocess")
    print("=" * 50)

    # run_preprocess_pipeline(
    #     settings.REVIEWED_TRAIN_MERGED_PATH,
    #     settings.REVIEWED_VALIDATION_MERGED_PATH
    # )

    run_preprocess_pipeline(
        settings.FILTERED_VALIDATED_TRAIN_PATH,
        settings.FILTERED_VALIDATED_VALIDATION_PATH,
    )

    print("=" * 50)
    print("2. Training")
    print("=" * 50)

    best_checkpoint = run_transformers_training_pipeline()

    print("=" * 50)
    print("3. Evaluation")
    print("=" * 50)

    run_evaluation(best_checkpoint)

    print("=" * 50)
    print("Pipeline completed")
    print("=" * 50)


if __name__ == "__main__":
    run_full_pipeline()