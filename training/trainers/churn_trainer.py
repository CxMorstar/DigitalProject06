"""Trainer adapter to run the current churn pipeline."""

from scripts.churn_analysis import main


def run_training_pipeline() -> None:
    """Run the end-to-end training/evaluation pipeline."""
    main()
