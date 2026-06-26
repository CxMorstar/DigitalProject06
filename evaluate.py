"""Project evaluation entry point."""

from pathlib import Path

import pandas as pd


if __name__ == "__main__":
    metrics_path = Path("output/model_metrics.csv")
    if not metrics_path.exists():
        raise FileNotFoundError("Run training first: python train.py")
    print(pd.read_csv(metrics_path))
