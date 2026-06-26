from pathlib import Path

import pandas as pd


def load_telco_dataset(path: str | Path) -> pd.DataFrame:
    """Load the Telco customer churn CSV dataset."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")
    return pd.read_csv(p)
