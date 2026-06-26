from pathlib import Path

import joblib


def save_model(obj, path: str | Path) -> None:
    """Save model object with joblib."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, p)


def load_model(path: str | Path):
    """Load model object with joblib."""
    return joblib.load(path)
