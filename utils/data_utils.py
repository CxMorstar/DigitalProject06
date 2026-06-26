from pathlib import Path


def ensure_path(path: str | Path) -> Path:
    """Ensure parent directory exists and return resolved path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
