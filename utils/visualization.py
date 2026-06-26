from pathlib import Path

import matplotlib.pyplot as plt


def save_figure(path: str | Path, dpi: int = 220) -> None:
    """Save current matplotlib figure to the target path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(p, dpi=dpi, bbox_inches="tight")
