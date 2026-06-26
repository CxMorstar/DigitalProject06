from pathlib import Path


def test_required_paths_exist():
    root = Path(__file__).resolve().parents[1]
    required = [
        root / "data" / "raw",
        root / "configs" / "default.yaml",
        root / "scripts" / "churn_analysis.py",
        root / "notebooks" / "churn_analysis.ipynb",
    ]
    for path in required:
        assert path.exists(), f"Missing required path: {path}"
