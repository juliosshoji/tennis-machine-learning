"""Model loading and prediction helpers."""

from __future__ import annotations

import json
from pathlib import Path


def load_model(path: str | Path) -> dict[str, float]:
    """Load a JSON model file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def predict(model: dict[str, float], feature_rows: list[list[float]]) -> list[float]:
    """Generate predictions for input rows using the trained model."""
    if model.get("model_type") != "mean_regressor":
        raise ValueError("unsupported model_type")

    target_mean = float(model["target_mean"])
    return [target_mean for _ in feature_rows]
