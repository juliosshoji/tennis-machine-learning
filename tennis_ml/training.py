"""Model training helpers."""

from __future__ import annotations

import json
from pathlib import Path


def train_mean_regressor(features: list[list[float]], targets: list[float]) -> dict[str, str | float]:
    """Train a simple baseline model that predicts the mean target value."""
    if not features or not targets:
        raise ValueError("features and targets must not be empty")

    if len(features) != len(targets):
        raise ValueError("features and targets must have the same length")

    return {"model_type": "mean_regressor", "target_mean": sum(targets) / len(targets)}


def save_model(model: dict[str, str | float], path: str | Path) -> None:
    """Persist the trained model as JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(model, indent=2), encoding="utf-8")
