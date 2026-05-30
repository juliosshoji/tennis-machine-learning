"""Model loading and prediction helpers."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import spmatrix


def load_model(path: str | Path) -> Any:
    """Load a pickled model file."""
    with open(Path(path), "rb") as f:
        return pickle.load(f)


def load_model_json(path: str | Path) -> dict:
    """Load a JSON model metadata file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def predict(model: Any, feature_rows: list[list[float]]) -> list[float]:
    """Generate predictions for input rows using a trained sklearn model."""
    if not hasattr(model, "predict"):
        raise ValueError("Model must have a predict method")
    
    predictions = model.predict(feature_rows)
    return [float(p) for p in predictions]


def predict_proba(model: Any, feature_rows: list[list[float]]) -> list[list[float]]:
    """Generate probability predictions for input rows (for classification models)."""
    if not hasattr(model, "predict_proba"):
        raise ValueError("Model must have a predict_proba method")
    
    predictions = model.predict_proba(feature_rows)
    return predictions.tolist()


def predict_mean_regressor(model: dict[str, str | float], feature_rows: list[list[float]]) -> list[float]:
    """Generate predictions for baseline mean regressor."""
    if model.get("model_type") != "mean_regressor":
        raise ValueError("unsupported model_type")

    target_mean = float(model["target_mean"])
    return [target_mean for _ in feature_rows]
