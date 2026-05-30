"""Data preprocessing helpers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


def load_csv_rows(path: str | Path) -> list[dict[str, str]]:
    """Load a CSV file into a list of dictionaries."""
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def preprocess_rows(
    rows: Iterable[dict[str, str]],
    feature_columns: list[str],
    target_column: str,
) -> tuple[list[list[float]], list[float]]:
    """Convert selected feature and target columns to float arrays."""
    features: list[list[float]] = []
    targets: list[float] = []

    for row in rows:
        if any(row.get(column, "") == "" for column in feature_columns + [target_column]):
            continue

        feature_values = [float(row[column]) for column in feature_columns]
        target_value = float(row[target_column])

        features.append(feature_values)
        targets.append(target_value)

    return features, targets
