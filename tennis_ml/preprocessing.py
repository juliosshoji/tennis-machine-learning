"""Data preprocessing helpers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


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


def load_and_preprocess_tennis_data(file_path: str | Path):
    """Load and preprocess tennis data for model training."""
    df = pd.read_csv(file_path)

    df.replace([-1, -1.0], np.nan, inplace=True)
    df["Player_1_Wins"] = (df["Winner"] == df["Player_1"]).astype(int)

    cols_to_drop = ["Winner", "Score"]
    df = df.drop(columns=cols_to_drop)

    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df = df.drop(columns=["Date"])

    X = df.drop(columns=["Player_1_Wins"])
    y = df["Player_1_Wins"]

    numerical_features = [
        "Rank_1",
        "Rank_2",
        "Pts_1",
        "Pts_2",
        "Odd_1",
        "Odd_2",
        "Best of",
        "Year",
        "Month",
        "Day",
    ]
    categorical_features = ["Tournament", "Series", "Court", "Surface", "Round"]

    player_encoder = LabelEncoder()
    all_players = pd.concat([X["Player_1"], X["Player_2"]]).unique()
    player_encoder.fit(all_players)

    X["Player_1"] = player_encoder.transform(X["Player_1"])
    X["Player_2"] = player_encoder.transform(X["Player_2"])

    numerical_features.extend(["Player_1", "Player_2"])

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    ohe_cat_features = (
        preprocessor.named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(categorical_features)
    )
    all_feature_names = numerical_features + list(ohe_cat_features)

    return X_train_processed, X_test_processed, y_train, y_test, all_feature_names
