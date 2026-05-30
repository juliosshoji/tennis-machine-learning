"""Model training helpers for tennis match outcome prediction."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import spmatrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def train_decision_tree(
    X_train: np.ndarray | spmatrix,
    y_train: np.ndarray,
    max_depth: int = 10,
    min_samples_split: int = 5,
) -> DecisionTreeClassifier:
    """Train a decision tree classifier for match outcome prediction."""
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def train_neural_network(
    X_train: np.ndarray | spmatrix,
    y_train: np.ndarray,
    hidden_layer_sizes: tuple[int, ...] = (100, 50),
    max_iter: int = 1000,
) -> MLPClassifier:
    """Train a multilayer perceptron (neural network) classifier."""
    # Check if we have enough samples for early stopping
    n_samples = X_train.shape[0] if hasattr(X_train, 'shape') else len(X_train)
    use_early_stopping = n_samples > 10  # Need at least 10 samples for validation split
    
    kwargs = {
        "hidden_layer_sizes": hidden_layer_sizes,
        "max_iter": max_iter,
        "random_state": 42,
        "early_stopping": use_early_stopping,
    }
    if use_early_stopping:
        kwargs["validation_fraction"] = 0.1
    
    model = MLPClassifier(**kwargs)
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: np.ndarray | spmatrix,
    y_train: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = 15,
) -> RandomForestClassifier:
    """Train a random forest classifier for enhanced predictions."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_kmeans_clustering(
    X_train: np.ndarray | spmatrix,
    n_clusters: int = 3,
) -> KMeans:
    """Train K-means clustering to discover match playing styles."""
    # Convert sparse matrix to dense if needed for clustering
    if spmatrix and isinstance(X_train, spmatrix):
        X_train = X_train.toarray()
    
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    model.fit(X_train)
    return model


def evaluate_model(
    model: Any,
    X_test: np.ndarray | spmatrix,
    y_test: np.ndarray,
    model_name: str = "Model",
) -> dict[str, float]:
    """Evaluate model performance on test data."""
    y_pred = model.predict(X_test)
    
    # Convert sparse matrix to dense if needed
    if spmatrix and isinstance(X_test, spmatrix):
        X_test = X_test.toarray()
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    
    # Calculate ROC AUC if model has probability predictions and both classes present
    try:
        # Check if both classes are present in y_test
        if len(np.unique(y_test)) > 1:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = roc_auc_score(y_test, y_pred_proba)
        else:
            metrics["roc_auc"] = None
    except (AttributeError, IndexError, ValueError):
        metrics["roc_auc"] = None
    
    return metrics


def get_feature_importance(
    model: Any,
    feature_names: list[str],
    top_n: int = 20,
) -> list[tuple[str, float]]:
    """Extract and return top N important features from the model."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return []
    
    # Get indices of top N features
    top_indices = np.argsort(importances)[-top_n:][::-1]
    
    # Return feature names and importances
    return [
        (feature_names[i], float(importances[i]))
        for i in top_indices
        if i < len(feature_names)
    ]


def save_model(model: Any, path: str | Path) -> None:
    """Persist the trained model using pickle for complex sklearn models."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    # Use pickle for sklearn models
    with open(destination, "wb") as f:
        pickle.dump(model, f)


def save_model_json(model_dict: dict, path: str | Path) -> None:
    """Persist model metadata as JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(model_dict, indent=2), encoding="utf-8")


def train_mean_regressor(features: list[list[float]], targets: list[float]) -> dict[str, str | float]:
    """Train a simple baseline model that predicts the mean target value."""
    if not features or not targets:
        raise ValueError("features and targets must not be empty")

    if len(features) != len(targets):
        raise ValueError("features and targets must have the same length")

    return {"model_type": "mean_regressor", "target_mean": sum(targets) / len(targets)}
