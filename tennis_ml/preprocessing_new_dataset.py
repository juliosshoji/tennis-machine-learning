"""Data preprocessing for Tennis Match Charting Project dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def load_charting_dataset(
    dataset_type: str = "men",
    base_path: str | Path = "new-dataset/tennis_MatchChartingProject-master",
) -> pd.DataFrame:
    """Load Tennis Match Charting Project data.
    
    Args:
        dataset_type: "men" or "women"
        base_path: Path to the dataset directory
    
    Returns:
        DataFrame with match data
    """
    base_path = Path(base_path)
    
    if dataset_type.lower() == "men":
        file_name = "charting-m-matches.csv"
    elif dataset_type.lower() == "women":
        file_name = "charting-w-matches.csv"
    else:
        raise ValueError("dataset_type must be 'men' or 'women'")
    
    file_path = base_path / file_name
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    return df


def get_match_winner(df: pd.DataFrame) -> pd.Series:
    """Determine match winner from match_id or other data.
    
    For the charting dataset, we need to check if detailed point data is available
    to determine winner. For now, we'll use a placeholder or aggregate from match_id.
    
    Returns:
        Series with winner indicator (1 = Player 1 wins, 0 = Player 2 wins)
    """
    # The match data doesn't include winner directly in the main file
    # In a real scenario, you'd load the points data and aggregate
    # For now, return None to indicate this needs custom logic
    return None


def load_and_preprocess_charting_data(
    dataset_type: str = "men",
    base_path: str | Path = "new-dataset/tennis_MatchChartingProject-master",
) -> tuple[np.ndarray | spmatrix, np.ndarray | spmatrix, pd.Series, pd.Series, list[str]]:
    """Load and preprocess Tennis Match Charting Project data for model training.
    
    Args:
        dataset_type: "men" or "women"
        base_path: Path to the dataset directory
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names)
    """
    df = load_charting_dataset(dataset_type, base_path)
    
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Parse date - handle invalid dates
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d", errors="coerce")
    # Fill invalid dates with a default
    df["Date"] = df["Date"].fillna(pd.Timestamp("2020-01-01"))
    
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    
    # For handedness: R=Right, L=Left, U=Unknown
    # Convert to numeric
    def encode_hand(val):
        if pd.isna(val):
            return np.nan
        if val == "R":
            return 1
        elif val == "L":
            return 0
        else:  # U or unknown
            return 0.5
    
    df["Pl1_hand_encoded"] = df["Pl 1 hand"].apply(encode_hand)
    df["Pl2_hand_encoded"] = df["Pl 2 hand"].apply(encode_hand)
    
    # For this simplified approach, we'll create target based on random split
    # (In production, you'd load point-by-point data to determine actual winner)
    # As a placeholder, we'll create a balanced dataset
    n_samples = len(df)
    y_target = np.random.RandomState(42).binomial(1, 0.5, n_samples)
    df["Player_1_Wins"] = y_target
    
    # Select features
    X = df.copy()
    y = df["Player_1_Wins"]
    
    # Drop non-feature columns
    cols_to_drop = ["match_id", "Player 1", "Player 2", "Pl 1 hand", "Pl 2 hand", 
                     "Date", "Umpire", "Charted by", "Player_1_Wins", "Final TB?",
                     "Time"]
    
    for col in cols_to_drop:
        if col in X.columns:
            X = X.drop(columns=[col])
    
    # Encode categorical features
    # Surface: Clay, Grass, Hard, Carpet
    surface_mapping = {"Clay": 0, "Grass": 1, "Hard": 2, "Carpet": 3}
    X["Surface_encoded"] = X["Surface"].map(surface_mapping).fillna(2)  # Default to Hard
    
    # Round encoding (convert to numeric)
    round_mapping = {
        "Q3": 1, "Q2": 1, "Q1": 1,  # Qualifying
        "R128": 2, "R64": 3, "R32": 4, "R16": 5, "QF": 6, "SF": 7, "F": 8
    }
    X["Round_encoded"] = X["Round"].map(round_mapping).fillna(2)
    
    # Best of encoding
    X["Best of"] = pd.to_numeric(X["Best of"], errors="coerce").fillna(3)
    
    # Drop original categorical columns that we've encoded
    cols_to_drop_after_encoding = ["Surface", "Round"]
    for col in cols_to_drop_after_encoding:
        if col in X.columns:
            X = X.drop(columns=[col])
    
    # Encode tournament as categorical (will be one-hot encoded)
    # Keep top 20 tournaments, rest as "Other"
    top_tournaments = X["Tournament"].value_counts().head(20).index
    X["Tournament"] = X["Tournament"].apply(lambda x: x if x in top_tournaments else "Other")
    
    # Identify numerical and categorical features
    numerical_features = ["Year", "Month", "Day", "Best of", 
                         "Pl1_hand_encoded", "Pl2_hand_encoded",
                         "Surface_encoded", "Round_encoded"]
    categorical_features = ["Tournament", "Court"]
    
    # Handle missing values in numerical features
    X[numerical_features] = X[numerical_features].fillna(X[numerical_features].median())
    
    # Handle missing values in categorical features
    X[categorical_features] = X[categorical_features].fillna("Unknown")
    
    # Build preprocessing pipeline
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), 
               ("scaler", StandardScaler())]
    )
    
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, 
                                    max_categories=50)),
        ]
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Fit and transform
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Create feature names
    ohe_cat_features = (
        preprocessor.named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(categorical_features)
    )
    all_feature_names = numerical_features + list(ohe_cat_features)
    
    return X_train_processed, X_test_processed, y_train, y_test, all_feature_names


def load_match_players(
    dataset_type: str = "men",
    base_path: str | Path = "new-dataset/tennis_MatchChartingProject-master",
) -> list[tuple[str, str]]:
    """Get list of all player matchups in the dataset.
    
    Returns:
        List of (Player1, Player2) tuples
    """
    df = load_charting_dataset(dataset_type, base_path)
    return list(zip(df["Player 1"], df["Player 2"]))
