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
    
    # ===== EXTRACT NEW FEATURES =====
    
    # 1. Day of week from date (Monday=0, Sunday=6)
    X["DayOfWeek"] = X["Date"].dt.dayofweek
    
    # 2. Match time features
    def extract_time_hour(time_str):
        """Extract hour from time string (e.g., '5pm' -> 17)"""
        if pd.isna(time_str) or time_str == "":
            return 12  # Default to noon if missing
        time_str = str(time_str).lower().strip()
        try:
            if "am" in time_str:
                hour = int(time_str.replace("am", "").strip())
                return hour if hour == 12 else hour
            elif "pm" in time_str:
                hour = int(time_str.replace("pm", "").strip())
                return 12 if hour == 12 else hour + 12
            else:
                return 12
        except:
            return 12
    
    X["Match_Hour"] = X["Time"].apply(extract_time_hour)
    
    # 3. Handedness matchup type (RR, RL, LR, LL, etc)
    def encode_matchup(hand1, hand2):
        """Encode handedness matchup type"""
        h1 = str(hand1).upper() if pd.notna(hand1) else "U"
        h2 = str(hand2).upper() if pd.notna(hand2) else "U"
        return f"{h1}{h2}"
    
    X["Handedness_Matchup"] = df.apply(
        lambda row: encode_matchup(row["Pl 1 hand"], row["Pl 2 hand"]), axis=1
    )
    
    # 4. Umpire presence (1 if umpire, 0 if not)
    X["Has_Umpire"] = (X["Umpire"].notna() & (X["Umpire"] != "")).astype(int)
    
    # 5. Charting quality (unique charting sources - more sources = more reliable)
    charting_quality = X["Charted by"].value_counts()
    X["Charting_Quality_Score"] = X["Charted by"].map(
        lambda x: min(charting_quality.get(x, 1), 10) if pd.notna(x) else 1
    )
    
    # 6. Final tiebreak indicator
    def encode_final_tb(val):
        """Encode Final TB? column"""
        if pd.isna(val) or val == "":
            return 0  # No data means no tiebreak
        val_str = str(val).upper()
        if val_str == "1":
            return 1  # Yes, went to final TB
        elif val_str == "0" or val_str == "A":
            return 0  # No final TB or retired
        else:
            return 0  # Default
    
    X["Final_Tiebreak"] = X["Final TB?"].apply(encode_final_tb)
    
    # 7. Tournament tier/level (Grand Slams are tier 1, Masters tier 2, etc)
    grand_slams = {"Australian Open", "French Open", "Wimbledon", "US Open", "Roland Garros"}
    masters_tournaments = {
        "Rome Masters", "Monte Carlo Masters", "Cincinnati Masters", 
        "Madrid Masters", "Miami Masters", "Canada Masters", "Shanghai Masters"
    }
    
    def tournament_tier(tournament_name):
        """Determine tournament tier"""
        if pd.isna(tournament_name):
            return 3
        t_name = str(tournament_name)
        if any(gs in t_name for gs in grand_slams):
            return 1  # Grand Slam
        elif any(m in t_name for m in masters_tournaments):
            return 2  # Masters 1000
        elif "500" in t_name or "ATP 500" in t_name:
            return 3  # ATP 500
        elif "250" in t_name or "ATP 250" in t_name:
            return 4  # ATP 250
        else:
            return 5  # Other
    
    X["Tournament_Tier"] = X["Tournament"].apply(tournament_tier)
    
    # 8. Court type (indoor/outdoor derived from court name)
    def classify_court_type(court_name):
        """Classify court as indoor/outdoor based on name"""
        if pd.isna(court_name):
            return 0  # Unknown
        court_str = str(court_name).lower()
        indoor_keywords = ["arena", "indoor", "coliseum", "colosseum", "center", "centre", "auditorium"]
        if any(keyword in court_str for keyword in indoor_keywords):
            return 1  # Indoor
        else:
            return 0  # Outdoor or unknown
    
    X["Is_Indoor_Court"] = X["Court"].apply(classify_court_type)
    
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
        "R128": 2, "R64": 3, "R32": 4, "R16": 5, "QF": 6, "SF": 7, "F": 8,
        "BR": 6, "RR": 5, "PO": 8, "PQ": 1  # Additional rounds
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
    # NEW FEATURES ADDED:
    # - DayOfWeek: day of week (0-6)
    # - Match_Hour: hour of day (0-23)
    # - Has_Umpire: binary indicator if umpire present
    # - Charting_Quality_Score: quality of charting data (1-10)
    # - Final_Tiebreak: binary indicator if match went to final TB
    # - Tournament_Tier: tier of tournament (1-5)
    # - Is_Indoor_Court: binary indicator if court is indoor
    # Total features now: 79 numerical + categorical one-hots
    
    numerical_features = ["Year", "Month", "Day", "DayOfWeek", "Match_Hour", "Best of", 
                         "Pl1_hand_encoded", "Pl2_hand_encoded",
                         "Surface_encoded", "Round_encoded", "Has_Umpire",
                         "Charting_Quality_Score", "Final_Tiebreak", "Tournament_Tier",
                         "Is_Indoor_Court"]
    categorical_features = ["Tournament", "Court", "Handedness_Matchup"]
    
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
