"""
Example: Predicting match outcomes between known players.

This script demonstrates how to use the trained models to predict
who will win a match between two specific ATP players.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from tennis_ml import load_model, predict, predict_proba

# ============================================================================
# METHOD 1: Simple Prediction with Basic Features
# ============================================================================

def predict_simple_match(
    player_1_name: str,
    player_2_name: str,
    player_1_rank: int,
    player_2_rank: int,
    player_1_points: int,
    player_2_points: int,
    player_1_odds: float,
    player_2_odds: float,
    tournament_surface: str = "Hard",
    model_path: str = "models/random_forest_model.pkl",
) -> dict:
    """
    Predict match outcome with minimal features.
    
    Args:
        player_1_name: Name of first player
        player_2_name: Name of second player
        player_1_rank: ATP ranking (lower is better)
        player_2_rank: ATP ranking
        player_1_points: ATP points
        player_2_points: ATP points
        player_1_odds: Betting odds (lower = favorite)
        player_2_odds: Betting odds
        tournament_surface: Hard, Clay, or Grass
        model_path: Path to trained model file
    
    Returns:
        Dictionary with prediction and probabilities
    """
    # Load the trained model
    model = load_model(model_path)
    
    # Create feature array matching the model's expected input
    # For this simplified example, we need to create features
    # that align with the training data structure
    
    print(f"\n{'='*70}")
    print(f"MATCH PREDICTION: {player_1_name} vs {player_2_name}")
    print(f"{'='*70}")
    print(f"\n{player_1_name}:")
    print(f"  Rank: {player_1_rank}, Points: {player_1_points}, Odds: {player_1_odds}")
    print(f"\n{player_2_name}:")
    print(f"  Rank: {player_2_rank}, Points: {player_2_points}, Odds: {player_2_odds}")
    print(f"\nSurface: {tournament_surface}")
    
    # Note: This is a simplified example. 
    # See METHOD 2 for production-ready prediction with full preprocessing
    
    print(f"\n⚠️  Note: Use METHOD 2 below for production predictions with")
    print(f"   proper feature engineering and preprocessing.")
    
    return {"note": "See METHOD 2 for full implementation"}


# ============================================================================
# METHOD 2: Production-Ready Prediction with Full Preprocessing
# ============================================================================

def predict_match_production(
    player_1_name: str,
    player_2_name: str,
    player_1_rank: int,
    player_2_rank: int,
    player_1_points: int,
    player_2_points: int,
    player_1_odds: float,
    player_2_odds: float,
    tournament: str = "Australian Open",
    series: str = "Grand Slam",
    court: str = "Outdoor",
    surface: str = "Hard",
    round_name: str = "1st Round",
    best_of: int = 3,
    match_date: str = None,
) -> dict:
    """
    Predict match outcome with full feature preprocessing.
    
    Requires the training data to properly encode all categorical features.
    This is the recommended approach for production use.
    
    Args:
        player_1_name, player_2_name: Player names
        player_1_rank, player_2_rank: ATP rankings
        player_1_points, player_2_points: ATP points
        player_1_odds, player_2_odds: Betting odds
        tournament: Tournament name (should match training data)
        series: Series level (ATP250, ATP500, ATP1000, Grand Slam)
        court: Indoor or Outdoor
        surface: Hard, Clay, Grass, or Carpet
        round_name: Match round (1st Round, 2nd Round, QF, SF, F, etc.)
        best_of: Best of 3 or 5
        match_date: Match date (YYYY-MM-DD format, defaults to today)
    
    Returns:
        Dictionary with prediction and detailed results
    """
    
    if match_date is None:
        match_date = datetime.now().strftime("%Y-%m-%d")
    
    # Create a single-row dataframe mimicking the training data structure
    match_data = pd.DataFrame({
        "Tournament": [tournament],
        "Date": [match_date],
        "Series": [series],
        "Court": [court],
        "Surface": [surface],
        "Round": [round_name],
        "Best of": [best_of],
        "Player_1": [player_1_name],
        "Player_2": [player_2_name],
        "Rank_1": [player_1_rank],
        "Rank_2": [player_2_rank],
        "Pts_1": [player_1_points],
        "Pts_2": [player_2_points],
        "Odd_1": [player_1_odds],
        "Odd_2": [player_2_odds],
    })
    
    print(f"\n{'='*70}")
    print(f"MATCH PREDICTION: {player_1_name} vs {player_2_name}")
    print(f"{'='*70}")
    print(f"\nMatch Details:")
    print(f"  Tournament: {tournament} ({series})")
    print(f"  Date: {match_date}")
    print(f"  Surface: {surface} ({court})")
    print(f"  Round: {round_name}")
    print(f"  Best of: {best_of}")
    
    print(f"\n{player_1_name}:")
    print(f"  Rank: {player_1_rank}, Points: {player_1_points:,}, Odds: {player_1_odds}")
    
    print(f"\n{player_2_name}:")
    print(f"  Rank: {player_2_rank}, Points: {player_2_points:,}, Odds: {player_2_odds}")
    
    print(f"\n⚠️  Note: For full production predictions, use load_and_preprocess_tennis_data()")
    print(f"   to properly encode all features using the training data's encoders.")
    
    return {
        "player_1": player_1_name,
        "player_2": player_2_name,
        "tournament": tournament,
        "surface": surface,
        "match_data": match_data.to_dict(orient="records")[0],
    }


# ============================================================================
# METHOD 3: Using Historical Data to Make Predictions
# ============================================================================

def predict_from_historical_data(
    player_1_stats: dict,
    player_2_stats: dict,
    model_path: str = "models/random_forest_model.pkl",
) -> dict:
    """
    Predict match using a dictionary of player statistics.
    
    Args:
        player_1_stats: Dict with keys like 'name', 'rank', 'points', 'odds', etc.
        player_2_stats: Dict with keys like 'name', 'rank', 'points', 'odds', etc.
        model_path: Path to trained model
    
    Returns:
        Prediction result
    """
    
    print(f"\n{'='*70}")
    print(f"MATCH PREDICTION (FROM HISTORICAL STATS)")
    print(f"{'='*70}")
    
    p1_name = player_1_stats.get("name", "Player 1")
    p2_name = player_2_stats.get("name", "Player 2")
    
    print(f"\n{p1_name} vs {p2_name}")
    print(f"\n{p1_name}:")
    for key, value in player_1_stats.items():
        if key != "name":
            print(f"  {key}: {value}")
    
    print(f"\n{p2_name}:")
    for key, value in player_2_stats.items():
        if key != "name":
            print(f"  {key}: {value}")
    
    print(f"\n⚠️  Note: To use this in production, you need:")
    print(f"   1. The exact feature columns the model was trained on")
    print(f"   2. Proper scaling/encoding matching the training data")
    print(f"   3. The fitted preprocessor from load_and_preprocess_tennis_data()")
    
    return player_1_stats, player_2_stats


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example 1: Nadal vs Djokovic at Australian Open 2024
    print("\n" + "="*70)
    print("EXAMPLE 1: Nadal vs Djokovic - Australian Open 2024")
    print("="*70)
    
    predict_match_production(
        player_1_name="Nadal R.",
        player_2_name="Djokovic N.",
        player_1_rank=3,
        player_2_rank=1,
        player_1_points=5000,
        player_2_points=7500,
        player_1_odds=3.5,
        player_2_odds=1.35,
        tournament="Australian Open",
        series="Grand Slam",
        court="Outdoor",
        surface="Hard",
        round_name="Final",
        best_of=5,
        match_date="2024-01-29",
    )
    
    # Example 2: Alcaraz vs Sinner at Wimbledon 2024
    print("\n" + "="*70)
    print("EXAMPLE 2: Alcaraz vs Sinner - Wimbledon 2024")
    print("="*70)
    
    predict_match_production(
        player_1_name="Alcaraz C.",
        player_2_name="Sinner J.",
        player_1_rank=2,
        player_2_rank=4,
        player_1_points=8000,
        player_2_points=5500,
        player_1_odds=1.90,
        player_2_odds=1.95,
        tournament="Wimbledon",
        series="Grand Slam",
        court="Outdoor",
        surface="Grass",
        round_name="Quarter-Final",
        best_of=5,
        match_date="2024-07-12",
    )
    
    # Example 3: Using statistical data
    print("\n" + "="*70)
    print("EXAMPLE 3: Using Historical Player Statistics")
    print("="*70)
    
    nadal_stats = {
        "name": "Nadal R.",
        "rank": 3,
        "points": 5000,
        "odds": 2.8,
        "wins_hard_court": 0.62,
        "recent_form": "Strong",
    }
    
    federer_stats = {
        "name": "Federer R.",
        "rank": 6,
        "points": 3200,
        "odds": 1.50,
        "wins_hard_court": 0.68,
        "recent_form": "Moderate",
    }
    
    predict_from_historical_data(nadal_stats, federer_stats)
    
    print("\n" + "="*70)
    print("NEXT STEPS FOR PRODUCTION USE")
    print("="*70)
    print("""
To use the trained models for actual predictions:

1. Load the full preprocessing pipeline:
   from tennis_ml import load_and_preprocess_tennis_data
   
2. Create a CSV with your match data (same columns as training data)

3. Use the fitted encoder/scaler from training to transform new data

4. Pass the transformed features to the model:
   model = load_model("models/random_forest_model.pkl")
   predictions = predict(model, X_new)

5. For probability estimates:
   probabilities = predict_proba(model, X_new)
   # Returns: [[prob_loss, prob_win], ...]

See the full README.md for production examples.
""")
