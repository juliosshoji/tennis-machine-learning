#!/usr/bin/env python3
"""
Quick Prediction Tool - Interactive Match Predictor

Usage:
    python quick_predict.py                    # Interactive mode
    python quick_predict.py --player1 Nadal --player2 Djokovic --odds1 3.5 --odds2 1.35
"""

import argparse
import sys
from datetime import datetime


def get_user_input():
    """Interactive mode: prompt user for match details."""
    
    print("\n" + "="*70)
    print("TENNIS MATCH PREDICTOR - INTERACTIVE MODE")
    print("="*70)
    
    # Player details
    player_1 = input("\nPlayer 1 name: ").strip()
    player_2 = input("Player 2 name: ").strip()
    
    # Odds
    while True:
        try:
            odds_1 = float(input(f"\n{player_1} betting odds (decimal): "))
            odds_2 = float(input(f"{player_2} betting odds (decimal): "))
            break
        except ValueError:
            print("Please enter valid numbers for odds")
    
    # Ranking (optional)
    try:
        rank_1 = int(input(f"\n{player_1} ATP ranking (or press Enter to skip): ") or "0")
        rank_2 = int(input(f"{player_2} ATP ranking (or press Enter to skip): ") or "0")
    except ValueError:
        rank_1 = rank_2 = 0
    
    # Points (optional)
    try:
        points_1 = int(input(f"\n{player_1} ATP points (or press Enter to skip): ") or "0")
        points_2 = int(input(f"{player_2} ATP points (or press Enter to skip): ") or "0")
    except ValueError:
        points_1 = points_2 = 0
    
    # Surface
    print("\nTournament surface (default: Hard):")
    print("  1. Hard")
    print("  2. Clay")
    print("  3. Grass")
    surface_choice = input("Select (1-3, or press Enter for Hard): ").strip()
    surfaces = {"1": "Hard", "2": "Clay", "3": "Grass"}
    surface = surfaces.get(surface_choice, "Hard")
    
    return {
        "player_1": player_1,
        "player_2": player_2,
        "odds_1": odds_1,
        "odds_2": odds_2,
        "rank_1": rank_1,
        "rank_2": rank_2,
        "points_1": points_1,
        "points_2": points_2,
        "surface": surface,
    }


def get_cli_args():
    """Parse command-line arguments."""
    
    parser = argparse.ArgumentParser(
        description="Quick Tennis Match Predictor"
    )
    parser.add_argument("--player1", help="Player 1 name")
    parser.add_argument("--player2", help="Player 2 name")
    parser.add_argument("--odds1", type=float, help="Player 1 odds")
    parser.add_argument("--odds2", type=float, help="Player 2 odds")
    parser.add_argument("--rank1", type=int, default=0, help="Player 1 ranking")
    parser.add_argument("--rank2", type=int, default=0, help="Player 2 ranking")
    parser.add_argument("--points1", type=int, default=0, help="Player 1 points")
    parser.add_argument("--points2", type=int, default=0, help="Player 2 points")
    parser.add_argument("--surface", default="Hard", help="Court surface")
    parser.add_argument("--examples", action="store_true", help="Show example predictions")
    
    return parser.parse_args()


def convert_odds_to_probability(odds_1: float, odds_2: float) -> tuple:
    """Convert decimal odds to win probabilities."""
    total = (1.0 / odds_1) + (1.0 / odds_2)
    prob_1 = (1.0 / odds_1) / total
    prob_2 = (1.0 / odds_2) / total
    return prob_1, prob_2


def predict_from_ranking(rank_1: int, rank_2: int) -> tuple:
    """Simple ranking-based prediction."""
    import numpy as np
    
    if rank_1 == 0 or rank_2 == 0:
        return None, None
    
    # Player with lower rank (better) more likely to win
    rank_diff = rank_2 - rank_1
    prob_1 = 1.0 / (1.0 + np.exp(-rank_diff / 50.0))
    prob_2 = 1.0 - prob_1
    return prob_1, prob_2


def make_prediction(match_data: dict) -> dict:
    """Make ensemble prediction from match data."""
    
    player_1 = match_data["player_1"]
    player_2 = match_data["player_2"]
    odds_1 = match_data["odds_1"]
    odds_2 = match_data["odds_2"]
    rank_1 = match_data["rank_1"]
    rank_2 = match_data["rank_2"]
    surface = match_data["surface"]
    
    # Get predictions from different methods
    odds_probs = convert_odds_to_probability(odds_1, odds_2)
    
    prob_p1_odds, prob_p2_odds = odds_probs
    
    # Ranking prediction (if available)
    if rank_1 > 0 and rank_2 > 0:
        prob_p1_rank, prob_p2_rank = predict_from_ranking(rank_1, rank_2)
        # Average them
        prob_p1 = (prob_p1_odds + prob_p1_rank) / 2
        prob_p2 = (prob_p2_odds + prob_p2_rank) / 2
        methods_used = "odds + ranking"
    else:
        prob_p1 = prob_p1_odds
        prob_p2 = prob_p2_odds
        methods_used = "odds only"
    
    winner = player_1 if prob_p1 > prob_p2 else player_2
    confidence = max(prob_p1, prob_p2)
    
    return {
        "player_1": player_1,
        "player_2": player_2,
        "surface": surface,
        "prob_player1": prob_p1,
        "prob_player2": prob_p2,
        "prediction": player_1 if prob_p1 > 0.5 else player_2,
        "confidence": confidence,
        "methods": methods_used,
    }


def display_result(result: dict):
    """Display prediction result."""
    
    print("\n" + "="*70)
    print("PREDICTION RESULT")
    print("="*70)
    
    print(f"\nMatch: {result['player_1']} vs {result['player_2']}")
    print(f"Surface: {result['surface']}")
    print(f"Methods used: {result['methods']}")
    
    print(f"\n{result['player_1']:.<40} {result['prob_player1']:>6.1%}")
    print(f"{result['player_2']:.<40} {result['prob_player2']:>6.1%}")
    
    print(f"\n{'PREDICTION':.<40} {result['prediction']}")
    print(f"{'CONFIDENCE':.<40} {result['confidence']:>6.1%}")
    
    # Interpretation
    if result['confidence'] < 0.52:
        interpretation = "TOSS-UP (virtually even)"
    elif result['confidence'] < 0.60:
        interpretation = "SLIGHT EDGE"
    elif result['confidence'] < 0.70:
        interpretation = "MODERATE EDGE"
    elif result['confidence'] < 0.80:
        interpretation = "STRONG EDGE"
    else:
        interpretation = "OVERWHELMING FAVORITE"
    
    print(f"{'INTERPRETATION':.<40} {interpretation}")
    
    print("\n" + "="*70)
    print("DISCLAIMER: This is an informational tool only.")
    print("Do not use for actual betting without additional analysis.")
    print("="*70 + "\n")


def display_examples():
    """Show example predictions."""
    
    print("\n" + "="*70)
    print("EXAMPLE PREDICTIONS")
    print("="*70)
    
    examples = [
        {
            "player_1": "Alcaraz C.",
            "player_2": "Sinner J.",
            "odds_1": 1.90,
            "odds_2": 1.95,
            "rank_1": 2,
            "rank_2": 4,
            "surface": "Grass",
        },
        {
            "player_1": "Nadal R.",
            "player_2": "Djokovic N.",
            "odds_1": 3.5,
            "odds_2": 1.35,
            "rank_1": 3,
            "rank_2": 1,
            "surface": "Hard",
        },
        {
            "player_1": "Young P.",
            "player_2": "Veteran M.",
            "odds_1": 5.0,
            "odds_2": 1.15,
            "rank_1": 100,
            "rank_2": 15,
            "surface": "Clay",
        },
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\nExample {i}: {example['player_1']} vs {example['player_2']}")
        result = make_prediction(example)
        print(f"  → {result['prediction']} ({result['confidence']:.1%})")


def main():
    """Main function."""
    
    args = get_cli_args()
    
    # Show examples if requested
    if args.examples:
        display_examples()
        return
    
    # Check if we have command-line arguments
    if args.player1 and args.player2 and args.odds1 and args.odds2:
        # CLI mode
        match_data = {
            "player_1": args.player1,
            "player_2": args.player2,
            "odds_1": args.odds1,
            "odds_2": args.odds2,
            "rank_1": args.rank1,
            "rank_2": args.rank2,
            "points_1": args.points1,
            "points_2": args.points2,
            "surface": args.surface,
        }
    else:
        # Check if user wants examples
        if len(sys.argv) == 1:
            print("\n" + "="*70)
            print("TENNIS MATCH PREDICTOR")
            print("="*70)
            print("\nUsage:")
            print("  1. python quick_predict.py                    # Interactive mode")
            print("  2. python quick_predict.py --examples          # Show examples")
            print("  3. python quick_predict.py --player1 X --player2 Y --odds1 Z --odds2 W")
            print("                                                  # Quick prediction")
            
            choice = input("\nSelect mode (1-3 or press Enter for interactive): ").strip()
            
            if choice == "2" or choice.lower() == "examples":
                display_examples()
                return
            else:
                # Interactive mode
                match_data = get_user_input()
        else:
            # Missing required arguments
            print("Error: Please provide --player1, --player2, --odds1, and --odds2")
            print("Or run without arguments for interactive mode")
            sys.exit(1)
    
    # Make prediction
    result = make_prediction(match_data)
    display_result(result)


if __name__ == "__main__":
    main()
