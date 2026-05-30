#!/usr/bin/env python3
"""
Player Search Tool - Find players in the Tennis Match Charting Project database

Usage:
    python player_search.py                    # Interactive search
    python player_search.py "Sinner"           # Search for player
    python player_search.py "Sinner" "Alcaraz" # Get predictions for matchup
"""

import pandas as pd
import sys
from pathlib import Path


def load_players():
    """Load all unique players from the Tennis Match Charting Project dataset."""
    base_path = Path("new-dataset/tennis_MatchChartingProject-master")
    
    if not base_path.exists():
        print("Error: new-dataset not found")
        sys.exit(1)
    
    # Load both men's and women's matches
    m_matches = pd.read_csv(base_path / "charting-m-matches.csv")
    w_matches = pd.read_csv(base_path / "charting-w-matches.csv")
    
    # Combine players
    players = sorted(list(
        set(m_matches["Player 1"].unique()) | 
        set(m_matches["Player 2"].unique()) |
        set(w_matches["Player 1"].unique()) | 
        set(w_matches["Player 2"].unique())
    ))
    
    return players


def search_players(query: str, players: list) -> list:
    """Search for players matching query."""
    query_lower = query.lower()
    results = [p for p in players if query_lower in p.lower()]
    return results


def get_player_stats(player_name: str, exact_match: bool = False) -> dict:
    """Get statistics for a specific player."""
    if not Path("atp_tennis.csv").exists():
        return {}
    
    df = pd.read_csv("atp_tennis.csv", low_memory=False)
    
    # Find matches where player was Player_1
    if exact_match:
        p1_matches = df[df["Player_1"] == player_name]
        p2_matches = df[df["Player_2"] == player_name]
    else:
        # Try to find close match
        p1_candidates = df[df["Player_1"].str.contains(player_name, case=False, na=False)]
        p2_candidates = df[df["Player_2"].str.contains(player_name, case=False, na=False)]
        
        if p1_candidates.empty and p2_candidates.empty:
            return {}
        
        # Get the first matching name
        if not p1_candidates.empty:
            actual_name = p1_candidates.iloc[0]["Player_1"]
        else:
            actual_name = p2_candidates.iloc[0]["Player_2"]
        
        p1_matches = df[df["Player_1"] == actual_name]
        p2_matches = df[df["Player_2"] == actual_name]
    
    # Calculate wins
    p1_wins = len(p1_matches[p1_matches["Winner"] == p1_matches["Player_1"]])
    p1_losses = len(p1_matches) - p1_wins
    
    p2_wins = len(p2_matches[p2_matches["Winner"] == p2_matches["Player_2"]])
    p2_losses = len(p2_matches) - p2_wins
    
    total_wins = p1_wins + p2_wins
    total_losses = p1_losses + p2_losses
    total_matches = total_wins + total_losses
    
    if total_matches == 0:
        return {}
    
    win_rate = total_wins / total_matches if total_matches > 0 else 0
    
    return {
        "name": actual_name if not exact_match else player_name,
        "total_matches": total_matches,
        "wins": total_wins,
        "losses": total_losses,
        "win_rate": win_rate,
        "avg_rank": None,  # Would require parsing rank data
    }


def display_player_info(player_name: str):
    """Display information about a player."""
    stats = get_player_stats(player_name, exact_match=False)
    
    if not stats:
        print(f"Player '{player_name}' not found")
        return
    
    print(f"\n{'='*70}")
    print(f"PLAYER INFORMATION")
    print(f"{'='*70}")
    print(f"\nName: {stats['name']}")
    print(f"Total Matches: {stats['total_matches']}")
    print(f"Wins: {stats['wins']}")
    print(f"Losses: {stats['losses']}")
    print(f"Win Rate: {stats['win_rate']:.1%}")
    print(f"{'='*70}\n")


def interactive_mode():
    """Interactive mode for player search and predictions."""
    
    print("\n" + "="*70)
    print("TENNIS MATCH CHARTING PROJECT - PLAYER SEARCH")
    print("="*70)
    
    players = load_players()
    print(f"\nLoaded {len(players)} players from Tennis Match Charting Project")
    
    while True:
        print("\nOptions:")
        print("  1. Search for a player")
        print("  2. Get player statistics")
        print("  3. Get match prediction")
        print("  4. List top players")
        print("  5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1":
            # Search for player
            query = input("\nEnter player name (partial OK): ").strip()
            results = search_players(query, players)
            
            if not results:
                print(f"No players found matching '{query}'")
            else:
                print(f"\nFound {len(results)} player(s):")
                for i, player in enumerate(results[:20], 1):
                    print(f"  {i:2d}. {player}")
                if len(results) > 20:
                    print(f"  ... and {len(results) - 20} more")
        
        elif choice == "2":
            # Get player stats
            query = input("\nEnter player name: ").strip()
            display_player_info(query)
        
        elif choice == "3":
            # Match prediction
            print("\nEnter match details:")
            p1 = input("Player 1 name: ").strip()
            p2 = input("Player 2 name: ").strip()
            
            try:
                odds1 = float(input("Player 1 odds: "))
                odds2 = float(input("Player 2 odds: "))
                
                # Try to import and use prediction
                try:
                    from predict_match import predict_ensemble
                    result = predict_ensemble(
                        player_1_name=p1,
                        player_2_name=p2,
                        player_1_odds=odds1,
                        player_2_odds=odds2,
                    )
                    print(f"\nPrediction: {result['prediction_label']} ({result['confidence']:.1%})")
                except ImportError:
                    print("Prediction module not available")
            except ValueError:
                print("Invalid odds format")
        
        elif choice == "4":
            # List some famous players
            print("\nSome famous players in the database:")
            famous = ["Federer R.", "Nadal R.", "Djokovic N.", "Murray A.", 
                     "Sampras P.", "Agassi A.", "Connors J.", "McEnroe J."]
            for player in famous:
                if player in players:
                    print(f"  ✓ {player}")
                else:
                    print(f"  ✗ {player} (not found)")
        
        elif choice == "5":
            print("\nGoodbye!")
            break
        
        else:
            print("Invalid option")


def command_line_mode(args: list):
    """Command-line mode."""
    players = load_players()
    
    if len(args) == 1:
        # Just one argument - search for player
        query = args[0]
        results = search_players(query, players)
        
        if not results:
            print(f"No players found matching '{query}'")
        else:
            print(f"\nFound {len(results)} player(s) matching '{query}':")
            print(f"{'='*70}")
            
            # Show in columns
            for i, player in enumerate(results):
                if (i + 1) % 3 == 0:
                    print(f"{player:30s}")
                else:
                    print(f"{player:30s}", end="  ")
            
            if len(results) % 3 != 0:
                print()
            
            print(f"{'='*70}")
    
    elif len(args) >= 2:
        # Two arguments - get match prediction
        p1 = args[0]
        p2 = args[1]
        
        # Try to find exact matches
        p1_matches = search_players(p1, players)
        p2_matches = search_players(p2, players)
        
        if not p1_matches:
            print(f"Player '{p1}' not found")
            return
        if not p2_matches:
            print(f"Player '{p2}' not found")
            return
        
        p1_exact = p1_matches[0]
        p2_exact = p2_matches[0]
        
        print(f"\nPlayers found:")
        print(f"  {p1_exact}")
        print(f"  {p2_exact}")
        
        # Get stats
        p1_stats = get_player_stats(p1_exact, exact_match=True)
        p2_stats = get_player_stats(p2_exact, exact_match=True)
        
        if p1_stats and p2_stats:
            print(f"\nStatistics:")
            print(f"  {p1_exact}: {p1_stats['wins']}-{p1_stats['losses']} ({p1_stats['win_rate']:.1%})")
            print(f"  {p2_exact}: {p2_stats['wins']}-{p2_stats['losses']} ({p2_stats['win_rate']:.1%})")
        
        print(f"\nTo get a match prediction, use:")
        print(f"  python quick_predict.py --player1 \"{p1_exact}\" --player2 \"{p2_exact}\" \\")
        print(f"    --odds1 <odds> --odds2 <odds>")


def main():
    """Main entry point."""
    
    if len(sys.argv) > 1 and sys.argv[1] not in ["--help", "-h"]:
        # Command-line mode
        args = sys.argv[1:]
        command_line_mode(args)
    elif "--help" in sys.argv or "-h" in sys.argv:
        print("\nPlayer Search Tool - Find players in Tennis Match Charting Project")
        print("\nUsage:")
        print("  python player_search.py                 # Interactive mode")
        print("  python player_search.py \"Sinner\"        # Search for player")
        print("  python player_search.py \"Sinner\" \"Alcaraz\"  # Search both players")
        print("\nExamples:")
        print("  python player_search.py")
        print("  python player_search.py \"Federer\"")
        print("  python player_search.py \"Nadal\"")
        print()
    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()
