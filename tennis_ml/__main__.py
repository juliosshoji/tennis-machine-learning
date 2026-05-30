"""CLI entrypoint for training and saving models."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from .preprocessing import load_and_preprocess_tennis_data, load_csv_rows, preprocess_rows
from .training import (
    save_model,
    save_model_json,
    train_mean_regressor,
    train_decision_tree,
    train_neural_network,
    train_random_forest,
    train_kmeans_clustering,
    evaluate_model,
    get_feature_importance,
)


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser for the training entrypoint."""
    parser = argparse.ArgumentParser(description="Train and save tennis match prediction models.")
    parser.add_argument("--data", required=True, help="Path to CSV tennis dataset.")
    parser.add_argument("--model-type", default="all", 
                       choices=["baseline", "tree", "neural", "forest", "all"],
                       help="Type of model to train.")
    parser.add_argument("--output-dir", default="models/", 
                       help="Output directory for model files.")
    parser.add_argument("--feature-columns", nargs="*",
                       help="Specific feature column names (for basic training mode).")
    parser.add_argument("--target-column", default="Player_1_Wins",
                       help="Target column name.")
    
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run CLI entrypoint."""
    args = build_parser().parse_args(argv)
    
    # Check if using simple mode (feature columns specified)
    if args.feature_columns:
        print("Running in basic preprocessing mode...")
        rows = load_csv_rows(args.data)
        features, targets = preprocess_rows(rows, args.feature_columns, args.target_column)
        model = train_mean_regressor(features, targets)
        save_model_json(model, f"{args.output_dir}baseline_model.json")
        print("✓ Baseline model saved")
        return 0
    
    # Full tennis data pipeline
    print(f"Loading and preprocessing tennis data from {args.data}...")
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_tennis_data(args.data)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Number of features: {len(feature_names)}")
    
    results = {"models": {}, "feature_importance": {}}
    
    # Train Decision Tree
    if args.model_type in ("tree", "all"):
        print("\n[1/4] Training Decision Tree...")
        dt_model = train_decision_tree(X_train, y_train)
        dt_metrics = evaluate_model(dt_model, X_test, y_test, "Decision Tree")
        save_model(dt_model, f"{args.output_dir}decision_tree_model.pkl")
        results["models"]["decision_tree"] = {
            "metrics": dt_metrics,
            "model_path": f"{args.output_dir}decision_tree_model.pkl",
        }
        dt_importance = get_feature_importance(dt_model, feature_names, top_n=20)
        results["feature_importance"]["decision_tree"] = dt_importance
        print(f"✓ Decision Tree - Accuracy: {dt_metrics['accuracy']:.4f}, F1: {dt_metrics['f1']:.4f}")
    
    # Train Neural Network
    if args.model_type in ("neural", "all"):
        print("[2/4] Training Neural Network (MLP)...")
        nn_model = train_neural_network(X_train, y_train)
        nn_metrics = evaluate_model(nn_model, X_test, y_test, "Neural Network")
        save_model(nn_model, f"{args.output_dir}neural_network_model.pkl")
        results["models"]["neural_network"] = {
            "metrics": nn_metrics,
            "model_path": f"{args.output_dir}neural_network_model.pkl",
        }
        nn_importance = get_feature_importance(nn_model, feature_names, top_n=20)
        results["feature_importance"]["neural_network"] = nn_importance
        print(f"✓ Neural Network - Accuracy: {nn_metrics['accuracy']:.4f}, F1: {nn_metrics['f1']:.4f}")
    
    # Train Random Forest
    if args.model_type in ("forest", "all"):
        print("[3/4] Training Random Forest...")
        rf_model = train_random_forest(X_train, y_train)
        rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
        save_model(rf_model, f"{args.output_dir}random_forest_model.pkl")
        results["models"]["random_forest"] = {
            "metrics": rf_metrics,
            "model_path": f"{args.output_dir}random_forest_model.pkl",
        }
        rf_importance = get_feature_importance(rf_model, feature_names, top_n=20)
        results["feature_importance"]["random_forest"] = rf_importance
        print(f"✓ Random Forest - Accuracy: {rf_metrics['accuracy']:.4f}, F1: {rf_metrics['f1']:.4f}")
    
    # K-means Clustering (exploratory)
    if args.model_type in ("all",):
        print("[4/4] Training K-means Clustering for exploratory analysis...")
        kmeans_model = train_kmeans_clustering(X_train, n_clusters=3)
        save_model(kmeans_model, f"{args.output_dir}kmeans_clustering_model.pkl")
        results["models"]["kmeans"] = {
            "n_clusters": 3,
            "model_path": f"{args.output_dir}kmeans_clustering_model.pkl",
        }
        print("✓ K-means Clustering complete (3 clusters identified)")
    
    # Save detailed results
    save_model_json(results, f"{args.output_dir}training_results.json")
    print(f"\n✓ All models trained and saved to {args.output_dir}")
    print(f"✓ Results summary saved to {args.output_dir}training_results.json")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
