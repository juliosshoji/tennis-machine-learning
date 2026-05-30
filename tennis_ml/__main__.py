"""CLI entrypoint for training and saving a baseline model."""

from __future__ import annotations

import argparse
from typing import Sequence

from .preprocessing import load_csv_rows, preprocess_rows
from .training import save_model, train_mean_regressor


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser for the training entrypoint."""
    parser = argparse.ArgumentParser(description="Train and save a mean baseline model.")
    parser.add_argument("--data", required=True, help="Path to CSV dataset.")
    parser.add_argument("--model", required=True, help="Output path for model JSON.")
    parser.add_argument(
        "--feature-columns",
        required=True,
        nargs="+",
        help="Feature column names to read from CSV.",
    )
    parser.add_argument("--target-column", required=True, help="Target column name in CSV.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run CLI entrypoint."""
    args = build_parser().parse_args(argv)
    rows = load_csv_rows(args.data)
    features, targets = preprocess_rows(rows, args.feature_columns, args.target_column)
    model = train_mean_regressor(features, targets)
    save_model(model, args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
