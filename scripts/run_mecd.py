#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mecd.config import DEFAULT_CONFIG, MECDConfig
from mecd.mecd_core import compute_mecd_from_returns
from mecd.utils import load_returns_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute MECD signal from a CSV of daily returns."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV file with returns.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output CSV file with MECD Z-scores.",
    )
    parser.add_argument(
        "--raw-output",
        default=None,
        help="Optional path to save raw MECD values as CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    raw_output_path = Path(args.raw_output) if args.raw_output else None

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    # Load returns
    returns = load_returns_csv(str(input_path))

    # Use default config; you can customize here if desired
    config: MECDConfig = DEFAULT_CONFIG

    # Compute MECD
    raw_mecd, mecd_z = compute_mecd_from_returns(returns, config)

    # Save outputs
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mecd_z.to_csv(output_path)

    if raw_output_path is not None:
        raw_output_path.parent.mkdir(parents=True, exist_ok=True)
        raw_mecd.to_csv(raw_output_path)

    print(f"Saved MECD Z-scores to {output_path}")
    if raw_output_path is not None:
        print(f"Saved raw MECD to {raw_output_path}")


if __name__ == "__main__":
    main()
