#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without installing the package
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from learning_branch_cut.experiments.runner import run_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a learning-augmented branch-and-cut experiment."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the YAML configuration file describing the experiment.",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "validation", "test"],
        help="Which dataset split to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_from_config(args.config, args.split)


if __name__ == "__main__":
    main()
