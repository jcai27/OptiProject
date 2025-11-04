#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from learning_branch_cut import ExperimentRunner, TrainingWorkflow, load_experiment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a minimal end-to-end workflow: solve instances then train models."
    )
    parser.add_argument("--config", required=True, type=Path, help="Experiment configuration file.")
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "validation", "test"],
        help="Dataset split to solve before training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)

    runner = ExperimentRunner(config)
    runner.run(args.split)

    workflow = TrainingWorkflow(config)
    workflow.run()


if __name__ == "__main__":
    main()
