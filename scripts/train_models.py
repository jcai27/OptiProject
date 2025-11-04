#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from learning_branch_cut import TRAINING_IMPORT_ERROR, TrainingWorkflow, load_experiment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lightweight models from telemetry logs.")
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the experiment configuration that produced the telemetry.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    if TrainingWorkflow is None:
        raise RuntimeError(
            "Training dependencies are missing. Install the ML extras (numpy, pandas, scikit-learn) "
            "and retry."
        ) from TRAINING_IMPORT_ERROR
    workflow = TrainingWorkflow(config)
    status = workflow.run()
    for name, state in status.items():
        print(f"{name}: {state}")


if __name__ == "__main__":
    main()
