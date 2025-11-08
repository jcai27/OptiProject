#!/usr/bin/env python

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from learning_branch_cut import ExperimentRunner, TrainingWorkflow, load_experiment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect telemetry with strong branching, train ML models, "
            "and evaluate a solver run that consumes the learned policies."
        )
    )
    parser.add_argument("--config", required=True, type=Path, help="Experiment configuration file.")
    parser.add_argument(
        "--train-split",
        default="train",
        choices=["train", "validation", "test"],
        help="Split used to gather telemetry before training.",
    )
    parser.add_argument(
        "--eval-split",
        default="validation",
        choices=["train", "validation", "test"],
        help="Split solved after training with the learned policies.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Run data collection + training only (skip the post-training evaluation solve).",
    )
    return parser.parse_args()


def with_policy_toggle(config, use_policies: bool):
    clone = copy.deepcopy(config)
    clone.solver.use_learned_policies = use_policies
    return clone


def maybe_warn_strong_branching(config) -> None:
    sb = config.instrumentation.strong_branching
    if not getattr(sb, "enable_labels", False):
        print(
            "[pipeline] Warning: strong branching labels are disabled; "
            "node model training will likely skip due to missing labels.",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    maybe_warn_strong_branching(config)

    print(
        f"[pipeline] Solving {args.train_split} split without learned policies to collect telemetry...",
        flush=True,
    )
    baseline_cfg = with_policy_toggle(config, use_policies=False)
    ExperimentRunner(baseline_cfg).run(args.train_split)

    print("[pipeline] Training node/cut models from collected telemetry...", flush=True)
    workflow = TrainingWorkflow(config)
    training_status = workflow.run()
    print(f"[pipeline] Training status: {training_status}", flush=True)

    if args.skip_eval:
        return

    print(
        f"[pipeline] Solving {args.eval_split} split with learned policies enabled...",
        flush=True,
    )
    ml_cfg = with_policy_toggle(config, use_policies=True)
    ExperimentRunner(ml_cfg).run(args.eval_split)


if __name__ == "__main__":
    main()
