# Learning-Augmented Branch-and-Cut Framework

This repository contains a Python scaffold for experimenting with learning-augmented branch-and-cut strategies on families of mixed-integer linear programs. The motivating domain is daily microgrid scheduling with batteries, photovoltaics, and flexible loads, but the framework is designed so that other repeated-solve problem classes can be plugged in with minimal changes.

## Early-stage scope

The current iteration provides:

- Project layout and configuration schema for describing datasets, solver backends, and experiment bundles.
- Abstractions for integrating with MILP solvers (SCIP / PySCIPOpt by default) with safe fallbacks when optional dependencies are missing.
- Logging interfaces that capture per-node and per-cut features plus solver statistics (bounds, LP iterations, queue sizes) required for training lightweight decision models.
- Placeholders for data curation, model training, and evaluation workflows.

The immediate next steps are to flesh out the solver instrumentation, connect to concrete datasets, and iterate on model training and guardrail logic.

## Repository structure

```
src/learning_branch_cut/
  configs/             # Pydantic-based configuration objects and YAML parsing helpers
  data/                # Instance generation, loading, and feature engineering
  experiments/         # Experiment runner orchestration and metrics aggregation
  instrumentation/     # Hooks for collecting solver telemetry
  ml/                  # Node / cut models and calibration logic
  solver/              # Solver abstraction layer and branch-and-cut hooks
scripts/               # CLI entry points (dataset prep, experiment runner, plotting)
docs/                  # Design notes, runbooks, and references
```

## Getting started

1. Create a Python environment (3.10+) and install dependencies with `pip install -e .` (add `.[solver]` if you want PySCIPOpt bindings).
2. Install an MILP solver with Python bindings (recommended: [SCIP 9 + PySCIPOpt](https://www.scipopt.org/)).
3. Generate synthetic microgrid instances with `python scripts/generate_microgrid_instances.py --output-dir data/microgrid_lp --instances 12 --time-steps 24 --generators 4 --evs 4` (or point to your own MILPs).
4. Copy an example configuration from `docs/examples/*.yml` and adjust dataset paths, solver options, and instrumentation toggles. When running with PySCIPOpt, consider disabling presolve/heuristics (see usage guide) so the solver explores a deeper branch-and-cut tree and produces diverse telemetry.
5. Run `python scripts/run_experiment.py --config path/to/config.yml` to execute the baseline workflow (use PowerShell/CMD if PySCIPOpt is installed; otherwise the dummy backend will be used).
6. Run `python scripts/train_models.py --config path/to/config.yml` to train node and cut models from telemetry (requires numpy, pandas, scikit-learn).
7. Use `python scripts/bootstrap_workflow.py --config path/to/config.yml` for an end-to-end solve+train loop.

## Status

This is a minimal working scaffold and not yet feature complete. Track progress via issues or the project board as instrumentation, learning, and evaluation pieces are implemented.
