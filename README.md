# Learning-Augmented Branch-and-Cut Framework

This repository contains a Python scaffold for experimenting with learning-augmented branch-and-cut strategies on families of mixed-integer linear programs. The motivating domain is daily microgrid scheduling with batteries, photovoltaics, and flexible loads, but the framework is designed so that other repeated-solve problem classes can be plugged in with minimal changes.

## High-level overview

This framework glues together four lightweight layers so you can prototype learning-augmented branch-and-cut ideas without rebuilding the plumbing each time:

1. **Configuration & datasets** – `ExperimentConfig` objects (parsed from YAML) describe datasets, solver backends, guardrails, and instrumentation toggles. The `InstanceDataset` helper maps logical IDs (e.g., `day_001`) onto `.lp` files plus optional metadata sidecars.
2. **Solver & instrumentation** – PySCIPOpt (when available) is wrapped with instrumentation callbacks that stream per-node and per-cut telemetry, optionally enriched with strong-branching probes. If PySCIPOpt is missing, the dummy backend keeps the rest of the workflow testable.
3. **Telemetry & ML models** – Rich JSONL logs capture the statistics needed to train small logistic/regression models for node selection and cut ordering. Trained bundles are cached under `outputs/models/`.
4. **Policy injection** – On subsequent runs, the solver automatically reloads the bundles (when present) and mixes the learned policies with the default best-bound strategy using guardrail weights, giving you a turnkey loop for collect → learn → deploy experiments.

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

## Low-level process walkthrough

1. **Define your experiment**
   - Start from `docs/examples/debug.yml` or `configs/sample_experiment.yml` and set the dataset root, solver backend, instrumentation toggles, and guardrail weights.
   - Strong-branching labels are optional but recommended for training the node model (`instrumentation.strong_branching.enable_labels: true`).
2. **Prepare data**
   - Point the dataset block at existing MILPs or synthesize new ones with `python scripts/generate_microgrid_instances.py --output-dir data/microgrid_lp --instances 12 --time-steps 24 --generators 4 --evs 4`.
   - (Optional) Drop YAML metadata beside each `.lp` using the same basename to log contextual fields.
3. **Collect telemetry**
   - Run `python scripts/run_experiment.py --config path/to/config.yml --split train` (or use `scripts/bootstrap_workflow.py`/`scripts/run_pipeline.py` if you prefer a scripted loop).
   - PySCIPOpt runs attach node/cut callbacks automatically; the dummy backend still emits placeholder entries so downstream tooling keeps working without SCIP.
4. **Train lightweight models**
   - Execute `python scripts/train_models.py --config path/to/config.yml` to convert telemetry JSONL files into NumPy datasets, fit the logistic (node) / elastic-net (cut) models, and persist bundles to `outputs/models/`.
   - Review `outputs/models/training_status.json` to confirm which models trained (`insufficient_labels` usually means the node log never saw both positive/negative labels).
5. **Deploy learned policies**
   - Flip `solver.use_learned_policies: true` in your config (or let `scripts/run_pipeline.py` do it for the evaluation split) so PySCIPOpt loads the bundles and blends ML scores with its native selectors.
   - Solve a validation/test split, inspect `outputs/results/<split>.json`, and iterate on feature logging, guardrails, or model hyperparameters as needed.

## Usage instructions

### Environment setup

```bash
python -m venv .venv && source .venv/bin/activate   # or use your preferred env manager
pip install -U pip
pip install -e .                                     # core deps (numpy, pandas, PyYAML, rich, scikit-learn)
pip install -e .[solver]                             # optional PySCIPOpt bindings (non-arm64 platforms)
```

Install SCIP + PySCIPOpt separately following the [SCIP docs](https://www.scipopt.org/) if you want full solver functionality; otherwise the dummy backend will keep scripts runnable.

### Running experiments

```bash
python scripts/run_experiment.py --config configs/sample_experiment.yml --split train
```

Key outputs:

- `outputs/results/train.json` – per-instance objective, status, solve time, node/cut counts.
- `outputs/telemetry/node_logs.jsonl` & `outputs/telemetry/cut_logs.jsonl` – rich telemetry for training/debugging.

Tweak `solver.settings` inside the YAML to control SCIP parameters (e.g., `limits/nodes`, `separating/maxrounds`, presolve toggles). Unknown parameters are ignored with a warning, so you can safely iterate.

### Training models only

```bash
python scripts/train_models.py --config configs/sample_experiment.yml
```

This step requires the ML dependencies (NumPy, pandas, scikit-learn). Successful runs produce:

- `outputs/models/node_model.pkl` and/or `cut_model.pkl`
- `outputs/models/training_status.json` summarizing which models trained and why others skipped.

### End-to-end pipeline

To automate collect → train → evaluate:

```bash
python scripts/run_pipeline.py \
  --config configs/sample_experiment.yml \
  --train-split train \
  --eval-split validation
```

The pipeline:

1. Solves the `train` split without learned policies (so telemetry logs reflect baseline behavior).
2. Trains node/cut models and saves bundles.
3. Reruns the `validation` split with `solver.use_learned_policies` enabled (skip with `--skip-eval`).

Use `scripts/bootstrap_workflow.py` if you only need a single collect+train pass (no evaluation solve).

## Status

This is a minimal working scaffold and not yet feature complete. Track progress via issues or the project board as instrumentation, learning, and evaluation pieces are implemented.
