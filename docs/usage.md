# Usage Guide

This document captures how to run the minimal framework and outlines the next implementation milestones.

## 1. Prepare a configuration

Create a YAML file following the schema below (see `docs/examples/debug.yml` for a runnable placeholder):

```yaml
experiment_name: debug
dataset:
  root_path: ./data/microgrid_lp
  file_extension: .lp
  train_instances: ["day_001", "day_002", "day_003"]
  validation_instances: ["day_004"]
  test_instances: ["day_005"]
solver:
  backend: dummy          # switch to pyscipopt once installed
  time_limit_seconds: 120
  models:
  node_model: none
  cut_model: none
guardrails:
  enable_guardrails: true
instrumentation:
  enable_node_logging: true
  enable_cut_logging: true
output_directory: ./outputs
```

## 2. Run an experiment

```bash
python3 scripts/run_experiment.py --config docs/examples/debug.yml --split train
```

Outputs include:

- `outputs/telemetry/node_logs.jsonl` and `outputs/telemetry/cut_logs.jsonl` with instrumentation records.
- `outputs/results/<split>.json` summarizing per-instance solver status.
- Node entries now capture depth, bounds, fractional variable counts, solving time, LP iteration counts, queue sizes, and cumulative cuts (branch-candidate queries are skipped automatically if the LP solution is no longer available); cut entries record the invoking context, current node id, LP statistics, and running cut totals.

## 3. Train lightweight models

```bash
python3 scripts/train_models.py --config docs/examples/debug.yml
```

This reads telemetry logs, trains the node and cut models when labels/targets are available, and saves bundles to `outputs/models/`.
Install the learning dependencies first with `pip install numpy pandas scikit-learn` (or `pip install -e .`).

## 4. End-to-end bootstrap

```bash
python3 scripts/bootstrap_workflow.py --config docs/examples/debug.yml --split train
```

Runs the chosen split and immediately trains models.

## 5. Generate synthetic microgrid instances

```bash
python3 scripts/generate_microgrid_instances.py --output-dir data/microgrid_lp --instances 6 --time-steps 8
```

This populates a repeated-solve dataset with MILP instances following the microgrid scheduling template. Add `--time-steps 24 --generators 4 --evs 4` (or similar) to make the problems tougher so the branch-and-cut tree grows and telemetry contains rich labels.

For PySCIPOpt runs that should explore deeper search trees (and thus yield varied node outcomes), set solver parameters in your config, for example:

```yaml
solver:
  backend: pyscipopt
  time_limit_seconds: 180
  settings:
    limits/nodes: 8000
    limits/time: 180
    presolving/maxrounds: 0
```
If your SCIP build supports additional parameters (heuristic emphasis, random branching, etc.), add them here—unknown names are ignored with a warning.

## 6. Extending the scaffold

- **Instrumentation** — with PySCIPOpt installed, the framework now attaches a node-event handler and a no-op separator that record telemetry without altering the search. Toggle them from the YAML `instrumentation` block or customize behaviour via `learning_branch_cut.instrumentation.scip_callbacks.CallbackOptions`.
- **Solver integration** — extend `learning_branch_cut.instrumentation.scip_callbacks` or replace the included callbacks to stream richer node and cut features into the `TelemetryLogger`.
- **Dataset tooling** — populate `InstanceDataset` with real microgrid traces; add samplers for generating synthetic scenarios.
- **Learning tasks** — connect `ml.models` to logged data, train models, and feed predictions back through solver guardrails.
- **Evaluation harness** — expand `ExperimentRunner` to compute primal/dual integrals, Dolan–Moré profiles, and statistical summaries.

Track these action items in issues or the project board to build toward the full research workflow.
