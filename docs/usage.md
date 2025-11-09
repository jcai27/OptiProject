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
  use_learned_policies: false
  models:
  node_model: none
  cut_model: none
guardrails:
  enable_guardrails: true
instrumentation:
  enable_node_logging: true
  enable_cut_logging: true
  strong_branching:
    enable_labels: true
    candidate_limit: 8
    iteration_limit: 50
    label_top_fraction: 0.25
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
Whenever `outputs/models/node_model.pkl` or `outputs/models/cut_model.pkl` exist, PySCIPOpt runs automatically reload them: the node selector reorders open nodes by the learned probabilities (blended with best bound using `guardrails.node_weight`), and the cut selector ranks candidate cuts using the regression predictions. If the bundles are missing, SCIP falls back to the default best-bound and internal cut selection strategies.

## 4. End-to-end pipeline

```bash
python3 scripts/run_pipeline.py --config docs/examples/debug.yml --train-split train --eval-split validation
```

This command solves the training split without ML policies (to log strong-branching labels), trains the node/cut models, then immediately solves the evaluation split with `solver.use_learned_policies` enabled so the learned selectors steer branching and cut ordering. For a simpler collect+train loop without the evaluation pass, you can still run `python3 scripts/bootstrap_workflow.py --config docs/examples/debug.yml --split train`.

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

- **Instrumentation** — with PySCIPOpt installed, the framework attaches node and row event handlers that log telemetry without altering the search. Toggle them from the YAML `instrumentation` block or customize behaviour via `learning_branch_cut.instrumentation.scip_callbacks.CallbackOptions`. The `strong_branching` sub-block can be enabled to compute approximate strong-branching scores/labels directly in the telemetry logs (the option auto-disables with a warning if the active PySCIPOpt build lacks the `getVarStrongbranch*` probes), which downstream training uses instead of heuristic node outcomes. Missing selector interfaces (`Nodesel`/`Cutsel`) are also detected automatically so runs keep progressing with SCIP’s native heuristics whenever PySCIPOpt was built without the hooks.
- **Solver integration** — extend `learning_branch_cut.instrumentation.scip_callbacks` or replace the included callbacks to stream richer node and cut features into the `TelemetryLogger`.
- **Dataset tooling** — populate `InstanceDataset` with real microgrid traces; add samplers for generating synthetic scenarios.
- **Learning tasks** — connect `ml.models` to logged data, train models, and feed predictions back through solver guardrails.
- **Evaluation harness** — expand `ExperimentRunner` to compute primal/dual integrals, Dolan–Moré profiles, and statistical summaries.

Track these action items in issues or the project board to build toward the full research workflow.
