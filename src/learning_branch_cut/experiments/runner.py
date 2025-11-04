from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List

from rich.console import Console

from ..configs.base import ExperimentConfig, load_experiment_config
from ..data.dataset import InstanceDataset, ProblemInstance
from ..instrumentation.logger import TelemetryLogger
from ..solver.base import SolverResult, build_solver_backend


class ExperimentRunner:
    """Co-ordinates dataset iteration, solver execution, and logging."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.console = Console()
        self.dataset = InstanceDataset(config.dataset)
        self.telemetry = TelemetryLogger(config.output_directory / "telemetry")
        self.solver = build_solver_backend(config, self.telemetry)

    def run(self, split: str) -> None:
        instances = self._resolve_split(split)
        self.console.rule(f"[bold green]Running split {split} ({len(instances)} instances)")
        results: List[Dict[str, object]] = []
        for problem in instances:
            self.console.log(f"Solving instance {problem.identifier}")
            result = self._solve_instance(problem)
            results.append(self._result_to_dict(problem, result))
        self._write_results(split, results)
        self.telemetry.flush()
        self.solver.close()

    def _solve_instance(self, problem: ProblemInstance) -> SolverResult:
        start = time.perf_counter()
        result: SolverResult | None = None
        try:
            result = self.solver.solve(str(problem.data_path))
        except Exception as exc:  # pragma: no cover - debugging guard
            self.console.log(f"[red]Solver failed on {problem.identifier}: {exc}[/red]")
            raise
        finally:
            elapsed = time.perf_counter() - start
            solver_time = getattr(result, "solve_time", float("nan"))
            self.console.log(
                f"Completed {problem.identifier} in {elapsed:.2f}s (solver {solver_time:.2f}s)"
            )
        assert result is not None
        return result

    def _resolve_split(self, split: str) -> List[ProblemInstance]:
        if split == "train":
            iterator = self.dataset.iter_train()
        elif split == "validation":
            iterator = self.dataset.iter_validation()
        elif split == "test":
            iterator = self.dataset.iter_test()
        else:
            raise ValueError(f"Unknown split '{split}'")
        return list(iterator)

    def _result_to_dict(
        self, problem: ProblemInstance, result: SolverResult
    ) -> Dict[str, object]:
        return {
            "instance": problem.identifier,
            "status": result.status,
            "objective": result.objective_value,
            "solve_time": result.solve_time,
            "nodes_processed": result.nodes_processed,
            "cuts_added": result.cuts_added,
            "metadata": problem.metadata,
        }

    def _write_results(self, split: str, results: List[Dict[str, object]]) -> None:
        output_dir = self.config.output_directory / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{split}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)


def run_from_config(path: Path, split: str) -> None:
    config = load_experiment_config(path)
    runner = ExperimentRunner(config)
    runner.run(split)
