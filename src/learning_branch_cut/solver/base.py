from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from ..configs.base import ExperimentConfig, InstrumentationConfig, SolverConfig
from ..instrumentation.logger import CutLogRecord, NodeLogRecord, TelemetryLogger
from ..instrumentation.scip_callbacks import CallbackOptions, register_logging_callbacks


@dataclass
class SolverResult:
    status: str
    objective_value: Optional[float]
    solve_time: float
    nodes_processed: int
    cuts_added: int


class SolverBackend:
    """Base API for solver integrations."""

    def __init__(self, config: SolverConfig, telemetry: TelemetryLogger) -> None:
        self.config = config
        self.telemetry = telemetry

    def solve(self, instance_path: str) -> SolverResult:
        raise NotImplementedError

    def close(self) -> None:
        """Cleanup resources if needed."""


class DummySolverBackend(SolverBackend):
    """Fallback backend used when no solver is installed."""

    def solve(self, instance_path: str) -> SolverResult:  # pragma: no cover - minimal stub
        start = time.perf_counter()
        time.sleep(0.01)  # Simulate minimal work
        timestamp = time.time()
        self.telemetry.record_node(
            NodeLogRecord(
                node_id=0,
                depth=0,
                dual_bound=0.0,
                primal_bound=0.0,
                gap=0.0,
                fractional_variables=0,
                timestamp=timestamp,
                outcome="optimal",
                estimate=0.0,
                solving_time=0.01,
                lp_iterations=0,
                nodes_processed=1,
                nodes_in_queue=0,
                cuts_applied=0,
                node_type="root",
            )
        )
        self.telemetry.record_cut(
            CutLogRecord(
                cut_id=0,
                separator="dummy",
                violation=0.0,
                efficacy=0.0,
                sparsity=0.0,
                application_time=timestamp,
                dual_improvement=0.0,
                node_id=0,
                solving_time=0.01,
                lp_iterations=0,
                cuts_applied_total=0,
                lp_rows=0,
                lp_cols=0,
                round_index=1,
            )
        )
        return SolverResult(
            status="dummy",
            objective_value=0.0,
            solve_time=time.perf_counter() - start,
            nodes_processed=1,
            cuts_added=1,
        )


class PySCIPOptBackend(SolverBackend):
    """Wrapper around PySCIPOpt with instrumentation hooks."""

    def __init__(
        self,
        config: SolverConfig,
        instrumentation: InstrumentationConfig,
        telemetry: TelemetryLogger,
    ) -> None:
        super().__init__(config, telemetry)
        try:
            import pyscipopt
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                f"PySCIPOpt import failed: {exc}. Install the optional `solver` dependency group."
            ) from exc
        self._pyscipopt = pyscipopt
        self._callback_options = CallbackOptions(
            enable_node_logging=instrumentation.enable_node_logging,
            enable_cut_logging=instrumentation.enable_cut_logging,
        )

    def _create_model(self):
        model = self._pyscipopt.Model()
        if self.config.time_limit_seconds:
            model.setRealParam("limits/time", float(self.config.time_limit_seconds))
        for name, value in self.config.settings.items():
            try:
                model.setParam(name, value)
            except KeyError:
                self.telemetry.console.log(
                    f"[yellow]Unknown SCIP parameter '{name}' ignored[/yellow]"
                )
            except Exception as exc:  # pragma: no cover - defensive
                self.telemetry.console.log(
                    f"[yellow]Failed to set SCIP parameter '{name}': {exc}[/yellow]"
                )
        register_logging_callbacks(model, self.telemetry, self._callback_options)
        return model

    def solve(self, instance_path: str) -> SolverResult:
        model = self._create_model()
        model.readProblem(str(instance_path))
        start = time.perf_counter()
        model.optimize()
        solve_time = time.perf_counter() - start
        status = model.getStatus()
        objective = model.getObjVal() if model.getNSols() > 0 else None
        nodes = model.getNNodes()
        cuts = model.getNCutsApplied()

        def safe_call(func, default=None):
            try:
                return func()
            except Exception:
                return default

        fractional_variables = 0
        get_lp_branch_cands = getattr(model, "getLPBranchCands", None)
        candidates = None
        if callable(get_lp_branch_cands):
            solstat = safe_call(getattr(model, "getLPSolstat", lambda: None))
            lp_ready = solstat in (None, 0)  # 0 corresponds to SCIP_LPSOLSTAT_OPTIMAL
            if lp_ready:
                candidates = safe_call(get_lp_branch_cands, None)
        if candidates is not None:
            if isinstance(candidates, tuple):
                fractional_variables = len(candidates[0])
            else:
                fractional_variables = len(candidates)

        get_solving_time = getattr(model, "getSolvingTime", None)
        solving_time_model = (
            safe_call(get_solving_time, solve_time) if get_solving_time else solve_time
        )

        get_lp_iterations = getattr(model, "getNLPIterations", None)
        lp_iterations = (
            safe_call(get_lp_iterations, None) if get_lp_iterations else None
        )

        get_nodes_left = getattr(model, "getNNodesLeft", None)
        nodes_remaining = safe_call(get_nodes_left, None) if get_nodes_left else None

        self.telemetry.record_node(
            NodeLogRecord(
                node_id=0,
                depth=0,
                dual_bound=model.getDualbound(),
                primal_bound=model.getPrimalbound(),
                gap=model.getGap(),
                fractional_variables=fractional_variables,
                timestamp=time.time(),
                outcome=status,
                estimate=None,
                solving_time=solving_time_model,
                lp_iterations=None if lp_iterations is None else int(lp_iterations),
                nodes_processed=int(nodes),
                nodes_in_queue=None if nodes_remaining is None else int(nodes_remaining),
                cuts_applied=int(cuts),
                node_type="root",
            )
        )
        self.telemetry.flush()
        free_method = getattr(model, "free", None)
        if callable(free_method):
            free_method()
        return SolverResult(
            status=status,
            objective_value=objective,
            solve_time=solve_time,
            nodes_processed=nodes,
            cuts_added=cuts,
        )

    def close(self) -> None:
        return None


def build_solver_backend(config: ExperimentConfig, telemetry: TelemetryLogger) -> SolverBackend:
    backend = config.solver.backend
    if backend == "pyscipopt":
        try:
            return PySCIPOptBackend(config.solver, config.instrumentation, telemetry)
        except RuntimeError as exc:
            telemetry.console.log(
                f"[yellow]PySCIPOpt unavailable ({exc}), falling back to dummy solver backend[/yellow]"
            )
    return DummySolverBackend(config.solver, telemetry)
