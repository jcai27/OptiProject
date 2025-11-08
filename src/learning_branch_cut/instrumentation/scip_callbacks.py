from __future__ import annotations

import time
import math
from dataclasses import dataclass
from typing import Optional

from .logger import CutLogRecord, NodeLogRecord, TelemetryLogger
from .strong_branching import StrongBranchingLabeller, StrongBranchingOptions

HAS_SCIP = False


def _to_int(value) -> int:
    try:
        return int(value)
    except TypeError:
        try:
            return int(value.value)  # some enums store numeric value in .value
        except Exception:
            return int(str(value))


def _event_type_name(event_type) -> str:
    try:
        mask = _to_int(event_type)
    except Exception:  # pragma: no cover
        return str(event_type)

    mapping = []
    for key in (
        "NODEFOCUSED",
        "NODEFEASIBLE",
        "NODEINFEASIBLE",
        "ROWADDEDSEPA",
        "ROWADDEDLP",
    ):
        const = globals().get(f"SCIP_EVENTTYPE_{key}")
        if const is None:
            continue
        try:
            const_mask = _to_int(const)
        except Exception:
            continue
        if mask & const_mask:
            mapping.append(key)
    return "|".join(mapping) if mapping else str(mask)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return default
        return numeric
    except Exception:
        return default


def _enum_name(value) -> str:
    if value is None:
        return "None"
    name = getattr(value, "name", None)
    if name:
        return name
    return str(value)


def _scip_result_value(name: str, fallback: int) -> int:
    if "SCIP_RESULT" not in globals():  # pragma: no cover - telemetry disabled
        return fallback
    result_enum = globals()["SCIP_RESULT"]
    # Attribute-style access (e.g., enum members)
    value = getattr(result_enum, name, None)
    if value is not None:
        return int(value)
    # Mapping-style access
    try:
        return int(result_enum[name])
    except Exception:
        return fallback


def _scip_result_ok() -> int:
    return _scip_result_value("OKAY", 0)


def _scip_result_didnotrun() -> int:
    return _scip_result_value("DIDNOTRUN", 0)


def _load_scip_classes():
    import pyscipopt

    Eventhdlr = getattr(pyscipopt, "Eventhdlr", None)
    Sepa = getattr(pyscipopt, "Sepa", None)

    # Try modern enum-style access first, fall back to legacy constants
    event_enum = getattr(pyscipopt, "SCIP_EVENTTYPE", None)

    def _event_value(name: str):
        if event_enum is not None:
            if hasattr(event_enum, name):
                return getattr(event_enum, name)
            try:
                return event_enum[name]
            except Exception:  # pragma: no cover - safety net
                pass
        legacy_name = f"SCIP_EVENTTYPE_{name}"
        return getattr(pyscipopt, legacy_name, None)

    SCIP_EVENTTYPE_NODEFOCUSED = _event_value("NODEFOCUSED")
    SCIP_EVENTTYPE_NODEFEASIBLE = _event_value("NODEFEASIBLE")
    SCIP_EVENTTYPE_NODEINFEASIBLE = _event_value("NODEINFEASIBLE")
    SCIP_EVENTTYPE_ROWADDEDSEPA = _event_value("ROWADDEDSEPA")
    SCIP_EVENTTYPE_ROWADDEDLP = _event_value("ROWADDEDLP")

    SCIP_RESULT = getattr(pyscipopt, "SCIP_RESULT", None)

    if None in (
        Eventhdlr,
        Sepa,
        SCIP_EVENTTYPE_NODEFOCUSED,
        SCIP_EVENTTYPE_NODEFEASIBLE,
        SCIP_EVENTTYPE_NODEINFEASIBLE,
        SCIP_EVENTTYPE_ROWADDEDSEPA,
        SCIP_EVENTTYPE_ROWADDEDLP,
        SCIP_RESULT,
    ):
        return None

    return {
        "Eventhdlr": Eventhdlr,
        "Sepa": Sepa,
        "SCIP_EVENTTYPE_NODEFEASIBLE": SCIP_EVENTTYPE_NODEFEASIBLE,
        "SCIP_EVENTTYPE_NODEFOCUSED": SCIP_EVENTTYPE_NODEFOCUSED,
        "SCIP_EVENTTYPE_NODEINFEASIBLE": SCIP_EVENTTYPE_NODEINFEASIBLE,
        "SCIP_EVENTTYPE_ROWADDEDSEPA": SCIP_EVENTTYPE_ROWADDEDSEPA,
        "SCIP_EVENTTYPE_ROWADDEDLP": SCIP_EVENTTYPE_ROWADDEDLP,
        "SCIP_RESULT": SCIP_RESULT,
    }


Eventhdlr = object
Sepa = object

try:  # Try at import time first; fall back to lazy load below
    _classes = _load_scip_classes()
except ImportError:  # pragma: no cover - optional dependency
    _classes = None
else:  # pragma: no cover - only executed when PySCIPOpt available at import
    if _classes is not None:
        globals().update(_classes)
        HAS_SCIP = True


@dataclass
class CallbackOptions:
    """Toggles for instrumentation callbacks."""

    enable_node_logging: bool = True
    enable_cut_logging: bool = True
    strong_branching: Optional[StrongBranchingOptions] = None


def register_logging_callbacks(model, telemetry: TelemetryLogger, options: CallbackOptions) -> None:
    """Attach logging callbacks to a PySCIPOpt model."""

    global HAS_SCIP, Eventhdlr, Sepa  # pylint:disable=global-statement

    if not HAS_SCIP:
        try:
            classes = _load_scip_classes()
        except ImportError as exc:  # pragma: no cover - guard against missing dependency
            raise RuntimeError("PySCIPOpt is not available, cannot register callbacks.") from exc
        if classes is None:
            telemetry.console.log(
                "[yellow]PySCIPOpt build missing event/separator interfaces; telemetry callbacks disabled.[/yellow]"
            )
            return
        globals().update(classes)
        HAS_SCIP = True

    if options.enable_node_logging:
        nodes = NodeEventHandler(telemetry, options)
        model.includeEventhdlr(nodes, "log_nodes", "Log node focus and status events.")

    if options.enable_cut_logging:
        cuts = CutEventHandler(telemetry)
        model.includeEventhdlr(cuts, "log_cuts", "Log cut additions.")


class NodeEventHandler(Eventhdlr):  # type: ignore[misc]
    """Records node focus events emitted by SCIP."""

    def __init__(self, telemetry: TelemetryLogger, options: CallbackOptions) -> None:
        super().__init__()
        self.telemetry = telemetry
        sb_options = getattr(options, "strong_branching", None)
        if sb_options is not None and not sb_options.enabled:
            sb_options = None
        self._sb_options = sb_options
        self._sb_labeller: Optional[StrongBranchingLabeller] = None
        self._sb_warned = False

    def eventinit(self) -> dict:
        self.model.catchEvent(SCIP_EVENTTYPE_NODEFOCUSED, self)
        self.model.catchEvent(SCIP_EVENTTYPE_NODEFEASIBLE, self)
        self.model.catchEvent(SCIP_EVENTTYPE_NODEINFEASIBLE, self)
        return {"result": _scip_result_ok()}

    def eventexit(self) -> dict:
        self.model.dropEvent(SCIP_EVENTTYPE_NODEFOCUSED, self)
        self.model.dropEvent(SCIP_EVENTTYPE_NODEFEASIBLE, self)
        self.model.dropEvent(SCIP_EVENTTYPE_NODEINFEASIBLE, self)
        return {"result": _scip_result_ok()}

    def eventexec(self, event) -> dict:  # pragma: no cover - exercised via solver runtime
        try:
            node = event.getNode()
            sb_score = None
            sb_label = None
            if self._sb_options is not None and self._is_focus_event(event):
                if self._sb_labeller is None:
                    self._sb_labeller = StrongBranchingLabeller(self.model, self._sb_options)
                    if not self._sb_labeller.supported:
                        if not self._sb_warned:
                            self.telemetry.console.log(
                                "[yellow]Strong branching labels requested but unsupported by this PySCIPOpt build; skipping label generation.[/yellow]"
                            )
                            self._sb_warned = True
                        self._sb_options = None
                        self._sb_labeller = None
                if self._sb_labeller is not None and self._sb_labeller.supported:
                    sb_score, sb_label = self._sb_labeller.evaluate_focus_node()
            estimate = self._safe_call(getattr(node, "getEstimate", None), None)
            node_type = self._safe_call(lambda: _enum_name(node.getType()), None)
            solving_time = self._safe_model_call("getSolvingTime", None)
            lp_iterations = self._safe_model_call("getNLPIterations", None)
            nodes_processed = self._safe_model_call("getNNodes", None)
            nodes_remaining = self._safe_model_call("getNNodesLeft", None)
            cuts_applied = self._safe_model_call("getNCutsApplied", None)
            record = NodeLogRecord(
                node_id=node.getNumber(),
                depth=node.getDepth(),
                dual_bound=float(node.getLowerbound()),
                primal_bound=float(self.model.getPrimalbound()),
                gap=float(self.model.getGap()),
                fractional_variables=self._fractional_count(),
                timestamp=time.time(),
                outcome=_event_type_name(event.getType()),
                estimate=None if estimate is None else float(estimate),
                solving_time=solving_time,
                lp_iterations=None if lp_iterations is None else int(lp_iterations),
                nodes_processed=None if nodes_processed is None else int(nodes_processed),
                nodes_in_queue=None if nodes_remaining is None else int(nodes_remaining),
                cuts_applied=None if cuts_applied is None else int(cuts_applied),
                node_type=node_type,
                strong_branch_score=sb_score,
                strong_branch_label=sb_label if sb_label is None else int(sb_label),
            )
            self.telemetry.record_node(record)
        except Exception as exc:  # pylint:disable=broad-except
            self.telemetry.console.log(
                f"[red]Failed to log node event ({type(exc).__name__}): {exc}[/red]"
            )
        return {"result": _scip_result_ok()}

    def _fractional_count(self) -> int:
        try:
            candidates = self.model.getLPBranchCands()
        except Exception:  # pragma: no cover - guard for unsupported models
            return 0
        if isinstance(candidates, tuple):
            cands = candidates[0]
        else:
            cands = candidates
        return len(cands)

    @staticmethod
    def _safe_call(func, default=None):
        if func is None:
            return default
        try:
            return func()
        except Exception:
            return default

    def _safe_model_call(self, name: str, default=None):
        func = getattr(self.model, name, None)
        return self._safe_call(func, default)

    @staticmethod
    def _is_focus_event(event) -> bool:
        try:
            event_mask = _to_int(event.getType())
            focus_mask = _to_int(SCIP_EVENTTYPE_NODEFOCUSED)
        except Exception:
            return False
        return bool(event_mask & focus_mask)


class CutEventHandler(Eventhdlr):  # type: ignore[misc]
    """Records real separator cuts by listening to row events."""

    def __init__(self, telemetry: TelemetryLogger) -> None:
        super().__init__()
        self.telemetry = telemetry
        self._cut_counter = 0
        self._round_counter = 0

    def eventinit(self) -> dict:
        self.model.catchEvent(SCIP_EVENTTYPE_ROWADDEDSEPA, self)
        self.model.catchEvent(SCIP_EVENTTYPE_ROWADDEDLP, self)
        return {"result": _scip_result_ok()}

    def eventexit(self) -> dict:
        self.model.dropEvent(SCIP_EVENTTYPE_ROWADDEDSEPA, self)
        self.model.dropEvent(SCIP_EVENTTYPE_ROWADDEDLP, self)
        return {"result": _scip_result_ok()}

    def eventexec(self, event) -> dict:  # pragma: no cover - exercised at runtime
        self._cut_counter += 1
        event_type = _event_type_name(event.getType())
        if "ROWADDEDLP" in event_type:
            self._round_counter += 1
        row = self._safe_call(event.getRow, None)
        if row is None:
            return {"result": _scip_result_ok()}
        node = self._safe_call(event.getNode, None)
        node_id = None if node is None else self._safe_call(node.getNumber, None)
        stats = self._gather_stats()
        violation = self._row_violation(row, stats)
        efficacy = self._safe_cut_efficacy(row)
        dual_signal = self._safe_call(row.getDualsol, 0.0)
        sparsity = self._row_sparsity(row, stats)
        record = CutLogRecord(
            cut_id=self._cut_counter,
            separator=event_type,
            violation=violation,
            efficacy=efficacy,
            sparsity=sparsity,
            application_time=time.time(),
            dual_improvement=_safe_float(dual_signal, efficacy),
            node_id=None if node_id is None else int(node_id),
            solving_time=stats["solving_time"],
            lp_iterations=None if stats["lp_iterations"] is None else int(stats["lp_iterations"]),
            cuts_applied_total=None
            if stats["cuts_applied_total"] is None
            else int(stats["cuts_applied_total"]),
            lp_rows=None if stats["lp_rows"] is None else int(stats["lp_rows"]),
            lp_cols=None if stats["lp_cols"] is None else int(stats["lp_cols"]),
            round_index=self._round_counter,
        )
        self.telemetry.record_cut(record)
        return {"result": _scip_result_ok()}

    def _gather_stats(self):
        return {
            "solving_time": self._safe_model_call("getSolvingTime"),
            "lp_iterations": self._safe_model_call("getNLPIterations"),
            "cuts_applied_total": self._safe_model_call("getNCutsApplied"),
            "lp_rows": self._safe_model_call("getNLPRows"),
            "lp_cols": self._safe_model_call("getNLPCols"),
        }

    def _row_violation(self, row, stats) -> float:
        try:
            activity = float(self.model.getRowLPActivity(row))
        except Exception:
            activity = 0.0
        lhs = self._safe_call(row.getLhs, -float("inf"))
        rhs = self._safe_call(row.getRhs, float("inf"))
        lhs_violation = max(0.0, lhs - activity)
        rhs_violation = max(0.0, activity - rhs)
        return float(max(lhs_violation, rhs_violation))

    def _safe_cut_efficacy(self, row) -> float:
        try:
            return float(self.model.getCutEfficacy(row))
        except Exception:
            return 0.0

    def _row_sparsity(self, row, stats) -> float:
        try:
            nnz = float(row.getNNonz())
        except Exception:
            nnz = 0.0
        cols = stats.get("lp_cols") or 1
        return float(nnz) / float(max(1, cols))

    @staticmethod
    def _safe_call(func, default=None):
        if func is None:
            return default
        try:
            return func()
        except Exception:
            return default

    def _safe_model_call(self, name: str):
        func = getattr(self.model, name, None)
        return self._safe_call(func, None)
