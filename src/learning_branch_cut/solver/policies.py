from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import numpy as np
except Exception:  # pragma: no cover - allow runtime without numpy
    np = None  # type: ignore[assignment]

from ..ml.features import CUT_FEATURES, NODE_FEATURES
from ..ml.models import CutModelBundle, NodeModelBundle

try:  # pragma: no cover - optional dependency
    import pyscipopt
except Exception:  # pragma: no cover - allows import when pyscipopt is missing
    Cutsel = Nodesel = object  # type: ignore[assignment]
    SCIP_RESULT = None  # type: ignore[assignment]
    Row = object  # type: ignore[assignment]
    HAS_NODESEL_CLASS = False
    HAS_CUTSEL_CLASS = False
else:  # pragma: no cover - executed when PySCIPOpt available
    Cutsel = getattr(pyscipopt, "Cutsel", None)
    Nodesel = getattr(pyscipopt, "Nodesel", None)
    Row = getattr(pyscipopt, "Row", None)
    if Cutsel is None or Nodesel is None or Row is None:
        scip_module = getattr(pyscipopt, "scip", None)
        if scip_module is not None:
            Cutsel = Cutsel or getattr(scip_module, "Cutsel", None)
            Nodesel = Nodesel or getattr(scip_module, "Nodesel", None)
            Row = Row or getattr(scip_module, "Row", None)
    Cutsel = Cutsel or object  # type: ignore[assignment]
    Nodesel = Nodesel or object  # type: ignore[assignment]
    Row = Row or object  # type: ignore[assignment]
    SCIP_RESULT = getattr(pyscipopt, "SCIP_RESULT", None)
    HAS_NODESEL_CLASS = Nodesel is not object
    HAS_CUTSEL_CLASS = Cutsel is not object


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        numeric = float(value)
        if math.isnan(numeric):
            return default
        if math.isinf(numeric):
            return default
        return numeric
    except Exception:
        return default


def _safe_int(value, default: int = 0) -> int:
    try:
        if value is None:
            return default
        numeric = int(value)
        return numeric
    except Exception:
        return default


class NodeSelectionPolicy:
    """Helper that evaluates open nodes with a trained logistic model."""

    def __init__(self, bundle: NodeModelBundle, node_weight: float) -> None:
        if np is None:
            raise RuntimeError("NumPy is required to evaluate the node model.")
        self.bundle = bundle
        self.node_weight = max(0.0, min(1.0, float(node_weight)))
        self.feature_names: Sequence[str] = tuple(bundle.feature_names)

    def score(self, scip_model, node) -> float:
        features = self._build_features(scip_model, node)
        if features is None:
            return -math.inf
        proba = float(self.bundle.model.predict_proba(features)[0, 1])
        bound_score = self._bound_score(scip_model, node)
        return (1.0 - self.node_weight) * proba + self.node_weight * bound_score

    def _build_features(self, scip_model, node) -> Optional[np.ndarray]:
        values: List[float] = []
        stats = _node_stats(scip_model)
        try:
            mapping: Dict[str, float] = {
                "depth": _safe_float(node.getDepth()),
                "dual_bound": _safe_float(node.getLowerbound()),
                "primal_bound": stats["primal_bound"],
                "gap": stats["gap"],
                "estimate": _safe_float(node.getEstimate(), stats["dual_bound"]),
                "solving_time": stats["solving_time"],
                "lp_iterations": stats["lp_iterations"],
                "nodes_processed": stats["nodes_processed"],
                "nodes_in_queue": stats["nodes_in_queue"],
                "cuts_applied": stats["cuts_applied"],
            }
        except Exception:
            return None
        for name in NODE_FEATURES:
            values.append(_safe_float(mapping.get(name), 0.0))
        data = np.asarray(values, dtype=float).reshape(1, -1)
        return self.bundle.scaler.transform(data)

    @staticmethod
    def _bound_score(scip_model, node) -> float:
        try:
            best_dual = _safe_float(scip_model.getDualbound(), node.getLowerbound())
        except Exception:
            best_dual = _safe_float(node.getLowerbound())
        gap = max(0.0, _safe_float(node.getLowerbound()) - best_dual)
        return 1.0 / (1.0 + gap)


class CutSelectionPolicy:
    """Helper that ranks cuts with a trained regression model."""

    def __init__(self, bundle: CutModelBundle) -> None:
        if np is None:
            raise RuntimeError("NumPy is required to evaluate the cut model.")
        self.bundle = bundle
        self.feature_names: Sequence[str] = tuple(bundle.feature_names)

    def score(self, scip_model, row: Row) -> float:
        data = self._build_features(scip_model, row)
        if data is None:
            return 0.0
        prediction = self.bundle.model.predict(data)[0]
        return float(prediction)

    def _build_features(self, scip_model, row: Row) -> Optional[np.ndarray]:
        stats = _cut_stats(scip_model)
        try:
            activity = _safe_float(scip_model.getRowLPActivity(row))
        except Exception:
            activity = 0.0
        lhs = _safe_float(row.getLhs(), -math.inf)
        rhs = _safe_float(row.getRhs(), math.inf)
        violation = max(max(0.0, activity - rhs), max(0.0, lhs - activity))
        try:
            efficacy = _safe_float(scip_model.getCutEfficacy(row))
        except Exception:
            efficacy = 0.0
        try:
            nnz = max(1, row.getNNonz())
        except Exception:
            nnz = 1
        total_cols = max(1, stats["lp_cols"])
        sparsity = float(nnz) / float(total_cols)
        mapping: Dict[str, float] = {
            "violation": violation,
            "efficacy": efficacy,
            "sparsity": sparsity,
            "solving_time": stats["solving_time"],
            "lp_iterations": stats["lp_iterations"],
            "cuts_applied_total": stats["cuts_applied_total"],
            "lp_rows": stats["lp_rows"],
            "lp_cols": stats["lp_cols"],
        }
        values = [_safe_float(mapping.get(name), 0.0) for name in CUT_FEATURES]
        data = np.asarray(values, dtype=float).reshape(1, -1)
        return self.bundle.scaler.transform(data)


class MLNodeSelector(Nodesel):  # type: ignore[misc]
    """Node selector that uses a trained probability model to rank open nodes."""

    def __init__(self, policy: NodeSelectionPolicy) -> None:
        super().__init__()
        self.policy = policy
        self._cache: Dict[int, float] = {}

    def nodeinit(self) -> None:
        self._cache.clear()

    def nodeselect(self):
        best_node = None
        best_score = -math.inf
        try:
            open_nodes = self.model.getOpenNodes()
        except Exception:
            open_nodes = ([], [], [])
        for bucket in open_nodes:
            for node in bucket:
                score = self._score(node)
                if score > best_score:
                    best_score = score
                    best_node = node
        if best_node is None:
            best_node = self.model.getBestNode() or self.model.getBestboundNode()
        return {"selnode": best_node}

    def nodecomp(self, node1, node2):
        score1 = self._score(node1)
        score2 = self._score(node2)
        if score1 > score2 + 1e-9:
            return -1
        if score1 < score2 - 1e-9:
            return 1
        return 0

    def _score(self, node) -> float:
        node_id = _safe_int(node.getNumber(), -1)
        cached = self._cache.get(node_id)
        if cached is not None:
            return cached
        score = self.policy.score(self.model, node)
        self._cache[node_id] = score
        return score


class MLCutSelector(Cutsel):  # type: ignore[misc]
    """Cut selector that sorts separator cuts by predicted gain."""

    def __init__(self, policy: CutSelectionPolicy) -> None:
        super().__init__()
        self.policy = policy

    def cutselselect(self, cuts, forcedcuts, root, maxnselectedcuts):
        scored: List[Tuple[float, Row]] = []
        for row in cuts:
            score = self.policy.score(self.model, row)
            scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        ordered = [row for _, row in scored]
        cuts[:] = ordered
        selection_cap = max(0, maxnselectedcuts)
        nselected = min(selection_cap, len(ordered) + len(forcedcuts))
        result = SCIP_RESULT.SUCCESS if SCIP_RESULT is not None else 0
        return {"cuts": cuts, "nselectedcuts": nselected, "result": result}


def _node_stats(model) -> Dict[str, float]:
    return {
        "primal_bound": _safe_float(_safe_call(model, "getPrimalbound")),
        "dual_bound": _safe_float(_safe_call(model, "getDualbound")),
        "gap": _safe_float(_safe_call(model, "getGap")),
        "solving_time": _safe_float(_safe_call(model, "getSolvingTime")),
        "lp_iterations": _safe_int(_safe_call(model, "getNLPIterations")),
        "nodes_processed": _safe_int(_safe_call(model, "getNNodes")),
        "nodes_in_queue": _safe_int(_safe_call(model, "getNNodesLeft")),
        "cuts_applied": _safe_int(_safe_call(model, "getNCutsApplied")),
    }


def _cut_stats(model) -> Dict[str, int]:
    return {
        "solving_time": _safe_float(_safe_call(model, "getSolvingTime")),
        "lp_iterations": _safe_int(_safe_call(model, "getNLPIterations")),
        "cuts_applied_total": _safe_int(_safe_call(model, "getNCutsApplied")),
        "lp_rows": _safe_int(_safe_call(model, "getNLPRows")),
        "lp_cols": _safe_int(_safe_call(model, "getNLPCols")),
    }


def _safe_call(model, name: str):
    func = getattr(model, name, None)
    if func is None:
        return None
    try:
        return func()
    except Exception:
        return None


def build_node_policy(bundle: Optional[NodeModelBundle], node_weight: float) -> Optional[NodeSelectionPolicy]:
    if bundle is None:
        return None
    try:
        return NodeSelectionPolicy(bundle, node_weight)
    except Exception:
        return None


def build_cut_policy(bundle: Optional[CutModelBundle]) -> Optional[CutSelectionPolicy]:
    if bundle is None:
        return None
    try:
        return CutSelectionPolicy(bundle)
    except Exception:
        return None
