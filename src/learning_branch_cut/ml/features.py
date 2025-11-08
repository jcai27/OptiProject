from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


NODE_FEATURES = [
    "depth",
    "dual_bound",
    "primal_bound",
    "gap",
    "estimate",
    "solving_time",
    "lp_iterations",
    "nodes_processed",
    "nodes_in_queue",
    "cuts_applied",
]

CUT_FEATURES = [
    "violation",
    "efficacy",
    "sparsity",
    "solving_time",
    "lp_iterations",
    "cuts_applied_total",
    "lp_rows",
    "lp_cols",
]


def build_node_dataset(
    records: List[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    features: List[List[float]] = []
    labels: List[int] = []
    for record in records:
        label = _strong_branch_label(record)
        if label is None:
            label = _node_label(record.get("outcome"))
        if label is None:
            continue
        labels.append(label)
        features.append([_to_float(record.get(name), 0.0) for name in NODE_FEATURES])

    if not labels:
        return np.empty((0, len(NODE_FEATURES))), np.empty((0,), dtype=int), NODE_FEATURES

    return np.asarray(features, dtype=float), np.asarray(labels, dtype=int), NODE_FEATURES


def build_cut_dataset(
    records: List[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    features: List[List[float]] = []
    targets: List[float] = []
    for record in records:
        target = record.get("dual_improvement")
        if target is None:
            continue
        targets.append(_to_float(target, 0.0))
        features.append([_to_float(record.get(name), 0.0) for name in CUT_FEATURES])

    if not targets:
        return np.empty((0, len(CUT_FEATURES))), np.empty((0,), dtype=float), CUT_FEATURES

    return np.asarray(features, dtype=float), np.asarray(targets, dtype=float), CUT_FEATURES


def _to_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _node_label(outcome: Optional[Any]) -> Optional[int]:
    if outcome is None:
        return None
    text = str(outcome).lower()
    if "infeasible" in text or "pruned" in text or "cutoff" in text:
        return 0
    if "nodefocused" in text:
        return 1
    if "feasible" in text or "optimal" in text:
        return 1
    # fall back to bit-mask interpretation if outcome is numeric
    try:
        mask = int(float(str(outcome)))
    except Exception:
        return None
    node_feasible_mask = 1 << 20  # SCIP_EVENTTYPE_NODEFEASIBLE
    node_infeasible_mask = 1 << 19  # SCIP_EVENTTYPE_NODEINFEASIBLE
    if mask & node_feasible_mask:
        return 1
    if mask & node_infeasible_mask:
        return 0
    return None


def _strong_branch_label(record: Dict[str, Any]) -> Optional[int]:
    value = record.get("strong_branch_label")
    if value is None:
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    try:
        numeric = int(float(value))
    except Exception:
        return None
    if numeric < 0:
        return 0
    if numeric > 0:
        return 1
    return 0
