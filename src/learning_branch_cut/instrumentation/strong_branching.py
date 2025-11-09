from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Iterable, Optional, Sequence


@dataclass
class StrongBranchingOptions:
    """Runtime options for deriving labels via strong branching."""

    enabled: bool = False
    candidate_limit: int = 8
    iteration_limit: int = 50
    top_fraction: float = 0.25
    score_history: int = 128
    min_positive_score: float = 1e-6

    @classmethod
    def from_config(cls, config: Any) -> "StrongBranchingOptions":
        if config is None:
            return cls()
        fraction = getattr(config, "label_top_fraction", 0.25)
        try:
            fraction = float(fraction)
        except Exception:  # pragma: no cover - defensive
            fraction = 0.25
        fraction = min(1.0, max(0.0, fraction))
        history = getattr(config, "score_history", 128)
        try:
            history = int(history)
        except Exception:  # pragma: no cover - defensive
            history = 128
        history = max(1, history)
        return cls(
            enabled=getattr(config, "enable_labels", False),
            candidate_limit=max(1, int(getattr(config, "candidate_limit", 8) or 1)),
            iteration_limit=max(1, int(getattr(config, "iteration_limit", 50) or 1)),
            top_fraction=fraction,
            score_history=history,
            min_positive_score=float(getattr(config, "min_positive_score", 1e-6) or 0.0),
        )


class StrongBranchingLabeller:
    """Computes per-node scores using strong branching primitives."""

    def __init__(self, model, options: StrongBranchingOptions) -> None:
        self.model = model
        self.options = options
        self._recent_scores: Deque[float] = deque(maxlen=self.options.score_history)
        self._branch_cand_func = self._resolve_branch_candidate_accessor()
        self._method_names = self._discover_strong_branch_methods()
        self.supported = bool(self._branch_cand_func) and bool(self._method_names)
        self.unsupported_reason: Optional[str] = None
        if not self._branch_cand_func:
            self.unsupported_reason = "missing branch-candidate accessor (expected `getLPBranchCands`)."
        elif not self._method_names:
            self.unsupported_reason = (
                "missing strong-branching probes (expected PySCIPOpt to expose `getVarStrongbranch*`)."
            )

    def evaluate_focus_node(self) -> tuple[Optional[float], Optional[int]]:
        if not self.options.enabled or not self.supported:
            return None, None
        candidates = self._branch_candidates()
        if not candidates:
            return None, None
        limit = min(len(candidates), max(1, self.options.candidate_limit))
        best_score: Optional[float] = None
        for variable in candidates[:limit]:
            score = self._score_variable(variable)
            if score is None:
                continue
            best_score = score if best_score is None else max(best_score, score)
        if best_score is None:
            return None, None
        self._recent_scores.append(best_score)
        threshold = self._dynamic_threshold()
        label = 1 if best_score >= threshold else 0
        return best_score, label

    def _branch_candidates(self) -> Sequence[Any]:
        func = self._branch_cand_func
        if func is None:
            return []
        try:
            result = func()
        except Exception:
            return []
        if result is None:
            return []
        if isinstance(result, tuple) and result:
            variables = result[0]
        else:
            variables = result
        if not variables:
            return []
        try:
            return list(variables)
        except TypeError:
            return []

    def _resolve_branch_candidate_accessor(self):
        for name in (
            "getLPBranchCands",
            "getBranchCands",
            "getLPBranchCandidates",
        ):
            func = getattr(self.model, name, None)
            if callable(func):
                return func
        for name in dir(self.model):
            lower = name.lower()
            if "branch" in lower and "cand" in lower:
                func = getattr(self.model, name, None)
                if callable(func):
                    return func
        return None

    def _discover_strong_branch_methods(self):
        preferred = (
            "getVarStrongbranchScore",
            "getVarStrongbranchScores",
            "getVarStrongbranch",
            "getStrongbranchScore",
        )
        discovered = [
            name
            for name in preferred
            if callable(getattr(self.model, name, None))
        ]
        if discovered:
            return discovered
        fallback = []
        for name in dir(self.model):
            attr = getattr(self.model, name, None)
            if not callable(attr):
                continue
            lower = name.lower()
            if lower.startswith("get") and "strongbranch" in lower:
                fallback.append(name)
        return fallback

    def _score_variable(self, variable) -> Optional[float]:
        for name in self._method_names:
            func = getattr(self.model, name, None)
            if func is None:
                continue
            try:
                result = func(variable, self.options.iteration_limit)
            except TypeError:
                result = func(variable)
            except Exception:
                continue
            score = self._coerce_score(result)
            if score is not None:
                return score
        return None

    def _coerce_score(self, value: Any) -> Optional[float]:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            if not math.isfinite(value):
                return None
            return float(value)
        if isinstance(value, (list, tuple)):
            numeric = []
            for entry in value:
                coerced = self._coerce_score(entry)
                if coerced is not None:
                    numeric.append(coerced)
            if not numeric:
                return None
            return max(numeric)
        return None

    def _dynamic_threshold(self) -> float:
        if not self._recent_scores:
            return max(self.options.min_positive_score, 0.0)
        sorted_scores = sorted(self._recent_scores, reverse=True)
        if not sorted_scores:
            return max(self.options.min_positive_score, 0.0)
        rank = int(math.ceil(len(sorted_scores) * self.options.top_fraction)) - 1
        rank = min(max(rank, 0), len(sorted_scores) - 1)
        dynamic = sorted_scores[rank]
        return max(self.options.min_positive_score, dynamic)
