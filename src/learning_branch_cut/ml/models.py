from __future__ import annotations
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

try:  # pragma: no cover - depends on optional dependency
    from sklearn.linear_model import ElasticNet, LogisticRegression
    from sklearn.preprocessing import StandardScaler

    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover - optional dependency missing
    ElasticNet = LogisticRegression = StandardScaler = None  # type: ignore[assignment]
    _HAS_SKLEARN = False


@dataclass
class NodeModelBundle:
    model: LogisticRegression
    scaler: StandardScaler
    feature_names: Sequence[str]


@dataclass
class CutModelBundle:
    model: ElasticNet
    scaler: StandardScaler
    feature_names: Sequence[str]


class NodePriorityModel:
    """Probability estimator for node usefulness using logistic regression."""

    def __init__(self) -> None:
        self.bundle: Optional[NodeModelBundle] = None

    def fit(self, features: np.ndarray, labels: np.ndarray, feature_names: Sequence[str]) -> None:
        if not _HAS_SKLEARN:
            raise RuntimeError(
                "scikit-learn is required to train the node model. Install with `pip install scikit-learn pandas numpy`."
            )
        scaler = StandardScaler()
        X = scaler.fit_transform(features)
        model = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
        )
        model.fit(X, labels)
        self.bundle = NodeModelBundle(model=model, scaler=scaler, feature_names=feature_names)

    def score(self, features: np.ndarray) -> np.ndarray:
        if self.bundle is None or not _HAS_SKLEARN:
            return np.zeros(features.shape[0])
        X = self.bundle.scaler.transform(features)
        return self.bundle.model.predict_proba(X)[:, 1]


class CutSelectionModel:
    """Predicts expected dual improvement for cuts via elastic net regression."""

    def __init__(self) -> None:
        self.bundle: Optional[CutModelBundle] = None

    def fit(self, features: np.ndarray, targets: np.ndarray, feature_names: Sequence[str]) -> None:
        if not _HAS_SKLEARN:
            raise RuntimeError(
                "scikit-learn is required to train the cut model. Install with `pip install scikit-learn pandas numpy`."
            )
        scaler = StandardScaler()
        X = scaler.fit_transform(features)
        model = ElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=5000)
        model.fit(X, targets)
        self.bundle = CutModelBundle(model=model, scaler=scaler, feature_names=feature_names)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.bundle is None or not _HAS_SKLEARN:
            return np.zeros(features.shape[0])
        X = self.bundle.scaler.transform(features)
        return self.bundle.model.predict(X)
