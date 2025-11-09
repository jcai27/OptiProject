from __future__ import annotations
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

try:  # pragma: no cover - depends on optional dependency
    from sklearn.linear_model import ElasticNet, LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    _SKLEARN_BACKEND = True
except ImportError:  # pragma: no cover - optional dependency missing
    ElasticNet = LogisticRegression = StandardScaler = None  # type: ignore[assignment]
    RandomForestClassifier = RandomForestRegressor = None  # type: ignore[assignment]
    _SKLEARN_BACKEND = False

try:  # pragma: no cover - optional dependency
    import xgboost as xgb

    _HAS_XGBOOST = True
except ImportError:  # pragma: no cover - optional dependency missing
    xgb = None  # type: ignore[assignment]
    _HAS_XGBOOST = False


class _FallbackStandardScaler:
    """Minimal standard scaler used when scikit-learn is unavailable."""

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "_FallbackStandardScaler":
        self.mean_ = np.mean(X, axis=0)
        scale = np.std(X, axis=0)
        scale[scale == 0.0] = 1.0
        self.scale_ = scale
        return self

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler not fitted.")
        return (X - self.mean_) / self.scale_


class _FallbackLogisticRegression:
    """Simple logistic regression optimizer using batch gradient descent."""

    def __init__(
        self,
        max_iter: int = 1000,
        lr: float = 0.1,
        reg: float = 1e-3,
        tol: float = 1e-5,
        **_: object,
    ):
        self.max_iter = max_iter
        self.lr = lr
        self.reg = reg
        self.tol = tol
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples, n_features = X.shape
        weights = np.zeros(n_features, dtype=float)
        bias = 0.0
        for _ in range(self.max_iter):
            logits = X @ weights + bias
            probs = 1.0 / (1.0 + np.exp(-logits))
            error = probs - y
            grad_w = (X.T @ error) / max(1, n_samples) + self.reg * weights
            grad_b = float(np.mean(error))
            weights -= self.lr * grad_w
            bias -= self.lr * grad_b
            if np.linalg.norm(grad_w) < self.tol:
                break
        self.coef_ = weights
        self.intercept_ = bias

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model not trained.")
        logits = X @ self.coef_ + self.intercept_
        probs = 1.0 / (1.0 + np.exp(-logits))
        probs = np.clip(probs, 1e-6, 1 - 1e-6)
        return np.column_stack([1.0 - probs, probs])


class _FallbackElasticNet:
    """Lightweight elastic-net-like regressor trained via gradient descent."""

    def __init__(
        self,
        alpha: float = 0.01,
        l1_ratio: float = 0.2,
        max_iter: int = 3000,
        lr: float = 0.001,
        tol: float = 1e-5,
        **_: object,
    ):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.lr = lr
        self.tol = tol
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples, n_features = X.shape
        weights = np.zeros(n_features, dtype=float)
        bias = float(np.mean(y)) if y.size else 0.0
        for _ in range(self.max_iter):
            preds = X @ weights + bias
            error = preds - y
            grad_w = (X.T @ error) / max(1, n_samples)
            if self.alpha > 0:
                l2_component = (1.0 - self.l1_ratio) * weights
                l1_component = self.l1_ratio * np.sign(weights)
                grad_w += self.alpha * (l2_component + l1_component)
            grad_b = float(np.mean(error))
            weights -= self.lr * grad_w
            bias -= self.lr * grad_b
            if np.linalg.norm(grad_w) < self.tol:
                break
        self.coef_ = weights
        self.intercept_ = bias

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model not trained.")
        return X @ self.coef_ + self.intercept_


if not _SKLEARN_BACKEND:  # pragma: no cover - fallback activation depends on environment
    StandardScaler = _FallbackStandardScaler  # type: ignore[assignment]
    LogisticRegression = _FallbackLogisticRegression  # type: ignore[assignment]
    ElasticNet = _FallbackElasticNet  # type: ignore[assignment]


@dataclass
class NodeModelBundle:
    model: Any
    scaler: StandardScaler
    feature_names: Sequence[str]
    model_type: str = "logistic_regression"


@dataclass
class CutModelBundle:
    model: Any
    scaler: StandardScaler
    feature_names: Sequence[str]
    model_type: str = "elastic_net"


class NodePriorityModel:
    """Probability estimator for node usefulness using logistic regression."""

    def __init__(self, model_type: str = "logistic_regression") -> None:
        self.model_type = (model_type or "logistic_regression").lower()
        self.bundle: Optional[NodeModelBundle] = None

    def fit(self, features: np.ndarray, labels: np.ndarray, feature_names: Sequence[str]) -> None:
        scaler = StandardScaler()
        X = scaler.fit_transform(features)
        model = self._build_estimator()
        model.fit(X, labels)
        self.bundle = NodeModelBundle(
            model=model, scaler=scaler, feature_names=feature_names, model_type=self.model_type
        )

    def score(self, features: np.ndarray) -> np.ndarray:
        if self.bundle is None:
            return np.zeros(features.shape[0])
        X = self.bundle.scaler.transform(features)
        return self.bundle.model.predict_proba(X)[:, 1]

    def _build_estimator(self):
        if self.model_type == "logistic_regression":
            if _SKLEARN_BACKEND:
                return LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=1000,
                    class_weight="balanced",
                )
            return LogisticRegression(max_iter=1500, lr=0.05, reg=1e-3)
        if self.model_type == "random_forest":
            self._require_sklearn("random_forest node model")
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=18,
                min_samples_leaf=2,
                n_jobs=-1,
                class_weight="balanced",
                random_state=13,
            )
        if self.model_type == "xgboost":
            self._require_xgboost("xgboost node model")
            return xgb.XGBClassifier(
                n_estimators=256,
                max_depth=10,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                reg_alpha=0.0,
                tree_method="hist",
                objective="binary:logistic",
                eval_metric="logloss",
                n_jobs=-1,
            )
        raise ValueError(f"Unknown node model type '{self.model_type}'.")

    @staticmethod
    def _require_sklearn(context: str) -> None:
        if not _SKLEARN_BACKEND:
            raise RuntimeError(
                f"{context} requires scikit-learn. Install it via `pip install scikit-learn`."
            )

    @staticmethod
    def _require_xgboost(context: str) -> None:
        if not _HAS_XGBOOST:
            raise RuntimeError(f"{context} requires xgboost. Install it via `pip install xgboost`.")


class CutSelectionModel:
    """Predicts expected dual improvement for cuts via elastic net regression."""

    def __init__(self, model_type: str = "elastic_net") -> None:
        self.model_type = (model_type or "elastic_net").lower()
        self.bundle: Optional[CutModelBundle] = None

    def fit(self, features: np.ndarray, targets: np.ndarray, feature_names: Sequence[str]) -> None:
        scaler = StandardScaler()
        X = scaler.fit_transform(features)
        model = self._build_estimator()
        model.fit(X, targets)
        self.bundle = CutModelBundle(
            model=model, scaler=scaler, feature_names=feature_names, model_type=self.model_type
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.bundle is None:
            return np.zeros(features.shape[0])
        X = self.bundle.scaler.transform(features)
        return self.bundle.model.predict(X)

    def _build_estimator(self):
        if self.model_type == "elastic_net":
            if _SKLEARN_BACKEND:
                return ElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=5000)
            return ElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=3000, lr=0.001)
        if self.model_type == "random_forest":
            self._require_sklearn("random_forest cut model")
            return RandomForestRegressor(
                n_estimators=300,
                max_depth=16,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=17,
            )
        if self.model_type == "xgboost":
            self._require_xgboost("xgboost cut model")
            return xgb.XGBRegressor(
                n_estimators=384,
                max_depth=10,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                reg_alpha=0.0,
                tree_method="hist",
                objective="reg:squarederror",
                n_jobs=-1,
            )
        raise ValueError(f"Unknown cut model type '{self.model_type}'.")

    @staticmethod
    def _require_sklearn(context: str) -> None:
        if not _SKLEARN_BACKEND:
            raise RuntimeError(
                f"{context} requires scikit-learn. Install it via `pip install scikit-learn`."
            )

    @staticmethod
    def _require_xgboost(context: str) -> None:
        if not _HAS_XGBOOST:
            raise RuntimeError(f"{context} requires xgboost. Install it via `pip install xgboost`.")
