from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .features import build_cut_dataset, build_node_dataset
from ..configs.base import ExperimentConfig
from .models import CutSelectionModel, NodePriorityModel
from .persistence import save_bundle


class TrainingWorkflow:
    """Simple end-to-end training pipeline for node and cut models."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.telemetry_dir = config.output_directory / "telemetry"
        self.model_dir = config.output_directory / "models"

    def run(self) -> Dict[str, str]:
        node_status = self._train_node_model()
        cut_status = self._train_cut_model()
        status = {"node_model": node_status, "cut_model": cut_status}
        self._write_status(status)
        return status

    def _train_node_model(self) -> str:
        records = _load_jsonl(self.telemetry_dir / "node_logs.jsonl")
        if not records:
            return "no_data"
        X, y, feature_names = build_node_dataset(records)
        if y.size == 0:
            return "no_labels"
        if np.unique(y).size < 2:
            return "insufficient_labels"
        model = NodePriorityModel()
        try:
            model.fit(X, y, feature_names)
        except RuntimeError as exc:
            message = str(exc)
            if "scikit-learn" in message.lower():
                return "sklearn_missing"
            return f"error:{message}"
        if model.bundle is None:
            return "training_failed"
        save_bundle(model.bundle, self.model_dir / "node_model.pkl")
        return "trained"

    def _train_cut_model(self) -> str:
        records = _load_jsonl(self.telemetry_dir / "cut_logs.jsonl")
        if not records:
            return "no_data"
        X, y, feature_names = build_cut_dataset(records)
        if y.size == 0:
            return "missing_targets"
        model = CutSelectionModel()
        try:
            model.fit(X, y, feature_names)
        except RuntimeError as exc:
            message = str(exc)
            if "scikit-learn" in message.lower():
                return "sklearn_missing"
            return f"error:{message}"
        if model.bundle is None:
            return "training_failed"
        save_bundle(model.bundle, self.model_dir / "cut_model.pkl")
        return "trained"

    def _write_status(self, status: Dict[str, str]) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        with (self.model_dir / "training_status.json").open("w", encoding="utf-8") as handle:
            json.dump(status, handle, indent=2)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows
