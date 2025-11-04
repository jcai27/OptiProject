"""Learning-augmented branch-and-cut framework."""

from .configs.base import ExperimentConfig, load_experiment_config
from .experiments.runner import ExperimentRunner

try:  # pragma: no cover - optional ML stack
    from .ml.training import TrainingWorkflow  # type: ignore
    _TRAINING_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - defer error for training command
    TrainingWorkflow = None  # type: ignore
    _TRAINING_IMPORT_ERROR = exc

TRAINING_IMPORT_ERROR = _TRAINING_IMPORT_ERROR

__all__ = [
    "ExperimentConfig",
    "ExperimentRunner",
    "TrainingWorkflow",
    "TRAINING_IMPORT_ERROR",
    "load_experiment_config",
]
