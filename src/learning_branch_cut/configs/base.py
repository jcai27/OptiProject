from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _expand_path(value: Optional[str | Path]) -> Optional[Path]:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


@dataclass
class DatasetConfig:
    name: str = "microgrid"
    root_path: Path = field(default_factory=lambda: Path("./data"))
    train_instances: List[str] = field(default_factory=list)
    validation_instances: List[str] = field(default_factory=list)
    test_instances: List[str] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)
    file_extension: str = ".yml"

    def __post_init__(self) -> None:
        self.root_path = _expand_path(self.root_path) or Path("./data").resolve()
        if not self.file_extension.startswith("."):
            self.file_extension = f".{self.file_extension}"


@dataclass
class SolverConfig:
    backend: str = "pyscipopt"
    settings: Dict[str, Any] = field(default_factory=dict)
    time_limit_seconds: Optional[int] = None
    log_directory: Optional[Path] = None
    use_learned_policies: bool = False

    def __post_init__(self) -> None:
        self.log_directory = _expand_path(self.log_directory)


@dataclass
class ModelConfig:
    node_model: str = "logistic_regression"
    cut_model: str = "elastic_net"
    retrain_epochs: int = 1
    calibration: str = "isotonic"


@dataclass
class GuardrailConfig:
    enable_guardrails: bool = True
    no_progress_node_limit: int = 250
    no_progress_time_seconds: int = 60
    cut_budget: int = 50
    node_weight: float = 0.15


@dataclass
class StrongBranchingConfig:
    """Controls strong-branching based labeling."""

    enable_labels: bool = False
    candidate_limit: int = 8
    iteration_limit: int = 50
    label_top_fraction: float = 0.25
    score_history: int = 128
    min_positive_score: float = 1e-6


@dataclass
class InstrumentationConfig:
    enable_node_logging: bool = True
    enable_cut_logging: bool = True
    strong_branching: StrongBranchingConfig = field(default_factory=StrongBranchingConfig)


@dataclass
class ExperimentConfig:
    experiment_name: str = "debug"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    guardrails: GuardrailConfig = field(default_factory=GuardrailConfig)
    instrumentation: InstrumentationConfig = field(default_factory=InstrumentationConfig)
    output_directory: Path = field(default_factory=lambda: Path("./outputs"))

    def __post_init__(self) -> None:
        self.output_directory = _expand_path(self.output_directory) or Path("./outputs").resolve()

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ExperimentConfig":
        instrumentation_raw = raw.get("instrumentation", {}) or {}
        strong_branching_raw = instrumentation_raw.get("strong_branching", {}) or {}
        instrumentation = InstrumentationConfig(
            enable_node_logging=instrumentation_raw.get("enable_node_logging", True),
            enable_cut_logging=instrumentation_raw.get("enable_cut_logging", True),
            strong_branching=StrongBranchingConfig(**strong_branching_raw),
        )
        return cls(
            experiment_name=raw.get("experiment_name", "debug"),
            dataset=DatasetConfig(**raw.get("dataset", {})),
            solver=SolverConfig(**raw.get("solver", {})),
            models=ModelConfig(**raw.get("models", {})),
            guardrails=GuardrailConfig(**raw.get("guardrails", {})),
            instrumentation=instrumentation,
            output_directory=raw.get("output_directory", "./outputs"),
        )


def load_experiment_config(path: Path) -> ExperimentConfig:
    with path.expanduser().open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return ExperimentConfig.from_dict(raw)
