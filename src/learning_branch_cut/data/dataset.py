from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

from ..configs.base import DatasetConfig


@dataclass
class ProblemInstance:
    """Lightweight representation of a single MILP instance."""

    identifier: str
    data_path: Path
    metadata: Dict[str, str] = field(default_factory=dict)


class InstanceDataset:
    """Utility for enumerating train/validation/test instances."""

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
        self._cache: Dict[str, ProblemInstance] = {}

    def _resolve(self, instance_id: str) -> ProblemInstance:
        data_path = self.config.root_path / f"{instance_id}{self.config.file_extension}"
        metadata_path = data_path.with_suffix(".meta.yml")
        metadata: Dict[str, str] = {}
        if metadata_path.exists():
            metadata = _load_metadata(metadata_path)
        problem = ProblemInstance(identifier=instance_id, data_path=data_path, metadata=metadata)
        self._cache[instance_id] = problem
        return problem

    def _stream(self, identifiers: Iterable[str]) -> Iterator[ProblemInstance]:
        for identifier in identifiers:
            yield self._cache.get(identifier) or self._resolve(identifier)

    def iter_train(self) -> Iterator[ProblemInstance]:
        return self._stream(self.config.train_instances)

    def iter_validation(self) -> Iterator[ProblemInstance]:
        return self._stream(self.config.validation_instances)

    def iter_test(self) -> Iterator[ProblemInstance]:
        return self._stream(self.config.test_instances)

    def available_splits(self) -> Dict[str, int]:
        return {
            "train": len(self.config.train_instances),
            "validation": len(self.config.validation_instances),
            "test": len(self.config.test_instances),
        }


def _load_metadata(path: Path) -> Dict[str, str]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - guard for optional dependency
        raise RuntimeError(
            "PyYAML is required to load metadata files. Install with `pip install pyyaml`."
        ) from exc

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
