from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Optional


def save_bundle(bundle: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(bundle, handle)


def load_bundle(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)
