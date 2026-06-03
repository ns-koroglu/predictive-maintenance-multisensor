"""Small utility helpers used throughout the baseline pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""

    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def prepare_experiment_directories(results_root: str | Path, experiment_name: str) -> Dict[str, Path]:
    """Create a predictable results layout for one experiment run."""

    root = ensure_directory(Path(results_root) / experiment_name)
    directories = {
        "root": root,
        "artifacts": ensure_directory(root / "artifacts"),
        "datasets": ensure_directory(root / "datasets"),
        "models": ensure_directory(root / "models"),
        "metrics": ensure_directory(root / "metrics"),
        "plots": ensure_directory(root / "plots"),
        "predictions": ensure_directory(root / "predictions"),
    }
    return directories


def _to_serializable(value: Any) -> Any:
    """Convert NumPy and pathlib objects into JSON-safe values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    """Save a JSON file using UTF-8 encoding and readable indentation."""

    target = Path(path)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(_to_serializable(payload), handle, indent=2)


def save_yaml(path: str | Path, payload: Dict[str, Any]) -> None:
    """Save a YAML file that captures experiment settings."""

    target = Path(path)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(_to_serializable(payload), handle, sort_keys=False)
