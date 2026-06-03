"""Data loading utilities for session-based multi-sensor experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


TIMESTAMP_COL = "timestamp"
SENSOR_FILES = {
    "ae": "ae.csv",
    "vibration": "vibration.csv",
    "thermal": "thermal.csv",
}
SENSOR_SCHEMAS = {
    "ae": {
        "required_columns": [TIMESTAMP_COL],
        "minimum_signal_columns": 1,
        "description": "AE data requires 'timestamp' and at least one signal column such as 'ae'.",
    },
    "vibration": {
        "required_columns": [TIMESTAMP_COL],
        "minimum_signal_columns": 1,
        "description": "Vibration data requires 'timestamp' and at least one axis such as 'ax', 'ay', or 'az'.",
    },
    "thermal": {
        "required_columns": [TIMESTAMP_COL, "t_mean", "t_max", "hotspot_area"],
        "minimum_signal_columns": 3,
        "description": "Thermal data requires 'timestamp', 't_mean', 't_max', and 'hotspot_area'.",
    },
}


class DataFormatError(Exception):
    """Raised when an input file does not match the expected experiment format."""


@dataclass
class SessionData:
    """Container that keeps the three sensor streams together with metadata."""

    session_id: str
    ae: pd.DataFrame
    vibration: pd.DataFrame
    thermal: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)

    def sensor_frames(self) -> Dict[str, pd.DataFrame]:
        """Return a modality-keyed view used by synchronization and windowing."""

        return {
            "ae": self.ae,
            "vibration": self.vibration,
            "thermal": self.thermal,
        }


def list_session_directories(data_root: str | Path) -> List[Path]:
    """List experiment folders sorted by name."""

    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Data root not found: {root}")
    return sorted(path for path in root.iterdir() if path.is_dir())


def load_metadata(session_dir: str | Path) -> Dict[str, Any]:
    """Load optional JSON metadata for one session directory."""

    metadata_path = Path(session_dir) / "metadata.json"
    if not metadata_path.exists():
        return {}

    with metadata_path.open("r", encoding="utf-8-sig") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise DataFormatError(f"{metadata_path} must contain a JSON object.")
    return metadata


def validate_sensor_frame(frame: pd.DataFrame, sensor_key: str, csv_name: str) -> None:
    """Validate required sensor columns and provide clear error messages."""

    if sensor_key not in SENSOR_SCHEMAS:
        raise ValueError(f"Unknown sensor key: {sensor_key}")

    schema = SENSOR_SCHEMAS[sensor_key]
    required_columns = schema["required_columns"]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise DataFormatError(
            f"{csv_name} is missing required columns {missing_columns}. {schema['description']}"
        )

    signal_columns = [column for column in frame.columns if column != TIMESTAMP_COL]
    if len(signal_columns) < int(schema["minimum_signal_columns"]):
        raise DataFormatError(f"{csv_name} does not contain enough signal columns. {schema['description']}")


def read_sensor_csv(path: str | Path, sensor_key: str) -> pd.DataFrame:
    """Read one sensor CSV file and validate that timestamps exist."""

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {sensor_key} file: {csv_path}")

    frame = pd.read_csv(csv_path)
    validate_sensor_frame(frame, sensor_key=sensor_key, csv_name=csv_path.name)
    return frame


def load_session(session_dir: str | Path) -> SessionData:
    """Load AE, vibration, thermal, and metadata from one session folder."""

    folder = Path(session_dir)
    metadata = load_metadata(folder)
    session_id = str(metadata.get("session_id", folder.name))

    ae = read_sensor_csv(folder / SENSOR_FILES["ae"], "ae")
    vibration = read_sensor_csv(folder / SENSOR_FILES["vibration"], "vibration")
    thermal = read_sensor_csv(folder / SENSOR_FILES["thermal"], "thermal")

    return SessionData(
        session_id=session_id,
        ae=ae,
        vibration=vibration,
        thermal=thermal,
        metadata=metadata,
    )


def numeric_sensor_columns(frame: pd.DataFrame) -> List[str]:
    """Return all columns except the timestamp column."""

    return [
        column
        for column in frame.columns
        if column not in {TIMESTAMP_COL, "sample_index"}
    ]


def metadata_to_feature_context(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Keep numeric metadata as explainable contextual features."""

    feature_context: Dict[str, Any] = {}
    for key, value in metadata.items():
        if key in {"session_id", "label", "note", "notes", "lubrication_state"}:
            continue
        if str(key).startswith("sync_") or str(key).endswith("_sampling_rate_hz"):
            continue
        if isinstance(value, bool):
            feature_context[f"meta_{key}"] = int(value)
        elif isinstance(value, (int, float)):
            feature_context[f"meta_{key}"] = float(value)
    return feature_context
