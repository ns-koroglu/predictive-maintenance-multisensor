"""Raw file parsers for the KAIST rotating machine dataset."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import re

import numpy as np
import pandas as pd
from nptdms import TdmsFile
from scipy.io import loadmat

from .schema import KaistAdapterError, SourceRecord


@dataclass
class ParsedStream:
    """One exported modality stream plus metadata needed for manifests."""

    frame: pd.DataFrame
    channel_names: List[str]
    sample_rate_hz: float
    duration_s: float
    absolute_start_time: Optional[str]
    units: Dict[str, str]
    source_format: str
    source_metadata: Dict[str, Any]

    @property
    def channel_count(self) -> int:
        """Number of signal columns in the export schema."""

        return len(self.channel_names)


@dataclass
class ParsedCurrentTemp:
    """Thermal and current exports derived from one TDMS file."""

    thermal: ParsedStream
    current: ParsedStream
    source_metadata: Dict[str, Any]
    warnings: List[str]


def _matlab_text(value: Any) -> str:
    """Convert MATLAB char arrays and object arrays into plain Python text."""

    if isinstance(value, np.ndarray) and value.dtype == object:
        if value.size == 0:
            return ""
        value = value.flat[0]
    if isinstance(value, np.ndarray):
        flattened = value.tolist()
        if isinstance(flattened, list):
            return "".join(str(item) for item in flattened)
        return str(flattened)
    if isinstance(value, list):
        if value and isinstance(value[0], list):
            return "".join(str(item) for item in value[0])
        return "".join(str(item) for item in value)
    return str(value)


def _parse_mat_absolute_time(raw_text: str) -> Optional[str]:
    """Normalize the Test.Lab absolute-time field into ISO 8601."""

    if not raw_text:
        return None
    match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ms ([0-9]+(?:\.[0-9]+)?)$", raw_text)
    if not match:
        return None
    base_time = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
    timestamp = base_time + timedelta(milliseconds=float(match.group(2)))
    return timestamp.isoformat()


def _build_time_columns(sample_count: int, increment: float) -> pd.DataFrame:
    """Create the common timestamp and sample-index columns."""

    sample_index = np.arange(sample_count, dtype=np.int64)
    timestamp = sample_index.astype(np.float64) * float(increment)
    return pd.DataFrame({"timestamp": timestamp, "sample_index": sample_index})


def parse_vibration_mat(record: SourceRecord) -> ParsedStream:
    """Parse one KAIST vibration MAT file into the strict export schema."""

    signal = loadmat(record.source_path, simplify_cells=True).get("Signal")
    if not isinstance(signal, dict):
        raise KaistAdapterError(f"{record.source_path.name} does not contain a MATLAB 'Signal' struct.")

    x_values = signal.get("x_values", {})
    y_values = signal.get("y_values", {})
    values = np.asarray(y_values.get("values"))
    if values.ndim != 2 or values.shape[1] != 4:
        raise KaistAdapterError(
            f"{record.source_path.name} must contain a 2D vibration array with 4 channels."
        )

    increment = float(x_values.get("increment"))
    sample_rate_hz = 1.0 / increment
    duration_s = float(values.shape[0] * increment)
    channel_names = [f"vibration_point_{index}_g" for index in range(1, 5)]

    frame = _build_time_columns(values.shape[0], increment)
    for column_index, column_name in enumerate(channel_names):
        frame[column_name] = values[:, column_index].astype(np.float64)

    annotations = signal.get("function_record", {}).get("TL_export_properties_annotation", {})
    source_metadata = {
        "source_path": str(record.source_path),
        "run_name": _matlab_text(annotations.get("run_name", "")),
        "orig_location": _matlab_text(annotations.get("orig_location", "")),
        "absolute_time_raw": _matlab_text(annotations.get("absolute_time", "")),
        "channel_group": _matlab_text(annotations.get("channel_group", "")),
        "sample_rate_hz": sample_rate_hz,
        "duration_s": duration_s,
        "channel_count": 4,
    }
    return ParsedStream(
        frame=frame,
        channel_names=channel_names,
        sample_rate_hz=sample_rate_hz,
        duration_s=duration_s,
        absolute_start_time=_parse_mat_absolute_time(source_metadata["absolute_time_raw"]),
        units={name: "g" for name in channel_names},
        source_format=".mat",
        source_metadata=source_metadata,
    )


def parse_acoustic_mat(record: SourceRecord) -> ParsedStream:
    """Parse one KAIST acoustic MAT file into the optional acoustic schema."""

    signal = loadmat(record.source_path, simplify_cells=True).get("Signal")
    if not isinstance(signal, dict):
        raise KaistAdapterError(f"{record.source_path.name} does not contain a MATLAB 'Signal' struct.")

    x_values = signal.get("x_values", {})
    y_values = signal.get("y_values", {})
    values = np.asarray(y_values.get("values"), dtype=np.float64).reshape(-1)

    increment = float(x_values.get("increment"))
    sample_rate_hz = 1.0 / increment
    duration_s = float(values.shape[0] * increment)
    frame = _build_time_columns(values.shape[0], increment)
    frame["acoustic_pa"] = values

    annotations = signal.get("function_record", {}).get("TL_export_properties_annotation", {})
    source_metadata = {
        "source_path": str(record.source_path),
        "run_name": _matlab_text(annotations.get("run_name", "")),
        "orig_location": _matlab_text(annotations.get("orig_location", "")),
        "absolute_time_raw": _matlab_text(annotations.get("absolute_time", "")),
        "channel_group": _matlab_text(annotations.get("channel_group", "")),
        "sample_rate_hz": sample_rate_hz,
        "duration_s": duration_s,
        "channel_count": 1,
    }
    return ParsedStream(
        frame=frame,
        channel_names=["acoustic_pa"],
        sample_rate_hz=sample_rate_hz,
        duration_s=duration_s,
        absolute_start_time=_parse_mat_absolute_time(source_metadata["absolute_time_raw"]),
        units={"acoustic_pa": "Pa"},
        source_format=".mat",
        source_metadata=source_metadata,
    )


def _normalize_tdms_name_property(property_name: str) -> Optional[str]:
    """Map a TDMS embedded run name to the canonical condition key when possible."""

    if not property_name or not property_name.startswith("LogFile_"):
        return None

    body = property_name.replace("LogFile_", "", 1)
    parts = body.split("_")
    if len(parts) < 2:
        return None

    load_code = parts[0]
    fault_token = parts[1]
    if fault_token == "Normal":
        return f"{load_code.lower()}_normal"
    if fault_token == "Inner" and len(parts) == 3:
        severity_map = {"03mm": "03", "1mm": "10", "3mm": "30"}
        severity_code = severity_map.get(parts[2])
        return f"{load_code.lower()}_bpfi_{severity_code}" if severity_code else None
    if fault_token == "Outer" and len(parts) == 3:
        severity_map = {"03mm": "03", "1mm": "10", "3mm": "30"}
        severity_code = severity_map.get(parts[2])
        return f"{load_code.lower()}_bpfo_{severity_code}" if severity_code else None
    if fault_token == "Misalignment" and len(parts) == 3:
        severity_map = {"0.1mm": "01", "0.3mm": "03", "0.5mm": "05"}
        severity_code = severity_map.get(parts[2])
        return f"{load_code.lower()}_misalignment_{severity_code}" if severity_code else None
    if fault_token == "Unbalance" and len(parts) == 3:
        severity_map = {
            "0.583": "0583mg",
            "0.1169": "1169mg",
            "0.1751": "1751mg",
            "0.2239": "2239mg",
            "0.3318": "3318mg",
            "1751mg": "1751mg",
            "2239mg": "2239mg",
            "3318mg": "3318mg",
        }
        severity_code = severity_map.get(parts[2])
        return f"{load_code.lower()}_unbalance_{severity_code}" if severity_code else None
    return None


def parse_current_temp_tdms(record: SourceRecord) -> ParsedCurrentTemp:
    """Parse one KAIST TDMS file into separate thermal and current exports."""

    tdms_file = TdmsFile.read(record.source_path)
    if not any(group.name == "Log" for group in tdms_file.groups()):
        raise KaistAdapterError(f"{record.source_path.name} does not contain a TDMS 'Log' group.")

    log_group = tdms_file["Log"]
    channels = list(log_group.channels())
    if not channels:
        raise KaistAdapterError(f"{record.source_path.name} does not contain TDMS channels.")

    thermal_channels = [channel for channel in channels if channel.properties.get("unit_string") == "°C"]
    current_channels = [channel for channel in channels if channel.properties.get("unit_string") == "A"]
    if len(thermal_channels) != 2:
        raise KaistAdapterError(
            f"{record.source_path.name} must contain exactly 2 thermal channels, found {len(thermal_channels)}."
        )
    if len(current_channels) != 3:
        raise KaistAdapterError(
            f"{record.source_path.name} must contain exactly 3 current channels, found {len(current_channels)}."
        )

    thermal_channels = sorted(thermal_channels, key=lambda channel: channel.name)
    current_channels = sorted(current_channels, key=lambda channel: channel.name)
    thermal_arrays = [np.asarray(channel[:], dtype=np.float64) for channel in thermal_channels]
    current_arrays = [np.asarray(channel[:], dtype=np.float64) for channel in current_channels]
    thermal_lengths = [len(array) for array in thermal_arrays]
    if any(length == 0 for length in thermal_lengths):
        raise KaistAdapterError(f"{record.source_path.name} contains an empty thermal channel.")
    thermal_length = min(thermal_lengths)
    warnings: List[str] = []
    if len(set(thermal_lengths)) != 1:
        warnings.append(
            f"{record.source_path.name}: thermal channels had unequal lengths {thermal_lengths}; "
            f"thermal export was cropped to the common length {thermal_length}."
        )

    non_empty_current_indices = [index for index, array in enumerate(current_arrays) if len(array) > 0]
    if not non_empty_current_indices:
        raise KaistAdapterError(f"{record.source_path.name} does not contain any non-empty current channels.")
    current_lengths = [len(array) for array in current_arrays]
    current_length = min(len(current_arrays[index]) for index in non_empty_current_indices)
    if len(set(current_lengths)) != 1:
        warnings.append(
            f"{record.source_path.name}: current channels had lengths {current_lengths}; "
            f"non-empty channels were cropped to {current_length} and empty channels were exported as NaN."
        )

    thermal_increment = float(thermal_channels[0].properties.get("wf_increment"))
    current_reference = current_channels[non_empty_current_indices[0]]
    current_increment = float(current_reference.properties.get("wf_increment"))
    thermal_frame = _build_time_columns(thermal_length, thermal_increment)
    current_frame = _build_time_columns(current_length, current_increment)

    thermal_names = ["temp_channel_1_c", "temp_channel_2_c"]
    current_names = ["current_channel_1_a", "current_channel_2_a", "current_channel_3_a"]
    for column_name, values in zip(thermal_names, thermal_arrays):
        thermal_frame[column_name] = values[:thermal_length]
    for column_name, values in zip(current_names, current_arrays):
        if len(values) == 0:
            current_frame[column_name] = np.full(current_length, np.nan)
        else:
            current_frame[column_name] = values[:current_length]

    file_name_property = str(tdms_file.properties.get("name", ""))
    embedded_condition_key = _normalize_tdms_name_property(file_name_property)
    if embedded_condition_key and embedded_condition_key != record.condition.condition_key:
        warnings.append(
            f"{record.source_path.name}: embedded TDMS name '{file_name_property}' maps to "
            f"'{embedded_condition_key}', but the filename maps to "
            f"'{record.condition.condition_key}'. Filename was kept as the label source."
        )

    source_metadata = {
        "source_path": str(record.source_path),
        "file_properties": {str(key): str(value) for key, value in tdms_file.properties.items()},
        "group_names": [group.name for group in tdms_file.groups()],
        "channel_names": [channel.name for channel in channels],
        "embedded_name_property": file_name_property,
        "embedded_condition_key": embedded_condition_key,
        "thermal_channel_names": [channel.name for channel in thermal_channels],
        "current_channel_names": [channel.name for channel in current_channels],
        "thermal_sample_rate_hz": 1.0 / thermal_increment,
        "current_sample_rate_hz": 1.0 / current_increment,
        "thermal_duration_s": float(thermal_length * thermal_increment),
        "current_duration_s": float(current_length * current_increment),
    }

    thermal_stream = ParsedStream(
        frame=thermal_frame,
        channel_names=thermal_names,
        sample_rate_hz=1.0 / thermal_increment,
        duration_s=float(thermal_length * thermal_increment),
        absolute_start_time=str(thermal_channels[0].properties.get("wf_start_time")),
        units={name: "degC" for name in thermal_names},
        source_format=".tdms",
        source_metadata=source_metadata,
    )
    current_stream = ParsedStream(
        frame=current_frame,
        channel_names=current_names,
        sample_rate_hz=1.0 / current_increment,
        duration_s=float(current_length * current_increment),
        absolute_start_time=str(current_reference.properties.get("wf_start_time")),
        units={name: "A" for name in current_names},
        source_format=".tdms",
        source_metadata=source_metadata,
    )
    return ParsedCurrentTemp(
        thermal=thermal_stream,
        current=current_stream,
        source_metadata=source_metadata,
        warnings=warnings,
    )
