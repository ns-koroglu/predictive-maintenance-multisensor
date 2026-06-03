"""Normalization rules and lightweight schema helpers for the KAIST adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


class KaistAdapterError(Exception):
    """Raised when the dataset cannot be normalized or exported safely."""


LOAD_MAP = {
    "0Nm": 0.0,
    "2Nm": 2.0,
    "4Nm": 4.0,
}

FAULT_MAP = {
    "Normal": "normal",
    "BPFI": "bpfi",
    "BPFO": "bpfo",
    "Misalign": "misalignment",
    "Unbalance": "unbalance",
    "Unbalalnce": "unbalance",
}

MM_SEVERITY_MAP = {
    "BPFI": {"03": 0.3, "10": 1.0, "30": 3.0},
    "BPFO": {"03": 0.3, "10": 1.0, "30": 3.0},
    "Misalign": {"01": 0.1, "03": 0.3, "05": 0.5},
}

UNBALANCE_SEVERITY_MAP = {
    "0583mg": 583.0,
    "1169mg": 1169.0,
    "1751mg": 1751.0,
    "2239mg": 2239.0,
    "3318mg": 3318.0,
}

CANONICAL_MODALITIES = ("vibration", "thermal", "current", "acoustic")
PRIMARY_MODALITIES = ("vibration", "thermal", "current")
FIRST_BASELINE_MODALITIES = ("vibration", "thermal")


def _format_condition_detail(fault_family: str, severity_value: float, severity_unit: str) -> str:
    """Create a short normalized condition label for reports and metadata."""

    if fault_family == "normal":
        return "normal"
    if severity_unit == "mg":
        return f"{fault_family}_{int(severity_value)}mg"
    if severity_unit == "mm":
        return f"{fault_family}_{severity_value:.1f}mm"
    return fault_family


@dataclass(frozen=True)
class NormalizedCondition:
    """Canonical condition identity derived from a KAIST filename."""

    load_code: str
    load_nm: float
    fault_family: str
    fault_family_raw: str
    severity_code: str
    severity_value: float
    severity_unit: str
    label: str
    multiclass_label: str
    condition_detail_label: str
    condition_key: str
    session_id: str
    acoustic_session_id: str


@dataclass
class SourceRecord:
    """One normalized source file entry before export."""

    modality: str
    source_path: Path
    condition: NormalizedCondition
    warnings: List[str] = field(default_factory=list)

    @property
    def source_stem(self) -> str:
        """Return the raw filename stem for quick auditing."""

        return self.source_path.stem


def parse_filename(path: str | Path, modality: str) -> SourceRecord:
    """Normalize one KAIST filename into a canonical condition record."""

    file_path = Path(path)
    parts = file_path.stem.split("_")
    if len(parts) < 2:
        raise KaistAdapterError(f"Unexpected KAIST filename: {file_path.name}")

    warnings: List[str] = []
    load_token = parts[0]
    if load_token not in LOAD_MAP:
        raise KaistAdapterError(f"Unsupported load token '{load_token}' in {file_path.name}")

    fault_token = parts[1]
    if fault_token not in FAULT_MAP:
        raise KaistAdapterError(f"Unsupported fault token '{fault_token}' in {file_path.name}")

    if fault_token == "Unbalalnce":
        warnings.append(
            f"{file_path.name}: normalized fault token 'Unbalalnce' to 'Unbalance'."
        )

    fault_family = FAULT_MAP[fault_token]
    if fault_family == "normal":
        severity_code = "none"
        severity_value = 0.0
        severity_unit = "none"
    else:
        if len(parts) != 3:
            raise KaistAdapterError(
                f"Expected a severity token in {file_path.name} for fault family '{fault_token}'."
            )
        raw_severity = parts[2]
        if fault_token in MM_SEVERITY_MAP:
            severity_map = MM_SEVERITY_MAP[fault_token]
            if raw_severity not in severity_map:
                raise KaistAdapterError(
                    f"Unsupported severity '{raw_severity}' for {fault_token} in {file_path.name}."
                )
            severity_code = raw_severity
            severity_value = severity_map[raw_severity]
            severity_unit = "mm"
        else:
            if raw_severity not in UNBALANCE_SEVERITY_MAP:
                raise KaistAdapterError(
                    f"Unsupported unbalance severity '{raw_severity}' in {file_path.name}."
                )
            severity_code = raw_severity
            severity_value = UNBALANCE_SEVERITY_MAP[raw_severity]
            severity_unit = "mg"

    load_code = load_token
    label = "healthy" if fault_family == "normal" else "faulty"
    multiclass_label = fault_family
    condition_key = f"{load_code.lower()}_{fault_family}"
    if severity_code != "none":
        condition_key = f"{condition_key}_{severity_code.lower()}"

    condition = NormalizedCondition(
        load_code=load_code,
        load_nm=LOAD_MAP[load_code],
        fault_family=fault_family,
        fault_family_raw=fault_token,
        severity_code=severity_code,
        severity_value=severity_value,
        severity_unit=severity_unit,
        label=label,
        multiclass_label=multiclass_label,
        condition_detail_label=_format_condition_detail(fault_family, severity_value, severity_unit),
        condition_key=condition_key,
        session_id=f"kaist_{condition_key}",
        acoustic_session_id=f"kaist_acoustic_{condition_key}",
    )
    return SourceRecord(modality=modality, source_path=file_path, condition=condition, warnings=warnings)


def missing_modalities(present_modalities: List[str]) -> List[str]:
    """Return canonical modalities not present in a session branch."""

    return [modality for modality in CANONICAL_MODALITIES if modality not in present_modalities]


def empty_modality_info() -> Dict[str, Any]:
    """Return a spec-compliant placeholder for a missing modality."""

    return {
        "present": False,
        "export_file": None,
        "source_format": None,
        "channel_names": [],
        "channel_count": 0,
        "sample_rate_hz": None,
        "duration_s": None,
        "timestamp_origin": "local_relative_seconds",
        "absolute_start_time": None,
        "units": {},
    }
