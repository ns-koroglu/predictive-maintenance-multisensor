"""Compact adapter for the Paderborn bearing dataset."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.io import loadmat

from ..config import ExperimentConfig
from ..datasets import get_dataset_metadata, standardize_feature_dataset
from ..features.common import linear_slope, rms
from ..features.thermal import extract_thermal_features
from ..features.vibration import extract_vibration_features
from ..io import TIMESTAMP_COL
from ..utils import ensure_directory, save_json


DEFAULT_VARIANT = "compact_multirate_snapshot"
FILE_NAME_PATTERN = re.compile(
    r"^(?P<n>N\d{2})_(?P<m>M\d{2})_(?P<f>F\d{2})_(?P<bearing>[A-Z0-9]+)_(?P<replicate>\d+)\.mat$"
)
HEALTHY_BEARING_PATTERN = re.compile(r"^K\d{3}$")
FAULTY_BEARING_PATTERN = re.compile(r"^K[A-Z]\d{2}$")
COMPONENT_PATTERN = re.compile(r"Component\s+([A-Z ]+?)\s+Position of damage", re.IGNORECASE | re.DOTALL)
MODE_PATTERN = re.compile(r"Mode\s+([A-Za-z \-]+?)\s+Sub-mode", re.IGNORECASE | re.DOTALL)
K002_LAYOUT_WARNING = (
    "Bearing K002 stores recordings inside a nested single subfolder and is preserved explicitly."
)
NOMINAL_RECORD_DURATION_SEC = 4.0
SAMPLING_RATE_VIBRATION_HZ = 64_000.0
SAMPLING_RATE_CURRENT_HZ = 64_000.0
SAMPLING_RATE_MECHANICAL_HZ = 4_000.0
SAMPLING_RATE_TEMPERATURE_HZ = 1.0
EXPECTED_SIGNAL_NAMES = {
    "vibration_1",
    "phase_current_1",
    "phase_current_2",
    "temp_2_bearing_module",
    "force",
    "speed",
    "torque",
}


class PaderbornAdapterError(Exception):
    """Raised when the compact Paderborn adapter cannot proceed safely."""


@dataclass(frozen=True)
class BearingFolder:
    """Resolved local structure for one bearing code folder."""

    bearing_code: str
    root_dir: Path
    recording_dir: Path
    source_folder_layout: str
    layout_warning: str | None
    mat_files: Tuple[Path, ...]
    profile_pdf: Path | None
    measuring_log_pdf: Path | None


@dataclass(frozen=True)
class FaultMetadata:
    """Conservative fault semantics extracted from one bearing profile."""

    label: str
    multiclass_label: str
    fault_component_normalized: str
    fault_origin_normalized: str
    fault_component_raw: str | None
    fault_origin_raw: str | None
    documented_fault_notes: str | None


@dataclass(frozen=True)
class ParsedRecording:
    """One parsed Paderborn measurement file plus nominal metadata."""

    signal_map: Dict[str, np.ndarray]
    raster_map: Dict[str, np.ndarray]
    nominal_record_duration_sec: float
    sampling_rate_vibration_hz: float
    sampling_rate_current_hz: float
    sampling_rate_mechanical_hz: float
    sampling_rate_temperature_hz: float


def _log(message: str) -> None:
    """Print progress for long compact builds."""

    print(message, flush=True)


def _normalize_whitespace(value: str | None) -> str | None:
    """Collapse line breaks and repeated whitespace."""

    if value is None:
        return None
    normalized = re.sub(r"\s+", " ", str(value)).strip()
    return normalized or None


def _bearing_family_prefix(bearing_code: str) -> str:
    """Return the conservative bearing family prefix."""

    if HEALTHY_BEARING_PATTERN.fullmatch(bearing_code):
        return "K"
    return str(bearing_code)[:2]


def _binary_label_from_bearing_code(bearing_code: str) -> str:
    """Return the strongly supported binary label."""

    if HEALTHY_BEARING_PATTERN.fullmatch(bearing_code):
        return "healthy"
    if FAULTY_BEARING_PATTERN.fullmatch(bearing_code):
        return "faulty"
    raise PaderbornAdapterError(f"Unsupported bearing code pattern: {bearing_code}")


def _session_id(bearing_code: str, condition_code: str, replicate_index: int) -> str:
    """Return the canonical measurement-file session id."""

    return f"paderborn_{bearing_code}_{condition_code}_{int(replicate_index):02d}"


def _group_id(bearing_code: str) -> str:
    """Return the canonical bearing-code group id."""

    return f"paderborn_{bearing_code}"


def _relative_path(root: Path, path: Path | None) -> str | None:
    """Store source paths relative to the extracted root."""

    if path is None:
        return None
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _extract_profile_text(profile_pdf: Path) -> str:
    """Extract profile PDF text using pypdf when available."""

    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("pypdf is not installed.") from error

    reader = PdfReader(str(profile_pdf))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _extract_component_raw(profile_text: str) -> str | None:
    """Extract the raw damage-component text block from a profile PDF."""

    match = COMPONENT_PATTERN.search(profile_text)
    return _normalize_whitespace(match.group(1) if match else None)


def _extract_mode_raw(profile_text: str) -> str | None:
    """Extract the raw damage-mode text block from a profile PDF."""

    match = MODE_PATTERN.search(profile_text)
    return _normalize_whitespace(match.group(1) if match else None)


def _normalize_fault_component(raw_component: str | None, label: str) -> str:
    """Normalize raw PDF component text conservatively."""

    if label == "healthy":
        return "none"
    if not raw_component:
        return "ambiguous"

    tokens = re.findall(r"[A-Z]{2}", raw_component.upper())
    token_set = set(tokens)
    if token_set == {"OR"}:
        return "outer_ring"
    if token_set == {"IR"}:
        return "inner_ring"
    if "IR" in token_set and "OR" in token_set:
        return "compound_mixed"
    return "ambiguous"


def _normalize_fault_origin(raw_mode: str | None, label: str) -> str:
    """Normalize raw PDF fault-origin text conservatively."""

    if label == "healthy":
        return "none"
    if not raw_mode:
        return "unknown"

    lowered = str(raw_mode).lower()
    detected = []
    if "artificial" in lowered:
        detected.append("artificial")
    if "fatigue" in lowered:
        detected.append("fatigue")
    if "plastic deformation" in lowered:
        detected.append("plastic_deformation")

    if not detected:
        return "unknown"
    unique_detected = sorted(set(detected))
    if len(unique_detected) == 1:
        return unique_detected[0]
    return "mixed"


def _extract_mat_struct_array(items: object) -> Sequence[object]:
    """Normalize MATLAB struct arrays loaded via scipy."""

    if isinstance(items, np.ndarray):
        return [item for item in items.reshape(-1).tolist()]
    return [items]


def _fault_metadata_from_profile(
    dataset_root: Path,
    bearing_folder: BearingFolder,
    audit_rows: List[Dict[str, object]],
) -> FaultMetadata:
    """Extract conservative fault semantics, keeping PDF failures non-blocking."""

    label = _binary_label_from_bearing_code(bearing_folder.bearing_code)
    if label == "healthy":
        return FaultMetadata(
            label="healthy",
            multiclass_label="unknown",
            fault_component_normalized="none",
            fault_origin_normalized="none",
            fault_component_raw=None,
            fault_origin_raw=None,
            documented_fault_notes=None,
        )

    if bearing_folder.profile_pdf is None:
        audit_rows.append(
            {
                "bearing_code": bearing_folder.bearing_code,
                "source_pdf": None,
                "issue_type": "missing_profile_pdf",
                "issue_detail": "Bearing profile PDF was missing; conservative fault semantics fell back to unknown.",
                "normalized_field": "fault_component_normalized",
                "resolved_value": "ambiguous",
            }
        )
        audit_rows.append(
            {
                "bearing_code": bearing_folder.bearing_code,
                "source_pdf": None,
                "issue_type": "missing_profile_pdf",
                "issue_detail": "Bearing profile PDF was missing; conservative fault semantics fell back to unknown.",
                "normalized_field": "fault_origin_normalized",
                "resolved_value": "unknown",
            }
        )
        return FaultMetadata(
            label="faulty",
            multiclass_label="unknown",
            fault_component_normalized="ambiguous",
            fault_origin_normalized="unknown",
            fault_component_raw=None,
            fault_origin_raw=None,
            documented_fault_notes="Bearing profile PDF missing; semantics kept conservative.",
        )

    try:
        profile_text = _extract_profile_text(bearing_folder.profile_pdf)
    except Exception as error:  # pragma: no cover - depends on local parser behavior
        relative_pdf = _relative_path(dataset_root, bearing_folder.profile_pdf)
        audit_rows.append(
            {
                "bearing_code": bearing_folder.bearing_code,
                "source_pdf": relative_pdf,
                "issue_type": "profile_pdf_parse_error",
                "issue_detail": f"Profile PDF could not be parsed: {error}",
                "normalized_field": "fault_component_normalized",
                "resolved_value": "ambiguous",
            }
        )
        audit_rows.append(
            {
                "bearing_code": bearing_folder.bearing_code,
                "source_pdf": relative_pdf,
                "issue_type": "profile_pdf_parse_error",
                "issue_detail": f"Profile PDF could not be parsed: {error}",
                "normalized_field": "fault_origin_normalized",
                "resolved_value": "unknown",
            }
        )
        return FaultMetadata(
            label="faulty",
            multiclass_label="unknown",
            fault_component_normalized="ambiguous",
            fault_origin_normalized="unknown",
            fault_component_raw=None,
            fault_origin_raw=None,
            documented_fault_notes="Profile PDF parse failed; semantics kept conservative.",
        )

    component_raw = _extract_component_raw(profile_text)
    origin_raw = _extract_mode_raw(profile_text)
    component_normalized = _normalize_fault_component(component_raw, label="faulty")
    origin_normalized = _normalize_fault_origin(origin_raw, label="faulty")

    relative_pdf = _relative_path(dataset_root, bearing_folder.profile_pdf)
    if component_normalized in {"compound_mixed", "ambiguous"}:
        audit_rows.append(
            {
                "bearing_code": bearing_folder.bearing_code,
                "source_pdf": relative_pdf,
                "issue_type": "conservative_fault_component_mapping",
                "issue_detail": f"Raw component value '{component_raw}' required a conservative normalization.",
                "normalized_field": "fault_component_normalized",
                "resolved_value": component_normalized,
            }
        )
    if origin_normalized in {"mixed", "unknown"}:
        audit_rows.append(
            {
                "bearing_code": bearing_folder.bearing_code,
                "source_pdf": relative_pdf,
                "issue_type": "conservative_fault_origin_mapping",
                "issue_detail": f"Raw origin value '{origin_raw}' required a conservative normalization.",
                "normalized_field": "fault_origin_normalized",
                "resolved_value": origin_normalized,
            }
        )

    notes = None
    if component_normalized == "ambiguous":
        notes = f"Raw component token '{component_raw}' remained ambiguous."
    elif component_normalized == "compound_mixed":
        notes = f"Profile PDF indicates mixed component involvement: '{component_raw}'."

    return FaultMetadata(
        label="faulty",
        multiclass_label="unknown",
        fault_component_normalized=component_normalized,
        fault_origin_normalized=origin_normalized,
        fault_component_raw=component_raw,
        fault_origin_raw=origin_raw,
        documented_fault_notes=notes,
    )


def _resolve_bearing_folder(bearing_dir: Path) -> BearingFolder:
    """Resolve flat versus nested bearing-folder layouts conservatively."""

    direct_mat_files = tuple(sorted(bearing_dir.glob("*.mat")))
    if direct_mat_files:
        recording_dir = bearing_dir
        layout = "flat_bearing_folder"
        layout_warning = None
    else:
        child_dirs = sorted(path for path in bearing_dir.iterdir() if path.is_dir())
        nested_candidates = [path for path in child_dirs if list(path.glob("*.mat"))]
        if len(nested_candidates) != 1:
            raise PaderbornAdapterError(
                f"Could not resolve a unique measurement directory for bearing {bearing_dir.name}."
            )
        recording_dir = nested_candidates[0]
        layout = "nested_single_subfolder"
        layout_warning = K002_LAYOUT_WARNING if bearing_dir.name == "K002" else (
            f"Bearing {bearing_dir.name} stores recordings inside a nested single subfolder."
        )
        direct_mat_files = tuple(sorted(recording_dir.glob("*.mat")))

    profile_pdf = recording_dir / f"{bearing_dir.name}.pdf"
    measuring_log_pdf = recording_dir / f"measuring_log_{bearing_dir.name}.pdf"
    return BearingFolder(
        bearing_code=bearing_dir.name,
        root_dir=bearing_dir,
        recording_dir=recording_dir,
        source_folder_layout=layout,
        layout_warning=layout_warning,
        mat_files=tuple(sorted(direct_mat_files)),
        profile_pdf=profile_pdf if profile_pdf.exists() else None,
        measuring_log_pdf=measuring_log_pdf if measuring_log_pdf.exists() else None,
    )


def discover_paderborn_bearings(data_root: str | Path) -> List[BearingFolder]:
    """Discover all local bearing folders and resolve layout anomalies explicitly."""

    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Paderborn data root not found: {root}")

    return [_resolve_bearing_folder(path) for path in sorted(root.iterdir()) if path.is_dir()]


def _load_paderborn_recording(recording_path: Path) -> ParsedRecording:
    """Load one Paderborn measurement file and keep only required signals."""

    payload = loadmat(recording_path, squeeze_me=True, struct_as_record=False)
    root_keys = [key for key in payload.keys() if not key.startswith("__")]
    if len(root_keys) != 1:
        raise PaderbornAdapterError(
            f"{recording_path.name} must contain exactly one root MATLAB variable, found {root_keys}."
        )
    root = payload[root_keys[0]]

    raster_map: Dict[str, np.ndarray] = {}
    for raster_struct in _extract_mat_struct_array(root.X):
        raster_name = str(getattr(raster_struct, "Raster", "")).strip()
        raster_data = np.asarray(getattr(raster_struct, "Data", []), dtype=float).reshape(-1)
        if raster_name:
            raster_map[raster_name] = raster_data

    signal_map: Dict[str, np.ndarray] = {}
    for signal_struct in _extract_mat_struct_array(root.Y):
        signal_name = str(getattr(signal_struct, "Name", "")).strip()
        signal_data = np.asarray(getattr(signal_struct, "Data", []), dtype=float).reshape(-1)
        if signal_name:
            signal_map[signal_name] = signal_data

    missing_signals = sorted(EXPECTED_SIGNAL_NAMES.difference(signal_map))
    if missing_signals:
        raise PaderbornAdapterError(
            f"{recording_path.name} is missing expected signals: {missing_signals}"
        )

    return ParsedRecording(
        signal_map=signal_map,
        raster_map=raster_map,
        nominal_record_duration_sec=NOMINAL_RECORD_DURATION_SEC,
        sampling_rate_vibration_hz=SAMPLING_RATE_VIBRATION_HZ,
        sampling_rate_current_hz=SAMPLING_RATE_CURRENT_HZ,
        sampling_rate_mechanical_hz=SAMPLING_RATE_MECHANICAL_HZ,
        sampling_rate_temperature_hz=SAMPLING_RATE_TEMPERATURE_HZ,
    )


def _frame_from_signal(timestamps: np.ndarray, column_name: str, signal: np.ndarray) -> pd.DataFrame:
    """Create a timestamped frame for the shared feature extractors."""

    return pd.DataFrame(
        {
            TIMESTAMP_COL: np.asarray(timestamps, dtype=float).reshape(-1),
            column_name: np.asarray(signal, dtype=float).reshape(-1),
        }
    )


def _extract_basic_signal_summary(
    signal: np.ndarray,
    timestamps: np.ndarray,
    prefix: str,
) -> Dict[str, float]:
    """Extract lightweight summary features for non-baseline signal blocks."""

    values = np.asarray(signal, dtype=float).reshape(-1)
    time_axis = np.asarray(timestamps, dtype=float).reshape(-1)
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_range": float(np.ptp(values)),
        f"{prefix}_rms": rms(values),
        f"{prefix}_start": float(values[0]),
        f"{prefix}_end": float(values[-1]),
        f"{prefix}_trend": float(values[-1] - values[0]) if len(values) > 1 else 0.0,
        f"{prefix}_slope": linear_slope(values, time_axis),
    }


def _extract_recording_features(recording: ParsedRecording) -> Dict[str, float]:
    """Build one compact feature row from all explicitly preserved signal blocks."""

    vibration_frame = _frame_from_signal(
        recording.raster_map["HostService"],
        "vibration_1",
        recording.signal_map["vibration_1"],
    )
    thermal_frame = _frame_from_signal(
        recording.raster_map["Temp_1Hz"],
        "bearing_temperature",
        recording.signal_map["temp_2_bearing_module"],
    )

    features: Dict[str, float] = {}
    features.update(
        extract_vibration_features(
            vibration_frame,
            fallback_sampling_rate=recording.sampling_rate_vibration_hz,
        )
    )
    features.update(extract_thermal_features(thermal_frame))

    current_timestamps = recording.raster_map["HostService"]
    for signal_name in ("phase_current_1", "phase_current_2"):
        features.update(
            _extract_basic_signal_summary(
                recording.signal_map[signal_name],
                current_timestamps,
                prefix=f"current_{signal_name}",
            )
        )

    mechanical_timestamps = recording.raster_map["Mech_4kHz"]
    for signal_name in ("force", "speed", "torque"):
        features.update(
            _extract_basic_signal_summary(
                recording.signal_map[signal_name],
                mechanical_timestamps,
                prefix=signal_name,
            )
        )
    return features


def _safe_sort_key(recording_row: Dict[str, object]) -> Tuple[str, str, int]:
    """Sort compact rows predictably by bearing, condition, and replicate."""

    return (
        str(recording_row["bearing_code"]),
        str(recording_row["condition_code"]),
        int(recording_row["replicate_index"]),
    )


def _write_adapter_summary_markdown(summary: Dict[str, object], output_path: Path) -> None:
    """Write a compact Markdown build report."""

    lines = [
        "# Paderborn Adapter Summary",
        "",
        f"- Status: `{summary['status']}`",
        f"- Data root: `{summary['data_root']}`",
        f"- Processed root: `{summary['processed_root']}`",
        f"- Discovered bearing codes: {summary['n_bearing_codes']}",
        f"- Measurement files discovered: {summary['n_recording_files']}",
        f"- Feature rows exported: {summary['n_feature_rows']}",
        f"- Binary label distribution: {summary['label_distribution']}",
        f"- Layout warnings: {summary['n_layout_warnings']}",
        f"- Normalization audit rows: {summary['n_audit_rows']}",
        "",
        "## Bearing Codes",
        ", ".join(summary.get("bearing_codes", [])) or "None",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_paderborn_compact_dataset(config: ExperimentConfig) -> Dict[str, object]:
    """Build the first compact Paderborn feature dataset and manifests."""

    dataset_root = Path(config.paths.data_root)
    processed_root = Path(config.paths.processed_root) / "paderborn"
    manifests_dir = ensure_directory(processed_root / "manifests")
    datasets_dir = ensure_directory(processed_root / "datasets")

    bearing_folders = discover_paderborn_bearings(dataset_root)
    layout_warning_rows: List[Dict[str, object]] = []
    audit_rows: List[Dict[str, object]] = []
    bearing_manifest_rows: List[Dict[str, object]] = []
    recording_manifest_rows: List[Dict[str, object]] = []
    feature_rows: List[Dict[str, object]] = []
    fault_metadata_map: Dict[str, Dict[str, object]] = {}

    total_files = sum(len(folder.mat_files) for folder in bearing_folders)
    _log(f"[Paderborn] Building compact dataset from {total_files} measurement files across {len(bearing_folders)} bearings.")

    processed_files = 0
    for bearing_folder in bearing_folders:
        if bearing_folder.layout_warning:
            layout_warning_rows.append(
                {
                    "bearing_code": bearing_folder.bearing_code,
                    "source_path": _relative_path(dataset_root, bearing_folder.recording_dir),
                    "warning_type": "nested_layout",
                    "warning_detail": bearing_folder.layout_warning,
                }
            )

        fault_metadata = _fault_metadata_from_profile(dataset_root, bearing_folder, audit_rows)
        fault_metadata_map[bearing_folder.bearing_code] = {
            "label": fault_metadata.label,
            "multiclass_label": fault_metadata.multiclass_label,
            "fault_component_normalized": fault_metadata.fault_component_normalized,
            "fault_origin_normalized": fault_metadata.fault_origin_normalized,
            "fault_component_raw": fault_metadata.fault_component_raw,
            "fault_origin_raw": fault_metadata.fault_origin_raw,
            "documented_fault_notes": fault_metadata.documented_fault_notes,
            "profile_pdf": _relative_path(dataset_root, bearing_folder.profile_pdf),
            "measuring_log_pdf": _relative_path(dataset_root, bearing_folder.measuring_log_pdf),
        }

        for mat_path in bearing_folder.mat_files:
            processed_files += 1
            if processed_files == 1 or processed_files % 200 == 0:
                _log(f"[Paderborn] Processing file {processed_files}/{total_files}: {mat_path.name}")

            match = FILE_NAME_PATTERN.fullmatch(mat_path.name)
            if not match:
                audit_rows.append(
                    {
                        "bearing_code": bearing_folder.bearing_code,
                        "source_pdf": None,
                        "issue_type": "unexpected_filename",
                        "issue_detail": f"Measurement file name did not match the strict pattern: {mat_path.name}",
                        "normalized_field": "session_id",
                        "resolved_value": "skipped",
                    }
                )
                continue

            condition_code = f"{match.group('n')}_{match.group('m')}_{match.group('f')}"
            replicate_index = int(match.group("replicate"))
            session_id = _session_id(bearing_folder.bearing_code, condition_code, replicate_index)
            group_id = _group_id(bearing_folder.bearing_code)

            base_row = {
                "session_id": session_id,
                "group_id": group_id,
                "split_group": group_id,
                "bearing_code": bearing_folder.bearing_code,
                "condition_code": condition_code,
                "operating_condition_n": match.group("n"),
                "operating_condition_m": match.group("m"),
                "operating_condition_f": match.group("f"),
                "replicate_index": replicate_index,
                "source_file_name": mat_path.name,
                "source_relative_path": _relative_path(dataset_root, mat_path),
                "source_folder_layout": bearing_folder.source_folder_layout,
                "layout_warning": bearing_folder.layout_warning,
                "label": fault_metadata.label,
                "multiclass_label": "unknown",
                "fault_component_normalized": fault_metadata.fault_component_normalized,
                "fault_origin_normalized": fault_metadata.fault_origin_normalized,
                "fault_component_raw": fault_metadata.fault_component_raw,
                "fault_origin_raw": fault_metadata.fault_origin_raw,
                "documented_fault_notes": fault_metadata.documented_fault_notes,
                "has_vibration": True,
                "has_current": True,
                "has_thermal": True,
                "has_force": True,
                "has_speed": True,
                "has_torque": True,
                "nominal_record_duration_sec": NOMINAL_RECORD_DURATION_SEC,
                "sampling_rate_vibration_hz": SAMPLING_RATE_VIBRATION_HZ,
                "sampling_rate_current_hz": SAMPLING_RATE_CURRENT_HZ,
                "sampling_rate_mechanical_hz": SAMPLING_RATE_MECHANICAL_HZ,
                "sampling_rate_temperature_hz": SAMPLING_RATE_TEMPERATURE_HZ,
                "window_pairing_strategy": "measurement_file_snapshot",
                "reference_region_role": "unknown",
            }

            record_status = "ok"
            feature_values: Dict[str, float] = {}
            try:
                recording = _load_paderborn_recording(mat_path)
                feature_values = _extract_recording_features(recording)
            except Exception as error:
                record_status = f"error: {error}"
                audit_rows.append(
                    {
                        "bearing_code": bearing_folder.bearing_code,
                        "source_pdf": None,
                        "issue_type": "recording_parse_error",
                        "issue_detail": f"{mat_path.name}: {error}",
                        "normalized_field": "record_status",
                        "resolved_value": record_status,
                    }
                )

            manifest_row = dict(base_row)
            manifest_row["record_status"] = record_status
            recording_manifest_rows.append(manifest_row)

            if record_status == "ok":
                feature_row = dict(base_row)
                feature_row["record_status"] = "ok"
                feature_row.update(feature_values)
                feature_rows.append(feature_row)

        condition_codes = {
            f"{FILE_NAME_PATTERN.fullmatch(path.name).group('n')}_{FILE_NAME_PATTERN.fullmatch(path.name).group('m')}_{FILE_NAME_PATTERN.fullmatch(path.name).group('f')}"
            for path in bearing_folder.mat_files
            if FILE_NAME_PATTERN.fullmatch(path.name)
        }
        bearing_audit_count = sum(1 for row in audit_rows if row["bearing_code"] == bearing_folder.bearing_code)
        bearing_manifest_rows.append(
            {
                "dataset_name": "paderborn",
                "dataset_variant": DEFAULT_VARIANT,
                "bearing_code": bearing_folder.bearing_code,
                "group_id": _group_id(bearing_folder.bearing_code),
                "split_group": _group_id(bearing_folder.bearing_code),
                "bearing_family_prefix": _bearing_family_prefix(bearing_folder.bearing_code),
                "label": fault_metadata.label,
                "multiclass_label": "unknown",
                "fault_component_normalized": fault_metadata.fault_component_normalized,
                "fault_origin_normalized": fault_metadata.fault_origin_normalized,
                "fault_component_raw": fault_metadata.fault_component_raw,
                "fault_origin_raw": fault_metadata.fault_origin_raw,
                "n_recordings": len(bearing_folder.mat_files),
                "n_conditions": len(condition_codes),
                "n_profile_pdfs": int(bearing_folder.profile_pdf is not None),
                "n_measuring_log_pdfs": int(bearing_folder.measuring_log_pdf is not None),
                "source_folder_layout": bearing_folder.source_folder_layout,
                "layout_warning": bearing_folder.layout_warning,
                "normalization_warning_count": bearing_audit_count,
                "documented_fault_notes": fault_metadata.documented_fault_notes,
            }
        )

    feature_frame = pd.DataFrame(sorted(feature_rows, key=_safe_sort_key))
    if feature_frame.empty:
        summary = {
            "dataset_name": "paderborn",
            "status": "no_data_available",
            "data_root": str(dataset_root),
            "processed_root": str(processed_root),
            "n_bearing_codes": len(bearing_folders),
            "n_recording_files": total_files,
            "n_feature_rows": 0,
            "label_distribution": {},
            "bearing_codes": [folder.bearing_code for folder in bearing_folders],
            "n_layout_warnings": len(layout_warning_rows),
            "n_audit_rows": len(audit_rows),
        }
        save_json(processed_root / "adapter_summary.json", summary)
        _write_adapter_summary_markdown(summary, processed_root / "adapter_summary.md")
        return summary

    metadata = get_dataset_metadata("paderborn")
    feature_frame = standardize_feature_dataset(
        frame=feature_frame,
        metadata=metadata,
        dataset_variant=DEFAULT_VARIANT,
    ).sort_values(["bearing_code", "condition_code", "replicate_index"]).reset_index(drop=True)

    bearing_manifest = pd.DataFrame(bearing_manifest_rows).sort_values("bearing_code").reset_index(drop=True)
    recording_manifest = pd.DataFrame(recording_manifest_rows).sort_values(
        ["bearing_code", "condition_code", "replicate_index"]
    ).reset_index(drop=True)
    audit_frame = pd.DataFrame(audit_rows)
    layout_frame = pd.DataFrame(layout_warning_rows)

    feature_dataset_path = datasets_dir / "paderborn_compact_feature_dataset.csv"
    bearing_manifest_path = manifests_dir / "bearing_manifest.csv"
    recording_manifest_path = manifests_dir / "recording_manifest.csv"
    audit_path = manifests_dir / "label_normalization_audit.csv"
    layout_path = manifests_dir / "layout_warnings.csv"
    fault_map_path = manifests_dir / "fault_metadata_map.json"

    feature_frame.to_csv(feature_dataset_path, index=False)
    bearing_manifest.to_csv(bearing_manifest_path, index=False)
    recording_manifest.to_csv(recording_manifest_path, index=False)
    if audit_frame.empty:
        audit_frame = pd.DataFrame(
            columns=["bearing_code", "source_pdf", "issue_type", "issue_detail", "normalized_field", "resolved_value"]
        )
    audit_frame.to_csv(audit_path, index=False)
    if layout_frame.empty:
        layout_frame = pd.DataFrame(columns=["bearing_code", "source_path", "warning_type", "warning_detail"])
    layout_frame.to_csv(layout_path, index=False)
    save_json(fault_map_path, {"dataset_name": "paderborn", "bearing_codes": fault_metadata_map})

    summary = {
        "dataset_name": "paderborn",
        "status": "ok",
        "data_root": str(dataset_root),
        "processed_root": str(processed_root),
        "n_bearing_codes": int(len(bearing_folders)),
        "bearing_codes": sorted(folder.bearing_code for folder in bearing_folders),
        "n_recording_files": int(total_files),
        "n_feature_rows": int(len(feature_frame)),
        "label_distribution": feature_frame["label"].astype(str).value_counts().to_dict(),
        "n_layout_warnings": int(len(layout_frame)),
        "n_audit_rows": int(len(audit_frame)),
        "feature_dataset_path": str(feature_dataset_path),
        "bearing_manifest_path": str(bearing_manifest_path),
        "recording_manifest_path": str(recording_manifest_path),
        "audit_path": str(audit_path),
        "layout_warnings_path": str(layout_path),
        "fault_metadata_map_path": str(fault_map_path),
    }
    save_json(processed_root / "adapter_summary.json", summary)
    _write_adapter_summary_markdown(summary, processed_root / "adapter_summary.md")
    return summary
