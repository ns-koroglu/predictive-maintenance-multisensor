"""Compact KAIST export and direct feature-building workflow."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from pdm_pipeline.features.common import safe_ratio
from pdm_pipeline.features.thermal import extract_thermal_features
from pdm_pipeline.features.vibration import extract_vibration_features
from pdm_pipeline.utils import ensure_directory, prepare_experiment_directories, save_json
from pdm_pipeline.windowing import slice_window

from .parsers import ParsedCurrentTemp, ParsedStream, parse_acoustic_mat, parse_current_temp_tdms, parse_vibration_mat
from .schema import (
    FIRST_BASELINE_MODALITIES,
    KaistAdapterError,
    SourceRecord,
    empty_modality_info,
    missing_modalities,
    parse_filename,
)


EXPECTED_SOURCE_DIRS = {
    "vibration": ("vibration", ".mat"),
    "current_temp": ("current_temp", ".tdms"),
    "acoustic": ("acoustic", ".mat"),
}


def _validate_dataset_root(dataset_root: Path) -> None:
    """Ensure the expected KAIST source folders exist before export begins."""

    if not dataset_root.exists():
        raise KaistAdapterError(f"KAIST dataset root not found: {dataset_root}")
    for subdirectory, _suffix in EXPECTED_SOURCE_DIRS.values():
        if not (dataset_root / subdirectory).exists():
            raise KaistAdapterError(
                f"Expected KAIST subdirectory '{subdirectory}' under {dataset_root}."
            )


def _relative_source_path(dataset_root: Path, source_path: Path) -> str:
    """Store source files relative to the KAIST dataset parent directory."""

    try:
        return str(source_path.relative_to(dataset_root.parent))
    except ValueError:
        return str(source_path)


def _scan_source_records(
    dataset_root: Path,
) -> Tuple[Dict[str, Dict[str, SourceRecord]], List[Dict[str, Any]], List[Dict[str, str]]]:
    """Collect normalized file records, inventory rows, and normalization warnings."""

    records_by_modality: Dict[str, Dict[str, SourceRecord]] = {
        "vibration": {},
        "current_temp": {},
        "acoustic": {},
    }
    inventory_rows: List[Dict[str, Any]] = []
    warning_rows: List[Dict[str, str]] = []

    for modality, (folder_name, suffix) in EXPECTED_SOURCE_DIRS.items():
        for source_path in sorted((dataset_root / folder_name).glob(f"*{suffix}")):
            record = parse_filename(source_path, modality=modality)
            if record.condition.condition_key in records_by_modality[modality]:
                raise KaistAdapterError(
                    f"Duplicate {modality} condition key '{record.condition.condition_key}'."
                )
            records_by_modality[modality][record.condition.condition_key] = record
            inventory_rows.append(
                {
                    "modality": modality,
                    "source_path": _relative_source_path(dataset_root, source_path),
                    "load_code": record.condition.load_code,
                    "fault_family_raw": record.condition.fault_family_raw,
                    "fault_family": record.condition.fault_family,
                    "severity_code": record.condition.severity_code,
                    "condition_key": record.condition.condition_key,
                    "label": record.condition.label,
                    "multiclass_label": record.condition.multiclass_label,
                }
            )
            for warning in record.warnings:
                warning_rows.append(
                    {
                        "condition_key": record.condition.condition_key,
                        "source_path": _relative_source_path(dataset_root, source_path),
                        "issue_type": "filename_normalization",
                        "issue_detail": warning,
                    }
                )
    return records_by_modality, inventory_rows, warning_rows


def _modality_info(parsed_stream: ParsedStream, export_file: str | None) -> Dict[str, Any]:
    """Build one modality-info object for metadata.json."""

    return {
        "present": True,
        "export_file": export_file,
        "source_format": parsed_stream.source_format,
        "channel_names": parsed_stream.channel_names,
        "channel_count": parsed_stream.channel_count,
        "sample_rate_hz": parsed_stream.sample_rate_hz,
        "duration_s": parsed_stream.duration_s,
        "timestamp_origin": "local_relative_seconds",
        "absolute_start_time": parsed_stream.absolute_start_time,
        "units": parsed_stream.units,
    }


def _session_metadata(
    session_id: str,
    session_branch: str,
    record: SourceRecord,
    source_files: Dict[str, Any],
    modality_info: Dict[str, Dict[str, Any]],
    present_modalities: List[str],
    warnings: List[str],
) -> Dict[str, Any]:
    """Build one session metadata payload that follows the approved KAIST spec."""

    condition = record.condition
    return {
        "schema_version": "kaist_adapter_v1",
        "dataset_name": "kaist_rotating_machine",
        "session_id": session_id,
        "session_branch": session_branch,
        "condition_key": condition.condition_key,
        "label": condition.label,
        "multiclass_label": condition.multiclass_label,
        "label_source": "normalized_filename",
        "load_code": condition.load_code,
        "load_nm": condition.load_nm,
        "fault_family": condition.fault_family,
        "fault_family_raw": condition.fault_family_raw,
        "severity_code": condition.severity_code,
        "severity_value": condition.severity_value,
        "severity_unit": condition.severity_unit,
        "condition_detail_label": condition.condition_detail_label,
        "available_modalities": present_modalities,
        "missing_modalities": missing_modalities(present_modalities),
        "sync_status": "condition_matched_unsynchronized",
        "shared_timebase": False,
        "cross_modality_alignment_allowed": False,
        "first_training_baseline_modalities": list(FIRST_BASELINE_MODALITIES),
        "window_pairing_strategy": "relative_elapsed_time_condition_matched",
        "current_preserved_but_not_required": True,
        "source_files": source_files,
        "modality_info": modality_info,
        "normalization_warnings": warnings,
    }


def _save_preview(frame: pd.DataFrame, path: Path, preview_rows: int) -> None:
    """Save a tiny preview CSV when requested."""

    if preview_rows <= 0:
        return
    frame.head(preview_rows).to_csv(path, index=False)


def _augment_thermal_window(thermal_window: pd.DataFrame) -> pd.DataFrame:
    """Create in-memory derived thermal columns without exporting them to disk."""

    augmented = thermal_window.copy()
    temp_columns = [column for column in augmented.columns if column.startswith("temp_channel_")]
    if len(temp_columns) >= 2:
        augmented["t_mean"] = augmented[temp_columns].mean(axis=1)
        augmented["t_max"] = augmented[temp_columns].max(axis=1)
    return augmented


def _mean_feature_value(feature_dict: Dict[str, float], prefix: str, suffix: str) -> float:
    """Average selected feature values when multiple channels are present."""

    values = [
        float(value)
        for key, value in feature_dict.items()
        if key.startswith(prefix) and key.endswith(suffix)
    ]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _extract_kaist_window_features(
    vibration_window: pd.DataFrame,
    thermal_window: pd.DataFrame,
    load_nm: float,
    vibration_sampling_rate_hz: float,
) -> Dict[str, float]:
    """Build explainable fused features for the first KAIST baseline."""

    thermal_augmented = _augment_thermal_window(thermal_window)
    vibration_features = extract_vibration_features(
        vibration_window,
        fallback_sampling_rate=vibration_sampling_rate_hz,
    )
    thermal_features = extract_thermal_features(thermal_augmented)

    features: Dict[str, float] = {}
    features.update(vibration_features)
    features.update(thermal_features)

    vibration_rms = _mean_feature_value(vibration_features, "vibration_", "_rms")
    thermal_mean = float(
        thermal_features.get(
            "thermal_t_mean_mean",
            _mean_feature_value(thermal_features, "thermal_temp_channel_", "_mean"),
        )
    )
    if vibration_rms > 0.0 and thermal_mean > 0.0:
        features["fusion_thermal_to_vibration_ratio"] = safe_ratio(thermal_mean, vibration_rms)
    features["meta_load_nm"] = float(load_nm)
    return features


def _build_feature_rows(
    session_id: str,
    condition_key: str,
    label: str,
    multiclass_label: str,
    load_nm: float,
    vibration_stream: ParsedStream,
    thermal_stream: ParsedStream,
    window_duration_sec: float,
    overlap: float,
    minimum_vibration_samples: int,
    minimum_thermal_samples: int,
) -> List[Dict[str, Any]]:
    """Window two unsynchronized modality streams on local time and extract features."""

    if window_duration_sec <= 0:
        raise KaistAdapterError("Window duration must be positive.")
    if not 0 <= overlap < 1:
        raise KaistAdapterError("Window overlap must be in the range [0, 1).")

    step_sec = window_duration_sec * (1.0 - overlap)
    if step_sec <= 0:
        raise KaistAdapterError("Window step size must be positive.")

    end_time = min(
        float(vibration_stream.frame["timestamp"].max()),
        float(thermal_stream.frame["timestamp"].max()),
    )
    feature_rows: List[Dict[str, Any]] = []
    window_index = 0
    current_start = 0.0
    while current_start + window_duration_sec <= end_time + 1e-12:
        current_end = current_start + window_duration_sec
        vibration_window = slice_window(vibration_stream.frame, current_start, current_end)
        thermal_window = slice_window(thermal_stream.frame, current_start, current_end)
        if len(vibration_window) >= minimum_vibration_samples and len(thermal_window) >= minimum_thermal_samples:
            row = {
                "session_id": session_id,
                "condition_key": condition_key,
                "label": label,
                "multiclass_label": multiclass_label,
                "window_index": window_index,
                "start_time": float(current_start),
                "end_time": float(current_end),
                "window_pairing_strategy": "relative_elapsed_time_condition_matched",
            }
            row.update(
                _extract_kaist_window_features(
                    vibration_window=vibration_window,
                    thermal_window=thermal_window,
                    load_nm=load_nm,
                    vibration_sampling_rate_hz=vibration_stream.sample_rate_hz,
                )
            )
            feature_rows.append(row)
        current_start += step_sec
        window_index += 1

    return feature_rows


def run_kaist_compact_workflow(
    dataset_root: str | Path = Path("data/external/kaist_rotating_machine/extracted"),
    processed_root: str | Path = Path("data/processed/kaist_rotating_machine"),
    results_root: str | Path = Path("results"),
    experiment_name: str = "kaist_feature_build",
    preview_rows: int = 128,
    window_duration_sec: float = 2.0,
    overlap: float = 0.5,
    minimum_vibration_samples: int = 16,
    minimum_thermal_samples: int = 2,
) -> Dict[str, Any]:
    """Run the storage-efficient KAIST compact export and feature build."""

    dataset_root = Path(dataset_root)
    processed_root = Path(processed_root)
    _validate_dataset_root(dataset_root)

    records_by_modality, inventory_rows, warning_rows = _scan_source_records(dataset_root)
    primary_keys = sorted(set(records_by_modality["vibration"]) & set(records_by_modality["current_temp"]))
    acoustic_keys = sorted(records_by_modality["acoustic"])

    manifests_dir = ensure_directory(processed_root / "manifests")
    primary_dir = ensure_directory(processed_root / "primary_sessions")
    acoustic_dir = ensure_directory(processed_root / "optional_acoustic_sessions")
    result_dirs = prepare_experiment_directories(results_root, experiment_name)

    session_rows: List[Dict[str, Any]] = []
    modality_rows: List[Dict[str, Any]] = []
    feature_rows: List[Dict[str, Any]] = []
    label_counter: Counter[str] = Counter()
    multiclass_counter: Counter[str] = Counter()

    for condition_key in primary_keys:
        vibration_record = records_by_modality["vibration"][condition_key]
        current_record = records_by_modality["current_temp"][condition_key]
        vibration_stream = parse_vibration_mat(vibration_record)
        current_temp = parse_current_temp_tdms(current_record)
        session_id = vibration_record.condition.session_id
        session_dir = ensure_directory(primary_dir / session_id)

        warnings = vibration_record.warnings + current_record.warnings + current_temp.warnings
        for warning in current_temp.warnings:
            warning_rows.append(
                {
                    "condition_key": condition_key,
                    "source_path": _relative_source_path(dataset_root, current_record.source_path),
                    "issue_type": "embedded_metadata_mismatch",
                    "issue_detail": warning,
                }
            )

        _save_preview(vibration_stream.frame, session_dir / "vibration_preview.csv", preview_rows)
        _save_preview(current_temp.thermal.frame, session_dir / "thermal_preview.csv", preview_rows)
        _save_preview(current_temp.current.frame, session_dir / "current_preview.csv", preview_rows)

        modality_info = {
            "vibration": _modality_info(
                vibration_stream,
                "vibration_preview.csv" if preview_rows > 0 else None,
            ),
            "thermal": _modality_info(
                current_temp.thermal,
                "thermal_preview.csv" if preview_rows > 0 else None,
            ),
            "current": _modality_info(
                current_temp.current,
                "current_preview.csv" if preview_rows > 0 else None,
            ),
            "acoustic": empty_modality_info(),
        }
        metadata = _session_metadata(
            session_id=session_id,
            session_branch="primary",
            record=vibration_record,
            source_files={
                "vibration": _relative_source_path(dataset_root, vibration_record.source_path),
                "current_temp": _relative_source_path(dataset_root, current_record.source_path),
                "acoustic": None,
            },
            modality_info=modality_info,
            present_modalities=["vibration", "thermal", "current"],
            warnings=warnings,
        )
        save_json(session_dir / "metadata.json", metadata)

        session_rows.append(
            {
                "session_id": session_id,
                "session_branch": "primary",
                "condition_key": condition_key,
                "label": vibration_record.condition.label,
                "multiclass_label": vibration_record.condition.multiclass_label,
                "load_code": vibration_record.condition.load_code,
                "load_nm": vibration_record.condition.load_nm,
                "fault_family": vibration_record.condition.fault_family,
                "severity_code": vibration_record.condition.severity_code,
                "severity_value": vibration_record.condition.severity_value,
                "severity_unit": vibration_record.condition.severity_unit,
                "available_modalities": "vibration;thermal;current",
                "missing_modalities": "acoustic",
                "sync_status": "condition_matched_unsynchronized",
            }
        )
        for modality_name, modality_payload in modality_info.items():
            modality_rows.append(
                {
                    "session_id": session_id,
                    "session_branch": "primary",
                    "modality": modality_name,
                    "present": modality_payload["present"],
                    "export_file": modality_payload["export_file"],
                    "sample_rate_hz": modality_payload["sample_rate_hz"],
                    "duration_s": modality_payload["duration_s"],
                    "channel_count": modality_payload["channel_count"],
                }
            )
        label_counter[vibration_record.condition.label] += 1
        multiclass_counter[vibration_record.condition.multiclass_label] += 1
        feature_rows.extend(
            _build_feature_rows(
                session_id=session_id,
                condition_key=condition_key,
                label=vibration_record.condition.label,
                multiclass_label=vibration_record.condition.multiclass_label,
                load_nm=vibration_record.condition.load_nm,
                vibration_stream=vibration_stream,
                thermal_stream=current_temp.thermal,
                window_duration_sec=window_duration_sec,
                overlap=overlap,
                minimum_vibration_samples=minimum_vibration_samples,
                minimum_thermal_samples=minimum_thermal_samples,
            )
        )

    for condition_key in acoustic_keys:
        acoustic_record = records_by_modality["acoustic"][condition_key]
        acoustic_stream = parse_acoustic_mat(acoustic_record)
        session_id = acoustic_record.condition.acoustic_session_id
        session_dir = ensure_directory(acoustic_dir / session_id)
        _save_preview(acoustic_stream.frame, session_dir / "acoustic_preview.csv", preview_rows)

        modality_info = {
            "vibration": empty_modality_info(),
            "thermal": empty_modality_info(),
            "current": empty_modality_info(),
            "acoustic": _modality_info(
                acoustic_stream,
                "acoustic_preview.csv" if preview_rows > 0 else None,
            ),
        }
        metadata = _session_metadata(
            session_id=session_id,
            session_branch="optional_acoustic",
            record=acoustic_record,
            source_files={
                "vibration": None,
                "current_temp": None,
                "acoustic": _relative_source_path(dataset_root, acoustic_record.source_path),
            },
            modality_info=modality_info,
            present_modalities=["acoustic"],
            warnings=acoustic_record.warnings,
        )
        save_json(session_dir / "metadata.json", metadata)

        session_rows.append(
            {
                "session_id": session_id,
                "session_branch": "optional_acoustic",
                "condition_key": condition_key,
                "label": acoustic_record.condition.label,
                "multiclass_label": acoustic_record.condition.multiclass_label,
                "load_code": acoustic_record.condition.load_code,
                "load_nm": acoustic_record.condition.load_nm,
                "fault_family": acoustic_record.condition.fault_family,
                "severity_code": acoustic_record.condition.severity_code,
                "severity_value": acoustic_record.condition.severity_value,
                "severity_unit": acoustic_record.condition.severity_unit,
                "available_modalities": "acoustic",
                "missing_modalities": "vibration;thermal;current",
                "sync_status": "condition_matched_unsynchronized",
            }
        )
        for modality_name, modality_payload in modality_info.items():
            modality_rows.append(
                {
                    "session_id": session_id,
                    "session_branch": "optional_acoustic",
                    "modality": modality_name,
                    "present": modality_payload["present"],
                    "export_file": modality_payload["export_file"],
                    "sample_rate_hz": modality_payload["sample_rate_hz"],
                    "duration_s": modality_payload["duration_s"],
                    "channel_count": modality_payload["channel_count"],
                }
            )

    inventory_frame = pd.DataFrame(inventory_rows)
    sessions_frame = pd.DataFrame(session_rows)
    modality_frame = pd.DataFrame(modality_rows)
    warnings_frame = pd.DataFrame(warning_rows)
    feature_frame = pd.DataFrame(feature_rows)
    feature_columns = [
        column
        for column in feature_frame.columns
        if column
        not in {
            "session_id",
            "condition_key",
            "label",
            "multiclass_label",
            "window_index",
            "start_time",
            "end_time",
            "window_pairing_strategy",
        }
    ]

    inventory_frame.to_csv(manifests_dir / "inventory.csv", index=False)
    warnings_frame.to_csv(manifests_dir / "normalization_audit.csv", index=False)
    sessions_frame.to_csv(manifests_dir / "sessions_manifest.csv", index=False)
    modality_frame.to_csv(manifests_dir / "modality_availability.csv", index=False)
    save_json(
        manifests_dir / "label_map.json",
        {
            "label_source": "normalized_filename",
            "binary_labels": {
                "normal": "healthy",
                "bpfi": "faulty",
                "bpfo": "faulty",
                "misalignment": "faulty",
                "unbalance": "faulty",
            },
            "multiclass_labels": {
                "normal": "normal",
                "bpfi": "bpfi",
                "bpfo": "bpfo",
                "misalignment": "misalignment",
                "unbalance": "unbalance",
            },
        },
    )
    save_json(
        manifests_dir / "compact_export_summary.json",
        {
            "primary_session_count": len(primary_keys),
            "optional_acoustic_session_count": len(acoustic_keys),
            "label_distribution": dict(label_counter),
            "multiclass_distribution": dict(multiclass_counter),
            "normalization_warning_count": len(warning_rows),
            "first_training_baseline_modalities": list(FIRST_BASELINE_MODALITIES),
            "current_exported_but_not_required": True,
        },
    )

    feature_frame.to_csv(result_dirs["datasets"] / "kaist_vibration_thermal_features.csv", index=False)
    sessions_frame.to_csv(result_dirs["datasets"] / "kaist_sessions_manifest.csv", index=False)
    warnings_frame.to_csv(result_dirs["datasets"] / "kaist_normalization_audit.csv", index=False)
    save_json(result_dirs["artifacts"] / "feature_columns.json", {"feature_columns": feature_columns})

    summary = {
        "dataset_root": str(dataset_root),
        "processed_root": str(processed_root),
        "results_root": str(result_dirs["root"]),
        "primary_session_count": len(primary_keys),
        "optional_acoustic_session_count": len(acoustic_keys),
        "sessions_used_for_first_baseline": len(primary_keys),
        "label_distribution": dict(label_counter),
        "multiclass_distribution": dict(multiclass_counter),
        "feature_row_count": int(len(feature_frame)),
        "feature_column_count": int(len(feature_columns)),
        "normalization_warning_count": len(warning_rows),
        "first_training_baseline_modalities": list(FIRST_BASELINE_MODALITIES),
        "current_exported_but_not_required": True,
        "modality_availability": {
            "primary": {
                "vibration": len(primary_keys),
                "thermal": len(primary_keys),
                "current": len(primary_keys),
                "acoustic": 0,
            },
            "optional_acoustic": {"acoustic": len(acoustic_keys)},
        },
    }
    save_json(result_dirs["metrics"] / "kaist_feature_build_summary.json", summary)

    summary_markdown = [
        "# KAIST Feature Build Summary",
        "",
        f"- Primary sessions used: {summary['primary_session_count']}",
        f"- Optional acoustic sessions exported separately: {summary['optional_acoustic_session_count']}",
        f"- Label distribution: {summary['label_distribution']}",
        f"- Multiclass distribution: {summary['multiclass_distribution']}",
        f"- Window rows produced: {summary['feature_row_count']}",
        f"- Feature columns produced: {summary['feature_column_count']}",
        f"- Normalization warnings: {summary['normalization_warning_count']}",
        f"- First baseline modalities: {', '.join(summary['first_training_baseline_modalities'])}",
        "- Current is preserved in metadata and compact previews, but not used in the first baseline feature set.",
        "- Window pairing strategy: relative elapsed time within each condition, explicitly not physical synchronization.",
    ]
    (result_dirs["root"] / "kaist_feature_build_summary.md").write_text(
        "\n".join(summary_markdown),
        encoding="utf-8",
    )

    return summary
