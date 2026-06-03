"""Manifest building and export logic for the KAIST rotating machine adapter."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Tuple

import pandas as pd

from pdm_pipeline.utils import ensure_directory, save_json

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


@dataclass
class ParsedCache:
    """In-memory cache of parsed modality files to avoid duplicate reads."""

    vibration: Dict[str, Tuple[SourceRecord, ParsedStream]]
    current_temp: Dict[str, Tuple[SourceRecord, ParsedCurrentTemp]]
    acoustic: Dict[str, Tuple[SourceRecord, ParsedStream]]


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
) -> Tuple[Dict[str, List[SourceRecord]], List[Dict[str, Any]], List[Dict[str, str]]]:
    """Collect normalized file records, inventory rows, and normalization warnings."""

    records_by_modality: Dict[str, List[SourceRecord]] = {"vibration": [], "current_temp": [], "acoustic": []}
    inventory_rows: List[Dict[str, Any]] = []
    warning_rows: List[Dict[str, str]] = []

    for modality, (folder_name, suffix) in EXPECTED_SOURCE_DIRS.items():
        for source_path in sorted((dataset_root / folder_name).glob(f"*{suffix}")):
            record = parse_filename(source_path, modality=modality)
            records_by_modality[modality].append(record)
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


def _write_empty_csv(path: Path, columns: List[str]) -> None:
    """Create a CSV with headers only when no rows are available."""

    pd.DataFrame(columns=columns).to_csv(path, index=False)


def _write_csv_safe(frame: pd.DataFrame, path: Path) -> None:
    """Write one CSV and raise a clear adapter error on disk exhaustion."""

    try:
        frame.to_csv(path, index=False)
    except OSError as exc:
        if getattr(exc, "errno", None) == 28:
            raise KaistAdapterError(
                f"Insufficient disk space while writing {path}. "
                "The KAIST raw-to-CSV export is storage-intensive because it preserves raw samples."
            ) from exc
        raise


def _materialize_processed_csv(source_path: Path, target_path: Path) -> None:
    """Reuse an interim CSV in the processed tree without rewriting the data."""

    if target_path.exists():
        target_path.unlink()
    try:
        os.link(source_path, target_path)
    except OSError:
        shutil.copy2(source_path, target_path)


def _modality_info(parsed_stream: ParsedStream, export_file: str) -> Dict[str, Any]:
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
        "current_preserved_but_not_required": True,
        "source_files": source_files,
        "modality_info": modality_info,
        "normalization_warnings": warnings,
    }


def _write_interim_exports(
    dataset_root: Path,
    interim_root: Path,
    records_by_modality: Dict[str, List[SourceRecord]],
    warning_rows: List[Dict[str, str]],
) -> ParsedCache:
    """Parse raw files once and export normalized modality-level CSV files."""

    ensure_directory(interim_root)
    ensure_directory(interim_root / "vibration")
    ensure_directory(interim_root / "thermal")
    ensure_directory(interim_root / "current")
    ensure_directory(interim_root / "acoustic")
    source_metadata_dir = ensure_directory(interim_root / "source_metadata")

    parsed_vibration: Dict[str, Tuple[SourceRecord, ParsedStream]] = {}
    parsed_current_temp: Dict[str, Tuple[SourceRecord, ParsedCurrentTemp]] = {}
    parsed_acoustic: Dict[str, Tuple[SourceRecord, ParsedStream]] = {}

    for record in records_by_modality["vibration"]:
        if record.condition.condition_key in parsed_vibration:
            raise KaistAdapterError(f"Duplicate vibration condition key '{record.condition.condition_key}'.")
        parsed = parse_vibration_mat(record)
        _write_csv_safe(
            parsed.frame,
            interim_root / "vibration" / f"{record.condition.condition_key}.csv",
        )
        save_json(source_metadata_dir / f"{record.condition.condition_key}__vibration.json", parsed.source_metadata)
        parsed_vibration[record.condition.condition_key] = (record, parsed)

    for record in records_by_modality["current_temp"]:
        if record.condition.condition_key in parsed_current_temp:
            raise KaistAdapterError(f"Duplicate current_temp condition key '{record.condition.condition_key}'.")
        parsed = parse_current_temp_tdms(record)
        _write_csv_safe(
            parsed.thermal.frame,
            interim_root / "thermal" / f"{record.condition.condition_key}.csv",
        )
        _write_csv_safe(
            parsed.current.frame,
            interim_root / "current" / f"{record.condition.condition_key}.csv",
        )
        save_json(
            source_metadata_dir / f"{record.condition.condition_key}__current_temp.json",
            parsed.source_metadata,
        )
        parsed_current_temp[record.condition.condition_key] = (record, parsed)
        for warning in parsed.warnings:
            warning_rows.append(
                {
                    "condition_key": record.condition.condition_key,
                    "source_path": _relative_source_path(dataset_root, record.source_path),
                    "issue_type": "embedded_metadata_mismatch",
                    "issue_detail": warning,
                }
            )

    for record in records_by_modality["acoustic"]:
        if record.condition.condition_key in parsed_acoustic:
            raise KaistAdapterError(f"Duplicate acoustic condition key '{record.condition.condition_key}'.")
        parsed = parse_acoustic_mat(record)
        _write_csv_safe(
            parsed.frame,
            interim_root / "acoustic" / f"{record.condition.condition_key}.csv",
        )
        save_json(source_metadata_dir / f"{record.condition.condition_key}__acoustic.json", parsed.source_metadata)
        parsed_acoustic[record.condition.condition_key] = (record, parsed)

    return ParsedCache(vibration=parsed_vibration, current_temp=parsed_current_temp, acoustic=parsed_acoustic)


def _write_processed_exports(
    dataset_root: Path,
    interim_root: Path,
    processed_root: Path,
    cache: ParsedCache,
    warning_rows: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Assemble processed session folders and manifests from parsed modality caches."""

    manifests_dir = ensure_directory(processed_root / "manifests")
    primary_dir = ensure_directory(processed_root / "primary_sessions")
    acoustic_dir = ensure_directory(processed_root / "optional_acoustic_sessions")

    session_rows: List[Dict[str, Any]] = []
    modality_rows: List[Dict[str, Any]] = []
    primary_condition_keys = sorted(set(cache.vibration) & set(cache.current_temp))
    label_counter: Counter[str] = Counter()
    multiclass_counter: Counter[str] = Counter()

    for condition_key in primary_condition_keys:
        vibration_record, vibration_stream = cache.vibration[condition_key]
        current_record, current_temp = cache.current_temp[condition_key]
        session_id = vibration_record.condition.session_id
        session_dir = ensure_directory(primary_dir / session_id)

        _materialize_processed_csv(
            interim_root / "vibration" / f"{condition_key}.csv",
            session_dir / "vibration.csv",
        )
        _materialize_processed_csv(
            interim_root / "thermal" / f"{condition_key}.csv",
            session_dir / "thermal.csv",
        )
        _materialize_processed_csv(
            interim_root / "current" / f"{condition_key}.csv",
            session_dir / "current.csv",
        )

        warnings = vibration_record.warnings + current_record.warnings + current_temp.warnings
        modality_info = {
            "vibration": _modality_info(vibration_stream, "vibration.csv"),
            "thermal": _modality_info(current_temp.thermal, "thermal.csv"),
            "current": _modality_info(current_temp.current, "current.csv"),
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

    acoustic_session_count = 0
    for condition_key, (record, acoustic_stream) in sorted(cache.acoustic.items()):
        session_id = record.condition.acoustic_session_id
        session_dir = ensure_directory(acoustic_dir / session_id)
        _materialize_processed_csv(
            interim_root / "acoustic" / f"{condition_key}.csv",
            session_dir / "acoustic.csv",
        )

        modality_info = {
            "vibration": empty_modality_info(),
            "thermal": empty_modality_info(),
            "current": empty_modality_info(),
            "acoustic": _modality_info(acoustic_stream, "acoustic.csv"),
        }
        metadata = _session_metadata(
            session_id=session_id,
            session_branch="optional_acoustic",
            record=record,
            source_files={
                "vibration": None,
                "current_temp": None,
                "acoustic": _relative_source_path(dataset_root, record.source_path),
            },
            modality_info=modality_info,
            present_modalities=["acoustic"],
            warnings=record.warnings,
        )
        save_json(session_dir / "metadata.json", metadata)

        session_rows.append(
            {
                "session_id": session_id,
                "session_branch": "optional_acoustic",
                "condition_key": condition_key,
                "label": record.condition.label,
                "multiclass_label": record.condition.multiclass_label,
                "load_code": record.condition.load_code,
                "load_nm": record.condition.load_nm,
                "fault_family": record.condition.fault_family,
                "severity_code": record.condition.severity_code,
                "severity_value": record.condition.severity_value,
                "severity_unit": record.condition.severity_unit,
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
        acoustic_session_count += 1

    pd.DataFrame(session_rows).to_csv(manifests_dir / "sessions_manifest.csv", index=False)
    pd.DataFrame(modality_rows).to_csv(manifests_dir / "modality_availability.csv", index=False)
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
        manifests_dir / "export_summary.json",
        {
            "primary_session_count": len(primary_condition_keys),
            "optional_acoustic_session_count": acoustic_session_count,
            "label_distribution": dict(label_counter),
            "multiclass_distribution": dict(multiclass_counter),
            "first_training_baseline_modalities": list(FIRST_BASELINE_MODALITIES),
            "current_exported_but_not_required": True,
            "normalization_warning_count": len(warning_rows),
        },
    )
    return {
        "primary_session_count": len(primary_condition_keys),
        "optional_acoustic_session_count": acoustic_session_count,
        "label_distribution": dict(label_counter),
        "multiclass_distribution": dict(multiclass_counter),
    }


def adapt_kaist_dataset(
    dataset_root: str | Path = Path("data/external/kaist_rotating_machine/extracted"),
    interim_root: str | Path = Path("data/interim/kaist_rotating_machine"),
    processed_root: str | Path = Path("data/processed/kaist_rotating_machine"),
) -> Dict[str, Any]:
    """Run the first strict KAIST export pass without touching training code."""

    dataset_root = Path(dataset_root)
    interim_root = Path(interim_root)
    processed_root = Path(processed_root)

    _validate_dataset_root(dataset_root)
    records_by_modality, inventory_rows, warning_rows = _scan_source_records(dataset_root)
    parsed_cache = _write_interim_exports(dataset_root, interim_root, records_by_modality, warning_rows)

    for row in inventory_rows:
        condition_key = row["condition_key"]
        if row["modality"] == "vibration":
            _record, parsed = parsed_cache.vibration[condition_key]
            row["sample_rate_hz"] = parsed.sample_rate_hz
            row["duration_s"] = parsed.duration_s
            row["channel_count"] = parsed.channel_count
        elif row["modality"] == "current_temp":
            _record, parsed = parsed_cache.current_temp[condition_key]
            row["sample_rate_hz"] = parsed.thermal.sample_rate_hz
            row["duration_s"] = parsed.thermal.duration_s
            row["channel_count"] = parsed.thermal.channel_count + parsed.current.channel_count
        else:
            _record, parsed = parsed_cache.acoustic[condition_key]
            row["sample_rate_hz"] = parsed.sample_rate_hz
            row["duration_s"] = parsed.duration_s
            row["channel_count"] = parsed.channel_count

    pd.DataFrame(inventory_rows).to_csv(interim_root / "inventory.csv", index=False)
    if warning_rows:
        pd.DataFrame(warning_rows).to_csv(interim_root / "normalization_audit.csv", index=False)
    else:
        _write_empty_csv(interim_root / "normalization_audit.csv", ["condition_key", "source_path", "issue_type", "issue_detail"])

    processed_summary = _write_processed_exports(
        dataset_root,
        interim_root,
        processed_root,
        parsed_cache,
        warning_rows,
    )
    modality_availability = {
        "primary": {
            "vibration": processed_summary["primary_session_count"],
            "thermal": processed_summary["primary_session_count"],
            "current": processed_summary["primary_session_count"],
            "acoustic": 0,
        },
        "optional_acoustic": {"acoustic": processed_summary["optional_acoustic_session_count"]},
    }
    return {
        "dataset_root": str(dataset_root),
        "interim_root": str(interim_root),
        "processed_root": str(processed_root),
        "primary_session_count": processed_summary["primary_session_count"],
        "optional_acoustic_session_count": processed_summary["optional_acoustic_session_count"],
        "label_distribution": processed_summary["label_distribution"],
        "multiclass_distribution": processed_summary["multiclass_distribution"],
        "modality_availability": modality_availability,
        "normalization_warning_count": len(warning_rows),
        "normalization_warnings": [row["issue_detail"] for row in warning_rows],
        "first_training_baseline_modalities": list(FIRST_BASELINE_MODALITIES),
        "current_exported_but_not_required": True,
    }
