"""Compact adapter for the KAIST ball bearing run-to-failure dataset."""

from __future__ import annotations

import time
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config import ExperimentConfig
from ..datasets import get_dataset_metadata, standardize_feature_dataset
from ..features.thermal import extract_thermal_features
from ..features.vibration import extract_vibration_features
from ..io import TIMESTAMP_COL
from ..utils import ensure_directory, save_json


SUPPORTED_EXTENSIONS = {".csv", ".txt", ".tsv"}
DEFAULT_SESSION_ID = "kaist_rtf_run_001"
EXPECTED_SIGNAL_COLUMNS = [
    "vibration_x",
    "vibration_y",
    "bearing_temperature_c",
    "ambient_temperature_c",
]
FORMAT_SAMPLE_BYTES = 8192


@dataclass(frozen=True)
class RunFileRecord:
    """One discovered hourly measurement file."""

    session_id: str
    group_id: str
    file_path: Path
    relative_path: str
    file_name: str
    extension: str
    progression_order_key: str
    progression_hint: str


@dataclass(frozen=True)
class CsvFormatSpec:
    """Compact description of a delimited file format."""

    delimiter: str
    has_header: bool
    engine: str = "c"
    encoding: str = "utf-8"


def _log(message: str, verbose: bool = True) -> None:
    """Print progress messages with flushing so long runs remain observable."""

    if verbose:
        print(message, flush=True)


def _normalize_name_token(value: str) -> str:
    """Normalize a file or folder token into a stable identifier."""

    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower()).strip("_")
    return normalized or "run"


def _candidate_session_id_from_parent(relative_parent: Path) -> str:
    """Map a relative parent path to a stable session identifier."""

    if str(relative_parent) in {"", "."}:
        return DEFAULT_SESSION_ID
    token = "__".join(_normalize_name_token(part) for part in relative_parent.parts if part)
    return f"kaist_rtf_{token}"


def _progression_sort_token(file_name: str) -> Tuple[str, str]:
    """Extract a progression token from a file name using datetime or numeric patterns."""

    stem = Path(file_name).stem
    datetime_patterns = [
        r"(20\d{2}[-_]?\d{2}[-_]?\d{2}[-_ ]?\d{2}[-_]?\d{2}[-_]?\d{2})",
        r"(\d{8}[-_]?\d{6})",
    ]
    for pattern in datetime_patterns:
        match = re.search(pattern, stem)
        if match:
            token = re.sub(r"[^0-9]", "", match.group(1))
            return token.zfill(14), "datetime_token"

    integer_tokens = re.findall(r"\d+", stem)
    if integer_tokens:
        token = integer_tokens[-1]
        return token.zfill(12), "numeric_token"

    return stem.lower(), "lexical_order"


def discover_run_files(data_root: str | Path) -> List[RunFileRecord]:
    """Discover candidate run-to-failure files without materializing raw CSV exports."""

    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"KAIST run-to-failure data root not found: {root}")

    discovered: List[RunFileRecord] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        relative_path = path.relative_to(root)
        session_id = _candidate_session_id_from_parent(relative_path.parent)
        order_key, hint = _progression_sort_token(path.name)
        discovered.append(
            RunFileRecord(
                session_id=session_id,
                group_id=session_id,
                file_path=path,
                relative_path=str(relative_path).replace("\\", "/"),
                file_name=path.name,
                extension=path.suffix.lower(),
                progression_order_key=order_key,
                progression_hint=hint,
            )
        )
    return discovered


def _decode_sample(file_path: Path, sample_bytes: int = FORMAT_SAMPLE_BYTES) -> str:
    """Read a small byte sample used for delimiter and header detection."""

    with file_path.open("rb") as handle:
        return handle.read(sample_bytes).decode("utf-8", errors="ignore")


def _detect_delimiter(sample_text: str) -> str:
    """Detect the delimiter from a small sample without scanning the whole file."""

    lines = [line for line in sample_text.splitlines() if line.strip()]
    if not lines:
        return ","

    candidates = [",", "\t", ";", "|"]
    scored = {delimiter: sum(line.count(delimiter) for line in lines[:3]) for delimiter in candidates}
    best_delimiter = max(scored, key=scored.get)
    if scored[best_delimiter] > 0:
        return best_delimiter
    return ","


def _detect_has_header(sample_text: str, delimiter: str) -> bool:
    """Infer whether the first non-empty line is a text header."""

    first_line = next((line for line in sample_text.splitlines() if line.strip()), "")
    if not first_line:
        return False
    first_tokens = [token.strip() for token in first_line.split(delimiter)]
    return any(any(character.isalpha() for character in token) for token in first_tokens)


def detect_csv_format(
    file_path: Path,
    format_cache: Dict[str, CsvFormatSpec] | None = None,
    cache_key: str | None = None,
    debug: bool = False,
    verbose: bool = True,
) -> CsvFormatSpec:
    """Detect a file format from a small sample and reuse it through a cache."""

    resolved_cache_key = cache_key or str(file_path.parent)
    if format_cache is not None and resolved_cache_key in format_cache:
        return format_cache[resolved_cache_key]

    sample_text = _decode_sample(file_path)
    delimiter = _detect_delimiter(sample_text)
    has_header = _detect_has_header(sample_text, delimiter)
    spec = CsvFormatSpec(delimiter=delimiter, has_header=has_header)
    if format_cache is not None:
        format_cache[resolved_cache_key] = spec
    if debug:
        _log(
            f"[kaist-rtf][debug] detected format for {file_path}: delimiter={repr(delimiter)}, "
            f"has_header={has_header}, engine={spec.engine}",
            verbose=verbose,
        )
    return spec


def _fast_read_delimited_file(file_path: Path, format_spec: CsvFormatSpec) -> pd.DataFrame:
    """Read only the required four numeric columns using the fast pandas parser path."""

    read_kwargs = {
        "sep": format_spec.delimiter,
        "engine": format_spec.engine,
        "header": 0 if format_spec.has_header else None,
        "usecols": [0, 1, 2, 3],
        "dtype": np.float32,
        "memory_map": True,
        # Some hourly files encode missing temperature samples as literal "NaN".
        # Keep NA parsing enabled so the C engine can still parse directly to floats.
        "keep_default_na": True,
        "na_values": ["NaN", "nan", "NAN"],
        "skip_blank_lines": True,
        "on_bad_lines": "error",
        "encoding": format_spec.encoding,
    }
    frame = pd.read_csv(file_path, **read_kwargs)
    frame.columns = EXPECTED_SIGNAL_COLUMNS
    return frame


def load_hourly_measurement(
    file_path: str | Path,
    sampling_rate_hz: float,
    format_spec: CsvFormatSpec | None = None,
    debug: bool = False,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Load one hourly file into vibration and thermal frames using the expected four-column layout."""

    source_path = Path(file_path)
    resolved_format = format_spec or detect_csv_format(source_path, debug=debug, verbose=verbose)
    try:
        frame = _fast_read_delimited_file(source_path, resolved_format)
    except Exception as error:
        raise ValueError(f"Failed to parse {source_path} with delimiter {repr(resolved_format.delimiter)}: {error}") from error

    if frame.shape[1] < 4:
        raise ValueError(
            f"{source_path.name} does not contain the expected four columns "
            "(vibration_x, vibration_y, bearing_temperature, ambient_temperature)."
        )
    if frame.empty:
        raise ValueError(f"{source_path.name} did not contain any readable numeric samples.")

    n_samples = int(len(frame))
    timestamps = np.arange(n_samples, dtype=np.float64) / float(sampling_rate_hz)
    vibration_frame = pd.DataFrame(
        {
            TIMESTAMP_COL: timestamps,
            "vibration_x": frame["vibration_x"].to_numpy(dtype=np.float64, copy=False),
            "vibration_y": frame["vibration_y"].to_numpy(dtype=np.float64, copy=False),
        }
    )
    thermal_frame = pd.DataFrame(
        {
            TIMESTAMP_COL: timestamps,
            "bearing_temperature_c": frame["bearing_temperature_c"].to_numpy(dtype=np.float64, copy=False),
            "ambient_temperature_c": frame["ambient_temperature_c"].to_numpy(dtype=np.float64, copy=False),
        }
    )
    measurement_summary = {
        "n_samples": n_samples,
        "duration_sec": float(timestamps[-1]) if n_samples > 1 else 0.0,
        "bearing_temp_mean": float(thermal_frame["bearing_temperature_c"].mean()),
        "ambient_temp_mean": float(thermal_frame["ambient_temperature_c"].mean()),
        "shape_rows": int(frame.shape[0]),
        "shape_cols": int(frame.shape[1]),
        "delimiter": resolved_format.delimiter,
        "has_header": bool(resolved_format.has_header),
    }
    if debug:
        _log(
            f"[kaist-rtf][debug] parsed {source_path.name}: rows={frame.shape[0]}, cols={frame.shape[1]}, "
            f"delimiter={repr(resolved_format.delimiter)}, has_header={resolved_format.has_header}",
            verbose=verbose,
        )
    return vibration_frame, thermal_frame, measurement_summary


def _gap_features(thermal_frame: pd.DataFrame) -> Dict[str, float]:
    """Add a few explicit bearing-to-ambient gap features for progression analysis."""

    gap = (
        thermal_frame["bearing_temperature_c"].to_numpy(dtype=float)
        - thermal_frame["ambient_temperature_c"].to_numpy(dtype=float)
    )
    timestamps = thermal_frame[TIMESTAMP_COL].to_numpy(dtype=float)
    if gap.size == 0:
        return {}

    trend = float(gap[-1] - gap[0]) if gap.size > 1 else 0.0
    slope = 0.0
    if gap.size > 1 and not np.allclose(timestamps, timestamps[0]):
        slope = float(np.polyfit(timestamps - timestamps[0], gap, deg=1)[0])
    return {
        "thermal_bearing_ambient_gap_mean": float(np.mean(gap)),
        "thermal_bearing_ambient_gap_max": float(np.max(gap)),
        "thermal_bearing_ambient_gap_trend": trend,
        "thermal_bearing_ambient_gap_slope": slope,
    }


def build_hourly_feature_row(
    record: RunFileRecord,
    progression_index: int,
    elapsed_hours: float,
    total_files: int,
    sampling_rate_hz: float,
    format_spec: CsvFormatSpec | None = None,
    debug: bool = False,
    verbose: bool = True,
) -> Tuple[Dict[str, object], Dict[str, float]]:
    """Convert one hourly file into a compact feature row."""

    vibration_frame, thermal_frame, measurement_summary = load_hourly_measurement(
        record.file_path,
        sampling_rate_hz=sampling_rate_hz,
        format_spec=format_spec,
        debug=debug,
        verbose=verbose,
    )
    features: Dict[str, object] = {
        "source_file_name": record.file_name,
        "source_relative_path": record.relative_path,
        "progression_index": int(progression_index),
        "elapsed_hours": float(elapsed_hours),
        "relative_progress": float(progression_index / max(total_files - 1, 1)),
        "measurement_duration_sec": float(measurement_summary["duration_sec"]),
        "sampling_rate_hz": float(sampling_rate_hz),
        "n_samples": int(measurement_summary["n_samples"]),
    }
    features.update(extract_vibration_features(vibration_frame, fallback_sampling_rate=sampling_rate_hz))
    features.update(extract_thermal_features(thermal_frame))
    features.update(_gap_features(thermal_frame))

    window_start = float(elapsed_hours * 3600.0)
    row: Dict[str, object] = {
        "session_id": record.session_id,
        "group_id": record.group_id,
        "label": "unknown",
        "multiclass_label": "unknown",
        "window_index": int(progression_index),
        "window_start": window_start,
        "window_end": window_start + float(measurement_summary["duration_sec"]),
        "window_pairing_strategy": "hourly_progression_order",
    }
    row.update(features)
    return row, measurement_summary


def _session_metadata(
    session_id: str,
    file_records: Sequence[RunFileRecord],
    dataset_variant: str,
) -> Dict[str, object]:
    """Build metadata.json content for one compact processed run."""

    return {
        "schema_version": "kaist_rtf_compact_v1",
        "dataset_name": "kaist_run_to_failure",
        "dataset_variant": dataset_variant,
        "session_id": session_id,
        "group_id": session_id,
        "label": "unknown",
        "multiclass_label": "unknown",
        "available_modalities": ["vibration", "thermal"],
        "missing_modalities": ["ae", "acoustic", "current"],
        "progression_type": "run_to_failure_hourly_measurements",
        "synchronized_modalities": True,
        "source_file_count": int(len(file_records)),
        "progression_hint_types": sorted({record.progression_hint for record in file_records}),
        "notes": (
            "This compact export preserves hourly progression order for anomaly-first and degradation-oriented analysis. "
            "No supervised health labels are provided by the adapter."
        ),
    }


def _empty_inventory_frame() -> pd.DataFrame:
    """Return a stable empty file-inventory frame."""

    return pd.DataFrame(
        columns=[
            "session_id",
            "group_id",
            "file_name",
            "relative_path",
            "extension",
            "progression_order_key",
            "progression_hint",
        ]
    )


def _empty_session_manifest_frame() -> pd.DataFrame:
    """Return a stable empty session-manifest frame."""

    return pd.DataFrame(
        columns=[
            "session_id",
            "group_id",
            "dataset_name",
            "dataset_variant",
            "n_hourly_files",
            "available_modalities",
            "missing_modalities",
            "progression_hint_types",
        ]
    )


def _empty_warning_frame() -> pd.DataFrame:
    """Return a stable warning table used for compact exports."""

    return pd.DataFrame(columns=["session_id", "file_name", "relative_path", "warning_type", "warning_detail"])


def build_kaist_rtf_compact_dataset(
    config: ExperimentConfig,
    debug: bool = False,
    max_files: int | None = None,
    progress_every: int = 10,
    verbose: bool = True,
) -> Dict[str, object]:
    """Build compact manifests and a feature dataset for the KAIST run-to-failure dataset."""

    start_time = time.perf_counter()
    data_root = Path(config.paths.data_root)
    processed_root = ensure_directory(Path(config.paths.processed_root) / "kaist_run_to_failure")
    manifests_dir = ensure_directory(processed_root / "manifests")
    sessions_dir = ensure_directory(processed_root / "sessions")
    datasets_dir = ensure_directory(processed_root / "datasets")

    if debug and max_files is None:
        max_files = 3
    progress_interval = max(1, int(progress_every))

    records = discover_run_files(data_root)
    total_discovered = len(records)
    if max_files is not None:
        records = records[: max(0, int(max_files))]

    _log(
        f"[kaist-rtf] discovered {total_discovered} candidate files under {data_root}. "
        f"Processing {len(records)} file(s).",
        verbose=verbose,
    )

    inventory_rows = [
        {
            "session_id": record.session_id,
            "group_id": record.group_id,
            "file_name": record.file_name,
            "relative_path": record.relative_path,
            "extension": record.extension,
            "progression_order_key": record.progression_order_key,
            "progression_hint": record.progression_hint,
        }
        for record in records
    ]
    inventory_frame = _empty_inventory_frame() if not inventory_rows else pd.DataFrame(inventory_rows)
    inventory_frame.to_csv(manifests_dir / "file_inventory.csv", index=False)

    metadata = get_dataset_metadata("kaist_run_to_failure")
    session_rows: List[Dict[str, object]] = []
    feature_rows: List[Dict[str, object]] = []
    warning_rows: List[Dict[str, object]] = []
    processed_files = 0
    format_cache: Dict[str, CsvFormatSpec] = {}
    first_successful_summary: Dict[str, object] | None = None

    for session_id in sorted(inventory_frame["session_id"].astype(str).unique().tolist()):
        ordered_records = sorted(
            [record for record in records if record.session_id == session_id],
            key=lambda item: (item.progression_order_key, item.file_name.lower()),
        )
        session_dir = ensure_directory(sessions_dir / session_id)
        session_manifest_rows: List[Dict[str, object]] = []
        total_files = len(ordered_records)
        session_format_spec = detect_csv_format(
            ordered_records[0].file_path,
            format_cache=format_cache,
            cache_key=session_id,
            debug=debug,
            verbose=verbose,
        ) if ordered_records else None

        for progression_index, record in enumerate(ordered_records):
            elapsed_hours = float(progression_index)
            session_manifest_rows.append(
                {
                    "session_id": record.session_id,
                    "group_id": record.group_id,
                    "progression_index": int(progression_index),
                    "elapsed_hours": elapsed_hours,
                    "file_name": record.file_name,
                    "relative_path": record.relative_path,
                    "progression_hint": record.progression_hint,
                }
            )
            file_start = time.perf_counter()
            try:
                row, measurement_summary = build_hourly_feature_row(
                    record=record,
                    progression_index=progression_index,
                    elapsed_hours=elapsed_hours,
                    total_files=total_files,
                    sampling_rate_hz=config.run_to_failure.sampling_rate_hz,
                    format_spec=session_format_spec,
                    debug=debug and processed_files == 0,
                    verbose=verbose,
                )
                feature_rows.append(row)
                processed_files += 1
                if first_successful_summary is None:
                    first_successful_summary = {
                        "file_name": record.file_name,
                        "relative_path": record.relative_path,
                        "delimiter": measurement_summary["delimiter"],
                        "has_header": measurement_summary["has_header"],
                        "rows": measurement_summary["shape_rows"],
                        "cols": measurement_summary["shape_cols"],
                        "n_samples": measurement_summary["n_samples"],
                    }
                    if debug:
                        _log(
                            f"[kaist-rtf][debug] first-file format summary: {first_successful_summary}",
                            verbose=verbose,
                        )
            except Exception as error:  # pragma: no cover - defensive path for malformed public files
                warning_detail = str(error)
                warning_rows.append(
                    {
                        "session_id": record.session_id,
                        "file_name": record.file_name,
                        "relative_path": record.relative_path,
                        "warning_type": "feature_build_failed",
                        "warning_detail": warning_detail,
                    }
                )
                _log(
                    f"[kaist-rtf][error] failed to parse {record.file_path}: {warning_detail}",
                    verbose=True,
                )
                continue

            if debug or processed_files == 1 or processed_files % progress_interval == 0 or processed_files == len(records):
                file_elapsed = time.perf_counter() - file_start
                total_elapsed = time.perf_counter() - start_time
                _log(
                    f"[kaist-rtf] processed {processed_files}/{len(records)} file(s) "
                    f"in {file_elapsed:.2f}s; total {total_elapsed:.2f}s; current={record.relative_path}",
                    verbose=verbose,
                )

        pd.DataFrame(session_manifest_rows).to_csv(session_dir / "file_manifest.csv", index=False)
        save_json(
            session_dir / "metadata.json",
            _session_metadata(
                session_id=session_id,
                file_records=ordered_records,
                dataset_variant=config.dataset.variant,
            ),
        )
        session_rows.append(
            {
                "session_id": session_id,
                "group_id": session_id,
                "dataset_name": metadata.key,
                "dataset_variant": config.dataset.variant,
                "n_hourly_files": int(total_files),
                "available_modalities": "vibration,thermal",
                "missing_modalities": "ae,acoustic,current",
                "progression_hint_types": ",".join(sorted({record.progression_hint for record in ordered_records})),
            }
        )

    session_manifest_frame = _empty_session_manifest_frame() if not session_rows else pd.DataFrame(session_rows)
    session_manifest_frame.to_csv(manifests_dir / "session_manifest.csv", index=False)
    warnings_frame = _empty_warning_frame() if not warning_rows else pd.DataFrame(warning_rows)
    warnings_frame.to_csv(manifests_dir / "normalization_warnings.csv", index=False)

    if feature_rows:
        feature_frame = pd.DataFrame(feature_rows)
        feature_frame = standardize_feature_dataset(
            frame=feature_frame,
            metadata=metadata,
            dataset_variant=config.dataset.variant,
            modality_flags={"vibration": True, "thermal": True, "current": False, "ae": False, "acoustic": False},
        )
        feature_frame = feature_frame.sort_values(["session_id", "progression_index"]).reset_index(drop=True)
    else:
        feature_frame = pd.DataFrame()

    feature_dataset_path = datasets_dir / "kaist_rtf_feature_dataset.csv"
    feature_frame.to_csv(feature_dataset_path, index=False)

    elapsed_seconds = time.perf_counter() - start_time
    summary = {
        "dataset_name": metadata.key,
        "dataset_variant": config.dataset.variant,
        "data_root": str(data_root),
        "processed_root": str(processed_root),
        "n_sessions": int(session_manifest_frame["session_id"].nunique()) if not session_manifest_frame.empty else 0,
        "n_source_files": int(total_discovered),
        "n_processed_files": int(processed_files),
        "n_feature_rows": int(len(feature_frame)),
        "feature_dataset_path": str(feature_dataset_path),
        "session_manifest_path": str(manifests_dir / "session_manifest.csv"),
        "warning_count": int(len(warnings_frame)),
        "elapsed_seconds": float(elapsed_seconds),
        "debug": bool(debug),
        "first_file_format_summary": first_successful_summary,
        "status": "ok" if total_discovered > 0 else "no_data_available",
        "note": (
            "No hourly files were found under the configured extracted root."
            if total_discovered == 0
            else "Compact run-to-failure manifests and feature dataset were generated successfully."
        ),
    }
    save_json(processed_root / "adapter_summary.json", summary)
    (processed_root / "adapter_summary.md").write_text(
        "\n".join(
            [
                "# KAIST Run-to-Failure Compact Adapter Summary",
                "",
                f"- Data root: `{data_root}`",
                f"- Source files discovered: {summary['n_source_files']}",
                f"- Files processed: {summary['n_processed_files']}",
                f"- Sessions discovered: {summary['n_sessions']}",
                f"- Feature rows created: {summary['n_feature_rows']}",
                f"- Warning count: {summary['warning_count']}",
                f"- Elapsed seconds: {summary['elapsed_seconds']:.2f}",
                f"- Status: `{summary['status']}`",
                f"- Note: {summary['note']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _log(
        "[kaist-rtf] summary: "
        f"files_processed={summary['n_processed_files']}, sessions_created={summary['n_sessions']}, "
        f"feature_rows={summary['n_feature_rows']}, elapsed_seconds={summary['elapsed_seconds']:.2f}",
        verbose=verbose,
    )
    return summary
