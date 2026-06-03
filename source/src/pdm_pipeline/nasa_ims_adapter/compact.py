"""Compact adapter for the NASA IMS bearing dataset."""

from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from ..config import ExperimentConfig
from ..datasets import get_dataset_metadata, standardize_feature_dataset
from ..features.vibration import extract_vibration_features
from ..io import TIMESTAMP_COL
from ..utils import ensure_directory, save_json


TIMESTAMP_NAME_PATTERN = re.compile(r"^\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}$")
SAMPLING_RATE_HZ = 20_000.0
NOMINAL_SNAPSHOT_DURATION_SEC = 1.0
EXPECTED_ROWS_PER_SNAPSHOT = 20_480
DEFAULT_VARIANT = "compact_bearing_progression"
NESTED_LAYOUT_WARNING = (
    "Local extracted layout contains a nested run path under 3rd_test/4th_test/txt; "
    "this run is preserved by observed path identity rather than assumed official set numbering."
)
DOCUMENTED_FAILURES: Dict[str, Dict[str, object]] = {
    "1st_test": {
        "notes": "Documented end-of-run defects: bearing 3 inner race defect, bearing 4 roller element defect.",
        "confidence": "documented",
        "bearings": {
            3: "inner_race_defect",
            4: "roller_element_defect",
        },
    },
    "2nd_test": {
        "notes": "Documented end-of-run defect: bearing 1 outer race failure.",
        "confidence": "documented",
        "bearings": {
            1: "outer_race_failure",
        },
    },
    "3rd_test__4th_test__txt": {
        "notes": (
            "The local extracted layout uses a nested path under 3rd_test/4th_test/txt. "
            "Official Set No. 3 failure metadata is therefore treated as uncertain for this local run."
        ),
        "confidence": "uncertain_local_layout",
        "bearings": {},
    },
}


@dataclass(frozen=True)
class SnapshotRecord:
    """One timestamp-named NASA IMS vibration snapshot file."""

    run_key: str
    group_id: str
    source_run_path: str
    layout_type: str
    layout_warning: str | None
    file_path: Path
    source_relative_path: str
    source_file_name: str
    snapshot_timestamp: datetime


def _log(message: str, verbose: bool = True) -> None:
    """Print progress messages with flushing for long compact exports."""

    if verbose:
        print(message, flush=True)


def _normalize_path_part(value: str) -> str:
    """Normalize one folder name into a stable run-key token."""

    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower()).strip("_")
    return normalized or "run"


def _run_key_from_relative_path(relative_parent: Path) -> str:
    """Convert a discovered file parent path into a stable run key."""

    return "__".join(_normalize_path_part(part) for part in relative_parent.parts if part)


def _group_id_from_run_key(run_key: str) -> str:
    """Return the canonical run-level group identifier."""

    return f"nasa_ims_{run_key}"


def _parse_snapshot_timestamp(file_name: str) -> datetime:
    """Parse the NASA IMS timestamp file naming convention."""

    if not TIMESTAMP_NAME_PATTERN.fullmatch(str(file_name)):
        raise ValueError(f"Unsupported NASA IMS file name: {file_name}")
    return datetime.strptime(str(file_name), "%Y.%m.%d.%H.%M.%S")


def discover_nasa_ims_snapshot_files(data_root: str | Path) -> List[SnapshotRecord]:
    """Discover all timestamp-named vibration snapshot files grouped by their leaf run folder."""

    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"NASA IMS data root not found: {root}")

    records: List[SnapshotRecord] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if not TIMESTAMP_NAME_PATTERN.fullmatch(path.name):
            continue
        relative_path = path.relative_to(root)
        run_path = relative_path.parent
        run_key = _run_key_from_relative_path(run_path)
        layout_type = "flat_run_folder" if len(run_path.parts) <= 1 else "nested_run_folder"
        layout_warning = NESTED_LAYOUT_WARNING if layout_type == "nested_run_folder" else None
        records.append(
            SnapshotRecord(
                run_key=run_key,
                group_id=_group_id_from_run_key(run_key),
                source_run_path=str(run_path).replace("\\", "/"),
                layout_type=layout_type,
                layout_warning=layout_warning,
                file_path=path,
                source_relative_path=str(relative_path).replace("\\", "/"),
                source_file_name=path.name,
                snapshot_timestamp=_parse_snapshot_timestamp(path.name),
            )
        )
    return records


def _first_non_empty_line(file_path: Path) -> str:
    """Read the first non-empty line of a snapshot file."""

    with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip():
                return line.strip()
    return ""


def _count_file_rows(file_path: Path) -> int:
    """Count rows in one snapshot file for run-level validation."""

    with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for line in handle if line.strip())


def _channel_count_from_file(file_path: Path) -> int:
    """Infer the run channel count from the first non-empty line."""

    first_line = _first_non_empty_line(file_path)
    if not first_line:
        raise ValueError(f"NASA IMS snapshot file is empty: {file_path}")
    return len(first_line.split())


def _load_snapshot_frame(file_path: Path, expected_channel_count: int) -> pd.DataFrame:
    """Read one NASA IMS vibration snapshot using a fast parser path."""

    try:
        frame = pd.read_csv(
            file_path,
            sep="\t",
            header=None,
            engine="c",
            dtype=np.float32,
            memory_map=True,
            na_filter=False,
            skip_blank_lines=True,
        )
    except (ParserError, EmptyDataError):
        frame = pd.read_csv(
            file_path,
            sep=r"\s+",
            header=None,
            engine="python",
            dtype=np.float32,
            skip_blank_lines=True,
        )

    if frame.empty:
        raise ValueError(f"NASA IMS snapshot file is empty after parsing: {file_path}")
    if frame.shape[1] != int(expected_channel_count):
        raise ValueError(
            f"{file_path.name} expected {expected_channel_count} columns but parsed {frame.shape[1]} columns."
        )
    return frame


def _bearing_channel_map(run_key: str, channel_count: int) -> Dict[int, List[int]]:
    """Return the run-specific channel-to-bearing mapping using 1-based channel indices."""

    if run_key == "1st_test":
        if int(channel_count) != 8:
            raise ValueError(f"1st_test must contain 8 channels, but {channel_count} were detected.")
        return {
            1: [1, 2],
            2: [3, 4],
            3: [5, 6],
            4: [7, 8],
        }

    if int(channel_count) == 4:
        return {
            1: [1],
            2: [2],
            3: [3],
            4: [4],
        }

    raise ValueError(f"Unsupported NASA IMS run/channel configuration: run_key={run_key}, channel_count={channel_count}")


def _documented_failure_entry(run_key: str) -> Dict[str, object]:
    """Return documented end-of-run metadata for one run."""

    default_entry = {
        "notes": "No local documented end-of-run failure metadata was available for this run.",
        "confidence": "uncertain_local_layout",
        "bearings": {},
    }
    return DOCUMENTED_FAILURES.get(run_key, default_entry)


def _documented_failure_metadata_for_bearing(run_key: str, bearing_id: int) -> Dict[str, object]:
    """Return bearing-level documented failure metadata while keeping uncertainty explicit."""

    run_entry = _documented_failure_entry(run_key)
    failure_map = run_entry.get("bearings", {})
    confidence = str(run_entry.get("confidence", "uncertain_local_layout"))
    if confidence == "uncertain_local_layout":
        failed_bearing = None
    else:
        failed_bearing = bool(int(bearing_id) in failure_map)

    return {
        "documented_failure_mode": failure_map.get(int(bearing_id)),
        "documented_failed_bearing": failed_bearing,
        "documented_failure_metadata_source": "ims_readme_pdf",
        "documented_failure_metadata_confidence": confidence,
        "documented_failure_notes": str(run_entry.get("notes", "")),
    }


def _session_id(run_key: str, bearing_id: int) -> str:
    """Return the canonical NASA IMS bearing-session identifier."""

    return f"nasa_ims_{run_key}_bearing_{int(bearing_id)}"


def _nominal_interval_minutes(timestamps: Sequence[datetime]) -> Dict[str, float]:
    """Summarize observed timestamp gaps for one run."""

    if len(timestamps) < 2:
        return {
            "nominal_interval_minutes": 0.0,
            "min_interval_minutes": 0.0,
            "max_interval_minutes": 0.0,
        }
    deltas = [round((current - previous).total_seconds() / 60.0, 3) for previous, current in zip(timestamps, timestamps[1:])]
    delta_counter = Counter(deltas)
    nominal_interval = float(delta_counter.most_common(1)[0][0])
    return {
        "nominal_interval_minutes": nominal_interval,
        "min_interval_minutes": float(min(deltas)),
        "max_interval_minutes": float(max(deltas)),
    }


def _sample_timestamps(n_rows: int, cache: Dict[int, np.ndarray]) -> np.ndarray:
    """Return cached per-snapshot sample timestamps for vibration features."""

    if n_rows not in cache:
        cache[n_rows] = np.arange(int(n_rows), dtype=np.float64) / SAMPLING_RATE_HZ
    return cache[n_rows]


def _delta_minutes(current: datetime, previous: datetime | None) -> float:
    """Return the gap from the previous snapshot in minutes."""

    if previous is None:
        return 0.0
    return float((current - previous).total_seconds() / 60.0)


def _relative_progress(progression_index: int, total_snapshots: int) -> float:
    """Return normalized run progress within one bearing session."""

    if total_snapshots <= 1:
        return 0.0
    return float(progression_index / float(total_snapshots - 1))


def _bearing_feature_row(
    record: SnapshotRecord,
    bearing_id: int,
    channel_indices: Sequence[int],
    snapshot_frame: pd.DataFrame,
    progression_index: int,
    total_snapshots: int,
    run_start_timestamp: datetime,
    nominal_interval_minutes: float,
    delta_minutes_from_previous: float,
    timestamp_cache: Dict[int, np.ndarray],
) -> Dict[str, object]:
    """Build one compact feature row for one bearing at one snapshot timestamp."""

    raw_array = snapshot_frame.to_numpy(dtype=np.float64, copy=False)
    sample_timestamps = _sample_timestamps(snapshot_frame.shape[0], timestamp_cache)

    vibration_frame = pd.DataFrame({TIMESTAMP_COL: sample_timestamps})
    for axis_index, source_channel in enumerate(channel_indices, start=1):
        vibration_frame[f"axis_{axis_index}"] = raw_array[:, int(source_channel) - 1]

    elapsed_minutes = float((record.snapshot_timestamp - run_start_timestamp).total_seconds() / 60.0)
    failure_metadata = _documented_failure_metadata_for_bearing(record.run_key, bearing_id)
    features = extract_vibration_features(vibration_frame, fallback_sampling_rate=SAMPLING_RATE_HZ)

    row: Dict[str, object] = {
        "session_id": _session_id(record.run_key, bearing_id),
        "group_id": record.group_id,
        "label": "unknown",
        "multiclass_label": "unknown",
        "reference_region_role": "unknown",
        "bearing_id": int(bearing_id),
        "run_key": record.run_key,
        "source_run_path": record.source_run_path,
        "source_file_name": record.source_file_name,
        "source_relative_path": record.source_relative_path,
        "snapshot_timestamp": record.snapshot_timestamp.isoformat(),
        "progression_index": int(progression_index),
        "elapsed_minutes": elapsed_minutes,
        "elapsed_hours": float(elapsed_minutes / 60.0),
        "relative_progress": _relative_progress(progression_index, total_snapshots),
        "nominal_snapshot_duration_sec": float(NOMINAL_SNAPSHOT_DURATION_SEC),
        "window_index": int(progression_index),
        "window_start": float(elapsed_minutes * 60.0),
        "window_end": float(elapsed_minutes * 60.0 + NOMINAL_SNAPSHOT_DURATION_SEC),
        "window_pairing_strategy": "file_timestamp_order",
        "axis_count": int(len(channel_indices)),
        "channel_indices": json.dumps([int(value) for value in channel_indices]),
        "delta_minutes_from_previous": float(delta_minutes_from_previous),
        "is_nominal_interval": bool(
            nominal_interval_minutes > 0 and np.isclose(delta_minutes_from_previous, nominal_interval_minutes, atol=0.1)
        ),
        "progression_gap_flag": bool(
            nominal_interval_minutes > 0 and delta_minutes_from_previous > float(1.5 * nominal_interval_minutes)
        ),
        "layout_type": record.layout_type,
        "layout_warning": record.layout_warning or "",
    }
    row.update(failure_metadata)
    row.update(features)
    return row


def build_nasa_ims_compact_dataset(
    config: ExperimentConfig,
    debug: bool = False,
    max_files: int | None = None,
    progress_every: int = 250,
    verbose: bool = True,
) -> Dict[str, object]:
    """Build compact manifests and a bearing-level feature dataset for NASA IMS."""

    start_time = time.perf_counter()
    data_root = Path(config.paths.data_root)
    processed_root = ensure_directory(Path(config.paths.processed_root) / "nasa_ims")
    manifests_dir = ensure_directory(processed_root / "manifests")
    datasets_dir = ensure_directory(processed_root / "datasets")
    metadata = get_dataset_metadata("nasa_ims")

    if debug and max_files is None:
        max_files = 50
    progress_interval = max(1, int(progress_every))

    discovered = discover_nasa_ims_snapshot_files(data_root)
    if max_files is not None:
        discovered = discovered[: max(0, int(max_files))]

    _log(
        f"[nasa-ims] discovered {len(discovered)} timestamp snapshot files under {data_root}.",
        verbose=verbose,
    )

    by_run: Dict[str, List[SnapshotRecord]] = defaultdict(list)
    for record in discovered:
        by_run[record.run_key].append(record)

    run_manifest_rows: List[Dict[str, object]] = []
    session_manifest_rows: List[Dict[str, object]] = []
    warning_rows: List[Dict[str, object]] = []
    feature_rows: List[Dict[str, object]] = []
    timestamp_cache: Dict[int, np.ndarray] = {}
    processed_files = 0

    for run_key in sorted(by_run):
        records = sorted(by_run[run_key], key=lambda item: item.snapshot_timestamp)
        if not records:
            continue

        run_entry = _documented_failure_entry(run_key)
        channel_count = _channel_count_from_file(records[0].file_path)
        rows_per_snapshot = _count_file_rows(records[0].file_path)
        bearing_map = _bearing_channel_map(run_key, channel_count)
        timestamps = [record.snapshot_timestamp for record in records]
        interval_summary = _nominal_interval_minutes(timestamps)
        layout_warning = records[0].layout_warning

        run_manifest_rows.append(
            {
                "dataset_name": metadata.key,
                "dataset_variant": config.dataset.variant,
                "run_key": run_key,
                "group_id": records[0].group_id,
                "source_run_path": records[0].source_run_path,
                "layout_type": records[0].layout_type,
                "layout_warning": layout_warning or "",
                "n_snapshot_files": int(len(records)),
                "start_timestamp": timestamps[0].isoformat(),
                "end_timestamp": timestamps[-1].isoformat(),
                "nominal_interval_minutes": float(interval_summary["nominal_interval_minutes"]),
                "min_interval_minutes": float(interval_summary["min_interval_minutes"]),
                "max_interval_minutes": float(interval_summary["max_interval_minutes"]),
                "sampling_rate_hz": float(SAMPLING_RATE_HZ),
                "nominal_snapshot_duration_sec": float(NOMINAL_SNAPSHOT_DURATION_SEC),
                "rows_per_snapshot": int(rows_per_snapshot),
                "channel_count": int(channel_count),
                "bearing_count": int(len(bearing_map)),
                "documented_failure_notes": str(run_entry.get("notes", "")),
                "documented_failure_metadata_confidence": str(run_entry.get("confidence", "uncertain_local_layout")),
            }
        )

        if layout_warning:
            warning_rows.append(
                {
                    "run_key": run_key,
                    "group_id": records[0].group_id,
                    "source_run_path": records[0].source_run_path,
                    "warning_type": "nested_packaging_layout",
                    "warning_detail": layout_warning,
                }
            )
        if int(rows_per_snapshot) != EXPECTED_ROWS_PER_SNAPSHOT:
            warning_rows.append(
                {
                    "run_key": run_key,
                    "group_id": records[0].group_id,
                    "source_run_path": records[0].source_run_path,
                    "warning_type": "unexpected_snapshot_row_count",
                    "warning_detail": (
                        f"Expected {EXPECTED_ROWS_PER_SNAPSHOT} rows per snapshot from documentation, "
                        f"but observed {rows_per_snapshot} rows in {records[0].source_file_name}."
                    ),
                }
            )

        run_start_timestamp = timestamps[0]
        previous_timestamp: datetime | None = None
        for progression_index, record in enumerate(records):
            try:
                snapshot_frame = _load_snapshot_frame(record.file_path, expected_channel_count=channel_count)
            except Exception as error:  # pragma: no cover - defensive path for malformed files
                warning_rows.append(
                    {
                        "run_key": run_key,
                        "group_id": record.group_id,
                        "source_run_path": record.source_run_path,
                        "warning_type": "snapshot_parse_failed",
                        "warning_detail": f"{record.source_relative_path}: {error}",
                    }
                )
                _log(f"[nasa-ims][error] failed to parse {record.file_path}: {error}", verbose=True)
                previous_timestamp = record.snapshot_timestamp
                continue

            if snapshot_frame.shape[0] != int(rows_per_snapshot):
                warning_rows.append(
                    {
                        "run_key": run_key,
                        "group_id": record.group_id,
                        "source_run_path": record.source_run_path,
                        "warning_type": "inconsistent_snapshot_row_count",
                        "warning_detail": (
                            f"{record.source_relative_path} has {snapshot_frame.shape[0]} rows, "
                            f"but the run reference count is {rows_per_snapshot}."
                        ),
                    }
                )

            delta_minutes_from_previous = _delta_minutes(record.snapshot_timestamp, previous_timestamp)
            for bearing_id, channel_indices in bearing_map.items():
                feature_rows.append(
                    _bearing_feature_row(
                        record=record,
                        bearing_id=bearing_id,
                        channel_indices=channel_indices,
                        snapshot_frame=snapshot_frame,
                        progression_index=progression_index,
                        total_snapshots=len(records),
                        run_start_timestamp=run_start_timestamp,
                        nominal_interval_minutes=float(interval_summary["nominal_interval_minutes"]),
                        delta_minutes_from_previous=delta_minutes_from_previous,
                        timestamp_cache=timestamp_cache,
                    )
                )

            processed_files += 1
            if debug or processed_files == 1 or processed_files % progress_interval == 0 or processed_files == len(discovered):
                elapsed = time.perf_counter() - start_time
                _log(
                    f"[nasa-ims] processed {processed_files}/{len(discovered)} snapshot files "
                    f"in {elapsed:.2f}s; current={record.source_relative_path}",
                    verbose=verbose,
                )
            previous_timestamp = record.snapshot_timestamp

        for bearing_id, channel_indices in bearing_map.items():
            failure_metadata = _documented_failure_metadata_for_bearing(run_key, bearing_id)
            session_manifest_rows.append(
                {
                    "dataset_name": metadata.key,
                    "dataset_variant": config.dataset.variant,
                    "run_key": run_key,
                    "group_id": records[0].group_id,
                    "session_id": _session_id(run_key, bearing_id),
                    "bearing_id": int(bearing_id),
                    "source_run_path": records[0].source_run_path,
                    "channel_indices": json.dumps([int(value) for value in channel_indices]),
                    "axis_count": int(len(channel_indices)),
                    "n_snapshots": int(len(records)),
                    "start_timestamp": timestamps[0].isoformat(),
                    "end_timestamp": timestamps[-1].isoformat(),
                    "sampling_rate_hz": float(SAMPLING_RATE_HZ),
                    "nominal_snapshot_duration_sec": float(NOMINAL_SNAPSHOT_DURATION_SEC),
                    "label": "unknown",
                    "multiclass_label": "unknown",
                    "reference_region_role": "unknown",
                    "documented_failure_mode": failure_metadata["documented_failure_mode"],
                    "documented_failed_bearing": failure_metadata["documented_failed_bearing"],
                    "documented_failure_metadata_confidence": failure_metadata["documented_failure_metadata_confidence"],
                    "layout_warning": layout_warning or "",
                }
            )

    run_manifest_frame = pd.DataFrame(run_manifest_rows)
    session_manifest_frame = pd.DataFrame(session_manifest_rows)
    warning_frame = pd.DataFrame(
        warning_rows,
        columns=["run_key", "group_id", "source_run_path", "warning_type", "warning_detail"],
    )

    run_manifest_frame.to_csv(manifests_dir / "run_manifest.csv", index=False)
    session_manifest_frame.to_csv(manifests_dir / "bearing_session_manifest.csv", index=False)
    warning_frame.to_csv(manifests_dir / "layout_warnings.csv", index=False)
    save_json(
        manifests_dir / "documented_failure_map.json",
        {
            "source": "ims_readme_pdf",
            "runs": {
                run_key: (
                    {
                        **{f"bearing_{bearing_id}": failure for bearing_id, failure in entry.get("bearings", {}).items()},
                    }
                    if entry.get("bearings")
                    else {"status": entry.get("confidence", "uncertain_local_layout")}
                )
                for run_key, entry in DOCUMENTED_FAILURES.items()
            },
        },
    )

    if feature_rows:
        feature_frame = pd.DataFrame(feature_rows)
        feature_frame = standardize_feature_dataset(
            frame=feature_frame,
            metadata=metadata,
            dataset_variant=config.dataset.variant or DEFAULT_VARIANT,
            group_column="group_id",
            label_column="label",
            multiclass_label_column="multiclass_label",
            modality_flags={"vibration": True, "thermal": False, "current": False, "ae": False, "acoustic": False},
        )
        feature_frame = feature_frame.sort_values(["group_id", "bearing_id", "progression_index"]).reset_index(drop=True)
    else:
        feature_frame = pd.DataFrame()

    feature_dataset_path = datasets_dir / "nasa_ims_bearing_feature_dataset.csv"
    feature_frame.to_csv(feature_dataset_path, index=False)

    elapsed_seconds = time.perf_counter() - start_time
    summary = {
        "dataset_name": metadata.key,
        "dataset_variant": config.dataset.variant or DEFAULT_VARIANT,
        "data_root": str(data_root),
        "processed_root": str(processed_root),
        "n_runs": int(run_manifest_frame["run_key"].nunique()) if not run_manifest_frame.empty else 0,
        "n_bearing_sessions": int(session_manifest_frame["session_id"].nunique()) if not session_manifest_frame.empty else 0,
        "n_source_files": int(len(discovered)),
        "n_processed_files": int(processed_files),
        "n_feature_rows": int(len(feature_frame)),
        "warning_count": int(len(warning_frame)),
        "feature_dataset_path": str(feature_dataset_path),
        "run_manifest_path": str(manifests_dir / "run_manifest.csv"),
        "session_manifest_path": str(manifests_dir / "bearing_session_manifest.csv"),
        "elapsed_seconds": float(elapsed_seconds),
        "status": "ok" if discovered else "no_data_available",
        "note": (
            "No NASA IMS snapshot files were discovered under the configured extracted root."
            if not discovered
            else "Compact NASA IMS manifests and bearing-level feature dataset were generated successfully."
        ),
    }
    save_json(processed_root / "adapter_summary.json", summary)
    (processed_root / "adapter_summary.md").write_text(
        "\n".join(
            [
                "# NASA IMS Compact Adapter Summary",
                "",
                f"- Data root: `{data_root}`",
                f"- Runs discovered: {summary['n_runs']}",
                f"- Bearing sessions discovered: {summary['n_bearing_sessions']}",
                f"- Snapshot files discovered: {summary['n_source_files']}",
                f"- Snapshot files processed: {summary['n_processed_files']}",
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
    return summary
