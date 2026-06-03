from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pdm_pipeline.config import ExperimentConfig
from pdm_pipeline.nasa_ims_adapter import build_nasa_ims_compact_dataset
from pdm_pipeline.nasa_ims_experiment import run_nasa_ims_experiment


def _write_snapshot(path: Path, n_columns: int, offset: float) -> None:
    rows = 64
    data = []
    for column_index in range(n_columns):
        base = np.sin(np.linspace(0.0, 4.0, rows) + column_index * 0.3) + offset + column_index * 0.05
        data.append(base)
    frame = pd.DataFrame(np.column_stack(data))
    frame.to_csv(path, index=False, header=False, sep="\t")


def test_nasa_ims_compact_adapter_builds_bearing_feature_dataset(tmp_path: Path) -> None:
    extracted_root = tmp_path / "extracted"
    (extracted_root / "1st_test").mkdir(parents=True)
    (extracted_root / "3rd_test" / "4th_test" / "txt").mkdir(parents=True)

    _write_snapshot(extracted_root / "1st_test" / "2003.10.22.12.06.24", 8, 0.0)
    _write_snapshot(extracted_root / "1st_test" / "2003.10.22.12.11.24", 8, 0.1)
    _write_snapshot(extracted_root / "3rd_test" / "4th_test" / "txt" / "2004.03.04.09.27.46", 4, 0.2)
    _write_snapshot(extracted_root / "3rd_test" / "4th_test" / "txt" / "2004.03.04.09.37.46", 4, 0.4)

    config = ExperimentConfig()
    config.dataset.name = "nasa_ims"
    config.dataset.variant = "compact_bearing_progression"
    config.paths.data_root = str(extracted_root)
    config.paths.processed_root = str(tmp_path / "processed")

    summary = build_nasa_ims_compact_dataset(config)

    assert summary["status"] == "ok"
    assert summary["n_runs"] == 2
    assert summary["n_bearing_sessions"] == 8
    assert summary["n_feature_rows"] == 16

    feature_frame = pd.read_csv(summary["feature_dataset_path"])
    session_manifest = pd.read_csv(Path(summary["session_manifest_path"]))
    warnings_frame = pd.read_csv(tmp_path / "processed" / "nasa_ims" / "manifests" / "layout_warnings.csv")

    assert feature_frame["has_vibration"].all()
    assert (feature_frame["reference_region_role"].astype(str) == "unknown").all()
    assert "3rd_test__4th_test__txt" in session_manifest["run_key"].astype(str).tolist()
    assert not warnings_frame.empty
    assert "[1, 2]" in set(session_manifest["channel_indices"].astype(str))


def test_nasa_ims_experiment_runs_from_processed_feature_dataset(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    dataset_dir = processed_root / "nasa_ims" / "datasets"
    dataset_dir.mkdir(parents=True)
    feature_dataset_path = dataset_dir / "nasa_ims_bearing_feature_dataset.csv"

    rows = []
    for session_id, group_id, bearing_id, documented_mode, scale in [
        ("nasa_ims_1st_test_bearing_1", "nasa_ims_1st_test", 1, None, 0.0),
        ("nasa_ims_2nd_test_bearing_1", "nasa_ims_2nd_test", 1, "outer_race_failure", 0.5),
    ]:
        for progression_index in range(12):
            rows.append(
                {
                    "dataset_name": "nasa_ims",
                    "dataset_variant": "compact_bearing_progression",
                    "dataset_display_name": "NASA IMS",
                    "session_id": session_id,
                    "group_id": group_id,
                    "label": "unknown",
                    "multiclass_label": "unknown",
                    "reference_region_role": "unknown",
                    "bearing_id": bearing_id,
                    "run_key": group_id.replace("nasa_ims_", ""),
                    "source_run_path": group_id.replace("nasa_ims_", ""),
                    "source_file_name": f"file_{progression_index:03d}",
                    "source_relative_path": f"{group_id}/file_{progression_index:03d}",
                    "snapshot_timestamp": f"2004-01-01T{progression_index:02d}:00:00",
                    "progression_index": progression_index,
                    "elapsed_minutes": float(progression_index * 10.0),
                    "elapsed_hours": float(progression_index / 6.0),
                    "relative_progress": float(progression_index / 11.0),
                    "nominal_snapshot_duration_sec": 1.0,
                    "window_index": progression_index,
                    "window_start": float(progression_index * 600.0),
                    "window_end": float(progression_index * 600.0 + 1.0),
                    "window_pairing_strategy": "file_timestamp_order",
                    "has_vibration": True,
                    "has_thermal": False,
                    "has_current": False,
                    "has_ae": False,
                    "has_acoustic": False,
                    "axis_count": 1,
                    "channel_indices": "[1]",
                    "documented_failure_mode": documented_mode,
                    "documented_failed_bearing": documented_mode is not None,
                    "documented_failure_metadata_source": "ims_readme_pdf",
                    "documented_failure_metadata_confidence": "documented",
                    "documented_failure_notes": "",
                    "vibration_feature_a": scale + (0.1 if progression_index < 4 else 1.0 + progression_index * 0.1),
                    "vibration_feature_b": scale + progression_index * 0.05,
                    "vibration_feature_c": scale + (0.2 if progression_index < 4 else 0.8 + progression_index * 0.08),
                }
            )

    pd.DataFrame(rows).to_csv(feature_dataset_path, index=False)

    config = ExperimentConfig()
    config.dataset.name = "nasa_ims"
    config.dataset.variant = "compact_bearing_progression"
    config.paths.processed_root = str(processed_root)
    config.paths.feature_dataset_path = str(feature_dataset_path)
    config.paths.results_root = str(tmp_path / "results")
    config.run_to_failure.healthy_reference_max_hours = 0.8
    config.run_to_failure.minimum_reference_files = 3
    config.run_to_failure.rolling_window_files = 3
    config.model.anomaly.minimum_healthy_windows = 4

    summary = run_nasa_ims_experiment(config)

    assert summary["status"] == "ok"
    assert summary["n_sessions"] == 2
    assert summary["n_feature_rows"] == 24
    trend_frame = pd.read_csv(Path(summary["results_dir"]) / "anomaly_trend.csv")
    crossing_frame = pd.read_csv(Path(summary["results_dir"]) / "threshold_crossing_summary.csv")
    assert set(crossing_frame["model_name"]) == {"isolation_forest", "one_class_svm"}
    assert {"reference_region_role", "anomaly_score_raw", "sustained_warning"}.issubset(trend_frame.columns)
