from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pdm_pipeline.config import ExperimentConfig
from pdm_pipeline.kaist_rtf_adapter import build_kaist_rtf_compact_dataset
from pdm_pipeline.kaist_rtf_experiment import run_kaist_rtf_experiment


def _write_hourly_file(path: Path, offset: float) -> None:
    rows = 32
    frame = pd.DataFrame(
        {
            0: np.sin(np.linspace(0.0, 4.0, rows)) + offset,
            1: np.cos(np.linspace(0.0, 4.0, rows)) + offset / 2.0,
            2: np.linspace(30.0 + offset, 31.0 + offset, rows),
            3: np.linspace(24.0, 24.3, rows),
        }
    )
    frame.to_csv(path, index=False, header=False)


def test_kaist_rtf_compact_adapter_builds_features(tmp_path: Path) -> None:
    extracted_root = tmp_path / "extracted"
    extracted_root.mkdir()
    _write_hourly_file(extracted_root / "hour_001.csv", 0.0)
    _write_hourly_file(extracted_root / "hour_002.csv", 0.2)
    _write_hourly_file(extracted_root / "hour_003.csv", 0.4)

    config = ExperimentConfig()
    config.dataset.name = "kaist_run_to_failure"
    config.dataset.variant = "compact_hourly_progression"
    config.paths.data_root = str(extracted_root)
    config.paths.processed_root = str(tmp_path / "processed")

    summary = build_kaist_rtf_compact_dataset(config)

    assert summary["status"] == "ok"
    assert summary["n_source_files"] == 3
    assert summary["n_feature_rows"] == 3
    feature_frame = pd.read_csv(summary["feature_dataset_path"])
    assert feature_frame["has_vibration"].all()
    assert feature_frame["has_thermal"].all()
    assert feature_frame["elapsed_hours"].tolist() == [0.0, 1.0, 2.0]


def test_kaist_rtf_experiment_reports_no_data_cleanly(tmp_path: Path) -> None:
    extracted_root = tmp_path / "empty_extracted"
    extracted_root.mkdir()

    config = ExperimentConfig()
    config.dataset.name = "kaist_run_to_failure"
    config.dataset.variant = "compact_hourly_progression"
    config.paths.data_root = str(extracted_root)
    config.paths.processed_root = str(tmp_path / "processed")
    config.paths.results_root = str(tmp_path / "results")

    summary = run_kaist_rtf_experiment(config)

    assert summary["status"] == "no_data_available"


def test_kaist_rtf_experiment_runs_from_processed_feature_dataset(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    dataset_dir = processed_root / "kaist_run_to_failure" / "datasets"
    dataset_dir.mkdir(parents=True)
    feature_dataset_path = dataset_dir / "kaist_rtf_feature_dataset.csv"

    n_rows = 12
    frame = pd.DataFrame(
        {
            "session_id": ["kaist_rtf_run_001"] * n_rows,
            "group_id": ["kaist_rtf_run_001"] * n_rows,
            "label": ["unknown"] * n_rows,
            "multiclass_label": ["unknown"] * n_rows,
            "window_index": list(range(n_rows)),
            "window_start": [float(index * 3600) for index in range(n_rows)],
            "window_end": [float(index * 3600 + 60.0) for index in range(n_rows)],
            "window_pairing_strategy": ["hourly_progression_order"] * n_rows,
            "source_file_name": [f"hour_{index:03d}.csv" for index in range(n_rows)],
            "source_relative_path": [f"hour_{index:03d}.csv" for index in range(n_rows)],
            "progression_index": list(range(n_rows)),
            "elapsed_hours": [float(index) for index in range(n_rows)],
            "relative_progress": np.linspace(0.0, 1.0, n_rows),
            "measurement_duration_sec": [60.0] * n_rows,
            "sampling_rate_hz": [25600.0] * n_rows,
            "n_samples": [128] * n_rows,
            "vibration_feature_a": np.concatenate([np.ones(6), np.linspace(1.5, 3.0, 6)]),
            "vibration_feature_b": np.concatenate([np.ones(6) * 0.5, np.linspace(0.7, 1.5, 6)]),
            "thermal_feature_a": np.linspace(30.0, 36.0, n_rows),
            "thermal_feature_b": np.concatenate([np.ones(6) * 24.0, np.linspace(24.5, 27.0, 6)]),
        }
    )
    frame.to_csv(feature_dataset_path, index=False)

    config = ExperimentConfig()
    config.dataset.name = "kaist_run_to_failure"
    config.dataset.variant = "compact_hourly_progression"
    config.paths.processed_root = str(processed_root)
    config.paths.feature_dataset_path = str(feature_dataset_path)
    config.paths.results_root = str(tmp_path / "results")
    config.run_to_failure.healthy_reference_max_hours = 4.0
    config.run_to_failure.minimum_reference_files = 3
    config.run_to_failure.rolling_window_files = 3
    config.model.anomaly.minimum_healthy_windows = 3

    summary = run_kaist_rtf_experiment(config)

    assert summary["status"] == "ok"
    assert summary["n_feature_rows"] == n_rows
    trend_frame = pd.read_csv(Path(summary["results_dir"]) / "anomaly_trend.csv")
    crossing_frame = pd.read_csv(Path(summary["results_dir"]) / "threshold_crossing_summary.csv")
    assert set(crossing_frame["model_name"]) == {"isolation_forest", "one_class_svm"}
    assert {"anomaly_score_raw", "rolling_anomaly_score_mean", "sustained_warning"}.issubset(trend_frame.columns)
