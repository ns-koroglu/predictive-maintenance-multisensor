from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat

from pdm_pipeline.config import ExperimentConfig
from pdm_pipeline.framework import run_configured_experiment
from pdm_pipeline.paderborn_adapter.compact import (
    _normalize_fault_component,
    _normalize_fault_origin,
    build_paderborn_compact_dataset,
)
from pdm_pipeline.paderborn_experiment import run_paderborn_baseline_experiment


def _write_fake_paderborn_mat(path: Path, offset: float) -> None:
    host_time = np.linspace(0.0, 4.0, 128)
    mech_time = np.linspace(0.0, 4.0, 32)
    temp_time = np.linspace(0.0, 4.0, 5)

    payload = {
        path.stem: {
            "Description": {
                "Measurement": {"Length": 4.0},
                "Recording": {"StopCondition": "Time limit: 4.000000"},
            },
            "X": np.array(
                [
                    {"Raster": "Mech_4kHz", "Data": mech_time},
                    {"Raster": "HostService", "Data": host_time},
                    {"Raster": "Temp_1Hz", "Data": temp_time},
                ],
                dtype=object,
            ),
            "Y": np.array(
                [
                    {"Name": "force", "Raster": "Mech_4kHz", "Data": np.sin(mech_time) + offset},
                    {"Name": "phase_current_1", "Raster": "HostService", "Data": np.cos(host_time) + offset},
                    {"Name": "phase_current_2", "Raster": "HostService", "Data": np.cos(host_time * 0.5) + offset},
                    {"Name": "speed", "Raster": "Mech_4kHz", "Data": np.ones_like(mech_time) * (900 + offset * 10)},
                    {"Name": "temp_2_bearing_module", "Raster": "Temp_1Hz", "Data": np.linspace(30 + offset, 31 + offset, 5)},
                    {"Name": "torque", "Raster": "Mech_4kHz", "Data": np.linspace(1.0 + offset, 1.3 + offset, mech_time.size)},
                    {"Name": "vibration_1", "Raster": "HostService", "Data": np.sin(host_time * 8.0) + offset},
                ],
                dtype=object,
            ),
        }
    }
    savemat(path, payload)


def test_paderborn_fault_normalization_is_conservative() -> None:
    assert _normalize_fault_component("OR", label="faulty") == "outer_ring"
    assert _normalize_fault_component("IR", label="faulty") == "inner_ring"
    assert _normalize_fault_component("IR OR", label="faulty") == "compound_mixed"
    assert _normalize_fault_component("AR", label="faulty") == "ambiguous"

    assert _normalize_fault_origin("artificial", label="faulty") == "artificial"
    assert _normalize_fault_origin("fatigue fatigue", label="faulty") == "fatigue"
    assert _normalize_fault_origin("plastic deformation", label="faulty") == "plastic_deformation"
    assert _normalize_fault_origin("fatigue plastic deformation", label="faulty") == "mixed"


def test_paderborn_compact_adapter_builds_dataset_and_keeps_pdf_fallback_non_blocking(tmp_path: Path) -> None:
    extracted_root = tmp_path / "extracted"
    (extracted_root / "K001").mkdir(parents=True)
    (extracted_root / "K002" / "K002").mkdir(parents=True)
    (extracted_root / "KA01").mkdir(parents=True)

    _write_fake_paderborn_mat(extracted_root / "K001" / "N09_M07_F10_K001_1.mat", 0.0)
    _write_fake_paderborn_mat(extracted_root / "K002" / "K002" / "N09_M07_F10_K002_1.mat", 0.1)
    _write_fake_paderborn_mat(extracted_root / "KA01" / "N09_M07_F10_KA01_1.mat", 0.5)

    (extracted_root / "KA01" / "KA01.pdf").write_text("not-a-real-pdf", encoding="utf-8")

    config = ExperimentConfig()
    config.dataset.name = "paderborn"
    config.dataset.variant = "compact_multirate_snapshot"
    config.paths.data_root = str(extracted_root)
    config.paths.processed_root = str(tmp_path / "processed")

    summary = build_paderborn_compact_dataset(config)

    assert summary["status"] == "ok"
    assert summary["n_bearing_codes"] == 3
    assert summary["n_feature_rows"] == 3

    feature_frame = pd.read_csv(summary["feature_dataset_path"])
    layout_frame = pd.read_csv(tmp_path / "processed" / "paderborn" / "manifests" / "layout_warnings.csv")
    audit_frame = pd.read_csv(tmp_path / "processed" / "paderborn" / "manifests" / "label_normalization_audit.csv")

    assert feature_frame["has_vibration"].all()
    assert feature_frame["has_current"].all()
    assert feature_frame["has_thermal"].all()
    assert (feature_frame["multiclass_label"].astype(str) == "unknown").all()
    assert "K002" in set(layout_frame["bearing_code"].astype(str))
    assert not audit_frame.empty


def test_paderborn_experiment_runs_vibration_only_with_group_split(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    dataset_dir = processed_root / "paderborn" / "datasets"
    dataset_dir.mkdir(parents=True)
    feature_dataset_path = dataset_dir / "paderborn_compact_feature_dataset.csv"

    rows = []
    for group_id, label, offset in [
        ("paderborn_K001", "healthy", 0.0),
        ("paderborn_K002", "healthy", 0.1),
        ("paderborn_KA01", "faulty", 1.0),
        ("paderborn_KI01", "faulty", 1.2),
    ]:
        bearing_code = group_id.replace("paderborn_", "")
        for replicate_index in range(1, 3):
            rows.append(
                {
                    "dataset_name": "paderborn",
                    "dataset_variant": "compact_multirate_snapshot",
                    "dataset_display_name": "Paderborn",
                    "session_id": f"paderborn_{bearing_code}_N09_M07_F10_{replicate_index:02d}",
                    "group_id": group_id,
                    "split_group": group_id,
                    "bearing_code": bearing_code,
                    "condition_code": "N09_M07_F10",
                    "operating_condition_n": "N09",
                    "operating_condition_m": "M07",
                    "operating_condition_f": "F10",
                    "replicate_index": replicate_index,
                    "label": label,
                    "multiclass_label": "unknown",
                    "fault_component_normalized": "none" if label == "healthy" else "outer_ring",
                    "fault_origin_normalized": "none" if label == "healthy" else "artificial",
                    "fault_component_raw": None,
                    "fault_origin_raw": None,
                    "source_file_name": f"N09_M07_F10_{bearing_code}_{replicate_index}.mat",
                    "source_relative_path": f"{bearing_code}/N09_M07_F10_{bearing_code}_{replicate_index}.mat",
                    "source_folder_layout": "flat_bearing_folder",
                    "layout_warning": None,
                    "documented_fault_notes": None,
                    "has_vibration": True,
                    "has_current": True,
                    "has_thermal": True,
                    "has_force": True,
                    "has_speed": True,
                    "has_torque": True,
                    "nominal_record_duration_sec": 4.0,
                    "sampling_rate_vibration_hz": 64000.0,
                    "sampling_rate_current_hz": 64000.0,
                    "sampling_rate_mechanical_hz": 4000.0,
                    "sampling_rate_temperature_hz": 1.0,
                    "record_status": "ok",
                    "vibration_feature_a": offset + replicate_index * 0.1,
                    "vibration_feature_b": offset + replicate_index * 0.2,
                    "current_feature_a": 10.0 + replicate_index,
                    "thermal_feature_a": 30.0 + replicate_index,
                    "force_feature_a": 5.0 + replicate_index,
                }
            )
    pd.DataFrame(rows).to_csv(feature_dataset_path, index=False)

    config = ExperimentConfig()
    config.dataset.name = "paderborn"
    config.dataset.variant = "compact_multirate_snapshot"
    config.paths.processed_root = str(processed_root)
    config.paths.feature_dataset_path = str(feature_dataset_path)
    config.paths.results_root = str(tmp_path / "results")
    config.evaluation.split_strategy = "group"
    config.evaluation.test_fraction = 0.25
    config.evaluation.positive_labels = ["faulty"]

    summary = run_paderborn_baseline_experiment(config)

    assert summary["status"] == "ok"
    assert summary["n_feature_columns"] == 2
    assert set(summary["split_summary"]["train_groups"]).isdisjoint(set(summary["split_summary"]["test_groups"]))

    results_dir = Path(summary["results_dir"])
    feature_columns = json.loads((results_dir / "artifacts" / "feature_columns.json").read_text(encoding="utf-8"))
    assert all(column.startswith("vibration_") for column in feature_columns["feature_columns"])
    assert (results_dir / "predictions" / "per_group_summary.csv").exists()
    assert (results_dir / "predictions" / "per_session_predictions.csv").exists()


def test_paderborn_framework_build_dispatch_runs(tmp_path: Path) -> None:
    extracted_root = tmp_path / "extracted"
    (extracted_root / "K001").mkdir(parents=True)
    _write_fake_paderborn_mat(extracted_root / "K001" / "N09_M07_F10_K001_1.mat", 0.0)

    config = ExperimentConfig()
    config.dataset.name = "paderborn"
    config.dataset.variant = "compact_multirate_snapshot"
    config.paths.data_root = str(extracted_root)
    config.paths.processed_root = str(tmp_path / "processed")

    summary = run_configured_experiment(config, stage="build")

    assert summary["status"] in {"ok", "no_data_available"}
