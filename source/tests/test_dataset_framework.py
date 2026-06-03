from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pdm_pipeline.config import ExperimentConfig
from pdm_pipeline.datasets import get_dataset_metadata, list_supported_datasets, standardize_feature_dataset
from pdm_pipeline.framework import resolve_dataset_metadata_from_config, run_configured_experiment


def test_dataset_registry_contains_requested_public_datasets() -> None:
    dataset_names = {item.key for item in list_supported_datasets()}

    assert "kaist_rotating_machine" in dataset_names
    assert "kaist_run_to_failure" in dataset_names
    assert "nasa_ims" in dataset_names
    assert "paderborn" in dataset_names
    assert "cwru" in dataset_names


def test_standardized_feature_dataset_adds_common_schema_fields() -> None:
    metadata = get_dataset_metadata("kaist_rotating_machine")
    frame = pd.DataFrame(
        {
            "session_id": ["s1", "s1"],
            "label": ["healthy", "healthy"],
            "window_index": [0, 1],
            "vibration_rms": [0.1, 0.2],
            "thermal_temp_mean": [30.0, 30.2],
        }
    )

    standardized = standardize_feature_dataset(
        frame=frame,
        metadata=metadata,
        dataset_variant="test_variant",
    )

    assert standardized["dataset_name"].eq("kaist_rotating_machine").all()
    assert standardized["dataset_variant"].eq("test_variant").all()
    assert standardized["group_id"].eq("s1").all()
    assert standardized["has_vibration"].all()
    assert standardized["has_thermal"].all()
    assert not standardized["has_ae"].any()
    assert not standardized["has_current"].any()


def test_framework_config_resolves_dataset_metadata(tmp_path: Path) -> None:
    config_path = tmp_path / "framework.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment_name: framework_test",
                "dataset:",
                "  name: kaist_rotating_machine",
                "  variant: compact_vibration_thermal_features",
                "paths:",
                "  feature_dataset_path: results/kaist_feature_build/datasets/kaist_vibration_thermal_features.csv",
            ]
        ),
        encoding="utf-8",
    )

    config = ExperimentConfig.from_yaml(config_path)
    resolved = resolve_dataset_metadata_from_config(config)

    assert resolved["key"] == "kaist_rotating_machine"
    assert resolved["variant"] == "compact_vibration_thermal_features"
    assert resolved["schema"]["group_column"] == "session_id"


def test_paderborn_registry_entry_is_implemented() -> None:
    metadata = get_dataset_metadata("paderborn")

    assert metadata.implementation_status == "implemented"
    assert metadata.recommended_baseline_modalities == ("vibration",)


def test_unimplemented_registered_dataset_raises_clear_error() -> None:
    config = ExperimentConfig()
    config.dataset.name = "cwru"

    with pytest.raises(NotImplementedError, match="scaffold only"):
        run_configured_experiment(config, stage="train")
