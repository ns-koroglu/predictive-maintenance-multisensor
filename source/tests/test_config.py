from __future__ import annotations

from pathlib import Path

from pdm_pipeline.config import ExperimentConfig


def test_config_loading_from_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment_name: test_run",
                "paths:",
                "  data_root: data/custom",
                "evaluation:",
                "  split_strategy: session",
                "model:",
                "  anomaly:",
                "    threshold_strategy: quantile",
                "    threshold_buffer_std: 2.0",
                "ablation:",
                "  enabled: true",
                "  include_metadata: false",
            ]
        ),
        encoding="utf-8",
    )

    config = ExperimentConfig.from_yaml(config_path)

    assert config.experiment_name == "test_run"
    assert config.paths.data_root == "data/custom"
    assert config.evaluation.split_strategy == "session"
    assert config.model.anomaly.threshold_strategy == "quantile"
    assert config.model.anomaly.threshold_buffer_std == 2.0
    assert config.ablation.enabled is True
    assert config.ablation.include_metadata is False
