"""Unified multi-dataset experiment dispatch layer."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .config import ExperimentConfig
from .datasets import build_schema_overview, get_dataset_metadata, standardize_feature_dataset
from .demo import run_demo
from .kaist_experiment import run_kaist_baseline_experiment
from .kaist_rtf_adapter import build_kaist_rtf_compact_dataset
from .kaist_rtf_experiment import run_kaist_rtf_experiment
from .nasa_ims_adapter import build_nasa_ims_compact_dataset
from .nasa_ims_experiment import run_nasa_ims_experiment
from .paderborn_adapter import build_paderborn_compact_dataset
from .paderborn_experiment import run_paderborn_baseline_experiment
from .pipeline import build_dataset_from_config, evaluate_saved_models, run_inference, run_training_experiment


def resolve_dataset_metadata_from_config(config: ExperimentConfig) -> Dict[str, object]:
    """Resolve the selected dataset into registry metadata plus schema details."""

    metadata = get_dataset_metadata(config.dataset.name)
    resolved = metadata.to_dict()
    resolved["variant"] = str(config.dataset.variant)
    resolved["schema"] = build_schema_overview(metadata)
    return resolved


def standardize_feature_dataset_for_config(
    feature_frame: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    """Apply the shared experiment schema using the dataset selected in the config."""

    metadata = get_dataset_metadata(config.dataset.name)
    return standardize_feature_dataset(
        frame=feature_frame,
        metadata=metadata,
        dataset_variant=config.dataset.variant,
        group_column=config.dataset.group_column,
        label_column=config.dataset.label_column,
        multiclass_label_column=config.dataset.multiclass_label_column,
    )


def load_standardized_feature_dataset(config: ExperimentConfig) -> pd.DataFrame:
    """Load a feature dataset path from config and standardize it for group-aware evaluation."""

    feature_dataset_path = config.paths.feature_dataset_path
    if not feature_dataset_path:
        raise ValueError("The selected dataset path requires paths.feature_dataset_path in the config.")

    dataset_path = Path(feature_dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Feature dataset not found: {dataset_path}")

    frame = pd.read_csv(dataset_path)
    return standardize_feature_dataset_for_config(frame, config)


def run_configured_experiment(
    config: ExperimentConfig,
    stage: str = "train",
    session_dir: Optional[str | Path] = None,
) -> Dict[str, object]:
    """Dispatch one experiment stage using the selected dataset family."""

    metadata = get_dataset_metadata(config.dataset.name)
    stage_name = str(stage).lower()

    if metadata.key == "kaist_rotating_machine":
        if stage_name != "train":
            raise NotImplementedError(
                "The current KAIST runner scaffold supports the publication-style training analysis path only."
            )
        summary = run_kaist_baseline_experiment(config)
        summary["dataset_metadata"] = resolve_dataset_metadata_from_config(config)
        return summary

    if metadata.key == "kaist_run_to_failure":
        if stage_name == "build":
            summary = build_kaist_rtf_compact_dataset(config)
        elif stage_name == "train":
            summary = run_kaist_rtf_experiment(config)
        else:
            raise NotImplementedError(
                "The current KAIST run-to-failure path supports only 'build' and 'train' stages."
            )
        summary["dataset_metadata"] = resolve_dataset_metadata_from_config(config)
        return summary

    if metadata.key == "nasa_ims":
        if stage_name == "build":
            summary = build_nasa_ims_compact_dataset(config)
        elif stage_name == "train":
            summary = run_nasa_ims_experiment(config)
        else:
            raise NotImplementedError(
                "The current NASA IMS path supports only 'build' and 'train' stages."
            )
        summary["dataset_metadata"] = resolve_dataset_metadata_from_config(config)
        return summary

    if metadata.key == "paderborn":
        if stage_name == "build":
            summary = build_paderborn_compact_dataset(config)
        elif stage_name == "train":
            summary = run_paderborn_baseline_experiment(config)
        else:
            raise NotImplementedError(
                "The current Paderborn path supports only 'build' and 'train' stages."
            )
        summary["dataset_metadata"] = resolve_dataset_metadata_from_config(config)
        return summary

    if metadata.key == "session_folder_baseline":
        if stage_name == "build":
            summary = build_dataset_from_config(config)
        elif stage_name == "train":
            summary = run_training_experiment(config)
        elif stage_name == "evaluate":
            summary = evaluate_saved_models(config)
        elif stage_name == "infer":
            summary = run_inference(config, session_dir=session_dir)
        elif stage_name == "demo":
            summary = run_demo(config)
        else:
            raise ValueError(f"Unsupported experiment stage: {stage_name}")
        summary["dataset_metadata"] = resolve_dataset_metadata_from_config(config)
        return summary

    raise NotImplementedError(
        f"Dataset '{metadata.display_name}' is registered as scaffold only. "
        "No adapter or experiment runner has been implemented yet."
    )
