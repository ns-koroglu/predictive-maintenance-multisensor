"""Binary vibration-only benchmark path for the Paderborn dataset."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .datasets import get_dataset_metadata, standardize_feature_dataset
from .evaluation import select_feature_columns, split_dataset, summarize_split
from .models import train_classifier, evaluate_classifier
from .paderborn_adapter import build_paderborn_compact_dataset
from .pipeline import _save_classifier_artifacts, _save_feature_columns, _save_scalar_metrics_csv, _save_split_artifacts
from .plots import plot_class_distribution, plot_confusion_matrix, plot_feature_importances
from .utils import prepare_experiment_directories, save_json, save_yaml


def _default_feature_dataset_path(config: ExperimentConfig) -> Path:
    """Resolve the processed Paderborn feature dataset path."""

    if config.paths.feature_dataset_path:
        return Path(config.paths.feature_dataset_path)
    return Path(config.paths.processed_root) / "paderborn" / "datasets" / "paderborn_compact_feature_dataset.csv"


def _load_or_build_feature_dataset(config: ExperimentConfig) -> tuple[Path, Dict[str, object]]:
    """Prefer the processed compact dataset and build it only if missing."""

    feature_dataset_path = _default_feature_dataset_path(config)
    processed_root = Path(config.paths.processed_root) / "paderborn"
    adapter_summary_path = processed_root / "adapter_summary.json"

    if feature_dataset_path.exists():
        if adapter_summary_path.exists():
            with adapter_summary_path.open("r", encoding="utf-8") as handle:
                return feature_dataset_path, json.load(handle)
        return feature_dataset_path, {
            "dataset_name": "paderborn",
            "status": "existing_feature_dataset",
            "feature_dataset_path": str(feature_dataset_path),
        }

    adapter_summary = build_paderborn_compact_dataset(config)
    return Path(adapter_summary["feature_dataset_path"]), adapter_summary


def _load_paderborn_feature_dataset(feature_dataset_path: Path) -> pd.DataFrame:
    """Load and validate the compact Paderborn feature dataset."""

    if not feature_dataset_path.exists():
        raise FileNotFoundError(f"Paderborn feature dataset not found: {feature_dataset_path}")

    frame = pd.read_csv(feature_dataset_path, low_memory=False)
    required_columns = {"session_id", "group_id", "split_group", "label"}
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ValueError(f"Paderborn feature dataset is missing required columns: {missing}")

    metadata = get_dataset_metadata("paderborn")
    frame = standardize_feature_dataset(
        frame=frame,
        metadata=metadata,
        dataset_variant="compact_multirate_snapshot",
    )
    return frame.sort_values(["bearing_code", "condition_code", "replicate_index"]).reset_index(drop=True)


def _vibration_only_feature_columns(frame: pd.DataFrame) -> List[str]:
    """Select only the vibration block for the first benchmark."""

    feature_columns = [column for column in select_feature_columns(frame) if str(column).startswith("vibration_")]
    if not feature_columns:
        raise ValueError("No vibration_* feature columns were found for the Paderborn baseline.")
    return feature_columns


def _build_group_summary(prediction_frame: pd.DataFrame, healthy_label: str) -> pd.DataFrame:
    """Aggregate measurement-file predictions by bearing-code group."""

    rows: List[Dict[str, object]] = []
    for group_id, frame in prediction_frame.groupby("group_id", sort=True):
        row: Dict[str, object] = {
            "group_id": str(group_id),
            "split_group": str(frame["split_group"].iloc[0]) if "split_group" in frame.columns else str(group_id),
            "true_label": str(frame["label"].iloc[0]) if "label" in frame.columns else "",
            "n_sessions": int(frame["session_id"].nunique()) if "session_id" in frame.columns else int(len(frame)),
            "majority_predicted_label": Counter(frame["predicted_label"].astype(str)).most_common(1)[0][0],
            "predicted_faulty_ratio": float((frame["predicted_label"].astype(str) != str(healthy_label)).mean()),
        }
        if "prob_faulty" in frame.columns:
            row["mean_faulty_probability"] = float(frame["prob_faulty"].mean())
            row["max_faulty_probability"] = float(frame["prob_faulty"].max())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("group_id").reset_index(drop=True)


def _write_experiment_summary_markdown(
    config: ExperimentConfig,
    split_summary: Dict[str, object],
    classifier_metrics: Dict[str, object],
    output_path: Path,
) -> None:
    """Write a compact benchmark report suitable for the thesis appendix."""

    lines = [
        f"# Experiment Summary: {config.experiment_name}",
        "",
        "## Problem Setting",
        "- Dataset: `paderborn`",
        "- Task: binary healthy vs faulty",
        "- Baseline: vibration-only Random Forest",
        f"- Split strategy: `{split_summary['split_strategy']}`",
        "",
        "## Split Summary",
        f"- Train groups: {split_summary['n_train_groups']}",
        f"- Test groups: {split_summary['n_test_groups']}",
        f"- Train sessions: {split_summary['n_train_sessions']}",
        f"- Test sessions: {split_summary['n_test_sessions']}",
        f"- Group overlap check: {split_summary.get('group_overlap') or 'none'}",
        "",
        "## Label Distribution",
        f"- Train: {split_summary.get('train_label_distribution', {})}",
        f"- Test: {split_summary.get('test_label_distribution', {})}",
        "",
        "## Main Metrics",
        f"- Balanced accuracy: {classifier_metrics.get('balanced_accuracy', 'n/a')}",
        f"- Macro precision: {classifier_metrics.get('precision_macro', 'n/a')}",
        f"- Macro recall: {classifier_metrics.get('recall_macro', 'n/a')}",
        f"- Macro F1: {classifier_metrics.get('f1_macro', 'n/a')}",
        f"- Accuracy: {classifier_metrics.get('accuracy', 'n/a')}",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_paderborn_baseline_experiment(config: ExperimentConfig) -> Dict[str, object]:
    """Run the first binary vibration-only Paderborn baseline experiment."""

    feature_dataset_path, adapter_summary = _load_or_build_feature_dataset(config)
    feature_frame = _load_paderborn_feature_dataset(feature_dataset_path)
    if feature_frame.empty:
        summary = {
            "dataset_name": "paderborn",
            "status": "no_data_available",
            "feature_dataset_path": str(feature_dataset_path),
        }
        return summary

    directories = prepare_experiment_directories(config.paths.results_root, config.experiment_name)
    save_yaml(directories["root"] / "config_snapshot.yaml", config.to_dict())

    feature_columns = _vibration_only_feature_columns(feature_frame)
    train_frame, test_frame = split_dataset(
        feature_frame,
        strategy=config.evaluation.split_strategy,
        test_fraction=config.evaluation.test_fraction,
        random_state=config.evaluation.random_state,
    )
    split_summary = summarize_split(train_frame, test_frame, strategy=config.evaluation.split_strategy)
    classifier_artifact = train_classifier(
        train_frame=train_frame,
        feature_columns=feature_columns,
        label_column="label",
        config=config.model.classifier,
    )
    classifier_metrics, prediction_frame = evaluate_classifier(
        artifact=classifier_artifact,
        test_frame=test_frame,
        feature_columns=feature_columns,
        label_column="label",
        healthy_label=config.evaluation.healthy_label,
        positive_labels=config.evaluation.positive_labels,
    )
    prediction_frame = prediction_frame.sort_values(["group_id", "session_id"]).reset_index(drop=True)
    group_summary = _build_group_summary(prediction_frame, healthy_label=config.evaluation.healthy_label)

    save_json(directories["metrics"] / "classifier_metrics.json", classifier_metrics)
    _save_scalar_metrics_csv(classifier_metrics, directories["metrics"] / "classifier_metrics.csv")
    save_json(directories["root"] / "experiment_summary.json", {
        "dataset_name": "paderborn",
        "status": "ok",
        "n_feature_rows": int(len(feature_frame)),
        "n_feature_columns": int(len(feature_columns)),
        "feature_dataset_path": str(feature_dataset_path),
        "split_summary": split_summary,
        "classifier_metrics": classifier_metrics,
    })
    _save_split_artifacts(train_frame, test_frame, split_summary, directories)
    _save_classifier_artifacts(classifier_artifact, directories)
    _save_feature_columns(feature_columns, directories)

    prediction_frame.to_csv(directories["predictions"] / "per_session_predictions.csv", index=False)
    group_summary.to_csv(directories["predictions"] / "per_group_summary.csv", index=False)

    class_distribution_frame = pd.DataFrame(
        {"label": feature_frame["label"].astype(str)}
    )
    plot_class_distribution(class_distribution_frame, directories["plots"] / "class_distribution.png")
    plot_confusion_matrix(
        matrix=np.asarray(classifier_metrics["confusion_matrix"], dtype=int),
        labels=classifier_metrics["labels"],
        output_path=directories["plots"] / "confusion_matrix.png",
        title="Paderborn Confusion Matrix",
    )
    importance_frame = pd.DataFrame(classifier_metrics.get("top_feature_importances", []))
    if not importance_frame.empty:
        plot_feature_importances(importance_frame, directories["plots"] / "feature_importance.png")
        importance_frame.to_csv(directories["metrics"] / "feature_importance.csv", index=False)

    _write_experiment_summary_markdown(
        config=config,
        split_summary=split_summary,
        classifier_metrics=classifier_metrics,
        output_path=directories["root"] / "experiment_summary.md",
    )

    summary = {
        "dataset_name": "paderborn",
        "status": "ok",
        "results_dir": str(directories["root"]),
        "feature_dataset_path": str(feature_dataset_path),
        "n_feature_rows": int(len(feature_frame)),
        "n_feature_columns": int(len(feature_columns)),
        "label_distribution": feature_frame["label"].astype(str).value_counts().to_dict(),
        "split_summary": split_summary,
        "classifier_metrics": classifier_metrics,
        "adapter_summary": adapter_summary,
    }
    return summary
