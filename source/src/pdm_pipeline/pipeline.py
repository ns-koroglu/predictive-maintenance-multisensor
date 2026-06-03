"""End-to-end baseline pipeline orchestration."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .datasets import get_dataset_metadata, standardize_feature_dataset
from .evaluation import select_feature_columns, split_dataset, summarize_split
from .features import extract_fused_features
from .io import list_session_directories, load_session
from .models import evaluate_anomaly_detector, evaluate_classifier, train_anomaly_detector, train_classifier
from .plots import (
    plot_anomaly_scores,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_feature_importances,
    plot_feature_trends,
)
from .preprocessing import preprocess_session, synchronize_session
from .utils import prepare_experiment_directories, save_json, save_yaml
from .windowing import create_session_windows


def _label_from_metadata(metadata: Dict[str, object]) -> str:
    """Read the session label in a consistent way."""

    label = metadata.get("label", "unknown")
    return str(label) if label is not None else "unknown"


def _notes_from_metadata(metadata: Dict[str, object]) -> Optional[str]:
    """Collect optional notes without affecting model features."""

    for key in ("note", "notes"):
        value = metadata.get(key)
        if value:
            return str(value)
    return None


def _save_feature_columns(feature_columns: List[str], directories: Dict[str, Path]) -> None:
    """Persist feature names as JSON and CSV for reproducible inference."""

    save_json(directories["artifacts"] / "feature_columns.json", {"feature_columns": feature_columns})
    pd.DataFrame({"feature": feature_columns}).to_csv(
        directories["artifacts"] / "feature_columns.csv",
        index=False,
    )


def _save_scalar_metrics_csv(metrics: Dict[str, object], output_path: Path) -> None:
    """Save flat scalar metrics to CSV for spreadsheet-friendly inspection."""

    scalar_metrics = {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (str, int, float)) and not isinstance(value, bool)
    }
    bool_metrics = {key: bool(value) for key, value in metrics.items() if isinstance(value, (bool, np.bool_))}
    scalar_metrics.update(bool_metrics)
    if scalar_metrics:
        pd.DataFrame([scalar_metrics]).to_csv(output_path, index=False)


def _save_classifier_metrics_tables(metrics: Dict[str, object], directories: Dict[str, Path]) -> None:
    """Persist classifier metrics as machine-readable tables."""

    save_json(directories["metrics"] / "classifier_metrics.json", metrics)
    _save_scalar_metrics_csv(metrics, directories["metrics"] / "classifier_scalar_metrics.csv")
    pd.DataFrame(metrics["classification_report"]).transpose().to_csv(
        directories["metrics"] / "classifier_classification_report.csv"
    )
    pd.DataFrame(
        metrics["confusion_matrix"],
        index=metrics["labels"],
        columns=metrics["labels"],
    ).to_csv(directories["metrics"] / "classifier_confusion_matrix.csv")


def _save_anomaly_metrics_tables(metrics: Dict[str, object], directories: Dict[str, Path]) -> None:
    """Persist anomaly metrics and calibration details."""

    save_json(directories["metrics"] / "anomaly_metrics.json", metrics)
    _save_scalar_metrics_csv(metrics, directories["metrics"] / "anomaly_scalar_metrics.csv")
    if "confusion_matrix" in metrics:
        pd.DataFrame(
            metrics["confusion_matrix"],
            index=metrics["labels"],
            columns=metrics["labels"],
        ).to_csv(directories["metrics"] / "anomaly_confusion_matrix.csv")
    if "calibration" in metrics:
        save_json(directories["metrics"] / "anomaly_calibration.json", metrics["calibration"])
        pd.DataFrame([metrics["calibration"]]).to_csv(
            directories["metrics"] / "anomaly_calibration.csv",
            index=False,
        )


def _save_split_artifacts(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    split_summary: Dict[str, object],
    directories: Dict[str, Path],
) -> None:
    """Persist session assignments and leakage checks."""

    session_rows: List[Dict[str, object]] = []
    for split_name, frame in (("train", train_frame), ("test", test_frame)):
        if "session_id" not in frame.columns:
            continue
        session_subset = frame[["session_id", "label"]].drop_duplicates().copy()
        session_subset["split"] = split_name
        session_rows.extend(session_subset.to_dict(orient="records"))

    if session_rows:
        pd.DataFrame(session_rows).sort_values(["split", "session_id"]).to_csv(
            directories["datasets"] / "session_split_manifest.csv",
            index=False,
        )

    group_rows: List[Dict[str, object]] = []
    for split_name, frame in (("train", train_frame), ("test", test_frame)):
        if "split_group" not in frame.columns:
            continue
        subset_columns = [column for column in ["split_group", "group_id", "label"] if column in frame.columns]
        group_subset = frame[subset_columns].drop_duplicates().copy()
        group_subset["split"] = split_name
        group_rows.extend(group_subset.to_dict(orient="records"))

    if group_rows:
        pd.DataFrame(group_rows).sort_values(["split", "split_group"]).to_csv(
            directories["datasets"] / "group_split_manifest.csv",
            index=False,
        )
    save_json(directories["metrics"] / "split_summary.json", split_summary)


def _save_classifier_artifacts(
    classifier_artifact: Dict[str, object],
    directories: Dict[str, Path],
) -> None:
    """Save classifier components and the combined artifact."""

    joblib.dump(classifier_artifact, directories["artifacts"] / "classifier_artifact.joblib")
    joblib.dump(classifier_artifact["preprocessor"], directories["artifacts"] / "classifier_preprocessor.joblib")
    joblib.dump(classifier_artifact["label_encoder"], directories["artifacts"] / "classifier_label_encoder.joblib")
    joblib.dump(classifier_artifact["model"], directories["models"] / "random_forest_model.joblib")


def _save_anomaly_artifacts(
    anomaly_artifact: Dict[str, object],
    directories: Dict[str, Path],
) -> None:
    """Save anomaly detector components and calibration artifacts."""

    joblib.dump(anomaly_artifact, directories["artifacts"] / "anomaly_artifact.joblib")
    joblib.dump(anomaly_artifact["preprocessor"], directories["artifacts"] / "anomaly_preprocessor.joblib")
    joblib.dump(anomaly_artifact["model"], directories["models"] / "isolation_forest_model.joblib")


def _remove_legacy_result_files(directories: Dict[str, Path]) -> None:
    """Remove obsolete artifact names left from older flat-script runs."""

    legacy_paths = [
        directories["models"] / "classifier.joblib",
        directories["models"] / "anomaly_detector.joblib",
        directories["metrics"] / "metrics.json",
    ]
    for legacy_path in legacy_paths:
        if legacy_path.exists():
            legacy_path.unlink()


def _build_session_prediction_summary(
    classifier_predictions: pd.DataFrame,
    anomaly_predictions: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Aggregate window predictions into thesis-friendly per-session summaries."""

    rows: List[Dict[str, object]] = []
    for session_id, frame in classifier_predictions.groupby("session_id"):
        row: Dict[str, object] = {
            "session_id": str(session_id),
            "n_windows": int(len(frame)),
            "true_label": str(frame["label"].iloc[0]) if "label" in frame.columns else "",
            "majority_predicted_label": Counter(frame["predicted_label"].astype(str)).most_common(1)[0][0],
        }
        if "label" in frame.columns:
            row["window_accuracy"] = float((frame["predicted_label"].astype(str) == frame["label"].astype(str)).mean())
        if "max_probability" in frame.columns:
            row["mean_max_probability"] = float(frame["max_probability"].mean())
        rows.append(row)

    summary = pd.DataFrame(rows)
    if anomaly_predictions is not None and not anomaly_predictions.empty:
        anomaly_summary = (
            anomaly_predictions.groupby("session_id")
            .agg(
                mean_anomaly_score=("anomaly_score", "mean"),
                max_anomaly_score=("anomaly_score", "max"),
                anomalous_window_ratio=("anomaly_flag", "mean"),
            )
            .reset_index()
        )
        summary = summary.merge(anomaly_summary, on="session_id", how="left")
    return summary.sort_values("session_id").reset_index(drop=True)


def _write_experiment_summary_markdown(
    config: ExperimentConfig,
    split_summary: Dict[str, object],
    classifier_metrics: Dict[str, object],
    anomaly_metrics: Dict[str, object],
    directories: Dict[str, Path],
) -> None:
    """Write a compact Markdown report for thesis appendices."""

    lines = [
        f"# Experiment Summary: {config.experiment_name}",
        "",
        "## Dataset and Split",
        f"- Data root: `{config.paths.data_root}`",
        f"- Split strategy: `{split_summary['split_strategy']}`",
        f"- Train windows: {split_summary['n_train_windows']}",
        f"- Test windows: {split_summary['n_test_windows']}",
        f"- Train sessions: {', '.join(split_summary['train_sessions']) or 'None'}",
        f"- Test sessions: {', '.join(split_summary['test_sessions']) or 'None'}",
        f"- Session overlap check: {split_summary['session_overlap'] or 'none'}",
        "",
        "## Classification",
        f"- Balanced accuracy: {classifier_metrics.get('balanced_accuracy', 'n/a')}",
        f"- Macro precision: {classifier_metrics.get('precision_macro', 'n/a')}",
        f"- Macro recall: {classifier_metrics.get('recall_macro', 'n/a')}",
        f"- Macro F1: {classifier_metrics.get('f1_macro', 'n/a')}",
        f"- ROC-AUC: {classifier_metrics.get('roc_auc', 'n/a')}",
        f"- PR-AUC: {classifier_metrics.get('pr_auc', 'n/a')}",
        "",
        "## Anomaly Detection",
        f"- Threshold: {anomaly_metrics.get('threshold', 'n/a')}",
        f"- Balanced accuracy: {anomaly_metrics.get('balanced_accuracy', 'n/a')}",
        f"- Precision: {anomaly_metrics.get('precision', 'n/a')}",
        f"- Recall: {anomaly_metrics.get('recall', 'n/a')}",
        f"- F1: {anomaly_metrics.get('f1', 'n/a')}",
        f"- ROC-AUC: {anomaly_metrics.get('roc_auc', 'n/a')}",
        f"- PR-AUC: {anomaly_metrics.get('pr_auc', 'n/a')}",
    ]
    if "calibration" in anomaly_metrics:
        lines.extend(
            [
                "",
                "## Anomaly Calibration",
                f"- Strategy: {anomaly_metrics['calibration'].get('strategy', 'n/a')}",
                f"- Base threshold: {anomaly_metrics['calibration'].get('base_threshold', 'n/a')}",
                f"- Buffer in std units: {anomaly_metrics['calibration'].get('threshold_buffer_std', 'n/a')}",
            ]
        )

    report_path = directories["root"] / "experiment_summary.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_warning_reason(
    predicted_class: str,
    healthy_label: str,
    anomalous_window_ratio: float,
    warning_ratio_threshold: float,
) -> tuple[bool, bool, str, str]:
    """Explain whether a warning came from the classifier, anomaly ratio, or both."""

    classifier_trigger = str(predicted_class).lower() != str(healthy_label).lower()
    anomaly_trigger = float(anomalous_window_ratio) >= float(warning_ratio_threshold)

    if classifier_trigger and anomaly_trigger:
        reason = "warning triggered by both"
    elif classifier_trigger:
        reason = "warning triggered by classifier"
    elif anomaly_trigger:
        reason = "warning triggered by anomaly ratio"
    else:
        reason = "no warning triggered"

    detail = (
        f"Predicted class is '{predicted_class}', anomaly ratio is {anomalous_window_ratio:.3f}, "
        f"and the anomaly warning threshold is {warning_ratio_threshold:.3f}. "
        f"The final decision is: {reason}."
    )
    return classifier_trigger, anomaly_trigger, reason, detail


def build_feature_rows_for_session(session_dir: str | Path, config: ExperimentConfig) -> List[Dict[str, object]]:
    """Load one session folder and convert it into fused feature rows."""

    session = load_session(session_dir)
    session = preprocess_session(
        session,
        interpolation_method=config.preprocessing.interpolation_method,
        fill_missing=config.preprocessing.fill_missing,
    )
    session = synchronize_session(session, method=config.synchronization.method)
    windows = create_session_windows(
        session,
        duration_sec=config.window.duration_sec,
        overlap=config.window.overlap,
        minimum_samples=config.window.minimum_samples,
    )

    rows: List[Dict[str, object]] = []
    for window in windows:
        feature_row = extract_fused_features(
            vibration_frame=window.vibration,
            ae_frame=window.ae,
            thermal_frame=window.thermal,
            metadata=window.metadata,
            vibration_sampling_rate_hz=float(window.metadata.get("vibration_sampling_rate_hz", 0.0)),
            ae_sampling_rate_hz=float(window.metadata.get("ae_sampling_rate_hz", 0.0)),
        )
        row: Dict[str, object] = {
            "session_id": window.session_id,
            "label": _label_from_metadata(window.metadata),
            "window_index": int(window.window_index),
            "window_start": float(window.start_time),
            "window_end": float(window.end_time),
            "diag_ae_samples": int(len(window.ae)),
            "diag_vibration_samples": int(len(window.vibration)),
            "diag_thermal_samples": int(len(window.thermal)),
        }
        if "lubrication_state" in window.metadata:
            row["lubrication_state"] = window.metadata["lubrication_state"]
        note = _notes_from_metadata(window.metadata)
        if note:
            row["note"] = note
        row.update(feature_row)
        rows.append(row)
    return rows


def build_feature_dataset(config: ExperimentConfig) -> pd.DataFrame:
    """Build the window-level fused dataset for all available sessions."""

    rows: List[Dict[str, object]] = []
    session_directories = list_session_directories(config.paths.data_root)
    if config.paths.session_whitelist:
        allowed = set(map(str, config.paths.session_whitelist))
        session_directories = [session_dir for session_dir in session_directories if session_dir.name in allowed]

    for session_dir in session_directories:
        try:
            session_rows = build_feature_rows_for_session(session_dir, config)
        except Exception as error:
            raise RuntimeError(f"Failed to process session '{session_dir.name}': {error}") from error
        if session_rows:
            rows.extend(session_rows)

    if not rows:
        raise RuntimeError("No feature rows were produced. Check the input data and window settings.")

    dataset = pd.DataFrame(rows)
    metadata = get_dataset_metadata(config.dataset.name)
    dataset = standardize_feature_dataset(
        frame=dataset,
        metadata=metadata,
        dataset_variant=config.dataset.variant,
        group_column=config.dataset.group_column,
        label_column=config.dataset.label_column,
        multiclass_label_column=config.dataset.multiclass_label_column,
    )
    dataset = dataset.sort_values(["session_id", "window_start"]).reset_index(drop=True)
    return dataset


def build_dataset_from_config(config: ExperimentConfig) -> Dict[str, object]:
    """Build and save only the feature dataset and config snapshot."""

    directories = prepare_experiment_directories(config.paths.results_root, config.experiment_name)
    save_yaml(directories["root"] / "config_snapshot.yaml", config.to_dict())

    dataset = build_feature_dataset(config)
    dataset_path = directories["datasets"] / "window_feature_dataset.csv"
    dataset.to_csv(dataset_path, index=False)

    summary = {
        "experiment_name": config.experiment_name,
        "dataset_path": str(dataset_path),
        "n_windows": int(len(dataset)),
        "n_sessions": int(dataset["session_id"].nunique()),
        "class_distribution": dataset["label"].astype(str).value_counts().to_dict(),
    }
    save_json(directories["metrics"] / "dataset_summary.json", summary)
    pd.DataFrame([summary]).to_csv(directories["metrics"] / "dataset_summary.csv", index=False)
    return summary


def run_training_experiment(config: ExperimentConfig) -> Dict[str, object]:
    """Train baseline models, evaluate them, and save plots plus artifacts."""

    directories = prepare_experiment_directories(config.paths.results_root, config.experiment_name)
    _remove_legacy_result_files(directories)
    save_yaml(directories["root"] / "config_snapshot.yaml", config.to_dict())

    dataset = build_feature_dataset(config)
    dataset.to_csv(directories["datasets"] / "window_feature_dataset.csv", index=False)

    labelled_dataset = dataset[dataset["label"].astype(str).str.lower() != "unknown"].copy()
    if labelled_dataset.empty:
        raise RuntimeError("Training requires labels in metadata.json.")

    feature_columns = select_feature_columns(labelled_dataset)
    if not feature_columns:
        raise RuntimeError("No numeric feature columns were found after feature extraction.")
    _save_feature_columns(feature_columns, directories)

    train_frame, test_frame = split_dataset(
        labelled_dataset,
        strategy=config.evaluation.split_strategy,
        test_fraction=config.evaluation.test_fraction,
        random_state=config.evaluation.random_state,
    )
    if train_frame.empty or test_frame.empty:
        raise RuntimeError("Train/test split failed. Provide more labelled sessions or reduce the test fraction.")

    split_summary = summarize_split(train_frame, test_frame, strategy=config.evaluation.split_strategy)
    if config.evaluation.split_strategy == "session" and split_summary["session_overlap"]:
        raise RuntimeError(f"Session leakage detected: {split_summary['session_overlap']}")
    _save_split_artifacts(train_frame, test_frame, split_summary, directories)

    train_frame.to_csv(directories["datasets"] / "train_windows.csv", index=False)
    test_frame.to_csv(directories["datasets"] / "test_windows.csv", index=False)

    classifier_artifact = train_classifier(
        train_frame=train_frame,
        feature_columns=feature_columns,
        label_column="label",
        config=config.model.classifier,
    )
    classifier_metrics, classifier_predictions = evaluate_classifier(
        artifact=classifier_artifact,
        test_frame=test_frame,
        feature_columns=feature_columns,
        label_column="label",
        healthy_label=config.evaluation.healthy_label,
        positive_labels=config.evaluation.positive_labels,
    )
    classifier_predictions.to_csv(directories["predictions"] / "classifier_test_predictions.csv", index=False)
    _save_classifier_metrics_tables(classifier_metrics, directories)
    _save_classifier_artifacts(classifier_artifact, directories)

    importance_frame = pd.DataFrame(classifier_metrics.get("top_feature_importances", []))
    if not importance_frame.empty:
        importance_frame.to_csv(directories["metrics"] / "classifier_feature_importances.csv", index=False)

    anomaly_metrics: Dict[str, object]
    anomaly_predictions: Optional[pd.DataFrame] = None
    anomaly_artifact: Optional[Dict[str, object]] = None
    try:
        anomaly_artifact = train_anomaly_detector(
            train_frame=train_frame,
            feature_columns=feature_columns,
            label_column="label",
            healthy_label=config.evaluation.healthy_label,
            config=config.model.anomaly,
        )
        anomaly_metrics, anomaly_predictions = evaluate_anomaly_detector(
            artifact=anomaly_artifact,
            test_frame=test_frame,
            label_column="label",
        )
        anomaly_predictions.to_csv(directories["predictions"] / "anomaly_test_scores.csv", index=False)
        _save_anomaly_metrics_tables(anomaly_metrics, directories)
        _save_anomaly_artifacts(anomaly_artifact, directories)
    except ValueError as error:
        anomaly_metrics = {"note": str(error)}
        save_json(directories["metrics"] / "anomaly_metrics.json", anomaly_metrics)
        _save_scalar_metrics_csv(anomaly_metrics, directories["metrics"] / "anomaly_scalar_metrics.csv")

    session_summary = _build_session_prediction_summary(classifier_predictions, anomaly_predictions)
    if not session_summary.empty:
        session_summary.to_csv(directories["predictions"] / "per_session_prediction_summary.csv", index=False)
        save_json(
            directories["predictions"] / "per_session_prediction_summary.json",
            {"rows": session_summary.to_dict(orient="records")},
        )

    plot_class_distribution(labelled_dataset, directories["plots"] / "class_distribution.png")
    plot_confusion_matrix(
        matrix=np.asarray(classifier_metrics["confusion_matrix"], dtype=int),
        labels=list(map(str, classifier_metrics["labels"])),
        output_path=directories["plots"] / "confusion_matrix.png",
        title="Random Forest Confusion Matrix",
    )

    top_feature_names = [
        item["feature"]
        for item in classifier_metrics.get("top_feature_importances", [])[: config.plots.top_n_feature_trends]
    ]
    if not top_feature_names:
        top_feature_names = feature_columns[: config.plots.top_n_feature_trends]
    plot_feature_trends(labelled_dataset, top_feature_names, directories["plots"] / "feature_trends.png")

    if not importance_frame.empty:
        plot_feature_importances(importance_frame, directories["plots"] / "feature_importances.png")

    if anomaly_predictions is not None and anomaly_artifact is not None:
        plot_anomaly_scores(
            anomaly_predictions,
            threshold=float(anomaly_artifact["threshold"]),
            output_path=directories["plots"] / "anomaly_scores.png",
        )

    summary = {
        "experiment_name": config.experiment_name,
        "results_dir": str(directories["root"]),
        "n_total_windows": int(len(labelled_dataset)),
        "n_train_windows": int(len(train_frame)),
        "n_test_windows": int(len(test_frame)),
        "train_sessions": split_summary["train_sessions"],
        "test_sessions": split_summary["test_sessions"],
        "split_strategy": split_summary["split_strategy"],
        "classifier_metrics_path": str(directories["metrics"] / "classifier_metrics.json"),
        "anomaly_metrics_path": str(directories["metrics"] / "anomaly_metrics.json"),
    }
    save_json(directories["root"] / "experiment_summary.json", summary)
    pd.DataFrame([summary]).to_csv(directories["root"] / "experiment_summary.csv", index=False)
    _write_experiment_summary_markdown(config, split_summary, classifier_metrics, anomaly_metrics, directories)
    return summary


def evaluate_saved_models(config: ExperimentConfig) -> Dict[str, object]:
    """Re-evaluate saved models on the stored train/test split."""

    directories = prepare_experiment_directories(config.paths.results_root, config.experiment_name)
    train_path = directories["datasets"] / "train_windows.csv"
    test_path = directories["datasets"] / "test_windows.csv"
    classifier_path = directories["artifacts"] / "classifier_artifact.joblib"
    if not train_path.exists() or not test_path.exists() or not classifier_path.exists():
        raise FileNotFoundError("Saved train/test split or classifier artifact not found. Run training first.")

    train_frame = pd.read_csv(train_path)
    test_frame = pd.read_csv(test_path)
    classifier_artifact = joblib.load(classifier_path)
    feature_columns = list(classifier_artifact["feature_columns"])

    split_summary = summarize_split(train_frame, test_frame, strategy=config.evaluation.split_strategy)
    classifier_metrics, classifier_predictions = evaluate_classifier(
        artifact=classifier_artifact,
        test_frame=test_frame,
        feature_columns=feature_columns,
        label_column="label",
        healthy_label=str(config.evaluation.healthy_label),
        positive_labels=list(config.evaluation.positive_labels),
    )
    classifier_predictions.to_csv(directories["predictions"] / "classifier_test_predictions.csv", index=False)
    _save_classifier_metrics_tables(classifier_metrics, directories)

    anomaly_metrics: Dict[str, object] = {}
    anomaly_path = directories["artifacts"] / "anomaly_artifact.joblib"
    anomaly_predictions: Optional[pd.DataFrame] = None
    if anomaly_path.exists():
        anomaly_artifact = joblib.load(anomaly_path)
        anomaly_metrics, anomaly_predictions = evaluate_anomaly_detector(
            artifact=anomaly_artifact,
            test_frame=test_frame,
            label_column="label",
        )
        anomaly_predictions.to_csv(directories["predictions"] / "anomaly_test_scores.csv", index=False)
        _save_anomaly_metrics_tables(anomaly_metrics, directories)
        plot_anomaly_scores(
            anomaly_predictions,
            threshold=float(anomaly_artifact["threshold"]),
            output_path=directories["plots"] / "anomaly_scores.png",
        )

    session_summary = _build_session_prediction_summary(classifier_predictions, anomaly_predictions)
    if not session_summary.empty:
        session_summary.to_csv(directories["predictions"] / "per_session_prediction_summary.csv", index=False)
        save_json(
            directories["predictions"] / "per_session_prediction_summary.json",
            {"rows": session_summary.to_dict(orient="records")},
        )

    plot_confusion_matrix(
        matrix=np.asarray(classifier_metrics["confusion_matrix"], dtype=int),
        labels=list(map(str, classifier_metrics["labels"])),
        output_path=directories["plots"] / "confusion_matrix.png",
        title="Random Forest Confusion Matrix",
    )
    importance_frame = pd.DataFrame(classifier_metrics.get("top_feature_importances", []))
    if not importance_frame.empty:
        importance_frame.to_csv(directories["metrics"] / "classifier_feature_importances.csv", index=False)
        plot_feature_importances(importance_frame, directories["plots"] / "feature_importances.png")

    summary = {
        "experiment_name": config.experiment_name,
        "results_dir": str(directories["root"]),
        "n_train_windows": int(len(train_frame)),
        "n_test_windows": int(len(test_frame)),
        "split_strategy": split_summary["split_strategy"],
        "classifier_metrics_path": str(directories["metrics"] / "classifier_metrics.json"),
        "anomaly_metrics_path": str(directories["metrics"] / "anomaly_metrics.json"),
        "anomaly_available": bool(anomaly_metrics),
    }
    save_json(directories["root"] / "evaluation_summary.json", summary)
    pd.DataFrame([summary]).to_csv(directories["root"] / "evaluation_summary.csv", index=False)
    _write_experiment_summary_markdown(config, split_summary, classifier_metrics, anomaly_metrics, directories)
    return summary


def run_inference(config: ExperimentConfig, session_dir: Optional[str | Path] = None) -> Dict[str, object]:
    """Run baseline inference on one unseen session directory."""

    directories = prepare_experiment_directories(config.paths.results_root, config.experiment_name)
    classifier_artifact = joblib.load(directories["artifacts"] / "classifier_artifact.joblib")
    anomaly_path = directories["artifacts"] / "anomaly_artifact.joblib"
    anomaly_artifact = joblib.load(anomaly_path) if anomaly_path.exists() else None

    target_session_dir = session_dir or config.paths.inference_session_dir
    if not target_session_dir:
        raise ValueError("No inference session directory was provided.")

    rows = build_feature_rows_for_session(target_session_dir, config)
    if not rows:
        raise RuntimeError("No windows were produced for the requested inference session.")

    inference_frame = pd.DataFrame(rows)
    feature_columns = list(classifier_artifact["feature_columns"])
    for column in feature_columns:
        if column not in inference_frame.columns:
            inference_frame[column] = np.nan

    classifier_metrics, classifier_predictions = evaluate_classifier(
        artifact=classifier_artifact,
        test_frame=inference_frame,
        feature_columns=feature_columns,
        label_column="label",
        healthy_label=str(config.evaluation.healthy_label),
        positive_labels=list(config.evaluation.positive_labels),
    )
    result_frame = classifier_predictions.copy()
    predictions = result_frame["predicted_label"].astype(str).tolist()

    summary: Dict[str, object] = {
        "session_id": str(inference_frame["session_id"].iloc[0]),
        "n_windows": int(len(inference_frame)),
        "predicted_class": Counter(predictions).most_common(1)[0][0],
        "majority_class": Counter(predictions).most_common(1)[0][0],
    }

    if "max_probability" in result_frame.columns:
        summary["mean_max_probability"] = float(result_frame["max_probability"].mean())

    anomaly_predictions: Optional[pd.DataFrame] = None
    if anomaly_artifact is not None:
        for column in anomaly_artifact["feature_columns"]:
            if column not in inference_frame.columns:
                inference_frame[column] = np.nan
        anomaly_metrics, anomaly_predictions = evaluate_anomaly_detector(
            artifact=anomaly_artifact,
            test_frame=inference_frame,
            label_column="label",
        )
        result_frame["anomaly_score"] = anomaly_predictions["anomaly_score"]
        result_frame["anomaly_flag"] = anomaly_predictions["anomaly_flag"]
        summary["anomaly_threshold"] = float(anomaly_artifact["threshold"])
        summary["mean_anomaly_score"] = float(anomaly_predictions["anomaly_score"].mean())
        summary["max_anomaly_score"] = float(anomaly_predictions["anomaly_score"].max())
        summary["anomalous_window_ratio"] = float(anomaly_predictions["anomaly_flag"].mean())
        save_json(directories["metrics"] / "last_inference_anomaly_metrics.json", anomaly_metrics)
    else:
        summary["anomalous_window_ratio"] = 0.0

    healthy_label = str(config.evaluation.healthy_label)
    classifier_trigger, anomaly_trigger, warning_reason, decision_summary = _resolve_warning_reason(
        predicted_class=str(summary["predicted_class"]),
        healthy_label=healthy_label,
        anomalous_window_ratio=float(summary["anomalous_window_ratio"]),
        warning_ratio_threshold=float(config.inference.warning_ratio_threshold),
    )
    summary["classifier_trigger"] = classifier_trigger
    summary["anomaly_ratio_trigger"] = anomaly_trigger
    summary["warning_ratio_threshold"] = float(config.inference.warning_ratio_threshold)
    summary["warning_trigger_reason"] = warning_reason
    summary["decision_summary"] = decision_summary
    summary["early_warning"] = bool(classifier_trigger or anomaly_trigger)
    summary["classifier_label_set"] = classifier_metrics.get("labels", [])

    session_id = str(summary["session_id"])
    result_frame.to_csv(directories["predictions"] / f"inference_windows_{session_id}.csv", index=False)
    session_summary = _build_session_prediction_summary(result_frame, anomaly_predictions)
    if not session_summary.empty:
        session_summary["predicted_class"] = str(summary["predicted_class"])
        session_summary["early_warning"] = bool(summary["early_warning"])
        session_summary["warning_trigger_reason"] = str(summary["warning_trigger_reason"])
        session_summary["decision_summary"] = str(summary["decision_summary"])
        session_summary.to_csv(directories["predictions"] / f"inference_session_summary_{session_id}.csv", index=False)
    save_json(directories["predictions"] / f"inference_summary_{session_id}.json", summary)
    return summary
