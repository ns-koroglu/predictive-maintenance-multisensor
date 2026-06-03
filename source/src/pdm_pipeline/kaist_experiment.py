"""KAIST baseline experiment runner using the compact feature dataset directly."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .datasets import get_dataset_metadata, standardize_feature_dataset
from .evaluation import select_feature_columns, split_dataset, summarize_split
from .models import (
    balance_training_frame,
    build_threshold_sweep,
    evaluate_anomaly_detector,
    evaluate_classifier,
    recommend_threshold_from_sweep,
    train_anomaly_detector,
    train_classifier,
    train_one_class_svm_detector,
)
from .pipeline import (
    _build_session_prediction_summary,
    _remove_legacy_result_files,
    _save_classifier_artifacts,
    _save_classifier_metrics_tables,
    _save_feature_columns,
    _save_scalar_metrics_csv,
    _save_split_artifacts,
)
from .plots import (
    plot_anomaly_baseline_comparison,
    plot_anomaly_scores,
    plot_class_distribution,
    plot_classifier_failure_overview,
    plot_confusion_matrix,
    plot_feature_importances,
    plot_threshold_sweep,
)
from .utils import ensure_directory, prepare_experiment_directories, save_json, save_yaml


def _load_kaist_feature_dataset(feature_dataset_path: str | Path, healthy_label: str) -> pd.DataFrame:
    """Load and validate the compact KAIST feature dataset."""

    dataset_path = Path(feature_dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"KAIST feature dataset not found: {dataset_path}")

    frame = pd.read_csv(dataset_path)
    required_columns = {"session_id", "label"}
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ValueError(f"KAIST feature dataset is missing required columns: {missing}")

    extra_columns: List[pd.Series] = []
    if "window_start" not in frame.columns and "start_time" in frame.columns:
        extra_columns.append(frame["start_time"].rename("window_start"))
    if "window_end" not in frame.columns and "end_time" in frame.columns:
        extra_columns.append(frame["end_time"].rename("window_end"))
    if extra_columns:
        frame = pd.concat([frame, *extra_columns], axis=1).copy()

    frame["label"] = np.where(
        frame["label"].astype(str).str.lower() == str(healthy_label).lower(),
        str(healthy_label),
        "faulty",
    )
    metadata = get_dataset_metadata("kaist_rotating_machine")
    frame = standardize_feature_dataset(
        frame=frame,
        metadata=metadata,
        dataset_variant="compact_vibration_thermal_features",
    )
    frame = frame.sort_values(
        [column for column in ["session_id", "window_start", "window_index"] if column in frame.columns]
    ).reset_index(drop=True)
    return frame


def _add_publication_directories(directories: Dict[str, Path]) -> Dict[str, Path]:
    """Add tables and figures directories used by the publication-ready analysis layer."""

    enriched = dict(directories)
    enriched["tables"] = ensure_directory(directories["root"] / "tables")
    enriched["figures"] = ensure_directory(directories["root"] / "figures")
    return enriched


def _format_table_value(value: object) -> str:
    """Format table values for compact publication-style markdown output."""

    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if np.isnan(float(value)):
            return ""
        return f"{float(value):.4f}"
    return str(value).replace("|", "\\|")


def _frame_to_markdown(frame: pd.DataFrame) -> str:
    """Convert a small DataFrame into a markdown table without extra dependencies."""

    if frame.empty:
        return "_No rows._\n"

    columns = list(map(str, frame.columns))
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        values = [_format_table_value(row[column]) for column in frame.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _save_publication_table(frame: pd.DataFrame, csv_path: Path, markdown_path: Path) -> None:
    """Save a publication-ready table as both CSV and Markdown."""

    frame.to_csv(csv_path, index=False)
    markdown_path.write_text(_frame_to_markdown(frame), encoding="utf-8")


def _save_named_anomaly_outputs(
    model_name: str,
    artifact: Dict[str, object],
    metrics: Dict[str, object],
    predictions: pd.DataFrame,
    directories: Dict[str, Path],
    save_as_default: bool,
) -> None:
    """Persist anomaly artifacts and metrics with model-specific filenames."""

    metrics_prefix = directories["metrics"] / model_name
    save_json(Path(f"{metrics_prefix}_metrics.json"), metrics)
    _save_scalar_metrics_csv(metrics, Path(f"{metrics_prefix}_scalar_metrics.csv"))
    if "confusion_matrix" in metrics:
        pd.DataFrame(
            metrics["confusion_matrix"],
            index=metrics["labels"],
            columns=metrics["labels"],
        ).to_csv(Path(f"{metrics_prefix}_confusion_matrix.csv"))
    if "calibration" in metrics:
        save_json(Path(f"{metrics_prefix}_calibration.json"), metrics["calibration"])
        pd.DataFrame([metrics["calibration"]]).to_csv(Path(f"{metrics_prefix}_calibration.csv"), index=False)

    predictions.to_csv(directories["predictions"] / f"{model_name}_test_scores.csv", index=False)
    joblib.dump(artifact, directories["artifacts"] / f"{model_name}_artifact.joblib")
    joblib.dump(artifact["preprocessor"], directories["artifacts"] / f"{model_name}_preprocessor.joblib")
    joblib.dump(artifact["model"], directories["models"] / f"{model_name}_model.joblib")

    if save_as_default:
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
        joblib.dump(artifact, directories["artifacts"] / "anomaly_artifact.joblib")
        joblib.dump(artifact["preprocessor"], directories["artifacts"] / "anomaly_preprocessor.joblib")
        joblib.dump(artifact["model"], directories["models"] / "anomaly_model.joblib")


def _select_primary_anomaly_result(anomaly_results: List[Dict[str, object]]) -> Dict[str, object]:
    """Choose the anomaly baseline used for the default report files."""

    successful = [result for result in anomaly_results if result.get("status") == "ok"]
    if not successful:
        raise RuntimeError("No anomaly baseline could be trained with the available healthy training windows.")

    for preferred_name in ("isolation_forest",):
        for result in successful:
            if str(result["model_name"]) == preferred_name:
                return result

    successful.sort(
        key=lambda result: (
            float(result["metrics"].get("balanced_accuracy", -1.0)),
            float(result["metrics"].get("f1", -1.0)),
            float(result["metrics"].get("pr_auc", -1.0)),
        ),
        reverse=True,
    )
    return successful[0]


def _build_anomaly_comparison_frame(
    anomaly_results: List[Dict[str, object]],
    selected_model_name: str,
) -> pd.DataFrame:
    """Create a compact comparison table for anomaly baselines."""

    rows: List[Dict[str, object]] = []
    for result in anomaly_results:
        if result["status"] != "ok":
            rows.append(
                {
                    "model_name": result["model_name"],
                    "status": "failed",
                    "selected_for_reporting": False,
                    "note": result.get("note", ""),
                }
            )
            continue

        metrics = result["metrics"]
        recommendation = result["threshold_recommendation"]
        rows.append(
            {
                "model_name": result["model_name"],
                "status": "ok",
                "selected_for_reporting": bool(str(result["model_name"]) == str(selected_model_name)),
                "threshold": float(metrics["threshold"]),
                "recommended_threshold": float(recommendation["recommended_threshold"]),
                "threshold_selection_rule": recommendation["selection_rule"],
                "n_healthy_train_windows": int(metrics["n_healthy_train_windows"]),
                "accuracy": float(metrics.get("accuracy", np.nan)),
                "balanced_accuracy": float(metrics.get("balanced_accuracy", np.nan)),
                "precision": float(metrics.get("precision", np.nan)),
                "recall": float(metrics.get("recall", np.nan)),
                "f1": float(metrics.get("f1", np.nan)),
                "roc_auc": float(metrics.get("roc_auc", np.nan)) if "roc_auc" in metrics else np.nan,
                "pr_auc": float(metrics.get("pr_auc", np.nan)) if "pr_auc" in metrics else np.nan,
                "recommended_precision": float(recommendation.get("recommended_precision", np.nan)),
                "recommended_recall": float(recommendation.get("recommended_recall", np.nan)),
                "recommended_f1": float(recommendation.get("recommended_f1", np.nan)),
                "recommended_balanced_accuracy": float(
                    recommendation.get("recommended_balanced_accuracy", np.nan)
                ),
            }
        )
    return pd.DataFrame(rows)


def _get_anomaly_result(
    anomaly_results: Sequence[Dict[str, object]],
    model_name: str,
) -> Optional[Dict[str, object]]:
    """Return one anomaly result by name when available."""

    for result in anomaly_results:
        if result.get("status") == "ok" and str(result.get("model_name")) == str(model_name):
            return result
    return None


def _evaluate_anomaly_models(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_columns: List[str],
    config: ExperimentConfig,
) -> List[Dict[str, object]]:
    """Train and evaluate all anomaly baselines for one train/test split."""

    anomaly_results: List[Dict[str, object]] = []
    anomaly_trainers = {
        "isolation_forest": train_anomaly_detector,
        "one_class_svm": train_one_class_svm_detector,
    }
    for model_name, train_fn in anomaly_trainers.items():
        try:
            artifact = train_fn(
                train_frame=train_frame,
                feature_columns=feature_columns,
                label_column="label",
                healthy_label=config.evaluation.healthy_label,
                config=config.model.anomaly,
            )
            metrics, predictions = evaluate_anomaly_detector(
                artifact=artifact,
                test_frame=test_frame,
                label_column="label",
            )
            sweep_frame = build_threshold_sweep(
                prediction_frame=predictions,
                model_name=model_name,
                calibrated_threshold=float(metrics["threshold"]),
                threshold_points=config.model.anomaly.threshold_sweep_points,
            )
            threshold_recommendation = recommend_threshold_from_sweep(
                sweep_frame=sweep_frame,
                calibrated_threshold=float(metrics["threshold"]),
            )
            metrics["threshold_analysis"] = threshold_recommendation
            anomaly_results.append(
                {
                    "status": "ok",
                    "model_name": model_name,
                    "artifact": artifact,
                    "metrics": metrics,
                    "predictions": predictions,
                    "sweep": sweep_frame,
                    "threshold_recommendation": threshold_recommendation,
                }
            )
        except ValueError as error:
            anomaly_results.append(
                {
                    "status": "failed",
                    "model_name": model_name,
                    "note": str(error),
                }
            )
    return anomaly_results


def _evaluate_kaist_split(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_columns: List[str],
    config: ExperimentConfig,
) -> Dict[str, object]:
    """Evaluate classifier and anomaly baselines for one fixed split."""

    balanced_train_frame, balance_summary = balance_training_frame(
        train_frame=train_frame,
        label_column="label",
        config=config.model.classifier,
    )
    classifier_artifact = train_classifier(
        train_frame=balanced_train_frame,
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
    importance_frame = pd.DataFrame(classifier_metrics.get("top_feature_importances", []))

    anomaly_results = _evaluate_anomaly_models(
        train_frame=train_frame,
        test_frame=test_frame,
        feature_columns=feature_columns,
        config=config,
    )
    primary_anomaly_result = _select_primary_anomaly_result(anomaly_results)
    anomaly_comparison = _build_anomaly_comparison_frame(
        anomaly_results=anomaly_results,
        selected_model_name=str(primary_anomaly_result["model_name"]),
    )
    threshold_sweeps = [
        result["sweep"]
        for result in anomaly_results
        if result.get("status") == "ok" and isinstance(result.get("sweep"), pd.DataFrame)
    ]
    threshold_sweep_frame = pd.concat(threshold_sweeps, ignore_index=True) if threshold_sweeps else pd.DataFrame()

    return {
        "balanced_train_frame": balanced_train_frame,
        "balance_summary": balance_summary,
        "classifier_artifact": classifier_artifact,
        "classifier_metrics": classifier_metrics,
        "classifier_predictions": classifier_predictions,
        "importance_frame": importance_frame,
        "anomaly_results": anomaly_results,
        "anomaly_comparison": anomaly_comparison,
        "threshold_sweep": threshold_sweep_frame,
        "primary_anomaly_result": primary_anomaly_result,
    }


def _build_classifier_metrics_table(
    classifier_metrics: Dict[str, object],
    balance_summary: Dict[str, object],
    test_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Build a clean single-row classifier metrics table."""

    before_counts = {str(key): int(value) for key, value in balance_summary.get("before_counts", {}).items()}
    after_counts = {str(key): int(value) for key, value in balance_summary.get("after_counts", {}).items()}
    test_counts = test_frame["label"].astype(str).value_counts().to_dict()

    return pd.DataFrame(
        [
            {
                "model_name": "random_forest_classifier",
                "train_balance_strategy": balance_summary.get("strategy", "none"),
                "train_faulty_before": before_counts.get("faulty", 0),
                "train_healthy_before": before_counts.get("healthy", 0),
                "train_faulty_after": after_counts.get("faulty", 0),
                "train_healthy_after": after_counts.get("healthy", 0),
                "test_faulty_windows": int(test_counts.get("faulty", 0)),
                "test_healthy_windows": int(test_counts.get("healthy", 0)),
                "accuracy": float(classifier_metrics.get("accuracy", np.nan)),
                "balanced_accuracy": float(classifier_metrics.get("balanced_accuracy", np.nan)),
                "macro_precision": float(classifier_metrics.get("precision_macro", np.nan)),
                "macro_recall": float(classifier_metrics.get("recall_macro", np.nan)),
                "macro_f1": float(classifier_metrics.get("f1_macro", np.nan)),
                "roc_auc": float(classifier_metrics.get("roc_auc", np.nan)),
                "pr_auc": float(classifier_metrics.get("pr_auc", np.nan)),
            }
        ]
    )


def _build_anomaly_metrics_table(selected_anomaly_result: Dict[str, object]) -> pd.DataFrame:
    """Build a single-row anomaly table that separates calibrated and sweep-based thresholds."""

    metrics = selected_anomaly_result["metrics"]
    recommendation = selected_anomaly_result["threshold_recommendation"]
    return pd.DataFrame(
        [
            {
                "reported_model_name": selected_anomaly_result["model_name"],
                "n_healthy_train_windows": int(metrics.get("n_healthy_train_windows", 0)),
                "calibrated_threshold": float(metrics.get("threshold", np.nan)),
                "calibrated_precision": float(metrics.get("precision", np.nan)),
                "calibrated_recall": float(metrics.get("recall", np.nan)),
                "calibrated_f1": float(metrics.get("f1", np.nan)),
                "calibrated_balanced_accuracy": float(metrics.get("balanced_accuracy", np.nan)),
                "calibrated_roc_auc": float(metrics.get("roc_auc", np.nan)),
                "calibrated_pr_auc": float(metrics.get("pr_auc", np.nan)),
                "recommended_threshold": float(recommendation.get("recommended_threshold", np.nan)),
                "recommended_precision": float(recommendation.get("recommended_precision", np.nan)),
                "recommended_recall": float(recommendation.get("recommended_recall", np.nan)),
                "recommended_f1": float(recommendation.get("recommended_f1", np.nan)),
                "recommended_balanced_accuracy": float(
                    recommendation.get("recommended_balanced_accuracy", np.nan)
                ),
                "threshold_note": "Recommended threshold is optimistic because it is chosen from the held-out sweep.",
            }
        ]
    )


def _build_threshold_sweep_summary_table(anomaly_comparison: pd.DataFrame) -> pd.DataFrame:
    """Build a compact table summarizing calibrated versus sweep-selected thresholds."""

    frame = anomaly_comparison.copy()
    if frame.empty:
        return frame

    keep_columns = {
        "model_name": "model_name",
        "threshold": "calibrated_threshold",
        "balanced_accuracy": "calibrated_balanced_accuracy",
        "precision": "calibrated_precision",
        "recall": "calibrated_recall",
        "f1": "calibrated_f1",
        "recommended_threshold": "recommended_threshold",
        "recommended_balanced_accuracy": "recommended_balanced_accuracy",
        "recommended_precision": "recommended_precision",
        "recommended_recall": "recommended_recall",
        "recommended_f1": "recommended_f1",
        "threshold_selection_rule": "selection_rule",
        "selected_for_reporting": "selected_for_reporting",
    }
    cleaned = frame.loc[frame["status"].astype(str) == "ok", list(keep_columns.keys())].rename(columns=keep_columns)
    return cleaned.reset_index(drop=True)


def _build_model_comparison_table(
    classifier_metrics: Dict[str, object],
    anomaly_comparison: pd.DataFrame,
    selected_model_name: str,
) -> pd.DataFrame:
    """Build a compact comparison table across classifier and anomaly baselines."""

    rows: List[Dict[str, object]] = [
        {
            "approach": "random_forest_classifier",
            "task_type": "supervised_classification",
            "operating_point": "class_weighted_train_time_balancing",
            "default_reported": False,
            "accuracy": float(classifier_metrics.get("accuracy", np.nan)),
            "balanced_accuracy": float(classifier_metrics.get("balanced_accuracy", np.nan)),
            "precision": float(classifier_metrics.get("precision_macro", np.nan)),
            "recall": float(classifier_metrics.get("recall_macro", np.nan)),
            "f1": float(classifier_metrics.get("f1_macro", np.nan)),
            "roc_auc": float(classifier_metrics.get("roc_auc", np.nan)),
            "pr_auc": float(classifier_metrics.get("pr_auc", np.nan)),
            "metric_note": "Precision, recall, and F1 are macro-averaged.",
        }
    ]

    for _, row in anomaly_comparison.loc[anomaly_comparison["status"].astype(str) == "ok"].iterrows():
        rows.append(
            {
                "approach": str(row["model_name"]),
                "task_type": "healthy_only_anomaly_detection",
                "operating_point": "calibrated_train_only_threshold",
                "default_reported": bool(str(row["model_name"]) == str(selected_model_name)),
                "accuracy": float(row["accuracy"]),
                "balanced_accuracy": float(row["balanced_accuracy"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1": float(row["f1"]),
                "roc_auc": float(row["roc_auc"]),
                "pr_auc": float(row["pr_auc"]),
                "metric_note": "Precision, recall, and F1 are binary anomaly metrics.",
            }
        )
    return pd.DataFrame(rows)


def _build_healthy_holdout_frames(
    dataset: pd.DataFrame,
    healthy_session_id: str,
    test_fraction: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Build one deterministic session-safe split with a specific healthy session held out."""

    session_labels = dataset[["session_id", "label"]].drop_duplicates().copy()
    all_sessions = session_labels["session_id"].astype(str).tolist()
    faulty_sessions = sorted(
        session_labels.loc[session_labels["label"].astype(str) != "healthy", "session_id"].astype(str).tolist()
    )
    if healthy_session_id not in set(all_sessions):
        raise ValueError(f"Requested healthy holdout session was not found: {healthy_session_id}")
    if len(faulty_sessions) < 2:
        raise ValueError("Split sensitivity requires at least two faulty sessions.")

    rng = np.random.default_rng(random_state)
    shuffled_faulty = faulty_sessions.copy()
    rng.shuffle(shuffled_faulty)
    n_faulty_test = max(1, int(round(len(shuffled_faulty) * test_fraction)))
    n_faulty_test = min(n_faulty_test, len(shuffled_faulty) - 1)
    test_faulty_sessions = sorted(shuffled_faulty[:n_faulty_test])

    test_sessions = sorted([healthy_session_id, *test_faulty_sessions])
    train_sessions = sorted(set(all_sessions).difference(test_sessions))

    train_frame = dataset[dataset["session_id"].astype(str).isin(train_sessions)].copy()
    test_frame = dataset[dataset["session_id"].astype(str).isin(test_sessions)].copy()
    split_summary = summarize_split(train_frame, test_frame, strategy="session")
    split_summary["held_out_healthy_session"] = str(healthy_session_id)
    return train_frame, test_frame, split_summary


def _run_split_sensitivity_analysis(
    dataset: pd.DataFrame,
    feature_columns: List[str],
    config: ExperimentConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate multiple session-safe splits while rotating the held-out healthy session."""

    healthy_sessions = sorted(
        dataset.loc[
            dataset["label"].astype(str).str.lower() == str(config.evaluation.healthy_label).lower(),
            "session_id",
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    per_split_rows: List[Dict[str, object]] = []
    for index, healthy_session_id in enumerate(healthy_sessions):
        train_frame, test_frame, split_summary = _build_healthy_holdout_frames(
            dataset=dataset,
            healthy_session_id=healthy_session_id,
            test_fraction=config.evaluation.test_fraction,
            random_state=config.evaluation.random_state + index,
        )
        split_result = _evaluate_kaist_split(
            train_frame=train_frame,
            test_frame=test_frame,
            feature_columns=feature_columns,
            config=config,
        )

        classifier_metrics = split_result["classifier_metrics"]
        isolation_forest_result = _get_anomaly_result(split_result["anomaly_results"], "isolation_forest")
        one_class_svm_result = _get_anomaly_result(split_result["anomaly_results"], "one_class_svm")

        row: Dict[str, object] = {
            "held_out_healthy_session": healthy_session_id,
            "n_train_sessions": int(split_summary["n_train_sessions"]),
            "n_test_sessions": int(split_summary["n_test_sessions"]),
            "n_train_windows": int(split_summary["n_train_windows"]),
            "n_test_windows": int(split_summary["n_test_windows"]),
            "classifier_accuracy": float(classifier_metrics.get("accuracy", np.nan)),
            "classifier_balanced_accuracy": float(classifier_metrics.get("balanced_accuracy", np.nan)),
            "classifier_macro_f1": float(classifier_metrics.get("f1_macro", np.nan)),
        }

        for model_name, result in (
            ("isolation_forest", isolation_forest_result),
            ("one_class_svm", one_class_svm_result),
        ):
            if result is None:
                row[f"{model_name}_balanced_accuracy"] = np.nan
                row[f"{model_name}_f1"] = np.nan
                row[f"{model_name}_recommended_balanced_accuracy"] = np.nan
                row[f"{model_name}_recommended_f1"] = np.nan
                continue

            metrics = result["metrics"]
            recommendation = result["threshold_recommendation"]
            row[f"{model_name}_balanced_accuracy"] = float(metrics.get("balanced_accuracy", np.nan))
            row[f"{model_name}_f1"] = float(metrics.get("f1", np.nan))
            row[f"{model_name}_recommended_balanced_accuracy"] = float(
                recommendation.get("recommended_balanced_accuracy", np.nan)
            )
            row[f"{model_name}_recommended_f1"] = float(recommendation.get("recommended_f1", np.nan))

        per_split_rows.append(row)

    per_split_frame = pd.DataFrame(per_split_rows)
    metric_columns = [
        "classifier_accuracy",
        "classifier_balanced_accuracy",
        "classifier_macro_f1",
        "isolation_forest_balanced_accuracy",
        "isolation_forest_f1",
        "isolation_forest_recommended_balanced_accuracy",
        "isolation_forest_recommended_f1",
        "one_class_svm_balanced_accuracy",
        "one_class_svm_f1",
        "one_class_svm_recommended_balanced_accuracy",
        "one_class_svm_recommended_f1",
    ]

    summary_rows: List[Dict[str, object]] = []
    for metric_name in metric_columns:
        if metric_name not in per_split_frame.columns:
            continue
        values = pd.to_numeric(per_split_frame[metric_name], errors="coerce").dropna()
        if values.empty:
            continue
        summary_rows.append(
            {
                "metric": metric_name,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    return per_split_frame, pd.DataFrame(summary_rows)


def _write_kaist_experiment_summary_markdown(
    config: ExperimentConfig,
    feature_dataset_path: Path,
    split_summary: Dict[str, object],
    test_frame: pd.DataFrame,
    balance_summary: Dict[str, object],
    classifier_metrics: Dict[str, object],
    classifier_predictions: pd.DataFrame,
    anomaly_comparison: pd.DataFrame,
    selected_anomaly_result: Dict[str, object],
    split_sensitivity_summary: pd.DataFrame,
    directories: Dict[str, Path],
) -> None:
    """Write a concise experiment summary with conservative interpretation."""

    test_counts = test_frame["label"].astype(str).value_counts()
    naive_accuracy = float(test_counts.max() / len(test_frame)) if len(test_frame) else 0.0
    predicted_counts = classifier_predictions["predicted_label"].astype(str).value_counts().to_dict()
    selected_metrics = selected_anomaly_result["metrics"]
    selected_recommendation = selected_anomaly_result.get("threshold_recommendation", {})

    def _metric_range(metric_name: str) -> str:
        row = split_sensitivity_summary.loc[split_sensitivity_summary["metric"] == metric_name]
        if row.empty:
            return "n/a"
        return f"{float(row.iloc[0]['min']):.4f} to {float(row.iloc[0]['max']):.4f}"

    lines = [
        f"# Experiment Summary: {config.experiment_name}",
        "",
        "## Input",
        f"- Feature dataset: `{feature_dataset_path}`",
        "- Baseline modalities: `vibration + thermal`",
        "- Current: exported and preserved in compact KAIST metadata, but not used in this first baseline",
        "- Session semantics: condition-matched and explicitly unsynchronized",
        "",
        "## Split",
        f"- Split strategy: `{split_summary['split_strategy']}`",
        f"- Train windows: {split_summary['n_train_windows']}",
        f"- Test windows: {split_summary['n_test_windows']}",
        f"- Train sessions: {', '.join(split_summary['train_sessions']) or 'None'}",
        f"- Test sessions: {', '.join(split_summary['test_sessions']) or 'None'}",
        f"- Session overlap: {split_summary['session_overlap'] or 'none'}",
        "",
        "## Class Imbalance",
        f"- Train label counts before balancing: {balance_summary.get('before_counts', {})}",
        f"- Train label counts used by the classifier: {balance_summary.get('after_counts', {})}",
        f"- Train-only balancing strategy: `{balance_summary.get('strategy', 'none')}`",
        f"- Majority-class test accuracy baseline: {naive_accuracy:.4f}",
        "- Raw accuracy is misleading here because most KAIST windows are faulty. A classifier can look strong by predicting only `faulty`, while still achieving zero healthy-class recall.",
        "",
        "## Classification",
        f"- Accuracy: {classifier_metrics.get('accuracy', 'n/a')}",
        f"- Balanced accuracy: {classifier_metrics.get('balanced_accuracy', 'n/a')}",
        f"- Macro precision: {classifier_metrics.get('precision_macro', 'n/a')}",
        f"- Macro recall: {classifier_metrics.get('recall_macro', 'n/a')}",
        f"- Macro F1: {classifier_metrics.get('f1_macro', 'n/a')}",
        f"- ROC-AUC: {classifier_metrics.get('roc_auc', 'n/a')}",
        f"- PR-AUC: {classifier_metrics.get('pr_auc', 'n/a')}",
        f"- Predicted label counts on the test set: {predicted_counts}",
        "",
        "## Anomaly Baselines",
    ]

    for _, row in anomaly_comparison.iterrows():
        if row.get("status") != "ok":
            lines.append(f"- {row['model_name']}: failed ({row.get('note', 'unknown error')})")
            continue
        lines.append(
            "- "
            f"{row['model_name']}: balanced_accuracy={row['balanced_accuracy']:.4f}, "
            f"precision={row['precision']:.4f}, recall={row['recall']:.4f}, f1={row['f1']:.4f}, "
            f"threshold={row['threshold']:.6f}, recommended_threshold={row['recommended_threshold']:.6f}"
        )

    lines.extend(
        [
            "",
            "## Selected Anomaly Baseline",
            f"- Default reported anomaly baseline: `{selected_anomaly_result['model_name']}`",
            "- Isolation Forest remains the default reported anomaly baseline for conservative interpretation.",
            f"- Calibrated threshold: {selected_metrics.get('threshold', 'n/a')}",
            f"- Recommended threshold from the sweep: {selected_recommendation.get('recommended_threshold', 'n/a')}",
            "- The recommended threshold is an evaluation-side analysis aid from the held-out sweep. The calibrated threshold remains the train-only operating point.",
            "",
            "## Split Sensitivity",
            f"- Classifier balanced accuracy range across healthy holdout splits: {_metric_range('classifier_balanced_accuracy')}",
            f"- Isolation Forest calibrated balanced accuracy range: {_metric_range('isolation_forest_balanced_accuracy')}",
            f"- Isolation Forest sweep-optimized balanced accuracy range: {_metric_range('isolation_forest_recommended_balanced_accuracy')}",
        ]
    )

    (directories["root"] / "experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_publication_results_summary(
    dataset: pd.DataFrame,
    classifier_metrics: Dict[str, object],
    selected_anomaly_result: Dict[str, object],
    split_sensitivity_summary: pd.DataFrame,
    directories: Dict[str, Path],
) -> None:
    """Write a concise manuscript-ready interpretation of the KAIST results."""

    label_counts = dataset["label"].astype(str).value_counts().to_dict()
    selected_metrics = selected_anomaly_result["metrics"]
    selected_recommendation = selected_anomaly_result["threshold_recommendation"]

    def _summary_value(metric_name: str, statistic: str) -> str:
        row = split_sensitivity_summary.loc[split_sensitivity_summary["metric"] == metric_name]
        if row.empty:
            return "n/a"
        return f"{float(row.iloc[0][statistic]):.4f}"

    lines = [
        "# Publication-Ready Results Summary",
        "",
        "## Problem Setting",
        "This analysis evaluates a thesis-oriented predictive maintenance baseline on the compact KAIST rotating machine feature dataset.",
        "The first integration baseline uses vibration and thermal features only. Sessions are condition-matched rather than safely time-synchronous, so all evaluation remains session-aware and leakage-safe.",
        "",
        "## Class Imbalance",
        f"The dataset contains {dataset['session_id'].nunique()} sessions and a binary label distribution of {label_counts}.",
        "Only three sessions are healthy, which makes generalization to unseen healthy conditions difficult and makes any window-level accuracy figure potentially deceptive.",
        "",
        "## Why Raw Accuracy Is Misleading",
        f"The supervised classifier reached an apparent accuracy of {float(classifier_metrics.get('accuracy', np.nan)):.4f}.",
        "However, this is nearly identical to the majority-class baseline because the test set is dominated by faulty windows.",
        f"The more informative metrics are balanced accuracy ({float(classifier_metrics.get('balanced_accuracy', np.nan)):.4f}) and macro F1 ({float(classifier_metrics.get('f1_macro', np.nan)):.4f}), which show that the classifier does not generalize to the healthy class.",
        "",
        "## Classifier Failure Case",
        "Under the current session-safe split, the binary Random Forest predicts every test window as faulty.",
        "This yields zero healthy-class recall despite train-time downsampling of the faulty class.",
        "The supervised result should therefore be interpreted as a failure case under extreme class imbalance, not as a strong baseline.",
        "",
        "## Anomaly-First Interpretation",
        "Healthy-only anomaly detection is more informative in this setting because it matches the small-healthy-data constraint more naturally.",
        "The default reported anomaly baseline remains Isolation Forest for conservative reporting.",
        f"Its calibrated operating point gives balanced accuracy {float(selected_metrics.get('balanced_accuracy', np.nan)):.4f}, precision {float(selected_metrics.get('precision', np.nan)):.4f}, recall {float(selected_metrics.get('recall', np.nan)):.4f}, and F1 {float(selected_metrics.get('f1', np.nan)):.4f}.",
        "One-Class SVM is also reported for comparison, but it is not used as the default headline result because its calibration is more sensitive to the tiny healthy training set.",
        "",
        "## Threshold Calibration Interpretation",
        f"For the default Isolation Forest baseline, the train-only calibrated threshold is {float(selected_metrics.get('threshold', np.nan)):.6f}.",
        f"The held-out sweep identifies a more favorable analysis threshold of {float(selected_recommendation.get('recommended_threshold', np.nan)):.6f}, with higher balanced accuracy {float(selected_recommendation.get('recommended_balanced_accuracy', np.nan)):.4f}.",
        "This sweep-selected threshold is intentionally reported as optimistic analysis only. It must not be conflated with the train-only calibrated operating point.",
        "",
        "## Split-Sensitivity Analysis",
        "A small split-sensitivity study rotates the held-out healthy session across all three healthy sessions while preserving session-safe splitting.",
        f"Across these splits, classifier balanced accuracy ranges from {_summary_value('classifier_balanced_accuracy', 'min')} to {_summary_value('classifier_balanced_accuracy', 'max')}.",
        f"Isolation Forest calibrated balanced accuracy ranges from {_summary_value('isolation_forest_balanced_accuracy', 'min')} to {_summary_value('isolation_forest_balanced_accuracy', 'max')}, while the optimistic sweep-selected value ranges from {_summary_value('isolation_forest_recommended_balanced_accuracy', 'min')} to {_summary_value('isolation_forest_recommended_balanced_accuracy', 'max')}.",
        "This reinforces the same conclusion: threshold choice matters, and any anomaly headline result should be interpreted conservatively.",
        "",
        "## Conservative Conclusion",
        "The current KAIST baseline does not support a reliable supervised healthy-vs-faulty classifier under the present session-safe split and class distribution.",
        "The anomaly-first view is more defensible, especially when reported with explicit separation between calibrated operating points and optimistic sweep-based analysis.",
    ]

    (directories["root"] / "publication_results_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_kaist_baseline_experiment(config: ExperimentConfig) -> Dict[str, object]:
    """Train and evaluate the KAIST baseline directly from the compact feature dataset."""

    feature_dataset_path = config.paths.feature_dataset_path
    if not feature_dataset_path:
        raise ValueError("KAIST experiment requires paths.feature_dataset_path in the config.")

    directories = _add_publication_directories(
        prepare_experiment_directories(config.paths.results_root, config.experiment_name)
    )
    _remove_legacy_result_files(directories)
    save_yaml(directories["root"] / "config_snapshot.yaml", config.to_dict())

    dataset = _load_kaist_feature_dataset(feature_dataset_path, healthy_label=config.evaluation.healthy_label)
    dataset.to_csv(directories["datasets"] / "kaist_feature_dataset.csv", index=False)

    feature_columns = select_feature_columns(dataset)
    if not feature_columns:
        raise RuntimeError("No numeric feature columns were found in the KAIST feature dataset.")
    _save_feature_columns(feature_columns, directories)

    train_frame, test_frame = split_dataset(
        dataset,
        strategy=config.evaluation.split_strategy,
        test_fraction=config.evaluation.test_fraction,
        random_state=config.evaluation.random_state,
    )
    if train_frame.empty or test_frame.empty:
        raise RuntimeError("KAIST train/test split failed. Adjust the split fraction or provide more sessions.")

    split_summary = summarize_split(train_frame, test_frame, strategy=config.evaluation.split_strategy)
    if config.evaluation.split_strategy == "session" and split_summary["session_overlap"]:
        raise RuntimeError(f"Session leakage detected: {split_summary['session_overlap']}")
    _save_split_artifacts(train_frame, test_frame, split_summary, directories)

    train_frame.to_csv(directories["datasets"] / "train_windows.csv", index=False)
    test_frame.to_csv(directories["datasets"] / "test_windows.csv", index=False)

    split_result = _evaluate_kaist_split(
        train_frame=train_frame,
        test_frame=test_frame,
        feature_columns=feature_columns,
        config=config,
    )
    balanced_train_frame = split_result["balanced_train_frame"]
    balance_summary = split_result["balance_summary"]
    classifier_artifact = split_result["classifier_artifact"]
    classifier_metrics = split_result["classifier_metrics"]
    classifier_predictions = split_result["classifier_predictions"]
    importance_frame = split_result["importance_frame"]
    anomaly_results = split_result["anomaly_results"]
    anomaly_comparison = split_result["anomaly_comparison"]
    threshold_sweep_frame = split_result["threshold_sweep"]
    primary_anomaly_result = split_result["primary_anomaly_result"]
    selected_anomaly_metrics = primary_anomaly_result["metrics"]
    selected_anomaly_predictions = primary_anomaly_result["predictions"]

    save_json(directories["metrics"] / "classifier_balance_summary.json", balance_summary)
    pd.DataFrame([balance_summary]).to_csv(directories["metrics"] / "classifier_balance_summary.csv", index=False)
    if config.model.classifier.balance_strategy != "none":
        balanced_train_frame.to_csv(directories["datasets"] / "classifier_train_windows_balanced.csv", index=False)

    classifier_predictions.to_csv(directories["predictions"] / "classifier_test_predictions.csv", index=False)
    _save_classifier_metrics_tables(classifier_metrics, directories)
    _save_classifier_artifacts(classifier_artifact, directories)

    if not importance_frame.empty:
        importance_frame.to_csv(directories["metrics"] / "classifier_feature_importances.csv", index=False)

    for result in anomaly_results:
        if result["status"] != "ok":
            continue
        _save_named_anomaly_outputs(
            model_name=str(result["model_name"]),
            artifact=result["artifact"],
            metrics=result["metrics"],
            predictions=result["predictions"],
            directories=directories,
            save_as_default=bool(str(result["model_name"]) == str(primary_anomaly_result["model_name"])),
        )

    anomaly_comparison.to_csv(directories["root"] / "anomaly_baseline_comparison.csv", index=False)
    save_json(
        directories["root"] / "anomaly_baseline_comparison.json",
        {"rows": anomaly_comparison.replace({np.nan: None}).to_dict(orient="records")},
    )

    if not threshold_sweep_frame.empty:
        threshold_sweep_frame.to_csv(directories["root"] / "threshold_sweep.csv", index=False)
        save_json(
            directories["root"] / "threshold_sweep.json",
            {"rows": threshold_sweep_frame.replace({np.nan: None}).to_dict(orient="records")},
        )
        plot_threshold_sweep(threshold_sweep_frame, directories["plots"] / "threshold_sweep.png")

    save_json(
        directories["metrics"] / "selected_anomaly_model.json",
        {
            "model_name": primary_anomaly_result["model_name"],
            "threshold": float(selected_anomaly_metrics["threshold"]),
            "threshold_recommendation": primary_anomaly_result["threshold_recommendation"],
        },
    )

    session_summary = _build_session_prediction_summary(classifier_predictions, selected_anomaly_predictions)
    if not session_summary.empty:
        session_summary["selected_anomaly_model"] = str(primary_anomaly_result["model_name"])
        session_summary.to_csv(directories["predictions"] / "per_session_prediction_summary.csv", index=False)
        save_json(
            directories["predictions"] / "per_session_prediction_summary.json",
            {"rows": session_summary.to_dict(orient="records")},
        )

    plot_class_distribution(dataset, directories["plots"] / "class_distribution.png")
    plot_confusion_matrix(
        matrix=np.asarray(classifier_metrics["confusion_matrix"], dtype=int),
        labels=list(map(str, classifier_metrics["labels"])),
        output_path=directories["plots"] / "confusion_matrix.png",
        title="KAIST Random Forest Confusion Matrix",
    )
    if not importance_frame.empty:
        plot_feature_importances(importance_frame, directories["plots"] / "feature_importances.png")
    plot_anomaly_scores(
        selected_anomaly_predictions,
        threshold=float(selected_anomaly_metrics["threshold"]),
        output_path=directories["plots"] / "anomaly_scores.png",
    )

    classifier_table = _build_classifier_metrics_table(
        classifier_metrics=classifier_metrics,
        balance_summary=balance_summary,
        test_frame=test_frame,
    )
    anomaly_table = _build_anomaly_metrics_table(primary_anomaly_result)
    threshold_summary_table = _build_threshold_sweep_summary_table(anomaly_comparison)
    model_comparison_table = _build_model_comparison_table(
        classifier_metrics=classifier_metrics,
        anomaly_comparison=anomaly_comparison,
        selected_model_name=str(primary_anomaly_result["model_name"]),
    )

    split_sensitivity_by_split, split_sensitivity_summary = _run_split_sensitivity_analysis(
        dataset=dataset,
        feature_columns=feature_columns,
        config=config,
    )

    _save_publication_table(
        classifier_table,
        directories["tables"] / "classifier_metrics_table.csv",
        directories["tables"] / "classifier_metrics_table.md",
    )
    _save_publication_table(
        anomaly_table,
        directories["tables"] / "anomaly_metrics_table.csv",
        directories["tables"] / "anomaly_metrics_table.md",
    )
    _save_publication_table(
        threshold_summary_table,
        directories["tables"] / "threshold_sweep_summary_table.csv",
        directories["tables"] / "threshold_sweep_summary_table.md",
    )
    _save_publication_table(
        model_comparison_table,
        directories["tables"] / "model_comparison_summary_table.csv",
        directories["tables"] / "model_comparison_summary_table.md",
    )
    _save_publication_table(
        split_sensitivity_by_split,
        directories["tables"] / "split_sensitivity_by_healthy_holdout.csv",
        directories["tables"] / "split_sensitivity_by_healthy_holdout.md",
    )
    _save_publication_table(
        split_sensitivity_summary,
        directories["tables"] / "split_sensitivity_summary.csv",
        directories["tables"] / "split_sensitivity_summary.md",
    )

    class_counts = dataset["label"].astype(str).value_counts().sort_index()
    plot_classifier_failure_overview(
        class_counts=class_counts,
        confusion_matrix=np.asarray(classifier_metrics["confusion_matrix"], dtype=int),
        labels=list(map(str, classifier_metrics["labels"])),
        output_path=directories["figures"] / "figure_01_classifier_failure_and_imbalance.png",
    )
    plot_anomaly_baseline_comparison(
        anomaly_comparison,
        directories["figures"] / "figure_02_anomaly_baseline_comparison.png",
    )
    if not threshold_sweep_frame.empty:
        plot_threshold_sweep(
            threshold_sweep_frame,
            directories["figures"] / "figure_03_threshold_sweep.png",
        )

    summary = {
        "experiment_name": config.experiment_name,
        "feature_dataset_path": str(feature_dataset_path),
        "results_dir": str(directories["root"]),
        "n_total_windows": int(len(dataset)),
        "n_total_sessions": int(dataset["session_id"].nunique()),
        "n_train_windows": int(len(train_frame)),
        "n_test_windows": int(len(test_frame)),
        "n_classifier_train_windows": int(len(balanced_train_frame)),
        "train_sessions": split_summary["train_sessions"],
        "test_sessions": split_summary["test_sessions"],
        "split_strategy": split_summary["split_strategy"],
        "label_distribution": dataset["label"].astype(str).value_counts().to_dict(),
        "classifier_balance_strategy": config.model.classifier.balance_strategy,
        "classifier_balance_summary": balance_summary,
        "selected_anomaly_model": str(primary_anomaly_result["model_name"]),
        "classifier_metrics_path": str(directories["metrics"] / "classifier_metrics.json"),
        "anomaly_metrics_path": str(directories["metrics"] / "anomaly_metrics.json"),
        "anomaly_comparison_path": str(directories["root"] / "anomaly_baseline_comparison.csv"),
        "threshold_sweep_path": str(directories["root"] / "threshold_sweep.csv"),
        "publication_summary_path": str(directories["root"] / "publication_results_summary.md"),
        "split_sensitivity_summary_path": str(directories["tables"] / "split_sensitivity_summary.csv"),
    }
    save_json(directories["root"] / "experiment_summary.json", summary)
    pd.DataFrame([summary]).to_csv(directories["root"] / "experiment_summary.csv", index=False)
    _write_kaist_experiment_summary_markdown(
        config=config,
        feature_dataset_path=Path(feature_dataset_path),
        split_summary=split_summary,
        test_frame=test_frame,
        balance_summary=balance_summary,
        classifier_metrics=classifier_metrics,
        classifier_predictions=classifier_predictions,
        anomaly_comparison=anomaly_comparison,
        selected_anomaly_result=primary_anomaly_result,
        split_sensitivity_summary=split_sensitivity_summary,
        directories=directories,
    )
    _write_publication_results_summary(
        dataset=dataset,
        classifier_metrics=classifier_metrics,
        selected_anomaly_result=primary_anomaly_result,
        split_sensitivity_summary=split_sensitivity_summary,
        directories=directories,
    )
    return summary
