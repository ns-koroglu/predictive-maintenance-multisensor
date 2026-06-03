"""Plotting helpers for simple thesis-ready experiment figures."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_class_distribution(frame: pd.DataFrame, output_path: str | Path, label_column: str = "label") -> None:
    """Plot the number of windows available for each class."""

    counts = frame[label_column].astype(str).value_counts().sort_index()
    figure, axis = plt.subplots(figsize=(7, 4))
    bars = axis.bar(counts.index.tolist(), counts.values.tolist(), color="#3b82f6")
    axis.set_title("Class Distribution")
    axis.set_ylabel("Window Count")
    axis.set_xlabel("Label")
    for bar in bars:
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{int(bar.get_height())}",
            ha="center",
            va="bottom",
        )
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_confusion_matrix(
    matrix: np.ndarray,
    labels: Sequence[str],
    output_path: str | Path,
    title: str = "Confusion Matrix",
) -> None:
    """Plot a confusion matrix using a readable heatmap style."""

    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks(np.arange(len(labels)))
    axis.set_yticks(np.arange(len(labels)))
    axis.set_xticklabels(labels, rotation=30, ha="right")
    axis.set_yticklabels(labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(title)

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            axis.text(col_index, row_index, str(int(matrix[row_index, col_index])), ha="center", va="center")

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_feature_trends(
    frame: pd.DataFrame,
    feature_columns: List[str],
    output_path: str | Path,
) -> None:
    """Plot selected feature trajectories per session."""

    if not feature_columns:
        return

    selected = feature_columns[: min(len(feature_columns), 6)]
    figure, axes = plt.subplots(len(selected), 1, figsize=(10, 3.4 * len(selected)), sharex=False)
    if len(selected) == 1:
        axes = [axes]

    for axis, feature_name in zip(axes, selected):
        for session_id, session_frame in frame.groupby("session_id"):
            ordered = session_frame.sort_values("window_start")
            label = str(ordered["label"].iloc[0]) if "label" in ordered.columns else str(session_id)
            time_axis = 0.5 * (
                ordered["window_start"].to_numpy(dtype=float) + ordered["window_end"].to_numpy(dtype=float)
            )
            axis.plot(
                time_axis,
                ordered[feature_name].to_numpy(dtype=float),
                marker="o",
                linewidth=1.5,
                label=f"{session_id} ({label})",
            )
        axis.set_ylabel(feature_name)
        axis.grid(alpha=0.3)

    axes[-1].set_xlabel("Window Mid Time (s)")
    axes[0].legend(loc="best", fontsize=8)
    figure.suptitle("Feature Trends Across Sessions", y=0.995)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_anomaly_scores(
    prediction_frame: pd.DataFrame,
    threshold: float,
    output_path: str | Path,
) -> None:
    """Plot anomaly score trajectories and the decision threshold."""

    figure, axis = plt.subplots(figsize=(10, 4.5))
    for session_id, session_frame in prediction_frame.groupby("session_id"):
        ordered = session_frame.sort_values("window_start")
        label = str(ordered["label"].iloc[0]) if "label" in ordered.columns else str(session_id)
        time_axis = 0.5 * (
            ordered["window_start"].to_numpy(dtype=float) + ordered["window_end"].to_numpy(dtype=float)
        )
        axis.plot(
            time_axis,
            ordered["anomaly_score"].to_numpy(dtype=float),
            marker="o",
            linewidth=1.5,
            label=f"{session_id} ({label})",
        )

    axis.axhline(threshold, color="#dc2626", linestyle="--", label="Threshold")
    axis.set_title("Anomaly Scores")
    axis.set_xlabel("Window Mid Time (s)")
    axis.set_ylabel("Isolation Forest Score")
    axis.grid(alpha=0.3)
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_feature_importances(
    importance_frame: pd.DataFrame,
    output_path: str | Path,
    top_n: int = 15,
) -> None:
    """Plot the most influential Random Forest features."""

    if importance_frame.empty:
        return

    top_frame = importance_frame.sort_values("importance", ascending=True).tail(top_n)
    figure, axis = plt.subplots(figsize=(9, max(4, 0.4 * len(top_frame))))
    axis.barh(top_frame["feature"], top_frame["importance"], color="#f97316")
    axis.set_title("Random Forest Feature Importances")
    axis.set_xlabel("Importance")
    axis.set_ylabel("Feature")
    axis.grid(axis="x", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_ablation_comparison(
    comparison_frame: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Plot classifier and anomaly F1 scores for each ablation setup."""

    if comparison_frame.empty:
        return

    frame = comparison_frame.copy()
    frame["anomaly_f1"] = frame["anomaly_f1"].fillna(0.0)
    x = np.arange(len(frame))
    width = 0.35

    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.bar(x - width / 2.0, frame["classifier_f1"], width=width, label="Classifier F1", color="#2563eb")
    axis.bar(x + width / 2.0, frame["anomaly_f1"], width=width, label="Anomaly F1", color="#f97316")

    axis.set_xticks(x)
    axis.set_xticklabels(frame["setup_name"], rotation=20, ha="right")
    axis.set_ylabel("F1 Score")
    axis.set_ylim(0.0, 1.05)
    axis.set_title("Ablation Study: Single Sensors vs Fusion")
    axis.grid(axis="y", alpha=0.3)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_sensor_group_importance(
    grouped_importance_frame: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Plot aggregated importance by sensor family."""

    if grouped_importance_frame.empty:
        return

    frame = grouped_importance_frame.copy()
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(frame["sensor_group"], frame["normalized_importance"], color=["#ef4444", "#2563eb", "#f97316"])
    axis.set_title("Sensor Group Contribution")
    axis.set_xlabel("Sensor Group")
    axis.set_ylabel("Normalized Importance")
    axis.set_ylim(0.0, 1.05)
    axis.grid(axis="y", alpha=0.3)
    for index, row in frame.iterrows():
        axis.text(index, float(row["normalized_importance"]), f"{float(row['normalized_importance']):.2f}", ha="center", va="bottom")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_threshold_sweep(
    sweep_frame: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Plot precision, recall, and F1 versus threshold for anomaly models."""

    if sweep_frame.empty:
        return

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    metric_colors = {
        "precision": "#2563eb",
        "recall": "#dc2626",
        "f1": "#16a34a",
    }

    for model_name, frame in sweep_frame.groupby("model_name"):
        ordered = frame.sort_values("threshold")
        for metric_name in ("precision", "recall"):
            axes[0].plot(
                ordered["threshold"],
                ordered[metric_name],
                linewidth=1.8,
                label=f"{model_name} {metric_name}",
                color=metric_colors[metric_name],
                alpha=0.7 if model_name == "isolation_forest" else 0.95,
                linestyle="-" if model_name == "isolation_forest" else "--",
            )
        axes[1].plot(
            ordered["threshold"],
            ordered["f1"],
            linewidth=2.0,
            label=f"{model_name} F1",
            color=metric_colors["f1"],
            alpha=0.7 if model_name == "isolation_forest" else 0.95,
            linestyle="-" if model_name == "isolation_forest" else "--",
        )

        calibrated_rows = ordered[ordered["is_calibrated_threshold"]]
        if not calibrated_rows.empty:
            calibrated_threshold = float(calibrated_rows.iloc[0]["threshold"])
            axes[0].axvline(calibrated_threshold, color="#6b7280", linestyle=":", alpha=0.6)
            axes[1].axvline(calibrated_threshold, color="#6b7280", linestyle=":", alpha=0.6)

    axes[0].set_title("Anomaly Threshold Sweep")
    axes[0].set_ylabel("Precision / Recall")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)

    axes[1].set_xlabel("Anomaly Score Threshold")
    axes[1].set_ylabel("F1 Score")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="best", fontsize=8)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_classifier_failure_overview(
    class_counts: pd.Series,
    confusion_matrix: np.ndarray,
    labels: Sequence[str],
    output_path: str | Path,
) -> None:
    """Plot class imbalance and classifier confusion matrix in one publication-friendly figure."""

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    bars = axes[0].bar(class_counts.index.tolist(), class_counts.values.tolist(), color=["#2563eb", "#dc2626"])
    axes[0].set_title("Class Imbalance")
    axes[0].set_xlabel("Label")
    axes[0].set_ylabel("Window Count")
    for bar in bars:
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{int(bar.get_height())}",
            ha="center",
            va="bottom",
        )

    image = axes[1].imshow(confusion_matrix, cmap="Blues")
    figure.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_xticks(np.arange(len(labels)))
    axes[1].set_yticks(np.arange(len(labels)))
    axes[1].set_xticklabels(labels, rotation=30, ha="right")
    axes[1].set_yticklabels(labels)
    axes[1].set_title("Classifier Failure Case")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    for row_index in range(confusion_matrix.shape[0]):
        for col_index in range(confusion_matrix.shape[1]):
            axes[1].text(
                col_index,
                row_index,
                str(int(confusion_matrix[row_index, col_index])),
                ha="center",
                va="center",
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_anomaly_baseline_comparison(
    comparison_frame: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Plot calibrated anomaly metrics for each baseline model."""

    frame = comparison_frame.copy()
    frame = frame[frame["status"].astype(str) == "ok"].copy()
    if frame.empty:
        return

    metrics = ["balanced_accuracy", "precision", "recall", "f1"]
    x = np.arange(len(metrics))
    width = 0.35 if len(frame) <= 2 else max(0.18, 0.7 / len(frame))

    figure, axis = plt.subplots(figsize=(10, 4.8))
    colors = ["#2563eb", "#f97316", "#16a34a", "#dc2626"]
    for index, (_, row) in enumerate(frame.iterrows()):
        offset = (index - (len(frame) - 1) / 2.0) * width
        axis.bar(
            x + offset,
            [float(row[metric]) for metric in metrics],
            width=width,
            label=str(row["model_name"]),
            color=colors[index % len(colors)],
        )

    axis.set_xticks(x)
    axis.set_xticklabels(["Balanced Acc.", "Precision", "Recall", "F1"])
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title("Calibrated Anomaly Baseline Comparison")
    axis.grid(axis="y", alpha=0.3)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_progression_anomaly_scores(
    trend_frame: pd.DataFrame,
    threshold: float,
    output_path: str | Path,
    rolling_window_files: int,
) -> None:
    """Plot anomaly scores and rolling summaries along run-to-failure progression."""

    if trend_frame.empty:
        return

    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for session_id, frame in trend_frame.groupby("session_id"):
        ordered = frame.sort_values("progression_index")
        x_axis = ordered["elapsed_hours"].to_numpy(dtype=float)
        axes[0].plot(
            x_axis,
            ordered["anomaly_score"].to_numpy(dtype=float),
            marker="o",
            linewidth=1.5,
            label=str(session_id),
        )
        if "rolling_anomaly_score_mean" in ordered.columns:
            axes[0].plot(
                x_axis,
                ordered["rolling_anomaly_score_mean"].to_numpy(dtype=float),
                linewidth=2.0,
                linestyle="--",
                label=f"{session_id} rolling",
            )
        if "rolling_anomaly_flag_ratio" in ordered.columns:
            axes[1].plot(
                x_axis,
                ordered["rolling_anomaly_flag_ratio"].to_numpy(dtype=float),
                marker="o",
                linewidth=1.5,
                label=str(session_id),
            )

    axes[0].axhline(threshold, color="#dc2626", linestyle="--", label="Calibrated Threshold")
    axes[0].set_title("Run-to-Failure Anomaly Score Progression")
    axes[0].set_ylabel("Anomaly Score")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)

    axes[1].set_title(f"Rolling Anomaly Ratio ({rolling_window_files} files)")
    axes[1].set_xlabel("Elapsed Hours")
    axes[1].set_ylabel("Rolling Flag Ratio")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="best", fontsize=8)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_rtf_anomaly_trend(
    trend_frame: pd.DataFrame,
    output_path: str | Path,
    rolling_window_files: int,
) -> None:
    """Plot publication-friendly anomaly trends for each anomaly model."""

    if trend_frame.empty or "model_name" not in trend_frame.columns:
        return

    model_names = sorted(trend_frame["model_name"].astype(str).unique().tolist())
    figure, axes = plt.subplots(len(model_names), 1, figsize=(11, max(4.5, 4.0 * len(model_names))), sharex=True)
    if len(model_names) == 1:
        axes = [axes]

    for axis, model_name in zip(axes, model_names):
        model_frame = trend_frame[trend_frame["model_name"].astype(str) == str(model_name)].copy()
        for session_id, session_frame in model_frame.groupby("session_id", sort=True):
            ordered = session_frame.sort_values("elapsed_hours").reset_index(drop=True)
            x_axis = ordered["elapsed_hours"].to_numpy(dtype=float)
            axis.plot(
                x_axis,
                ordered["anomaly_score_raw"].to_numpy(dtype=float),
                linewidth=1.5,
                color="#2563eb",
                label=f"{session_id} raw score",
            )
            axis.plot(
                x_axis,
                ordered["rolling_anomaly_score_mean"].to_numpy(dtype=float),
                linewidth=2.0,
                linestyle="--",
                color="#f97316",
                label=f"{session_id} rolling mean ({rolling_window_files})",
            )

            if ordered["is_reference_baseline"].astype(bool).any():
                reference_end_hour = float(
                    ordered.loc[ordered["is_reference_baseline"].astype(bool), "elapsed_hours"].max()
                )
                axis.axvspan(0.0, reference_end_hour, color="#dcfce7", alpha=0.35, label="Calibration region")

            threshold = float(ordered["calibrated_threshold"].iloc[0])
            axis.axhline(threshold, color="#dc2626", linestyle="--", linewidth=1.4, label="Calibrated threshold")

            threshold_rows = ordered[
                ordered["post_reference_threshold_crossed"].astype(bool)
                & ~ordered["post_reference_threshold_crossed"].astype(bool).shift(fill_value=False)
            ]
            if not threshold_rows.empty:
                crossing = threshold_rows.iloc[0]
                axis.scatter(
                    [float(crossing["elapsed_hours"])],
                    [float(crossing["anomaly_score_raw"])],
                    color="#dc2626",
                    s=45,
                    zorder=5,
                    label="First threshold crossing",
                )

            warning_rows = ordered[
                ordered["sustained_warning"].astype(bool)
                & ~ordered["sustained_warning"].astype(bool).shift(fill_value=False)
            ]
            if not warning_rows.empty:
                warning = warning_rows.iloc[0]
                axis.scatter(
                    [float(warning["elapsed_hours"])],
                    [float(warning["anomaly_score_raw"])],
                    color="#7c3aed",
                    marker="*",
                    s=120,
                    zorder=6,
                    label="First sustained warning",
                )

        axis.set_title(f"Anomaly Trend: {str(model_name).replace('_', ' ').title()}")
        axis.set_ylabel("Anomaly Score")
        axis.grid(alpha=0.3)
        handles, labels = axis.get_legend_handles_labels()
        deduped = dict(zip(labels, handles))
        axis.legend(deduped.values(), deduped.keys(), loc="best", fontsize=8)

    axes[-1].set_xlabel("Elapsed Hours")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)
