"""One-command presentation demo runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .ablation import run_ablation_study
from .config import ExperimentConfig
from .demo_data import generate_demo_raw_sessions
from .explanations import generate_feature_explanations
from .pipeline import build_dataset_from_config, evaluate_saved_models, run_inference, run_training_experiment
from .utils import prepare_experiment_directories, save_json


def _format_markdown_table(frame: pd.DataFrame) -> List[str]:
    """Render a small dataframe as a simple Markdown table without extra dependencies."""

    if frame.empty:
        return ["No rows available."]

    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        values = [str(row[column]) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _build_demo_session_results(results_dir: Path, inference_summary: Dict[str, object]) -> pd.DataFrame:
    """Assemble one presentation-friendly per-session table."""

    dataset_path = results_dir / "datasets" / "window_feature_dataset.csv"
    split_path = results_dir / "datasets" / "session_split_manifest.csv"
    evaluation_path = results_dir / "predictions" / "per_session_prediction_summary.csv"

    dataset = pd.read_csv(dataset_path)
    session_frame = (
        dataset.groupby(["session_id", "label"], as_index=False)
        .agg(n_windows=("window_index", "count"))
        .rename(columns={"label": "true_label"})
    )

    if split_path.exists():
        split_frame = pd.read_csv(split_path)
        session_frame = session_frame.merge(split_frame, on="session_id", how="left")

    if evaluation_path.exists():
        evaluation_frame = pd.read_csv(evaluation_path)
        session_frame = session_frame.merge(evaluation_frame, on="session_id", how="left", suffixes=("", "_eval"))
        if "true_label_eval" in session_frame.columns:
            session_frame["true_label"] = session_frame["true_label"].fillna(session_frame["true_label_eval"])
        if "n_windows_eval" in session_frame.columns:
            session_frame["n_windows"] = session_frame["n_windows"].fillna(session_frame["n_windows_eval"])

    inference_session_id = str(inference_summary.get("session_id", ""))
    session_frame["demo_role"] = session_frame["session_id"].apply(
        lambda session_id: "inference_target" if str(session_id) == inference_session_id else "dataset_session"
    )
    session_frame["inference_predicted_class"] = ""
    session_frame["inference_early_warning"] = ""
    session_frame["warning_trigger_reason"] = ""
    session_frame["decision_summary"] = ""

    mask = session_frame["session_id"].astype(str) == inference_session_id
    if mask.any():
        session_frame.loc[mask, "inference_predicted_class"] = str(inference_summary.get("predicted_class", ""))
        session_frame.loc[mask, "inference_early_warning"] = str(inference_summary.get("early_warning", ""))
        session_frame.loc[mask, "warning_trigger_reason"] = str(inference_summary.get("warning_trigger_reason", ""))
        session_frame.loc[mask, "decision_summary"] = str(inference_summary.get("decision_summary", ""))

    ordered_columns = [
        "session_id",
        "true_label",
        "split",
        "demo_role",
        "n_windows",
        "majority_predicted_label",
        "window_accuracy",
        "mean_max_probability",
        "mean_anomaly_score",
        "anomalous_window_ratio",
        "inference_predicted_class",
        "inference_early_warning",
        "warning_trigger_reason",
        "decision_summary",
    ]
    existing_columns = [column for column in ordered_columns if column in session_frame.columns]
    return session_frame[existing_columns].sort_values("session_id").reset_index(drop=True)


def _write_demo_summary(
    config: ExperimentConfig,
    results_dir: Path,
    generated_sessions: List[Dict[str, object]],
    dataset_summary: Dict[str, object],
    split_summary: Dict[str, object],
    classifier_metrics: Dict[str, object],
    anomaly_metrics: Dict[str, object],
    inference_summary: Dict[str, object],
    session_results: pd.DataFrame,
    ablation_result: Dict[str, object] | None,
    explanation_result: Dict[str, object] | None,
) -> None:
    """Write a concise presentation-grade Markdown summary."""

    lines: List[str] = [
        f"# Demo Summary: {config.experiment_name}",
        "",
        "## Sessions Used",
        f"- Data root: `{config.paths.data_root}`",
        f"- Session whitelist: {', '.join(config.paths.session_whitelist)}",
        f"- Generated sessions: {len(generated_sessions)}",
        "",
        "## Class Distribution",
        f"- Window counts: {dataset_summary.get('class_distribution', {})}",
        "",
        "## Train/Test Split",
        f"- Split strategy: `{split_summary.get('split_strategy', 'n/a')}`",
        f"- Train sessions: {', '.join(split_summary.get('train_sessions', []))}",
        f"- Test sessions: {', '.join(split_summary.get('test_sessions', []))}",
        "",
        "## Key Metrics",
        f"- Classifier macro precision: {classifier_metrics.get('precision_macro', 'n/a')}",
        f"- Classifier macro recall: {classifier_metrics.get('recall_macro', 'n/a')}",
        f"- Classifier macro F1: {classifier_metrics.get('f1_macro', 'n/a')}",
        f"- Classifier ROC-AUC: {classifier_metrics.get('roc_auc', 'n/a')}",
        f"- Anomaly precision: {anomaly_metrics.get('precision', 'n/a')}",
        f"- Anomaly recall: {anomaly_metrics.get('recall', 'n/a')}",
        f"- Anomaly F1: {anomaly_metrics.get('f1', 'n/a')}",
        f"- Anomaly threshold: {anomaly_metrics.get('threshold', 'n/a')}",
        "",
        "## Inference Result",
        f"- Session: `{inference_summary.get('session_id', '')}`",
        f"- Predicted class: `{inference_summary.get('predicted_class', '')}`",
        f"- Early warning: `{inference_summary.get('early_warning', '')}`",
        f"- Warning trigger reason: `{inference_summary.get('warning_trigger_reason', '')}`",
        f"- Decision summary: {inference_summary.get('decision_summary', '')}",
        "",
        "## Why Fusion?",
        ablation_result.get("narrative", "Ablation study was not run.") if ablation_result else "Ablation study was not run.",
        "",
        "## Which Sensor Signals Contributed Most?",
        explanation_result.get("summary_text", "Feature explanation outputs were not generated.")
        if explanation_result
        else "Feature explanation outputs were not generated.",
        "",
        "## Per-Session Results",
    ]
    lines.extend(_format_markdown_table(session_results))
    lines.extend(
        [
            "",
            "## Ablation Outputs",
            "- `ablation_comparison.csv`",
            "- `ablation_comparison.json`",
            "- `ablation_summary.md`",
            "- `plots/ablation_comparison.png`",
            "",
            "## Explanation Outputs",
            "- `top_features.csv`",
            "- `top_features.json`",
            "- `feature_explanation_summary.md`",
            "- `sensor_group_importance.csv`",
            "- `sensor_group_importance.json`",
            "- `plots/top_features.png`",
            "- `plots/sensor_group_importance.png`",
            "",
            "## Presentation Figures",
            "- `plots/class_distribution.png`",
            "- `plots/confusion_matrix.png`",
            "- `plots/anomaly_scores.png`",
            "- `plots/feature_importances.png`",
        ]
    )

    (results_dir / "demo_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_demo(config: ExperimentConfig) -> Dict[str, object]:
    """Generate raw smoke-test data, run the full pipeline, and save a compact demo summary."""

    generated_sessions = generate_demo_raw_sessions(config.paths.data_root)
    dataset_summary = build_dataset_from_config(config)
    training_summary = run_training_experiment(config)
    evaluation_summary = evaluate_saved_models(config)
    inference_summary = run_inference(config)

    results_dir = prepare_experiment_directories(config.paths.results_root, config.experiment_name)["root"]
    with (results_dir / "metrics" / "split_summary.json").open("r", encoding="utf-8") as handle:
        split_summary = json.load(handle)
    with (results_dir / "metrics" / "classifier_metrics.json").open("r", encoding="utf-8") as handle:
        classifier_metrics = json.load(handle)
    with (results_dir / "metrics" / "anomaly_metrics.json").open("r", encoding="utf-8") as handle:
        anomaly_metrics = json.load(handle)

    ablation_result = run_ablation_study(config, results_dir) if config.ablation.enabled else None
    explanation_result = generate_feature_explanations(results_dir)
    session_results = _build_demo_session_results(results_dir, inference_summary)
    session_results.to_csv(results_dir / "demo_session_results.csv", index=False)
    save_json(
        results_dir / "demo_summary.json",
        {
            "generated_sessions": generated_sessions,
            "dataset_summary": dataset_summary,
            "training_summary": training_summary,
            "evaluation_summary": evaluation_summary,
            "inference_summary": inference_summary,
            "ablation_result": ablation_result,
            "explanation_result": explanation_result,
        },
    )
    _write_demo_summary(
        config=config,
        results_dir=results_dir,
        generated_sessions=generated_sessions,
        dataset_summary=dataset_summary,
        split_summary=split_summary,
        classifier_metrics=classifier_metrics,
        anomaly_metrics=anomaly_metrics,
        inference_summary=inference_summary,
        session_results=session_results,
        ablation_result=ablation_result,
        explanation_result=explanation_result,
    )

    compact_summary = {
        "sessions_used": [item["session_id"] for item in generated_sessions],
        "class_distribution": dataset_summary.get("class_distribution", {}),
        "train_sessions": training_summary.get("train_sessions", []),
        "test_sessions": training_summary.get("test_sessions", []),
        "classifier_macro_f1": classifier_metrics.get("f1_macro"),
        "anomaly_f1": anomaly_metrics.get("f1"),
        "ablation_narrative": ablation_result.get("narrative") if ablation_result else None,
        "dominant_sensor_group": explanation_result.get("dominant_sensor_group") if explanation_result else None,
        "inference_result": {
            "session_id": inference_summary.get("session_id"),
            "predicted_class": inference_summary.get("predicted_class"),
            "early_warning": inference_summary.get("early_warning"),
            "warning_trigger_reason": inference_summary.get("warning_trigger_reason"),
        },
    }
    return compact_summary
