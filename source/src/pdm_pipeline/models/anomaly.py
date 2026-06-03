"""Classical anomaly baselines for early anomaly warning."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from ..config import AnomalyConfig


AnomalyArtifact = Dict[str, object]


def _healthy_training_frame(
    train_frame: pd.DataFrame,
    label_column: str,
    healthy_label: str,
    minimum_healthy_windows: int,
) -> pd.DataFrame:
    """Return healthy training windows and validate the minimum sample count."""

    healthy_mask = train_frame[label_column].astype(str).str.lower() == healthy_label.lower()
    healthy_frame = train_frame.loc[healthy_mask].copy()
    if healthy_frame.empty:
        raise ValueError("Anomaly model requires at least one healthy training window.")
    if len(healthy_frame) < minimum_healthy_windows:
        raise ValueError(
            f"Anomaly model requires at least {minimum_healthy_windows} healthy windows, "
            f"but only {len(healthy_frame)} were available."
        )
    return healthy_frame


def _build_preprocessor() -> Pipeline:
    """Return the shared anomaly preprocessing pipeline."""

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def _score_model(model_name: str, model: object, transformed_features: np.ndarray) -> np.ndarray:
    """Return anomaly scores where larger values mean more anomalous windows."""

    if model_name == "one_class_svm":
        return -np.asarray(model.decision_function(transformed_features), dtype=float).reshape(-1)
    return -np.asarray(model.score_samples(transformed_features), dtype=float).reshape(-1)


def score_anomaly_detector(
    artifact: AnomalyArtifact,
    test_frame: pd.DataFrame,
    label_column: str,
) -> pd.DataFrame:
    """Score windows with a fitted anomaly detector without fixing a threshold yet."""

    preprocessor = artifact["preprocessor"]
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    healthy_label = str(artifact["healthy_label"])
    model_name = str(artifact.get("model_name", "isolation_forest"))

    x_test = preprocessor.transform(test_frame[feature_columns])
    scores = _score_model(model_name, model, x_test)
    target = (test_frame[label_column].astype(str).str.lower() != healthy_label.lower()).astype(int)

    prediction_frame = test_frame[
        [column for column in ["session_id", "window_index", "window_start", "window_end", label_column] if column in test_frame]
    ].copy()
    prediction_frame["true_anomaly"] = target.to_numpy(dtype=int)
    prediction_frame["anomaly_score"] = scores
    return prediction_frame


def score_anomaly_sequence(
    artifact: AnomalyArtifact,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Score an unlabeled chronological sequence with a fitted anomaly detector."""

    preprocessor = artifact["preprocessor"]
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    model_name = str(artifact.get("model_name", "isolation_forest"))

    transformed = preprocessor.transform(frame[feature_columns])
    scores = _score_model(model_name, model, transformed)

    columns = [
        column
        for column in [
            "session_id",
            "group_id",
            "window_index",
            "window_start",
            "window_end",
            "progression_index",
            "elapsed_hours",
        ]
        if column in frame.columns
    ]
    scored = frame[columns].copy()
    scored["anomaly_score"] = scores
    return scored


def compute_anomaly_metrics(
    prediction_frame: pd.DataFrame,
    threshold: float,
) -> Dict[str, object]:
    """Convert anomaly scores into binary predictions and compute summary metrics."""

    target = prediction_frame["true_anomaly"].astype(int)
    flags = prediction_frame["anomaly_score"].to_numpy(dtype=float) >= float(threshold)

    precision, recall, f1, _ = precision_recall_fscore_support(
        target,
        flags.astype(int),
        average="binary",
        zero_division=0,
    )
    metrics: Dict[str, object] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(target, flags.astype(int))),
        "balanced_accuracy": float(balanced_accuracy_score(target, flags.astype(int))),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(target, flags.astype(int), labels=[0, 1]).tolist(),
        "labels": ["normal", "anomaly"],
    }

    if target.nunique() >= 2:
        metrics["roc_auc"] = float(roc_auc_score(target, prediction_frame["anomaly_score"].to_numpy(dtype=float)))
        metrics["pr_auc"] = float(average_precision_score(target, prediction_frame["anomaly_score"].to_numpy(dtype=float)))
    return metrics


def calibrate_anomaly_threshold(healthy_scores: np.ndarray, config: AnomalyConfig) -> Dict[str, float | str]:
    """Calibrate an anomaly threshold from healthy training scores only."""

    scores = np.asarray(healthy_scores, dtype=float)
    if scores.size == 0:
        raise ValueError("Threshold calibration requires at least one healthy score.")

    score_mean = float(np.mean(scores))
    score_std = float(np.std(scores))
    score_median = float(np.median(scores))
    score_max = float(np.max(scores))

    if config.threshold_strategy == "quantile":
        base_threshold = float(np.quantile(scores, config.threshold_quantile))
    elif config.threshold_strategy == "mean_std":
        base_threshold = score_mean + float(config.threshold_quantile) * score_std
    else:
        raise ValueError(f"Unsupported anomaly threshold strategy: {config.threshold_strategy}")

    threshold = float(base_threshold + config.threshold_buffer_std * score_std)
    return {
        "strategy": config.threshold_strategy,
        "threshold": threshold,
        "base_threshold": float(base_threshold),
        "threshold_quantile": float(config.threshold_quantile),
        "threshold_buffer_std": float(config.threshold_buffer_std),
        "healthy_score_min": float(np.min(scores)),
        "healthy_score_max": score_max,
        "healthy_score_mean": score_mean,
        "healthy_score_median": score_median,
        "healthy_score_std": score_std,
        "healthy_score_q90": float(np.quantile(scores, 0.90)),
        "healthy_score_q95": float(np.quantile(scores, 0.95)),
        "healthy_score_q99": float(np.quantile(scores, 0.99)),
    }


def train_anomaly_detector(
    train_frame: pd.DataFrame,
    feature_columns: List[str],
    label_column: str,
    healthy_label: str,
    config: AnomalyConfig,
) -> AnomalyArtifact:
    """Train an Isolation Forest on healthy windows only."""

    healthy_frame = _healthy_training_frame(
        train_frame=train_frame,
        label_column=label_column,
        healthy_label=healthy_label,
        minimum_healthy_windows=config.minimum_healthy_windows,
    )
    if not 0 < config.threshold_quantile <= 1:
        raise ValueError("Anomaly threshold_quantile must be between 0 and 1.")

    preprocessor = _build_preprocessor()
    x_healthy = preprocessor.fit_transform(healthy_frame[feature_columns])

    model = IsolationForest(
        n_estimators=config.n_estimators,
        contamination=config.contamination,
        random_state=config.random_state,
    )
    model.fit(x_healthy)

    healthy_scores = -model.score_samples(x_healthy)
    calibration = calibrate_anomaly_threshold(healthy_scores, config)
    return {
        "model_name": "isolation_forest",
        "preprocessor": preprocessor,
        "model": model,
        "feature_columns": list(feature_columns),
        "healthy_label": healthy_label,
        "threshold": float(calibration["threshold"]),
        "calibration": calibration,
        "n_healthy_train_windows": int(len(healthy_frame)),
        "healthy_train_sessions": sorted(healthy_frame["session_id"].astype(str).unique().tolist())
        if "session_id" in healthy_frame
        else [],
    }


def train_one_class_svm_detector(
    train_frame: pd.DataFrame,
    feature_columns: List[str],
    label_column: str,
    healthy_label: str,
    config: AnomalyConfig,
) -> AnomalyArtifact:
    """Train a One-Class SVM on healthy windows only."""

    healthy_frame = _healthy_training_frame(
        train_frame=train_frame,
        label_column=label_column,
        healthy_label=healthy_label,
        minimum_healthy_windows=config.minimum_healthy_windows,
    )
    if not 0 < config.threshold_quantile <= 1:
        raise ValueError("Anomaly threshold_quantile must be between 0 and 1.")

    preprocessor = _build_preprocessor()
    x_healthy = preprocessor.fit_transform(healthy_frame[feature_columns])

    model = OneClassSVM(
        kernel=config.one_class_svm_kernel,
        nu=config.one_class_svm_nu,
        gamma=config.one_class_svm_gamma,
    )
    model.fit(x_healthy)

    healthy_scores = _score_model("one_class_svm", model, x_healthy)
    calibration = calibrate_anomaly_threshold(healthy_scores, config)
    return {
        "model_name": "one_class_svm",
        "preprocessor": preprocessor,
        "model": model,
        "feature_columns": list(feature_columns),
        "healthy_label": healthy_label,
        "threshold": float(calibration["threshold"]),
        "calibration": calibration,
        "n_healthy_train_windows": int(len(healthy_frame)),
        "healthy_train_sessions": sorted(healthy_frame["session_id"].astype(str).unique().tolist())
        if "session_id" in healthy_frame
        else [],
    }


def evaluate_anomaly_detector(
    artifact: AnomalyArtifact,
    test_frame: pd.DataFrame,
    label_column: str,
) -> tuple[Dict[str, object], pd.DataFrame]:
    """Score windows, derive anomaly flags, and compute binary anomaly metrics."""

    prediction_frame = score_anomaly_detector(
        artifact=artifact,
        test_frame=test_frame,
        label_column=label_column,
    )
    threshold = float(artifact["threshold"])
    metrics = compute_anomaly_metrics(prediction_frame, threshold=threshold)
    prediction_frame["anomaly_flag"] = (
        prediction_frame["anomaly_score"].to_numpy(dtype=float) >= threshold
    ).astype(int)
    metrics.update(
        {
            "model_name": str(artifact.get("model_name", "isolation_forest")),
            "n_healthy_train_windows": int(artifact["n_healthy_train_windows"]),
            "healthy_train_sessions": list(artifact.get("healthy_train_sessions", [])),
            "calibration": artifact["calibration"],
        }
    )

    return metrics, prediction_frame


def build_threshold_sweep(
    prediction_frame: pd.DataFrame,
    model_name: str,
    calibrated_threshold: float,
    threshold_points: int,
) -> pd.DataFrame:
    """Evaluate precision-recall trade-offs over a score threshold sweep."""

    if prediction_frame.empty:
        return pd.DataFrame()

    scores = prediction_frame["anomaly_score"].to_numpy(dtype=float)
    if np.allclose(scores.min(), scores.max()):
        thresholds = np.asarray([scores.min()], dtype=float)
    else:
        quantiles = np.linspace(0.0, 1.0, max(2, int(threshold_points)))
        thresholds = np.unique(np.quantile(scores, quantiles))

    rows: List[Dict[str, float | int | str | bool]] = []
    for threshold in thresholds:
        metrics = compute_anomaly_metrics(prediction_frame, threshold=float(threshold))
        flags = prediction_frame["anomaly_score"].to_numpy(dtype=float) >= float(threshold)
        rows.append(
            {
                "model_name": str(model_name),
                "threshold": float(threshold),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
                "accuracy": float(metrics["accuracy"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "anomalous_window_ratio": float(flags.mean()),
                "is_calibrated_threshold": bool(np.isclose(float(threshold), float(calibrated_threshold))),
            }
        )

    sweep_frame = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    return sweep_frame


def recommend_threshold_from_sweep(
    sweep_frame: pd.DataFrame,
    calibrated_threshold: float,
) -> Dict[str, object]:
    """Choose a readable operating threshold from the sweep table for analysis."""

    if sweep_frame.empty:
        return {
            "recommended_threshold": float(calibrated_threshold),
            "selection_rule": "fallback_to_calibrated",
        }

    ranked = sweep_frame.copy()
    ranked["distance_to_calibrated"] = (ranked["threshold"] - float(calibrated_threshold)).abs()
    ranked = ranked.sort_values(
        ["balanced_accuracy", "f1", "precision", "distance_to_calibrated"],
        ascending=[False, False, False, True],
    )
    best_row = ranked.iloc[0]
    return {
        "recommended_threshold": float(best_row["threshold"]),
        "selection_rule": "max_balanced_accuracy_then_f1",
        "recommended_precision": float(best_row["precision"]),
        "recommended_recall": float(best_row["recall"]),
        "recommended_f1": float(best_row["f1"]),
        "recommended_balanced_accuracy": float(best_row["balanced_accuracy"]),
        "calibrated_threshold": float(calibrated_threshold),
    }
