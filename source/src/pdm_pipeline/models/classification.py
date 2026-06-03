"""Random Forest baseline for health-state classification."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, label_binarize

from ..config import ClassifierConfig


ClassifierArtifact = Dict[str, object]


def _resolve_positive_label(
    class_names: Sequence[str],
    positive_labels: Sequence[str],
    healthy_label: str,
) -> str | None:
    """Choose the positive class used for binary ROC-AUC reporting."""

    class_list = [str(name) for name in class_names]
    for label in positive_labels:
        if label in class_list:
            return str(label)

    for label in class_list:
        if label.lower() != healthy_label.lower():
            return label
    return class_list[0] if class_list else None


def train_classifier(
    train_frame: pd.DataFrame,
    feature_columns: List[str],
    label_column: str,
    config: ClassifierConfig,
) -> ClassifierArtifact:
    """Train a Random Forest classifier and persist explicit preprocessing artifacts."""

    preprocessor = SimpleImputer(strategy="median")
    x_train = preprocessor.fit_transform(train_frame[feature_columns])

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_frame[label_column].astype(str))

    model = RandomForestClassifier(
        n_estimators=config.n_estimators,
        random_state=config.random_state,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        n_jobs=config.n_jobs,
        class_weight=config.class_weight,
    )
    model.fit(x_train, y_train)

    return {
        "preprocessor": preprocessor,
        "label_encoder": label_encoder,
        "model": model,
        "feature_columns": list(feature_columns),
    }


def balance_training_frame(
    train_frame: pd.DataFrame,
    label_column: str,
    config: ClassifierConfig,
) -> tuple[pd.DataFrame, Dict[str, object]]:
    """Optionally rebalance the training frame without touching the test distribution."""

    counts = train_frame[label_column].astype(str).value_counts()
    summary: Dict[str, object] = {
        "strategy": config.balance_strategy,
        "target_ratio": float(config.balance_target_ratio),
        "before_counts": counts.to_dict(),
    }

    if config.balance_strategy == "none":
        summary["after_counts"] = counts.to_dict()
        summary["n_rows_before"] = int(len(train_frame))
        summary["n_rows_after"] = int(len(train_frame))
        return train_frame.copy(), summary

    if counts.empty or len(counts) < 2:
        summary["note"] = "Balancing was skipped because at least two classes are required."
        summary["after_counts"] = counts.to_dict()
        summary["n_rows_before"] = int(len(train_frame))
        summary["n_rows_after"] = int(len(train_frame))
        return train_frame.copy(), summary

    ratio = float(config.balance_target_ratio)
    if ratio <= 0:
        raise ValueError("Classifier balance_target_ratio must be positive.")

    rng = np.random.default_rng(config.random_state)
    class_frames = {
        str(label): subset.copy()
        for label, subset in train_frame.groupby(train_frame[label_column].astype(str), sort=False)
    }

    minority_count = int(counts.min())
    majority_count = int(counts.max())

    balanced_parts: List[pd.DataFrame] = []
    if config.balance_strategy == "downsample_majority":
        target_majority = max(minority_count, int(round(minority_count * ratio)))
        for label, subset in class_frames.items():
            if len(subset) > target_majority:
                chosen = rng.choice(subset.index.to_numpy(), size=target_majority, replace=False)
                balanced_parts.append(subset.loc[chosen].copy())
            else:
                balanced_parts.append(subset)
    elif config.balance_strategy == "upsample_minority":
        target_minority = max(1, int(round(majority_count / ratio)))
        for label, subset in class_frames.items():
            if len(subset) < target_minority:
                chosen = rng.choice(subset.index.to_numpy(), size=target_minority, replace=True)
                balanced_parts.append(subset.loc[chosen].copy())
            else:
                balanced_parts.append(subset)
    else:
        raise ValueError(f"Unsupported classifier balance strategy: {config.balance_strategy}")

    balanced_frame = (
        pd.concat(balanced_parts, axis=0, ignore_index=True)
        .sample(frac=1.0, random_state=config.random_state)
        .reset_index(drop=True)
    )
    after_counts = balanced_frame[label_column].astype(str).value_counts().to_dict()

    summary["after_counts"] = after_counts
    summary["n_rows_before"] = int(len(train_frame))
    summary["n_rows_after"] = int(len(balanced_frame))
    return balanced_frame, summary


def classifier_feature_importances(
    artifact: ClassifierArtifact,
    feature_columns: List[str],
) -> List[Dict[str, float]]:
    """Return feature importances sorted from most to least influential."""

    estimator = artifact["model"]
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return []

    ranking = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)
    return ranking.to_dict(orient="records")


def evaluate_classifier(
    artifact: ClassifierArtifact,
    test_frame: pd.DataFrame,
    feature_columns: List[str],
    label_column: str,
    healthy_label: str,
    positive_labels: Sequence[str],
) -> tuple[Dict[str, object], pd.DataFrame]:
    """Compute thesis-oriented metrics and store per-window predictions."""

    preprocessor = artifact["preprocessor"]
    label_encoder = artifact["label_encoder"]
    model = artifact["model"]

    x_test = preprocessor.transform(test_frame[feature_columns])
    y_true = test_frame[label_column].astype(str)
    y_pred_encoded = model.predict(x_test)
    y_pred = pd.Series(label_encoder.inverse_transform(y_pred_encoded), index=test_frame.index, dtype=str)
    class_names = [str(name) for name in label_encoder.classes_]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    prediction_frame = test_frame[
        [
            column
            for column in [
                "session_id",
                "group_id",
                "split_group",
                "window_index",
                "window_start",
                "window_end",
                label_column,
            ]
            if column in test_frame
        ]
    ].copy()
    prediction_frame["predicted_label"] = y_pred

    metrics: Dict[str, object] = {
        "labels": class_names,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=class_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=class_names).tolist(),
        "top_feature_importances": classifier_feature_importances(artifact, feature_columns)[:20],
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_test)
        for class_index, class_name in enumerate(class_names):
            prediction_frame[f"prob_{class_name}"] = probabilities[:, class_index]
        prediction_frame["max_probability"] = np.max(probabilities, axis=1)

        unique_labels = sorted(y_true.unique().tolist())
        if len(unique_labels) >= 2:
            if len(class_names) == 2:
                positive_label = _resolve_positive_label(class_names, positive_labels, healthy_label)
                if positive_label is not None and positive_label in class_names:
                    positive_index = class_names.index(positive_label)
                    binary_target = (y_true == positive_label).astype(int)
                    metrics["roc_auc"] = float(roc_auc_score(binary_target, probabilities[:, positive_index]))
                    metrics["pr_auc"] = float(
                        average_precision_score(binary_target, probabilities[:, positive_index])
                    )
                    metrics["roc_auc_positive_label"] = positive_label
                    metrics["pr_auc_positive_label"] = positive_label
            elif len(class_names) > 2:
                binarized_target = label_binarize(y_true, classes=class_names)
                if binarized_target.shape[1] == probabilities.shape[1]:
                    metrics["roc_auc"] = float(
                        roc_auc_score(
                            binarized_target,
                            probabilities,
                            average="macro",
                            multi_class="ovr",
                        )
                    )
                    metrics["pr_auc"] = float(
                        average_precision_score(
                            binarized_target,
                            probabilities,
                            average="macro",
                        )
                    )
    return metrics, prediction_frame
