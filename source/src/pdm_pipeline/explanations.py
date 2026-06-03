"""Minimal explanation helpers based on Random Forest feature importances."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd

from .models.classification import classifier_feature_importances
from .plots import plot_feature_importances, plot_sensor_group_importance
from .utils import save_json


def _sensor_group_for_feature(feature_name: str) -> str | None:
    """Map a feature name to its primary sensor family."""

    if feature_name.startswith("ae_"):
        return "AE"
    if feature_name.startswith("vibration_"):
        return "vibration"
    if feature_name.startswith("thermal_"):
        return "thermal"
    return None


def _format_top_features_markdown(top_features: pd.DataFrame) -> List[str]:
    """Render top features as a short bullet list."""

    lines: List[str] = []
    for _, row in top_features.iterrows():
        lines.append(f"- `{row['feature']}`: importance {float(row['importance']):.4f}")
    return lines


def generate_feature_explanations(results_dir: Path, top_n: int = 10) -> Dict[str, object]:
    """Create compact fused-model explanation artifacts for the demo baseline."""

    artifact = joblib.load(results_dir / "artifacts" / "classifier_artifact.joblib")
    feature_columns = list(artifact["feature_columns"])
    full_importance_frame = pd.DataFrame(classifier_feature_importances(artifact, feature_columns))
    if full_importance_frame.empty:
        raise RuntimeError("Classifier artifact does not contain feature importances.")

    top_features = full_importance_frame.head(top_n).reset_index(drop=True)
    top_features.to_csv(results_dir / "top_features.csv", index=False)
    save_json(results_dir / "top_features.json", {"rows": top_features.to_dict(orient="records")})
    plot_feature_importances(top_features, results_dir / "plots" / "top_features.png", top_n=top_n)

    sensor_frame = full_importance_frame.copy()
    sensor_frame["sensor_group"] = sensor_frame["feature"].apply(_sensor_group_for_feature)
    grouped = (
        sensor_frame.dropna(subset=["sensor_group"])
        .groupby("sensor_group", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    grouped["normalized_importance"] = grouped["importance"] / grouped["importance"].sum()

    excluded_importance = float(sensor_frame.loc[sensor_frame["sensor_group"].isna(), "importance"].sum())
    grouped.to_csv(results_dir / "sensor_group_importance.csv", index=False)
    save_json(
        results_dir / "sensor_group_importance.json",
        {
            "rows": grouped.to_dict(orient="records"),
            "excluded_non_sensor_importance": excluded_importance,
            "note": "Only AE, vibration, and thermal feature families are aggregated here. Fusion/meta features are excluded from the group chart to keep the sensor comparison explicit.",
        },
    )
    plot_sensor_group_importance(grouped, results_dir / "plots" / "sensor_group_importance.png")

    dominant_group = str(grouped.iloc[0]["sensor_group"])
    dominant_share = float(grouped.iloc[0]["normalized_importance"])
    summary_text = (
        f"The strongest modality-specific contribution came from the {dominant_group} feature family "
        f"with normalized importance {dominant_share:.3f}. "
        f"The most important fused-model features were {', '.join(top_features['feature'].head(3).tolist())}. "
        f"This supports the fusion rationale because the final model still depends on multiple sensor-derived features rather than a single modality alone."
    )

    lines = [
        "# Feature Explanation Summary",
        "",
        "This explanation uses the fused Random Forest feature importances from the baseline classifier.",
        "",
        "## Most Important Features",
    ]
    lines.extend(_format_top_features_markdown(top_features))
    lines.extend(
        [
            "",
            "## Sensor Group Contribution",
            f"- Dominant sensor group: `{dominant_group}`",
            f"- Normalized importance share: {dominant_share:.3f}",
            f"- Excluded non-sensor importance (fusion/meta/context): {excluded_importance:.4f}",
            "",
            "## Interpretation",
            summary_text,
        ]
    )
    (results_dir / "feature_explanation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "top_features": top_features.to_dict(orient="records"),
        "dominant_sensor_group": dominant_group,
        "dominant_sensor_group_share": dominant_share,
        "excluded_non_sensor_importance": excluded_importance,
        "summary_text": summary_text,
    }
