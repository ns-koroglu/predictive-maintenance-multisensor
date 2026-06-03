"""Anomaly-first chronological experiment path for the KAIST run-to-failure dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from .config import ExperimentConfig
from .evaluation import select_feature_columns
from .kaist_rtf_adapter import build_kaist_rtf_compact_dataset
from .models import (
    score_anomaly_sequence,
    train_anomaly_detector,
    train_one_class_svm_detector,
)
from .plots import plot_rtf_anomaly_trend
from .utils import prepare_experiment_directories, save_json, save_yaml


NON_MODEL_COLUMNS = {
    "source_file_name",
    "source_relative_path",
    "progression_index",
    "elapsed_hours",
    "relative_progress",
    "measurement_duration_sec",
    "sampling_rate_hz",
    "n_samples",
}
MODEL_BUILDERS = {
    "isolation_forest": train_anomaly_detector,
    "one_class_svm": train_one_class_svm_detector,
}


def _load_feature_dataset(feature_dataset_path: Path) -> pd.DataFrame:
    """Load a compact run-to-failure feature dataset if it exists."""

    if not feature_dataset_path.exists():
        raise FileNotFoundError(f"Run-to-failure feature dataset not found: {feature_dataset_path}")
    try:
        frame = pd.read_csv(feature_dataset_path)
    except EmptyDataError:
        return pd.DataFrame()
    if frame.empty:
        return frame
    return frame.sort_values(["session_id", "progression_index"]).reset_index(drop=True)


def _load_json_if_exists(path: Path) -> Dict[str, object]:
    """Load a JSON file when it exists, otherwise return an empty dictionary."""

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _default_feature_dataset_path(config: ExperimentConfig) -> Path:
    """Resolve the processed feature dataset path used by the experiment."""

    if config.paths.feature_dataset_path:
        return Path(config.paths.feature_dataset_path)
    return Path(config.paths.processed_root) / "kaist_run_to_failure" / "datasets" / "kaist_rtf_feature_dataset.csv"


def _load_or_build_feature_dataset(config: ExperimentConfig) -> Tuple[Path, Dict[str, object]]:
    """Prefer the processed feature dataset and only rebuild it if missing."""

    feature_dataset_path = _default_feature_dataset_path(config)
    processed_root = Path(config.paths.processed_root) / "kaist_run_to_failure"
    adapter_summary_path = processed_root / "adapter_summary.json"

    if feature_dataset_path.exists():
        adapter_summary = _load_json_if_exists(adapter_summary_path)
        if not adapter_summary:
            adapter_summary = {
                "dataset_name": "kaist_run_to_failure",
                "status": "existing_feature_dataset",
                "feature_dataset_path": str(feature_dataset_path),
            }
        return feature_dataset_path, adapter_summary

    adapter_summary = build_kaist_rtf_compact_dataset(config)
    return Path(adapter_summary["feature_dataset_path"]), adapter_summary


def _reference_row_count(n_rows: int, config: ExperimentConfig) -> int:
    """Decide how many early files should define the healthy reference region."""

    fraction_count = int(np.ceil(float(n_rows) * float(config.run_to_failure.healthy_reference_fraction)))
    minimum_count = int(config.run_to_failure.minimum_reference_files)
    count = max(1, max(fraction_count, minimum_count))
    return min(count, max(1, n_rows - 1)) if n_rows > 1 else 1


def _build_reference_frame(
    feature_frame: pd.DataFrame,
    config: ExperimentConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Select the early progression region used for healthy-reference calibration."""

    reference_rows: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, object]] = []

    for session_id, session_frame in feature_frame.groupby("session_id", sort=True):
        ordered = session_frame.sort_values("progression_index").reset_index(drop=True).copy()
        max_reference_rows = max(1, len(ordered) - 1) if len(ordered) > 1 else 1
        minimum_count = min(max_reference_rows, max(1, int(config.run_to_failure.minimum_reference_files)))
        reference_mode = "early_fraction_or_minimum"

        max_hours = config.run_to_failure.healthy_reference_max_hours
        if max_hours is not None and float(max_hours) > 0:
            reference_subset = ordered.loc[ordered["elapsed_hours"].astype(float) < float(max_hours)].copy()
            reference_mode = f"first_{float(max_hours):g}_hours"
            if reference_subset.empty or len(reference_subset) < minimum_count:
                reference_count = min(max_reference_rows, max(minimum_count, len(reference_subset)))
                reference_subset = ordered.head(reference_count).copy()
                reference_mode = f"{reference_mode}_with_minimum_extension"
            else:
                reference_subset = reference_subset.head(max_reference_rows).copy()
        else:
            reference_count = _reference_row_count(len(ordered), config)
            reference_subset = ordered.head(reference_count).copy()

        reference_subset["reference_label"] = "healthy"
        reference_subset["reference_mode"] = reference_mode
        reference_subset["is_reference_baseline"] = True
        reference_rows.append(reference_subset)

        summary_rows.append(
            {
                "session_id": str(session_id),
                "reference_mode": reference_mode,
                "reference_file_count": int(len(reference_subset)),
                "session_file_count": int(len(ordered)),
                "reference_fraction_of_run": float(len(reference_subset) / max(len(ordered), 1)),
                "reference_start_index": int(reference_subset["progression_index"].min()),
                "reference_end_index": int(reference_subset["progression_index"].max()),
                "reference_start_hour": float(reference_subset["elapsed_hours"].min()),
                "reference_end_hour": float(reference_subset["elapsed_hours"].max()),
                "evaluation_file_count": int(len(ordered) - len(reference_subset)),
            }
        )

    reference_frame = (
        pd.concat(reference_rows, ignore_index=True)
        if reference_rows
        else pd.DataFrame(columns=list(feature_frame.columns) + ["reference_label", "reference_mode", "is_reference_baseline"])
    )
    reference_summary = pd.DataFrame(summary_rows)
    return reference_frame, reference_summary


def _reference_lookup(reference_frame: pd.DataFrame) -> set[Tuple[str, int]]:
    """Create a stable lookup for identifying reference rows in the full run."""

    return set(
        zip(
            reference_frame["session_id"].astype(str),
            reference_frame["progression_index"].astype(int),
        )
    )


def _build_model_trend_frame(
    artifact: Dict[str, object],
    feature_frame: pd.DataFrame,
    reference_points: set[Tuple[str, int]],
    rolling_window_files: int,
    warning_ratio_threshold: float,
) -> pd.DataFrame:
    """Score the chronological run and derive compact early-warning signals."""

    scored = score_anomaly_sequence(artifact=artifact, frame=feature_frame)
    trend_frame = feature_frame[
        [
            column
            for column in [
                "session_id",
                "group_id",
                "source_file_name",
                "source_relative_path",
                "progression_index",
                "elapsed_hours",
                "relative_progress",
                "window_start",
                "window_end",
            ]
            if column in feature_frame.columns
        ]
    ].copy()

    scores = scored["anomaly_score"].to_numpy(dtype=float)
    calibration = artifact["calibration"]
    healthy_mean = float(calibration["healthy_score_mean"])
    healthy_std = float(calibration["healthy_score_std"])
    healthy_std = healthy_std if healthy_std > 1e-12 else 1.0
    threshold = float(artifact["threshold"])

    trend_frame["model_name"] = str(artifact.get("model_name", "isolation_forest"))
    trend_frame["anomaly_score_raw"] = scores
    trend_frame["anomaly_score_reference_z"] = (scores - healthy_mean) / healthy_std
    trend_frame["calibrated_threshold"] = threshold
    trend_frame["threshold_reference_z"] = (threshold - healthy_mean) / healthy_std
    trend_frame["is_reference_baseline"] = [
        (str(session_id), int(progression_index)) in reference_points
        for session_id, progression_index in zip(
            trend_frame["session_id"].astype(str),
            trend_frame["progression_index"].astype(int),
        )
    ]
    trend_frame["post_reference_region"] = ~trend_frame["is_reference_baseline"].astype(bool)
    trend_frame["threshold_crossed"] = (
        trend_frame["anomaly_score_raw"].to_numpy(dtype=float) >= threshold
    ).astype(int)
    trend_frame["post_reference_threshold_crossed"] = (
        trend_frame["threshold_crossed"].astype(bool) & trend_frame["post_reference_region"].astype(bool)
    )

    trend_frame = trend_frame.sort_values(["session_id", "progression_index"]).reset_index(drop=True)
    trend_frame["rolling_anomaly_score_mean"] = (
        trend_frame.groupby("session_id")["anomaly_score_raw"]
        .transform(lambda series: series.rolling(window=rolling_window_files, min_periods=1).mean())
    )
    trend_frame["rolling_anomaly_score_z_mean"] = (
        trend_frame.groupby("session_id")["anomaly_score_reference_z"]
        .transform(lambda series: series.rolling(window=rolling_window_files, min_periods=1).mean())
    )
    trend_frame["rolling_anomaly_flag_ratio"] = (
        trend_frame.groupby("session_id")["threshold_crossed"]
        .transform(lambda series: series.rolling(window=rolling_window_files, min_periods=1).mean())
    )
    trend_frame["sustained_warning"] = (
        trend_frame["post_reference_region"].astype(bool)
        & (trend_frame["rolling_anomaly_flag_ratio"].to_numpy(dtype=float) >= float(warning_ratio_threshold))
    )
    return trend_frame


def _first_event(
    frame: pd.DataFrame,
    mask_column: str,
) -> Tuple[int | None, float | None]:
    """Return the first progression index and elapsed hour where a boolean mask is true."""

    flagged = frame.loc[frame[mask_column].astype(bool)]
    if flagged.empty:
        return None, None
    return int(flagged.iloc[0]["progression_index"]), float(flagged.iloc[0]["elapsed_hours"])


def _build_threshold_crossing_summary(
    trend_frame: pd.DataFrame,
    reference_summary_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize threshold crossings and sustained warnings per model and session."""

    rows: List[Dict[str, object]] = []
    reference_lookup = {
        str(row["session_id"]): row for row in reference_summary_frame.to_dict(orient="records")
    }

    for (model_name, session_id), session_frame in trend_frame.groupby(["model_name", "session_id"], sort=True):
        ordered = session_frame.sort_values("progression_index").reset_index(drop=True)
        reference_info = reference_lookup.get(str(session_id), {})
        evaluation_frame = ordered.loc[ordered["post_reference_region"].astype(bool)].copy()

        first_cross_index, first_cross_hour = _first_event(evaluation_frame, "post_reference_threshold_crossed")
        first_warning_index, first_warning_hour = _first_event(evaluation_frame, "sustained_warning")
        if first_warning_index is not None:
            first_alarm_index = first_warning_index
            first_alarm_hour = first_warning_hour
            first_alarm_reason = "sustained_warning"
        elif first_cross_index is not None:
            first_alarm_index = first_cross_index
            first_alarm_hour = first_cross_hour
            first_alarm_reason = "threshold_crossing"
        else:
            first_alarm_index = None
            first_alarm_hour = None
            first_alarm_reason = "no_alarm"

        rows.append(
            {
                "model_name": str(model_name),
                "session_id": str(session_id),
                "calibration_mode": str(reference_info.get("reference_mode", "unknown")),
                "calibration_file_count": int(reference_info.get("reference_file_count", 0)),
                "calibration_start_index": reference_info.get("reference_start_index"),
                "calibration_end_index": reference_info.get("reference_end_index"),
                "calibration_start_hour": reference_info.get("reference_start_hour"),
                "calibration_end_hour": reference_info.get("reference_end_hour"),
                "evaluation_file_count": int(reference_info.get("evaluation_file_count", len(evaluation_frame))),
                "calibrated_threshold": float(ordered["calibrated_threshold"].iloc[0]),
                "first_threshold_crossing_index": first_cross_index,
                "first_threshold_crossing_hour": first_cross_hour,
                "first_sustained_warning_index": first_warning_index,
                "first_sustained_warning_hour": first_warning_hour,
                "first_alarm_index": first_alarm_index,
                "first_alarm_hour": first_alarm_hour,
                "first_alarm_reason": first_alarm_reason,
                "post_reference_crossing_count": int(evaluation_frame["post_reference_threshold_crossed"].astype(bool).sum()),
                "post_reference_flag_ratio_mean": float(evaluation_frame["threshold_crossed"].mean())
                if not evaluation_frame.empty
                else 0.0,
                "max_anomaly_score_raw": float(ordered["anomaly_score_raw"].max()),
                "max_anomaly_score_reference_z": float(ordered["anomaly_score_reference_z"].max()),
                "final_rolling_anomaly_score_mean": float(ordered["rolling_anomaly_score_mean"].iloc[-1]),
                "final_rolling_anomaly_score_z_mean": float(ordered["rolling_anomaly_score_z_mean"].iloc[-1]),
                "final_rolling_anomaly_flag_ratio": float(ordered["rolling_anomaly_flag_ratio"].iloc[-1]),
                "status": "alarm_detected" if first_alarm_index is not None else "no_alarm_detected",
            }
        )
    return pd.DataFrame(rows)


def _build_model_summary_frame(
    artifacts: Iterable[Dict[str, object]],
    threshold_crossing_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Collect calibration and early-warning outcomes in one compact table."""

    rows: List[Dict[str, object]] = []
    for artifact in artifacts:
        model_name = str(artifact.get("model_name", "unknown"))
        calibration = artifact["calibration"]
        model_crossings = threshold_crossing_summary[
            threshold_crossing_summary["model_name"].astype(str) == model_name
        ].copy()
        if model_crossings.empty:
            crossing: Dict[str, object] = {}
        else:
            with_alarm = model_crossings[model_crossings["first_alarm_hour"].notna()].copy()
            crossing = (
                with_alarm.sort_values("first_alarm_hour").iloc[0].to_dict()
                if not with_alarm.empty
                else model_crossings.iloc[0].to_dict()
            )
        rows.append(
            {
                "model_name": model_name,
                "n_reference_files": int(artifact["n_healthy_train_windows"]),
                "calibration_strategy": str(calibration["strategy"]),
                "calibrated_threshold": float(artifact["threshold"]),
                "healthy_score_mean": float(calibration["healthy_score_mean"]),
                "healthy_score_std": float(calibration["healthy_score_std"]),
                "healthy_score_q95": float(calibration["healthy_score_q95"]),
                "healthy_score_q99": float(calibration["healthy_score_q99"]),
                "first_alarm_index": crossing.get("first_alarm_index"),
                "first_alarm_hour": crossing.get("first_alarm_hour"),
                "first_alarm_reason": crossing.get("first_alarm_reason"),
                "status": crossing.get("status", "unknown"),
            }
        )
    return pd.DataFrame(rows)


def _write_experiment_summary_markdown(
    config: ExperimentConfig,
    adapter_summary: Dict[str, object],
    feature_frame: pd.DataFrame,
    reference_summary_frame: pd.DataFrame,
    model_summary_frame: pd.DataFrame,
    n_model_features: int,
    n_dropped_all_missing_features: int,
    output_path: Path,
) -> None:
    """Write a compact experiment report focused on calibration and trend analysis."""

    lines = [
        f"# Experiment Summary: {config.experiment_name}",
        "",
        "## Problem Setting",
        "- Dataset: `kaist_run_to_failure`",
        "- Analysis type: anomaly-first chronological degradation analysis",
        "- Signals used: vibration x/y, bearing temperature, ambient temperature",
        "- Supervised classification: intentionally not used",
        "",
        "## Data Summary",
        f"- Feature dataset: `{_default_feature_dataset_path(config)}`",
        f"- Source files discovered: {adapter_summary.get('n_source_files', 0)}",
        f"- Sessions in feature dataset: {feature_frame['session_id'].nunique() if not feature_frame.empty else 0}",
        f"- Hourly feature rows: {len(feature_frame)}",
        f"- Model features used: {n_model_features}",
        f"- All-missing features dropped before calibration: {n_dropped_all_missing_features}",
        "",
        "## Calibration Region",
        f"- Healthy reference fraction: {config.run_to_failure.healthy_reference_fraction}",
        f"- Healthy reference max hours: {config.run_to_failure.healthy_reference_max_hours}",
        f"- Minimum reference files: {config.run_to_failure.minimum_reference_files}",
        f"- Rolling window size: {config.run_to_failure.rolling_window_files} files",
        f"- Warning ratio threshold: {config.inference.warning_ratio_threshold}",
    ]
    if not reference_summary_frame.empty:
        for row in reference_summary_frame.to_dict(orient="records"):
            lines.append(
                "- "
                f"{row['session_id']}: {row['reference_file_count']} calibration files "
                f"from hour {row['reference_start_hour']} to {row['reference_end_hour']} "
                f"using `{row['reference_mode']}`"
            )

    lines.extend(["", "## Model Outcomes"])
    if model_summary_frame.empty:
        lines.append("- No anomaly models were trained.")
    else:
        for row in model_summary_frame.to_dict(orient="records"):
            lines.append(
                "- "
                f"{row['model_name']}: threshold={row['calibrated_threshold']:.6f}, "
                f"first_alarm_hour={row['first_alarm_hour']}, reason={row['first_alarm_reason']}, "
                f"status={row['status']}"
            )

    lines.extend(
        [
            "",
            "## Conservative Interpretation",
            "- The calibration region is used for model fitting and threshold calibration only.",
            "- Later threshold crossings indicate deviation from the early run behavior, not a proven failure prediction timestamp.",
            "- Because explicit failure-onset labels are unavailable, the results should be interpreted as anomaly-trend evidence rather than validated RUL or fault-time estimation.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_early_warning_summary_markdown(
    config: ExperimentConfig,
    reference_summary_frame: pd.DataFrame,
    threshold_crossing_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write a publication-friendly early-warning summary for the chronological run."""

    lines = [
        f"# Early Warning Summary: {config.experiment_name}",
        "",
        "## Calibration Region",
        "- The anomaly models were trained only on the early reference region of the run.",
        "- This region is treated as the healthy operating baseline for score normalization and threshold calibration.",
    ]
    if reference_summary_frame.empty:
        lines.append("- No calibration region was available.")
    else:
        for row in reference_summary_frame.to_dict(orient="records"):
            lines.append(
                "- "
                f"{row['session_id']}: calibration files={row['reference_file_count']}, "
                f"hours={row['reference_start_hour']} to {row['reference_end_hour']}, "
                f"mode=`{row['reference_mode']}`"
            )

    lines.extend(["", "## Early Warning Results"])
    if threshold_crossing_summary.empty:
        lines.append("- No model results were available.")
    else:
        for row in threshold_crossing_summary.to_dict(orient="records"):
            lines.append(f"### {str(row['model_name']).replace('_', ' ').title()}")
            lines.append(f"- Calibrated threshold: {float(row['calibrated_threshold']):.6f}")
            lines.append(
                f"- First threshold crossing: index={row['first_threshold_crossing_index']}, "
                f"hour={row['first_threshold_crossing_hour']}"
            )
            lines.append(
                f"- First sustained warning: index={row['first_sustained_warning_index']}, "
                f"hour={row['first_sustained_warning_hour']}"
            )
            lines.append(f"- First reported alarm: index={row['first_alarm_index']}, hour={row['first_alarm_hour']}")
            lines.append(f"- Alarm reason: `{row['first_alarm_reason']}`")
            lines.append(f"- Final rolling anomaly-flag ratio: {float(row['final_rolling_anomaly_flag_ratio']):.3f}")

    lines.extend(
        [
            "",
            "## Conservative Reporting Note",
            "- These alarm points mark when the run first deviated from the calibrated healthy-reference regime.",
            "- They should not be described as verified failure-prediction times unless external failure-onset evidence is added later.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_kaist_rtf_experiment(config: ExperimentConfig) -> Dict[str, object]:
    """Run the first real anomaly-trend experiment for the KAIST run-to-failure dataset."""

    directories = prepare_experiment_directories(config.paths.results_root, config.experiment_name)
    save_yaml(directories["root"] / "config_snapshot.yaml", config.to_dict())

    feature_dataset_path, adapter_summary = _load_or_build_feature_dataset(config)
    feature_frame = _load_feature_dataset(feature_dataset_path)

    if feature_frame.empty:
        summary = {
            "experiment_name": config.experiment_name,
            "results_dir": str(directories["root"]),
            "feature_dataset_path": str(feature_dataset_path),
            "status": "no_data_available",
            "note": "No processed run-to-failure feature dataset was available for anomaly-trend analysis.",
        }
        save_json(directories["root"] / "experiment_summary.json", summary)
        pd.DataFrame([summary]).to_csv(directories["root"] / "experiment_summary.csv", index=False)
        _write_experiment_summary_markdown(
            config=config,
            adapter_summary=adapter_summary,
            feature_frame=feature_frame,
            reference_summary_frame=pd.DataFrame(),
            model_summary_frame=pd.DataFrame(),
            n_model_features=0,
            n_dropped_all_missing_features=0,
            output_path=directories["root"] / "experiment_summary.md",
        )
        _write_early_warning_summary_markdown(
            config=config,
            reference_summary_frame=pd.DataFrame(),
            threshold_crossing_summary=pd.DataFrame(),
            output_path=directories["root"] / "early_warning_summary.md",
        )
        return summary

    feature_frame.to_csv(directories["datasets"] / "kaist_rtf_feature_dataset.csv", index=False)

    model_frame = feature_frame.drop(columns=[column for column in NON_MODEL_COLUMNS if column in feature_frame.columns])
    feature_columns = select_feature_columns(model_frame)
    dropped_all_missing_features = sorted(
        column for column in feature_columns if feature_frame[column].isna().all()
    )
    if dropped_all_missing_features:
        pd.DataFrame({"feature_name": dropped_all_missing_features}).to_csv(
            directories["root"] / "dropped_all_missing_features.csv",
            index=False,
        )
    feature_columns = [column for column in feature_columns if column not in dropped_all_missing_features]
    if not feature_columns:
        raise RuntimeError("No numeric model features were available for the KAIST run-to-failure experiment.")

    reference_frame, reference_summary_frame = _build_reference_frame(feature_frame, config)
    if reference_frame.empty:
        raise RuntimeError("No reference files were available for anomaly calibration.")

    reference_frame.to_csv(directories["datasets"] / "healthy_reference_region.csv", index=False)
    reference_summary_frame.to_csv(directories["root"] / "reference_region_summary.csv", index=False)
    save_json(
        directories["root"] / "reference_region_summary.json",
        {"rows": reference_summary_frame.to_dict(orient="records")},
    )

    reference_points = _reference_lookup(reference_frame)
    rolling_window = int(config.run_to_failure.rolling_window_files)
    warning_ratio_threshold = float(config.inference.warning_ratio_threshold)

    artifacts: List[Dict[str, object]] = []
    trend_frames: List[pd.DataFrame] = []
    calibration_rows: List[Dict[str, object]] = []

    for model_name, builder in MODEL_BUILDERS.items():
        artifact = builder(
            train_frame=reference_frame,
            feature_columns=feature_columns,
            label_column="reference_label",
            healthy_label="healthy",
            config=config.model.anomaly,
        )
        artifacts.append(artifact)
        joblib.dump(artifact, directories["artifacts"] / f"{model_name}_artifact.joblib")
        joblib.dump(artifact["model"], directories["models"] / f"{model_name}.joblib")
        joblib.dump(artifact["preprocessor"], directories["artifacts"] / f"{model_name}_preprocessor.joblib")

        trend_frame = _build_model_trend_frame(
            artifact=artifact,
            feature_frame=feature_frame,
            reference_points=reference_points,
            rolling_window_files=rolling_window,
            warning_ratio_threshold=warning_ratio_threshold,
        )
        trend_frames.append(trend_frame)

        calibration_rows.append(
            {
                "model_name": str(artifact["model_name"]),
                "n_reference_files": int(artifact["n_healthy_train_windows"]),
                **artifact["calibration"],
            }
        )

    anomaly_trend_frame = pd.concat(trend_frames, ignore_index=True).sort_values(
        ["model_name", "session_id", "progression_index"]
    ).reset_index(drop=True)
    anomaly_trend_frame.to_csv(directories["root"] / "anomaly_trend.csv", index=False)
    anomaly_trend_frame.to_csv(directories["predictions"] / "anomaly_trend.csv", index=False)

    calibration_frame = pd.DataFrame(calibration_rows)
    calibration_frame.to_csv(directories["metrics"] / "anomaly_model_calibration.csv", index=False)
    save_json(
        directories["metrics"] / "anomaly_model_calibration.json",
        {"rows": calibration_frame.to_dict(orient="records")},
    )

    threshold_crossing_summary = _build_threshold_crossing_summary(
        trend_frame=anomaly_trend_frame,
        reference_summary_frame=reference_summary_frame,
    )
    threshold_crossing_summary.to_csv(directories["root"] / "threshold_crossing_summary.csv", index=False)
    threshold_crossing_summary.to_csv(directories["predictions"] / "threshold_crossing_summary.csv", index=False)
    save_json(
        directories["predictions"] / "threshold_crossing_summary.json",
        {"rows": threshold_crossing_summary.to_dict(orient="records")},
    )

    model_summary_frame = _build_model_summary_frame(artifacts=artifacts, threshold_crossing_summary=threshold_crossing_summary)
    model_summary_frame.to_csv(directories["root"] / "anomaly_model_summary.csv", index=False)
    save_json(
        directories["root"] / "anomaly_model_summary.json",
        {"rows": model_summary_frame.to_dict(orient="records")},
    )

    plot_rtf_anomaly_trend(
        trend_frame=anomaly_trend_frame,
        output_path=directories["root"] / "anomaly_trend.png",
        rolling_window_files=rolling_window,
    )

    summary = {
        "experiment_name": config.experiment_name,
        "results_dir": str(directories["root"]),
        "feature_dataset_path": str(feature_dataset_path),
        "n_sessions": int(feature_frame["session_id"].nunique()),
        "n_feature_rows": int(len(feature_frame)),
        "n_reference_files": int(len(reference_frame)),
        "n_model_features": int(len(feature_columns)),
        "n_dropped_all_missing_features": int(len(dropped_all_missing_features)),
        "reference_range_hours": reference_summary_frame[
            ["session_id", "reference_start_hour", "reference_end_hour", "reference_file_count"]
        ].to_dict(orient="records"),
        "model_names": model_summary_frame["model_name"].astype(str).tolist(),
        "status": "ok",
        "anomaly_trend_path": str(directories["root"] / "anomaly_trend.csv"),
        "threshold_crossing_summary_path": str(directories["root"] / "threshold_crossing_summary.csv"),
    }
    save_json(directories["root"] / "experiment_summary.json", summary)
    pd.DataFrame([summary]).to_csv(directories["root"] / "experiment_summary.csv", index=False)
    _write_experiment_summary_markdown(
        config=config,
        adapter_summary=adapter_summary,
        feature_frame=feature_frame,
        reference_summary_frame=reference_summary_frame,
        model_summary_frame=model_summary_frame,
        n_model_features=len(feature_columns),
        n_dropped_all_missing_features=len(dropped_all_missing_features),
        output_path=directories["root"] / "experiment_summary.md",
    )
    _write_early_warning_summary_markdown(
        config=config,
        reference_summary_frame=reference_summary_frame,
        threshold_crossing_summary=threshold_crossing_summary,
        output_path=directories["root"] / "early_warning_summary.md",
    )
    return summary
