"""Anomaly-first chronological experiment path for the NASA IMS bearing dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import pandas as pd
from pandas.errors import EmptyDataError

from .config import ExperimentConfig
from .evaluation import select_feature_columns
from .models import score_anomaly_sequence, train_anomaly_detector, train_one_class_svm_detector
from .nasa_ims_adapter import build_nasa_ims_compact_dataset
from .plots import plot_rtf_anomaly_trend
from .utils import prepare_experiment_directories, save_json, save_yaml


NON_MODEL_COLUMNS = {
    "dataset_name",
    "dataset_variant",
    "dataset_display_name",
    "session_id",
    "group_id",
    "run_key",
    "label",
    "multiclass_label",
    "reference_region_role",
    "bearing_id",
    "source_run_path",
    "source_file_name",
    "source_relative_path",
    "snapshot_timestamp",
    "progression_index",
    "elapsed_minutes",
    "elapsed_hours",
    "relative_progress",
    "nominal_snapshot_duration_sec",
    "window_index",
    "window_start",
    "window_end",
    "window_pairing_strategy",
    "split_group",
    "axis_count",
    "channel_indices",
    "delta_minutes_from_previous",
    "is_nominal_interval",
    "progression_gap_flag",
    "layout_type",
    "layout_warning",
    "documented_failure_mode",
    "documented_failed_bearing",
    "documented_failure_metadata_source",
    "documented_failure_metadata_confidence",
    "documented_failure_notes",
}
MODEL_BUILDERS = {
    "isolation_forest": train_anomaly_detector,
    "one_class_svm": train_one_class_svm_detector,
}


def _load_json_if_exists(path: Path) -> Dict[str, object]:
    """Load a JSON file if present, otherwise return an empty dictionary."""

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _default_feature_dataset_path(config: ExperimentConfig) -> Path:
    """Resolve the processed NASA IMS feature dataset path."""

    if config.paths.feature_dataset_path:
        return Path(config.paths.feature_dataset_path)
    return Path(config.paths.processed_root) / "nasa_ims" / "datasets" / "nasa_ims_bearing_feature_dataset.csv"


def _load_or_build_feature_dataset(config: ExperimentConfig) -> Tuple[Path, Dict[str, object]]:
    """Prefer the processed dataset and rebuild it only if missing."""

    feature_dataset_path = _default_feature_dataset_path(config)
    processed_root = Path(config.paths.processed_root) / "nasa_ims"
    adapter_summary_path = processed_root / "adapter_summary.json"

    if feature_dataset_path.exists():
        adapter_summary = _load_json_if_exists(adapter_summary_path)
        if not adapter_summary:
            adapter_summary = {
                "dataset_name": "nasa_ims",
                "status": "existing_feature_dataset",
                "feature_dataset_path": str(feature_dataset_path),
            }
        return feature_dataset_path, adapter_summary

    adapter_summary = build_nasa_ims_compact_dataset(config)
    return Path(adapter_summary["feature_dataset_path"]), adapter_summary


def _load_feature_dataset(feature_dataset_path: Path) -> pd.DataFrame:
    """Load the compact NASA IMS bearing feature dataset."""

    if not feature_dataset_path.exists():
        raise FileNotFoundError(f"NASA IMS feature dataset not found: {feature_dataset_path}")
    try:
        frame = pd.read_csv(feature_dataset_path, low_memory=False)
    except EmptyDataError:
        return pd.DataFrame()
    if frame.empty:
        return frame
    return frame.sort_values(["group_id", "session_id", "progression_index"]).reset_index(drop=True)


def _reference_row_count(n_rows: int, config: ExperimentConfig) -> int:
    """Return the early healthy-reference row count for one bearing session."""

    fraction_count = max(1, int(round(float(n_rows) * float(config.run_to_failure.healthy_reference_fraction))))
    minimum_count = max(1, int(config.run_to_failure.minimum_reference_files))
    count = max(fraction_count, minimum_count)
    if n_rows <= 1:
        return 1
    return min(count, n_rows - 1)


def _build_reference_frame(
    feature_frame: pd.DataFrame,
    config: ExperimentConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Select an early chronological reference region for each bearing session."""

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

        reference_subset["reference_region_role"] = "calibration"
        reference_subset["reference_label"] = "healthy"
        reference_rows.append(reference_subset)

        summary_rows.append(
            {
                "group_id": str(reference_subset["group_id"].iloc[0]),
                "session_id": str(session_id),
                "bearing_id": int(reference_subset["bearing_id"].iloc[0]),
                "run_key": str(reference_subset["run_key"].iloc[0]),
                "reference_mode": reference_mode,
                "reference_file_count": int(len(reference_subset)),
                "session_file_count": int(len(ordered)),
                "reference_fraction_of_session": float(len(reference_subset) / max(len(ordered), 1)),
                "reference_start_index": int(reference_subset["progression_index"].min()),
                "reference_end_index": int(reference_subset["progression_index"].max()),
                "reference_start_hour": float(reference_subset["elapsed_hours"].min()),
                "reference_end_hour": float(reference_subset["elapsed_hours"].max()),
                "evaluation_file_count": int(len(ordered) - len(reference_subset)),
                "documented_failure_mode": reference_subset["documented_failure_mode"].iloc[0]
                if "documented_failure_mode" in reference_subset.columns
                else None,
                "documented_failed_bearing": reference_subset["documented_failed_bearing"].iloc[0]
                if "documented_failed_bearing" in reference_subset.columns
                else None,
                "documented_failure_metadata_confidence": reference_subset["documented_failure_metadata_confidence"].iloc[0]
                if "documented_failure_metadata_confidence" in reference_subset.columns
                else None,
            }
        )

    reference_frame = pd.concat(reference_rows, ignore_index=True) if reference_rows else pd.DataFrame()
    reference_summary_frame = pd.DataFrame(summary_rows)
    return reference_frame, reference_summary_frame


def _reference_lookup(reference_frame: pd.DataFrame) -> set[Tuple[str, int]]:
    """Build a stable lookup for calibration rows."""

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
    """Score the full chronology and derive rolling anomaly trend signals."""

    scored = score_anomaly_sequence(artifact=artifact, frame=feature_frame)
    trend_frame = feature_frame[
        [
            column
            for column in [
                "group_id",
                "session_id",
                "run_key",
                "bearing_id",
                "source_run_path",
                "source_file_name",
                "source_relative_path",
                "snapshot_timestamp",
                "progression_index",
                "elapsed_minutes",
                "elapsed_hours",
                "relative_progress",
                "window_start",
                "window_end",
                "channel_indices",
                "documented_failure_mode",
                "documented_failed_bearing",
                "documented_failure_metadata_confidence",
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
    trend_frame["reference_region_role"] = [
        "calibration" if (str(session_id), int(progression_index)) in reference_points else "evaluation"
        for session_id, progression_index in zip(
            trend_frame["session_id"].astype(str),
            trend_frame["progression_index"].astype(int),
        )
    ]
    trend_frame["is_reference_baseline"] = trend_frame["reference_region_role"].astype(str) == "calibration"
    trend_frame["post_reference_region"] = trend_frame["reference_region_role"].astype(str) == "evaluation"
    trend_frame["threshold_crossed"] = (
        trend_frame["anomaly_score_raw"].to_numpy(dtype=float) >= threshold
    ).astype(int)
    trend_frame["post_reference_threshold_crossed"] = (
        trend_frame["threshold_crossed"].astype(bool) & trend_frame["post_reference_region"].astype(bool)
    )

    trend_frame = trend_frame.sort_values(["group_id", "session_id", "progression_index"]).reset_index(drop=True)
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


def _first_event(frame: pd.DataFrame, mask_column: str) -> Tuple[int | None, float | None]:
    """Return the first progression index and elapsed hour where a mask becomes true."""

    flagged = frame.loc[frame[mask_column].astype(bool)]
    if flagged.empty:
        return None, None
    return int(flagged.iloc[0]["progression_index"]), float(flagged.iloc[0]["elapsed_hours"])


def _build_threshold_crossing_summary(
    trend_frame: pd.DataFrame,
    reference_summary_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize threshold crossings and sustained warnings per model and bearing session."""

    rows: List[Dict[str, object]] = []
    reference_lookup = {
        str(row["session_id"]): row for row in reference_summary_frame.to_dict(orient="records")
    }

    for (model_name, session_id), session_frame in trend_frame.groupby(["model_name", "session_id"], sort=True):
        ordered = session_frame.sort_values("progression_index").reset_index(drop=True)
        reference_info = reference_lookup.get(str(session_id), {})
        evaluation_frame = ordered.loc[ordered["reference_region_role"].astype(str) == "evaluation"].copy()

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
                "group_id": str(ordered["group_id"].iloc[0]),
                "session_id": str(session_id),
                "run_key": str(ordered["run_key"].iloc[0]) if "run_key" in ordered.columns else "",
                "bearing_id": int(ordered["bearing_id"].iloc[0]) if "bearing_id" in ordered.columns else None,
                "calibration_mode": str(reference_info.get("reference_mode", "unknown")),
                "calibration_file_count": int(reference_info.get("reference_file_count", 0)),
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
                "documented_failure_mode": ordered["documented_failure_mode"].iloc[0]
                if "documented_failure_mode" in ordered.columns
                else None,
                "documented_failed_bearing": ordered["documented_failed_bearing"].iloc[0]
                if "documented_failed_bearing" in ordered.columns
                else None,
                "documented_failure_metadata_confidence": ordered["documented_failure_metadata_confidence"].iloc[0]
                if "documented_failure_metadata_confidence" in ordered.columns
                else None,
                "status": "alarm_detected" if first_alarm_index is not None else "no_alarm_detected",
            }
        )

    return pd.DataFrame(rows)


def _build_model_summary_frame(
    artifacts: Iterable[Dict[str, object]],
    threshold_crossing_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Collect one compact model-level summary row per anomaly baseline."""

    rows: List[Dict[str, object]] = []
    for artifact in artifacts:
        model_name = str(artifact.get("model_name", "unknown"))
        calibration = artifact["calibration"]
        model_crossings = threshold_crossing_summary[
            threshold_crossing_summary["model_name"].astype(str) == model_name
        ].copy()
        sessions_with_alarm = int(model_crossings["first_alarm_hour"].notna().sum()) if not model_crossings.empty else 0
        rows.append(
            {
                "model_name": model_name,
                "n_reference_files": int(artifact["n_healthy_train_windows"]),
                "n_reference_sessions": int(len(artifact.get("healthy_train_sessions", []))),
                "calibration_strategy": str(calibration["strategy"]),
                "calibrated_threshold": float(artifact["threshold"]),
                "healthy_score_mean": float(calibration["healthy_score_mean"]),
                "healthy_score_std": float(calibration["healthy_score_std"]),
                "healthy_score_q95": float(calibration["healthy_score_q95"]),
                "healthy_score_q99": float(calibration["healthy_score_q99"]),
                "sessions_with_alarm": sessions_with_alarm,
            }
        )
    return pd.DataFrame(rows)


def _write_experiment_summary_markdown(
    config: ExperimentConfig,
    adapter_summary: Dict[str, object],
    feature_frame: pd.DataFrame,
    reference_summary_frame: pd.DataFrame,
    model_summary_frame: pd.DataFrame,
    threshold_crossing_summary: pd.DataFrame,
    n_model_features: int,
    n_dropped_all_missing_features: int,
    output_path: Path,
) -> None:
    """Write a compact thesis-ready NASA IMS anomaly-first experiment summary."""

    lines = [
        f"# Experiment Summary: {config.experiment_name}",
        "",
        "## Problem Setting",
        "- Dataset: `nasa_ims`",
        "- Analysis type: anomaly-first bearing-level degradation analysis",
        "- Signals used: vibration only",
        "- Session semantics: one bearing trajectory per session, one test run per group",
        "- Supervised classification: intentionally not used",
        "",
        "## Data Summary",
        f"- Feature dataset: `{_default_feature_dataset_path(config)}`",
        f"- Discovered runs: {adapter_summary.get('n_runs', 0)}",
        f"- Bearing sessions: {feature_frame['session_id'].nunique() if not feature_frame.empty else 0}",
        f"- Feature rows: {len(feature_frame)}",
        f"- Model features used: {n_model_features}",
        f"- All-missing features dropped before calibration: {n_dropped_all_missing_features}",
        "",
        "## Calibration Region",
        f"- Healthy reference fraction: {config.run_to_failure.healthy_reference_fraction}",
        f"- Healthy reference max hours: {config.run_to_failure.healthy_reference_max_hours}",
        f"- Minimum reference files: {config.run_to_failure.minimum_reference_files}",
        f"- Rolling window size: {config.run_to_failure.rolling_window_files} snapshots",
        f"- Warning ratio threshold: {config.inference.warning_ratio_threshold}",
    ]
    for row in reference_summary_frame.head(12).to_dict(orient="records"):
        lines.append(
            "- "
            f"{row['session_id']}: {row['reference_file_count']} calibration snapshots, "
            f"hours {row['reference_start_hour']} to {row['reference_end_hour']}, "
            f"mode=`{row['reference_mode']}`"
        )

    lines.extend(["", "## Model Outcomes"])
    if model_summary_frame.empty:
        lines.append("- No anomaly models were trained.")
    else:
        for row in model_summary_frame.to_dict(orient="records"):
            lines.append(
                "- "
                f"{row['model_name']}: threshold={row['calibrated_threshold']:.6f}, "
                f"reference_files={row['n_reference_files']}, sessions_with_alarm={row['sessions_with_alarm']}"
            )

    lines.extend(["", "## First Alarm Behavior"])
    if threshold_crossing_summary.empty:
        lines.append("- No threshold crossing results were available.")
    else:
        for row in threshold_crossing_summary.head(12).to_dict(orient="records"):
            lines.append(
                "- "
                f"{row['model_name']} / {row['session_id']}: first_alarm_hour={row['first_alarm_hour']}, "
                f"reason={row['first_alarm_reason']}, documented_failure_mode={row['documented_failure_mode']}"
            )

    lines.extend(
        [
            "",
            "## Conservative Interpretation",
            "- Snapshot-level labels remain unknown throughout the experiment.",
            "- Documented end-of-run bearing failures are stored separately as metadata and are not used as dense targets.",
            "- Threshold crossings indicate deviation from the early calibrated regime, not a verified failure prediction time.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_nasa_ims_experiment(config: ExperimentConfig) -> Dict[str, object]:
    """Run the first compact anomaly-first experiment for NASA IMS."""

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
            "note": "No processed NASA IMS bearing feature dataset was available for anomaly analysis.",
        }
        save_json(directories["root"] / "experiment_summary.json", summary)
        pd.DataFrame([summary]).to_csv(directories["root"] / "experiment_summary.csv", index=False)
        _write_experiment_summary_markdown(
            config=config,
            adapter_summary=adapter_summary,
            feature_frame=feature_frame,
            reference_summary_frame=pd.DataFrame(),
            model_summary_frame=pd.DataFrame(),
            threshold_crossing_summary=pd.DataFrame(),
            n_model_features=0,
            n_dropped_all_missing_features=0,
            output_path=directories["root"] / "experiment_summary.md",
        )
        return summary

    feature_frame.to_csv(directories["datasets"] / "nasa_ims_bearing_feature_dataset.csv", index=False)

    model_frame = feature_frame.drop(columns=[column for column in NON_MODEL_COLUMNS if column in feature_frame.columns])
    feature_columns = select_feature_columns(model_frame)
    dropped_all_missing_features = sorted(column for column in feature_columns if feature_frame[column].isna().all())
    if dropped_all_missing_features:
        pd.DataFrame({"feature_name": dropped_all_missing_features}).to_csv(
            directories["root"] / "dropped_all_missing_features.csv",
            index=False,
        )
    feature_columns = [column for column in feature_columns if column not in dropped_all_missing_features]
    if not feature_columns:
        raise RuntimeError("No numeric model features were available for the NASA IMS experiment.")

    reference_frame, reference_summary_frame = _build_reference_frame(feature_frame, config)
    if reference_frame.empty:
        raise RuntimeError("No reference region was available for NASA IMS anomaly calibration.")

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
        ["model_name", "group_id", "session_id", "progression_index"]
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
        "n_runs": int(feature_frame["group_id"].nunique()),
        "n_sessions": int(feature_frame["session_id"].nunique()),
        "n_feature_rows": int(len(feature_frame)),
        "n_reference_files": int(len(reference_frame)),
        "n_model_features": int(len(feature_columns)),
        "n_dropped_all_missing_features": int(len(dropped_all_missing_features)),
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
        threshold_crossing_summary=threshold_crossing_summary,
        n_model_features=len(feature_columns),
        n_dropped_all_missing_features=len(dropped_all_missing_features),
        output_path=directories["root"] / "experiment_summary.md",
    )
    return summary
