"""Dataset splitting and feature-column selection helpers."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


NON_FEATURE_COLUMNS = {
    "dataset_name",
    "dataset_variant",
    "dataset_display_name",
    "session_id",
    "group_id",
    "bearing_code",
    "condition_code",
    "operating_condition_n",
    "operating_condition_m",
    "operating_condition_f",
    "replicate_index",
    "run_key",
    "bearing_id",
    "condition_key",
    "label",
    "multiclass_label",
    "reference_region_role",
    "window_index",
    "window_start",
    "window_end",
    "nominal_snapshot_duration_sec",
    "start_time",
    "end_time",
    "snapshot_timestamp",
    "sample_index",
    "split_group",
    "window_pairing_strategy",
    "channel_indices",
    "axis_count",
    "documented_failed_bearing",
    "documented_failure_mode",
    "documented_failure_metadata_confidence",
    "fault_component_normalized",
    "fault_origin_normalized",
    "fault_component_raw",
    "fault_origin_raw",
    "source_file_name",
    "source_relative_path",
    "source_folder_layout",
    "layout_warning",
    "documented_fault_notes",
    "nominal_record_duration_sec",
    "sampling_rate_vibration_hz",
    "sampling_rate_current_hz",
    "sampling_rate_mechanical_hz",
    "sampling_rate_temperature_hz",
    "record_status",
    "note",
    "notes",
    "lubrication_state",
}
NON_FEATURE_PREFIXES = (
    "diag_",
    "has_",
    "vibration_sample_index_",
    "thermal_sample_index_",
    "ae_sample_index_",
    "current_sample_index_",
)


def select_feature_columns(frame: pd.DataFrame) -> List[str]:
    """Select numeric feature columns while excluding identifiers and diagnostics."""

    columns: List[str] = []
    for column in frame.columns:
        if column in NON_FEATURE_COLUMNS:
            continue
        if any(column.startswith(prefix) for prefix in NON_FEATURE_PREFIXES):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(column)
    return columns


def split_dataset_by_session(
    frame: pd.DataFrame,
    test_fraction: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform a session-level split so a session never appears in both sets."""

    if "session_id" not in frame.columns:
        raise ValueError("Session-aware splitting requires a 'session_id' column.")
    if frame["session_id"].nunique() < 2:
        raise ValueError(
            "Session-aware splitting requires at least two sessions. "
            "To allow leakage intentionally, switch evaluation.split_strategy to 'window'."
        )

    session_labels = frame[["session_id", "label"]].drop_duplicates().copy()
    rng = np.random.default_rng(random_state)

    train_sessions: List[str] = []
    test_sessions: List[str] = []
    for label, group in session_labels.groupby("label"):
        sessions = group["session_id"].astype(str).tolist()
        rng.shuffle(sessions)
        if len(sessions) == 1:
            train_sessions.extend(sessions)
            continue

        n_test = max(1, int(round(len(sessions) * test_fraction)))
        n_test = min(n_test, len(sessions) - 1)
        test_sessions.extend(sessions[:n_test])
        train_sessions.extend(sessions[n_test:])

    if not test_sessions:
        sessions = session_labels["session_id"].astype(str).tolist()
        rng.shuffle(sessions)
        split_index = max(1, int(len(sessions) * test_fraction))
        test_sessions = sessions[:split_index]
        train_sessions = sessions[split_index:]

    train_frame = frame[frame["session_id"].astype(str).isin(train_sessions)].copy()
    test_frame = frame[frame["session_id"].astype(str).isin(test_sessions)].copy()

    if train_frame.empty or test_frame.empty:
        session_ids = session_labels["session_id"].astype(str).tolist()
        rng.shuffle(session_ids)
        split_index = max(1, int(round(len(session_ids) * test_fraction)))
        split_index = min(split_index, len(session_ids) - 1)
        test_sessions = session_ids[:split_index]
        train_sessions = session_ids[split_index:]
        train_frame = frame[frame["session_id"].astype(str).isin(train_sessions)].copy()
        test_frame = frame[frame["session_id"].astype(str).isin(test_sessions)].copy()

    overlap = sorted(
        set(train_frame["session_id"].astype(str).unique()).intersection(
            set(test_frame["session_id"].astype(str).unique())
        )
    )
    if overlap:
        raise RuntimeError(f"Leakage detected: train/test session overlap found: {overlap}")

    return train_frame, test_frame


def split_dataset_by_group(
    frame: pd.DataFrame,
    test_fraction: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform a group-aware split so one split_group never appears in both sets."""

    if "split_group" not in frame.columns:
        raise ValueError("Group-aware splitting requires a 'split_group' column.")
    if frame["split_group"].nunique() < 2:
        raise ValueError(
            "Group-aware splitting requires at least two groups. "
            "To allow leakage intentionally, switch evaluation.split_strategy to 'window'."
        )

    group_labels = frame[["split_group", "label"]].drop_duplicates().copy()
    group_label_counts = group_labels["split_group"].astype(str).value_counts()
    inconsistent_groups = sorted(group_label_counts[group_label_counts > 1].index.astype(str).tolist())
    if inconsistent_groups:
        raise ValueError(
            "Group-aware splitting requires exactly one label per split_group. "
            f"Inconsistent groups: {inconsistent_groups}"
        )

    rng = np.random.default_rng(random_state)
    train_groups: List[str] = []
    test_groups: List[str] = []

    for label, group in group_labels.groupby("label"):
        groups = group["split_group"].astype(str).tolist()
        rng.shuffle(groups)
        if len(groups) == 1:
            train_groups.extend(groups)
            continue

        n_test = max(1, int(round(len(groups) * test_fraction)))
        n_test = min(n_test, len(groups) - 1)
        test_groups.extend(groups[:n_test])
        train_groups.extend(groups[n_test:])

    if not test_groups:
        all_groups = group_labels["split_group"].astype(str).tolist()
        rng.shuffle(all_groups)
        split_index = max(1, int(round(len(all_groups) * test_fraction)))
        split_index = min(split_index, len(all_groups) - 1)
        test_groups = all_groups[:split_index]
        train_groups = all_groups[split_index:]

    train_frame = frame[frame["split_group"].astype(str).isin(train_groups)].copy()
    test_frame = frame[frame["split_group"].astype(str).isin(test_groups)].copy()

    if train_frame.empty or test_frame.empty:
        all_groups = group_labels["split_group"].astype(str).tolist()
        rng.shuffle(all_groups)
        split_index = max(1, int(round(len(all_groups) * test_fraction)))
        split_index = min(split_index, len(all_groups) - 1)
        test_groups = all_groups[:split_index]
        train_groups = all_groups[split_index:]
        train_frame = frame[frame["split_group"].astype(str).isin(train_groups)].copy()
        test_frame = frame[frame["split_group"].astype(str).isin(test_groups)].copy()

    overlap = sorted(
        set(train_frame["split_group"].astype(str).unique()).intersection(
            set(test_frame["split_group"].astype(str).unique())
        )
    )
    if overlap:
        raise RuntimeError(f"Leakage detected: train/test group overlap found: {overlap}")

    return train_frame, test_frame


def split_dataset_by_window(
    frame: pd.DataFrame,
    test_fraction: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform a row-level split when the user explicitly accepts leakage risk."""

    if frame.empty:
        raise ValueError("Cannot split an empty dataset.")
    shuffled = frame.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    split_index = max(1, int(round(len(shuffled) * (1.0 - test_fraction))))
    split_index = min(split_index, len(shuffled) - 1)
    return shuffled.iloc[:split_index].copy(), shuffled.iloc[split_index:].copy()


def split_dataset(
    frame: pd.DataFrame,
    strategy: str,
    test_fraction: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset using an explicit leakage policy."""

    if strategy == "session":
        return split_dataset_by_session(frame, test_fraction=test_fraction, random_state=random_state)
    if strategy == "group":
        return split_dataset_by_group(frame, test_fraction=test_fraction, random_state=random_state)
    if strategy == "window":
        return split_dataset_by_window(frame, test_fraction=test_fraction, random_state=random_state)
    raise ValueError(f"Unsupported split strategy: {strategy}")


def summarize_split(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    strategy: str,
) -> Dict[str, object]:
    """Create a compact split summary for saved experiment reports."""

    train_sessions = sorted(train_frame.get("session_id", pd.Series(dtype=str)).astype(str).unique().tolist())
    test_sessions = sorted(test_frame.get("session_id", pd.Series(dtype=str)).astype(str).unique().tolist())
    train_groups = sorted(train_frame.get("split_group", pd.Series(dtype=str)).astype(str).unique().tolist())
    test_groups = sorted(test_frame.get("split_group", pd.Series(dtype=str)).astype(str).unique().tolist())
    overlap = sorted(set(train_sessions).intersection(set(test_sessions)))
    group_overlap = sorted(set(train_groups).intersection(set(test_groups)))
    return {
        "split_strategy": strategy,
        "n_train_windows": int(len(train_frame)),
        "n_test_windows": int(len(test_frame)),
        "n_train_sessions": int(len(train_sessions)),
        "n_test_sessions": int(len(test_sessions)),
        "n_train_groups": int(len(train_groups)),
        "n_test_groups": int(len(test_groups)),
        "train_sessions": train_sessions,
        "test_sessions": test_sessions,
        "train_groups": train_groups,
        "test_groups": test_groups,
        "session_overlap": overlap,
        "group_overlap": group_overlap,
        "train_label_distribution": train_frame["label"].astype(str).value_counts().to_dict()
        if "label" in train_frame
        else {},
        "test_label_distribution": test_frame["label"].astype(str).value_counts().to_dict()
        if "label" in test_frame
        else {},
    }
