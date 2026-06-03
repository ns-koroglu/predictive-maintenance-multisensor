"""Preprocessing and synchronization helpers for multi-rate sensor streams."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .io import DataFormatError, SessionData, TIMESTAMP_COL


def preprocess_sensor_frame(
    frame: pd.DataFrame,
    sensor_name: str,
    interpolation_method: str = "linear",
    fill_missing: bool = True,
) -> pd.DataFrame:
    """Clean a raw sensor frame while preserving a thesis-friendly workflow."""

    work = frame.copy()
    work[TIMESTAMP_COL] = pd.to_numeric(work[TIMESTAMP_COL], errors="coerce")

    signal_columns = [column for column in work.columns if column != TIMESTAMP_COL]
    if not signal_columns:
        raise DataFormatError(f"{sensor_name} data does not contain any signal columns")

    for column in signal_columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")

    work = work.dropna(subset=[TIMESTAMP_COL]).sort_values(TIMESTAMP_COL)
    work = work.groupby(TIMESTAMP_COL, as_index=False).mean(numeric_only=True)

    numeric_columns = [column for column in work.columns if column != TIMESTAMP_COL]
    work = work.dropna(how="all", subset=numeric_columns)
    if work.empty:
        raise DataFormatError(f"{sensor_name} data is empty after timestamp cleaning")

    if fill_missing and numeric_columns:
        work[numeric_columns] = work[numeric_columns].interpolate(
            method=interpolation_method,
            limit_direction="both",
        )
        work[numeric_columns] = work[numeric_columns].ffill().bfill()

    return work.reset_index(drop=True)


def preprocess_session(
    session: SessionData,
    interpolation_method: str = "linear",
    fill_missing: bool = True,
) -> SessionData:
    """Apply the same cleaning logic to all modalities in one session."""

    return SessionData(
        session_id=session.session_id,
        ae=preprocess_sensor_frame(
            session.ae,
            sensor_name="AE",
            interpolation_method=interpolation_method,
            fill_missing=fill_missing,
        ),
        vibration=preprocess_sensor_frame(
            session.vibration,
            sensor_name="vibration",
            interpolation_method=interpolation_method,
            fill_missing=fill_missing,
        ),
        thermal=preprocess_sensor_frame(
            session.thermal,
            sensor_name="thermal",
            interpolation_method=interpolation_method,
            fill_missing=fill_missing,
        ),
        metadata=dict(session.metadata),
    )


def estimate_sampling_rate(frame: pd.DataFrame) -> float:
    """Estimate the sampling rate from timestamp differences."""

    timestamps = frame[TIMESTAMP_COL].to_numpy(dtype=float)
    if timestamps.size < 2:
        return 0.0

    deltas = np.diff(timestamps)
    deltas = deltas[deltas > 0]
    if deltas.size == 0:
        return 0.0
    return float(1.0 / np.median(deltas))


def trim_frame_to_overlap(frame: pd.DataFrame, start_time: float, end_time: float) -> pd.DataFrame:
    """Trim a sensor frame to the common time interval shared by all modalities."""

    mask = (frame[TIMESTAMP_COL] >= start_time) & (frame[TIMESTAMP_COL] <= end_time)
    return frame.loc[mask].reset_index(drop=True)


def synchronize_session(session: SessionData, method: str = "trim_to_overlap") -> SessionData:
    """Synchronize the three sensor streams on their common valid time interval."""

    if method != "trim_to_overlap":
        raise ValueError(f"Unsupported synchronization method: {method}")

    frames: Dict[str, pd.DataFrame] = session.sensor_frames()
    start_time = max(float(frame[TIMESTAMP_COL].min()) for frame in frames.values())
    end_time = min(float(frame[TIMESTAMP_COL].max()) for frame in frames.values())
    if end_time <= start_time:
        raise DataFormatError(
            f"Session {session.session_id} has no common overlap across AE, vibration, and thermal data"
        )

    synchronized = SessionData(
        session_id=session.session_id,
        ae=trim_frame_to_overlap(session.ae, start_time, end_time),
        vibration=trim_frame_to_overlap(session.vibration, start_time, end_time),
        thermal=trim_frame_to_overlap(session.thermal, start_time, end_time),
        metadata=dict(session.metadata),
    )
    synchronized.metadata["sync_start_time"] = start_time
    synchronized.metadata["sync_end_time"] = end_time
    synchronized.metadata["ae_sampling_rate_hz"] = estimate_sampling_rate(synchronized.ae)
    synchronized.metadata["vibration_sampling_rate_hz"] = estimate_sampling_rate(synchronized.vibration)
    synchronized.metadata["thermal_sampling_rate_hz"] = estimate_sampling_rate(synchronized.thermal)
    return synchronized
