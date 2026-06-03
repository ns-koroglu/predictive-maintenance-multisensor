"""Thermal summary feature extraction for trend-based monitoring."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from ..io import TIMESTAMP_COL, numeric_sensor_columns
from .common import linear_slope


def extract_thermal_features(thermal_frame: pd.DataFrame) -> Dict[str, float]:
    """Extract simple but interpretable thermal trend features."""

    timestamps = thermal_frame[TIMESTAMP_COL].to_numpy(dtype=float)
    features: Dict[str, float] = {}

    for column in numeric_sensor_columns(thermal_frame):
        values = thermal_frame[column].to_numpy(dtype=float)
        prefix = f"thermal_{column}"

        features[f"{prefix}_mean"] = float(np.mean(values))
        features[f"{prefix}_std"] = float(np.std(values))
        features[f"{prefix}_min"] = float(np.min(values))
        features[f"{prefix}_max"] = float(np.max(values))
        features[f"{prefix}_range"] = float(np.ptp(values))
        features[f"{prefix}_start"] = float(values[0])
        features[f"{prefix}_end"] = float(values[-1])
        features[f"{prefix}_trend"] = float(values[-1] - values[0]) if len(values) > 1 else 0.0
        features[f"{prefix}_slope"] = linear_slope(values, timestamps)

    if {"t_mean", "t_max"}.issubset(thermal_frame.columns):
        delta = (
            thermal_frame["t_max"].to_numpy(dtype=float)
            - thermal_frame["t_mean"].to_numpy(dtype=float)
        )
        features["thermal_temp_gap_mean"] = float(np.mean(delta))
        features["thermal_temp_gap_max"] = float(np.max(delta))
        features["thermal_temp_gap_trend"] = float(delta[-1] - delta[0]) if len(delta) > 1 else 0.0
        features["thermal_temp_gap_slope"] = linear_slope(delta, timestamps)

    if {"hotspot_area", "t_max"}.issubset(thermal_frame.columns):
        hotspot_energy = (
            thermal_frame["hotspot_area"].to_numpy(dtype=float)
            * thermal_frame["t_max"].to_numpy(dtype=float)
        )
        features["thermal_hotspot_energy_mean"] = float(np.mean(hotspot_energy))
        features["thermal_hotspot_energy_trend"] = (
            float(hotspot_energy[-1] - hotspot_energy[0]) if len(hotspot_energy) > 1 else 0.0
        )

    return features
