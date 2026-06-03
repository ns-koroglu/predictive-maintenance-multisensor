"""Feature extraction for acoustic emission windows."""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from ..io import TIMESTAMP_COL, numeric_sensor_columns
from .common import extract_signal_features


def extract_ae_features(
    ae_frame: pd.DataFrame,
    fallback_sampling_rate: Optional[float] = None,
) -> Dict[str, float]:
    """Extract explainable AE features from one window."""

    timestamps = ae_frame[TIMESTAMP_COL].to_numpy(dtype=float)
    features: Dict[str, float] = {}
    for column in numeric_sensor_columns(ae_frame):
        signal = ae_frame[column].to_numpy(dtype=float)
        features.update(
            extract_signal_features(
                signal=signal,
                timestamps=timestamps,
                prefix=f"ae_{column}",
                fallback_sampling_rate=fallback_sampling_rate,
            )
        )
    return features
