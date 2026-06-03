"""Feature-level fusion across AE, vibration, thermal, and numeric metadata."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..io import metadata_to_feature_context
from .ae import extract_ae_features
from .common import safe_ratio
from .thermal import extract_thermal_features
from .vibration import extract_vibration_features


def _mean_feature_value(feature_dict: Dict[str, float], prefix: str, suffix: str) -> float:
    """Average selected feature values when multiple signal channels are present."""

    values = [
        float(value)
        for key, value in feature_dict.items()
        if key.startswith(prefix) and key.endswith(suffix)
    ]
    if not values:
        return 0.0
    return float(np.mean(values))


def extract_fused_features(
    vibration_frame: pd.DataFrame,
    ae_frame: pd.DataFrame,
    thermal_frame: pd.DataFrame,
    metadata: Optional[Dict[str, object]] = None,
    vibration_sampling_rate_hz: Optional[float] = None,
    ae_sampling_rate_hz: Optional[float] = None,
) -> Dict[str, float]:
    """Create modality-specific features first, then add lightweight fusion ratios."""

    features: Dict[str, float] = {}
    vibration_features = extract_vibration_features(vibration_frame, vibration_sampling_rate_hz)
    ae_features = extract_ae_features(ae_frame, ae_sampling_rate_hz)
    thermal_features = extract_thermal_features(thermal_frame)

    features.update(vibration_features)
    features.update(ae_features)
    features.update(thermal_features)

    vibration_rms = _mean_feature_value(vibration_features, "vibration_", "_rms")
    ae_rms = _mean_feature_value(ae_features, "ae_", "_rms")
    thermal_mean = float(features.get("thermal_t_mean_mean", features.get("thermal_t_max_mean", 0.0)))

    if vibration_rms > 0 and ae_rms > 0:
        features["fusion_ae_to_vibration_rms_ratio"] = safe_ratio(ae_rms, vibration_rms)
        features["fusion_vibration_to_ae_rms_ratio"] = safe_ratio(vibration_rms, ae_rms)

    if thermal_mean > 0 and vibration_rms > 0:
        features["fusion_thermal_to_vibration_ratio"] = safe_ratio(thermal_mean, vibration_rms)

    if thermal_mean > 0 and ae_rms > 0:
        features["fusion_thermal_to_ae_ratio"] = safe_ratio(thermal_mean, ae_rms)

    if "thermal_temp_gap_mean" in features and vibration_rms > 0:
        features["fusion_temp_gap_to_vibration_ratio"] = safe_ratio(
            float(features["thermal_temp_gap_mean"]),
            vibration_rms,
        )

    if metadata:
        features.update(metadata_to_feature_context(metadata))

    return features
