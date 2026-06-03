from __future__ import annotations

from pathlib import Path

import pandas as pd

from pdm_pipeline.evaluation import split_dataset_by_group, split_dataset_by_session
from pdm_pipeline.features.fusion import extract_fused_features
from pdm_pipeline.io import load_session
from pdm_pipeline.preprocessing import preprocess_session, synchronize_session
from pdm_pipeline.windowing import create_session_windows


def test_feature_extraction_output_shape(sample_session_dir: Path) -> None:
    session = synchronize_session(preprocess_session(load_session(sample_session_dir)))
    window = create_session_windows(
        session,
        duration_sec=1.0,
        overlap=0.0,
        minimum_samples={"ae": 2, "vibration": 2, "thermal": 2},
    )[0]

    features = extract_fused_features(
        vibration_frame=window.vibration,
        ae_frame=window.ae,
        thermal_frame=window.thermal,
        metadata=window.metadata,
        vibration_sampling_rate_hz=2.0,
        ae_sampling_rate_hz=2.0,
    )

    assert len(features) > 10
    assert "thermal_t_mean_mean" in features
    assert any(key.startswith("ae_") for key in features)
    assert any(key.startswith("vibration_") for key in features)


def test_session_split_is_leakage_safe() -> None:
    frame = pd.DataFrame(
        {
            "session_id": ["s1", "s1", "s2", "s2", "s3", "s3", "s4", "s4"],
            "label": ["healthy", "healthy", "healthy", "healthy", "faulty", "faulty", "faulty", "faulty"],
            "feature_1": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )

    train_frame, test_frame = split_dataset_by_session(frame, test_fraction=0.25, random_state=42)

    assert set(train_frame["session_id"]).isdisjoint(set(test_frame["session_id"]))


def test_group_split_is_leakage_safe() -> None:
    frame = pd.DataFrame(
        {
            "session_id": ["s1", "s2", "s3", "s4", "s5", "s6"],
            "split_group": ["g1", "g1", "g2", "g2", "g3", "g4"],
            "label": ["healthy", "healthy", "healthy", "healthy", "faulty", "faulty"],
            "feature_1": [1, 2, 3, 4, 5, 6],
        }
    )

    train_frame, test_frame = split_dataset_by_group(frame, test_fraction=0.25, random_state=42)

    assert set(train_frame["split_group"]).isdisjoint(set(test_frame["split_group"]))
