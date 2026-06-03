from __future__ import annotations

from pathlib import Path

from pdm_pipeline.io import load_session
from pdm_pipeline.preprocessing import preprocess_session, synchronize_session
from pdm_pipeline.windowing import create_session_windows


def test_timestamp_synchronization(sample_session_dir: Path) -> None:
    session = load_session(sample_session_dir)
    cleaned = preprocess_session(session)
    synchronized = synchronize_session(cleaned)

    assert synchronized.ae["timestamp"].min() == synchronized.vibration["timestamp"].min()
    assert synchronized.thermal["timestamp"].max() == synchronized.vibration["timestamp"].max()


def test_window_generation(sample_session_dir: Path) -> None:
    session = synchronize_session(preprocess_session(load_session(sample_session_dir)))
    windows = create_session_windows(
        session,
        duration_sec=1.0,
        overlap=0.5,
        minimum_samples={"ae": 2, "vibration": 2, "thermal": 2},
    )

    assert len(windows) >= 2
    assert windows[0].end_time > windows[0].start_time
    assert len(windows[0].ae) >= 2
