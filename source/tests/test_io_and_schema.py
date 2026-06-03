from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pdm_pipeline.io import DataFormatError, load_session, read_sensor_csv


def test_session_loading(sample_session_dir: Path) -> None:
    session = load_session(sample_session_dir)

    assert session.session_id == "session_001"
    assert list(session.ae.columns) == ["timestamp", "ae"]
    assert "ax" in session.vibration.columns
    assert "t_mean" in session.thermal.columns


def test_invalid_thermal_schema_raises(tmp_path: Path) -> None:
    thermal_path = tmp_path / "thermal.csv"
    pd.DataFrame({"timestamp": [0.0, 0.5], "t_mean": [30.0, 30.1]}).to_csv(thermal_path, index=False)

    with pytest.raises(DataFormatError):
        read_sensor_csv(thermal_path, "thermal")
