from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@pytest.fixture
def sample_session_dir(tmp_path: Path) -> Path:
    session_dir = tmp_path / "session_001"
    session_dir.mkdir()

    timestamps = [round(0.1 * index, 3) for index in range(21)]
    ae = pd.DataFrame(
        {
            "timestamp": timestamps,
            "ae": [0.1 + 0.02 * ((index % 5) - 2) for index in range(21)],
        }
    )
    vibration = pd.DataFrame(
        {
            "timestamp": timestamps,
            "ax": [1.0 + 0.1 * ((index % 4) - 1) for index in range(21)],
            "ay": [0.8 + 0.08 * ((index % 3) - 1) for index in range(21)],
        }
    )
    thermal = pd.DataFrame(
        {
            "timestamp": timestamps,
            "t_mean": [30.0 + 0.05 * index for index in range(21)],
            "t_max": [32.0 + 0.06 * index for index in range(21)],
            "hotspot_area": [10.0 + 0.03 * index for index in range(21)],
        }
    )
    metadata = {
        "session_id": "session_001",
        "label": "healthy",
        "rpm": 1200,
        "load_level": 0.2,
        "ambient_temp_c": 24.5,
    }

    ae.to_csv(session_dir / "ae.csv", index=False)
    vibration.to_csv(session_dir / "vibration.csv", index=False)
    thermal.to_csv(session_dir / "thermal.csv", index=False)
    (session_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return session_dir
