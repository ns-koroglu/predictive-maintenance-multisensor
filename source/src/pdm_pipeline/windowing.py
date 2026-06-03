"""Window creation for synchronized multi-sensor sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd

from .io import SessionData, TIMESTAMP_COL


@dataclass
class WindowedSample:
    """One synchronized analysis window across AE, vibration, and thermal signals."""

    session_id: str
    window_index: int
    start_time: float
    end_time: float
    ae: pd.DataFrame
    vibration: pd.DataFrame
    thermal: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)


def slice_window(frame: pd.DataFrame, start_time: float, end_time: float) -> pd.DataFrame:
    """Select samples that belong to one half-open time interval."""

    mask = (frame[TIMESTAMP_COL] >= start_time) & (frame[TIMESTAMP_COL] < end_time)
    return frame.loc[mask].reset_index(drop=True)


def create_session_windows(
    session: SessionData,
    duration_sec: float,
    overlap: float,
    minimum_samples: Dict[str, int],
) -> List[WindowedSample]:
    """Convert one synchronized session into fixed-length overlapping windows."""

    if duration_sec <= 0:
        raise ValueError("Window duration must be positive")
    if not 0 <= overlap < 1:
        raise ValueError("Window overlap must be in the range [0, 1)")

    step_sec = duration_sec * (1.0 - overlap)
    if step_sec <= 0:
        raise ValueError("Window step size must be positive")

    start_time = max(
        float(session.ae[TIMESTAMP_COL].min()),
        float(session.vibration[TIMESTAMP_COL].min()),
        float(session.thermal[TIMESTAMP_COL].min()),
    )
    end_time = min(
        float(session.ae[TIMESTAMP_COL].max()),
        float(session.vibration[TIMESTAMP_COL].max()),
        float(session.thermal[TIMESTAMP_COL].max()),
    )

    windows: List[WindowedSample] = []
    window_index = 0
    current_start = start_time
    while current_start + duration_sec <= end_time + 1e-12:
        current_end = current_start + duration_sec
        ae_window = slice_window(session.ae, current_start, current_end)
        vibration_window = slice_window(session.vibration, current_start, current_end)
        thermal_window = slice_window(session.thermal, current_start, current_end)

        if (
            len(ae_window) >= minimum_samples.get("ae", 1)
            and len(vibration_window) >= minimum_samples.get("vibration", 1)
            and len(thermal_window) >= minimum_samples.get("thermal", 1)
        ):
            windows.append(
                WindowedSample(
                    session_id=session.session_id,
                    window_index=window_index,
                    start_time=float(current_start),
                    end_time=float(current_end),
                    ae=ae_window,
                    vibration=vibration_window,
                    thermal=thermal_window,
                    metadata=dict(session.metadata),
                )
            )

        current_start += step_sec
        window_index += 1

    return windows
