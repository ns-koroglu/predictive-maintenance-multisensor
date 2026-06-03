"""Reproducible synthetic raw-session generator for presentation demos."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .utils import ensure_directory


@dataclass(frozen=True)
class DemoSessionSpec:
    """Compact description of one synthetic demo session."""

    session_id: str
    label: str
    seed: int
    rpm: int
    load_level: float
    ambient_temp_c: float
    lubrication_state: str
    note: str


DEMO_SESSION_SPECS: List[DemoSessionSpec] = [
    DemoSessionSpec(
        session_id="session_001_healthy",
        label="healthy",
        seed=101,
        rpm=1200,
        load_level=0.20,
        ambient_temp_c=24.5,
        lubrication_state="normal",
        note="Seeded healthy synthetic session with stable low-variance signals.",
    ),
    DemoSessionSpec(
        session_id="session_002_healthy",
        label="healthy",
        seed=202,
        rpm=1325,
        load_level=0.26,
        ambient_temp_c=24.7,
        lubrication_state="normal",
        note="Second healthy seeded session used for session-safe split demonstration.",
    ),
    DemoSessionSpec(
        session_id="session_003_developing_fault",
        label="developing_fault",
        seed=303,
        rpm=1260,
        load_level=0.40,
        ambient_temp_c=25.0,
        lubrication_state="degraded",
        note="Seeded developing-fault session with higher AE activity and moderately larger vibration.",
    ),
    DemoSessionSpec(
        session_id="session_004_developing_fault",
        label="developing_fault",
        seed=404,
        rpm=1410,
        load_level=0.46,
        ambient_temp_c=25.2,
        lubrication_state="degraded",
        note="Seeded developing-fault session used as the default inference target.",
    ),
]


def _gaussian_pulses(time_axis: np.ndarray, centers: List[float], width: float, amplitudes: List[float]) -> np.ndarray:
    """Create deterministic pulse-like energy bursts for AE simulation."""

    signal = np.zeros_like(time_axis, dtype=float)
    for center, amplitude in zip(centers, amplitudes):
        signal += amplitude * np.exp(-0.5 * ((time_axis - center) / width) ** 2)
    return signal


def _generate_vibration(session: DemoSessionSpec, duration_sec: float = 12.0, sampling_rate_hz: float = 50.0) -> pd.DataFrame:
    """Generate tri-axial vibration with clear but simple healthy/fault separation."""

    rng = np.random.default_rng(session.seed)
    time_axis = np.arange(0.0, duration_sec + 1e-9, 1.0 / sampling_rate_hz)

    if session.label == "healthy":
        base_amp = 0.10
        noise_std = 0.010
        modulation = 1.0 + 0.03 * np.sin(2 * np.pi * 0.15 * time_axis)
        harmonic_boost = 0.04
    else:
        base_amp = 0.18
        noise_std = 0.016
        modulation = 1.0 + 0.08 * np.sin(2 * np.pi * 0.18 * time_axis)
        harmonic_boost = 0.09

    ax = modulation * (
        base_amp * np.sin(2 * np.pi * 8.0 * time_axis)
        + harmonic_boost * np.sin(2 * np.pi * 15.0 * time_axis + 0.35)
    )
    ay = modulation * (
        0.85 * base_amp * np.sin(2 * np.pi * 7.5 * time_axis + 0.6)
        + 0.90 * harmonic_boost * np.sin(2 * np.pi * 12.5 * time_axis)
    )
    az = modulation * (
        1.05 * base_amp * np.sin(2 * np.pi * 8.8 * time_axis + 0.2)
        + 1.10 * harmonic_boost * np.sin(2 * np.pi * 16.0 * time_axis + 0.8)
    )

    if session.label != "healthy":
        az += 0.03 * np.sin(2 * np.pi * 3.0 * time_axis)

    ax += rng.normal(0.0, noise_std, size=time_axis.size)
    ay += rng.normal(0.0, noise_std, size=time_axis.size)
    az += rng.normal(0.0, noise_std, size=time_axis.size)

    return pd.DataFrame(
        {
            "timestamp": time_axis,
            "ax": ax,
            "ay": ay,
            "az": az,
        }
    )


def _generate_ae(session: DemoSessionSpec, duration_sec: float = 12.0, sampling_rate_hz: float = 100.0) -> pd.DataFrame:
    """Generate AE signals with mild bursts for developing-fault sessions."""

    rng = np.random.default_rng(session.seed + 1000)
    time_axis = np.arange(0.0, duration_sec + 1e-9, 1.0 / sampling_rate_hz)

    if session.label == "healthy":
        base = (
            0.018 * np.sin(2 * np.pi * 12.0 * time_axis)
            + 0.010 * np.sin(2 * np.pi * 21.0 * time_axis + 0.4)
        )
        bursts = _gaussian_pulses(time_axis, centers=[4.0, 8.3], width=0.06, amplitudes=[0.020, 0.018])
        noise_std = 0.004
    else:
        base = (
            0.030 * np.sin(2 * np.pi * 12.0 * time_axis)
            + 0.020 * np.sin(2 * np.pi * 24.0 * time_axis + 0.35)
        )
        bursts = _gaussian_pulses(
            time_axis,
            centers=[2.8, 5.7, 8.8, 10.4],
            width=0.07,
            amplitudes=[0.060, 0.075, 0.065, 0.070],
        )
        noise_std = 0.007

    ae = base + bursts + rng.normal(0.0, noise_std, size=time_axis.size)
    return pd.DataFrame({"timestamp": time_axis, "ae": ae})


def _generate_thermal(session: DemoSessionSpec, duration_sec: float = 12.0, sampling_period_sec: float = 0.5) -> pd.DataFrame:
    """Generate thermal summary features with mild drift and measurement noise."""

    rng = np.random.default_rng(session.seed + 2000)
    time_axis = np.arange(0.0, duration_sec + 1e-9, sampling_period_sec)

    if session.label == "healthy":
        mean_start = 31.0
        mean_drift = 0.045
        gap_start = 2.10
        gap_drift = 0.010
        hotspot_start = 9.80
        hotspot_drift = 0.025
        noise_std = 0.025
    else:
        mean_start = 33.8
        mean_drift = 0.110
        gap_start = 2.45
        gap_drift = 0.018
        hotspot_start = 11.20
        hotspot_drift = 0.055
        noise_std = 0.035

    t_mean = mean_start + mean_drift * time_axis + 0.05 * np.sin(2 * np.pi * 0.08 * time_axis)
    t_mean += rng.normal(0.0, noise_std, size=time_axis.size)

    gap = gap_start + gap_drift * time_axis + 0.03 * np.sin(2 * np.pi * 0.05 * time_axis + 0.3)
    t_max = t_mean + gap

    hotspot_area = hotspot_start + hotspot_drift * time_axis + 0.08 * np.sin(2 * np.pi * 0.06 * time_axis)
    hotspot_area += rng.normal(0.0, noise_std * 0.6, size=time_axis.size)

    return pd.DataFrame(
        {
            "timestamp": time_axis,
            "t_mean": t_mean,
            "t_max": t_max,
            "hotspot_area": hotspot_area,
        }
    )


def generate_demo_raw_sessions(data_root: str | Path) -> List[Dict[str, object]]:
    """Generate the four thesis-demo raw sessions in a reproducible way."""

    root = ensure_directory(data_root)
    generated_sessions: List[Dict[str, object]] = []

    for session in DEMO_SESSION_SPECS:
        session_dir = ensure_directory(root / session.session_id)
        ae_frame = _generate_ae(session)
        vibration_frame = _generate_vibration(session)
        thermal_frame = _generate_thermal(session)

        ae_frame.to_csv(session_dir / "ae.csv", index=False, float_format="%.6f")
        vibration_frame.to_csv(session_dir / "vibration.csv", index=False, float_format="%.6f")
        thermal_frame.to_csv(session_dir / "thermal.csv", index=False, float_format="%.6f")

        metadata = {
            "session_id": session.session_id,
            "label": session.label,
            "rpm": session.rpm,
            "load_level": session.load_level,
            "ambient_temp_c": session.ambient_temp_c,
            "lubrication_state": session.lubrication_state,
            "notes": session.note,
        }
        with (session_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        generated_sessions.append(
            {
                "session_id": session.session_id,
                "label": session.label,
                "n_ae_samples": int(len(ae_frame)),
                "n_vibration_samples": int(len(vibration_frame)),
                "n_thermal_samples": int(len(thermal_frame)),
                "seed": session.seed,
            }
        )

    return generated_sessions
