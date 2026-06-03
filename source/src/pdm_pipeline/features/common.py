"""Shared signal-processing feature utilities."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis, skew


EPS = 1e-12


def safe_ratio(numerator: float, denominator: float) -> float:
    """Compute ratios without dividing by zero."""

    return float(numerator / (denominator + EPS))


def rms(signal: np.ndarray) -> float:
    """Root-mean-square amplitude."""

    signal = np.asarray(signal, dtype=float)
    if signal.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(signal))))


def spectral_entropy(power_spectral_density: np.ndarray) -> float:
    """Shannon entropy of the normalized power spectrum."""

    density = np.asarray(power_spectral_density, dtype=float)
    if density.size == 0:
        return 0.0
    density = np.maximum(density, EPS)
    density = density / np.sum(density)
    return float(-np.sum(density * np.log2(density)))


def bandpower(
    frequencies: np.ndarray,
    power_spectral_density: np.ndarray,
    low_cutoff: float,
    high_cutoff: float,
) -> float:
    """Integrate the spectrum over a frequency band."""

    mask = (frequencies >= low_cutoff) & (frequencies < high_cutoff)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(power_spectral_density[mask], frequencies[mask]))


def linear_slope(values: np.ndarray, timestamps: Optional[np.ndarray] = None) -> float:
    """Estimate a linear trend slope for a window."""

    signal = np.asarray(values, dtype=float)
    if signal.size < 2:
        return 0.0

    if timestamps is None:
        time_axis = np.arange(signal.size, dtype=float)
    else:
        time_axis = np.asarray(timestamps, dtype=float)
        if time_axis.size != signal.size:
            return 0.0
        time_axis = time_axis - time_axis[0]

    if np.allclose(time_axis, time_axis[0]):
        return 0.0
    return float(np.polyfit(time_axis, signal, deg=1)[0])


def estimate_sampling_rate_from_timestamps(
    timestamps: Iterable[float],
    fallback_sampling_rate: Optional[float] = None,
) -> float:
    """Estimate sampling rate directly from timestamps."""

    time_array = np.asarray(list(timestamps), dtype=float)
    if time_array.size < 2:
        return float(fallback_sampling_rate or 0.0)

    deltas = np.diff(time_array)
    deltas = deltas[deltas > 0]
    if deltas.size == 0:
        return float(fallback_sampling_rate or 0.0)
    return float(1.0 / np.median(deltas))


def _clean_statistic(value: float, default: float = 0.0) -> float:
    """Replace NaN or infinite statistics with a stable fallback."""

    if np.isnan(value) or np.isinf(value):
        return default
    return float(value)


def extract_time_domain_features(signal: np.ndarray, prefix: str) -> Dict[str, float]:
    """Hand-crafted time-domain features commonly used in condition monitoring."""

    x = np.asarray(signal, dtype=float)
    if x.size < 4:
        return {}

    absolute_x = np.abs(x)
    peak = float(np.max(absolute_x))
    mean_abs = float(np.mean(absolute_x))
    rms_value = rms(x)
    root_amplitude = float(np.square(np.mean(np.sqrt(absolute_x + EPS))))

    if np.allclose(x, x[0]):
        skewness = 0.0
        kurtosis_value = 0.0
    else:
        skewness = _clean_statistic(float(skew(x, bias=False)), default=0.0)
        kurtosis_value = _clean_statistic(float(kurtosis(x, fisher=False, bias=False)), default=0.0)

    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_rms": rms_value,
        f"{prefix}_min": float(np.min(x)),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_p2p": float(np.ptp(x)),
        f"{prefix}_skew": skewness,
        f"{prefix}_kurtosis": kurtosis_value,
        f"{prefix}_crest_factor": safe_ratio(peak, rms_value),
        f"{prefix}_shape_factor": safe_ratio(rms_value, mean_abs),
        f"{prefix}_impulse_factor": safe_ratio(peak, mean_abs),
        f"{prefix}_margin_factor": safe_ratio(peak, root_amplitude),
        f"{prefix}_zero_crossings": float(np.sum(np.diff(np.signbit(x)) != 0)),
    }


def extract_frequency_domain_features(
    signal: np.ndarray,
    sampling_rate_hz: float,
    prefix: str,
    band_edges: tuple[float, float, float] = (0.1, 0.3, 0.6),
) -> Dict[str, float]:
    """Welch-spectrum features that remain explainable in a thesis context."""

    x = np.asarray(signal, dtype=float)
    if x.size < 8 or sampling_rate_hz <= 0:
        return {}

    nperseg = min(len(x), 256)
    frequencies, power_spectral_density = welch(x, fs=sampling_rate_hz, nperseg=nperseg)
    if frequencies.size == 0 or power_spectral_density.size == 0:
        return {}

    total_power = float(np.trapezoid(power_spectral_density, frequencies))
    dominant_index = int(np.argmax(power_spectral_density))
    dominant_frequency = float(frequencies[dominant_index])

    spectrum_sum = float(np.sum(power_spectral_density))
    spectral_centroid = safe_ratio(float(np.sum(frequencies * power_spectral_density)), spectrum_sum)
    spectral_bandwidth = safe_ratio(
        float(np.sqrt(np.sum(((frequencies - spectral_centroid) ** 2) * power_spectral_density))),
        spectrum_sum,
    )

    nyquist = sampling_rate_hz / 2.0
    edge_1 = min(max(band_edges[0] * nyquist, 0.0), nyquist)
    edge_2 = min(max(band_edges[1] * nyquist, edge_1), nyquist)
    edge_3 = min(max(band_edges[2] * nyquist, edge_2), nyquist)
    bands = [
        (0.0, edge_1),
        (edge_1, edge_2),
        (edge_2, edge_3),
        (edge_3, nyquist + EPS),
    ]

    features = {
        f"{prefix}_spec_total_power": total_power,
        f"{prefix}_spec_dominant_freq": dominant_frequency,
        f"{prefix}_spec_centroid": spectral_centroid,
        f"{prefix}_spec_bandwidth": spectral_bandwidth,
        f"{prefix}_spec_entropy": spectral_entropy(power_spectral_density),
    }
    for band_index, (low_cutoff, high_cutoff) in enumerate(bands, start=1):
        features[f"{prefix}_bandpower_{band_index}"] = bandpower(
            frequencies,
            power_spectral_density,
            low_cutoff,
            high_cutoff,
        )
    return features


def extract_signal_features(
    signal: np.ndarray,
    timestamps: np.ndarray,
    prefix: str,
    fallback_sampling_rate: Optional[float] = None,
) -> Dict[str, float]:
    """Combine time-domain and frequency-domain features for one signal."""

    sampling_rate_hz = estimate_sampling_rate_from_timestamps(timestamps, fallback_sampling_rate)
    features = extract_time_domain_features(signal, prefix)
    features.update(extract_frequency_domain_features(signal, sampling_rate_hz, prefix))
    return features
