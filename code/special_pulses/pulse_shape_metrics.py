"""Shared waveform metrics for pulse shape analysis.

Analysis part: special-pulse metrics (used by prototype plots and property analyses).
Dependencies: waveform_rule_metrics.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.signal import find_peaks

from special_pulses.waveform_rule_metrics import (
    compute_half_max_width,
    detect_double_pulse,
    get_representative_waveform,
)


def baseline_correct(trace: np.ndarray) -> np.ndarray:
    baseline_window = max(1, len(trace) // 5)
    return trace - np.median(trace[:baseline_window])


def normalize_trace(trace: np.ndarray) -> np.ndarray:
    peak = np.max(trace)
    if peak <= 0:
        return trace
    return trace / peak


def width_at_fraction(signal: np.ndarray, fraction: float = 0.1) -> dict:
    """Left/right widths from peak to crossings at fraction × peak amplitude."""
    peak_idx = int(np.argmax(signal))
    peak_amp = float(signal[peak_idx])
    if peak_amp <= 0:
        return {
            "left_width_samples": np.nan,
            "right_width_samples": np.nan,
            "left_width_sec": np.nan,
            "right_width_sec": np.nan,
            "peak_idx": peak_idx,
            "threshold": np.nan,
        }

    threshold = fraction * peak_amp
    left = peak_idx
    while left > 0 and signal[left] > threshold:
        left -= 1
    right = peak_idx
    while right < len(signal) - 1 and signal[right] > threshold:
        right += 1

    left_width = peak_idx - left
    right_width = right - peak_idx
    return {
        "left_width_samples": left_width,
        "right_width_samples": right_width,
        "left_idx": left,
        "right_idx": right,
        "peak_idx": peak_idx,
        "threshold": threshold,
    }


def symmetry_at_fraction(signal: np.ndarray, sample_rate: float, fraction: float = 0.1) -> dict:
    """Per-pulse symmetry measured at fraction of peak amplitude."""
    info = width_at_fraction(signal, fraction=fraction)
    left = info["left_width_samples"]
    right = info["right_width_samples"]
    if np.isnan(left) or np.isnan(right) or (left + right) == 0:
        ratio = np.nan
    else:
        ratio = min(left, right) / max(left, right)
    return {
        **info,
        "left_width_sec": left / sample_rate,
        "right_width_sec": right / sample_rate,
        "symmetry_ratio": ratio,
        "asymmetry": abs(left - right) / sample_rate,
    }


def double_pulse_metrics(signal: np.ndarray, sample_rate: float) -> dict:
    """Peak separation, trough depth, and half-width for one double pulse."""
    is_double, info = detect_double_pulse(signal[:, np.newaxis], sample_rate)
    metrics = {
        "is_double": bool(is_double),
        "peak_separation_sec": np.nan,
        "peak_separation_ms": np.nan,
        "trough_depth_ratio": np.nan,
        "half_width_sec": np.nan,
        "half_width_ms": np.nan,
    }
    if not is_double or "peaks" not in info:
        return metrics

    peaks = np.sort(info["peaks"])
    p1, p2 = peaks
    valley_idx = p1 + int(np.argmin(signal[p1 : p2 + 1]))
    peak_heights = [signal[p1], signal[p2]]
    higher_peak = max(peak_heights)
    trough = signal[valley_idx]

    width_sec, _ = compute_half_max_width(signal, sample_rate)
    metrics.update(
        {
            "peak_separation_sec": (p2 - p1) / sample_rate,
            "peak_separation_ms": (p2 - p1) / sample_rate * 1000,
            "trough_depth_ratio": trough / higher_peak if higher_peak > 0 else np.nan,
            "half_width_sec": width_sec,
            "half_width_ms": width_sec * 1000,
            "peaks": peaks,
            "valley_idx": valley_idx,
        }
    )
    return metrics


def paired_symmetry_test(left_widths: np.ndarray, right_widths: np.ndarray) -> dict:
    """Paired t-test: left vs right width at 10% amplitude (H0: symmetric)."""
    mask = ~(np.isnan(left_widths) | np.isnan(right_widths))
    left = left_widths[mask]
    right = right_widths[mask]
    if len(left) < 3:
        return {"n": len(left), "t_stat": np.nan, "p_value": np.nan}
    t_stat, p_value = stats.ttest_rel(left, right)
    return {"n": len(left), "t_stat": float(t_stat), "p_value": float(p_value)}


def shift_waveform(trace: np.ndarray, shift: int) -> np.ndarray:
    shifted = np.zeros_like(trace)
    if shift > 0:
        shifted[shift:] = trace[:-shift]
    elif shift < 0:
        shifted[:shift] = trace[-shift:]
    else:
        shifted = trace.copy()
    return shifted
