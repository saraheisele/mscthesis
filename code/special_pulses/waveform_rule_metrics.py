"""Rule-based / metric helpers for special pulse shapes.

The ML classifier is the production shape decision. These helpers remain for
metrics, tuning, and legacy rule-based paths.

Callers must pass ``WAVEFORM_FS`` (see ``pulse_config``) as ``sample_rate`` for
predetected ``raw_pulses`` snippets. Those arrays share a 48 kHz-equivalent grid
even when the recording metadata says 24 kHz.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# Raised from 0.7 to suppress very small noisy pulses in special-shape detection.
MIN_AMPLITUDE_THRESHOLD = 5.0


def get_representative_waveform(pulse_waveform):
    """
    Select the strongest channel of the pulse waveform and
    enforce positive dominant polarity.

    Parameters:
    -----------
    pulse_waveform : np.ndarray
        Shape: (num_samples, num_channels)

    Returns:
    --------
    trace : np.ndarray
        1D waveform from strongest channel
    best_channel : int
        Index of selected channel
    """

    # Find strongest channel by absolute peak amplitude
    channel_strengths = np.max(np.abs(pulse_waveform), axis=0)
    best_channel = np.argmax(channel_strengths)

    # Extract waveform of strongest channel
    trace = pulse_waveform[:, best_channel].copy()

    # Flip polarity if dominant peak is negative
    if abs(np.min(trace)) > np.max(trace):
        trace *= -1

    return trace, best_channel


# width measurement function
def compute_half_max_width(signal, sample_rate):
    """
    Compute pulse width at half maximum amplitude.

    Parameters
    ----------
    signal : np.ndarray
        1D waveform
    sample_rate : float
        Sampling rate in Hz

    Returns
    -------
    width_sec : float
        Width in seconds
    info : dict
        Debug information
    """

    peak_idx = np.argmax(signal)
    peak_amp = signal[peak_idx]

    half_height = 0.5 * peak_amp

    # Find left crossing
    left = peak_idx
    while left > 0 and signal[left] > half_height:
        left -= 1

    # Find right crossing
    right = peak_idx
    while right < len(signal) - 1 and signal[right] > half_height:
        right += 1

    width_samples = right - left
    width_sec = width_samples / sample_rate

    return width_sec, {
        "peak_idx": peak_idx,
        "half_height": half_height,
        "left_idx": left,
        "right_idx": right,
        "width_samples": width_samples,
    }


# new version for unified double pulse detection (for detection and visualization)
def detect_double_pulse(
    pulse_waveform,
    sample_rate,
    amplitude_threshold=MIN_AMPLITUDE_THRESHOLD,
    min_peak_distance=0.0005,
    max_peak_distance=0.002,
    min_valley_ratio=0.4,
    max_valley_ratio=0.95,
    prominence_ratio=0.1,
    max_amplitude_diff=0.6,
):
    """
    Unified double pulse detector.

    Returns:
    --------
    is_double : bool
    info : dict (for visualization/debugging)
    """
    ## this approach just averages all 16 channels/waveforms of each pulse, prone to artifacts
    # trace = np.mean(pulse_waveform, axis=1)

    # # --- 1. Determine dominant polarity ---
    # if np.max(trace) >= abs(np.min(trace)):
    #     signal = trace
    #     polarity = 1
    # else:
    #     signal = -trace
    #     polarity = -1

    # use helper function to run double pulse detection on strongest waveform of each pulse only
    signal, best_channel = get_representative_waveform(pulse_waveform)
    polarity = 1

    max_amp = np.max(signal)

    if max_amp < amplitude_threshold:
        return False, {"reason": "below_threshold"}

    # --- 2. Find peaks on signed signal ---
    prominence = max(prominence_ratio * max_amp, 1e-12)
    peaks, _ = find_peaks(signal, prominence=prominence)

    # --- 3. Keep only strong peaks ---
    peaks = np.array([p for p in peaks if signal[p] >= amplitude_threshold])

    if len(peaks) != 2:
        return False, {"reason": f"{len(peaks)}_peaks"}

    peaks = np.sort(peaks)
    p1, p2 = peaks

    a1, a2 = signal[p1], signal[p2]

    # --- 4. One peak must be GLOBAL maximum ---
    if not (np.isclose(a1, max_amp) or np.isclose(a2, max_amp)):
        return False, {"reason": "no_global_max_peak"}

    # --- 5. Distance constraint ---
    dt = (p2 - p1) / sample_rate
    if dt > max_peak_distance:
        return False, {"reason": "too_far", "dt": dt}
    if dt < min_peak_distance:
        return False, {"reason": "too_close", "dt": dt}

    # --- 6. Valley constraint ---
    # baseline correction: take first 5th of pulse snippet and use median as baseline
    baseline = np.median(signal[: len(signal) // 5])
    signal_corrected = signal - baseline
    a1_corrected = signal_corrected[p1]
    a2_corrected = signal_corrected[p2]

    # determine which is the highest peak
    peak_ref = max(a1_corrected, a2_corrected)

    valley = np.min(signal_corrected[p1 : p2 + 1])
    if valley < min_valley_ratio * peak_ref:
        return False, {"reason": "valley_too_deep"}
    if valley > max_valley_ratio * peak_ref:
        return False, {"reason": "valley_too_shallow"}

    # --- Peak amplitude similarity constraint ---
    a_high = max(a1_corrected, a2_corrected)
    a_low = min(a1_corrected, a2_corrected)

    if a_low < max_amplitude_diff * a_high:
        return False, {"reason": "peak_amplitude_mismatch"}

    return True, {
        "peaks": peaks,
        "amplitudes": [a1, a2],
        "valley": valley,
        "polarity": polarity,
        "dt": dt,
    }


# shape analysis function for wide pulses
def check_pulse_shape_gaussian_exponential(signal, peak_idx, sample_rate):
    """
    Check if pulse follows a Gaussian-then-exponential shape:
    - Gaussian (symmetric) rise/peak
    - Exponential decay after peak

    Parameters:
    -----------
    signal : np.ndarray
        1D corrected waveform
    peak_idx : int
        Index of the peak
    sample_rate : float
        Sample rate in Hz

    Returns:
    --------
    is_valid_shape : bool
        True if pulse matches expected shape
    info : dict
        Debug information with fit quality metrics
    """

    # Fit Gaussian function to left side (rise) of peak
    def gaussian(x, amp, mu, sigma):
        return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    # Fit exponential function to right side (decay) of peak
    def exponential(x, amp, tau):
        return amp * np.exp(-x / tau)

    info = {}

    # --- LEFT SIDE (RISE): Fit Gaussian ---
    # Take data from beginning up to peak
    left_idx = max(0, peak_idx - int(0.002 * sample_rate))  # ~2ms before peak
    left_x = np.arange(peak_idx - left_idx)
    left_y = signal[left_idx : peak_idx + 1]

    if len(left_x) > 3:
        try:
            # Initial guess
            amp_left = np.max(left_y)
            mu_left = len(left_x) // 2
            sigma_left = len(left_x) / 4

            popt_left, _ = curve_fit(
                gaussian, left_x, left_y, p0=[amp_left, mu_left, sigma_left], maxfev=500
            )

            # Calculate R² for left fit
            y_pred_left = gaussian(left_x, *popt_left)
            ss_res_left = np.sum((left_y - y_pred_left) ** 2)
            ss_tot_left = np.sum((left_y - np.mean(left_y)) ** 2)
            r2_left = 1 - (ss_res_left / ss_tot_left) if ss_tot_left > 0 else 0
            info["r2_gaussian_rise"] = r2_left
        except:  # noqa: E722
            r2_left = -1
            info["r2_gaussian_rise"] = -1
    else:
        r2_left = -1
        info["r2_gaussian_rise"] = -1

    # --- RIGHT SIDE (DECAY): Fit Exponential ---
    # Take data from peak onwards
    right_idx = min(len(signal), peak_idx + int(0.002 * sample_rate))  # ~2ms after peak
    right_x = np.arange(right_idx - peak_idx)
    right_y = signal[peak_idx:right_idx]

    if len(right_x) > 3 and np.max(right_y) > 0:
        try:
            # Initial guess
            amp_right = right_y[0]
            tau_right = len(right_x) / 3  # characteristic decay time

            popt_right, _ = curve_fit(
                exponential, right_x, right_y, p0=[amp_right, tau_right], maxfev=500
            )

            # Calculate R² for right fit
            y_pred_right = exponential(right_x, *popt_right)
            ss_res_right = np.sum((right_y - y_pred_right) ** 2)
            ss_tot_right = np.sum((right_y - np.mean(right_y)) ** 2)
            r2_right = 1 - (ss_res_right / ss_tot_right) if ss_tot_right > 0 else 0
            info["r2_exponential_decay"] = r2_right
        except:  # noqa: E722
            r2_right = -1
            info["r2_exponential_decay"] = -1
    else:
        r2_right = -1
        info["r2_exponential_decay"] = -1

    # --- Asymmetry check: decay should be faster than rise ---
    # Calculate slope metrics
    rise_region = signal[max(0, peak_idx - int(0.001 * sample_rate)) : peak_idx]
    decay_region = signal[
        peak_idx : min(len(signal), peak_idx + int(0.001 * sample_rate))
    ]

    if len(rise_region) > 1:
        rise_slope = (signal[peak_idx] - rise_region[0]) / len(rise_region)
    else:
        rise_slope = 0

    if len(decay_region) > 1:
        decay_slope = (decay_region[0] - decay_region[-1]) / len(decay_region)
    else:
        decay_slope = 0

    # Decay should be steeper than rise for exponential
    if rise_slope > 1e-12:
        slope_ratio = decay_slope / rise_slope  # should be > 0.5
        info["decay_to_rise_slope_ratio"] = slope_ratio
    else:
        slope_ratio = 0
        info["decay_to_rise_slope_ratio"] = 0

    # Validation: Both fits should be reasonably good
    # We require at least one side to have decent fit (R² > 0.5)
    # or at least show the expected asymmetry
    gaussian_ok = r2_left > 0.4
    exponential_ok = r2_right > 0.4
    asymmetry_ok = slope_ratio > 0.3  # decay steeper than rise

    is_valid_shape = (gaussian_ok or exponential_ok) and (
        gaussian_ok and exponential_ok or asymmetry_ok
    )

    info["is_valid_shape"] = is_valid_shape

    return is_valid_shape, info


# detection of wide pulses
def detect_wide_pulse(
    pulse_waveform,
    sample_rate,
    amplitude_threshold=MIN_AMPLITUDE_THRESHOLD,
    width_threshold_ms=2.3,
    max_width_ms=4,
    isolation_window_ms=3.0,
    prominence_ratio=0.1,
    check_shape=True,
):
    """
    Detect wide pulses using width at half maximum and shape analysis.

    Parameters:
    -----------
    pulse_waveform : np.ndarray
        2D array of pulse data (num_samples, num_channels)
    sample_rate : float
        Sample rate in Hz
    amplitude_threshold : float
        Minimum amplitude for a pulse to be considered
    width_threshold_ms : float
        Minimum width at half maximum in milliseconds
    max_width_ms : float
        Maximum width at half maximum in milliseconds (excludes overly wide pulses).
        Use None for no upper limit.
    isolation_window_ms : float
        Time window (in milliseconds) before and after the peak to check for isolated peak
    prominence_ratio : float
        Minimum prominence ratio relative to peak amplitude (e.g., 0.1 = 10%).
        The peak must stand out at least this much from the surrounding signal.
    check_shape : bool
        If True, verify that pulse follows Gaussian-then-exponential shape pattern.
        Helps reject flat signals that happen to be wide.

    Returns:
    --------
    bool
        True if pulse is a wide pulse, False otherwise
    dict
        Debug information
    """

    signal, best_channel = get_representative_waveform(pulse_waveform)

    # baseline correction: take first 5th of pulse snippet and use median as baseline
    baseline = np.median(signal[: len(signal) // 5])
    signal_corrected = signal - baseline

    max_amp = np.max(signal_corrected)
    peak_idx = np.argmax(signal_corrected)

    if max_amp < amplitude_threshold:
        return False, {"reason": "below_threshold"}

    # --- NEW: Prominence check ---
    # Ensure the peak is actually prominent, not just a flat signal
    # Check prominence by looking at the surrounding signal
    window_samples = int(isolation_window_ms * sample_rate / 1000)
    start_idx = max(0, peak_idx - window_samples)
    end_idx = min(len(signal_corrected), peak_idx + window_samples + 1)

    # Use scipy's find_peaks with prominence
    peaks_in_region, peak_props = find_peaks(
        signal_corrected[start_idx:end_idx], prominence=prominence_ratio * max_amp
    )

    # The main peak should have sufficient prominence
    if len(peaks_in_region) == 0:
        return False, {"reason": "insufficient_prominence"}

    # Adjust peak index if needed (find_peaks returns indices relative to the region)
    main_peak_in_region = np.argmin(np.abs(peaks_in_region - (peak_idx - start_idx)))
    if main_peak_in_region >= len(peak_props["prominences"]):
        return False, {"reason": "peak_not_found_in_region"}

    peak_prominence = peak_props["prominences"][main_peak_in_region]
    if peak_prominence < prominence_ratio * max_amp:
        return False, {
            "reason": "low_prominence",
            "prominence": peak_prominence,
            "required": prominence_ratio * max_amp,
        }

    # Convert isolation window from ms to samples
    window_samples = int(isolation_window_ms * sample_rate / 1000)

    # Define the isolation window around the peak
    start_idx = max(0, peak_idx - window_samples)
    end_idx = min(len(signal_corrected), peak_idx + window_samples + 1)

    # Find all local maxima in the isolation window
    window_signal = signal_corrected[start_idx:end_idx]
    peaks_in_window, _ = find_peaks(window_signal)

    # Count peaks that are significantly high (above 50% of max amplitude)
    high_peaks = [p for p in peaks_in_window if window_signal[p] >= 0.5 * max_amp]

    # Should only have one dominant peak (the main peak itself)
    if len(high_peaks) > 1:
        return False, {
            "reason": "multiple_local_maxima_in_window",
            "num_peaks": len(high_peaks),
        }

    width_sec, width_info = compute_half_max_width(signal_corrected, sample_rate)

    width_ms = width_sec * 1000

    # Check minimum width constraint
    if width_ms < width_threshold_ms:
        return False, {
            "reason": "too_narrow",
            "width_ms": width_ms,
            **width_info,
        }

    # Check maximum width constraint
    if max_width_ms is not None and width_ms > max_width_ms:
        return False, {
            "reason": "too_wide",
            "width_ms": width_ms,
            **width_info,
        }

    # --- NEW: Shape analysis ---
    if check_shape:
        is_valid_shape, shape_info = check_pulse_shape_gaussian_exponential(
            signal_corrected, peak_idx, sample_rate
        )
        if not is_valid_shape:
            return False, {
                "reason": "invalid_pulse_shape",
                **shape_info,
            }
    else:
        shape_info = {}

    return True, {
        "width_ms": width_ms,
        "prominence": peak_prominence,
        **width_info,
        **shape_info,
    }


# unified detection wrapper
def detect_pulse(pulse_waveform, sample_rate, mode="double"):
    """
    Unified detector wrapper.

    Parameters
    ----------
    mode : str
        ``"double"`` or ``"wide"``. Callers that follow DETECTION_MODE should
        pass that value explicitly (the facade does).
    """

    if mode == "double":
        return detect_double_pulse(pulse_waveform, sample_rate)

    elif mode == "wide":
        return detect_wide_pulse(pulse_waveform, sample_rate)

    else:
        raise ValueError(f"Unknown detection mode: {mode}")

