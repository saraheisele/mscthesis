"""Detect and label special pulse shapes (double, wide, fat) in predetected .h5 files.

Analysis part: special-pulse detection and ML classifier (Part 2 of Berlin activity analysis).
Dependencies: data_paths, h5_io; writes marker arrays back into input .h5 files.

Rule-based detectors annotate each pulse; optional supervised workflow trains a
Random Forest classifier on manually labeled examples.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from path_setup import setup_script_paths

setup_script_paths(__file__)

import csv
import json
import pickle
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import numpy as np
import nixio
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_paths import H5_DIR, SPECIAL_PULSE_CLASSIFIER_DIR
from h5_io import (
    get_path_list,
    get_pulse_block,
    load_marker_array,
    open_h5,
    open_h5_readwrite_or_readonly,
    save_marker_sidecar,
)

# Initialize console for logging
con = Console()

#################################
############# MODE ##############
#################################

DETECTION_MODE = "fat"
# options:
# "double"
# "wide"
# "fat"

MODE_CONFIG = {
    "double": {
        "array_name": "is_double_peak",
        "display_name": "double peak",
    },
    "wide": {
        "array_name": "is_wide_pulse",
        "display_name": "wide pulse",
    },
    "fat": {
        "array_name": "is_fat_pulse",
        "display_name": "fat pulse",
    },
}

ARRAY_NAME = MODE_CONFIG[DETECTION_MODE]["array_name"]
DISPLAY_NAME = MODE_CONFIG[DETECTION_MODE]["display_name"]

# Raised from 0.7 to suppress very small noisy pulses in special-shape detection.
MIN_AMPLITUDE_THRESHOLD = 5.0

SPECIAL_PULSE_CLASSES = {
    0: "normal",
    1: "double",
    2: "wide",
    3: "fat",
}

LABELING_PULSE_CLASSES = {
    0: "normal",
    1: "double",
    2: "wide",
    3: "fat",
}

SPECIAL_CLASS_ARRAYS = {
    1: "is_double_peak",
    2: "is_wide_pulse",
    3: "is_fat_pulse",
}

LABELING_CLASS_ARRAYS = {
    1: ("is_double_peak",),
    2: ("is_wide_pulse",),
    3: ("is_fat_pulse",),
}

MULTICLASS_ARRAY_NAME = "special_pulse_class"

# Max PCA dimensions for classifier pipelines (RobustPCA step).
PCA_MAX_CLASSIFIER_COMPONENTS = 20
# Max PCA dimensions computed for exploratory scatter plots.
PCA_MAX_PLOT_COMPONENTS = 10


class RobustPCA(BaseEstimator, TransformerMixin):
    """
    PCA fit on inlier samples identified by MinCovDet (robust covariance).

    Outliers are excluded from the fit but still projected at transform time.
    """

    def __init__(self, n_components=2, random_state=42, support_fraction=None):
        self.n_components = n_components
        self.random_state = random_state
        self.support_fraction = support_fraction

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        n_components = min(self.n_components, n_samples - 1, n_features)
        n_components = max(1, n_components)
        self.n_components_ = n_components

        fit_mask = self._robust_inlier_mask(X)
        self.n_inliers_ = int(np.sum(fit_mask))
        self.n_outliers_ = int(np.sum(~fit_mask))

        self.pca_ = PCA(n_components=n_components, random_state=self.random_state)
        self.pca_.fit(X[fit_mask])
        return self

    def _robust_inlier_mask(self, X):
        n_samples, n_features = X.shape
        if n_samples < 3:
            return np.ones(n_samples, dtype=bool)

        # MinCovDet is expensive in high dimensions; detect outliers in a
        # compact PCA subspace, then fit the final PCA on those inliers.
        n_pre = min(50, n_samples - 1, n_features)
        if n_features > n_pre:
            pre_pca = PCA(n_components=n_pre, random_state=self.random_state)
            X_for_mcd = pre_pca.fit_transform(X)
        else:
            X_for_mcd = X

        support_fraction = self.support_fraction
        if support_fraction is None:
            support_fraction = min(0.75, (n_samples - 1) / n_samples)

        mcd = MinCovDet(
            support_fraction=support_fraction,
            random_state=self.random_state,
        )
        mcd.fit(X_for_mcd)
        return mcd.support_

    def transform(self, X):
        return self.pca_.transform(np.asarray(X, dtype=float))

    @property
    def explained_variance_ratio_(self):
        return self.pca_.explained_variance_ratio_

    @property
    def components_(self):
        return self.pca_.components_


#################################
############# ANALYSIS ##########
#################################


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


def detect_fat_pulse(
    pulse_waveform,
    sample_rate,
    amplitude_threshold=MIN_AMPLITUDE_THRESHOLD,
    width_threshold_ms=4,
    max_width_ms=None,
    isolation_window_ms=3.0,
    prominence_ratio=0.1,
):
    """
    Detect fat pulses using the wide-pulse detector without shape validation.
    """
    return detect_wide_pulse(
        pulse_waveform,
        sample_rate,
        amplitude_threshold=amplitude_threshold,
        width_threshold_ms=width_threshold_ms,
        max_width_ms=max_width_ms,
        isolation_window_ms=isolation_window_ms,
        prominence_ratio=prominence_ratio,
        check_shape=False,
    )


# unified detection wrapper
def detect_pulse(pulse_waveform, sample_rate):
    """
    Unified detector wrapper.
    """

    if DETECTION_MODE == "double":
        return detect_double_pulse(pulse_waveform, sample_rate)

    elif DETECTION_MODE == "wide":
        return detect_wide_pulse(pulse_waveform, sample_rate)

    elif DETECTION_MODE == "fat":
        return detect_fat_pulse(pulse_waveform, sample_rate)

    else:
        raise ValueError(f"Unknown DETECTION_MODE: {DETECTION_MODE}")


def detect_special_pulses_in_file(file_path):
    """
    Detect the selected special pulse type in all pulses of a single h5 file
    and add the results as a data array.

    Parameters:
    -----------
    file_path : Path or str
        Path to the h5 file to process

    Returns:
    --------
    dict
        Statistics about the file: number of pulses and detected pulses
    """
    con.log(f"Processing: {Path(file_path).name}")

    # Open h5 file with read/write mode
    file, write_mode = open_h5_readwrite_or_readonly(file_path)
    if file is None:
        return {
            "file": Path(file_path).name,
            "status": "skipped",
            "reason": "locked_or_unreadable",
        }

    try:
        # Access pulses block
        block = get_pulse_block(file)
        data_array_names = [da.name for da in block.data_arrays]

        # Check if required data arrays exist
        if "raw_pulses" not in data_array_names:
            con.log("  ⚠ File does not contain 'raw_pulses' array. Skipping.")
            return {
                "file": Path(file_path).name,
                "status": "skipped",
                "reason": "no raw_pulses",
            }

        if "predicted_labels" not in data_array_names:
            con.log("  ⚠ File does not contain 'predicted_labels' array. Skipping.")
            return {
                "file": Path(file_path).name,
                "status": "skipped",
                "reason": "no predicted_labels",
            }

        # Load pulse data
        raw_pulses = block.data_arrays["raw_pulses"]
        predicted_labels = block.data_arrays["predicted_labels"]

        # Get sample rate from metadata
        fs = file.sections["pulses_metadata"]["metadata"]["samplerate"]
        ### TODO: CURRENTLY AT STEP 8 OF CHATS IMPLEMENTATION
        num_pulses = len(raw_pulses)
        con.log(f"  Found {num_pulses} pulses.")

        # Previous detector arrays are used to keep rule-based categories exclusive
        # when running the modes in order: double, wide, fat.
        predicted = predicted_labels[:]
        candidate_indices = np.where(predicted == 1)[0]

        if DETECTION_MODE in {"wide", "fat"}:
            double_marker = load_marker_array(file_path, "is_double_peak", block)
            if double_marker is not None:
                is_double_peak = expand_marker_to_all_pulses(
                    double_marker,
                    candidate_indices,
                    num_pulses,
                    "is_double_peak",
                )
            else:
                is_double_peak = np.zeros(num_pulses, dtype=np.int64)
        else:
            is_double_peak = np.zeros(num_pulses, dtype=np.int64)

        if DETECTION_MODE == "fat":
            wide_marker = load_marker_array(file_path, "is_wide_pulse", block)
            if wide_marker is not None:
                is_wide_pulse = expand_marker_to_all_pulses(
                    wide_marker,
                    candidate_indices,
                    num_pulses,
                    "is_wide_pulse",
                )
            else:
                is_wide_pulse = np.zeros(num_pulses, dtype=np.int64)
        else:
            is_wide_pulse = np.zeros(num_pulses, dtype=np.int64)

        # Analyze each pulse for the selected pulse type
        is_detection_array = np.zeros(num_pulses, dtype=np.int64)
        positive_count = 0
        negative_count = 0

        # detection loop
        for i, pulse in enumerate(raw_pulses):
            # TODO: ask patrick if all pulses in the h5 files have a 1 for predicted_labels!!
            # Only analyze predicted positive pulses
            if predicted[i] != 1:
                continue

            # Wide pulses cannot also be double peaks
            if DETECTION_MODE == "wide" and is_double_peak[i] == 1:
                negative_count += 1
                continue

            # Fat pulses cannot also be double or wide pulses
            if DETECTION_MODE == "fat" and (
                is_double_peak[i] == 1 or is_wide_pulse[i] == 1
            ):
                negative_count += 1
                continue

            is_positive, _ = detect_pulse(pulse[:], fs)

            if is_positive:
                is_detection_array[i] = 1
                positive_count += 1
            else:
                negative_count += 1

            if (i + 1) % max(1, num_pulses // 10) == 0:
                con.log(f"  Processed {i + 1}/{num_pulses} pulses...")

        # Create or overwrite the "is_detection" data array in the h5 file
        if write_mode == "h5":
            if ARRAY_NAME in data_array_names:
                con.log(f"  Updating existing '{ARRAY_NAME}' array...")
                block.data_arrays[ARRAY_NAME][:] = is_detection_array
            else:
                con.log(f"  Creating new '{ARRAY_NAME}' array...")
                block.create_data_array(
                    ARRAY_NAME,
                    ARRAY_NAME,
                    data=is_detection_array,
                )
        else:
            sidecar = save_marker_sidecar(file_path, ARRAY_NAME, is_detection_array)
            con.log(f"  Saved '{ARRAY_NAME}' markers to {sidecar.name}")

        # Log summary
        con.log(
            f"  ✓ Completed: {positive_count} {DISPLAY_NAME}s detected, {negative_count} rejected"
        )

        return {
            "file": Path(file_path).name,
            "status": "completed",
            "num_pulses": num_pulses,
            "num_detected": positive_count,
            "num_rejected": negative_count,
            "ratio": positive_count / num_pulses if num_pulses > 0 else 0,
        }

    finally:
        file.close()


def process_all_h5_files(data_path):
    """
    Process all h5 files in a directory to detect the selected pulse type.

    Parameters:
    -----------
    data_path : Path or str
        Path to directory containing h5 files

    Returns:
    --------
    list
        List of dictionaries with statistics for each file
    """
    data_path = Path(data_path)
    path_list = get_path_list(data_path)

    if not path_list:
        con.log("No h5 files found.")
        return []

    results = []
    con.log(f"\n{'=' * 60}")
    con.log(f"Processing {len(path_list)} h5 files for {DISPLAY_NAME} detection")
    con.log(f"{'=' * 60}\n")

    for i, fp in enumerate(path_list, 1):
        con.log(f"[{i}/{len(path_list)}]")
        result = detect_special_pulses_in_file(fp)
        results.append(result)
        con.log()

    # Print summary statistics
    con.log(f"\n{'=' * 60}")
    con.log("SUMMARY")
    con.log(f"{'=' * 60}")

    # get total numbers across all files
    total_pulses = sum(r.get("num_pulses", 0) for r in results)
    total_detected = sum(r.get("num_detected", 0) for r in results)
    total_rejected = sum(r.get("num_rejected", 0) for r in results)

    # summary logging
    con.log(f"Total files processed: {len(results)}")
    con.log(f"Total pulses analyzed: {total_pulses}")
    con.log(f"Total {DISPLAY_NAME}s: {total_detected}")
    con.log(f"Total rejected: {total_rejected}")
    con.log(
        f"{DISPLAY_NAME.capitalize()} ratio: {total_detected / total_pulses if total_pulses > 0 else 0:.2%}"
    )

    return results


############################################
############# SUPERVISED LEARNING ##########
############################################


def get_default_ml_paths():
    base_path = SPECIAL_PULSE_CLASSIFIER_DIR
    return {
        "base": base_path,
        "labels": base_path / "labeled_special_pulses.npz",
        "model": base_path / "special_pulse_rf_pca.pkl",
        "pca_plot": base_path / "labeled_pulses_pca_space.png",
    }


def get_first_available_array(block, data_array_names, array_names):
    for array_name in array_names:
        if array_name in data_array_names:
            return block.data_arrays[array_name][:], array_name
    return None, None


def expand_marker_to_all_pulses(marker, candidate_indices, num_pulses, array_name):
    """
    Return a full-length marker array, accepting both full-pulse arrays and arrays
    written only for predicted-positive pulses.
    """
    marker = np.asarray(marker, dtype=np.int64)

    if len(marker) == num_pulses:
        return marker

    if len(marker) == len(candidate_indices):
        full_marker = np.zeros(num_pulses, dtype=np.int64)
        full_marker[candidate_indices] = marker
        return full_marker

    raise ValueError(
        f"Array '{array_name}' has length {len(marker)}, but expected either "
        f"{num_pulses} pulses or {len(candidate_indices)} predicted-positive pulses."
    )


def load_balanced_detector_labeled_pulses(
    data_path,
    pulses_per_type=300,
    random_seed=42,
):
    """
    Load a balanced manual-labeling set using old detector output arrays.

    Sampling pools:
        normal: predicted-positive pulses with double == 0, wide == 0, and fat == 0
        double: predicted-positive pulses with double == 1
        wide: predicted-positive pulses with wide == 1
        fat: predicted-positive pulses with fat == 1
    """
    path_list = get_path_list(Path(data_path))
    rng = np.random.default_rng(random_seed)
    pools = {label_id: [] for label_id in LABELING_PULSE_CLASSES}

    for file_idx, file_path in enumerate(path_list, 1):
        con.log(f"  Loading candidates [{file_idx}/{len(path_list)}] {file_path.name}")
        file = open_h5(file_path, nixio.FileMode.ReadOnly)
        if file is None:
            continue

        try:
            block = get_pulse_block(file)
            data_array_names = [da.name for da in block.data_arrays]

            if "raw_pulses" not in data_array_names:
                con.log("    No raw_pulses array. Skipping.")
                continue

            raw_pulses = block.data_arrays["raw_pulses"]
            num_pulses = len(raw_pulses)

            if "predicted_labels" in data_array_names:
                predicted_labels = block.data_arrays["predicted_labels"][:]
                candidate_indices = np.where(predicted_labels == 1)[0]
            else:
                candidate_indices = np.arange(num_pulses)

            if len(candidate_indices) == 0:
                continue

            detector_markers = {}
            missing = []
            for label_id in (1, 2, 3):
                marker, array_name = get_first_available_array(
                    block, data_array_names, LABELING_CLASS_ARRAYS[label_id]
                )
                if marker is None:
                    missing.append(LABELING_CLASS_ARRAYS[label_id][0])
                    continue

                detector_markers[label_id] = expand_marker_to_all_pulses(
                    marker, candidate_indices, num_pulses, array_name
                )

            if missing:
                con.log(f"    Missing {', '.join(missing)}. Skipping.")
                continue

            double_marker = detector_markers[1]
            wide_marker = detector_markers[2]
            fat_marker = detector_markers[3]

            fs = file.sections["pulses_metadata"]["metadata"]["samplerate"]

            candidate_mask = np.zeros(num_pulses, dtype=bool)
            candidate_mask[candidate_indices] = True
            masks = {
                0: candidate_mask
                & (double_marker == 0)
                & (wide_marker == 0)
                & (fat_marker == 0),
                1: candidate_mask & (double_marker == 1),
                2: candidate_mask & (wide_marker == 1),
                3: candidate_mask & (fat_marker == 1),
            }

            for label_id, mask in masks.items():
                for pulse_idx in np.where(mask)[0]:
                    pools[label_id].append(
                        {
                            "file_path": str(file_path),
                            "pulse_idx": int(pulse_idx),
                            "fs": float(fs),
                            "sampling_pool": LABELING_PULSE_CLASSES[label_id],
                        }
                    )

        finally:
            file.close()

    selected_records = []
    for label_id, class_name in LABELING_PULSE_CLASSES.items():
        pool = pools[label_id]
        if not pool:
            con.log(f"  No {class_name} candidates found.")
            continue

        sample_size = min(pulses_per_type, len(pool))
        if len(pool) < pulses_per_type:
            con.log(
                f"  Only {len(pool)} {class_name} candidates available; "
                f"using all of them."
            )
        else:
            con.log(f"  Sampling {sample_size} {class_name} candidates.")

        selected_indices = rng.choice(len(pool), size=sample_size, replace=False)
        selected_records.extend(pool[int(i)] for i in selected_indices)

    if not selected_records:
        return np.empty((0, 0)), []

    shuffle_order = rng.permutation(len(selected_records))
    selected_records = [selected_records[int(i)] for i in shuffle_order]

    waveforms = []
    records = []
    waveform_cache = {}
    for record in selected_records:
        file_path = record["file_path"]
        if file_path not in waveform_cache:
            file = open_h5(file_path, nixio.FileMode.ReadOnly)
            if file is None:
                continue
            try:
                block = get_pulse_block(file)
                waveform_cache[file_path] = block.data_arrays["raw_pulses"][:]
            finally:
                file.close()

        pulse_data = waveform_cache[file_path][record["pulse_idx"]]
        trace, best_channel = get_representative_waveform(pulse_data)
        waveforms.append(trace)
        records.append(
            {
                "file_path": record["file_path"],
                "pulse_idx": record["pulse_idx"],
                "fs": record["fs"],
                "best_channel": int(best_channel),
                "sampling_pool": record["sampling_pool"],
                "all_channels": np.asarray(pulse_data, dtype=float),
            }
        )

    return np.asarray(waveforms, dtype=float), records


def normalize_waveforms_for_pca(waveforms):
    """
    Baseline-correct and amplitude-normalize waveforms before PCA.
    """
    waveforms = np.asarray(waveforms, dtype=float)
    corrected = waveforms.copy()

    baseline_window = max(1, corrected.shape[1] // 5)
    baseline = np.median(corrected[:, :baseline_window], axis=1, keepdims=True)
    corrected -= baseline

    scale = np.max(np.abs(corrected), axis=1, keepdims=True)
    scale[scale == 0] = 1.0
    corrected /= scale

    return corrected


def _make_pca_estimator(n_components, random_state=42):
    return RobustPCA(n_components=n_components, random_state=random_state)


def _get_pca_n_components(X_train):
    X_train = np.asarray(X_train)
    n_components = min(
        PCA_MAX_CLASSIFIER_COMPONENTS,
        X_train.shape[0] - 1,
        X_train.shape[1],
    )
    return max(1, n_components)


def _get_pca_n_components_for_plot(X):
    X = np.asarray(X)
    n_components = min(
        PCA_MAX_PLOT_COMPONENTS,
        X.shape[0] - 1,
        X.shape[1],
    )
    return max(1, n_components)


def _select_meaningful_pc_pairs(explained_variance_ratio, max_pairs=4):
    """
    Choose PC scatter-plot axes beyond PC1/PC2.

    Always includes (PC1, PC2). Adds pairs that involve the next strongest
    components while both axes retain at least 2% explained variance.
    """
    explained = np.asarray(explained_variance_ratio, dtype=float)
    n_components = len(explained)
    if n_components < 2:
        return [(0, 0)]

    candidate_pairs = [(0, 1)]
    if n_components >= 3:
        candidate_pairs.extend([(0, 2), (1, 2)])
    if n_components >= 4:
        candidate_pairs.extend([(0, 3), (2, 3)])
    if n_components >= 5:
        candidate_pairs.append((1, 3))

    seen = set()
    selected = []
    min_variance = 0.02
    for i, j in candidate_pairs:
        if i >= n_components or j >= n_components:
            continue
        if explained[i] < min_variance or explained[j] < min_variance:
            continue
        pair = (min(i, j), max(i, j))
        if pair in seen:
            continue
        seen.add(pair)
        selected.append(pair)
        if len(selected) >= max_pairs:
            break

    return selected or [(0, 1)]


def _fit_pca_projection(waveforms, n_components):
    pca_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", _make_pca_estimator(n_components)),
        ]
    )
    projected = pca_pipeline.fit_transform(waveforms)
    explained = pca_pipeline.named_steps["pca"].explained_variance_ratio_
    return projected, explained, pca_pipeline


def _scatter_labeled_pca_pairs(
    ax,
    projected,
    labels,
    pc_x,
    pc_y,
    explained,
    labels_sorted,
    colors,
    split_name=None,
    split_styles=None,
):
    split_styles = split_styles or {
        "default": {"s": 42, "alpha": 0.75, "linewidths": 0.3, "zorder": 1},
    }
    style_key = split_name if split_name in split_styles else "default"
    style = split_styles[style_key]

    for color, label_id in zip(colors, labels_sorted):
        mask = labels == label_id
        if not np.any(mask):
            continue
        class_name = SPECIAL_PULSE_CLASSES.get(int(label_id), f"class {label_id}")
        legend_label = f"{class_name} (n={int(np.sum(mask))})"
        if split_name is not None:
            legend_label = f"{class_name} ({split_name}, n={int(np.sum(mask))})"

        ax.scatter(
            projected[mask, pc_x],
            projected[mask, pc_y],
            s=style["s"],
            alpha=style["alpha"],
            edgecolors="black",
            linewidths=style["linewidths"],
            color=color,
            zorder=style["zorder"],
            label=legend_label,
        )

    pc_x_var = explained[pc_x] * 100 if pc_x < len(explained) else 0.0
    pc_y_var = explained[pc_y] * 100 if pc_y < len(explained) else 0.0
    ax.set_xlabel(f"PC{pc_x + 1} ({pc_x_var:.1f}% variance)")
    ax.set_ylabel(f"PC{pc_y + 1} ({pc_y_var:.1f}% variance)")
    ax.grid(True, alpha=0.25, linestyle="--")


def normalize_channels_for_label_plot(pulse_data):
    """
    Baseline-correct all channels and normalize them with one shared scale.
    """
    pulse_data = np.asarray(pulse_data, dtype=float)
    corrected = pulse_data.copy()

    baseline_window = max(1, corrected.shape[0] // 5)
    baseline = np.median(corrected[:baseline_window, :], axis=0, keepdims=True)
    corrected -= baseline

    scale = np.max(np.abs(corrected))
    if scale == 0:
        scale = 1.0

    return corrected / scale


def plot_labeling_pulse(ax, waveform, record, label_counts, current_idx, total):
    ax.clear()
    fs = record["fs"]

    if "all_channels" in record:
        pulse_data = normalize_channels_for_label_plot(record["all_channels"])
        time_axis = np.arange(pulse_data.shape[0]) / fs * 1000
        best_channel = record.get("best_channel")

        for channel_idx in range(pulse_data.shape[1]):
            is_best_channel = channel_idx == best_channel
            ax.plot(
                time_axis,
                pulse_data[:, channel_idx],
                linewidth=1.8 if is_best_channel else 0.9,
                alpha=0.95 if is_best_channel else 0.45,
                label=f"ch {channel_idx}" if is_best_channel else None,
            )
    else:
        time_axis = np.arange(len(waveform)) / fs * 1000
        ax.plot(time_axis, waveform, linewidth=2.0, color="steelblue")

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (normalized)")
    ax.grid(True, alpha=0.3, linestyle="--")

    counts = " | ".join(
        f"{name}: {label_counts.get(label_id, 0)}"
        for label_id, name in LABELING_PULSE_CLASSES.items()
    )
    sampled_as = record.get("sampling_pool", "candidate")
    title = (
        f"Label pulse {current_idx + 1}/{total} | "
        f"{Path(record['file_path']).name}, pulse {record['pulse_idx']} | "
        f"sampled as: {sampled_as}\n"
        "[0] normal  [1] double  [2] wide  [3] fat  [S] skip  [Q] finish | "
        f"{counts}"
    )
    ax.set_title(title, fontsize=11, fontweight="bold")


def interactive_label_pulses(
    data_path,
    labels_path=None,
    pulses_per_type=300,
    random_seed=42,
):
    """
    Prompt the user to label pulses in a matplotlib window.

    Labels:
        0 = normal
        1 = double
        2 = wide
        3 = fat
    """
    ml_paths = get_default_ml_paths()
    labels_path = Path(labels_path) if labels_path else ml_paths["labels"]
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    con.log("Loading pulse candidates for manual labeling...")
    waveforms, records = load_balanced_detector_labeled_pulses(
        data_path, pulses_per_type=pulses_per_type, random_seed=random_seed
    )

    if len(waveforms) == 0:
        con.log("No pulse candidates found.")
        return None

    waveforms = normalize_waveforms_for_pca(waveforms)
    labels = np.full(len(waveforms), -1, dtype=np.int64)

    con.log("\nLabeling instructions:")
    con.log("  0 = normal/non-special pulse")
    con.log("  1 = double pulse")
    con.log("  2 = wide pulse")
    con.log("  3 = fat pulse")
    con.log("  S = skip current pulse")
    con.log("  Q = finish and save labels collected so far")

    state = {"idx": 0, "quit": False}
    label_counts = {}

    fig, ax = plt.subplots(figsize=(12, 5))

    def advance():
        while state["idx"] < len(waveforms) and labels[state["idx"]] != -1:
            state["idx"] += 1

        if state["idx"] >= len(waveforms):
            state["quit"] = True
            plt.close(fig)
            return

        plot_labeling_pulse(
            ax,
            waveforms[state["idx"]],
            records[state["idx"]],
            label_counts,
            state["idx"],
            len(waveforms),
        )
        fig.canvas.draw_idle()

    def on_key_press(event):
        if event.key is None:
            return

        key = event.key.lower()
        if key in {"0", "1", "2", "3"}:
            label = int(key)
            labels[state["idx"]] = label
            label_counts[label] = label_counts.get(label, 0) + 1
            con.log(
                f"  Labeled pulse {state['idx'] + 1}/{len(waveforms)} as "
                f"{LABELING_PULSE_CLASSES[label]}"
            )
            state["idx"] += 1
            advance()
        elif key == "s":
            con.log(f"  Skipped pulse {state['idx'] + 1}/{len(waveforms)}")
            state["idx"] += 1
            advance()
        elif key == "q":
            state["quit"] = True
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key_press)
    advance()
    plt.show(block=True)

    labeled_mask = labels != -1
    if not np.any(labeled_mask):
        con.log("No labels collected.")
        return None

    labeled_records = np.array(
        [
            (
                records[i]["file_path"],
                records[i]["pulse_idx"],
                records[i]["fs"],
                records[i]["best_channel"],
            )
            for i in np.where(labeled_mask)[0]
        ],
        dtype=[
            ("file_path", "U512"),
            ("pulse_idx", "i8"),
            ("fs", "f8"),
            ("best_channel", "i8"),
        ],
    )

    save_waveforms = waveforms[labeled_mask]
    save_labels = labels[labeled_mask]
    save_records = labeled_records

    if labels_path.exists():
        existing = np.load(labels_path, allow_pickle=False)
        save_waveforms = np.vstack([existing["waveforms"], save_waveforms])
        save_labels = np.concatenate([existing["labels"], save_labels])
        save_records = np.concatenate([existing["records"], save_records])

    np.savez_compressed(
        labels_path,
        waveforms=save_waveforms,
        labels=save_labels,
        records=save_records,
    )
    con.log(
        f"Saved {np.sum(labeled_mask)} new labels "
        f"({len(save_labels)} total) to {labels_path}"
    )

    return labels_path


def load_labeled_dataset(labels_path):
    labels_path = Path(labels_path)
    if not labels_path.exists():
        raise FileNotFoundError(f"Label dataset does not exist: {labels_path}")

    data = np.load(labels_path, allow_pickle=False)
    return data["waveforms"], data["labels"], data["records"]


def plot_labeled_pulses_pca_space(waveforms, labels, output_path=None, show=True):
    """
    Plot robust-PCA projections of manually labeled pulse waveforms.

    Computes up to PCA_MAX_PLOT_COMPONENTS components and renders the most
    informative PC pairs (PC1/PC2 plus additional high-variance axes).
    """
    if len(labels) < 2:
        con.log("Need at least two labeled pulses to plot PCA space.")
        return None, None

    labels_sorted = sorted(np.unique(labels))
    n_components = _get_pca_n_components_for_plot(waveforms)
    if n_components < 1:
        con.log("No waveform features available to plot PCA space.")
        return None, None

    projected, explained, pca_pipeline = _fit_pca_projection(waveforms, n_components)
    pca_step = pca_pipeline.named_steps["pca"]
    con.log(
        f"Robust PCA plot: {n_components} components "
        f"({pca_step.n_inliers_} inliers, {pca_step.n_outliers_} outliers excluded from fit)"
    )

    pc_pairs = _select_meaningful_pc_pairs(explained)
    n_panels = len(pc_pairs)
    n_cols = min(2, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.5 * n_cols, 5.5 * n_rows),
        squeeze=False,
    )
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(labels_sorted), 1)))

    for panel_idx, (pc_x, pc_y) in enumerate(pc_pairs):
        row_idx, col_idx = divmod(panel_idx, n_cols)
        ax = axes[row_idx, col_idx]
        _scatter_labeled_pca_pairs(
            ax,
            projected,
            labels,
            pc_x,
            pc_y,
            explained,
            labels_sorted,
            colors,
        )
        if panel_idx == 0:
            ax.legend(title="Manual label", frameon=True, fontsize=8)

    for panel_idx in range(n_panels, n_rows * n_cols):
        row_idx, col_idx = divmod(panel_idx, n_cols)
        axes[row_idx, col_idx].set_axis_off()

    fig.suptitle(
        "Robust PCA Space of Manually Labeled Pulses",
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        con.log(f"Saved labeled-pulse PCA plot to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes


def _prepare_classifier_train_test_split(waveforms, labels, records=None):
    """
    Shared stratified train/test split for classifier training and benchmarking.
    """
    waveforms = np.asarray(waveforms, dtype=float)
    labels = np.asarray(labels, dtype=np.int64)
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    if len(unique_labels) < 2:
        raise ValueError("Need at least two labeled classes to train a classifier.")

    min_class_count = int(np.min(label_counts))
    num_classes = len(unique_labels)
    enough_for_stratify = min_class_count >= 2 and len(labels) >= 2 * num_classes
    stratify = labels if enough_for_stratify else None

    if stratify is not None:
        test_count = max(num_classes, int(np.ceil(0.25 * len(labels))))
        test_size = test_count / len(labels)
    else:
        test_size = 0.25 if len(labels) >= 8 else 0.5

    split_indices = np.arange(len(waveforms))
    train_idx, test_idx = train_test_split(
        split_indices,
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )

    labels_sorted = sorted(unique_labels)
    target_names = [SPECIAL_PULSE_CLASSES[int(i)] for i in labels_sorted]
    result = {
        "X_train": waveforms[train_idx],
        "X_test": waveforms[test_idx],
        "y_train": labels[train_idx],
        "y_test": labels[test_idx],
        "labels_sorted": labels_sorted,
        "target_names": target_names,
    }
    if records is not None:
        records = np.asarray(records)
        result["test_records"] = records[test_idx]
    return result


def _compute_multiclass_metrics(y_test, y_pred, labels_sorted):
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=labels_sorted,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_test,
            y_pred,
            labels=labels_sorted,
            average="weighted",
            zero_division=0,
        )
    )
    return {
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
    }


def _build_benchmark_classifier_pipelines():
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "svc_rbf": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "svc",
                    SVC(
                        kernel="rbf",
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=5),
    }


def _wrap_benchmark_pipeline(estimator, feature_space, n_components):
    if isinstance(estimator, Pipeline):
        if feature_space == "pca":
            return Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("pca", _make_pca_estimator(n_components)),
                    ("classifier", estimator.named_steps["svc"]),
                ]
            )
        return estimator

    steps = [("scaler", StandardScaler())]
    if feature_space == "pca":
        steps.append(("pca", _make_pca_estimator(n_components)))
    steps.append(("classifier", estimator))
    return Pipeline(steps=steps)


def plot_benchmark_pca_scatter(
    X_train,
    y_train,
    X_test,
    y_test,
    output_path,
    title,
    fit_on="train",
    show_splits=("train", "test"),
):
    """
    Plot labeled pulses in robust-PCA space across multiple PC pairs.

    fit_on:
        "train" — fit StandardScaler + RobustPCA on the training split only
        (matches the classifier pipeline).
        "full" — fit on train + test combined (exploratory view of full labels).
    show_splits:
        Which splits to draw, e.g. ("train", "test") or ("train", "test") for both.
    """
    show_splits = tuple(show_splits)
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_train = np.asarray(y_train, dtype=np.int64)
    y_test = np.asarray(y_test, dtype=np.int64)

    if fit_on == "train":
        fit_waveforms = X_train
    elif fit_on in {"full", "all"}:
        fit_waveforms = np.vstack([X_train, X_test])
    else:
        raise ValueError("fit_on must be 'train' or 'full'")

    n_components = _get_pca_n_components_for_plot(fit_waveforms)
    projected, explained, pca_pipeline = _fit_pca_projection(
        fit_waveforms, n_components
    )
    pca_step = pca_pipeline.named_steps["pca"]

    split_data = {}
    if "train" in show_splits:
        projected_train = pca_pipeline.transform(X_train)
        split_data["train"] = (projected_train, y_train)
    if "test" in show_splits:
        projected_test = pca_pipeline.transform(X_test)
        split_data["test"] = (projected_test, y_test)

    all_labels = [labels for _, labels in split_data.values()]
    labels_sorted = sorted(np.unique(np.concatenate(all_labels)))
    pc_pairs = _select_meaningful_pc_pairs(explained)
    n_panels = len(pc_pairs)
    n_cols = min(2, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.5 * n_cols, 5.5 * n_rows),
        squeeze=False,
    )
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(labels_sorted), 1)))
    split_styles = {
        "train": {"s": 28, "alpha": 0.35, "linewidths": 0.0, "zorder": 1},
        "test": {"s": 46, "alpha": 0.9, "linewidths": 0.35, "zorder": 2},
    }

    for panel_idx, (pc_x, pc_y) in enumerate(pc_pairs):
        row_idx, col_idx = divmod(panel_idx, n_cols)
        ax = axes[row_idx, col_idx]
        for split_name, (projected, labels) in split_data.items():
            _scatter_labeled_pca_pairs(
                ax,
                projected,
                labels,
                pc_x,
                pc_y,
                explained,
                labels_sorted,
                colors,
                split_name=split_name,
                split_styles=split_styles,
            )
        if panel_idx == 0:
            ax.legend(title="True label (split)", frameon=True, fontsize=7, loc="best")

    for panel_idx in range(n_panels, n_rows * n_cols):
        row_idx, col_idx = divmod(panel_idx, n_cols)
        axes[row_idx, col_idx].set_axis_off()

    fit_note = "fit on train" if fit_on == "train" else "fit on full dataset"
    fig.suptitle(
        (
            f"{title}\n({fit_note}; {n_components} components, "
            f"{pca_step.n_inliers_} inliers, {pca_step.n_outliers_} outliers excluded)"
        ),
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_benchmark_waveform_sanity(
    X_test,
    y_test,
    y_pred,
    test_records,
    output_path,
    title,
    examples_per_type=2,
):
    labels_sorted = sorted(np.unique(y_test))
    n_rows = len(labels_sorted)
    n_cols = 2
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(11, 2.8 * n_rows),
        squeeze=False,
    )

    for row_idx, label_id in enumerate(labels_sorted):
        class_name = SPECIAL_PULSE_CLASSES.get(int(label_id), f"class {label_id}")
        class_mask = y_test == label_id
        class_indices = np.where(class_mask)[0]

        correct_indices = class_indices[y_pred[class_indices] == label_id]
        wrong_indices = class_indices[y_pred[class_indices] != label_id]

        selections = [
            ("correct", correct_indices),
            ("misclassified", wrong_indices),
        ]

        for col_idx, (example_type, candidate_indices) in enumerate(selections):
            ax = axes[row_idx, col_idx]
            if len(candidate_indices) == 0:
                ax.text(
                    0.5,
                    0.5,
                    f"No {example_type} examples",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_axis_off()
                continue

            pick_count = min(examples_per_type, len(candidate_indices))
            picked = candidate_indices[:pick_count]

            for example_idx in picked:
                waveform = X_test[example_idx]
                record = test_records[example_idx]
                fs = float(record["fs"]) if record is not None else 1.0
                time_axis = np.arange(len(waveform)) / fs * 1000
                pred_name = SPECIAL_PULSE_CLASSES.get(
                    int(y_pred[example_idx]), f"class {y_pred[example_idx]}"
                )
                ax.plot(
                    time_axis,
                    waveform,
                    alpha=0.85,
                    linewidth=1.4,
                    label=f"pred={pred_name}",
                )

            ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Amplitude")
            ax.grid(True, alpha=0.25, linestyle="--")
            ax.legend(fontsize=8, loc="upper right")
            ax.set_title(
                f"{class_name}: {example_type} (true={class_name})",
                fontsize=10,
                fontweight="bold",
            )

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _print_benchmark_comparison_table(results):
    table = Table(title="Pulse classifier benchmark (held-out test set)")
    table.add_column("Run", style="cyan")
    table.add_column("Classifier")
    table.add_column("Features")
    table.add_column("Macro P/R/F1")
    table.add_column("Weighted P/R/F1")
    table.add_column("Best?", justify="center")

    best_macro_f1 = max(row["macro_f1"] for row in results)
    best_rows = [row for row in results if row["macro_f1"] == best_macro_f1]

    for row in results:
        is_best = row in best_rows
        table.add_row(
            row["run_id"],
            row["classifier"],
            row["feature_space"],
            (
                f"{row['macro_precision']:.3f} / "
                f"{row['macro_recall']:.3f} / "
                f"{row['macro_f1']:.3f}"
            ),
            (
                f"{row['weighted_precision']:.3f} / "
                f"{row['weighted_recall']:.3f} / "
                f"{row['weighted_f1']:.3f}"
            ),
            "★" if is_best else "",
        )

    con.print(table)
    return best_rows


def _summarize_benchmark_findings(results, best_rows):
    pca_runs = [row for row in results if row["feature_space"] == "pca"]
    raw_runs = [row for row in results if row["feature_space"] == "raw"]
    best_pca = max(pca_runs, key=lambda row: row["macro_f1"])
    best_raw = max(raw_runs, key=lambda row: row["macro_f1"])
    pca_vs_raw_gap = best_pca["macro_f1"] - best_raw["macro_f1"]

    classifier_best = {}
    for row in results:
        current = classifier_best.get(row["classifier"])
        if current is None or row["macro_f1"] > current["macro_f1"]:
            classifier_best[row["classifier"]] = row

    ranked_classifiers = sorted(
        classifier_best.values(), key=lambda row: row["macro_f1"], reverse=True
    )
    top = ranked_classifiers[0]
    second = ranked_classifiers[1] if len(ranked_classifiers) > 1 else None
    classifier_gap = top["macro_f1"] - second["macro_f1"] if second else 0.0

    con.log("\nBenchmark summary:")
    con.log(
        f"  Best overall: {best_rows[0]['classifier']} + {best_rows[0]['feature_space']} "
        f"(macro F1={best_rows[0]['macro_f1']:.3f})"
    )
    if len(best_rows) > 1:
        tied = ", ".join(
            f"{row['classifier']}+{row['feature_space']}" for row in best_rows
        )
        con.log(f"  Tied best runs: {tied}")

    if abs(pca_vs_raw_gap) < 0.02:
        pca_verdict = "marginal difference"
    elif pca_vs_raw_gap > 0:
        pca_verdict = f"PCA slightly better by {pca_vs_raw_gap:.3f} macro F1"
    else:
        pca_verdict = f"raw waveforms slightly better by {-pca_vs_raw_gap:.3f} macro F1"
    con.log(f"  PCA vs raw: {pca_verdict}")

    if classifier_gap < 0.02:
        classifier_verdict = "marginal difference between top classifiers"
    else:
        classifier_verdict = (
            f"{top['classifier']} leads by {classifier_gap:.3f} macro F1 "
            f"over {second['classifier']}"
        )
    con.log(f"  Classifier spread: {classifier_verdict}")


def benchmark_pulse_classifiers(labels_path=None):
    """
    Compare multiclass classifiers on PCA features vs raw waveforms.

    Saves metrics and plots under SPECIAL_PULSE_CLASSIFIER_DIR/benchmark/.
    """
    ml_paths = get_default_ml_paths()
    labels_path = Path(labels_path) if labels_path else ml_paths["labels"]
    benchmark_dir = ml_paths["base"] / "benchmark"
    pca_plot_dir = benchmark_dir / "pca"
    waveform_plot_dir = benchmark_dir / "waveforms"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    if not labels_path.exists():
        con.log(f"No labeled dataset found at {labels_path}.")
        con.log("Run the supervised labeling workflow first (main menu option 2).")
        return None

    waveforms, labels, records = load_labeled_dataset(labels_path)
    active_label_mask = np.isin(labels, list(LABELING_PULSE_CLASSES))
    if not np.all(active_label_mask):
        ignored_count = int(np.sum(~active_label_mask))
        con.log(f"Ignoring {ignored_count} labels outside normal/double/wide/fat.")
        waveforms = waveforms[active_label_mask]
        labels = labels[active_label_mask]
        records = records[active_label_mask]

    if len(labels) == 0:
        con.log("No normal/double/wide/fat labels available for benchmarking.")
        return None

    split = _prepare_classifier_train_test_split(waveforms, labels, records=records)
    X_train = split["X_train"]
    X_test = split["X_test"]
    y_train = split["y_train"]
    y_test = split["y_test"]
    labels_sorted = split["labels_sorted"]
    test_records = split["test_records"]
    n_components = _get_pca_n_components(X_train)

    con.log("\n" + "=" * 60)
    con.log("PULSE CLASSIFIER BENCHMARK")
    con.log("=" * 60)
    con.log(f"Labels: {labels_path}")
    con.log(f"Train/test split: {len(y_train)} / {len(y_test)} pulses")
    con.log(f"PCA components (when used): {n_components}")
    con.log(f"Output directory: {benchmark_dir}")
    con.log("=" * 60)

    pca_plot_train_test_path = pca_plot_dir / "dataset_pca_train_test.png"
    pca_plot_full_fit_path = pca_plot_dir / "dataset_pca_full_fit.png"
    con.log("Saving shared PCA scatter plots (classifier-independent)...")
    plot_benchmark_pca_scatter(
        X_train,
        y_train,
        X_test,
        y_test,
        pca_plot_train_test_path,
        title="Labeled pulses in PCA space",
        fit_on="train",
        show_splits=("train", "test"),
    )
    plot_benchmark_pca_scatter(
        X_train,
        y_train,
        X_test,
        y_test,
        pca_plot_full_fit_path,
        title="Labeled pulses in PCA space",
        fit_on="full",
        show_splits=("train", "test"),
    )
    con.log(
        "PCA plots are identical across classifier runs because they visualize "
        "waveform geometry, not model predictions."
    )

    classifier_estimators = _build_benchmark_classifier_pipelines()
    feature_spaces = ("pca", "raw")
    results = []

    for classifier_name, estimator in classifier_estimators.items():
        for feature_space in feature_spaces:
            run_id = f"{classifier_name}__{feature_space}"
            con.log(f"Training {classifier_name} on {feature_space} features...")

            pipeline = _wrap_benchmark_pipeline(
                estimator, feature_space, n_components
            )
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            metrics = _compute_multiclass_metrics(y_test, y_pred, labels_sorted)

            waveform_plot_path = waveform_plot_dir / f"{run_id}_waveforms.png"
            plot_benchmark_waveform_sanity(
                X_test,
                y_test,
                y_pred,
                test_records,
                waveform_plot_path,
                title=(
                    f"Test-set waveform examples | {classifier_name} | "
                    f"{feature_space} features"
                ),
            )

            run_result = {
                "run_id": run_id,
                "classifier": classifier_name,
                "feature_space": feature_space,
                **metrics,
                "pca_plot_train_test": str(pca_plot_train_test_path),
                "pca_plot_full_fit": str(pca_plot_full_fit_path),
                "waveform_plot": str(waveform_plot_path),
            }
            results.append(run_result)

            con.log(
                f"  macro F1={metrics['macro_f1']:.3f}, "
                f"weighted F1={metrics['weighted_f1']:.3f}"
            )

    best_rows = _print_benchmark_comparison_table(results)
    _summarize_benchmark_findings(results, best_rows)

    metrics_json_path = benchmark_dir / "benchmark_metrics.json"
    metrics_csv_path = benchmark_dir / "benchmark_metrics.csv"
    best_macro_f1 = max(row["macro_f1"] for row in results)

    with metrics_json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "labels_path": str(labels_path),
                "train_size": int(len(y_train)),
                "test_size": int(len(y_test)),
                "pca_n_components": int(n_components),
                "pca_plot_train_test": str(pca_plot_train_test_path),
                "pca_plot_full_fit": str(pca_plot_full_fit_path),
                "best_macro_f1": float(best_macro_f1),
                "best_runs": [
                    {
                        "run_id": row["run_id"],
                        "classifier": row["classifier"],
                        "feature_space": row["feature_space"],
                        "macro_f1": row["macro_f1"],
                    }
                    for row in best_rows
                ],
                "results": results,
            },
            f,
            indent=2,
        )

    fieldnames = [
        "run_id",
        "classifier",
        "feature_space",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "is_best",
        "pca_plot_train_test",
        "pca_plot_full_fit",
        "waveform_plot",
    ]
    with metrics_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    **{key: row[key] for key in fieldnames if key != "is_best"},
                    "is_best": row["macro_f1"] == best_macro_f1,
                }
            )

    con.log(f"Saved benchmark metrics to {metrics_json_path}")
    con.log(f"Saved benchmark metrics to {metrics_csv_path}")
    con.log(f"Saved PCA plots to {pca_plot_dir}")
    con.log(f"Saved waveform sanity plots to {waveform_plot_dir}")

    return {
        "benchmark_dir": benchmark_dir,
        "results": results,
        "best_runs": best_rows,
    }


def train_special_pulse_classifier(labels_path=None, model_path=None, show_pca_plot=False):
    """
    Train a robust PCA + random forest multiclass classifier and print precision/recall/f1.
    """
    ml_paths = get_default_ml_paths()
    labels_path = Path(labels_path) if labels_path else ml_paths["labels"]
    model_path = Path(model_path) if model_path else ml_paths["model"]
    model_path.parent.mkdir(parents=True, exist_ok=True)

    waveforms, labels, _ = load_labeled_dataset(labels_path)
    active_label_mask = np.isin(labels, list(LABELING_PULSE_CLASSES))
    if not np.all(active_label_mask):
        ignored_count = int(np.sum(~active_label_mask))
        con.log(f"Ignoring {ignored_count} labels outside normal/double/wide/fat.")
        waveforms = waveforms[active_label_mask]
        labels = labels[active_label_mask]
    if len(labels) == 0:
        raise ValueError(
            "No normal/double/wide/fat labels available to train a classifier."
        )

    unique_labels, label_counts = np.unique(labels, return_counts=True)
    if len(unique_labels) < 2:
        raise ValueError("Need at least two labeled classes to train a classifier.")

    plot_labeled_pulses_pca_space(
        waveforms,
        labels,
        output_path=ml_paths["pca_plot"],
        show=show_pca_plot,
    )

    split = _prepare_classifier_train_test_split(waveforms, labels)
    X_train = split["X_train"]
    X_test = split["X_test"]
    y_train = split["y_train"]
    y_test = split["y_test"]
    labels_sorted = split["labels_sorted"]
    target_names = split["target_names"]
    unique_labels = labels_sorted

    n_components = _get_pca_n_components(X_train)

    classifier = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", _make_pca_estimator(n_components)),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    con.log("\nClassifier performance on held-out labeled pulses:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=labels_sorted,
            target_names=target_names,
            zero_division=0,
        )
    )
    con.log("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred, labels=labels_sorted))

    metrics = _compute_multiclass_metrics(y_test, y_pred, labels_sorted)
    macro_precision = metrics["macro_precision"]
    macro_recall = metrics["macro_recall"]
    macro_f1 = metrics["macro_f1"]
    weighted_precision = metrics["weighted_precision"]
    weighted_recall = metrics["weighted_recall"]
    weighted_f1 = metrics["weighted_f1"]

    with model_path.open("wb") as f:
        pickle.dump(
            {
                "classifier": classifier,
                "classes": SPECIAL_PULSE_CLASSES,
                "array_names": SPECIAL_CLASS_ARRAYS,
                "multiclass_array": MULTICLASS_ARRAY_NAME,
            },
            f,
        )

    con.log(f"Saved trained classifier to {model_path}")
    con.log("\nFinal held-out classifier metrics:")
    print(
        f"macro    precision={macro_precision:.3f} "
        f"recall={macro_recall:.3f} f1={macro_f1:.3f}"
    )
    print(
        f"weighted precision={weighted_precision:.3f} "
        f"recall={weighted_recall:.3f} f1={weighted_f1:.3f}"
    )
    return model_path


def write_or_create_data_array(block, array_name, values):
    data_array_names = [da.name for da in block.data_arrays]
    values = np.asarray(values, dtype=np.int64)

    if array_name in data_array_names:
        block.data_arrays[array_name][:] = values
    else:
        block.create_data_array(array_name, array_name, data=values)


def predict_special_pulses_in_file(file_path, classifier):
    file = open_h5(file_path, nixio.FileMode.ReadWrite)
    if file is None:
        return {"status": "skipped", "reason": "locked_or_unreadable"}

    try:
        block = get_pulse_block(file)
        data_array_names = [da.name for da in block.data_arrays]

        if "raw_pulses" not in data_array_names:
            con.log(f"  {Path(file_path).name}: no raw_pulses array. Skipping.")
            return {"status": "skipped", "reason": "no raw_pulses"}

        raw_pulses = block.data_arrays["raw_pulses"]
        num_pulses = len(raw_pulses)

        if "predicted_labels" in data_array_names:
            predicted_labels = block.data_arrays["predicted_labels"][:]
            candidate_indices = np.where(predicted_labels == 1)[0]
        else:
            candidate_indices = np.arange(num_pulses)

        predicted_classes = np.zeros(num_pulses, dtype=np.int64)

        if len(candidate_indices) > 0:
            waveforms = []
            for pulse_idx in candidate_indices:
                pulse_data = raw_pulses[int(pulse_idx)][:]
                trace, _ = get_representative_waveform(pulse_data)
                waveforms.append(trace)

            waveforms = normalize_waveforms_for_pca(np.asarray(waveforms, dtype=float))
            predicted_classes[candidate_indices] = classifier.predict(waveforms)

        write_or_create_data_array(block, MULTICLASS_ARRAY_NAME, predicted_classes)

        for label_id, array_name in SPECIAL_CLASS_ARRAYS.items():
            binary_values = (predicted_classes == label_id).astype(np.int64)
            write_or_create_data_array(block, array_name, binary_values)

        counts = {
            SPECIAL_PULSE_CLASSES[label_id]: int(np.sum(predicted_classes == label_id))
            for label_id in SPECIAL_PULSE_CLASSES
        }

        con.log(f"  {Path(file_path).name}: {counts}")
        return {"status": "completed", "counts": counts}

    finally:
        file.close()


def apply_special_pulse_classifier(data_path, model_path=None):
    ml_paths = get_default_ml_paths()
    model_path = Path(model_path) if model_path else ml_paths["model"]

    with model_path.open("rb") as f:
        model_data = pickle.load(f)

    classifier = model_data["classifier"]
    path_list = get_path_list(Path(data_path))

    results = []
    for file_idx, file_path in enumerate(path_list, 1):
        con.log(f"Predicting [{file_idx}/{len(path_list)}] {file_path.name}")
        results.append(predict_special_pulses_in_file(file_path, classifier))

    return results


def supervised_learning_workflow(data_path):
    """
    End-to-end workflow:
    1. optionally label pulses,
    2. train/evaluate PCA + random forest classifier,
    3. apply predictions to h5 files.
    """
    ml_paths = get_default_ml_paths()

    con.log("\n" + "=" * 60)
    con.log("SUPERVISED SPECIAL PULSE CLASSIFIER")
    con.log("=" * 60)
    con.log(f"Label dataset: {ml_paths['labels']}")
    con.log(f"Model file:     {ml_paths['model']}")
    con.log("=" * 60)

    should_label = input("Label pulses now? [y/N]: ").strip().lower() == "y"
    if should_label:
        pulses_per_type_raw = input(
            "Pulses to sample per type for labeling [300]: "
        ).strip()
        pulses_per_type = int(pulses_per_type_raw) if pulses_per_type_raw else 300
        interactive_label_pulses(
            data_path,
            labels_path=ml_paths["labels"],
            pulses_per_type=pulses_per_type,
        )

    should_train = (
        input("Train classifier from labeled pulses? [Y/n]: ").strip().lower()
    )
    if should_train != "n":
        train_special_pulse_classifier(
            labels_path=ml_paths["labels"],
            model_path=ml_paths["model"],
        )

    should_apply = input("Apply classifier to h5 files now? [Y/n]: ").strip().lower()
    if should_apply != "n":
        apply_special_pulse_classifier(data_path, model_path=ml_paths["model"])


if __name__ == "__main__":
    data_path = H5_DIR

    con.log("\n" + "=" * 60)
    con.log("SPECIAL PULSE DETECTION OPTIONS")
    con.log("=" * 60)
    con.log("1. Run old rule-based detector")
    con.log("2. Run supervised PCA + random forest workflow")
    con.log("3. Benchmark pulse classifiers (PCA vs raw)")
    con.log("=" * 60)

    mode = input("Select mode (1, 2, or 3): ").strip()

    if mode == "1":
        # Process all h5 files to detect double peaks or wide pulses
        results = process_all_h5_files(data_path)
    elif mode == "2":
        supervised_learning_workflow(data_path)
    elif mode == "3":
        benchmark_pulse_classifiers()
    else:
        con.log("Invalid selection. Please enter 1, 2, or 3.")
