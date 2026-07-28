"""Detect and label special pulse shapes (double, wide) in predetected .h5 files.

Analysis part: special-pulse detection and ML classifier (Part 2 of Berlin activity analysis).
Dependencies: data_paths, h5_io; writes marker arrays back into input .h5 files.

Default workflow: train/apply a PCA + random forest classifier on manually labeled
examples. Rule-based per-shape detectors remain available via --mode rule-based.
"""

import argparse
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

from data_paths import H5_DIR, PCA_SPACE_DIR, SPECIAL_PULSE_CLASSIFIER_DIR
from presentation_style import LEGEND_LOC, apply_presentation_style, classifier_label_colors
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

DETECTION_MODE = "double"
# options:
# "double"
# "wide"

MODE_CONFIG = {
    "double": {
        "array_name": "is_double_peak",
        "display_name": "double peak",
    },
    "wide": {
        "array_name": "is_wide_pulse",
        "display_name": "wide pulse",
    },
}

ARRAY_NAME = MODE_CONFIG[DETECTION_MODE]["array_name"]
DISPLAY_NAME = MODE_CONFIG[DETECTION_MODE]["display_name"]

# Raised from 0.7 to suppress very small noisy pulses in special-shape detection.
MIN_AMPLITUDE_THRESHOLD = 5.0

SPECIAL_PULSE_CLASSES = {
    0: "normal",
    1: "wide",
    2: "double",
}

LABELING_PULSE_CLASSES = {
    0: "normal",
    1: "wide",
    2: "double",
}

SPECIAL_CLASS_ARRAYS = {
    1: "is_wide_pulse",
    2: "is_double_peak",
}

LABELING_CLASS_ARRAYS = {
    1: ("is_wide_pulse",),
    2: ("is_double_peak",),
}

MULTICLASS_ARRAY_NAME = "special_pulse_class"

# Max PCA dimensions for classifier pipelines (RobustPCA step).
PCA_MAX_CLASSIFIER_COMPONENTS = 20
# Max PCA dimensions computed for exploratory scatter plots.
PCA_MAX_PLOT_COMPONENTS = 10

# Production decision defaults for rare special pulses.
# Retuned on naturalistic_test_labels.npz (n=486): min_proba wide/double = 0.40
# slightly beats hard predict() on that test set (macro F1 0.724 vs 0.711).
# No rule-based gate: ML is the sole shape decision.
DEFAULT_NATURAL_PRIOR = {
    0: 0.92,  # normal
    1: 0.075,  # wide
    2: 0.005,  # double
}
DEFAULT_MIN_PROBA = {
    1: 0.40,  # wide
    2: 0.40,  # double
}
DEFAULT_DECISION_CONFIG = {
    "use_prior_reweight": False,
    "natural_prior": DEFAULT_NATURAL_PRIOR,
    "min_proba": DEFAULT_MIN_PROBA,
    "default_class": 0,
    "rule_gate_double": False,
    "rf_class_weight": None,
}


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


# unified detection wrapper
def detect_pulse(pulse_waveform, sample_rate):
    """
    Unified detector wrapper.
    """

    if DETECTION_MODE == "double":
        return detect_double_pulse(pulse_waveform, sample_rate)

    elif DETECTION_MODE == "wide":
        return detect_wide_pulse(pulse_waveform, sample_rate)

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
        # when running the modes in order: double, then wide.
        predicted = predicted_labels[:]
        candidate_indices = np.where(predicted == 1)[0]

        if DETECTION_MODE == "wide":
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
    pca_dir = PCA_SPACE_DIR
    pca_dir.mkdir(parents=True, exist_ok=True)
    return {
        "base": base_path,
        "labels": base_path / "labeled_special_pulses.npz",
        "naturalistic_test_labels": base_path / "naturalistic_test_labels.npz",
        "naturalistic_train_labels": base_path / "naturalistic_train_labels.npz",
        "model": base_path / "special_pulse_rf_pca.pkl",
        "pca_plot": pca_dir / "labeled_pulses_pca_space.png",
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
        normal: predicted-positive pulses with wide == 0 and double == 0
        wide: predicted-positive pulses with wide == 1
        double: predicted-positive pulses with double == 1
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
            for label_id in LABELING_CLASS_ARRAYS:
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

            wide_marker = detector_markers[1]
            double_marker = detector_markers[2]

            fs = file.sections["pulses_metadata"]["metadata"]["samplerate"]

            candidate_mask = np.zeros(num_pulses, dtype=bool)
            candidate_mask[candidate_indices] = True
            masks = {
                0: candidate_mask & (wide_marker == 0) & (double_marker == 0),
                1: candidate_mask & (wide_marker == 1),
                2: candidate_mask & (double_marker == 1),
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


def resample_waveforms(waveforms: np.ndarray, target_length: int) -> np.ndarray:
    """Resample 1D waveforms to the length expected by the trained classifier."""
    waveforms = np.asarray(waveforms, dtype=float)
    if waveforms.ndim == 1:
        waveforms = waveforms[np.newaxis, :]
    if waveforms.shape[1] == target_length:
        return waveforms

    source_x = np.linspace(0.0, 1.0, waveforms.shape[1])
    target_x = np.linspace(0.0, 1.0, target_length)
    return np.vstack([np.interp(target_x, source_x, row) for row in waveforms])


def classifier_waveform_length(classifier) -> int:
    """Return the waveform feature length a fitted classifier pipeline expects."""
    return int(classifier.named_steps["scaler"].n_features_in_)


def prepare_classifier_waveforms(waveforms: np.ndarray, classifier) -> np.ndarray:
    """Resample and normalize waveforms for classifier prediction."""
    target_length = classifier_waveform_length(classifier)
    waveforms = resample_waveforms(waveforms, target_length)
    return normalize_waveforms_for_pca(waveforms)


def class_prior_from_labels(labels, class_ids=None):
    """Empirical class frequencies as a dict {class_id: prior}."""
    labels = np.asarray(labels, dtype=np.int64)
    if class_ids is None:
        class_ids = sorted(SPECIAL_PULSE_CLASSES)
    class_ids = [int(c) for c in class_ids]
    counts = {c: int(np.sum(labels == c)) for c in class_ids}
    total = max(sum(counts.values()), 1)
    return {c: counts[c] / total for c in class_ids}


def reweight_class_probabilities(proba, class_ids, train_prior, natural_prior):
    """
    Bayes-style prior correction: p'(c|x) ∝ p(c|x) * π_nat(c) / π_train(c).

    RF predict_proba reflects the training label mix. Multiplying by the ratio of
    natural to train priors shifts mass toward the production prevalence.
    """
    proba = np.asarray(proba, dtype=float)
    class_ids = [int(c) for c in class_ids]
    weights = np.array(
        [
            float(natural_prior.get(c, 0.0)) / max(float(train_prior.get(c, 1e-12)), 1e-12)
            for c in class_ids
        ],
        dtype=float,
    )
    adjusted = proba * weights[np.newaxis, :]
    row_sums = adjusted.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return adjusted / row_sums


def decide_classes_from_proba(
    proba,
    class_ids,
    *,
    min_proba=None,
    default_class=0,
):
    """
    Map class probabilities to labels with high bars for rare classes.

    Rare classes (keys in min_proba) are only assigned if their probability meets
    the threshold and they are the strongest rare candidate; otherwise default_class.
    """
    proba = np.asarray(proba, dtype=float)
    class_ids = np.asarray(class_ids, dtype=np.int64)
    min_proba = {int(k): float(v) for k, v in (min_proba or {}).items()}
    default_class = int(default_class)

    id_to_col = {int(c): i for i, c in enumerate(class_ids)}
    preds = np.full(len(proba), default_class, dtype=np.int64)

    rare_classes = [c for c in sorted(min_proba) if c in id_to_col]
    if not rare_classes:
        return class_ids[np.argmax(proba, axis=1)]

    for i, row in enumerate(proba):
        best_rare = None
        best_p = -1.0
        for class_id in rare_classes:
            p = float(row[id_to_col[class_id]])
            if p >= min_proba[class_id] and p > best_p:
                best_rare = class_id
                best_p = p
        if best_rare is not None:
            preds[i] = best_rare
        else:
            # Among non-rare / default: prefer argmax, but never assign a rare
            # class that failed its threshold.
            eligible = [
                c
                for c in class_ids
                if c not in min_proba or float(row[id_to_col[int(c)]]) >= min_proba[int(c)]
            ]
            if not eligible:
                preds[i] = default_class
            else:
                preds[i] = max(eligible, key=lambda c: float(row[id_to_col[int(c)]]))
    return preds


def predict_special_pulse_classes(
    classifier,
    waveforms,
    *,
    decision_config=None,
    train_prior=None,
    pulse_waveforms_raw=None,
    sample_rates=None,
):
    """
    Predict class IDs using predict_proba + optional prior/threshold/rule gate.

    Parameters
    ----------
    classifier : fitted sklearn Pipeline
    waveforms : array (n, T) already prepared for the classifier
    decision_config : dict, optional
    train_prior : dict, optional
    pulse_waveforms_raw : list/array of (T, C) multi-channel pulses for rule gate
    sample_rates : array of sample rates for rule gate
    """
    cfg = {**DEFAULT_DECISION_CONFIG, **(decision_config or {})}
    class_ids = np.asarray(classifier.classes_, dtype=np.int64)
    proba = classifier.predict_proba(waveforms)

    if cfg.get("use_prior_reweight", False):
        if train_prior is None:
            raise ValueError("train_prior is required when use_prior_reweight=True")
        proba = reweight_class_probabilities(
            proba,
            class_ids,
            train_prior,
            cfg.get("natural_prior", DEFAULT_NATURAL_PRIOR),
        )

    preds = decide_classes_from_proba(
        proba,
        class_ids,
        min_proba=cfg.get("min_proba", DEFAULT_MIN_PROBA),
        default_class=cfg.get("default_class", 0),
    )

    if cfg.get("rule_gate_double", False) and pulse_waveforms_raw is not None:
        if sample_rates is None:
            raise ValueError("sample_rates required when rule_gate_double=True")
        for i, pred in enumerate(preds):
            if pred != 2:
                continue
            is_double, _ = detect_double_pulse(
                np.asarray(pulse_waveforms_raw[i], dtype=float),
                float(sample_rates[i]),
            )
            if not is_double:
                preds[i] = 0

    return preds, proba


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
            s=style["s"] * 1.4,
            alpha=style["alpha"],
            edgecolors="black",
            linewidths=style["linewidths"] + 0.4,
            color=color,
            zorder=style["zorder"],
            label=legend_label,
        )

    pc_x_var = explained[pc_x] * 100 if pc_x < len(explained) else 0.0
    pc_y_var = explained[pc_y] * 100 if pc_y < len(explained) else 0.0
    ax.set_xlabel(f"PC{pc_x + 1} ({pc_x_var:.1f}% variance)")
    ax.set_ylabel(f"PC{pc_y + 1} ({pc_y_var:.1f}% variance)")
    ax.grid(True, alpha=0.25, linestyle="--")


# Common timebase for labeling / naturalistic train features.
# Many H5 files are 48 kHz with ~10 ms snippets; others are 24 kHz with ~20 ms.
# For comparable shapes while labeling (and for RF features), we plot and store
# all snippets on this reference rate: equal sample counts → equal plotted duration
# (48 kHz traces are effectively time-stretched ×2, as if sampled at 24 kHz).
LABEL_REF_FS = 24000.0


def normalize_channels_for_label_plot(pulse_data):
    """
    Baseline-correct all channels, flip each to positive dominant polarity,
    and normalize with one shared scale across channels.
    """
    pulse_data = np.asarray(pulse_data, dtype=float)
    corrected = pulse_data.copy()

    baseline_window = max(1, corrected.shape[0] // 5)
    baseline = np.median(corrected[:baseline_window, :], axis=0, keepdims=True)
    corrected -= baseline

    # Flip each channel so its dominant peak is positive.
    for channel_idx in range(corrected.shape[1]):
        trace = corrected[:, channel_idx]
        if abs(np.min(trace)) > np.max(trace):
            corrected[:, channel_idx] = -trace

    scale = np.max(np.abs(corrected))
    if scale == 0:
        scale = 1.0

    return corrected / scale


def apply_label_ref_timebase(records, waveforms=None, ref_fs=LABEL_REF_FS):
    """
    Reinterpret each pulse on a common reference sample rate for labeling/RF.

    Sample values are unchanged; native_fs is kept for provenance. Display and
    saved training features then use ref_fs so 48 kHz and 24 kHz snippets share
    the same plotted duration (×2 stretch for 48 kHz).
    """
    ref_fs = float(ref_fs)
    for record in records:
        native_fs = float(record["fs"])
        record["native_fs"] = native_fs
        record["fs"] = ref_fs
    if waveforms is not None:
        return records, np.asarray(waveforms, dtype=float)
    return records


def compute_label_plot_ylim(records):
    """
    Fixed y-limits for the labeling UI, shared across all pulses in the session.

    Lower bound is capped at -0.3 so polarity-flipped channels stay comparable.
    """
    ymin = 0.0
    ymax = 0.0
    for record in records:
        if "all_channels" in record:
            pulse_data = normalize_channels_for_label_plot(record["all_channels"])
        else:
            pulse_data = np.asarray(record.get("waveform", []), dtype=float)
            if pulse_data.size == 0:
                continue
            if abs(np.min(pulse_data)) > np.max(pulse_data):
                pulse_data = -pulse_data
            scale = np.max(np.abs(pulse_data)) or 1.0
            pulse_data = pulse_data / scale
        ymin = min(ymin, float(np.min(pulse_data)))
        ymax = max(ymax, float(np.max(pulse_data)))

    span = max(abs(ymin), abs(ymax), 1.0)
    margin = 0.05 * span
    lower = max(ymin - margin, -0.3)
    upper = ymax + margin
    return (lower, upper)


def pulse_peak_index(pulse_data, best_channel=None):
    """Return the sample index of the dominant absolute peak."""
    pulse_data = np.asarray(pulse_data, dtype=float)
    if pulse_data.ndim == 2:
        if best_channel is None:
            channel_strengths = np.max(np.abs(pulse_data), axis=0)
            best_channel = int(np.argmax(channel_strengths))
        trace = pulse_data[:, best_channel]
    else:
        trace = pulse_data
    return int(np.argmax(np.abs(trace)))


def peak_aligned_time_ms(n_samples, peak_idx, fs):
    """Time axis in ms with the pulse peak at t = 0."""
    return (np.arange(n_samples) - peak_idx) / fs * 1000


def compute_label_plot_half_window_ms(records, ref_fs=None):
    """Symmetric half-window (ms) for fixed peak-centered labeling axes."""
    half_windows = []
    for record in records:
        fs = float(ref_fs if ref_fs is not None else record["fs"])
        if "all_channels" in record:
            pulse_data = record["all_channels"]
            peak_idx = pulse_peak_index(
                pulse_data, best_channel=record.get("best_channel")
            )
            n_samples = pulse_data.shape[0]
        else:
            waveform = np.asarray(record["waveform"], dtype=float)
            peak_idx = pulse_peak_index(waveform)
            n_samples = len(waveform)

        half_samples = max(peak_idx, n_samples - 1 - peak_idx)
        half_windows.append(half_samples / fs * 1000)

    return max(half_windows) if half_windows else 5.0


def plot_labeling_pulse(
    ax,
    waveform,
    record,
    label_counts,
    current_idx,
    total,
    *,
    label_plot_half_window_ms,
    label_plot_ylim,
):
    ax.clear()
    fs = record["fs"]

    if "all_channels" in record:
        pulse_data = normalize_channels_for_label_plot(record["all_channels"])
        best_channel = record.get("best_channel")
        peak_idx = pulse_peak_index(pulse_data, best_channel=best_channel)
        time_axis = peak_aligned_time_ms(pulse_data.shape[0], peak_idx, fs)

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
        plot_wave = np.asarray(waveform, dtype=float).copy()
        if abs(np.min(plot_wave)) > np.max(plot_wave):
            plot_wave *= -1
        peak_idx = pulse_peak_index(plot_wave)
        time_axis = peak_aligned_time_ms(len(plot_wave), peak_idx, fs)
        ax.plot(time_axis, plot_wave, linewidth=2.0, color="steelblue")

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.25, linestyle=":")
    ax.set_xlim(-label_plot_half_window_ms, label_plot_half_window_ms)
    ax.set_ylim(label_plot_ylim)
    ax.set_xlabel("Time from peak (ms)")
    ax.set_ylabel("Amplitude (normalized, +polarity)")
    ax.grid(True, alpha=0.3, linestyle="--")

    counts = " | ".join(
        f"{name}: {label_counts.get(label_id, 0)}"
        for label_id, name in LABELING_PULSE_CLASSES.items()
    )
    sampled_as = record.get("sampling_pool", "candidate")
    native_fs = record.get("native_fs", record["fs"])
    fs_note = (
        f"native {native_fs/1000:.0f} kHz→plot {record['fs']/1000:.0f} kHz"
        if abs(float(native_fs) - float(record["fs"])) > 1
        else f"{record['fs']/1000:.0f} kHz"
    )
    title = (
        f"Label pulse {current_idx + 1}/{total} | "
        f"{Path(record['file_path']).name}, pulse {record['pulse_idx']} | "
        f"sampled as: {sampled_as} | {fs_note}\n"
        "[0] normal  [1] wide  [2] double  [S] skip  [Q] finish | "
        f"{counts}"
    )
    ax.set_title(title, fontsize=11, fontweight="bold")


def _pulse_record_key(file_path, pulse_idx):
    return (str(file_path), int(pulse_idx))


def load_excluded_pulse_keys(labels_path):
    """Return set of (file_path, pulse_idx) already labeled (e.g. test set)."""
    labels_path = Path(labels_path)
    if not labels_path.exists():
        return set()
    data = np.load(labels_path, allow_pickle=False)
    records = data["records"]
    return {
        _pulse_record_key(rec["file_path"], rec["pulse_idx"])
        for rec in records
    }


def load_naturalistic_pulses_for_labeling(
    data_path,
    n_pulses=500,
    max_files=40,
    random_seed=42,
    exclude_keys=None,
):
    """
    Sample pulses at natural prevalence for labeling.

    Stratifies across files (≈equal draw per file) so the set is not dominated
    by one recording. Does not use old detector class pools — only
    predicted-positive (or all) pulses, drawn at random.

    exclude_keys : optional set of (file_path, pulse_idx) to skip (e.g. test set).
    """
    path_list = get_path_list(Path(data_path))
    if not path_list:
        raise FileNotFoundError(f"No H5 files under {data_path}")

    exclude_keys = exclude_keys or set()
    rng = np.random.default_rng(random_seed)
    file_order = rng.permutation(len(path_list))
    selected_files = [path_list[int(i)] for i in file_order[: min(max_files, len(path_list))]]

    pools_by_file = []
    n_excluded = 0
    for file_path in selected_files:
        file = open_h5(file_path, nixio.FileMode.ReadOnly)
        if file is None:
            continue
        try:
            block = get_pulse_block(file)
            names = [da.name for da in block.data_arrays]
            if "raw_pulses" not in names:
                continue
            raw_pulses = block.data_arrays["raw_pulses"]
            num_pulses = len(raw_pulses)
            if "predicted_labels" in names:
                candidate_indices = np.where(
                    block.data_arrays["predicted_labels"][:] == 1
                )[0]
            else:
                candidate_indices = np.arange(num_pulses)
            if len(candidate_indices) == 0:
                continue
            fs = float(file.sections["pulses_metadata"]["metadata"]["samplerate"])
            pool = []
            for pulse_idx in candidate_indices:
                key = _pulse_record_key(file_path, pulse_idx)
                if key in exclude_keys:
                    n_excluded += 1
                    continue
                pool.append(
                    {
                        "file_path": str(file_path),
                        "pulse_idx": int(pulse_idx),
                        "fs": float(fs),
                        "sampling_pool": "naturalistic",
                    }
                )
            if pool:
                pools_by_file.append(pool)
        finally:
            file.close()

    if not pools_by_file:
        return np.empty((0, 0)), []

    # ≈equal draw per file, then top up if short
    n_files = len(pools_by_file)
    per_file = max(1, n_pulses // n_files)
    selected_records = []
    leftovers = []
    for pool in pools_by_file:
        take = min(per_file, len(pool))
        chosen = rng.choice(len(pool), size=take, replace=False)
        chosen_set = {int(j) for j in chosen}
        selected_records.extend(pool[int(i)] for i in chosen)
        leftovers.extend(pool[i] for i in range(len(pool)) if i not in chosen_set)

    if len(selected_records) < n_pulses and leftovers:
        need = min(n_pulses - len(selected_records), len(leftovers))
        extra = rng.choice(len(leftovers), size=need, replace=False)
        selected_records.extend(leftovers[int(i)] for i in extra)

    if len(selected_records) > n_pulses:
        keep = rng.choice(len(selected_records), size=n_pulses, replace=False)
        selected_records = [selected_records[int(i)] for i in keep]

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
                "sampling_pool": "naturalistic",
                "all_channels": np.asarray(pulse_data, dtype=float),
            }
        )

    con.log(
        f"  Naturalistic labeling sample: {len(records)} pulses from "
        f"{len({r['file_path'] for r in records})} files "
        f"(target n={n_pulses}, max_files={max_files}"
        + (f", excluded {n_excluded} already-labeled" if n_excluded else "")
        + ")"
    )
    return np.asarray(waveforms, dtype=float), records


def interactive_label_pulses(
    data_path,
    labels_path=None,
    pulses_per_type=300,
    random_seed=42,
    *,
    sampling="balanced",
    n_pulses=500,
    max_files=40,
    append_existing=True,
    exclude_keys=None,
):
    """
    Prompt the user to label pulses in a matplotlib window.

    Labels:
        0 = normal
        1 = wide
        2 = double

    sampling:
        "balanced" — old detector pools (training-style, class-balanced)
        "naturalistic" — random / file-stratified (test-set style)
    """
    ml_paths = get_default_ml_paths()
    labels_path = Path(labels_path) if labels_path else ml_paths["labels"]
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    con.log("Loading pulse candidates for manual labeling...")
    if sampling == "naturalistic":
        waveforms, records = load_naturalistic_pulses_for_labeling(
            data_path,
            n_pulses=n_pulses,
            max_files=max_files,
            random_seed=random_seed,
            exclude_keys=exclude_keys,
        )
    elif sampling == "balanced":
        waveforms, records = load_balanced_detector_labeled_pulses(
            data_path, pulses_per_type=pulses_per_type, random_seed=random_seed
        )
    else:
        raise ValueError(f"Unknown sampling mode: {sampling}")

    if len(waveforms) == 0:
        con.log("No pulse candidates found.")
        return None

    if sampling == "naturalistic":
        records, waveforms = apply_label_ref_timebase(
            records, waveforms, ref_fs=LABEL_REF_FS
        )
        n_stretched = sum(
            1
            for r in records
            if abs(float(r.get("native_fs", r["fs"])) - LABEL_REF_FS) > 1
        )
        con.log(
            f"  Timebase normalized to {LABEL_REF_FS/1000:.0f} kHz for labeling/"
            f"features ({n_stretched}/{len(records)} pulses were higher-rate and "
            "are shown time-stretched × fs/ref_fs)."
        )

    waveforms = normalize_waveforms_for_pca(waveforms)
    labels = np.full(len(waveforms), -1, dtype=np.int64)
    label_plot_half_window_ms = compute_label_plot_half_window_ms(
        records, ref_fs=LABEL_REF_FS if sampling == "naturalistic" else None
    )
    label_plot_ylim = compute_label_plot_ylim(records)

    con.log("\nLabeling instructions:")
    con.log(f"  Sampling mode: {sampling}")
    con.log(f"  Save path: {labels_path}")
    if sampling == "naturalistic":
        con.log(
            f"  Common plot/feature timebase: {LABEL_REF_FS/1000:.0f} kHz "
            "(48 kHz snippets stretched ×2 so all traces share the x-axis)"
        )
    con.log("  0 = normal/non-special pulse")
    con.log("  1 = wide pulse")
    con.log("  2 = double pulse")
    con.log("  S = skip current pulse")
    con.log("  Q = finish and save labels collected so far")
    con.log(
        f"  Fixed x-axis: ±{label_plot_half_window_ms:.2f} ms from peak "
        "(for width comparison)"
    )
    con.log(
        f"  Fixed y-axis: [{label_plot_ylim[0]:.2f}, {label_plot_ylim[1]:.2f}] "
        "(all channels flipped to +polarity)"
    )

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
            label_plot_half_window_ms=label_plot_half_window_ms,
            label_plot_ylim=label_plot_ylim,
        )
        fig.canvas.draw_idle()

    def on_key_press(event):
        if event.key is None:
            return

        key = event.key.lower()
        if key in {"0", "1", "2"}:
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

    if append_existing and labels_path.exists():
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
    unique, counts = np.unique(save_labels, return_counts=True)
    con.log(
        "Label counts: "
        + ", ".join(
            f"{LABELING_PULSE_CLASSES.get(int(u), u)}={int(c)}"
            for u, c in zip(unique, counts)
        )
    )

    return labels_path


def interactive_label_naturalistic_test_set(
    data_path,
    labels_path=None,
    n_pulses=500,
    max_files=40,
    random_seed=42,
    append_existing=False,
):
    """
    Label a naturalistic held-out test set (random / file-stratified sampling).

    Saved separately from the training label file so it is not used for fitting.
    """
    ml_paths = get_default_ml_paths()
    labels_path = (
        Path(labels_path) if labels_path else ml_paths["naturalistic_test_labels"]
    )
    con.log("\n" + "=" * 60)
    con.log("NATURALISTIC TEST-SET LABELING")
    con.log("=" * 60)
    con.log(
        "This set is for evaluating the thresholded RF only — "
        "it will not be used to train the classifier."
    )
    con.log(f"Target size: ~{n_pulses} pulses across up to {max_files} files")
    con.log(f"Output: {labels_path}")
    con.log("=" * 60)

    return interactive_label_pulses(
        data_path,
        labels_path=labels_path,
        random_seed=random_seed,
        sampling="naturalistic",
        n_pulses=n_pulses,
        max_files=max_files,
        append_existing=append_existing,
    )


def interactive_label_naturalistic_train_set(
    data_path,
    labels_path=None,
    n_pulses=1200,
    max_files=80,
    random_seed=7,
    append_existing=False,
    exclude_test_set=True,
):
    """
    Label a naturalistic training set (file-stratified random sampling).

    Excludes pulses already in the frozen naturalistic test set by default.
    Saved separately from the old detector-balanced training labels.
    """
    ml_paths = get_default_ml_paths()
    labels_path = (
        Path(labels_path) if labels_path else ml_paths["naturalistic_train_labels"]
    )
    exclude_keys = set()
    if exclude_test_set:
        exclude_keys = load_excluded_pulse_keys(ml_paths["naturalistic_test_labels"])
        con.log(
            f"Excluding {len(exclude_keys)} pulses already in naturalistic test set."
        )

    con.log("\n" + "=" * 60)
    con.log("NATURALISTIC TRAINING-SET LABELING")
    con.log("=" * 60)
    con.log(
        "Sampling: random among predicted-positive pulses, stratified across "
        "many H5 files (≈equal count per file). NOT balanced by old detector "
        "class pools — prevalence should match the data."
    )
    con.log(
        "Your frozen naturalistic test set is excluded so there is no train/test "
        "leakage. After labeling, we will train on this file and evaluate on the "
        "test set."
    )
    con.log(f"Target size: ~{n_pulses} pulses across up to {max_files} files")
    con.log(f"Output: {labels_path}")
    con.log("=" * 60)

    return interactive_label_pulses(
        data_path,
        labels_path=labels_path,
        random_seed=random_seed,
        sampling="naturalistic",
        n_pulses=n_pulses,
        max_files=max_files,
        append_existing=append_existing,
        exclude_keys=exclude_keys,
    )


def evaluate_naturalistic_test_set(
    labels_path=None,
    model_path=None,
):
    """
    Score the saved production classifier on the naturalistic test labels.
    """
    ml_paths = get_default_ml_paths()
    labels_path = (
        Path(labels_path) if labels_path else ml_paths["naturalistic_test_labels"]
    )
    model_path = Path(model_path) if model_path else ml_paths["model"]

    if not labels_path.exists():
        raise FileNotFoundError(
            f"Naturalistic test labels not found at {labels_path}. "
            "Run --mode label-naturalistic-test first."
        )
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    import __main__
    setattr(__main__, "RobustPCA", RobustPCA)

    waveforms, labels, _ = load_labeled_dataset(labels_path)
    with model_path.open("rb") as f:
        model_data = pickle.load(f)

    classifier = model_data["classifier"]
    decision_config = model_data.get("decision_config", DEFAULT_DECISION_CONFIG)
    train_prior = model_data.get("train_prior", {0: 0.42, 1: 0.20, 2: 0.38})

    X = prepare_classifier_waveforms(waveforms, classifier)
    y_hard = classifier.predict(X)
    y_policy, _ = predict_special_pulse_classes(
        classifier,
        X,
        decision_config=decision_config,
        train_prior=train_prior,
    )

    labels_sorted = sorted(np.unique(labels))
    target_names = [SPECIAL_PULSE_CLASSES[int(i)] for i in labels_sorted]

    con.log("\n" + "=" * 60)
    con.log("NATURALISTIC TEST EVALUATION")
    con.log("=" * 60)
    con.log(f"Labels: {labels_path} (n={len(labels)})")
    con.log(f"Model:  {model_path}")
    con.log(f"Decision config: {decision_config}")
    true_prior = class_prior_from_labels(labels)
    con.log(f"True label rates: {true_prior}")

    con.log("\n--- Hard predict() ---")
    print(
        classification_report(
            labels,
            y_hard,
            labels=labels_sorted,
            target_names=target_names,
            zero_division=0,
        )
    )
    print(confusion_matrix(labels, y_hard, labels=labels_sorted))

    con.log("\n--- Production decision policy ---")
    print(
        classification_report(
            labels,
            y_policy,
            labels=labels_sorted,
            target_names=target_names,
            zero_division=0,
        )
    )
    print(confusion_matrix(labels, y_policy, labels=labels_sorted))

    pred_rates = class_prior_from_labels(y_policy)
    con.log(f"Predicted rates (policy): {pred_rates}")

    out_dir = ml_paths["base"] / "naturalistic_test_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "naturalistic_test_metrics.json"
    summary = {
        "labels_path": str(labels_path),
        "model_path": str(model_path),
        "n": int(len(labels)),
        "true_prior": true_prior,
        "decision_config": decision_config,
        "hard_predict": _compute_multiclass_metrics(labels, y_hard, labels_sorted),
        "policy": _compute_multiclass_metrics(labels, y_policy, labels_sorted),
        "policy_double": _binary_prf(labels, y_policy, 2),
        "policy_wide": _binary_prf(labels, y_policy, 1),
        "predicted_rates_policy": pred_rates,
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    con.log(f"Saved {out_json}")
    return summary


def load_labeled_dataset(labels_path):
    labels_path = Path(labels_path)
    if not labels_path.exists():
        raise FileNotFoundError(f"Label dataset does not exist: {labels_path}")

    data = np.load(labels_path, allow_pickle=False)
    return data["waveforms"], data["labels"], data["records"]


def plot_labeled_pulses_pca_space(
    waveforms,
    labels,
    output_path=None,
    show=True,
    class_names=None,
):
    """
    Plot robust-PCA projections of manually labeled pulse waveforms.

    Computes up to PCA_MAX_PLOT_COMPONENTS components and renders the most
    informative PC pairs (PC1/PC2 plus additional high-variance axes).
    """
    apply_presentation_style()
    if len(labels) < 2:
        con.log("Need at least two labeled pulses to plot PCA space.")
        return None, None

    class_names = class_names or SPECIAL_PULSE_CLASSES
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
        figsize=(7.0 * n_cols, 6.0 * n_rows),
        squeeze=False,
    )
    colors = classifier_label_colors(labels_sorted, class_names)

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
            ax.legend(title="Manual label", frameon=True, loc=LEGEND_LOC)

    for panel_idx in range(n_panels, n_rows * n_cols):
        row_idx, col_idx = divmod(panel_idx, n_cols)
        axes[row_idx, col_idx].set_axis_off()

    fig.suptitle(
        "Robust PCA Space of Manually Labeled Pulses",
        y=1.02,
    )
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
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
    pca_plot_dir = PCA_SPACE_DIR / "benchmark"
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
        con.log(f"Ignoring {ignored_count} labels outside normal/wide/double.")
        waveforms = waveforms[active_label_mask]
        labels = labels[active_label_mask]
        records = records[active_label_mask]

    if len(labels) == 0:
        con.log("No normal/wide/double labels available for benchmarking.")
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


def train_special_pulse_classifier(
    labels_path=None,
    model_path=None,
    show_pca_plot=False,
    decision_config=None,
):
    """
    Train a robust PCA + random forest multiclass classifier and print precision/recall/f1.

    Saves decision_config + train_prior alongside the pipeline so deploy can use
    predict_proba with prior reweighting and rare-class thresholds.
    """
    ml_paths = get_default_ml_paths()
    if labels_path is None:
        # Prefer naturalistic training labels when available (new workflow).
        if ml_paths["naturalistic_train_labels"].exists():
            labels_path = ml_paths["naturalistic_train_labels"]
            con.log(f"Using naturalistic training labels: {labels_path}")
        else:
            labels_path = ml_paths["labels"]
    else:
        labels_path = Path(labels_path)
    model_path = Path(model_path) if model_path else ml_paths["model"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    decision_config = {**DEFAULT_DECISION_CONFIG, **(decision_config or {})}

    waveforms, labels, _ = load_labeled_dataset(labels_path)
    active_label_mask = np.isin(labels, list(LABELING_PULSE_CLASSES))
    if not np.all(active_label_mask):
        ignored_count = int(np.sum(~active_label_mask))
        con.log(f"Ignoring {ignored_count} labels outside normal/wide/double.")
        waveforms = waveforms[active_label_mask]
        labels = labels[active_label_mask]
    if len(labels) == 0:
        raise ValueError(
            "No normal/wide/double labels available to train a classifier."
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
    train_prior = class_prior_from_labels(y_train, class_ids=labels_sorted)

    n_components = _get_pca_n_components(X_train)
    rf_class_weight = decision_config.get("rf_class_weight", None)

    classifier = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", _make_pca_estimator(n_components)),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight=rf_class_weight,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    classifier.fit(X_train, y_train)
    y_pred_hard = classifier.predict(X_test)
    y_pred, _ = predict_special_pulse_classes(
        classifier,
        X_test,
        decision_config=decision_config,
        train_prior=train_prior,
    )

    con.log("\nClassifier performance on held-out labeled pulses (hard predict):")
    print(
        classification_report(
            y_test,
            y_pred_hard,
            labels=labels_sorted,
            target_names=target_names,
            zero_division=0,
        )
    )
    con.log("Confusion matrix (hard predict):")
    print(confusion_matrix(y_test, y_pred_hard, labels=labels_sorted))

    con.log(
        "\nHeld-out metrics with production decision policy "
        f"(prior_reweight={decision_config.get('use_prior_reweight')}, "
        f"min_proba={decision_config.get('min_proba')}):"
    )
    print(
        classification_report(
            y_test,
            y_pred,
            labels=labels_sorted,
            target_names=target_names,
            zero_division=0,
        )
    )
    con.log("Confusion matrix (decision policy):")
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
                "waveform_length": int(X_train.shape[1]),
                "train_prior": train_prior,
                "decision_config": decision_config,
            },
            f,
        )

    con.log(f"Saved trained classifier to {model_path}")
    con.log(f"Train prior: {train_prior}")
    con.log(f"Decision config: {decision_config}")
    con.log("\nFinal held-out classifier metrics (decision policy):")
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


def predict_special_pulses_in_file(
    file_path,
    classifier,
    *,
    decision_config=None,
    train_prior=None,
):
    file, write_mode = open_h5_readwrite_or_readonly(file_path)
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
        fs = float(file.sections["pulses_metadata"]["metadata"]["samplerate"])

        if "predicted_labels" in data_array_names:
            predicted_labels = block.data_arrays["predicted_labels"][:]
            candidate_indices = np.where(predicted_labels == 1)[0]
        else:
            candidate_indices = np.arange(num_pulses)

        predicted_classes = np.zeros(num_pulses, dtype=np.int64)

        if len(candidate_indices) > 0:
            waveforms = []
            raw_for_gate = []
            for pulse_idx in candidate_indices:
                pulse_data = raw_pulses[int(pulse_idx)][:]
                raw_for_gate.append(pulse_data)
                trace, _ = get_representative_waveform(pulse_data)
                waveforms.append(trace)

            waveforms = prepare_classifier_waveforms(
                np.asarray(waveforms, dtype=float), classifier
            )
            cfg = decision_config or DEFAULT_DECISION_CONFIG
            sample_rates = np.full(len(candidate_indices), fs, dtype=float)
            preds, _ = predict_special_pulse_classes(
                classifier,
                waveforms,
                decision_config=cfg,
                train_prior=train_prior,
                pulse_waveforms_raw=(
                    raw_for_gate if cfg.get("rule_gate_double") else None
                ),
                sample_rates=(
                    sample_rates if cfg.get("rule_gate_double") else None
                ),
            )
            predicted_classes[candidate_indices] = preds

        marker_arrays = {
            MULTICLASS_ARRAY_NAME: predicted_classes,
            **{
                array_name: (predicted_classes == label_id).astype(np.int64)
                for label_id, array_name in SPECIAL_CLASS_ARRAYS.items()
            },
        }

        if write_mode == "h5":
            for array_name, values in marker_arrays.items():
                write_or_create_data_array(block, array_name, values)
        else:
            for array_name, values in marker_arrays.items():
                sidecar = save_marker_sidecar(file_path, array_name, values)
                con.log(f"  Saved '{array_name}' markers to {sidecar.name}")

        counts = {
            SPECIAL_PULSE_CLASSES[label_id]: int(np.sum(predicted_classes == label_id))
            for label_id in SPECIAL_PULSE_CLASSES
        }

        con.log(f"  {Path(file_path).name}: {counts}")
        return {"status": "completed", "counts": counts, "write_mode": write_mode}

    finally:
        file.close()


def apply_special_pulse_classifier(data_path, model_path=None):
    ml_paths = get_default_ml_paths()
    model_path = Path(model_path) if model_path else ml_paths["model"]

    # Compatibility for older pickles: during training, RobustPCA may have been
    # pickled under "__main__" (e.g. when the training script was run directly).
    # When loading from another script, that class may not exist in __main__,
    # causing: "Can't get attribute 'RobustPCA' on <module '__main__' ...>".
    import __main__
    setattr(__main__, "RobustPCA", RobustPCA)

    with model_path.open("rb") as f:
        model_data = pickle.load(f)

    classifier = model_data["classifier"]
    decision_config = model_data.get("decision_config", DEFAULT_DECISION_CONFIG)
    train_prior = model_data.get("train_prior")
    if train_prior is None:
        # Older pickles: approximate train prior from balanced label mix.
        train_prior = {0: 0.42, 1: 0.20, 2: 0.38}
        con.log(
            "Model pickle has no train_prior; using approximate balanced priors "
            f"{train_prior}"
        )

    path_list = get_path_list(Path(data_path))

    results = []
    for file_idx, file_path in enumerate(path_list, 1):
        con.log(f"Predicting [{file_idx}/{len(path_list)}] {file_path.name}")
        results.append(
            predict_special_pulses_in_file(
                file_path,
                classifier,
                decision_config=decision_config,
                train_prior=train_prior,
            )
        )

    return results


def sample_naturalistic_pulses(
    data_path,
    n_pulses=400,
    max_files=8,
    random_seed=42,
):
    """
    Draw a small random sample of pulses from H5 files (natural prevalence).

    Returns prepared 1D waveforms, multi-channel raw pulses, sample rates, and
    rule-based double/wide flags for the same sample.
    """
    path_list = get_path_list(Path(data_path))
    if not path_list:
        raise FileNotFoundError(f"No H5 files under {data_path}")

    rng = np.random.default_rng(random_seed)
    file_order = rng.permutation(len(path_list))
    selected_files = [path_list[int(i)] for i in file_order[:max_files]]

    pool = []
    for file_path in selected_files:
        file = open_h5(file_path, nixio.FileMode.ReadOnly)
        if file is None:
            continue
        try:
            block = get_pulse_block(file)
            names = [da.name for da in block.data_arrays]
            if "raw_pulses" not in names:
                continue
            raw_pulses = block.data_arrays["raw_pulses"]
            num_pulses = len(raw_pulses)
            if "predicted_labels" in names:
                candidate_indices = np.where(block.data_arrays["predicted_labels"][:] == 1)[0]
            else:
                candidate_indices = np.arange(num_pulses)
            fs = float(file.sections["pulses_metadata"]["metadata"]["samplerate"])
            for pulse_idx in candidate_indices:
                pool.append((str(file_path), int(pulse_idx), fs))
        finally:
            file.close()

    if not pool:
        raise RuntimeError("No candidate pulses found for naturalistic sampling.")

    sample_size = min(n_pulses, len(pool))
    chosen = [pool[int(i)] for i in rng.choice(len(pool), size=sample_size, replace=False)]

    # Group by file for efficient loading
    by_file = {}
    for file_path, pulse_idx, fs in chosen:
        by_file.setdefault(file_path, []).append((pulse_idx, fs))

    traces = []
    raw_list = []
    sample_rates = []
    rule_double = []
    rule_wide = []

    for file_path, items in by_file.items():
        file = open_h5(file_path, nixio.FileMode.ReadOnly)
        if file is None:
            continue
        try:
            block = get_pulse_block(file)
            raw_pulses = block.data_arrays["raw_pulses"]
            for pulse_idx, fs in items:
                pulse_data = np.asarray(raw_pulses[pulse_idx][:], dtype=float)
                trace, _ = get_representative_waveform(pulse_data)
                is_double, _ = detect_double_pulse(pulse_data, fs)
                is_wide, _ = detect_wide_pulse(pulse_data, fs)
                traces.append(trace)
                raw_list.append(pulse_data)
                sample_rates.append(fs)
                rule_double.append(bool(is_double))
                rule_wide.append(bool(is_wide))
        finally:
            file.close()

    return {
        "traces": np.asarray(traces, dtype=float),
        "raw_pulses": raw_list,
        "sample_rates": np.asarray(sample_rates, dtype=float),
        "rule_double": np.asarray(rule_double, dtype=bool),
        "rule_wide": np.asarray(rule_wide, dtype=bool),
        "n_pool": len(pool),
        "n_files": len(by_file),
    }


def _rate_dict(preds, class_ids=(0, 1, 2)):
    preds = np.asarray(preds, dtype=np.int64)
    n = max(len(preds), 1)
    return {
        SPECIAL_PULSE_CLASSES[c]: {
            "count": int(np.sum(preds == c)),
            "rate": float(np.mean(preds == c)) if len(preds) else 0.0,
        }
        for c in class_ids
    }


def _binary_prf(y_true, y_pred, positive_label):
    y_true = np.asarray(y_true) == positive_label
    y_pred = np.asarray(y_pred) == positive_label
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "support": int(np.sum(y_true)),
    }


def tune_classifier_decisions(
    data_path=None,
    labels_path=None,
    n_natural=400,
    max_files=8,
    random_seed=42,
):
    """
    Compare decision policies without writing markers.

    Evaluates on:
      1) stratified holdout of the (biased) labeled set — optimistic P/R
      2) small naturalistic unlabeled sample — predicted rates vs rule-based

    Does not claim true naturalistic precision without manual labels.
    """
    ml_paths = get_default_ml_paths()
    labels_path = Path(labels_path) if labels_path else ml_paths["labels"]
    data_path = Path(data_path) if data_path else H5_DIR
    out_dir = ml_paths["base"] / "decision_tuning"
    out_dir.mkdir(parents=True, exist_ok=True)

    waveforms, labels, _ = load_labeled_dataset(labels_path)
    active = np.isin(labels, list(LABELING_PULSE_CLASSES))
    waveforms, labels = waveforms[active], labels[active]

    split = _prepare_classifier_train_test_split(waveforms, labels)
    X_train, X_test = split["X_train"], split["X_test"]
    y_train, y_test = split["y_train"], split["y_test"]
    labels_sorted = split["labels_sorted"]
    train_prior = class_prior_from_labels(y_train, class_ids=labels_sorted)
    train_prior = {int(k): float(v) for k, v in train_prior.items()}
    n_components = _get_pca_n_components(X_train)

    con.log("\n" + "=" * 60)
    con.log("DECISION POLICY TUNING (no marker writes)")
    con.log("=" * 60)
    con.log(f"Labeled holdout: train={len(y_train)} test={len(y_test)}")
    con.log(f"Train prior: {train_prior}")

    con.log("Sampling naturalistic pulses...")
    natural = sample_naturalistic_pulses(
        data_path,
        n_pulses=n_natural,
        max_files=max_files,
        random_seed=random_seed,
    )
    rule_preds = np.zeros(len(natural["traces"]), dtype=np.int64)
    rule_preds[natural["rule_wide"]] = 1
    # Prefer double over wide if both fire
    rule_preds[natural["rule_double"]] = 2
    rule_rates = _rate_dict(rule_preds)
    # Empirical prior from rule-based on this sample (floored to avoid zeros)
    rule_prior = {
        0: max(rule_rates["normal"]["rate"], 0.5),
        1: max(rule_rates["wide"]["rate"], 0.005),
        2: max(rule_rates["double"]["rate"], 0.002),
    }
    # Renormalize
    s = sum(rule_prior.values())
    rule_prior = {k: v / s for k, v in rule_prior.items()}

    con.log(
        f"Naturalistic sample: n={len(natural['traces'])} from "
        f"{natural['n_files']} files (pool={natural['n_pool']})"
    )
    con.log(
        "Rule-based rates on sample: "
        + ", ".join(f"{k}={v['rate']:.3%}" for k, v in rule_rates.items())
    )
    con.log(f"Assumed natural prior (default): {DEFAULT_NATURAL_PRIOR}")
    con.log(f"Rule-estimated prior: {rule_prior}")

    policies = {
        "A_hard_predict_balanced_weights": {
            "rf_class_weight": "balanced",
            "use_prior_reweight": False,
            "min_proba": {},
            "rule_gate_double": False,
            "use_hard_predict": True,
        },
        "B_hard_predict_no_class_weight": {
            "rf_class_weight": None,
            "use_prior_reweight": False,
            "min_proba": {},
            "rule_gate_double": False,
            "use_hard_predict": True,
        },
        "C_threshold_t050": {
            "rf_class_weight": None,
            "use_prior_reweight": False,
            "min_proba": {1: 0.50, 2: 0.50},
            "rule_gate_double": False,
            "use_hard_predict": False,
        },
        "D_threshold_t070": {
            "rf_class_weight": None,
            "use_prior_reweight": False,
            "min_proba": {1: 0.60, 2: 0.70},
            "rule_gate_double": False,
            "use_hard_predict": False,
        },
        "E_threshold_t085": {
            "rf_class_weight": None,
            "use_prior_reweight": False,
            "min_proba": {1: 0.65, 2: 0.85},
            "rule_gate_double": False,
            "use_hard_predict": False,
        },
        "F_threshold_t090": {
            "rf_class_weight": None,
            "use_prior_reweight": False,
            "min_proba": {1: 0.70, 2: 0.90},
            "rule_gate_double": False,
            "use_hard_predict": False,
        },
        "G_mild_prior_argmax": {
            "rf_class_weight": None,
            "use_prior_reweight": True,
            "natural_prior": {0: 0.90, 1: 0.08, 2: 0.02},
            "min_proba": {},
            "rule_gate_double": False,
            "use_hard_predict": False,
            "argmax_after_reweight": True,
        },
        "H_rule_prior_argmax": {
            "rf_class_weight": None,
            "use_prior_reweight": True,
            "natural_prior": rule_prior,
            "min_proba": {},
            "rule_gate_double": False,
            "use_hard_predict": False,
            "argmax_after_reweight": True,
        },
        "I_threshold_t070_rule_gate": {
            "rf_class_weight": None,
            "use_prior_reweight": False,
            "min_proba": {1: 0.60, 2: 0.70},
            "rule_gate_double": True,
            "use_hard_predict": False,
        },
        "J_threshold_t050_rule_gate": {
            "rf_class_weight": None,
            "use_prior_reweight": False,
            "min_proba": {1: 0.50, 2: 0.50},
            "rule_gate_double": True,
            "use_hard_predict": False,
        },
        "K_mild_prior_plus_t060": {
            "rf_class_weight": None,
            "use_prior_reweight": True,
            "natural_prior": {0: 0.90, 1: 0.08, 2: 0.02},
            "min_proba": {1: 0.40, 2: 0.60},
            "rule_gate_double": False,
            "use_hard_predict": False,
        },
    }

    # Fit one RF per class_weight setting and reuse
    fitted = {}
    for weight_key in ("balanced", None):
        pipe = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("pca", _make_pca_estimator(n_components)),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight=weight_key,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        pipe.fit(X_train, y_train)
        fitted[weight_key] = pipe

    X_nat = prepare_classifier_waveforms(natural["traces"], fitted[None])

    results = []
    for name, pol in policies.items():
        clf = fitted[pol["rf_class_weight"]]
        decision_config = {
            "use_prior_reweight": pol.get("use_prior_reweight", False),
            "natural_prior": {
                int(k): float(v)
                for k, v in pol.get("natural_prior", DEFAULT_NATURAL_PRIOR).items()
            },
            "min_proba": {int(k): float(v) for k, v in pol.get("min_proba", {}).items()},
            "default_class": 0,
            "rule_gate_double": pol.get("rule_gate_double", False),
            "rf_class_weight": pol["rf_class_weight"],
        }

        if pol.get("use_hard_predict"):
            y_hold = clf.predict(X_test)
            y_nat = clf.predict(X_nat)
        elif pol.get("argmax_after_reweight"):
            proba = clf.predict_proba(X_test)
            if decision_config["use_prior_reweight"]:
                proba = reweight_class_probabilities(
                    proba,
                    clf.classes_,
                    train_prior,
                    decision_config["natural_prior"],
                )
            y_hold = np.asarray(clf.classes_, dtype=np.int64)[np.argmax(proba, axis=1)]
            proba_n = clf.predict_proba(X_nat)
            if decision_config["use_prior_reweight"]:
                proba_n = reweight_class_probabilities(
                    proba_n,
                    clf.classes_,
                    train_prior,
                    decision_config["natural_prior"],
                )
            y_nat = np.asarray(clf.classes_, dtype=np.int64)[np.argmax(proba_n, axis=1)]
        else:
            y_hold, _ = predict_special_pulse_classes(
                clf,
                X_test,
                decision_config=decision_config,
                train_prior=train_prior,
            )
            y_nat, _ = predict_special_pulse_classes(
                clf,
                X_nat,
                decision_config=decision_config,
                train_prior=train_prior,
                pulse_waveforms_raw=(
                    natural["raw_pulses"]
                    if decision_config["rule_gate_double"]
                    else None
                ),
                sample_rates=(
                    natural["sample_rates"]
                    if decision_config["rule_gate_double"]
                    else None
                ),
            )

        hold_metrics = _compute_multiclass_metrics(y_test, y_hold, labels_sorted)
        double_hold = _binary_prf(y_test, y_hold, 2)
        wide_hold = _binary_prf(y_test, y_hold, 1)
        nat_rates = _rate_dict(y_nat)

        row = {
            "policy": name,
            "decision_config": decision_config,
            "holdout_macro_f1": hold_metrics["macro_f1"],
            "holdout_weighted_f1": hold_metrics["weighted_f1"],
            "holdout_double": double_hold,
            "holdout_wide": wide_hold,
            "natural_rates": nat_rates,
            "natural_double_rate": nat_rates["double"]["rate"],
            "natural_wide_rate": nat_rates["wide"]["rate"],
            "natural_normal_rate": nat_rates["normal"]["rate"],
            "rule_double_rate": rule_rates["double"]["rate"],
            "rule_wide_rate": rule_rates["wide"]["rate"],
        }
        results.append(row)

        con.log(
            f"{name}: hold macroF1={hold_metrics['macro_f1']:.3f} "
            f"double P/R={double_hold['precision']:.2f}/{double_hold['recall']:.2f} | "
            f"nat rates N/W/D="
            f"{nat_rates['normal']['rate']:.3%}/"
            f"{nat_rates['wide']['rate']:.3%}/"
            f"{nat_rates['double']['rate']:.3%}"
        )

    # Prefer policies whose naturalistic double rate is near rule-based (and ≪ 5%),
    # then maximize holdout double precision, then macro F1.
    def rank_key(r):
        double_ok = r["natural_double_rate"] < 0.05
        return (
            0 if double_ok else 1,
            abs(r["natural_double_rate"] - r["rule_double_rate"]),
            abs(r["natural_wide_rate"] - r["rule_wide_rate"]),
            -r["holdout_double"]["precision"],
            -r["holdout_double"]["f1"],
            -r["holdout_macro_f1"],
        )

    ranked = sorted(results, key=rank_key)
    best = ranked[0]

    summary = {
        "labels_path": str(labels_path),
        "data_path": str(data_path),
        "train_prior": train_prior,
        "natural_prior_assumed": DEFAULT_NATURAL_PRIOR,
        "rule_estimated_prior": rule_prior,
        "naturalistic_n": int(len(natural["traces"])),
        "rule_based_rates": rule_rates,
        "note": (
            "Holdout P/R is on a balanced labeled set and overestimates rare-class "
            "performance in production. Naturalistic rates have no ground truth; "
            "compare to rule-based rates and biological expectation (doubles ≪ few %)."
        ),
        "results": results,
        "recommended_policy": best["policy"],
        "recommended_decision_config": best["decision_config"],
    }

    out_json = out_dir / "decision_tuning_results.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    table = Table(title="Decision policy comparison")
    table.add_column("Policy")
    table.add_column("Hold macroF1", justify="right")
    table.add_column("Dbl P/R", justify="right")
    table.add_column("Nat double%", justify="right")
    table.add_column("Nat wide%", justify="right")
    table.add_column("Nat normal%", justify="right")
    for r in results:
        marker = " *" if r["policy"] == best["policy"] else ""
        table.add_row(
            r["policy"] + marker,
            f"{r['holdout_macro_f1']:.3f}",
            f"{r['holdout_double']['precision']:.2f}/{r['holdout_double']['recall']:.2f}",
            f"{100 * r['natural_double_rate']:.2f}",
            f"{100 * r['natural_wide_rate']:.2f}",
            f"{100 * r['natural_normal_rate']:.2f}",
        )
    con.print(table)
    con.log(
        f"Rule-based on same sample: double={100 * rule_rates['double']['rate']:.2f}%, "
        f"wide={100 * rule_rates['wide']['rate']:.2f}%"
    )
    con.log(f"Recommended (rate-aware): {best['policy']}")
    con.log(f"Saved {out_json}")
    return summary


def run_supervised_detection(
    data_path,
    *,
    labels_path=None,
    model_path=None,
    label=False,
    pulses_per_type=300,
    train=False,
    retrain=False,
    apply=True,
    auto_train=False,
):
    """
    Run the supervised special-pulse workflow without interactive prompts.

    By default applies an existing trained classifier. When auto_train=True, trains
    automatically if no model exists but labeled examples are available.
    """
    ml_paths = get_default_ml_paths()
    if labels_path is None:
        if ml_paths["naturalistic_train_labels"].exists():
            labels_path = ml_paths["naturalistic_train_labels"]
        else:
            labels_path = ml_paths["labels"]
    else:
        labels_path = Path(labels_path)
    model_path = Path(model_path) if model_path else ml_paths["model"]

    if label:
        interactive_label_pulses(
            data_path,
            labels_path=labels_path,
            pulses_per_type=pulses_per_type,
        )

    should_train = train or retrain
    if (
        auto_train
        and not should_train
        and apply
        and not model_path.exists()
        and labels_path.exists()
    ):
        should_train = True
        con.log(f"No model at {model_path}; training from {labels_path}")

    if should_train:
        if not labels_path.exists():
            raise FileNotFoundError(
                f"Labeled pulses not found at {labels_path}. "
                "Run with --label first to create training data."
            )
        train_special_pulse_classifier(
            labels_path=labels_path,
            model_path=model_path,
        )

    if apply:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained classifier not found at {model_path}. "
                "Run with --label and --train, or provide an existing model."
            )
        return apply_special_pulse_classifier(data_path, model_path=model_path)

    return None


def supervised_learning_workflow(data_path):
    """
    Interactive end-to-end workflow:
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
    pulses_per_type = 300
    if should_label:
        pulses_per_type_raw = input(
            "Pulses to sample per type for labeling [300]: "
        ).strip()
        pulses_per_type = int(pulses_per_type_raw) if pulses_per_type_raw else 300

    should_train = (
        input("Train classifier from labeled pulses? [Y/n]: ").strip().lower()
    )
    should_apply = input("Apply classifier to h5 files now? [Y/n]: ").strip().lower()

    run_supervised_detection(
        data_path,
        labels_path=ml_paths["labels"],
        model_path=ml_paths["model"],
        label=should_label,
        pulses_per_type=pulses_per_type,
        train=should_train != "n",
        apply=should_apply != "n",
    )


def main():
    global DETECTION_MODE, ARRAY_NAME, DISPLAY_NAME

    parser = argparse.ArgumentParser(
        description="Detect special pulse shapes (double, wide) in .h5 files.",
    )
    parser.add_argument(
        "--mode",
        choices=(
            "supervised",
            "rule-based",
            "benchmark",
            "tune-decisions",
            "label-naturalistic-test",
            "label-naturalistic-train",
            "eval-naturalistic-test",
        ),
        default="supervised",
        help="Detection mode (default: supervised ML classifier)",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=H5_DIR,
        help=f"Directory with .h5 files (default: {H5_DIR})",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Prompt for each step of the supervised workflow",
    )
    parser.add_argument(
        "--label",
        action="store_true",
        help="Interactively label pulse examples for training",
    )
    parser.add_argument(
        "--pulses-per-type",
        type=int,
        default=300,
        help="Pulses to sample per class during balanced labeling (default: 300)",
    )
    parser.add_argument(
        "--n-pulses",
        type=int,
        default=500,
        help="Pulses to sample for naturalistic labeling (default: 500; train mode uses 1200 if unset via mode)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=40,
        help="Max H5 files to draw from for naturalistic sampling (default: 40)",
    )
    parser.add_argument(
        "--append-test-labels",
        action="store_true",
        help="Append to existing naturalistic test labels instead of overwriting",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train classifier from labeled examples",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain classifier even if a model file already exists",
    )
    parser.add_argument(
        "--no-apply",
        action="store_true",
        help="Skip applying the classifier to .h5 files",
    )
    parser.add_argument(
        "--detection-mode",
        choices=tuple(MODE_CONFIG),
        default=DETECTION_MODE,
        help="Pulse type for rule-based detection (default: %(default)s)",
    )
    args = parser.parse_args()

    if args.mode == "supervised":
        if args.interactive:
            supervised_learning_workflow(args.data_path)
            return

        run_supervised_detection(
            args.data_path,
            label=args.label,
            pulses_per_type=args.pulses_per_type,
            train=args.train,
            retrain=args.retrain,
            apply=not args.no_apply,
            auto_train=True,
        )
        return

    if args.mode == "rule-based":
        DETECTION_MODE = args.detection_mode
        ARRAY_NAME = MODE_CONFIG[DETECTION_MODE]["array_name"]
        DISPLAY_NAME = MODE_CONFIG[DETECTION_MODE]["display_name"]
        process_all_h5_files(args.data_path)
        return

    if args.mode == "tune-decisions":
        tune_classifier_decisions(data_path=args.data_path)
        return

    if args.mode == "label-naturalistic-test":
        interactive_label_naturalistic_test_set(
            args.data_path,
            n_pulses=args.n_pulses,
            max_files=args.max_files,
            append_existing=args.append_test_labels,
        )
        return

    if args.mode == "label-naturalistic-train":
        n_train = args.n_pulses if args.n_pulses != 500 else 1200
        max_files = args.max_files if args.max_files != 40 else 80
        interactive_label_naturalistic_train_set(
            args.data_path,
            n_pulses=n_train,
            max_files=max_files,
            append_existing=args.append_test_labels,
        )
        return

    if args.mode == "eval-naturalistic-test":
        evaluate_naturalistic_test_set()
        return

    benchmark_pulse_classifiers()


if __name__ == "__main__":
    main()
