##### attempt to filter double peaked peaks #####

# wave show function in lib rosa (plots hist of audio data)
# let this run for whole dataset and get info of in which files and times double peaks are -> make histogram

# This script detects special pulse shapes in h5 files.
# Double peaks are identified by analyzing the peak structure of each pulse waveform.
# The script adds an "is_double_peak", "is_wide_pulse", or "is_fat_pulse" data array to each h5 file containing binary labels (0 or 1).


from pathlib import Path
import pickle
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import numpy as np
import nixio
import matplotlib.pyplot as plt
from rich.console import Console
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_paths import H5_DIR, SPECIAL_PULSE_CLASSIFIER_DIR

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


#################################
############# ANALYSIS ##########
#################################


# old version from claude
def analyze_pulse_for_double_peak(
    pulse_waveform,
    sample_rate,
    amplitude_threshold=0.7,
    max_peak_distance=0.002,
    min_valley_ratio=0.5,
    local_max_window=0.004,
):
    """
    Analyze a single pulse waveform to determine if it's a double peak.

    Parameters:
    -----------
    pulse_waveform : np.ndarray
        2D array of shape (num_samples, num_channels)
    sample_rate : float
        Sample rate in Hz
    amplitude_threshold : float
        Minimum amplitude for a pulse to be considered (excludes very low amplitude pulses)
    max_peak_distance : float
        Maximum time distance (in seconds) between two peaks to be considered part of the same double peak
    min_valley_ratio : float
        Minimum ratio of valley amplitude to peak amplitude (0 to 1).
        Valley amplitude must be >= min_valley_ratio * peak_max to be considered a valid double peak.
    local_max_window : float
        Time window (in seconds) for checking if a peak is a local maximum. At least one peak must be
        the highest within this window.

    Returns:
    --------
    bool
        True if the pulse is detected as a double peak, False otherwise
    """
    # Compute the mean absolute amplitude across all channels for each time sample
    # This gives us a 1D representation of the pulse strength over time
    pulse_strength = np.mean(np.abs(pulse_waveform), axis=1)

    # Get non-absolute mean for sign checking
    pulse_waveform_mean = np.mean(pulse_waveform, axis=1)

    max_strength = np.max(pulse_strength)

    # Exclude pulses with amplitude below threshold
    if max_strength < amplitude_threshold:
        return False

    # Find peaks with prominence threshold
    # Use 10% of max strength as minimum prominence
    prominence = 0.1 * max_strength
    peaks, peak_properties = find_peaks(pulse_strength, prominence=prominence)

    # A double peak must have EXACTLY 2 peaks
    if len(peaks) != 2:
        return False

    # Convert time thresholds to samples
    distance_threshold_samples = max_peak_distance * sample_rate
    window_samples = int(local_max_window * sample_rate)

    peak1_idx = peaks[0]
    peak2_idx = peaks[1]

    # Check if peaks are within the maximum time distance
    peak_distance_samples = peak2_idx - peak1_idx
    if peak_distance_samples > distance_threshold_samples:
        return False

    # Check that both peaks have the same sign (both above 0 or both below 0)
    peak1_value = pulse_waveform_mean[peak1_idx]
    peak2_value = pulse_waveform_mean[peak2_idx]

    # If peaks have different signs, reject
    if (peak1_value > 0 and peak2_value < 0) or (peak1_value < 0 and peak2_value > 0):
        return False

    # Check if at least one peak is a local maximum within the time window
    def is_local_max_in_window(signal, idx, window_size):
        start = max(0, idx - window_size)
        end = min(len(signal), idx + window_size + 1)
        return signal[idx] == np.max(signal[start:end])

    peak1_is_local_max = is_local_max_in_window(
        pulse_strength, peak1_idx, window_samples
    )
    peak2_is_local_max = is_local_max_in_window(
        pulse_strength, peak2_idx, window_samples
    )

    if not (peak1_is_local_max or peak2_is_local_max):
        return False

    # Find the valley (minimum amplitude) between the two peaks
    valley_amplitude = np.min(pulse_strength[peak1_idx : peak2_idx + 1])
    peak_max = max(pulse_strength[peak1_idx], pulse_strength[peak2_idx])

    # Check if valley is not too deep (amplitude doesn't drop below min_valley_ratio threshold)
    if valley_amplitude >= min_valley_ratio * peak_max:
        return True

    return False


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
    amplitude_threshold=0.7,
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
    amplitude_threshold=0.7,
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
    amplitude_threshold=0.7,
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


def get_path_list(datapath):
    """
    Get a sorted list of all .h5 files in the given directory or subdirectories.
    """
    con.log("Discovering h5 files.")

    if not datapath.exists():
        raise FileNotFoundError(f"Path {datapath} does not exist.")

    path_list = []

    if datapath.is_file():
        if datapath.suffix == ".h5":
            path_list.append(datapath)
    elif datapath.is_dir():
        for file in datapath.rglob("*.h5"):
            path_list.append(file)
    else:
        raise FileNotFoundError(f"Path {datapath} is not valid.")

    con.log(f"Found {len(path_list)} h5 files.")
    return sorted(path_list)


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
    file = nixio.File.open(str(file_path), nixio.FileMode.ReadWrite)

    try:
        # Access pulses block
        block = file.blocks["pulses"]
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

        if DETECTION_MODE in {"wide", "fat"} and "is_double_peak" in data_array_names:
            is_double_peak = expand_marker_to_all_pulses(
                block.data_arrays["is_double_peak"][:],
                candidate_indices,
                num_pulses,
                "is_double_peak",
            )
        else:
            is_double_peak = np.zeros(num_pulses, dtype=np.int64)

        if DETECTION_MODE == "fat" and "is_wide_pulse" in data_array_names:
            is_wide_pulse = expand_marker_to_all_pulses(
                block.data_arrays["is_wide_pulse"][:],
                candidate_indices,
                num_pulses,
                "is_wide_pulse",
            )
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


def load_predicted_positive_pulses(data_path, max_pulses=None, random_seed=42):
    """
    Load predicted-positive pulses and their source locations from h5 files.
    """
    path_list = get_path_list(Path(data_path))
    records = []
    waveforms = []

    for file_idx, file_path in enumerate(path_list, 1):
        con.log(f"  Loading candidates [{file_idx}/{len(path_list)}] {file_path.name}")
        file = nixio.File.open(str(file_path), nixio.FileMode.ReadOnly)

        try:
            block = file.blocks["pulses"]
            data_array_names = [da.name for da in block.data_arrays]

            if "raw_pulses" not in data_array_names:
                con.log("    No raw_pulses array. Skipping.")
                continue

            raw_pulses = block.data_arrays["raw_pulses"]
            if "predicted_labels" in data_array_names:
                predicted_labels = block.data_arrays["predicted_labels"][:]
                pulse_indices = np.where(predicted_labels == 1)[0]
            else:
                pulse_indices = np.arange(len(raw_pulses))

            fs = file.sections["pulses_metadata"]["metadata"]["samplerate"]

            for pulse_idx in pulse_indices:
                pulse_data = raw_pulses[int(pulse_idx)][:]
                trace, best_channel = get_representative_waveform(pulse_data)
                waveforms.append(trace)
                records.append(
                    {
                        "file_path": str(file_path),
                        "pulse_idx": int(pulse_idx),
                        "fs": float(fs),
                        "best_channel": int(best_channel),
                    }
                )

        finally:
            file.close()

    if not waveforms:
        return np.empty((0, 0)), []

    waveforms = np.asarray(waveforms, dtype=float)

    if max_pulses is not None and len(waveforms) > max_pulses:
        rng = np.random.default_rng(random_seed)
        selected = np.sort(rng.choice(len(waveforms), size=max_pulses, replace=False))
        waveforms = waveforms[selected]
        records = [records[i] for i in selected]

    return waveforms, records


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
        file = nixio.File.open(str(file_path), nixio.FileMode.ReadOnly)

        try:
            block = file.blocks["pulses"]
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
            file = nixio.File.open(file_path, nixio.FileMode.ReadOnly)
            try:
                block = file.blocks["pulses"]
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
    Plot the first two PCA components of the manually labeled pulse waveforms.
    """
    if len(labels) < 2:
        con.log("Need at least two labeled pulses to plot PCA space.")
        return None, None

    labels_sorted = sorted(np.unique(labels))
    n_components = min(2, waveforms.shape[0], waveforms.shape[1])
    if n_components < 1:
        con.log("No waveform features available to plot PCA space.")
        return None, None

    pca_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=42)),
        ]
    )
    projected = pca_pipeline.fit_transform(waveforms)

    if n_components == 1:
        projected = np.column_stack([projected[:, 0], np.zeros(len(projected))])

    explained = pca_pipeline.named_steps["pca"].explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(labels_sorted), 1)))

    for color, label_id in zip(colors, labels_sorted):
        mask = labels == label_id
        class_name = SPECIAL_PULSE_CLASSES.get(int(label_id), f"class {label_id}")
        ax.scatter(
            projected[mask, 0],
            projected[mask, 1],
            s=42,
            alpha=0.75,
            edgecolors="black",
            linewidths=0.3,
            color=color,
            label=f"{class_name} (n={int(np.sum(mask))})",
        )

    pc1_var = explained[0] * 100 if len(explained) > 0 else 0
    pc2_var = explained[1] * 100 if len(explained) > 1 else 0
    ax.set_xlabel(f"PC1 ({pc1_var:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pc2_var:.1f}% variance)")
    ax.set_title("PCA Space of Manually Labeled Pulses", fontweight="bold")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(title="Manual label", frameon=True)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        con.log(f"Saved labeled-pulse PCA plot to {output_path}")

    if show:
        plt.show()

    return fig, ax


def train_special_pulse_classifier(labels_path=None, model_path=None):
    """
    Train a PCA + random forest multiclass classifier and print precision/recall/f1.
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
        show=True,
    )

    min_class_count = int(np.min(label_counts))
    num_classes = len(unique_labels)
    enough_for_stratify = min_class_count >= 2 and len(labels) >= 2 * num_classes
    stratify = labels if enough_for_stratify else None

    if stratify is not None:
        test_count = max(num_classes, int(np.ceil(0.25 * len(labels))))
        test_size = test_count / len(labels)
    else:
        test_size = 0.25 if len(labels) >= 8 else 0.5

    X_train, X_test, y_train, y_test = train_test_split(
        waveforms,
        labels,
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )

    n_components = min(20, X_train.shape[0] - 1, X_train.shape[1])
    n_components = max(1, n_components)

    classifier = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=42)),
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

    target_names = [SPECIAL_PULSE_CLASSES[i] for i in sorted(unique_labels)]
    labels_sorted = sorted(unique_labels)

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
    file = nixio.File.open(str(file_path), nixio.FileMode.ReadWrite)

    try:
        block = file.blocks["pulses"]
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
    con.log("=" * 60)

    mode = input("Select mode (1 or 2): ").strip()

    if mode == "1":
        # Process all h5 files to detect double peaks or wide pulses
        results = process_all_h5_files(data_path)
    elif mode == "2":
        supervised_learning_workflow(data_path)
    else:
        con.log("Invalid selection. Please enter 1 or 2.")
