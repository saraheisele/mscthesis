##### attempt to filter double peaked peaks #####

# wave show function in lib rosa (plots hist of audio data)
# let this run for whole dataset and get info of in which files and times double peaks are -> make histogram

# This script detects which pulses in h5 files are "double peaks" (wider pulses with 2 peaks instead of 1).
# Double peaks are identified by analyzing the peak structure of each pulse waveform.
# The script adds an "is_double_peak" or "is_wide_pulse" data array to each h5 file containing binary labels (0 or 1).

# TODO: plotting von double peak detection trennen, double peaks finden verbessern

from pathlib import Path
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import numpy as np
import nixio
from rich.console import Console

# Initialize console for logging
con = Console()

#################################
############# MODE ##############
#################################

DETECTION_MODE = "wide"
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
        except:
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
        except:
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
    max_width_ms=4.0,
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
        Maximum width at half maximum in milliseconds (excludes overly wide pulses)
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
    if width_ms > max_width_ms:
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
    Detect double peaks or wide pulsesin all pulses of a single h5 file and add the results as a data array.

    Parameters:
    -----------
    file_path : Path or str
        Path to the h5 file to process

    Returns:
    --------
    dict
        Statistics about the file: num_pulses, num_double_peaks, num_single_peaks
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

        # Optional exclusion array for wide pulse mode
        if "is_double_peak" in data_array_names:
            is_double_peak = block.data_arrays["is_double_peak"][:]
        else:
            is_double_peak = np.zeros(num_pulses, dtype=np.int64)

        # Analyze each pulse for double peaks or wide peaks
        is_detection_list = []
        positive_count = 0
        negative_count = 0

        # Get predicted labels to filter only actual detected pulses
        predicted = predicted_labels[:]

        # detection loop
        for i, pulse in enumerate(raw_pulses):
            # TODO: ask patrick if all pulses in the h5 files have a 1 for predicted_labels!!
            # Only analyze predicted positive pulses
            if predicted[i] != 1:
                continue

            # Wide pulses cannot also be double peaks
            if DETECTION_MODE == "wide" and is_double_peak[i] == 1:
                is_detection_list.append(0)
                negative_count += 1
                continue

            is_positive, _ = detect_pulse(pulse[:], fs)

            if is_positive:
                is_detection_list.append(1)
                positive_count += 1
            else:
                is_detection_list.append(0)
                negative_count += 1

            if (i + 1) % max(1, num_pulses // 10) == 0:
                con.log(f"  Processed {i + 1}/{num_pulses} pulses...")

        # Convert to numpy array
        is_detection_array = np.array(is_detection_list, dtype=np.int64)

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
    Process all h5 files in a directory to detect double peaks.

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
    con.log(f"Processing {len(path_list)} h5 files for double peak detection")
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


if __name__ == "__main__":
    # Path to directory containing h5 files with detected pulses
    data_path = Path(
        "/home/eisele/wrk/mscthesis/data/raw/eels-mfn2021_dummy_pulses_redetected/subtestset/"
    )

    # Process all h5 files to detect double peaks or wide pulses
    results = process_all_h5_files(data_path)
