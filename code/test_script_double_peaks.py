##### attempt to filter double peaked peaks #####

# wave show function in lib rosa (plots hist of audio data)
# let this run for whole dataset and get info of in which files and times double peaks are -> make histogram

# This script detects which pulses in h5 files are "double peaks" (wider pulses with 2 peaks instead of 1).
# Double peaks are identified by analyzing the peak structure of each pulse waveform.
# The script adds an "is_double_peak" data array to each h5 file containing binary labels (0 or 1).

# TODO: plotting von double peak detection trennen, double peaks finden verbessern

# import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import nixio
from rich.console import Console

# Initialize console for logging
con = Console()


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


def detect_double_peaks_in_file(file_path):
    """
    Detect double peaks in all pulses of a single h5 file and add the results as a data array.

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

        num_pulses = len(raw_pulses)
        con.log(f"  Found {num_pulses} pulses.")

        # Analyze each pulse for double peaks
        is_double_peak_list = []
        double_peak_count = 0
        single_peak_count = 0

        # Get predicted labels to filter only actual detected pulses
        predicted = predicted_labels[:]

        for i, pulse in enumerate(raw_pulses):
            # Only analyze pulses that were predicted as positive (label == 1)
            if predicted[i] == 1:
                # is_double = analyze_pulse_for_double_peak(pulse[:], fs)
                is_double, _ = detect_double_pulse(pulse[:], fs)
                if is_double:
                    is_double_peak_list.append(1)
                    double_peak_count += 1
                else:
                    is_double_peak_list.append(0)
                    single_peak_count += 1
            else:
                # Skip predicted negatives - don't add them to the analysis
                pass

            if (i + 1) % max(1, num_pulses // 10) == 0:
                con.log(f"  Processed {i + 1}/{num_pulses} pulses...")

        # Convert to numpy array
        is_double_peak_array = np.array(is_double_peak_list, dtype=np.int64)

        # Create or overwrite the "is_double_peak" data array in the h5 file
        if "is_double_peak" in data_array_names:
            con.log("  Updating existing 'is_double_peak' array...")
            block.data_arrays["is_double_peak"][:] = is_double_peak_array
        else:
            con.log("  Creating new 'is_double_peak' array...")
            block.create_data_array(
                "is_double_peak", "is_double_peak", data=is_double_peak_array
            )

        # Log summary
        con.log(
            f"  ✓ Completed: {double_peak_count} double peaks, {single_peak_count} single peaks"
        )

        return {
            "file": Path(file_path).name,
            "status": "completed",
            "num_pulses": num_pulses,
            "num_double_peaks": double_peak_count,
            "num_single_peaks": single_peak_count,
            "double_peak_ratio": double_peak_count / num_pulses
            if num_pulses > 0
            else 0,
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
        result = detect_double_peaks_in_file(fp)
        results.append(result)
        con.log()

    # Print summary statistics
    con.log(f"\n{'=' * 60}")
    con.log("SUMMARY")
    con.log(f"{'=' * 60}")

    total_pulses = sum(r.get("num_pulses", 0) for r in results)
    total_double_peaks = sum(r.get("num_double_peaks", 0) for r in results)
    total_single_peaks = sum(r.get("num_single_peaks", 0) for r in results)

    con.log(f"Total files processed: {len(results)}")
    con.log(f"Total pulses analyzed: {total_pulses}")
    con.log(f"Total double peaks: {total_double_peaks}")
    con.log(f"Total single peaks: {total_single_peaks}")
    con.log(
        f"Double peak ratio: {total_double_peaks / total_pulses if total_pulses > 0 else 0:.2%}"
    )

    return results


#################################
############# VISUALIZATION #####
#################################


def plot_double_peaks_from_file(file_path, max_plots=20, figsize=(16, 10)):
    """
    Load double peaks from h5 file and plot them using librosa waveshow.

    Parameters:
    -----------
    file_path : Path or str
        Path to the h5 file to visualize
    max_plots : int
        Maximum number of double peaks to plot (if there are many)
    figsize : tuple
        Figure size (width, height)

    Returns:
    --------
    fig, axes
        Matplotlib figure and axes objects
    """
    file_path = Path(file_path)
    con.log(f"Loading double peaks from: {file_path.name}")

    # Open h5 file
    file = nixio.File.open(str(file_path), nixio.FileMode.ReadOnly)

    try:
        block = file.blocks["pulses"]
        data_array_names = [da.name for da in block.data_arrays]

        if "is_double_peak" not in data_array_names:
            con.log(
                "  ⚠ File does not have 'is_double_peak' array. Run detection first."
            )
            return None, None

        # Load data
        is_double_peak = block.data_arrays["is_double_peak"][:]
        raw_pulses = block.data_arrays["raw_pulses"]
        fs = file.sections["pulses_metadata"]["metadata"]["samplerate"]

        # Get indices of double peaks
        double_peak_indices = np.where(is_double_peak == 1)[0]
        num_double_peaks = len(double_peak_indices)

        if num_double_peaks == 0:
            con.log("  No double peaks found in this file.")
            return None, None

        con.log(
            f"  Found {num_double_peaks} double peaks. Plotting first {min(max_plots, num_double_peaks)}..."
        )

        # Limit number of plots
        indices_to_plot = double_peak_indices[:max_plots]
        num_to_plot = len(indices_to_plot)

        # Create subplots
        n_cols = min(4, num_to_plot)  # Max 4 columns
        n_rows = (num_to_plot + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if num_to_plot == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten()

        # Plot each double peak
        for plot_idx, pulse_idx in enumerate(indices_to_plot):
            ax = axes[plot_idx]
            pulse_data = raw_pulses[pulse_idx]  # Shape: (num_samples, 16 channels)

            # Use mean across channels for visualization
            # pulse_mean = np.mean(pulse_data, axis=1)
            pulse_mean, best_channel = get_representative_waveform(pulse_data)

            # Plot waveform
            time_axis = np.arange(len(pulse_mean)) / fs
            ax.plot(time_axis, pulse_mean, linewidth=1.5, color="steelblue", alpha=0.8)
            ax.fill_between(time_axis, pulse_mean, alpha=0.3, color="steelblue")

            # # Add peaks detection visualization using same logic as detection function
            # pulse_strength = np.mean(np.abs(pulse_data), axis=1)
            # pulse_waveform_mean = np.mean(pulse_data, axis=1)
            # max_strength = np.max(pulse_strength)
            # prominence = 0.1 * max_strength
            # peaks, _ = find_peaks(pulse_strength, prominence=prominence)

            # # Apply same filtering criteria as detection function
            # filtered_peaks = []
            # if len(peaks) == 2:
            #     # Check same sign
            #     peak1_value = pulse_waveform_mean[peaks[0]]
            #     peak2_value = pulse_waveform_mean[peaks[1]]

            #     # Check if peaks have the same sign (both above 0 or both below 0)
            #     same_sign = not (
            #         (peak1_value > 0 and peak2_value < 0)
            #         or (peak1_value < 0 and peak2_value > 0)
            #     )

            #     if same_sign:
            #         # Check if at least one is a local maximum within 0.004s window
            #         window_samples = int(0.004 * fs)

            #         def is_local_max_in_window(signal, idx, window_size):
            #             start = max(0, idx - window_size)
            #             end = min(len(signal), idx + window_size + 1)
            #             return signal[idx] == np.max(signal[start:end])

            #         peak1_local_max = is_local_max_in_window(
            #             pulse_strength, peaks[0], window_samples
            #         )
            #         peak2_local_max = is_local_max_in_window(
            #             pulse_strength, peaks[1], window_samples
            #         )

            #         if peak1_local_max or peak2_local_max:
            #             filtered_peaks = list(peaks)

            # # Mark only the valid peaks
            # if filtered_peaks:
            #     peak_times = np.array(filtered_peaks) / fs
            #     peak_values = pulse_mean[filtered_peaks]
            #     ax.scatter(
            #         peak_times,
            #         peak_values,
            #         color="red",
            #         s=100,
            #         marker="*",
            #         zorder=5,
            #         label=f"{len(filtered_peaks)} peaks",
            #     )
            # else:
            #     ax.text(
            #         0.5,
            #         0.5,
            #         "No valid double peak",
            #         ha="center",
            #         va="center",
            #         transform=ax.transAxes,
            #         fontsize=9,
            #         color="gray",
            #     )

            # use same logic as detection function
            is_double, info = detect_double_pulse(pulse_data, fs)

            if is_double:
                peaks = info["peaks"]
                peak_times = np.array(peaks) / fs
                peak_vals = pulse_mean[peaks]

                ax.scatter(
                    peak_times,
                    peak_vals,
                    color="red",
                    s=100,
                    marker="*",
                    zorder=5,
                    label="2 peaks",
                )

            else:
                ax.text(
                    0.5,
                    0.5,
                    info.get("reason", "not double"),
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="gray",
                )

            # Formatting
            ax.set_title(
                f"Double Peak #{pulse_idx}\n(Pulse index: {pulse_idx})",
                fontsize=10,
                fontweight="bold",
            )
            ax.set_xlabel("Time (s)", fontsize=9)
            ax.set_ylabel("Amplitude (a.u.)", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        # Hide unused subplots
        for plot_idx in range(num_to_plot, len(axes)):
            axes[plot_idx].set_visible(False)

        plt.suptitle(
            f"Double Peaks from {file_path.name}\n(First {num_to_plot} of {num_double_peaks} double peaks)",
            fontsize=14,
            fontweight="bold",
            y=0.995,
        )
        plt.tight_layout()

        return fig, axes

    finally:
        file.close()


def plot_double_peaks_overlay(file_path, figsize=(14, 6)):
    """
    Plot all detected double peaks overlaid on each other for comparison.

    Parameters:
    -----------
    file_path : Path or str
        Path to the h5 file to visualize
    figsize : tuple
        Figure size (width, height)

    Returns:
    --------
    fig, ax
        Matplotlib figure and axes objects
    """
    file_path = Path(file_path)
    con.log(f"Creating overlay plot from: {file_path.name}")

    # Open h5 file
    file = nixio.File.open(str(file_path), nixio.FileMode.ReadOnly)

    try:
        block = file.blocks["pulses"]
        data_array_names = [da.name for da in block.data_arrays]

        if "is_double_peak" not in data_array_names:
            con.log(
                "  ⚠ File does not have 'is_double_peak' array. Run detection first."
            )
            return None, None

        # Load data
        is_double_peak = block.data_arrays["is_double_peak"][:]
        raw_pulses = block.data_arrays["raw_pulses"]
        fs = file.sections["pulses_metadata"]["metadata"]["samplerate"]

        # Get indices of double peaks
        double_peak_indices = np.where(is_double_peak == 1)[0]
        num_double_peaks = len(double_peak_indices)

        if num_double_peaks == 0:
            con.log("  No double peaks found in this file.")
            return None, None

        con.log(f"  Overlaying {num_double_peaks} double peaks...")

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot each double peak with transparency
        colors = plt.cm.viridis(np.linspace(0, 1, num_double_peaks))

        for i, pulse_idx in enumerate(double_peak_indices):
            pulse_data = raw_pulses[pulse_idx]
            # pulse_mean = np.mean(pulse_data, axis=1)
            pulse_mean, best_channel = get_representative_waveform(pulse_data)
            time_axis = np.arange(len(pulse_mean)) / fs

            ax.plot(
                time_axis,
                pulse_mean,
                alpha=0.6,
                linewidth=1.5,
                color=colors[i],
                label=f"Pulse {pulse_idx}",
            )

        # Formatting
        ax.set_xlabel("Time (s)", fontsize=11)
        ax.set_ylabel("Amplitude (a.u.)", fontsize=11)
        ax.set_title(
            f"All Double Peaks Overlay - {file_path.name}\n({num_double_peaks} double peaks)",
            fontsize=13,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)

        # Only show legend if not too many peaks
        if num_double_peaks <= 20:
            ax.legend(fontsize=8, loc="upper right", ncol=2)

        plt.tight_layout()

        return fig, ax

    finally:
        file.close()


def plot_double_peaks_statistics(file_path, figsize=(14, 5)):
    """
    Plot statistics about single vs double peaks.

    Parameters:
    -----------
    file_path : Path or str
        Path to the h5 file to visualize
    figsize : tuple
        Figure size (width, height)

    Returns:
    --------
    fig, axes
        Matplotlib figure and axes objects
    """
    file_path = Path(file_path)
    con.log(f"Creating statistics plot from: {file_path.name}")

    # Open h5 file
    file = nixio.File.open(str(file_path), nixio.FileMode.ReadOnly)

    try:
        block = file.blocks["pulses"]
        data_array_names = [da.name for da in block.data_arrays]

        if "is_double_peak" not in data_array_names:
            con.log(
                "  ⚠ File does not have 'is_double_peak' array. Run detection first."
            )
            return None, None

        # Load data
        is_double_peak = block.data_arrays["is_double_peak"][:]
        raw_pulses = block.data_arrays["raw_pulses"]

        num_double = np.sum(is_double_peak)
        num_single = len(is_double_peak) - num_double

        # Compute pulse widths for both categories
        single_peak_indices = np.where(is_double_peak == 0)[0]
        double_peak_indices = np.where(is_double_peak == 1)[0]

        single_widths = []
        for idx in single_peak_indices:
            pulse_strength = np.mean(np.abs(raw_pulses[idx]), axis=1)
            threshold = 0.1 * np.max(pulse_strength)
            above_threshold = np.where(pulse_strength > threshold)[0]
            if len(above_threshold) > 0:
                width = above_threshold[-1] - above_threshold[0]
                single_widths.append(width)

        double_widths = []
        for idx in double_peak_indices:
            pulse_strength = np.mean(np.abs(raw_pulses[idx]), axis=1)
            threshold = 0.1 * np.max(pulse_strength)
            above_threshold = np.where(pulse_strength > threshold)[0]
            if len(above_threshold) > 0:
                width = above_threshold[-1] - above_threshold[0]
                double_widths.append(width)

        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Plot 1: Pie chart of single vs double
        ax = axes[0]
        sizes = [num_single, num_double]
        colors_pie = ["#3498db", "#e74c3c"]
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=["Single Peak", "Double Peak"],
            autopct="%1.1f%%",
            colors=colors_pie,
            startangle=90,
        )
        ax.set_title("Peak Type Distribution", fontsize=11, fontweight="bold")
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")

        # Plot 2: Bar chart of counts
        ax = axes[1]
        ax.bar(
            ["Single Peak", "Double Peak"],
            [num_single, num_double],
            color=colors_pie,
            alpha=0.7,
            edgecolor="black",
        )
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title("Peak Counts", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        for i, v in enumerate([num_single, num_double]):
            ax.text(i, v + 5, str(v), ha="center", fontweight="bold")

        # Plot 3: Pulse width comparison
        ax = axes[2]
        if single_widths and double_widths:
            bp = ax.boxplot(
                [single_widths, double_widths],
                labels=["Single Peak", "Double Peak"],
                patch_artist=True,
            )
            for patch, color in zip(bp["boxes"], colors_pie):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.set_ylabel("Pulse Width (samples)", fontsize=10)
            ax.set_title("Pulse Width Comparison", fontsize=11, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="y")
        else:
            ax.text(
                0.5,
                0.5,
                "Insufficient data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        plt.suptitle(
            f"Statistics - {file_path.name}", fontsize=13, fontweight="bold", y=1.00
        )
        plt.tight_layout()

        return fig, axes

    finally:
        file.close()


def plot_all_double_peaks_combined(data_path, max_per_file=10, figsize_per_plot=(4, 3)):
    """
    Plot all double peaks from all h5 files in a combined grid visualization.

    Parameters:
    -----------
    data_path : Path or str
        Path to directory containing h5 files
    max_per_file : int
        Maximum number of double peaks to plot per file (randomly sampled if more exist)
    figsize_per_plot : tuple
        Figure size per subplot (width, height)

    Returns:
    --------
    dict
        Dictionary with file names as keys and (fig, axes) tuples as values
    """
    data_path = Path(data_path)
    path_list = get_path_list(data_path)

    all_plots = {}
    total_double_peaks = 0

    con.log("\n" + "=" * 60)
    con.log("Creating combined double peaks visualization")
    con.log("=" * 60 + "\n")

    for file_path in path_list:
        file = nixio.File.open(str(file_path), nixio.FileMode.ReadOnly)

        try:
            block = file.blocks["pulses"]
            data_array_names = [da.name for da in block.data_arrays]

            if "is_double_peak" not in data_array_names:
                con.log(f"⊘ {file_path.name}: No 'is_double_peak' array")
                continue

            # Load data
            is_double_peak = block.data_arrays["is_double_peak"][:]
            raw_pulses = block.data_arrays["raw_pulses"]
            fs = file.sections["pulses_metadata"]["metadata"]["samplerate"]

            # Get indices of double peaks
            double_peak_indices = np.where(is_double_peak == 1)[0]
            num_double_peaks = len(double_peak_indices)

            if num_double_peaks == 0:
                con.log(f"  {file_path.name}: No double peaks found")
                continue

            con.log(f"  {file_path.name}: Found {num_double_peaks} double peaks")

            # Randomly sample if too many
            if num_double_peaks > max_per_file:
                selected_indices = np.random.choice(
                    double_peak_indices, size=max_per_file, replace=False
                )
                con.log(f"    → Randomly sampling {max_per_file} for visualization")
            else:
                selected_indices = double_peak_indices

            num_to_plot = len(selected_indices)
            total_double_peaks += num_to_plot

            # Create subplots grid
            n_cols = min(4, num_to_plot)  # Max 4 columns
            n_rows = (num_to_plot + n_cols - 1) // n_cols

            figsize = (figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

            if num_to_plot == 1:
                axes = np.array([axes])
            else:
                axes = axes.flatten()

            # Plot each sampled double peak
            for plot_idx, pulse_idx in enumerate(selected_indices):
                ax = axes[plot_idx]
                pulse_data = raw_pulses[pulse_idx]  # Shape: (num_samples, 16 channels)

                # Use mean across channels for visualization
                # pulse_mean = np.mean(pulse_data, axis=1)
                pulse_mean, best_channel = get_representative_waveform(pulse_data)

                # Plot waveform
                time_axis = np.arange(len(pulse_mean)) / fs
                ax.plot(
                    time_axis, pulse_mean, linewidth=1.5, color="steelblue", alpha=0.8
                )
                ax.fill_between(time_axis, pulse_mean, alpha=0.3, color="steelblue")

                # # Add peaks detection visualization using same logic as detection function
                # pulse_strength = np.mean(np.abs(pulse_data), axis=1)
                # pulse_waveform_mean = np.mean(pulse_data, axis=1)
                # max_strength = np.max(pulse_strength)
                # prominence = 0.1 * max_strength
                # peaks, _ = find_peaks(pulse_strength, prominence=prominence)

                # # Apply same filtering criteria as detection function
                # filtered_peaks = []
                # if len(peaks) == 2:
                #     # Check same sign
                #     peak1_value = pulse_waveform_mean[peaks[0]]
                #     peak2_value = pulse_waveform_mean[peaks[1]]

                #     # Check if peaks have the same sign (both above 0 or both below 0)
                #     same_sign = not (
                #         (peak1_value > 0 and peak2_value < 0)
                #         or (peak1_value < 0 and peak2_value > 0)
                #     )

                #     if same_sign:
                #         # Check if at least one is a local maximum within 0.004s window
                #         window_samples = int(0.004 * fs)

                #         def is_local_max_in_window(signal, idx, window_size):
                #             start = max(0, idx - window_size)
                #             end = min(len(signal), idx + window_size + 1)
                #             return signal[idx] == np.max(signal[start:end])

                #         peak1_local_max = is_local_max_in_window(
                #             pulse_strength, peaks[0], window_samples
                #         )
                #         peak2_local_max = is_local_max_in_window(
                #             pulse_strength, peaks[1], window_samples
                #         )

                #         if peak1_local_max or peak2_local_max:
                #             filtered_peaks = list(peaks)

                # # Mark peaks
                # if filtered_peaks:
                #     peak_times = np.array(filtered_peaks) / fs
                #     peak_values = pulse_mean[filtered_peaks]
                #     ax.scatter(
                #         peak_times,
                #         peak_values,
                #         color="red",
                #         s=100,
                #         marker="*",
                #         zorder=5,
                #         label=f"{len(filtered_peaks)} peaks",
                #     )

                # use same logic as detection function
                is_double, info = detect_double_pulse(pulse_data, fs)

                if is_double:
                    peaks = info["peaks"]
                    peak_times = np.array(peaks) / fs
                    peak_vals = pulse_mean[peaks]

                    ax.scatter(
                        peak_times,
                        peak_vals,
                        color="red",
                        s=100,
                        marker="*",
                        zorder=5,
                        label="2 peaks",
                    )

                else:
                    ax.text(
                        0.5,
                        0.5,
                        info.get("reason", "not double"),
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="gray",
                    )

                # Formatting
                ax.set_title(f"Pulse {pulse_idx}", fontsize=9, fontweight="bold")
                ax.set_xlabel("Time (s)", fontsize=8)
                ax.set_ylabel("Amplitude (a.u.)", fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=7)

            # Hide unused subplots
            for plot_idx in range(num_to_plot, len(axes)):
                axes[plot_idx].set_visible(False)

            plt.suptitle(
                f"Double Peaks - {file_path.name}\n({num_to_plot} of {num_double_peaks} double peaks)",
                fontsize=12,
                fontweight="bold",
                y=0.995,
            )
            plt.tight_layout()

            all_plots[file_path.name] = (fig, axes)

        finally:
            file.close()

    con.log(f"\n✓ Total double peaks to visualize: {total_double_peaks}")
    con.log(f"✓ Created {len(all_plots)} figures")

    return all_plots


def interactive_double_peak_verification(data_path):
    """
    Interactive CLI-based tool to verify double peak classifications.

    Displays each double peak waveform one at a time and allows the user to verify or correct
    the classification by pressing Y (confirm double peak) or N (not a double peak).
    Changes are saved to the h5 files only after all files have been reviewed.

    Parameters:
    -----------
    data_path : Path or str
        Path to directory containing h5 files

    Returns:
    --------
    dict
        Summary statistics of corrections made
    """
    data_path = Path(data_path)
    path_list = get_path_list(data_path)

    if not path_list:
        con.log("No h5 files found.")
        return {}

    # Dictionary to track changes: {file_path: {pulse_idx: new_label}}
    changes = {}
    total_verified = 0
    total_corrected = 0

    con.log(f"\n{'=' * 60}")
    con.log("INTERACTIVE DOUBLE PEAK VERIFICATION")
    con.log(f"{'=' * 60}\n")
    con.log("Instructions:")
    con.log("  Press 'Y' to confirm this is a double peak (keep as 1)")
    con.log("  Press 'N' to mark this as NOT a double peak (change to 0)")
    con.log("  Press 'Q' to quit without saving")
    con.log(f"\n{'=' * 60}\n")

    try:
        for file_idx, file_path in enumerate(path_list, 1):
            con.log(f"\n[File {file_idx}/{len(path_list)}] {file_path.name}")

            # Open file in read mode first to check content
            file = nixio.File.open(str(file_path), nixio.FileMode.ReadOnly)

            try:
                block = file.blocks["pulses"]
                data_array_names = [da.name for da in block.data_arrays]

                if "is_double_peak" not in data_array_names:
                    con.log("  ⚠ No 'is_double_peak' array found. Skipping.")
                    continue

                # Load data
                is_double_peak = block.data_arrays["is_double_peak"][:]
                raw_pulses = block.data_arrays["raw_pulses"]
                fs = file.sections["pulses_metadata"]["metadata"]["samplerate"]

                # Get indices of double peaks
                double_peak_indices = np.where(is_double_peak == 1)[0]
                num_double_peaks = len(double_peak_indices)

                if num_double_peaks == 0:
                    con.log("  No double peaks to verify in this file.")
                    continue

                con.log(f"  Found {num_double_peaks} double peaks to verify")

                # Initialize changes for this file
                if str(file_path) not in changes:
                    changes[str(file_path)] = {}

                # Verify each double peak
                for pulse_num, pulse_idx in enumerate(double_peak_indices, 1):
                    pulse_data = raw_pulses[pulse_idx]
                    pulse_mean, best_channel = get_representative_waveform(pulse_data)

                    # Detect peaks for visualization
                    is_double, info = detect_double_pulse(pulse_data, fs)

                    # Create figure
                    fig, ax = plt.subplots(figsize=(10, 5))

                    # Plot waveform
                    time_axis = np.arange(len(pulse_mean)) / fs
                    ax.plot(
                        time_axis,
                        pulse_mean,
                        linewidth=2,
                        color="steelblue",
                        alpha=0.8,
                        label="Waveform",
                    )
                    ax.fill_between(time_axis, pulse_mean, alpha=0.2, color="steelblue")

                    # Mark detected peaks
                    if is_double and "peaks" in info:
                        peaks = info["peaks"]
                        peak_times = np.array(peaks) / fs
                        peak_vals = pulse_mean[peaks]
                        ax.scatter(
                            peak_times,
                            peak_vals,
                            color="red",
                            s=80,
                            marker="*",
                            zorder=5,
                            label="Detected peaks",
                        )

                    # Formatting
                    ax.set_xlabel("Time (s)", fontsize=11)
                    ax.set_ylabel("Amplitude (a.u.)", fontsize=11)
                    ax.set_title(
                        f"Pulse {pulse_idx} [{pulse_num}/{num_double_peaks}] from {file_path.name}\n"
                        f"Channel {best_channel} (strongest) - Verify: Is this a double peak?",
                        fontsize=12,
                        fontweight="bold",
                    )
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=10, loc="upper right")

                    plt.tight_layout()
                    plt.show(block=False)
                    plt.pause(0.1)

                    # Wait for user input
                    valid_input = False
                    while not valid_input:
                        user_input = (
                            input("\n>>> Verify double peak? (Y/N/Q): ").strip().upper()
                        )

                        if user_input == "Y":
                            # Keep as double peak (1)
                            changes[str(file_path)][pulse_idx] = 1
                            con.log(
                                f"    ✓ Confirmed double peak for pulse {pulse_idx}"
                            )
                            total_verified += 1
                            valid_input = True
                        elif user_input == "N":
                            # Mark as NOT double peak (0)
                            changes[str(file_path)][pulse_idx] = 0
                            con.log(
                                f"    ✗ Corrected pulse {pulse_idx} to NOT a double peak"
                            )
                            total_corrected += 1
                            valid_input = True
                        elif user_input == "Q":
                            con.log("\n⚠ Exiting without saving...")
                            plt.close("all")
                            return {
                                "status": "cancelled",
                                "message": "Verification cancelled by user",
                            }
                        else:
                            con.log("    Invalid input. Please press Y, N, or Q.")

                    plt.close("all")

            finally:
                file.close()

        # Save all changes
        con.log(f"\n{'=' * 60}")
        con.log("SAVING CHANGES...")
        con.log(f"{'=' * 60}\n")

        for file_path_str, pulse_changes in changes.items():
            if not pulse_changes:
                continue

            file_path = Path(file_path_str)
            con.log(f"Updating {file_path.name}...")

            # Open file in read/write mode to save changes
            file = nixio.File.open(str(file_path), nixio.FileMode.ReadWrite)

            try:
                block = file.blocks["pulses"]
                is_double_peak_array = block.data_arrays["is_double_peak"][:]

                # Apply changes
                for pulse_idx, new_value in pulse_changes.items():
                    is_double_peak_array[pulse_idx] = new_value

                # Write back to file
                block.data_arrays["is_double_peak"][:] = is_double_peak_array
                con.log(f"  ✓ Saved {len(pulse_changes)} changes")

            finally:
                file.close()

        # Print summary
        con.log(f"\n{'=' * 60}")
        con.log("VERIFICATION COMPLETE")
        con.log(f"{'=' * 60}")
        con.log(f"Total pulses verified: {total_verified}")
        con.log(f"Total corrections made: {total_corrected}")
        con.log(
            f"Total changes saved: {sum(len(changes) for changes in changes.values())}"
        )

        return {
            "status": "completed",
            "verified": total_verified,
            "corrected": total_corrected,
        }

    except KeyboardInterrupt:
        con.log("\n⚠ Interrupted by user. No changes saved.")
        return {"status": "interrupted", "message": "Process interrupted"}


if __name__ == "__main__":
    # Path to directory containing h5 files with detected pulses
    data_path = Path(
        "/home/eisele/wrk/mscthesis/data/raw/eels-mfn2021_dummy_pulses_redetected/berlin_tank_site/"
    )

    # Process all h5 files to detect double peaks
    results = process_all_h5_files(data_path)

    # Start interactive verification of double peaks
    verification_results = interactive_double_peak_verification(data_path)

    # TODO: maybe change UI so it doesnt switch between command line and matplotlib windows, but instead shows all double peaks in a grid and allows user to click on each one to verify? would be more user friendly and less disruptive than showing one at a time. could also add "confirm all" button for quick verification if most look correct.
