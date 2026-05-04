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


def analyze_pulse_for_double_peak(
    pulse_waveform, prominence_threshold=0.3, distance_threshold=20
):
    """
    Analyze a single pulse waveform to determine if it's a double peak.

    Parameters:
    -----------
    pulse_waveform : np.ndarray
        2D array of shape (num_samples, num_channels)
    prominence_threshold : float
        Prominence threshold for finding peaks (as fraction of max amplitude)
    distance_threshold : int
        Minimum distance (in samples) between two peaks for them to be considered separate

    Returns:
    --------
    bool
        True if the pulse is detected as a double peak, False otherwise
    """
    # Compute the mean absolute amplitude across all channels for each time sample
    # This gives us a 1D representation of the pulse strength over time
    pulse_strength = np.mean(np.abs(pulse_waveform), axis=1)

    # Normalize pulse strength to [0, 1] range
    max_strength = np.max(pulse_strength)
    if max_strength == 0:
        return False

    pulse_strength_normalized = pulse_strength / max_strength

    # Calculate prominence threshold as a fraction of the normalized max
    prominence = prominence_threshold * max_strength

    # Find peaks in the pulse waveform
    # We look for local maxima with sufficient prominence and spacing
    peaks, peak_properties = find_peaks(
        pulse_strength, prominence=prominence, distance=distance_threshold
    )

    # A double peak has 2 or more prominent peaks
    # A single peak has only 1 prominent peak
    # Zero or 1 peak -> single pulse (0)
    # 2+ peaks -> double pulse (1)
    is_double_peak = len(peaks) >= 2

    return is_double_peak


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

        num_pulses = len(raw_pulses)
        con.log(f"  Found {num_pulses} pulses.")

        # Analyze each pulse for double peaks
        is_double_peak_list = []
        double_peak_count = 0
        single_peak_count = 0

        for i, pulse in enumerate(raw_pulses):
            is_double = analyze_pulse_for_double_peak(pulse[:])
            is_double_peak_list.append(1 if is_double else 0)

            if is_double:
                double_peak_count += 1
            else:
                single_peak_count += 1

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
            pulse_mean = np.mean(pulse_data, axis=1)

            # Plot waveform
            time_axis = np.arange(len(pulse_mean)) / fs
            ax.plot(time_axis, pulse_mean, linewidth=1.5, color="steelblue", alpha=0.8)
            ax.fill_between(time_axis, pulse_mean, alpha=0.3, color="steelblue")

            # Add peaks detection visualization
            pulse_strength = np.mean(np.abs(pulse_data), axis=1)
            prominence = 0.3 * np.max(pulse_strength)
            peaks, _ = find_peaks(pulse_strength, prominence=prominence, distance=20)

            # Mark peaks
            peak_times = peaks / fs
            peak_values = pulse_mean[peaks]
            ax.scatter(
                peak_times,
                peak_values,
                color="red",
                s=100,
                marker="*",
                zorder=5,
                label=f"{len(peaks)} peaks",
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
            pulse_mean = np.mean(pulse_data, axis=1)
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
                pulse_mean = np.mean(pulse_data, axis=1)

                # Plot waveform
                time_axis = np.arange(len(pulse_mean)) / fs
                ax.plot(
                    time_axis, pulse_mean, linewidth=1.5, color="steelblue", alpha=0.8
                )
                ax.fill_between(time_axis, pulse_mean, alpha=0.3, color="steelblue")

                # Add peaks detection visualization
                pulse_strength = np.mean(np.abs(pulse_data), axis=1)
                prominence = 0.3 * np.max(pulse_strength)
                peaks, _ = find_peaks(
                    pulse_strength, prominence=prominence, distance=20
                )

                # Mark peaks
                peak_times = peaks / fs
                peak_values = pulse_mean[peaks]
                ax.scatter(
                    peak_times,
                    peak_values,
                    color="red",
                    s=100,
                    marker="*",
                    zorder=5,
                    label=f"{len(peaks)} peaks",
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


if __name__ == "__main__":
    # Path to directory containing h5 files with detected pulses
    data_path = Path(
        "/home/eisele/wrk/mscthesis/data/newdata/eels-mfn2021_dummy_pulses_redetected/berlin_tank_site"
    )

    # Process all h5 files to detect double peaks
    results = process_all_h5_files(data_path)

    # Create combined visualization of all double peaks
    all_double_peak_plots = plot_all_double_peaks_combined(data_path, max_per_file=10)

    # Display all plots
    if all_double_peak_plots:
        con.log(f"\n{'=' * 60}")
        con.log(f"Displaying {len(all_double_peak_plots)} visualization(s)")
        con.log(f"{'=' * 60}\n")
        plt.show()
    else:
        con.log("\n⚠ No double peaks found to visualize.")

    # ========== ALTERNATIVE VISUALIZATIONS ==========
    # You can uncomment any of these to see different visualization styles:

    # Option 1: View statistics about single vs double peaks from a specific file
    # file_to_stats = data_path / "recordings_2023-11-22_pulses.h5"
    # fig, axes = plot_double_peaks_statistics(file_to_stats)
    # if fig:
    #     plt.show()

    # Option 2: Overlay all double peaks from one file for direct comparison
    # file_to_overlay = data_path / "recordings_2023-11-22_pulses.h5"
    # fig, ax = plot_double_peaks_overlay(file_to_overlay)
    # if fig:
    #     plt.show()

    # Option 3: Individual subplot view of double peaks from one file
    # file_to_individual = data_path / "recordings_2023-11-22_pulses.h5"
    # fig, axes = plot_double_peaks_from_file(file_to_individual, max_plots=20)
    # if fig:
    #     plt.show()
