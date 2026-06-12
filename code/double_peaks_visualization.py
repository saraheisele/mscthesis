"""
Visualization and interactive verification for detected pulses (double peaks or wide pulses).

This script reads detection results from h5 files and provides:
- Static visualization functions (grids, overlays, statistics)
- Interactive verification tool for manual review and correction

The detection should have been run first using double_peaks_detection.py
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import nixio
from rich.console import Console

# Import detection functions and utilities from detection script
from double_peaks_detection import (
    get_representative_waveform,
    detect_pulse,
    compute_half_max_width,
    get_path_list,
    DETECTION_MODE,
    ARRAY_NAME,
    DISPLAY_NAME,
)

from data_paths import H5_DIR

# Initialize console for logging
con = Console()


#################################
############# VISUALIZATION #####
#################################


def plot_detected_pulses_from_file(file_path, max_plots=20, figsize=(16, 10)):
    """
    Load detected pulses from h5 file and plot them in a grid.

    Parameters:
    -----------
    file_path : Path or str
        Path to the h5 file to visualize
    max_plots : int
        Maximum number of pulses to plot (if there are many)
    figsize : tuple
        Figure size (width, height)

    Returns:
    --------
    fig, axes
        Matplotlib figure and axes objects
    """
    file_path = Path(file_path)
    con.log(f"Loading {DISPLAY_NAME}s from: {file_path.name}")

    # Open h5 file
    file = nixio.File.open(str(file_path), nixio.FileMode.ReadOnly)

    try:
        block = file.blocks["pulses"]
        data_array_names = [da.name for da in block.data_arrays]

        if ARRAY_NAME not in data_array_names:
            con.log(
                f"  ⚠ File does not have '{ARRAY_NAME}' array. Run detection first."
            )
            return None, None

        # Load data
        detection_array = block.data_arrays[ARRAY_NAME][:]
        raw_pulses = block.data_arrays["raw_pulses"]
        fs = file.sections["pulses_metadata"]["metadata"]["samplerate"]

        # Get indices of detected pulses
        detected_indices = np.where(detection_array == 1)[0]
        num_detected = len(detected_indices)

        if num_detected == 0:
            con.log(f"  No {DISPLAY_NAME}s found in this file.")
            return None, None

        con.log(
            f"  Found {num_detected} {DISPLAY_NAME}s. Plotting first {min(max_plots, num_detected)}..."
        )

        # Limit number of plots
        indices_to_plot = detected_indices[:max_plots]
        num_to_plot = len(indices_to_plot)

        # Create subplots
        n_cols = min(4, num_to_plot)  # Max 4 columns
        n_rows = (num_to_plot + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if num_to_plot == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten()

        # Plot each detected pulse
        for plot_idx, pulse_idx in enumerate(indices_to_plot):
            ax = axes[plot_idx]
            pulse_data = raw_pulses[pulse_idx]  # Shape: (num_samples, num_channels)

            # Get representative waveform
            pulse_mean, best_channel = get_representative_waveform(pulse_data)

            # Plot waveform
            time_axis = np.arange(len(pulse_mean)) / fs
            ax.plot(time_axis, pulse_mean, linewidth=1.5, color="steelblue", alpha=0.8)
            ax.fill_between(time_axis, pulse_mean, alpha=0.3, color="steelblue")

            # Use same logic as detection function for visualization
            is_positive, info = detect_pulse(pulse_data, fs)

            # Mode specific markers for visualization
            if DETECTION_MODE == "wide" and is_positive:
                ax.axhline(
                    info["half_height"],
                    color="orange",
                    linestyle="--",
                    alpha=0.7,
                )

                ax.axvspan(
                    info["left_idx"] / fs,
                    info["right_idx"] / fs,
                    color="orange",
                    alpha=0.2,
                    label=f"Width = {info['width_samples'] / fs * 1000:.2f} ms",
                )

            elif DETECTION_MODE == "double" and is_positive:
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
                    info.get("reason", "not detected"),
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="gray",
                )

            # Formatting
            ax.set_title(
                f"{DISPLAY_NAME.capitalize()} #{pulse_idx}\n(Pulse index: {pulse_idx})",
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
            f"{DISPLAY_NAME.capitalize()}s from {file_path.name}\n(First {num_to_plot} of {num_detected} {DISPLAY_NAME}s)",
            fontsize=14,
            fontweight="bold",
            y=0.995,
        )
        plt.tight_layout()

        return fig, axes

    finally:
        file.close()


def plot_detected_pulses_overlay(file_path, figsize=(14, 6)):
    """
    Plot all detected pulses overlaid on each other for comparison.

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

        if ARRAY_NAME not in data_array_names:
            con.log(
                f"  ⚠ File does not have '{ARRAY_NAME}' array. Run detection first."
            )
            return None, None

        # Load data
        detection_array = block.data_arrays[ARRAY_NAME][:]
        raw_pulses = block.data_arrays["raw_pulses"]
        fs = file.sections["pulses_metadata"]["metadata"]["samplerate"]

        # Get indices of detected pulses
        detected_indices = np.where(detection_array == 1)[0]
        num_detected = len(detected_indices)

        if num_detected == 0:
            con.log(f"  No {DISPLAY_NAME}s found in this file.")
            return None, None

        con.log(f"  Overlaying {num_detected} {DISPLAY_NAME}s...")

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot each pulse with transparency
        colors = plt.cm.viridis(np.linspace(0, 1, num_detected))

        for i, pulse_idx in enumerate(detected_indices):
            pulse_data = raw_pulses[pulse_idx]
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
            f"All {DISPLAY_NAME.capitalize()}s Overlay - {file_path.name}\n({num_detected} {DISPLAY_NAME}s)",
            fontsize=13,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)

        # Only show legend if not too many pulses
        if num_detected <= 20:
            ax.legend(fontsize=8, loc="upper right", ncol=2)

        plt.tight_layout()

        return fig, ax

    finally:
        file.close()


def plot_detection_statistics(file_path, figsize=(14, 5)):
    """
    Plot statistics about detected vs non-detected pulses.

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

        if ARRAY_NAME not in data_array_names:
            con.log(
                f"  ⚠ File does not have '{ARRAY_NAME}' array. Run detection first."
            )
            return None, None

        # Load data
        detection_array = block.data_arrays[ARRAY_NAME][:]
        raw_pulses = block.data_arrays["raw_pulses"]

        num_detected = np.sum(detection_array)
        num_not_detected = len(detection_array) - num_detected

        # Compute pulse widths for both categories
        not_detected_indices = np.where(detection_array == 0)[0]
        detected_indices = np.where(detection_array == 1)[0]

        not_detected_widths = []
        for idx in not_detected_indices:
            pulse_strength = np.mean(np.abs(raw_pulses[idx]), axis=1)
            threshold = 0.1 * np.max(pulse_strength)
            above_threshold = np.where(pulse_strength > threshold)[0]
            if len(above_threshold) > 0:
                width = above_threshold[-1] - above_threshold[0]
                not_detected_widths.append(width)

        detected_widths = []
        for idx in detected_indices:
            pulse_strength = np.mean(np.abs(raw_pulses[idx]), axis=1)
            threshold = 0.1 * np.max(pulse_strength)
            above_threshold = np.where(pulse_strength > threshold)[0]
            if len(above_threshold) > 0:
                width = above_threshold[-1] - above_threshold[0]
                detected_widths.append(width)

        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Plot 1: Pie chart of detected vs not detected
        ax = axes[0]
        sizes = [num_not_detected, num_detected]
        colors_pie = ["#3498db", "#e74c3c"]
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=["Not Detected", f"Detected\n({DISPLAY_NAME})"],
            autopct="%1.1f%%",
            colors=colors_pie,
            startangle=90,
        )
        ax.set_title("Detection Distribution", fontsize=11, fontweight="bold")
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")

        # Plot 2: Bar chart of counts
        ax = axes[1]
        ax.bar(
            ["Not Detected", "Detected"],
            [num_not_detected, num_detected],
            color=colors_pie,
            alpha=0.7,
            edgecolor="black",
        )
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title("Pulse Counts", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        for i, v in enumerate([num_not_detected, num_detected]):
            ax.text(i, v + 5, str(v), ha="center", fontweight="bold")

        # Plot 3: Pulse width comparison
        ax = axes[2]
        if not_detected_widths and detected_widths:
            bp = ax.boxplot(
                [not_detected_widths, detected_widths],
                labels=["Not Detected", "Detected"],
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
            f"Detection Statistics - {file_path.name}",
            fontsize=13,
            fontweight="bold",
            y=1.00,
        )
        plt.tight_layout()

        return fig, axes

    finally:
        file.close()


def plot_all_pulses_width_histogram(data_path, figsize=(14, 6), bins=50):
    """
    Create a histogram of half-max widths for ALL pulses from all h5 files.
    Separates detected pulses from non-detected for comparison.

    Parameters:
    -----------
    data_path : Path or str
        Path to directory containing h5 files
    figsize : tuple
        Figure size (width, height)
    bins : int
        Number of histogram bins

    Returns:
    --------
    fig, axes
        Matplotlib figure and axes objects
    dict
        Statistics about the pulse widths
    """
    data_path = Path(data_path)
    path_list = get_path_list(data_path)

    if not path_list:
        con.log("No h5 files found.")
        return None, None, {}

    con.log("\nComputing half-max widths for ALL pulses from all h5 files...")

    detected_widths_ms = []
    not_detected_widths_ms = []
    total_pulses = 0
    total_detected = 0
    total_not_detected = 0

    for file_idx, file_path in enumerate(path_list, 1):
        con.log(f"  [{file_idx}/{len(path_list)}] {file_path.name}", end="")

        file = nixio.File.open(str(file_path), nixio.FileMode.ReadOnly)

        try:
            block = file.blocks["pulses"]
            data_array_names = [da.name for da in block.data_arrays]

            raw_pulses = block.data_arrays["raw_pulses"]
            fs = file.sections["pulses_metadata"]["metadata"]["samplerate"]

            # Check if detection array exists
            if ARRAY_NAME in data_array_names:
                detection_array = block.data_arrays[ARRAY_NAME][:]
            else:
                detection_array = np.zeros(len(raw_pulses), dtype=np.int64)

            num_pulses = len(raw_pulses)
            total_pulses += num_pulses

            # Process each pulse
            for pulse_idx, pulse_data in enumerate(raw_pulses):
                # Get representative waveform
                signal, best_channel = get_representative_waveform(pulse_data)

                # Apply baseline correction (same as detection does)
                baseline = np.median(signal[: len(signal) // 5])
                signal_corrected = signal - baseline

                # Compute half-max width
                width_sec, width_info = compute_half_max_width(signal_corrected, fs)
                width_ms = width_sec * 1000

                # Categorize as detected or not detected
                is_detected = detection_array[pulse_idx] == 1

                if is_detected:
                    detected_widths_ms.append(width_ms)
                    total_detected += 1
                else:
                    not_detected_widths_ms.append(width_ms)
                    total_not_detected += 1

            con.log(f" ✓ ({num_pulses} pulses)")

            file.close()

        except Exception as e:
            con.log(f" ✗ Error processing file: {e}")
            file.close()  # Ensure file is closed before continuing
            continue

    con.log(f"\n✓ Total pulses processed: {total_pulses}")
    con.log(f"  - Detected as {DISPLAY_NAME}: {total_detected}")
    con.log(f"  - Not detected: {total_not_detected}")

    # Compute statistics
    all_widths = detected_widths_ms + not_detected_widths_ms
    stats = {
        "total_pulses": total_pulses,
        "total_detected": total_detected,
        "total_not_detected": total_not_detected,
        "all_widths": {
            "mean": np.mean(all_widths) if all_widths else 0,
            "median": np.median(all_widths) if all_widths else 0,
            "std": np.std(all_widths) if all_widths else 0,
            "min": np.min(all_widths) if all_widths else 0,
            "max": np.max(all_widths) if all_widths else 0,
        },
        "detected_widths": {
            "mean": np.mean(detected_widths_ms) if detected_widths_ms else 0,
            "median": np.median(detected_widths_ms) if detected_widths_ms else 0,
            "std": np.std(detected_widths_ms) if detected_widths_ms else 0,
            "min": np.min(detected_widths_ms) if detected_widths_ms else 0,
            "max": np.max(detected_widths_ms) if detected_widths_ms else 0,
        },
        "not_detected_widths": {
            "mean": np.mean(not_detected_widths_ms) if not_detected_widths_ms else 0,
            "median": np.median(not_detected_widths_ms)
            if not_detected_widths_ms
            else 0,
            "std": np.std(not_detected_widths_ms) if not_detected_widths_ms else 0,
            "min": np.min(not_detected_widths_ms) if not_detected_widths_ms else 0,
            "max": np.max(not_detected_widths_ms) if not_detected_widths_ms else 0,
        },
    }

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- Plot 1: Overlaid histogram of all pulses ---
    ax = axes[0]

    # Create uniform bin edges based on combined data range
    all_widths_combined = detected_widths_ms + not_detected_widths_ms
    if all_widths_combined:
        bin_edges = np.linspace(
            min(all_widths_combined), max(all_widths_combined), bins + 1
        )
    else:
        bin_edges = bins

    ax.hist(
        not_detected_widths_ms,
        bins=bin_edges,
        alpha=0.6,
        label=f"Not detected ({total_not_detected})",
        color="#3498db",
        edgecolor="black",
    )

    ax.hist(
        detected_widths_ms,
        bins=bin_edges,
        alpha=0.6,
        label=f"Detected {DISPLAY_NAME} ({total_detected})",
        color="#e74c3c",
        edgecolor="black",
    )

    ax.set_xlabel("Half-max Width (ms)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("All Pulses - Overlaid Distribution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Add vertical lines for means
    if detected_widths_ms:
        ax.axvline(
            stats["detected_widths"]["mean"],
            color="#e74c3c",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"Detected mean: {stats['detected_widths']['mean']:.2f} ms",
        )

    if not_detected_widths_ms:
        ax.axvline(
            stats["not_detected_widths"]["mean"],
            color="#3498db",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"Not detected mean: {stats['not_detected_widths']['mean']:.2f} ms",
        )

    # --- Plot 2: Side-by-side comparison ---
    ax = axes[1]

    positions = [1, 2]
    widths_data = [not_detected_widths_ms, detected_widths_ms]
    labels = [
        f"Not Detected\n({total_not_detected})",
        f"Detected {DISPLAY_NAME.title()}\n({total_detected})",
    ]
    colors = ["#3498db", "#e74c3c"]

    bp = ax.boxplot(
        widths_data,
        positions=positions,
        labels=labels,
        patch_artist=True,
        widths=0.6,
    )

    # Color the box plots
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add violin plots for more detail
    parts = ax.violinplot(
        widths_data,
        positions=positions,
        showmeans=True,
        showmedians=True,
    )

    for pc in parts["bodies"]:
        pc.set_facecolor("lightgray")
        pc.set_alpha(0.3)

    ax.set_ylabel("Half-max Width (ms)", fontsize=11)
    ax.set_title("Width Distribution Comparison", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Add statistics text box
    stats_text = f"""Statistics:
All pulses (n={total_pulses}):
  Mean: {stats["all_widths"]["mean"]:.2f} ms
  Median: {stats["all_widths"]["median"]:.2f} ms
  Range: [{stats["all_widths"]["min"]:.2f}, {stats["all_widths"]["max"]:.2f}] ms

Detected (n={total_detected}):
  Mean: {stats["detected_widths"]["mean"]:.2f} ms
  Median: {stats["detected_widths"]["median"]:.2f} ms

Not Detected (n={total_not_detected}):
  Mean: {stats["not_detected_widths"]["mean"]:.2f} ms
  Median: {stats["not_detected_widths"]["median"]:.2f} ms"""

    plt.figtext(
        0.98,
        0.97,
        stats_text,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        family="monospace",
    )

    plt.suptitle(
        f"Half-Max Width Histogram - All Pulses from All Files\n({DETECTION_MODE.capitalize()} Detection Mode)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    return fig, axes, stats


def plot_all_detected_pulses_combined(
    data_path, max_per_file=10, figsize_per_plot=(4, 3)
):
    """
    Plot detected pulses from all h5 files in a combined grid visualization.

    Parameters:
    -----------
    data_path : Path or str
        Path to directory containing h5 files
    max_per_file : int
        Maximum number of pulses to plot per file (randomly sampled if more exist)
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
    total_detected = 0

    con.log("\n" + "=" * 60)
    con.log(f"Creating combined {DISPLAY_NAME} visualization")
    con.log("=" * 60 + "\n")

    for file_path in path_list:
        file = nixio.File.open(str(file_path), nixio.FileMode.ReadOnly)

        try:
            block = file.blocks["pulses"]
            data_array_names = [da.name for da in block.data_arrays]

            if ARRAY_NAME not in data_array_names:
                con.log(f"⊘ {file_path.name}: No '{ARRAY_NAME}' array")
                continue

            # Load data
            detection_array = block.data_arrays[ARRAY_NAME][:]
            raw_pulses = block.data_arrays["raw_pulses"]
            fs = file.sections["pulses_metadata"]["metadata"]["samplerate"]

            # Get indices of detected pulses
            detected_indices = np.where(detection_array == 1)[0]
            num_detected = len(detected_indices)

            if num_detected == 0:
                con.log(f"  {file_path.name}: No {DISPLAY_NAME}s found")
                continue

            con.log(f"  {file_path.name}: Found {num_detected} {DISPLAY_NAME}s")

            # Randomly sample if too many
            if num_detected > max_per_file:
                selected_indices = np.random.choice(
                    detected_indices, size=max_per_file, replace=False
                )
                con.log(f"    → Randomly sampling {max_per_file} for visualization")
            else:
                selected_indices = detected_indices

            num_to_plot = len(selected_indices)
            total_detected += num_to_plot

            # Create subplots grid
            n_cols = min(4, num_to_plot)  # Max 4 columns
            n_rows = (num_to_plot + n_cols - 1) // n_cols

            figsize = (figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

            if num_to_plot == 1:
                axes = np.array([axes])
            else:
                axes = axes.flatten()

            # Plot each sampled pulse
            for plot_idx, pulse_idx in enumerate(selected_indices):
                ax = axes[plot_idx]
                pulse_data = raw_pulses[pulse_idx]

                # Get representative waveform
                pulse_mean, best_channel = get_representative_waveform(pulse_data)

                # Plot waveform
                time_axis = np.arange(len(pulse_mean)) / fs
                ax.plot(
                    time_axis, pulse_mean, linewidth=1.5, color="steelblue", alpha=0.8
                )
                ax.fill_between(time_axis, pulse_mean, alpha=0.3, color="steelblue")

                # Use same logic as detection function
                is_positive, info = detect_pulse(pulse_data, fs)

                # Mode specific markers
                if DETECTION_MODE == "wide" and is_positive:
                    ax.axhline(
                        info["half_height"],
                        color="orange",
                        linestyle="--",
                        alpha=0.7,
                    )

                    ax.axvspan(
                        info["left_idx"] / fs,
                        info["right_idx"] / fs,
                        color="orange",
                        alpha=0.2,
                        label=f"Width = {info['width_samples'] / fs * 1000:.2f} ms",
                    )

                elif DETECTION_MODE == "double" and is_positive:
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
                        info.get("reason", "not detected"),
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
                f"{DISPLAY_NAME.capitalize()}s - {file_path.name}\n({num_to_plot} of {num_detected} {DISPLAY_NAME}s)",
                fontsize=12,
                fontweight="bold",
                y=0.995,
            )
            plt.tight_layout()

            all_plots[file_path.name] = (fig, axes)

        finally:
            file.close()

    con.log(f"\n✓ Total {DISPLAY_NAME}s to visualize: {total_detected}")
    con.log(f"✓ Created {len(all_plots)} figures")

    return all_plots


############################################
############# INTERACTIVE VERIFICATION #####
############################################


def interactive_pulse_verification(data_path):
    """
    Interactive matplotlib-based verification tool for pulse classifications.

    Displays each detected pulse waveform in a matplotlib window and allows verification
    using keyboard input (Y to confirm, N to reject) without switching focus away from
    the plot window. Changes are saved to h5 files after all files have been reviewed.

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

    # State management
    state = {
        "current_pulse_idx": 0,
        "changes": {},  # {file_path: {pulse_idx: new_label}}
        "total_verified": 0,
        "total_corrected": 0,
        "quit_requested": False,
        "all_detected_pulses": [],
        "current_fig": None,
    }

    # Pre-load all detected pulses from all files
    con.log(f"\n{'=' * 60}")
    con.log(f"INTERACTIVE {DISPLAY_NAME.upper()} VERIFICATION")
    con.log(f"{'=' * 60}\n")
    con.log(f"Loading {DISPLAY_NAME}s from all files...")

    for file_idx, file_path in enumerate(path_list, 1):
        con.log(f"  [{file_idx}/{len(path_list)}] {file_path.name}", end="")

        file = nixio.File.open(str(file_path), nixio.FileMode.ReadOnly)

        try:
            block = file.blocks["pulses"]
            data_array_names = [da.name for da in block.data_arrays]

            if ARRAY_NAME not in data_array_names:
                con.log(f" - No {ARRAY_NAME} array. Skipping.")
                continue

            detection_array = block.data_arrays[ARRAY_NAME][:]
            raw_pulses = block.data_arrays["raw_pulses"]
            fs = file.sections["pulses_metadata"]["metadata"]["samplerate"]

            detected_indices = np.where(detection_array == 1)[0]
            num_detected = len(detected_indices)

            con.log(f" - Found {num_detected} {DISPLAY_NAME}s")

            # Initialize changes dict for this file
            if str(file_path) not in state["changes"]:
                state["changes"][str(file_path)] = {}

            # Load pulse data
            for pulse_idx in detected_indices:
                pulse_data = raw_pulses[pulse_idx]

                is_positive, info = detect_pulse(pulse_data, fs)

                pulse_mean, best_channel = get_representative_waveform(pulse_data)

                state["all_detected_pulses"].append(
                    {
                        "file_path": file_path,
                        "file_path_str": str(file_path),
                        "pulse_data": pulse_data,
                        "pulse_idx": pulse_idx,
                        "fs": fs,
                        "is_positive": is_positive,
                        "info": info,
                        "best_channel": best_channel,
                        "pulse_mean": pulse_mean,
                    }
                )

        finally:
            file.close()

    total_pulses = len(state["all_detected_pulses"])

    if total_pulses == 0:
        con.log(f"\n⚠ No {DISPLAY_NAME}s found to verify.")
        return {}

    con.log(f"\n✓ Loaded {total_pulses} {DISPLAY_NAME}s to verify")

    con.log("\nInstructions (use keys while plot window is active):")
    con.log(f"  [Y] - Confirm {DISPLAY_NAME} (keep as 1)")
    con.log(f"  [N] - Reject {DISPLAY_NAME} (change to 0)")
    con.log("  [Q] - Quit without saving")
    con.log("\nStarting verification...\n")

    def save_changes():
        """Save all changes to h5 files."""

        con.log(f"\n{'=' * 60}")
        con.log("SAVING CHANGES...")
        con.log(f"{'=' * 60}\n")

        for file_path_str, pulse_changes in state["changes"].items():
            if not pulse_changes:
                continue

            file_path = Path(file_path_str)

            con.log(f"  Updating {file_path.name}...")

            file = nixio.File.open(
                str(file_path),
                nixio.FileMode.ReadWrite,
            )

            try:
                block = file.blocks["pulses"]

                detection_array = block.data_arrays[ARRAY_NAME][:]

                # Apply changes
                for pulse_idx, new_value in pulse_changes.items():
                    detection_array[pulse_idx] = new_value

                # Write back to file
                block.data_arrays[ARRAY_NAME][:] = detection_array

                con.log(f"    ✓ Saved {len(pulse_changes)} changes")

            finally:
                file.close()

        # Print summary
        con.log(f"\n{'=' * 60}")
        con.log("VERIFICATION COMPLETE")
        con.log(f"{'=' * 60}")

        con.log(f"Total pulses verified: {state['total_verified']}")
        con.log(f"Total corrections made: {state['total_corrected']}")
        con.log(f"Total reviewed: {state['total_verified'] + state['total_corrected']}")

        con.log(
            f"Total changes saved: {sum(len(c) for c in state['changes'].values())}"
        )

    def display_next_pulse():
        """Display the next pulse for verification."""

        if state["quit_requested"] or state["current_pulse_idx"] >= total_pulses:
            return

        pulse_data = state["all_detected_pulses"][state["current_pulse_idx"]]
        pulse_data = state["all_detected_pulses"][state["current_pulse_idx"]]

        fig, ax = plt.subplots(figsize=(12, 5))

        state["current_fig"] = fig

        # Plot waveform
        time_axis = np.arange(len(pulse_data["pulse_mean"])) / pulse_data["fs"]

        ax.plot(
            time_axis,
            pulse_data["pulse_data"],
            linewidth=2.5,
            color="steelblue",
            alpha=0.85,
            label="Waveform",
        )

        # ax.fill_between(
        #     time_axis,
        #     pulse_data["pulse_mean"],
        #     alpha=0.25,
        #     color="steelblue",
        # )

        # DOUBLE PEAK VISUALIZATION
        if DETECTION_MODE == "double":
            if pulse_data["is_positive"] and "peaks" in pulse_data["info"]:
                peaks = pulse_data["info"]["peaks"]

                peak_times = np.array(peaks) / pulse_data["fs"]

                peak_vals = pulse_data["pulse_mean"][peaks]

                ax.scatter(
                    peak_times,
                    peak_vals,
                    color="red",
                    s=120,
                    marker="*",
                    zorder=5,
                    label="Detected peaks",
                )

        # WIDE PULSE VISUALIZATION
        elif DETECTION_MODE == "wide":
            info = pulse_data["info"]

            if pulse_data["is_positive"] and "half_height" in info:
                left_t = info["left_idx"] / pulse_data["fs"]
                right_t = info["right_idx"] / pulse_data["fs"]

                width_ms = info["width_samples"] / pulse_data["fs"] * 1000

                ax.axhline(
                    info["half_height"],
                    color="orange",
                    linestyle="--",
                    alpha=0.8,
                    label="Half height",
                )

                ax.axvspan(
                    left_t,
                    right_t,
                    color="orange",
                    alpha=0.25,
                    label=f"Width = {width_ms:.2f} ms",
                )

        # Title with instructions
        file_name = pulse_data["file_path"].name

        pulse_num = state["current_pulse_idx"] + 1

        title = (
            f"{DISPLAY_NAME.capitalize()} verification | "
            f"Pulse {pulse_data['pulse_idx']} "
            f"[{pulse_num}/{total_pulses}] - {file_name}\n"
            f"Channel {pulse_data['best_channel']} (strongest) | "
            f"Press [Y]es / [N]o / [Q]uit"
        )

        ax.set_title(
            title,
            fontsize=12,
            fontweight="bold",
            pad=15,
        )

        ax.set_xlabel("Time (s)", fontsize=11)

        ax.set_ylabel("Amplitude (a.u.)", fontsize=11)

        ax.grid(True, alpha=0.3, linestyle="--")

        ax.legend(fontsize=10, loc="upper right")

        plt.tight_layout()

        # Register key handler and show
        fig.canvas.mpl_connect(
            "key_press_event",
            on_key_press,
        )

        plt.show(block=False)

    def on_key_press(event):
        """Handle keyboard input for verification."""

        if event.key is None or state["quit_requested"]:
            return

        key = event.key.upper()

        # CONFIRM
        if key == "Y":
            pulse_data = state["all_detected_pulses"][state["current_pulse_idx"]]

            state["changes"][pulse_data["file_path_str"]][pulse_data["pulse_idx"]] = 1

            state["total_verified"] += 1

            con.log(
                f"  ✓ "
                f"[{state['current_pulse_idx'] + 1}/{total_pulses}] "
                f"Confirmed {DISPLAY_NAME} "
                f"(pulse {pulse_data['pulse_idx']})"
            )

            state["current_pulse_idx"] += 1

            if state["current_pulse_idx"] >= total_pulses:
                plt.close("all")

                save_changes()

            else:
                plt.close(state["current_fig"])

                display_next_pulse()

        # REJECT
        elif key == "N":
            pulse_data = state["all_detected_pulses"][state["current_pulse_idx"]]

            state["changes"][pulse_data["file_path_str"]][pulse_data["pulse_idx"]] = 0

            state["total_corrected"] += 1

            con.log(
                f"  ✗ "
                f"[{state['current_pulse_idx'] + 1}/{total_pulses}] "
                f"Marked as NOT {DISPLAY_NAME} "
                f"(pulse {pulse_data['pulse_idx']})"
            )

            state["current_pulse_idx"] += 1

            if state["current_pulse_idx"] >= total_pulses:
                plt.close("all")

                save_changes()

            else:
                plt.close(state["current_fig"])

                display_next_pulse()

        # QUIT
        elif key == "Q":
            con.log("\n⚠ Quit requested. Closing without saving...")

            state["quit_requested"] = True

            plt.close("all")

    # START VERIFICATION
    try:
        display_next_pulse()

        plt.show(block=True)

        if not state["quit_requested"]:
            return {
                "status": "completed",
                "verified": state["total_verified"],
                "corrected": state["total_corrected"],
                "total_reviewed": (state["total_verified"] + state["total_corrected"]),
            }

        else:
            return {
                "status": "cancelled",
                "message": "Verification cancelled by user",
            }

    except Exception as e:
        con.log(f"\n⚠ Error during verification: {e}")

        import traceback

        traceback.print_exc()

        return {
            "status": "error",
            "message": str(e),
        }


if __name__ == "__main__":
    data_path = H5_DIR

    # Choose visualization mode
    con.log("\n" + "=" * 60)
    con.log("PULSE VISUALIZATION OPTIONS")
    con.log("=" * 60)
    con.log("1. Plot all detected pulses (combined from all files)")
    con.log("2. Interactive verification (review and correct detections)")
    con.log("3. Histogram of half-widths for ALL pulses")
    con.log("=" * 60)

    mode = input("Select mode (1, 2, or 3): ").strip()

    if mode == "1":
        # Plot all detected pulses from all files
        con.log("\nGenerating combined visualizations...")
        all_plots = plot_all_detected_pulses_combined(data_path, max_per_file=10)
        if all_plots:
            plt.show()
        else:
            con.log("No detections found to visualize.")

    elif mode == "2":
        # Run interactive verification
        con.log("\nStarting interactive verification...")
        verification_results = interactive_pulse_verification(data_path)
        if verification_results:
            con.log(f"Verification results: {verification_results}")

    elif mode == "3":
        # Plot histogram of all pulse widths
        con.log("\nGenerating width histogram for all pulses...")
        fig, axes, stats = plot_all_pulses_width_histogram(data_path)
        if fig:
            con.log("\n✓ Width statistics computed:")
            con.log(f"  Total pulses: {stats['total_pulses']}")
            con.log(
                f"  Detected: {stats['total_detected']} ({stats['total_detected'] / stats['total_pulses'] * 100:.1f}%)"
            )
            con.log(
                f"  Not detected: {stats['total_not_detected']} ({stats['total_not_detected'] / stats['total_pulses'] * 100:.1f}%)"
            )
            con.log(
                f"\n  All pulses width - Mean: {stats['all_widths']['mean']:.2f} ms, Median: {stats['all_widths']['median']:.2f} ms"
            )
            con.log(
                f"  Detected widths - Mean: {stats['detected_widths']['mean']:.2f} ms, Median: {stats['detected_widths']['median']:.2f} ms"
            )
            con.log(
                f"  Not detected widths - Mean: {stats['not_detected_widths']['mean']:.2f} ms, Median: {stats['not_detected_widths']['median']:.2f} ms"
            )
            plt.show()
        else:
            con.log("Failed to generate histogram.")

    else:
        con.log("Invalid selection. Please enter 1, 2, or 3.")
