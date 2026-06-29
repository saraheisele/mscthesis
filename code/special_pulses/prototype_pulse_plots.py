"""Plot prototype pulse waveforms for each pulse shape.

Analysis part: special-pulse visualization (Part 2c of Berlin activity analysis).
Dependencies: double_peaks_detection, data_paths, h5_io.

Randomly samples pulses, aligns waveforms at shape-specific reference points,
and overlays individual traces with their mean for normal/double/wide/fat types.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from path_setup import setup_script_paths

setup_script_paths(__file__)

import json

import matplotlib.pyplot as plt
import nixio
import numpy as np
from rich.console import Console
from scipy import stats
from scipy.signal import find_peaks

from data_paths import H5_DIR, PULSE_SHAPE_PROTOTYPES_DIR
from double_peaks_detection import (
    MIN_AMPLITUDE_THRESHOLD,
    compute_half_max_width,
    detect_double_pulse,
    detect_fat_pulse,
    detect_wide_pulse,
)
from h5_io import get_path_list, get_pulse_block, open_h5
from pulse_shape_metrics import (
    double_pulse_metrics,
    paired_symmetry_test,
    symmetry_at_fraction,
)

console = Console()

OUTPUT_DIR = PULSE_SHAPE_PROTOTYPES_DIR
SAMPLE_SIZE = 100
RANDOM_SEED = 42

CLIP_RATIO = 0.995
CLIP_MIN_CONSECUTIVE = 8

PULSE_SHAPES = {
    "normal": {
        "label": "Normal pulse",
        "array_name": None,
        "color": "#2980b9",
        "align": "maximum",
        "detect": None,
        "criteria": [
            "Predicted-positive pulse",
            "Not classified as double, wide, or fat",
            "Single narrow EOD waveform",
        ],
    },
    "double": {
        "label": "Double pulse",
        "array_name": "is_double_peak",
        "color": "#c0392b",
        "align": "valley",
        "detect": detect_double_pulse,
        "criteria": [
            "Amplitude ≥ {:.1f} (baseline-corrected)".format(MIN_AMPLITUDE_THRESHOLD),
            "Exactly 2 peaks, prominence ≥ 10% of max",
            "Peak separation: 0.5–2.0 ms",
            "Valley between peaks: 40–95% of higher peak",
            "Peak amplitudes within 60% of each other",
            "One peak is the global maximum",
        ],
    },
    "wide": {
        "label": "Wide pulse",
        "array_name": "is_wide_pulse",
        "color": "#e67e22",
        "align": "maximum",
        "detect": lambda wf, fs: detect_wide_pulse(wf, fs, check_shape=True),
        "criteria": [
            "Amplitude ≥ {:.1f} (baseline-corrected)".format(MIN_AMPLITUDE_THRESHOLD),
            "Half-max width: 2.3–4.0 ms",
            "Single prominent peak (3 ms isolation window)",
            "Prominence ≥ 10% of peak amplitude",
            "Gaussian rise + exponential decay shape",
            "Excludes double pulses",
        ],
    },
    "fat": {
        "label": "Fat pulse",
        "array_name": "is_fat_pulse",
        "color": "#8e44ad",
        "align": "maximum",
        "detect": detect_fat_pulse,
        "criteria": [
            "Amplitude ≥ {:.1f} (baseline-corrected)".format(MIN_AMPLITUDE_THRESHOLD),
            "Half-max width ≥ 4.0 ms (no upper limit)",
            "Single prominent peak (3 ms isolation window)",
            "Prominence ≥ 10% of peak amplitude",
            "Excludes double and wide pulses",
            "No Gaussian/exponential shape constraint",
        ],
    },
}


def longest_run_at_peak(trace, ratio=CLIP_RATIO):
    abs_trace = np.abs(trace)
    peak = np.max(abs_trace)
    if peak == 0:
        return 0

    mask = abs_trace >= ratio * peak
    max_run = current = 0
    for value in mask:
        current = current + 1 if value else 0
        max_run = max(max_run, current)
    return max_run


def is_channel_clipped(trace, min_consecutive=CLIP_MIN_CONSECUTIVE, ratio=CLIP_RATIO):
    return longest_run_at_peak(trace, ratio=ratio) >= min_consecutive


def get_biggest_unclipped_waveform(pulse_waveform):
    """
    Return the waveform from the strongest channel that does not show clipping.

    Clipping is detected as a flat top: >= CLIP_MIN_CONSECUTIVE consecutive
    samples within CLIP_RATIO of the channel peak.
    """
    channel_strengths = np.max(np.abs(pulse_waveform), axis=0)
    channel_order = np.argsort(channel_strengths)[::-1]

    for channel_idx in channel_order:
        trace = pulse_waveform[:, channel_idx]
        if not is_channel_clipped(trace):
            trace = trace.copy()
            if abs(np.min(trace)) > np.max(trace):
                trace *= -1
            return trace, int(channel_idx)

    best_channel = int(channel_order[0])
    trace = pulse_waveform[:, best_channel].copy()
    if abs(np.min(trace)) > np.max(trace):
        trace *= -1
    return trace, best_channel


def baseline_correct(trace):
    baseline_window = max(1, len(trace) // 5)
    baseline = np.median(trace[:baseline_window])
    return trace - baseline


def normalize_trace(trace):
    peak = np.max(trace)
    if peak <= 0:
        return trace
    return trace / peak


def shift_waveform(trace, shift):
    shifted = np.zeros_like(trace)
    if shift > 0:
        shifted[shift:] = trace[:-shift]
    elif shift < 0:
        shifted[:shift] = trace[-shift:]
    else:
        shifted = trace.copy()
    return shifted


def double_peak_indices(trace, fs):
    is_positive, info = detect_double_pulse(trace[:, np.newaxis], fs)
    if is_positive and "peaks" in info:
        return np.sort(info["peaks"])

    prominence = 0.05 * np.max(trace)
    peaks, _ = find_peaks(trace, prominence=prominence)
    if len(peaks) < 2:
        return None

    min_sep = int(0.0005 * fs)
    max_sep = int(0.002 * fs)
    best_pair = None
    best_score = -1.0
    for i, p1 in enumerate(peaks):
        for p2 in peaks[i + 1 :]:
            separation = p2 - p1
            if min_sep <= separation <= max_sep:
                score = trace[p1] + trace[p2]
                if score > best_score:
                    best_score = score
                    best_pair = (p1, p2)

    if best_pair is not None:
        return np.asarray(best_pair)

    return np.sort(peaks[np.argsort(trace[peaks])[-2:]])


def double_valley_index(trace, fs):
    peaks = double_peak_indices(trace, fs)
    if peaks is None:
        return int(np.argmax(trace))

    p1, p2 = peaks
    return int(p1 + np.argmin(trace[p1 : p2 + 1]))


def alignment_reference_index(trace, align_mode, fs):
    if align_mode == "valley":
        return double_valley_index(trace, fs)
    return int(np.argmax(trace))


def align_waveforms(
    normalized_waveforms, corrected_waveforms, align_mode, fs
):
    center = len(normalized_waveforms[0]) // 2
    return np.asarray(
        [
            shift_waveform(
                normalized_trace,
                center
                - alignment_reference_index(corrected_trace, align_mode, fs),
            )
            for normalized_trace, corrected_trace in zip(
                normalized_waveforms, corrected_waveforms
            )
        ]
    )


def expand_marker(marker, candidate_indices, num_pulses):
    marker = np.asarray(marker, dtype=np.int64)
    if len(marker) == num_pulses:
        return marker

    full_marker = np.zeros(num_pulses, dtype=np.int64)
    full_marker[candidate_indices] = marker
    return full_marker


def collect_pulse_indices(data_path, array_name):
    entries = []
    for file_path in get_path_list(Path(data_path)):
        file = open_h5(file_path, nixio.FileMode.ReadOnly)
        if file is None:
            continue
        try:
            block = get_pulse_block(file)
            data_array_names = [da.name for da in block.data_arrays]
            if array_name not in data_array_names:
                continue

            marker = block.data_arrays[array_name][:]
            detected_indices = np.where(marker == 1)[0]
            for pulse_idx in detected_indices:
                entries.append((file_path, int(pulse_idx)))
        finally:
            file.close()
    return entries


def collect_normal_pulse_indices(data_path):
    entries = []
    for file_path in get_path_list(Path(data_path)):
        file = open_h5(file_path, nixio.FileMode.ReadOnly)
        if file is None:
            continue
        try:
            block = get_pulse_block(file)
            data_array_names = [da.name for da in block.data_arrays]

            if "predicted_labels" not in data_array_names:
                continue

            raw_pulses = block.data_arrays["raw_pulses"]
            num_pulses = len(raw_pulses)
            predicted_labels = block.data_arrays["predicted_labels"][:]
            candidate_indices = np.where(predicted_labels == 1)[0]

            if len(candidate_indices) == 0:
                continue

            marker_names = ("is_double_peak", "is_wide_pulse", "is_fat_pulse")
            if not all(name in data_array_names for name in marker_names):
                continue

            double_marker = expand_marker(
                block.data_arrays["is_double_peak"][:], candidate_indices, num_pulses
            )
            wide_marker = expand_marker(
                block.data_arrays["is_wide_pulse"][:], candidate_indices, num_pulses
            )
            fat_marker = expand_marker(
                block.data_arrays["is_fat_pulse"][:], candidate_indices, num_pulses
            )

            for pulse_idx in candidate_indices:
                if (
                    double_marker[pulse_idx] == 0
                    and wide_marker[pulse_idx] == 0
                    and fat_marker[pulse_idx] == 0
                ):
                    entries.append((file_path, int(pulse_idx)))
        finally:
            file.close()
    return entries


def load_sampled_waveforms(entries, sample_size, random_seed, align_mode=None):
    if not entries:
        return [], [], None

    rng = np.random.default_rng(random_seed)
    shuffled_entries = list(entries)
    rng.shuffle(shuffled_entries)

    corrected_waveforms = []
    normalized_waveforms = []
    fs = None

    for file_path, pulse_idx in shuffled_entries:
        if len(normalized_waveforms) >= sample_size:
            break

        file = open_h5(file_path, nixio.FileMode.ReadOnly)
        if file is None:
            continue
        try:
            block = get_pulse_block(file)
            pulse_data = block.data_arrays["raw_pulses"][pulse_idx][:]
            if fs is None:
                fs = float(file.sections["pulses_metadata"]["metadata"]["samplerate"])

            trace, _ = get_biggest_unclipped_waveform(pulse_data)
            corrected = baseline_correct(trace)

            if align_mode == "valley" and double_peak_indices(corrected, fs) is None:
                continue

            corrected_waveforms.append(corrected)
            normalized_waveforms.append(normalize_trace(corrected))
        finally:
            file.close()

    return corrected_waveforms, normalized_waveforms, fs


def add_half_max_markers(ax, mean_trace, fs, color):
    width_sec, width_info = compute_half_max_width(mean_trace, fs)
    left_t = width_info["left_idx"] / fs * 1000
    right_t = width_info["right_idx"] / fs * 1000
    ax.axhline(
        width_info["half_height"],
        color=color,
        linestyle="--",
        alpha=0.8,
        linewidth=1.2,
        label="Half maximum",
    )
    ax.axvspan(
        left_t,
        right_t,
        color=color,
        alpha=0.15,
        label=f"Width = {width_sec * 1000:.2f} ms",
    )


def add_detection_markers(ax, mean_trace, fs, pulse_shape):
    color = pulse_shape["color"]
    center_idx = len(mean_trace) // 2
    center_t = center_idx / fs * 1000

    if pulse_shape["align"] == "valley":
        peaks = double_peak_indices(mean_trace, fs)
        if peaks is not None:
            peak_times = peaks / fs * 1000
            ax.scatter(
                peak_times,
                mean_trace[peaks],
                color=color,
                s=90,
                marker="*",
                zorder=6,
                label="Detected peaks",
            )
            ax.scatter(
                center_t,
                mean_trace[center_idx],
                color=color,
                s=70,
                marker="v",
                zorder=6,
                label="Inter-peak trough",
            )
            dt_ms = (peaks[1] - peaks[0]) / fs * 1000
            ax.annotate(
                f"Δt = {dt_ms:.2f} ms",
                xy=(np.mean(peak_times), np.mean(mean_trace[peaks])),
                xytext=(8, 12),
                textcoords="offset points",
                fontsize=9,
                color=color,
            )
        add_half_max_markers(ax, mean_trace, fs, color)
        return

    if pulse_shape["align"] == "maximum":
        ax.scatter(
            center_t,
            mean_trace[center_idx],
            color=color,
            s=90,
            marker="*",
            zorder=6,
            label="Aligned maximum",
        )

    if pulse_shape["array_name"] in {"is_wide_pulse", "is_fat_pulse"}:
        add_half_max_markers(ax, mean_trace, fs, color)
        return

    add_half_max_markers(ax, mean_trace, fs, color)


def load_all_waveforms(entries, align_mode=None, max_waveforms=None):
    """Load waveforms for mean computation (optionally capped for memory)."""
    corrected_waveforms = []
    normalized_waveforms = []
    fs = None

    for file_path, pulse_idx in entries:
        if max_waveforms is not None and len(normalized_waveforms) >= max_waveforms:
            break
        file = open_h5(file_path, nixio.FileMode.ReadOnly)
        if file is None:
            continue
        try:
            block = get_pulse_block(file)
            pulse_data = block.data_arrays["raw_pulses"][pulse_idx][:]
            if fs is None:
                fs = float(file.sections["pulses_metadata"]["metadata"]["samplerate"])
            trace, _ = get_biggest_unclipped_waveform(pulse_data)
            corrected = baseline_correct(trace)
            if align_mode == "valley" and double_peak_indices(corrected, fs) is None:
                continue
            corrected_waveforms.append(corrected)
            normalized_waveforms.append(normalize_trace(corrected))
        finally:
            file.close()

    return corrected_waveforms, normalized_waveforms, fs


def plot_prototype_pulse_shape(
    pulse_key,
    pulse_shape,
    corrected_waveforms,
    normalized_waveforms,
    fs,
    output_dir,
    mean_trace_all=None,
    total_count=None,
):
    """Plot sampled pulses with mean computed from all waveforms when provided."""
    if not normalized_waveforms:
        console.log(f"[yellow]No pulses found for {pulse_shape['label']}. Skipping.")
        return None

    aligned_sample = align_waveforms(
        normalized_waveforms, corrected_waveforms, pulse_shape["align"], fs
    )
    if mean_trace_all is not None:
        mean_trace = mean_trace_all
        mean_label = f"Mean (n={total_count:,})"
    else:
        mean_trace = np.mean(aligned_sample, axis=0)
        mean_label = f"Mean (n={len(aligned_sample)})"

    time_ms = np.arange(aligned_sample.shape[1]) / fs * 1000

    fig, ax = plt.subplots(figsize=(11, 6))

    for trace in aligned_sample:
        ax.plot(time_ms, trace, color="gray", alpha=0.25, linewidth=0.8)

    ax.plot(
        time_ms,
        mean_trace,
        color=pulse_shape["color"],
        linewidth=2.8,
        label=mean_label,
        zorder=5,
    )

    add_detection_markers(ax, mean_trace, fs, pulse_shape)

    criteria_text = "\n".join(f"• {line}" for line in pulse_shape["criteria"])
    ax.text(
        0.98,
        0.97,
        criteria_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="#cccccc"),
    )

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Normalized amplitude")
    if np.min(aligned_sample) < -1:
        ax.set_ylim(-1, None)
    ax.set_title(
        f"Prototype {pulse_shape['label'].lower()}s "
        f"(showing {len(aligned_sample)} of {total_count or len(aligned_sample):,})",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"prototype_{pulse_key}_pulse.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    console.log(f"Saved {output_path}")
    return output_path


def plot_symmetry_analysis(
    pulse_key, pulse_shape, corrected_waveforms, fs, output_dir, fraction=0.1
):
    """Whisker plot and paired t-test for left/right width at fraction of peak."""
    if pulse_key not in {"normal", "wide", "fat"}:
        return

    left_widths = []
    right_widths = []
    for trace in corrected_waveforms:
        sym = symmetry_at_fraction(trace, fs, fraction=fraction)
        if np.isnan(sym["left_width_sec"]) or np.isnan(sym["right_width_sec"]):
            continue
        left_widths.append(sym["left_width_sec"] * 1000)
        right_widths.append(sym["right_width_sec"] * 1000)

    if len(left_widths) < 3:
        return

    left = np.asarray(left_widths)
    right = np.asarray(right_widths)
    test = paired_symmetry_test(left / 1000, right / 1000)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(
        [left, right],
        tick_labels=[f"Left @ {int(fraction * 100)}%", f"Right @ {int(fraction * 100)}%"],
        patch_artist=True,
    )
    ax.set_ylabel("Width (ms)")
    ax.set_title(
        f"{pulse_shape['label']} symmetry at {int(fraction * 100)}% amplitude\n"
        f"Paired t-test: t={test['t_stat']:.3f}, p={test['p_value']:.2e} (n={test['n']})"
    )
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = output_dir / f"symmetry_{pulse_key}_10pct.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    console.log(f"Saved {out}")

    stats_path = output_dir / f"symmetry_{pulse_key}_10pct_stats.json"
    with open(stats_path, "w") as handle:
        json.dump(
            {
                "pulse_shape": pulse_key,
                "fraction": fraction,
                "n": test["n"],
                "mean_left_ms": float(np.mean(left)),
                "mean_right_ms": float(np.mean(right)),
                "paired_ttest": test,
            },
            handle,
            indent=2,
        )


def plot_normal_double_overlay(
    normal_mean,
    double_mean,
    fs,
    output_dir,
):
    """Overlay mean double pulse with two mean normal pulses (peak-aligned to each peak)."""
    dp_peaks = double_peak_indices(double_mean, fs)
    if dp_peaks is None:
        return

    normal_peak = int(np.argmax(normal_mean))
    p1, p2 = dp_peaks

    norm1 = shift_waveform(normal_mean, p1 - normal_peak)
    norm2 = shift_waveform(normal_mean, p2 - normal_peak)

    time_ms = np.arange(len(double_mean)) / fs * 1000
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        time_ms,
        double_mean,
        color=PULSE_SHAPES["double"]["color"],
        linewidth=2.5,
        label="Mean double pulse",
        zorder=4,
    )
    ax.plot(
        time_ms,
        norm1,
        color=PULSE_SHAPES["normal"]["color"],
        linewidth=2,
        linestyle="--",
        label="Mean normal pulse (aligned to 1st peak)",
        zorder=3,
    )
    ax.plot(
        time_ms,
        norm2,
        color=PULSE_SHAPES["normal"]["color"],
        linewidth=2,
        linestyle=":",
        label="Mean normal pulse (aligned to 2nd peak)",
        zorder=3,
    )
    ax.scatter(
        dp_peaks / fs * 1000,
        double_mean[dp_peaks],
        color=PULSE_SHAPES["double"]["color"],
        s=80,
        marker="*",
        zorder=5,
        label="Double peaks",
    )
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Normalized amplitude")
    ax.set_title("Mean double pulse vs two mean normal pulses (peak-aligned)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = output_dir / "normal_pulses_aligned_to_double.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    console.log(f"Saved {out}")


def save_double_peak_separation_stats(corrected_waveforms, fs, output_dir):
    separations_ms = []
    for trace in corrected_waveforms:
        metrics = double_pulse_metrics(trace, fs)
        if not np.isnan(metrics["peak_separation_ms"]):
            separations_ms.append(metrics["peak_separation_ms"])
    if not separations_ms:
        return
    arr = np.asarray(separations_ms)
    stats_dict = {
        "n": int(arr.size),
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "std_ms": float(np.std(arr)),
        "mean_sec": float(np.mean(arr) / 1000),
    }
    with open(output_dir / "double_peak_separation_stats.json", "w") as handle:
        json.dump(stats_dict, handle, indent=2)
    console.log(
        f"  Double peak separation: mean={stats_dict['mean_ms']:.3f} ms "
        f"(n={stats_dict['n']:,})"
    )


def main(data_path=H5_DIR, sample_size=SAMPLE_SIZE, random_seed=RANDOM_SEED):
    console.log("Collecting and plotting prototype pulses for each pulse shape...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pulse_counts = {}
    mean_traces = {}

    for pulse_key, pulse_shape in PULSE_SHAPES.items():
        console.log(f"\n{pulse_shape['label']}:")
        if pulse_key == "normal":
            entries = collect_normal_pulse_indices(data_path)
        else:
            entries = collect_pulse_indices(data_path, pulse_shape["array_name"])
        total_count = len(entries)
        pulse_counts[pulse_key] = total_count
        console.log(f"  Found {total_count} detected pulses")

        all_corrected, all_normalized, fs = load_all_waveforms(
            entries, align_mode=pulse_shape["align"]
        )
        if not all_normalized:
            continue

        all_aligned = align_waveforms(
            all_normalized, all_corrected, pulse_shape["align"], fs
        )
        mean_all = np.mean(all_aligned, axis=0)
        mean_traces[pulse_key] = (mean_all, fs, all_corrected)

        rng = np.random.default_rng(random_seed)
        sample_idx = rng.choice(len(all_normalized), size=min(sample_size, len(all_normalized)), replace=False)
        sample_corrected = [all_corrected[i] for i in sample_idx]
        sample_normalized = [all_normalized[i] for i in sample_idx]

        plot_prototype_pulse_shape(
            pulse_key,
            pulse_shape,
            sample_corrected,
            sample_normalized,
            fs,
            OUTPUT_DIR,
            mean_trace_all=mean_all,
            total_count=total_count,
        )

        if pulse_key == "double":
            save_double_peak_separation_stats(all_corrected, fs, OUTPUT_DIR)
        if pulse_key in {"normal", "wide", "fat"}:
            plot_symmetry_analysis(
                pulse_key, pulse_shape, all_corrected, fs, OUTPUT_DIR
            )

    with open(OUTPUT_DIR / "pulse_shape_counts.json", "w") as handle:
        json.dump(pulse_counts, handle, indent=2)

    if "normal" in mean_traces and "double" in mean_traces:
        normal_mean, fs_n, _ = mean_traces["normal"]
        double_mean, fs_d, _ = mean_traces["double"]
        if fs_n == fs_d:
            plot_normal_double_overlay(normal_mean, double_mean, fs_n, OUTPUT_DIR)


if __name__ == "__main__":
    main()
