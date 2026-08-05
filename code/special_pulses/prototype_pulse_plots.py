"""Plot prototype pulse waveforms for each pulse shape.

Analysis part: special-pulse visualization (Part 2c of Berlin activity analysis).
Dependencies: double_peaks_detection, data_paths, h5_io, pulse_shape_metrics.

Randomly samples up to MAX_WAVEFORMS_PER_CLASS pulses per shape for the
prototype median, aligns at shape-specific reference points, and overlays
SAMPLE_SIZE individual traces drawn from that capped set. Uses existing
``special_pulse_class`` labels (classifier apply is off by default).

Aligned mean/median waveforms are saved under PULSE_SHAPE_PROTOTYPES_DIR as
``prototype_<shape>_mean.npz`` for reuse by downstream analyses. Displayed
prototypes and companion panels use the median by default.
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
from presentation_style import LEGEND_LOC, apply_presentation_style, pulse_shape_color, save_thesis_figure
from double_peaks_detection import (
    MULTICLASS_ARRAY_NAME,
    SPECIAL_PULSE_CLASSES,
    apply_special_pulse_classifier,
    compute_half_max_width,
    detect_double_pulse,
    get_default_ml_paths,
)
from h5_io import get_path_list, get_pulse_block, load_marker_array, open_h5
from pulse_config import WAVEFORM_FS
from pulse_shape_metrics import (
    baseline_correct,
    double_pulse_metrics,
    normalize_trace,
    paired_symmetry_test,
    shift_waveform,
    symmetry_at_fraction,
)

console = Console()

OUTPUT_DIR = PULSE_SHAPE_PROTOTYPES_DIR
SAMPLE_SIZE = 100  # gray overlay traces in prototype plots
MAX_WAVEFORMS_PER_CLASS = 20_000  # random-sample cap for prototype aggregates
PROTOTYPE_STAT = "median"  # overlay / panel / double-alignment display default
RANDOM_SEED = 42
# raw_pulses are already on a common 48 kHz-equivalent grid (24 kHz files were
# interpolated upstream). Always use WAVEFORM_FS for snippet sample↔time.
TARGET_FS = WAVEFORM_FS  # alias kept for callers / cached APIs

CLIP_RATIO = 0.995
CLIP_MIN_CONSECUTIVE = 8

CLASS_IDS = {name: label_id for label_id, name in SPECIAL_PULSE_CLASSES.items()}

PULSE_SHAPES = {
    "normal": {
        "label": "Normal pulse",
        "class_id": CLASS_IDS["normal"],
        "color": pulse_shape_color("normal"),
        "align": "maximum",
        "criteria": [
            "Predicted-positive pulse",
            "Random Forest class: normal",
            "Robust PCA + RF on strongest channel waveform",
        ],
    },
    "double": {
        "label": "Double pulse",
        "class_id": CLASS_IDS["double"],
        "color": pulse_shape_color("double"),
        "align": "valley",
        "criteria": [
            "Random Forest class: double",
            "Robust PCA + RF multiclass classifier",
            "Aligned at inter-peak trough",
        ],
    },
    "wide": {
        "label": "Wide pulse",
        "class_id": CLASS_IDS["wide"],
        "color": pulse_shape_color("wide"),
        "align": "maximum",
        "criteria": [
            "Random Forest class: wide",
            "Robust PCA + RF multiclass classifier",
            "Aligned at peak maximum",
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


def save_prototype_mean_waveforms(
    pulse_key: str,
    aligned_waveforms: np.ndarray,
    fs: float,
    output_dir: Path,
    *,
    n_classified: int,
    align_mode: str,
) -> Path:
    """Save aligned mean/median prototype waveforms for later reuse."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mean_trace = np.mean(aligned_waveforms, axis=0)
    median_trace = np.median(aligned_waveforms, axis=0)
    out = output_dir / f"prototype_{pulse_key}_mean.npz"
    np.savez(
        out,
        mean=np.asarray(mean_trace, dtype=float),
        median=np.asarray(median_trace, dtype=float),
        fs=float(fs),
        n_in_mean=int(len(aligned_waveforms)),
        n_classified=int(n_classified),
        align_mode=str(align_mode),
        pulse_key=str(pulse_key),
    )
    console.log(
        f"  Saved prototype mean/median ({len(aligned_waveforms):,} waveforms) to {out.name}"
    )
    return out


def load_prototype_mean_waveforms(
    pulse_key: str, output_dir: Path | None = None
) -> dict | None:
    """Load a previously saved prototype mean npz, or None if missing."""
    output_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    path = output_dir / f"prototype_{pulse_key}_mean.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    return {
        "mean": data["mean"],
        "median": data["median"],
        "fs": float(data["fs"]),
        "n_in_mean": int(data["n_in_mean"]),
        "n_classified": int(data["n_classified"]),
        "align_mode": str(data["align_mode"]),
        "pulse_key": str(data["pulse_key"]) if "pulse_key" in data.files else pulse_key,
        "path": path,
    }


def collect_classifier_pulse_indices(
    data_path, class_id, max_entries=None, random_seed=None
):
    """Collect pulse indices for one RF classifier class (special_pulse_class).

    Returns ``(entries, n_classified)``. If ``max_entries`` is set, files are
    visited in random order and collection stops once enough matches are held
    (``n_classified`` is then a lower bound: ``len(entries)``).
    """
    entries = []
    rng = np.random.default_rng(random_seed)
    file_paths = list(get_path_list(Path(data_path)))
    if max_entries is not None:
        rng.shuffle(file_paths)

    seen = 0
    for file_path in file_paths:
        file = open_h5(file_path, nixio.FileMode.ReadOnly)
        if file is None:
            continue
        try:
            block = get_pulse_block(file)
            data_array_names = [da.name for da in block.data_arrays]

            if "raw_pulses" not in data_array_names:
                continue

            classes = load_marker_array(file_path, MULTICLASS_ARRAY_NAME, block)
            if classes is None:
                continue

            num_pulses = len(block.data_arrays["raw_pulses"])
            if "predicted_labels" in data_array_names:
                candidate_indices = np.where(
                    block.data_arrays["predicted_labels"][:] == 1
                )[0]
            else:
                candidate_indices = np.arange(num_pulses)

            classes_arr = np.asarray(classes)
            matching = candidate_indices[classes_arr[candidate_indices] == class_id]
            if matching.size == 0:
                continue

            if max_entries is None:
                entries.extend((file_path, int(i)) for i in matching)
                seen += int(matching.size)
            else:
                need = max_entries - len(entries)
                if need <= 0:
                    break
                if matching.size <= need:
                    entries.extend((file_path, int(i)) for i in matching)
                else:
                    pick = rng.choice(matching.size, size=need, replace=False)
                    entries.extend((file_path, int(matching[j])) for j in pick)
                seen = len(entries)
                if len(entries) >= max_entries:
                    break
        finally:
            file.close()
    return entries, seen if max_entries is None else seen

def load_sampled_waveforms(entries, sample_size, random_seed, align_mode=None):
    if not entries:
        return [], [], None

    rng = np.random.default_rng(random_seed)
    shuffled_entries = list(entries)
    rng.shuffle(shuffled_entries)

    corrected_waveforms = []
    normalized_waveforms = []
    fs = None
    open_files = {}

    try:
        for file_path, pulse_idx in shuffled_entries:
            if len(normalized_waveforms) >= sample_size:
                break

            file_key = str(file_path)
            if file_key not in open_files:
                file = open_h5(file_path, nixio.FileMode.ReadOnly)
                if file is None:
                    continue
                block = get_pulse_block(file)
                open_files[file_key] = (file, block)

            file, block = open_files[file_key]
            pulse_data = block.data_arrays["raw_pulses"][pulse_idx][:]
            if fs is None:
                fs = float(WAVEFORM_FS)

            trace, _ = get_biggest_unclipped_waveform(pulse_data)
            corrected = baseline_correct(trace)

            if align_mode == "valley" and double_peak_indices(corrected, fs) is None:
                continue

            corrected_waveforms.append(corrected)
            normalized_waveforms.append(normalize_trace(corrected))
    finally:
        for file, _ in open_files.values():
            file.close()

    return corrected_waveforms, normalized_waveforms, fs


def add_half_max_markers(ax, mean_trace, fs, color, time_offset_ms=0.0):
    """Mark half-maximum level and FWHM span of the mean waveform."""
    width_sec, width_info = compute_half_max_width(mean_trace, fs)
    if np.isnan(width_sec):
        return
    left_t = width_info["left_idx"] / fs * 1000 + time_offset_ms
    right_t = width_info["right_idx"] / fs * 1000 + time_offset_ms
    half_h = width_info["half_height"]
    width_ms = width_sec * 1000

    ax.axhline(
        half_h,
        color=color,
        linestyle="--",
        alpha=0.55,
        linewidth=1.0,
        zorder=3,
    )
    ax.axvspan(
        left_t,
        right_t,
        color=color,
        alpha=0.12,
        zorder=2,
    )
    # Explicit width bar (legend entry for half-width).
    ax.plot(
        [left_t, right_t],
        [half_h, half_h],
        color=color,
        linestyle="-",
        linewidth=2.0,
        marker="|",
        markersize=14,
        markeredgewidth=2.0,
        alpha=0.95,
        zorder=6,
        label=f"Half-width = {width_ms:.2f} ms",
    )


def add_symmetry_fraction_markers(ax, mean_trace, fs, color, fraction=0.1):
    """Mark 10% amplitude crossings and left/right widths on a prototype plot."""
    sym = symmetry_at_fraction(mean_trace, fs, fraction=fraction)
    threshold = sym["threshold"]
    if np.isnan(threshold):
        return

    left_t = sym["left_idx"] / fs * 1000
    right_t = sym["right_idx"] / fs * 1000
    left_w = sym["left_width_sec"] * 1000
    right_w = sym["right_width_sec"] * 1000
    pct = int(fraction * 100)

    ax.axhline(
        threshold,
        color=color,
        linestyle=":",
        linewidth=2.0,
        alpha=0.9,
        label=f"{pct}% amplitude",
    )
    ax.axvline(left_t, color=color, linestyle=":", alpha=0.75, linewidth=1.5)
    ax.axvline(right_t, color=color, linestyle=":", alpha=0.75, linewidth=1.5)
    ax.scatter([left_t, right_t], [threshold, threshold], color=color, s=70, zorder=7)
    ax.annotate(
        f"L = {left_w:.2f} ms",
        xy=(left_t, threshold),
        xytext=(-36, -22),
        textcoords="offset points",
        fontsize=10,
        color=color,
    )
    ax.annotate(
        f"R = {right_w:.2f} ms",
        xy=(right_t, threshold),
        xytext=(10, -22),
        textcoords="offset points",
        fontsize=10,
        color=color,
    )


def add_detection_markers(ax, mean_trace, fs, pulse_shape, pulse_key=None):
    color = pulse_shape["color"]
    center_idx = len(mean_trace) // 2
    center_t = center_idx / fs * 1000

    if pulse_shape["align"] == "valley":
        peaks = double_peak_indices(mean_trace, fs)
        if peaks is not None:
            peak_times = peaks / fs * 1000
            dt_ms = (peaks[1] - peaks[0]) / fs * 1000
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
            # Peak-to-peak Δt: visual bar + dedicated legend entry.
            peak_y = float(np.mean(mean_trace[peaks]))
            ax.plot(
                peak_times,
                [peak_y, peak_y],
                color=color,
                linestyle=":",
                linewidth=1.6,
                marker="|",
                markersize=10,
                markeredgewidth=1.6,
                alpha=0.9,
                zorder=5,
                label=f"$\\Delta t$ = {dt_ms:.2f} ms",
            )
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


def match_trace_length(trace, target_len: int):
    """Center-crop or zero-pad a 1D trace to ``target_len`` samples."""
    trace = np.asarray(trace, dtype=float)
    n = len(trace)
    if n == target_len:
        return trace
    if n > target_len:
        start = (n - target_len) // 2
        return trace[start : start + target_len]
    out = np.zeros(target_len, dtype=float)
    start = (target_len - n) // 2
    out[start : start + n] = trace
    return out


def load_all_waveforms(
    entries,
    align_mode=None,
    max_waveforms=None,
    random_seed=None,
    required_fs: float | None = None,
    target_fs: float = WAVEFORM_FS,
    target_len: int | None = None,
):
    """Load waveforms for prototype aggregates (optionally capped via random sample).

    Snippet arrays are already on the shared ``WAVEFORM_FS`` grid (24 kHz files
    were interpolated upstream). ``target_fs`` is ignored except as the returned
    timebase (always ``WAVEFORM_FS``). Length is matched to ``target_len`` (or
    the first accepted pulse) via center crop/pad.

    If ``required_fs`` is set, only files whose *native* metadata rate matches
    are used (provenance filter; does not change the waveform timebase).
    Entries are shuffled, then grouped by file in shuffle order so HDF5 reads
    stay sequential without alphabetically biasing the capped sample.
    """
    entries = list(entries)
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
        rng.shuffle(entries)
    # Preserve shuffle order of first file appearance; sequential within file.
    grouped = {}
    file_order = []
    for file_path, pulse_idx in entries:
        key = str(file_path)
        if key not in grouped:
            grouped[key] = []
            file_order.append(file_path)
        grouped[key].append((file_path, pulse_idx))
    entries = [item for path in file_order for item in grouped[str(path)]]

    corrected_waveforms = []
    normalized_waveforms = []
    fs = float(WAVEFORM_FS)
    open_files = {}
    n_skipped_rate = 0
    n_skipped_len = 0
    n_skipped_morphology = 0
    _ = target_fs  # callers may still pass it; timebase is always WAVEFORM_FS

    try:
        for file_path, pulse_idx in entries:
            if max_waveforms is not None and len(normalized_waveforms) >= max_waveforms:
                break

            file_key = str(file_path)
            if file_key not in open_files:
                file = open_h5(file_path, nixio.FileMode.ReadOnly)
                if file is None:
                    continue
                block = get_pulse_block(file)
                file_fs = float(file.sections["pulses_metadata"]["metadata"]["samplerate"])
                if required_fs is not None and abs(file_fs - required_fs) >= 1e-6:
                    file.close()
                    open_files[file_key] = None
                    continue
                open_files[file_key] = (file, block, file_fs)

            cached = open_files[file_key]
            if cached is None:
                n_skipped_rate += 1
                continue
            file, block, _file_fs = cached
            pulse_data = block.data_arrays["raw_pulses"][pulse_idx][:]
            trace, _ = get_biggest_unclipped_waveform(pulse_data)

            if target_len is None:
                target_len = len(trace)
            elif len(trace) != target_len:
                # Prefer exact match; fall back to pad/crop for rare mismatches.
                if abs(len(trace) - target_len) > max(8, target_len // 20):
                    n_skipped_len += 1
                    continue
                trace = match_trace_length(trace, int(target_len))
            else:
                trace = np.asarray(trace, dtype=float)

            corrected = baseline_correct(trace)
            if align_mode == "valley" and double_peak_indices(corrected, fs) is None:
                n_skipped_morphology += 1
                continue
            corrected_waveforms.append(corrected)
            normalized_waveforms.append(normalize_trace(corrected))
    finally:
        for cached in open_files.values():
            if cached is not None:
                cached[0].close()

    if n_skipped_rate or n_skipped_len or n_skipped_morphology:
        console.log(
            f"  Waveform load @ {fs:.0f} Hz: kept {len(normalized_waveforms):,}, "
            f"skipped native-rate filter={n_skipped_rate:,}, "
            f"skipped length={n_skipped_len:,}, "
            f"skipped no-valley={n_skipped_morphology:,}"
        )

    return corrected_waveforms, normalized_waveforms, fs


def plot_prototype_pulse_shape(
    pulse_key,
    pulse_shape,
    corrected_waveforms,
    normalized_waveforms,
    fs,
    output_dir,
    median_trace_all=None,
    n_in_median=None,
    total_count=None,
    prototype_stat=PROTOTYPE_STAT,
):
    """Plot SAMPLE_SIZE gray traces with the colored median prototype overlay."""
    if not normalized_waveforms:
        console.log(f"[yellow]No pulses found for {pulse_shape['label']}. Skipping.")
        return None
    if prototype_stat != "median":
        raise ValueError("prototype plots display the median; got {!r}".format(prototype_stat))

    aligned_sample = align_waveforms(
        normalized_waveforms, corrected_waveforms, pulse_shape["align"], fs
    )
    pool_n = n_in_median if n_in_median is not None else len(aligned_sample)
    if median_trace_all is not None:
        median_trace = median_trace_all
    else:
        median_trace = np.median(aligned_sample, axis=0)
    median_label = f"Median (n={pool_n:,})"

    time_ms = np.arange(aligned_sample.shape[1]) / fs * 1000

    fig, ax = plt.subplots(figsize=(11, 6))

    for trace in aligned_sample:
        ax.plot(time_ms, trace, color="gray", alpha=0.25, linewidth=0.8)

    ax.plot(
        time_ms,
        median_trace,
        color=pulse_shape["color"],
        linewidth=2.8,
        label=median_label,
        zorder=5,
    )

    add_detection_markers(ax, median_trace, fs, pulse_shape, pulse_key=pulse_key)
    add_half_max_markers(ax, median_trace, fs, pulse_shape["color"])

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
    ax.set_ylim(-0.2, 1.05)
    if total_count is not None and total_count != pool_n:
        title = (
            f"Prototype {pulse_shape['label'].lower()}s "
            f"(showing {len(aligned_sample)} of {pool_n:,}; "
            f"{total_count:,} classified)"
        )
    else:
        title = (
            f"Prototype {pulse_shape['label'].lower()}s "
            f"(showing {len(aligned_sample)} of {pool_n:,})"
        )
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"prototype_{pulse_key}_pulse.png"
    fig.savefig(output_path, dpi=300)
    save_thesis_figure(f"pulse_shapes/prototype_{pulse_key}_pulse.png", fig)
    plt.close(fig)
    console.log(f"Saved {output_path}")
    return output_path


def plot_symmetry_analysis(
    pulse_key, pulse_shape, corrected_waveforms, fs, output_dir, fraction=0.1
):
    """Whisker plot and paired t-test for left/right width at fraction of peak."""
    if pulse_key not in {"normal", "wide"}:
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


def plot_mean_pulse_shapes_panel(
    output_dir: Path | None = None,
    *,
    panel_order=("normal", "double", "wide"),
    save_name: str = "mean_pulse_shapes_panel.png",
    prototype_stat: str = PROTOTYPE_STAT,
) -> Path | None:
    """Three-panel talk figure: median normal / double / wide waveforms side by side.

    Filename kept as ``mean_pulse_shapes_panel.png`` for thesis path stability;
    the plotted aggregate is ``prototype_stat`` (default: median).
    """
    output_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    apply_presentation_style()
    if prototype_stat not in ("mean", "median"):
        raise ValueError("prototype_stat must be 'mean' or 'median'")

    traces = {}
    for key in panel_order:
        loaded = load_prototype_mean_waveforms(key, output_dir)
        if loaded is None:
            console.log(f"[yellow]Missing prototype_{key}_mean.npz — skip panel figure.")
            return None
        traces[key] = loaded

    for k in panel_order:
        stored = float(traces[k]["fs"])
        if abs(stored - WAVEFORM_FS) >= 1e-6:
            console.log(
                f"[yellow]prototype_{k}: cached fs={stored:.0f} Hz → "
                f"plotting as WAVEFORM_FS={WAVEFORM_FS:.0f} Hz"
            )

    fig, axes = plt.subplots(
        1,
        len(panel_order),
        figsize=(14, 4.2),
        sharey=True,
        constrained_layout=True,
    )
    if len(panel_order) == 1:
        axes = [axes]

    y_max = 0.0
    y_min = 0.0
    for ax, key in zip(axes, panel_order):
        shape = PULSE_SHAPES[key]
        trace = np.asarray(traces[key][prototype_stat], dtype=float)
        n = int(traces[key]["n_in_mean"])
        time_ms = np.arange(len(trace)) / WAVEFORM_FS * 1000
        # Center time on the alignment sample so shapes line up visually.
        time_ms = time_ms - time_ms[len(trace) // 2]

        color = shape["color"]
        ax.plot(time_ms, trace, color=color, linewidth=3.0, solid_capstyle="round")
        ax.axhline(0.0, color="#bbbbbb", linewidth=1.0, zorder=0)
        ax.axvline(0.0, color="#dddddd", linewidth=1.0, linestyle=":", zorder=0)

        ax.set_title(shape["label"], color=color, pad=10)
        ax.set_xlabel("Time (ms)")
        ax.grid(True, alpha=0.28)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(
            0.97,
            0.95,
            f"n = {n:,}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=11,
            color="#555555",
        )
        y_max = max(y_max, float(np.nanmax(trace)))
        y_min = min(y_min, float(np.nanmin(trace)))

    axes[0].set_ylabel("Normalized amplitude")
    pad = 0.08 * max(y_max - y_min, 1.0)
    axes[0].set_ylim(y_min - pad, y_max + pad)
    for ax in axes:
        ax.set_xlim(-5.0, 5.0)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / save_name
    fig.savefig(output_path, dpi=300)
    thesis_path = save_thesis_figure(f"pulse_shapes/{save_name}", fig)
    plt.close(fig)
    console.log(f"Saved {output_path} ({prototype_stat})")
    console.log(f"Saved thesis figure {thesis_path}")
    return output_path


def plot_normal_double_overlay(
    normal_trace,
    double_trace,
    fs,
    output_dir,
    prototype_stat: str = PROTOTYPE_STAT,
):
    """Overlay median double pulse with two median normal pulses (peak-aligned)."""
    dp_peaks = double_peak_indices(double_trace, fs)
    if dp_peaks is None:
        console.log(
            "[yellow]Skipping normal_pulses_aligned_to_double.png: "
            "no two peaks detected on double prototype."
        )
        return None

    normal_peak = int(np.argmax(normal_trace))
    p1, p2 = dp_peaks

    norm1 = shift_waveform(normal_trace, p1 - normal_peak)
    norm2 = shift_waveform(normal_trace, p2 - normal_peak)
    stat_label = prototype_stat.capitalize()

    time_ms = np.arange(len(double_trace)) / fs * 1000
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        time_ms,
        double_trace,
        color=PULSE_SHAPES["double"]["color"],
        linewidth=2.5,
        label=f"{stat_label} double pulse",
        zorder=4,
    )
    ax.plot(
        time_ms,
        norm1,
        color=PULSE_SHAPES["normal"]["color"],
        linewidth=2,
        linestyle="--",
        label=f"{stat_label} normal pulse (aligned to 1st peak)",
        zorder=3,
    )
    ax.plot(
        time_ms,
        norm2,
        color=PULSE_SHAPES["normal"]["color"],
        linewidth=2,
        linestyle=":",
        label=f"{stat_label} normal pulse (aligned to 2nd peak)",
        zorder=3,
    )
    ax.scatter(
        dp_peaks / fs * 1000,
        double_trace[dp_peaks],
        color=PULSE_SHAPES["double"]["color"],
        s=80,
        marker="*",
        zorder=5,
        label="Double peaks",
    )
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Normalized amplitude")
    ax.set_title(
        f"{stat_label} double pulse vs two {prototype_stat} normal pulses (peak-aligned)"
    )
    ax.legend(loc=LEGEND_LOC, fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = output_dir / "normal_pulses_aligned_to_double.png"
    fig.savefig(out, dpi=300)
    save_thesis_figure("pulse_shapes/normal_pulses_aligned_to_double.png", fig)
    plt.close(fig)
    console.log(f"Saved {out}")
    return out


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


def refresh_aligned_overlay_from_npz(output_dir: Path | None = None) -> Path | None:
    """Rebuild aligned-to-double plot from saved prototypes on WAVEFORM_FS."""
    output_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    apply_presentation_style()
    normal = load_prototype_mean_waveforms("normal", output_dir)
    double = load_prototype_mean_waveforms("double", output_dir)
    if normal is None or double is None:
        console.log(
            "[yellow]WARNING: skipped normal_pulses_aligned_to_double.png — "
            "missing prototype_normal/double_mean.npz."
        )
        return None

    normal_u = np.asarray(normal[PROTOTYPE_STAT], dtype=float)
    double_u = np.asarray(double[PROTOTYPE_STAT], dtype=float)
    target_len = max(len(normal_u), len(double_u))
    normal_u = match_trace_length(normal_u, target_len)
    double_u = match_trace_length(double_u, target_len)
    return plot_normal_double_overlay(normal_u, double_u, float(WAVEFORM_FS), output_dir)


def main(
    data_path=H5_DIR,
    sample_size=SAMPLE_SIZE,
    max_waveforms=MAX_WAVEFORMS_PER_CLASS,
    random_seed=RANDOM_SEED,
    apply_classifier=False,
):
    apply_presentation_style()
    console.log("Collecting and plotting prototype pulses for each pulse shape...")
    console.log(
        f"Prototype {PROTOTYPE_STAT} cap: {max_waveforms:,} pulses/class; "
        f"plot overlay: {sample_size} traces"
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if apply_classifier:
        model_path = get_default_ml_paths()["model"]
        if model_path.exists():
            console.log("Applying Random Forest classifier to h5 files...")
            apply_special_pulse_classifier(data_path)
        else:
            console.log(
                "Classifier model not found; using existing special_pulse_class labels."
            )
    else:
        console.log(
            "Using existing special_pulse_class labels (skipping classifier apply)."
        )

    pulse_counts = {}
    proto_traces = {}

    for pulse_key, pulse_shape in PULSE_SHAPES.items():
        console.log(f"\n{pulse_shape['label']}:")
        class_seed = random_seed + int(pulse_shape["class_id"])
        collect_cap = None
        if max_waveforms is not None:
            # Valley/fs/length filters reject many candidates; oversample.
            oversample = 8.0 if pulse_shape["align"] == "valley" else 2.0
            collect_cap = int(max_waveforms * oversample)
        entries, sampled_count = collect_classifier_pulse_indices(
            data_path,
            pulse_shape["class_id"],
            max_entries=collect_cap,
            random_seed=class_seed,
        )
        # Prefer full class totals from a previous full scan when early-stopping.
        prior_counts_path = OUTPUT_DIR / "pulse_shape_counts.json"
        total_count = sampled_count
        if collect_cap is not None and prior_counts_path.exists():
            with open(prior_counts_path) as handle:
                prior = json.load(handle)
            if pulse_key in prior and int(prior[pulse_key]) >= sampled_count:
                total_count = int(prior[pulse_key])
        pulse_counts[pulse_key] = total_count
        console.log(
            f"  Class total {total_count:,} "
            f"(holding {len(entries):,} for load)"
        )

        all_corrected, all_normalized, fs = load_all_waveforms(
            entries,
            align_mode=pulse_shape["align"],
            max_waveforms=max_waveforms,
            random_seed=class_seed,
        )
        if not all_normalized:
            continue

        all_aligned = align_waveforms(
            all_normalized, all_corrected, pulse_shape["align"], fs
        )
        median_all = np.median(all_aligned, axis=0)
        n_in_median = len(all_aligned)
        proto_traces[pulse_key] = (median_all, fs, all_corrected)
        save_prototype_mean_waveforms(
            pulse_key,
            all_aligned,
            fs,
            OUTPUT_DIR,
            n_classified=total_count,
            align_mode=pulse_shape["align"],
        )
        console.log(
            f"  Mean+median from {n_in_median:,} morph-aligned waveforms "
            f"(display={PROTOTYPE_STAT}; cap {max_waveforms:,}; "
            f"{total_count:,} RF-classified)"
        )

        rng = np.random.default_rng(class_seed)
        sample_idx = rng.choice(
            len(all_normalized),
            size=min(sample_size, len(all_normalized)),
            replace=False,
        )
        sample_corrected = [all_corrected[i] for i in sample_idx]
        sample_normalized = [all_normalized[i] for i in sample_idx]

        plot_prototype_pulse_shape(
            pulse_key,
            pulse_shape,
            sample_corrected,
            sample_normalized,
            fs,
            OUTPUT_DIR,
            median_trace_all=median_all,
            n_in_median=n_in_median,
            total_count=total_count,
        )

        if pulse_key == "double":
            save_double_peak_separation_stats(all_corrected, fs, OUTPUT_DIR)
        if pulse_key in {"normal", "wide"}:
            plot_symmetry_analysis(
                pulse_key, pulse_shape, all_corrected, fs, OUTPUT_DIR
            )

    with open(OUTPUT_DIR / "pulse_shape_counts.json", "w") as handle:
        json.dump(pulse_counts, handle, indent=2)

    if "normal" in proto_traces and "double" in proto_traces:
        normal_med, fs_n, _ = proto_traces["normal"]
        double_med, fs_d, _ = proto_traces["double"]
        if abs(float(fs_n) - float(fs_d)) < 1e-6:
            plot_normal_double_overlay(normal_med, double_med, fs_n, OUTPUT_DIR)
        else:
            console.log(
                "[yellow]WARNING: in-memory sample-rate mismatch "
                f"normal={fs_n:.0f} Hz vs double={fs_d:.0f} Hz; "
                "rebuilding overlay from NPZs on WAVEFORM_FS."
            )
            refresh_aligned_overlay_from_npz(OUTPUT_DIR)
    else:
        missing = [k for k in ("normal", "double") if k not in proto_traces]
        console.log(
            "[yellow]WARNING: skipped normal_pulses_aligned_to_double.png — "
            f"missing prototype traces for {missing}."
        )

    if plot_mean_pulse_shapes_panel(OUTPUT_DIR) is None:
        console.log(
            "[yellow]WARNING: mean_pulse_shapes_panel.png was not produced."
        )


if __name__ == "__main__":
    main()
