"""Split pulses at the half-width histogram valley and plot waveforms.

Archived WIP — not part of ``run_analysis.sh``. Kept for exploratory
thresholding of half-width modes. See ``archived/README.md``.

Analysis part: exploratory special-pulse waveforms.
Dependencies: data_paths, h5_io, pulse_shape_metrics, prototype helpers.
"""

Analysis part: exploratory WIP around the bimodal half-width distribution
(figures/pulse_shapes/half_width_kde_all_shapes.png).

For each predicted-positive pulse, compute half-max width and sampling rate,
find the local minimum between the two density peaks, then compare waveforms
below vs above that threshold.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from path_setup import setup_script_paths

setup_script_paths(__file__)

import matplotlib.pyplot as plt
import nixio
import numpy as np
from rich.console import Console
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

from data_paths import H5_DIR, active_thesis_figures_dir
from h5_io import get_path_list, get_pulse_block, open_h5
from presentation_style import apply_presentation_style
from special_pulses.double_peaks_detection import compute_half_max_width
from special_pulses.prototype_pulse_plots import get_biggest_unclipped_waveform
from special_pulses.pulse_shape_metrics import (
    baseline_correct,
    normalize_trace,
    shift_waveform,
)

console = Console()

OUTPUT_DIR = active_thesis_figures_dir() / "workinprogress"
CACHE_PATH = OUTPUT_DIR / "half_width_pulse_catalog.npz"

RANDOM_SEED = 42
MEAN_SAMPLE_SIZE = 3000
RAW_SAMPLE_SIZE = 10
# Common time grid for mean/median across mixed sampling rates (ms relative to peak)
TIME_MS_MIN = -4.0
TIME_MS_MAX = 8.0
TIME_MS_STEP = 0.02


def collect_catalog(data_path: Path, cache_path: Path) -> dict:
    """Collect half-width, fs, and pulse locations for all candidate pulses."""
    if cache_path.exists():
        console.log(f"Loading cached catalog from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return {
            "file_paths": data["file_paths"],
            "pulse_indices": data["pulse_indices"].astype(np.int64),
            "half_width_ms": data["half_width_ms"].astype(float),
            "fs": data["fs"].astype(float),
        }

    file_paths: list[str] = []
    pulse_indices: list[int] = []
    half_widths: list[float] = []
    sample_rates: list[float] = []

    paths = get_path_list(Path(data_path))
    console.log(f"Collecting half-widths from {len(paths)} h5 files...")

    for file_idx, file_path in enumerate(paths, 1):
        file = open_h5(file_path, nixio.FileMode.ReadOnly)
        if file is None:
            continue
        try:
            block = get_pulse_block(file)
            names = [da.name for da in block.data_arrays]
            if "raw_pulses" not in names or "predicted_labels" not in names:
                continue

            fs = float(file.sections["pulses_metadata"]["metadata"]["samplerate"])
            raw = block.data_arrays["raw_pulses"]
            pred = block.data_arrays["predicted_labels"][:]
            candidates = np.where(pred == 1)[0]
            if candidates.size == 0:
                continue

            for pulse_idx in candidates:
                trace, _ = get_biggest_unclipped_waveform(raw[int(pulse_idx)][:])
                corrected = baseline_correct(trace)
                width_sec, _ = compute_half_max_width(corrected, fs)
                width_ms = float(width_sec * 1000)
                if not np.isfinite(width_ms) or width_ms <= 0:
                    continue
                file_paths.append(str(file_path))
                pulse_indices.append(int(pulse_idx))
                half_widths.append(width_ms)
                sample_rates.append(fs)

            if file_idx % 10 == 0 or file_idx == len(paths):
                console.log(
                    f"  [{file_idx}/{len(paths)}] catalog size={len(half_widths):,}"
                )
        finally:
            file.close()

    catalog = {
        "file_paths": np.asarray(file_paths, dtype=object),
        "pulse_indices": np.asarray(pulse_indices, dtype=np.int64),
        "half_width_ms": np.asarray(half_widths, dtype=float),
        "fs": np.asarray(sample_rates, dtype=float),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **catalog)
    console.log(f"Saved catalog ({len(half_widths):,} pulses) to {cache_path}")
    return catalog


def find_valley_threshold(widths_ms: np.ndarray) -> dict:
    """Find local minimum between the two main peaks of the half-width density."""
    widths = widths_ms[np.isfinite(widths_ms)]
    # Focus on the bulk of the distribution (exclude extreme tail)
    lo, hi = np.percentile(widths, [0.5, 99.5])
    clipped = widths[(widths >= lo) & (widths <= hi)]

    kde = gaussian_kde(clipped, bw_method=0.08)
    x = np.linspace(lo, hi, 2000)
    y = kde(x)

    peaks, _ = find_peaks(y, prominence=np.max(y) * 0.05)
    if peaks.size < 2:
        # Fallback: smoothed histogram valley between modes
        hist, edges = np.histogram(clipped, bins=120, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        smooth = gaussian_filter1d(hist.astype(float), sigma=2.0)
        peaks, _ = find_peaks(smooth, prominence=np.max(smooth) * 0.05)
        if peaks.size < 2:
            raise RuntimeError("Could not find two peaks in half-width distribution")
        p1, p2 = sorted(peaks, key=lambda i: smooth[i], reverse=True)[:2]
        p1, p2 = sorted([p1, p2])
        valley_idx = p1 + int(np.argmin(smooth[p1 : p2 + 1]))
        threshold = float(centers[valley_idx])
        return {
            "threshold_ms": threshold,
            "peak1_ms": float(centers[p1]),
            "peak2_ms": float(centers[p2]),
            "x": centers,
            "density": smooth,
            "method": "smoothed_histogram",
        }

    # Take the two highest peaks, ordered by x
    top2 = sorted(peaks, key=lambda i: y[i], reverse=True)[:2]
    p1, p2 = sorted(top2)
    valley_idx = p1 + int(np.argmin(y[p1 : p2 + 1]))
    threshold = float(x[valley_idx])
    return {
        "threshold_ms": threshold,
        "peak1_ms": float(x[p1]),
        "peak2_ms": float(x[p2]),
        "x": x,
        "density": y,
        "method": "kde",
    }


def load_waveforms(entries: list[tuple[str, int]]) -> tuple[list[np.ndarray], list[float]]:
    """Load baseline-corrected waveforms for (file_path, pulse_idx) entries."""
    corrected: list[np.ndarray] = []
    rates: list[float] = []
    open_files: dict[str, tuple] = {}

    try:
        for file_path, pulse_idx in entries:
            key = str(file_path)
            if key not in open_files:
                file = open_h5(file_path, nixio.FileMode.ReadOnly)
                if file is None:
                    continue
                block = get_pulse_block(file)
                open_files[key] = (file, block)

            file, block = open_files[key]
            pulse_data = block.data_arrays["raw_pulses"][int(pulse_idx)][:]
            fs = float(file.sections["pulses_metadata"]["metadata"]["samplerate"])
            trace, _ = get_biggest_unclipped_waveform(pulse_data)
            corrected.append(baseline_correct(trace))
            rates.append(fs)
    finally:
        for file, _ in open_files.values():
            file.close()

    return corrected, rates


def peak_align_to_time_grid(
    waveforms: list[np.ndarray],
    rates: list[float],
    time_ms: np.ndarray,
) -> np.ndarray:
    """Peak-align waveforms and interpolate onto a common time axis in ms."""
    grid = []
    for trace, fs in zip(waveforms, rates):
        if len(trace) < 5:
            continue
        peak_idx = int(np.argmax(trace))
        t_ms = (np.arange(len(trace)) - peak_idx) / fs * 1000.0
        norm = normalize_trace(trace)
        grid.append(np.interp(time_ms, t_ms, norm, left=np.nan, right=np.nan))
    if not grid:
        return np.empty((0, len(time_ms)))
    return np.asarray(grid, dtype=float)


def sample_indices(mask: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    idxs = np.flatnonzero(mask)
    if idxs.size == 0:
        return idxs
    if idxs.size <= n:
        return idxs
    return rng.choice(idxs, size=n, replace=False)


def plot_histogram_with_threshold(
    widths: np.ndarray,
    valley: dict,
    output_dir: Path,
):
    thr = valley["threshold_ms"]
    below = widths < thr
    above = widths >= thr

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(widths, bins=80, color="steelblue", alpha=0.75, edgecolor="white", density=True)
    ax.plot(valley["x"], valley["density"], color="black", linewidth=1.5, label="Density")
    ax.axvline(thr, color="crimson", linestyle="--", linewidth=2, label=f"Threshold={thr:.2f} ms")
    ax.axvline(valley["peak1_ms"], color="gray", linestyle=":", alpha=0.8, label="Peaks")
    ax.axvline(valley["peak2_ms"], color="gray", linestyle=":", alpha=0.8)
    ax.set_xlabel("Half width (ms)")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Half-width distribution with valley threshold\n"
        f"below n={below.sum():,} | above n={above.sum():,} ({valley['method']})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "half_width_threshold.png", dpi=300)
    plt.close(fig)


def plot_mean_median(
    below_grid: np.ndarray,
    above_grid: np.ndarray,
    time_ms: np.ndarray,
    thr: float,
    output_dir: Path,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, grid, title, color in [
        (axes[0], below_grid, f"Below threshold (<{thr:.2f} ms)", "#1f77b4"),
        (axes[1], above_grid, f"Above threshold (≥{thr:.2f} ms)", "#d62728"),
    ]:
        if grid.size == 0:
            ax.set_title(f"{title}\n(n=0)")
            continue
        mean = np.nanmean(grid, axis=0)
        median = np.nanmedian(grid, axis=0)
        ax.plot(time_ms, mean, color=color, linewidth=2.0, label=f"Mean (n={len(grid):,})")
        ax.plot(
            time_ms,
            median,
            color="black",
            linewidth=1.5,
            linestyle="--",
            label="Median",
        )
        ax.axvline(0, color="gray", linestyle=":", alpha=0.6)
        ax.set_xlabel("Time from peak (ms)")
        ax.set_ylabel("Normalized amplitude")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Mean / median waveforms by half-width population", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "mean_median_waveforms_by_threshold.png", dpi=300)
    plt.close(fig)


def plot_raw_examples(
    below_waves: list[np.ndarray],
    below_fs: list[float],
    above_waves: list[np.ndarray],
    above_fs: list[float],
    thr: float,
    output_dir: Path,
):
    fig, axes = plt.subplots(2, RAW_SAMPLE_SIZE, figsize=(18, 5), sharey=True)
    for col in range(RAW_SAMPLE_SIZE):
        for row, waves, rates, color, label in [
            (0, below_waves, below_fs, "#1f77b4", f"Below (<{thr:.2f} ms)"),
            (1, above_waves, above_fs, "#d62728", f"Above (≥{thr:.2f} ms)"),
        ]:
            ax = axes[row, col]
            if col >= len(waves):
                ax.axis("off")
                continue
            trace = waves[col]
            fs = rates[col]
            peak_idx = int(np.argmax(trace))
            shifted = shift_waveform(normalize_trace(trace), len(trace) // 2 - peak_idx)
            t_ms = (np.arange(len(shifted)) - len(shifted) // 2) / fs * 1000.0
            ax.plot(t_ms, shifted, color=color, linewidth=1.0)
            ax.set_xlim(TIME_MS_MIN, TIME_MS_MAX)
            ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
            ax.grid(True, alpha=0.25)
            if row == 0:
                ax.set_title(f"#{col + 1}", fontsize=9)
            if col == 0:
                ax.set_ylabel(label, fontsize=9)
            if row == 1:
                ax.set_xlabel("ms")
    fig.suptitle("10 random raw waveforms per half-width population", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "raw_waveforms_by_threshold.png", dpi=300)
    plt.close(fig)


def plot_fs_distribution(fs_below: np.ndarray, fs_above: np.ndarray, thr: float, output_dir: Path):
    def counts(fs_arr: np.ndarray) -> dict[str, int]:
        return {
            "24 kHz": int(np.sum(np.isclose(fs_arr, 24000))),
            "48 kHz": int(np.sum(np.isclose(fs_arr, 48000))),
            "other": int(np.sum(~(np.isclose(fs_arr, 24000) | np.isclose(fs_arr, 48000)))),
        }

    below_c = counts(fs_below)
    above_c = counts(fs_above)
    labels = ["24 kHz", "48 kHz"]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    b_vals = [below_c["24 kHz"], below_c["48 kHz"]]
    a_vals = [above_c["24 kHz"], above_c["48 kHz"]]
    bars1 = ax.bar(x - width / 2, b_vals, width, label=f"Below (<{thr:.2f} ms)", color="#1f77b4")
    bars2 = ax.bar(x + width / 2, a_vals, width, label=f"Above (≥{thr:.2f} ms)", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Pulse count")
    ax.set_title("Sampling-frequency distribution by half-width population")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{int(h):,}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    if below_c["other"] or above_c["other"]:
        ax.text(
            0.98,
            0.98,
            f"other fs: below={below_c['other']}, above={above_c['other']}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_dir / "sampling_frequency_by_threshold.png", dpi=300)
    plt.close(fig)

    summary = {
        "threshold_ms": thr,
        "below": {"n": int(fs_below.size), **below_c},
        "above": {"n": int(fs_above.size), **above_c},
    }
    with open(output_dir / "sampling_frequency_by_threshold.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main(data_path=H5_DIR):
    global OUTPUT_DIR, CACHE_PATH
    apply_presentation_style()
    OUTPUT_DIR = active_thesis_figures_dir() / "workinprogress"
    CACHE_PATH = OUTPUT_DIR / "half_width_pulse_catalog.npz"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    catalog = collect_catalog(Path(data_path), CACHE_PATH)
    widths = catalog["half_width_ms"]
    fs_all = catalog["fs"]
    console.log(f"Catalog pulses with finite half-width: {widths.size:,}")

    valley = find_valley_threshold(widths)
    thr = valley["threshold_ms"]
    console.log(
        f"Valley threshold={thr:.3f} ms "
        f"(peaks at {valley['peak1_ms']:.2f} and {valley['peak2_ms']:.2f} ms)"
    )

    below_mask = widths < thr
    above_mask = widths >= thr
    console.log(f"Below: {below_mask.sum():,} | Above: {above_mask.sum():,}")

    plot_histogram_with_threshold(widths, valley, OUTPUT_DIR)
    fs_summary = plot_fs_distribution(fs_all[below_mask], fs_all[above_mask], thr, OUTPUT_DIR)
    console.log(f"Sampling-frequency counts: {fs_summary}")

    below_sel = sample_indices(below_mask, MEAN_SAMPLE_SIZE, rng)
    above_sel = sample_indices(above_mask, MEAN_SAMPLE_SIZE, rng)
    below_raw_sel = sample_indices(below_mask, RAW_SAMPLE_SIZE, rng)
    above_raw_sel = sample_indices(above_mask, RAW_SAMPLE_SIZE, rng)

    def entries_from(sel: np.ndarray) -> list[tuple[str, int]]:
        return [
            (str(catalog["file_paths"][i]), int(catalog["pulse_indices"][i]))
            for i in sel
        ]

    console.log(
        f"Loading mean/median subsets "
        f"(below={len(below_sel):,}, above={len(above_sel):,})..."
    )
    below_waves, below_rates = load_waveforms(entries_from(below_sel))
    above_waves, above_rates = load_waveforms(entries_from(above_sel))

    time_ms = np.arange(TIME_MS_MIN, TIME_MS_MAX + TIME_MS_STEP, TIME_MS_STEP)
    below_grid = peak_align_to_time_grid(below_waves, below_rates, time_ms)
    above_grid = peak_align_to_time_grid(above_waves, above_rates, time_ms)
    plot_mean_median(below_grid, above_grid, time_ms, thr, OUTPUT_DIR)

    console.log("Loading 10 raw examples per population...")
    below_raw, below_raw_fs = load_waveforms(entries_from(below_raw_sel))
    above_raw, above_raw_fs = load_waveforms(entries_from(above_raw_sel))
    plot_raw_examples(below_raw, below_raw_fs, above_raw, above_raw_fs, thr, OUTPUT_DIR)

    meta = {
        "threshold_ms": thr,
        "peak1_ms": valley["peak1_ms"],
        "peak2_ms": valley["peak2_ms"],
        "method": valley["method"],
        "n_total": int(widths.size),
        "n_below": int(below_mask.sum()),
        "n_above": int(above_mask.sum()),
        "mean_sample_size_requested": MEAN_SAMPLE_SIZE,
        "n_below_in_mean": int(below_grid.shape[0]),
        "n_above_in_mean": int(above_grid.shape[0]),
        "sampling_frequency": fs_summary,
        "data_path": str(data_path),
        "random_seed": RANDOM_SEED,
    }
    with open(OUTPUT_DIR / "half_width_threshold_summary.json", "w") as handle:
        json.dump(meta, handle, indent=2)

    console.log(f"Saved figures and summary to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
