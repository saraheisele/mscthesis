"""Plot spatial eel usage along the Berlin line logger.

Analysis part: position visualization (Part 5b of Berlin activity analysis).
Dependencies: data_paths, position_utils; requires position_data_preprocessing.py output.

Produces mean-position time series, occupancy heatmaps, bright/dark fractions,
and session-wise position scatter plots.
"""

from __future__ import annotations

from pathlib import Path

from path_setup import setup_script_paths

setup_script_paths(__file__)

import matplotlib.pyplot as plt
import numpy as np

from data_paths import POSITION_FIGURES_DIR, activity_hist_dir, position_hist_dir
from plotting_utils import format_x_axis
from presentation_style import (
    LEGEND_LOC,
    THESIS_COLORS,
    add_bright_dark_boundary_horizontal,
    add_dark_electrode_boundary,
    apply_presentation_style,
    save_thesis_figure,
    shade_dark_electrodes,
    shade_dark_region_above,
)

POSITION_SCATTER_COLOR = THESIS_COLORS[0]
POSITION_MEDIAN_COLOR = THESIS_COLORS[1]
from pulse_config import PULSE_TYPES
from position_utils import (
    DEFAULT_BRIGHT_DARK_BOUNDARY_M,
    ELECTRODE_SPACING_M,
    LINE_LENGTH_M,
    N_ELECTRODES,
    default_electrode_positions_m,
)

DARK_ELECTRODE_START = int(round(DEFAULT_BRIGHT_DARK_BOUNDARY_M / ELECTRODE_SPACING_M))

POSITION_METHOD = "peak_positive"

POSITION_PANEL_TIMESCALES = [
    ("hour", "24 h — hour"),
    ("month", "12 month — month"),
    ("month_since_start", "months since start"),
    ("year", "years since start — year"),
]


def estimate_total_recording_seconds() -> float:
    """Approximate active recording duration from activity histograms."""
    data_path = activity_hist_dir(PULSE_TYPES["all"]["hist_subdir"])
    counts = np.load(data_path / "berlin_dummypulses_count_hist_dict.npz")["day"]
    rates = np.load(data_path / "berlin_dummypulses_pulse_rate_hz_hist_dict.npz")["day"]
    total_pulses = float(np.nansum(counts))
    positive_rates = rates[(~np.isnan(rates)) & (rates > 0)]
    mean_hz = float(np.mean(positive_rates)) if positive_rates.size else np.nan
    if not mean_hz or np.isnan(mean_hz):
        return max(total_pulses, 1.0)
    return max(total_pulses / mean_hz, 1.0)


def load_metadata(data_path):
    metadata_path = data_path / "berlin_dummypulses_hist_metadata.npz"
    if metadata_path.exists():
        metadata = np.load(metadata_path)
        return (
            int(metadata["first_month_year"]),
            int(metadata["first_month_month"]),
            int(metadata["first_year"]),
        )
    return 2023, 1, 2023


def plot_mean_position(timescale, mean_position, save_path, suffix, meta):
    if timescale not in mean_position:
        return
    first_month_year, first_month_month, first_year = meta
    values = mean_position[timescale]
    x = np.arange(len(values))
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.plot(x, values, color="darkgreen")
    format_x_axis(
        ax,
        timescale,
        len(values),
        first_year=first_year,
        first_month_year=first_month_year,
        first_month_month=first_month_month,
        data=values if timescale == "year" else None,
    )
    ax.set_ylabel("mean head position (m)")
    ax.set_ylim(0, LINE_LENGTH_M)
    shade_dark_region_above(ax, DEFAULT_BRIGHT_DARK_BOUNDARY_M, y_max=LINE_LENGTH_M)
    add_bright_dark_boundary_horizontal(ax, DEFAULT_BRIGHT_DARK_BOUNDARY_M)
    ax.legend(loc=LEGEND_LOC)
    titles = {
        "minute": "24-hour mean position (1-min bins)",
        "hour": "24-hour mean position (hourly bins)",
        "month": "monthly mean position",
        "month_since_start": "monthly mean position since recording start",
        "year": "yearly mean position",
    }
    fig.suptitle(titles.get(timescale, timescale))
    plt.tight_layout()
    plt.savefig(save_path / f"{timescale}_mean_position{suffix}.png", dpi=300)
    plt.close()


def plot_occurrence_heatmap(occurrence, save_path, suffix, meta):
    for timescale, data in occurrence.items():
        if data.size == 0:
            continue
        normalized = data.astype(float)
        row_sums = normalized.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = np.divide(
                normalized, row_sums, out=np.zeros_like(normalized), where=row_sums > 0
            )

        fig, ax = plt.subplots(figsize=(20, 6))
        electrode_positions = default_electrode_positions_m()
        im = ax.imshow(
            normalized.T,
            aspect="auto",
            origin="lower",
            extent=[-0.5, data.shape[0] - 0.5, -0.5, len(electrode_positions) - 0.5],
            cmap="viridis",
        )
        ax.set_yticks(np.arange(len(electrode_positions)))
        ax.set_yticklabels([f"{pos:.2f}" for pos in electrode_positions])
        format_x_axis(
            ax,
            timescale,
            data.shape[0],
            first_year=meta[2],
            first_month_year=meta[0],
            first_month_month=meta[1],
            data=None,
        )
        ax.set_ylabel("position along line (m)")
        ax.set_title(f"spatial occupancy heatmap ({timescale} bins)")
        fig.colorbar(im, ax=ax, label="fraction of pulses")
        plt.tight_layout()
        plt.savefig(save_path / f"{timescale}_occupancy_heatmap{suffix}.png", dpi=300)
        plt.close()


def plot_bright_dark_fraction(bright_count, dark_count, save_path, suffix, meta):
    for timescale in ("minute", "hour"):
        if timescale not in bright_count:
            continue
        bright = bright_count[timescale].astype(float)
        dark = dark_count[timescale].astype(float)
        total = bright + dark
        fraction_dark = np.full_like(total, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            np.divide(dark, total, out=fraction_dark, where=total > 0)

        x = np.arange(len(fraction_dark))
        fig, ax = plt.subplots(figsize=(20, 6))
        ax.plot(x, fraction_dark, color="midnightblue")
        format_x_axis(
            ax,
            timescale,
            len(fraction_dark),
            first_year=meta[2],
            first_month_year=meta[0],
            first_month_month=meta[1],
        )
        ax.set_ylabel("fraction in dark area")
        ax.set_ylim(0, 1)
        ax.set_title(f"dark-area occupancy fraction ({timescale} bins)")
        plt.tight_layout()
        plt.savefig(save_path / f"{timescale}_dark_fraction{suffix}.png", dpi=300)
        plt.close()


def plot_session_mean_position_summary(timescale, arr, save_path, suffix, meta):
    x = np.arange(arr.shape[1])
    fig, ax = plt.subplots(figsize=(12, 5))

    for i in range(arr.shape[0]):
        valid = ~np.isnan(arr[i])
        ax.scatter(x[valid], arr[i][valid], alpha=0.2, s=10, color="tab:blue")

    active_arr = np.where(~np.isnan(arr), arr, np.nan)
    median = np.full(arr.shape[1], np.nan)
    p_lo = np.full(arr.shape[1], np.nan)
    p_hi = np.full(arr.shape[1], np.nan)
    for j in range(arr.shape[1]):
        values = active_arr[:, j]
        values = values[~np.isnan(values)]
        if values.size == 0:
            continue
        median[j] = np.median(values)
        p_lo[j] = np.percentile(values, 16)
        p_hi[j] = np.percentile(values, 84)

    ax.plot(x, np.ma.masked_invalid(median), color="tab:red", linewidth=1.5, label="median")
    ax.fill_between(
        x,
        np.ma.masked_invalid(p_lo),
        np.ma.masked_invalid(p_hi),
        color="tab:red",
        alpha=0.25,
        label="16-84th pct",
    )
    format_x_axis(
        ax,
        timescale,
        arr.shape[1],
        first_year=meta[2],
        first_month_year=meta[0],
        first_month_month=meta[1],
    )
    ax.set_ylabel("mean head position (m)")
    ax.set_ylim(0, LINE_LENGTH_M)
    shade_dark_region_above(ax, DEFAULT_BRIGHT_DARK_BOUNDARY_M, y_max=LINE_LENGTH_M)
    add_bright_dark_boundary_horizontal(ax, DEFAULT_BRIGHT_DARK_BOUNDARY_M)
    ax.legend(loc=LEGEND_LOC, fontsize="small")
    plt.title(f"session-wise mean position ({timescale})")
    plt.tight_layout()
    plt.savefig(save_path / f"{timescale}_session_mean_position{suffix}.png", dpi=300)
    plt.close()


def plot_overall_position_distribution(occurrence_hour, save_path, suffix):
    """Bar chart of pulse rate (Hz) per electrode collapsed over all hours."""
    counts = occurrence_hour.sum(axis=0)
    if counts.sum() == 0:
        return
    electrode_indices = np.arange(min(len(counts), N_ELECTRODES))
    rates_hz = counts[: len(electrode_indices)].astype(float) / estimate_total_recording_seconds()

    fig, ax = plt.subplots(figsize=(12, 5))
    ymax = max(float(np.max(rates_hz)) * 1.1, 0.01)
    ax.set_ylim(0, ymax)
    ax.set_xlim(-0.5, len(electrode_indices) - 0.5)
    shade_dark_electrodes(
        ax,
        DARK_ELECTRODE_START,
        n_electrodes=len(electrode_indices),
        y_min=0,
        y_max=ymax,
    )
    add_dark_electrode_boundary(ax, DARK_ELECTRODE_START)
    ax.bar(electrode_indices, rates_hz, width=0.7, color=POSITION_SCATTER_COLOR, linewidth=0, zorder=3)
    ax.set_xticks(electrode_indices)
    ax.set_xlabel("electrode")
    ax.set_ylabel("pulse rate (Hz)")
    ax.set_title("overall position distribution (hourly bins collapsed)")
    ax.legend(loc=LEGEND_LOC, fontsize=10)
    ax.set_xlim(-0.5, len(electrode_indices) - 0.5)
    plt.tight_layout()
    filename = f"overall_position_distribution{suffix}.png"
    plt.savefig(save_path / filename, dpi=300)
    save_thesis_figure(f"position_estimation/{filename}")
    plt.close()


def plot_position_panel_figure(session_data, save_path, suffix, meta):
    """Four-panel session-wise median position overview."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    first_month_year, first_month_month, first_year = meta
    for ax, (timescale, title) in zip(axes.ravel(), POSITION_PANEL_TIMESCALES):
        if timescale not in session_data.files:
            ax.set_axis_off()
            continue
        arr = session_data[timescale]
        x = np.arange(arr.shape[1])
        for i in range(arr.shape[0]):
            valid = ~np.isnan(arr[i])
            ax.scatter(x[valid], arr[i][valid], alpha=0.15, s=8, color=POSITION_SCATTER_COLOR)

        median = np.full(arr.shape[1], np.nan)
        p_lo = np.full(arr.shape[1], np.nan)
        p_hi = np.full(arr.shape[1], np.nan)
        for j in range(arr.shape[1]):
            values = arr[:, j]
            values = values[~np.isnan(values)]
            if values.size == 0:
                continue
            median[j] = np.median(values)
            p_lo[j] = np.percentile(values, 16)
            p_hi[j] = np.percentile(values, 84)

        ax.plot(x, np.ma.masked_invalid(median), color=POSITION_MEDIAN_COLOR, linewidth=2.5, label="median")
        ax.fill_between(
            x,
            np.ma.masked_invalid(p_lo),
            np.ma.masked_invalid(p_hi),
            color=POSITION_MEDIAN_COLOR,
            alpha=0.25,
            label="16-84th pct",
        )
        format_x_axis(
            ax,
            timescale,
            arr.shape[1],
            first_year=first_year,
            first_month_year=first_month_year,
            first_month_month=first_month_month,
        )
        ax.set_ylim(0, LINE_LENGTH_M)
        shade_dark_region_above(ax, DEFAULT_BRIGHT_DARK_BOUNDARY_M, y_max=LINE_LENGTH_M)
        add_bright_dark_boundary_horizontal(ax, DEFAULT_BRIGHT_DARK_BOUNDARY_M)
        ax.set_ylabel("mean head position (m)")
        ax.set_title(title)
        ax.legend(loc=LEGEND_LOC, fontsize=10)
        ax.grid(True, alpha=0.25)

    fig.suptitle("Spatial usage over time (session-wise median + percentiles)")
    plt.tight_layout()
    filename = f"position_panels{suffix}.png"
    plt.savefig(save_path / filename, dpi=300)
    save_thesis_figure(f"position_estimation/{filename}")
    plt.close()


def main():
    apply_presentation_style()
    suffix = f"_{POSITION_METHOD[:4]}"
    data_path = position_hist_dir(POSITION_METHOD)
    save_path = POSITION_FIGURES_DIR / POSITION_METHOD
    save_path.mkdir(parents=True, exist_ok=True)
    meta = load_metadata(data_path)

    mean_position = np.load(
        data_path / f"berlin_position_{POSITION_METHOD}_mean_position_hist_dict.npz"
    )
    mean_position = {k: mean_position[k] for k in mean_position.files}

    bright_count = np.load(
        data_path / f"berlin_position_{POSITION_METHOD}_bright_count_hist_dict.npz"
    )
    bright_count = {k: bright_count[k] for k in bright_count.files}
    dark_count = np.load(
        data_path / f"berlin_position_{POSITION_METHOD}_dark_count_hist_dict.npz"
    )
    dark_count = {k: dark_count[k] for k in dark_count.files}

    occurrence = np.load(
        data_path / f"berlin_position_{POSITION_METHOD}_occurrence_hist.npz"
    )
    occurrence = {k: occurrence[k] for k in occurrence.files}

    for timescale in ("minute", "hour", "month", "month_since_start", "year"):
        plot_mean_position(timescale, mean_position, save_path, suffix, meta)

    plot_occurrence_heatmap(occurrence, save_path, suffix, meta)
    plot_bright_dark_fraction(bright_count, dark_count, save_path, suffix, meta)

    if "hour" in occurrence:
        plot_overall_position_distribution(occurrence["hour"], save_path, suffix)

    session_path = data_path / f"berlin_position_{POSITION_METHOD}_session_mean_position.npz"
    if session_path.exists():
        session_data = np.load(session_path)
        for timescale in session_data.files:
            if timescale == "day":
                continue
            plot_session_mean_position_summary(
                timescale, session_data[timescale], save_path, suffix, meta
            )
        plot_position_panel_figure(session_data, save_path, suffix, meta)


if __name__ == "__main__":
    main()
