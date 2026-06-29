"""Plot spatial eel usage along the Berlin line logger.

Analysis part: position visualization (Part 5b of Berlin activity analysis).
Dependencies: data_paths, position_utils; requires position_data_preprocessing.py output.

Produces mean-position time series, occupancy heatmaps, bright/dark fractions,
and session-wise position scatter plots.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from path_setup import setup_script_paths

setup_script_paths(__file__)

import matplotlib.pyplot as plt
import numpy as np
from dateutil.relativedelta import relativedelta

from data_paths import POSITION_FIGURES_DIR, position_hist_dir
from position_utils import (
    DEFAULT_BRIGHT_DARK_BOUNDARY_M,
    LINE_LENGTH_M,
    default_electrode_positions_m,
)

POSITION_METHOD = "peak_positive"


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


def format_x_axis(axis, timescale, n_bins, first_year, first_month_year, first_month_month, data=None):
    x = np.arange(n_bins)
    if timescale == "year" and data is not None:
        valid_indices = np.where(~np.isnan(data))[0]
        if len(valid_indices) > 0:
            n_bins = valid_indices[-1] + 1
            x = np.arange(n_bins)

    if timescale == "minute":
        tick_positions = np.arange(0, n_bins, 60)
        tick_labels = [f"{h:02d}:00" for h in range(len(tick_positions))]
        xlabel = "time of day"
    elif timescale == "hour":
        tick_positions = x
        tick_labels = [f"{h:02d}:00" for h in x]
        xlabel = "time of day"
    elif timescale == "month":
        tick_positions = x
        tick_labels = [datetime(2000, m, 1).strftime("%b") for m in range(1, 13)]
        xlabel = "month"
    elif timescale == "month_since_start":
        tick_step = max(1, n_bins // 18)
        tick_positions = x[::tick_step]
        first_month = datetime(first_month_year, first_month_month, 1)
        all_labels = [
            (first_month + relativedelta(months=i)).strftime("%b %Y") for i in range(n_bins)
        ]
        tick_labels = [all_labels[i] for i in tick_positions]
        xlabel = "month since recording start"
    elif timescale == "year":
        tick_positions = x
        tick_labels = [str(y) for y in range(first_year, first_year + n_bins)]
        xlabel = "year"
    else:
        tick_positions = x
        tick_labels = [str(i) for i in x]
        xlabel = "bin index"

    axis.set_xticks(tick_positions)
    axis.set_xticklabels(tick_labels, rotation=45, ha="right")
    axis.set_xlabel(xlabel)
    axis.set_xlim(-0.5, n_bins - 0.5)


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
        first_year,
        first_month_year,
        first_month_month,
        data=values if timescale == "year" else None,
    )
    ax.set_ylabel("mean head position (m)")
    ax.set_ylim(0, LINE_LENGTH_M)
    ax.axhline(
        DEFAULT_BRIGHT_DARK_BOUNDARY_M,
        color="gray",
        linestyle="--",
        linewidth=1,
        label="bright/dark boundary",
    )
    ax.legend(loc="upper right")
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
            ax, timescale, data.shape[0], meta[2], meta[0], meta[1], data=None
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
            meta[2],
            meta[0],
            meta[1],
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
    format_x_axis(ax, timescale, arr.shape[1], meta[2], meta[0], meta[1])
    ax.set_ylabel("mean head position (m)")
    ax.set_ylim(0, LINE_LENGTH_M)
    ax.axhline(DEFAULT_BRIGHT_DARK_BOUNDARY_M, color="gray", linestyle="--", linewidth=1)
    ax.legend(loc="upper right", fontsize="small")
    plt.title(f"session-wise mean position ({timescale})")
    plt.tight_layout()
    plt.savefig(save_path / f"{timescale}_session_mean_position{suffix}.png", dpi=300)
    plt.close()


def plot_overall_position_distribution(occurrence_hour, save_path, suffix):
    """Bar chart of pulse counts per electrode collapsed over all hours."""
    counts = occurrence_hour.sum(axis=0)
    if counts.sum() == 0:
        return
    positions = default_electrode_positions_m()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(positions, counts, width=0.2, color="steelblue", edgecolor="white")
    ax.axvline(
        DEFAULT_BRIGHT_DARK_BOUNDARY_M,
        color="gray",
        linestyle="--",
        linewidth=1,
        label="bright/dark boundary",
    )
    ax.set_xlabel("position along line (m)")
    ax.set_ylabel("pulse count")
    ax.set_title("overall position distribution (hourly bins collapsed)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path / f"overall_position_distribution{suffix}.png", dpi=300)
    plt.close()


def main():
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


if __name__ == "__main__":
    main()
