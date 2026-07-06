"""Plot pulse-rate histograms from eel_data_preprocessing.py output.

Analysis part: pulse activity visualization (Part 1b of Berlin activity analysis).
Dependencies: data_paths, pulse_config, plotting_utils; requires preprocessing .npz files.

Generates global line plots and session-wise scatter plots with median and
bootstrap confidence intervals (or percentile bands) for each timescale.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from path_setup import setup_script_paths

setup_script_paths(__file__)

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from data_paths import activity_hist_dir, processed_figures_dir
from plotting_utils import format_x_axis
from presentation_style import apply_presentation_style, pulse_shape_color
from pulse_config import PULSE_TYPES, select_pulse_type

USE_BOOTSTRAP = True
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_CI_LEVEL = 95

GLOBAL_PLOTS = [
    ("minute", "24-hour pulse rate histogram (1-min bins)", "24h_minute{suffix}.png"),
    ("hour", "24-hour pulse rate histogram (hourly bins)", "24h_hour{suffix}.png"),
    ("month", "monthly pulse rate histogram (monthly bins)", "12month_month{suffix}.png"),
    (
        "month_since_start",
        "monthly pulse rate histogram since recording start (monthly bins)",
        "months_since_start_month{suffix}.png",
    ),
    ("year", "yearly pulse rate histogram (yearly bins)", "years_year{suffix}.png"),
]

CIRCADIAN_PANEL_TIMESCALES = [
    ("hour", "24 h — hour"),
    ("month", "12 month — month"),
    ("month_since_start", "months since start"),
    ("year", "years since start — year"),
]


def start_y_axis_at_zero(axes):
    for axis in np.ravel(axes):
        axis.set_ylim(bottom=0)


def load_histogram_metadata(data_path):
    metadata_path = data_path / "berlin_dummypulses_hist_metadata.npz"
    if metadata_path.exists():
        metadata = np.load(metadata_path)
        return {
            "first_month_year": int(metadata["first_month_year"]),
            "first_month_month": int(metadata["first_month_month"]),
            "first_year": int(metadata["first_year"]),
        }
    return {"first_month_year": 2023, "first_month_month": 1, "first_year": 2023}


def nan_summary(arr):
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

    return median, p_lo, p_hi


def bootstrap_ci_summary(arr, n_bootstrap=10000, ci_level=95):
    median = np.full(arr.shape[1], np.nan)
    ci_lo = np.full(arr.shape[1], np.nan)
    ci_hi = np.full(arr.shape[1], np.nan)
    alpha = (100 - ci_level) / 2

    for j in tqdm(range(arr.shape[1]), desc="Computing bootstrap CI"):
        values = arr[:, j]
        values = values[~np.isnan(values)]
        if values.size == 0:
            continue

        median[j] = np.median(values)
        bootstrap_medians = [
            np.median(np.random.choice(values, size=len(values), replace=True))
            for _ in range(n_bootstrap)
        ]
        ci_lo[j] = np.percentile(bootstrap_medians, alpha)
        ci_hi[j] = np.percentile(bootstrap_medians, 100 - alpha)

    return median, ci_lo, ci_hi


def plot_pulse_rate(
    pulse_rate_hist_dict,
    save_path,
    axis_meta,
    timescale,
    title,
    filename,
):
    if timescale not in pulse_rate_hist_dict:
        return

    rate = pulse_rate_hist_dict[timescale]
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.plot(np.arange(len(rate)), rate, color="green")
    format_x_axis(
        ax,
        timescale,
        len(rate),
        data=rate if timescale == "year" else None,
        **axis_meta,
    )
    ax.set_ylabel("Pulse Rate (Hz)")
    start_y_axis_at_zero(ax)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path / filename, dpi=300)
    plt.close()


def plot_session_median_on_axis(
    ax,
    arr,
    timescale,
    axis_meta,
    *,
    color,
    title,
):
    x = np.arange(arr.shape[1])
    for i in range(arr.shape[0]):
        valid = ~np.isnan(arr[i])
        ax.scatter(x[valid], arr[i][valid], alpha=0.15, s=8, color=color)

    active_arr = np.where(arr > 0, arr, np.nan)
    if USE_BOOTSTRAP:
        median, ci_lo, ci_hi = bootstrap_ci_summary(
            active_arr,
            n_bootstrap=BOOTSTRAP_ITERATIONS,
            ci_level=BOOTSTRAP_CI_LEVEL,
        )
        band_label = f"bootstrap {BOOTSTRAP_CI_LEVEL}% CI"
    else:
        median, ci_lo, ci_hi = nan_summary(active_arr)
        band_label = "active-session 16-84th pct"

    ax.plot(
        x,
        np.ma.masked_invalid(median),
        color=color,
        linewidth=2.8,
        label="active-session median",
    )
    ax.fill_between(
        x,
        np.ma.masked_invalid(ci_lo),
        np.ma.masked_invalid(ci_hi),
        color=color,
        alpha=0.25,
        label=band_label,
    )
    format_x_axis(
        ax,
        timescale,
        arr.shape[1],
        data=median if timescale == "year" else None,
        **axis_meta,
    )
    ax.set_ylabel("Pulse rate (Hz)")
    ax.set_ylim(bottom=0)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.25)


def plot_circadian_panel_figure(
    session_data,
    save_path,
    suffix,
    axis_meta,
    pulse_label,
    pulse_type_key,
):
    """Four-panel overview: 24h, 12 month, months since start, years."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    color = pulse_shape_color(pulse_type_key)
    for ax, (timescale, title) in zip(axes.ravel(), CIRCADIAN_PANEL_TIMESCALES):
        if timescale not in session_data.files:
            ax.set_axis_off()
            continue
        plot_session_median_on_axis(
            ax,
            session_data[timescale],
            timescale,
            axis_meta,
            color=color,
            title=title,
        )
    fig.suptitle(f"Pulse rate over time — {pulse_label}")
    plt.tight_layout()
    fig.savefig(save_path / f"circadian_panels{suffix}.png", dpi=300)
    plt.close(fig)


def plot_session_pulse_rate_summary(
    save_path,
    axis_meta,
    suffix,
    timescale,
    arr,
):
    if suffix == "_dp":
        color = pulse_shape_color("double")
    elif suffix == "_wide":
        color = pulse_shape_color("wide")
    elif suffix == "_fat":
        color = pulse_shape_color("fat")
    else:
        color = pulse_shape_color("all")

    fig, ax = plt.subplots(figsize=(12, 5))
    plot_session_median_on_axis(
        ax,
        arr,
        timescale,
        axis_meta,
        color=color,
        title=timescale,
    )
    plt.tight_layout()
    plt.savefig(save_path / f"{timescale}{suffix}.png", dpi=300)
    plt.close()


def main():
    apply_presentation_style()
    pulse_type = select_pulse_type(default="all")
    pulse_config = PULSE_TYPES[pulse_type]
    suffix = pulse_config["suffix"]
    data_path = activity_hist_dir(pulse_config["hist_subdir"])
    save_path = processed_figures_dir(pulse_config["figures_subdir"])
    save_path.mkdir(parents=True, exist_ok=True)

    pulse_rate_data = np.load(data_path / "berlin_dummypulses_pulse_rate_hz_hist_dict.npz")
    pulse_rate_hist_dict = {k: pulse_rate_data[k] for k in pulse_rate_data.files}
    axis_meta = load_histogram_metadata(data_path)

    for timescale, title, filename in GLOBAL_PLOTS:
        plot_pulse_rate(
            pulse_rate_hist_dict,
            save_path,
            axis_meta,
            timescale,
            title,
            filename.format(suffix=suffix),
        )

    session_data = np.load(data_path / "berlin_dummypulses_session_pulse_rate_hz.npz")
    for timescale in session_data.files:
        if timescale == "day":
            continue
        plot_session_pulse_rate_summary(
            save_path,
            axis_meta,
            suffix,
            timescale,
            session_data[timescale],
        )

    plot_circadian_panel_figure(
        session_data,
        save_path,
        suffix,
        axis_meta,
        pulse_config["label"],
        pulse_type,
    )

    print(f"Saved figures to {save_path}")


if __name__ == "__main__":
    main()
