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

from correlations.mating_notes_utils import extract_mating_events
from data_paths import activity_hist_dir, processed_figures_dir, resolve_activity_hist_npz
from plotting_utils import format_x_axis
from presentation_style import apply_presentation_style, legend_on_upper_right_subplot, pulse_shape_color, save_thesis_figure
from pulse_config import PULSE_TYPES, PULSE_TYPE_DISPLAY_ORDER, select_pulse_type

USE_BOOTSTRAP = True
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_CI_LEVEL = 95
MATING_MARKER_COLOR = "crimson"

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
    metadata_path = resolve_activity_hist_npz(data_path, "hist_metadata")
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


def month_since_start_index(event_time, first_month_year: int, first_month_month: int) -> int:
    """Return 0-based month_since_start bin for an event timestamp."""
    return (event_time.year - first_month_year) * 12 + (event_time.month - first_month_month)


def mark_mating_on_month_since_start(
    ax,
    axis_meta: dict,
    n_bins: int,
    mating_events=None,
    *,
    label: str = "Mating note",
):
    """Overlay vertical markers for mating notes on a month_since_start axis."""
    if mating_events is None:
        mating_events = extract_mating_events()
    if not mating_events:
        return

    first_year = int(axis_meta["first_month_year"])
    first_month = int(axis_meta["first_month_month"])
    marked = False
    y_max = ax.get_ylim()[1]
    for event in mating_events:
        event_time = event["event_time"]
        idx = month_since_start_index(event_time, first_year, first_month)
        if idx < 0 or idx >= n_bins:
            continue
        ax.axvline(
            idx,
            color=MATING_MARKER_COLOR,
            linestyle="--",
            alpha=0.7,
            linewidth=1.2,
        )
        ax.scatter(
            [idx],
            [0.97 * y_max],
            marker="v",
            color=MATING_MARKER_COLOR,
            s=55,
            zorder=6,
            label=label if not marked else None,
        )
        marked = True
    if marked:
        ax.legend()


def plot_pulse_rate(
    pulse_rate_hist_dict,
    save_path,
    axis_meta,
    timescale,
    title,
    filename,
    *,
    mating_events=None,
    save_exploratory: bool = False,
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
    if timescale == "month_since_start" and mating_events is not None:
        mark_mating_on_month_since_start(ax, axis_meta, len(rate), mating_events)
        title = f"{title} — mating notes marked"
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path / filename, dpi=300)
    if save_exploratory:
        save_thesis_figure(f"exploratory_mating_corr/{filename}", fig)
    plt.close()


def plot_session_median_on_axis(
    ax,
    arr,
    timescale,
    axis_meta,
    *,
    color,
    title,
    show_scatter=True,
    median_label="active-session median",
    decorate_axis=True,
    include_band_in_legend=True,
    show_legend=True,
):
    x = np.arange(arr.shape[1])
    if show_scatter:
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
        band_label = f"bootstrap {BOOTSTRAP_CI_LEVEL}% CI" if include_band_in_legend else None
    else:
        median, ci_lo, ci_hi = nan_summary(active_arr)
        band_label = "active-session 16-84th pct" if include_band_in_legend else None

    ax.plot(
        x,
        np.ma.masked_invalid(median),
        color=color,
        linewidth=2.8,
        label=median_label,
    )
    ax.fill_between(
        x,
        np.ma.masked_invalid(ci_lo),
        np.ma.masked_invalid(ci_hi),
        color=color,
        alpha=0.25,
        label=band_label,
    )
    if decorate_axis:
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
        if show_legend:
            ax.legend()
        ax.grid(True, alpha=0.25)
    else:
        ax.set_ylim(bottom=0)


def plot_circadian_panel_figure(
    session_data,
    save_path,
    suffix,
    axis_meta,
    pulse_label,
    pulse_type_key,
):
    """Four-panel overview: 24h, 12 month, months since start, years."""
    apply_presentation_style()
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
            show_legend=False,
        )
    legend_on_upper_right_subplot(axes)
    fig.suptitle(f"Pulse rate over time — {pulse_label}")
    plt.tight_layout()
    filename = f"circadian_panels{suffix}.png"
    fig.savefig(save_path / filename, dpi=300)
    save_thesis_figure(f"activity_timescales/{filename}", fig)
    plt.close(fig)


def plot_circadian_panels_all_shapes(save_path=None):
    """Four-panel overview with all pulse categories overlaid."""
    apply_presentation_style()
    if save_path is None:
        save_path = processed_figures_dir(PULSE_TYPES["all"]["figures_subdir"])
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    axis_meta = load_histogram_metadata(
        activity_hist_dir(PULSE_TYPES["all"]["hist_subdir"])
    )
    session_by_pulse = {}
    for pulse_type, pulse_config in PULSE_TYPES.items():
        data_path = activity_hist_dir(pulse_config["hist_subdir"])
        session_npz = resolve_activity_hist_npz(data_path, "session_pulse_rate_hz")
        if session_npz.exists():
            session_by_pulse[pulse_type] = np.load(session_npz)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    for ax, (timescale, panel_title) in zip(axes.ravel(), CIRCADIAN_PANEL_TIMESCALES):
        plotted = False
        for pulse_type in PULSE_TYPE_DISPLAY_ORDER:
            session_data = session_by_pulse.get(pulse_type)
            if session_data is None or timescale not in session_data.files:
                continue
            plot_session_median_on_axis(
                ax,
                session_data[timescale],
                timescale,
                axis_meta,
                color=pulse_shape_color(pulse_type),
                title=panel_title,
                show_scatter=False,
                median_label=PULSE_TYPES[pulse_type]["label"],
                decorate_axis=False,
                include_band_in_legend=False,
            )
            plotted = True
        if not plotted:
            ax.set_axis_off()
            continue
        year_data = None
        if timescale == "year":
            all_arr = session_by_pulse["all"][timescale]
            active_arr = np.where(all_arr > 0, all_arr, np.nan)
            year_data = np.nanmedian(active_arr, axis=0)
        format_x_axis(
            ax,
            timescale,
            session_by_pulse["all"][timescale].shape[1],
            data=year_data,
            **axis_meta,
        )
        ax.set_ylabel("Pulse rate (Hz)")
        ax.set_ylim(bottom=0)
        ax.set_title(panel_title)
        ax.grid(True, alpha=0.25)

    legend_on_upper_right_subplot(axes)
    fig.suptitle("Pulse rate over time — all pulse categories")
    plt.tight_layout()
    filename = "circadian_panels_all_shapes.png"
    out_path = save_path / filename
    fig.savefig(out_path, dpi=300)
    save_thesis_figure(f"activity_timescales/{filename}", fig)
    plt.close(fig)
    print(f"Saved combined circadian panels to {out_path}")


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

    pulse_rate_data = np.load(resolve_activity_hist_npz(data_path, "pulse_rate_hz_hist_dict"))
    pulse_rate_hist_dict = {k: pulse_rate_data[k] for k in pulse_rate_data.files}
    axis_meta = load_histogram_metadata(data_path)
    mating_events = extract_mating_events() if pulse_type == "all" else None

    for timescale, title, filename in GLOBAL_PLOTS:
        plot_pulse_rate(
            pulse_rate_hist_dict,
            save_path,
            axis_meta,
            timescale,
            title,
            filename.format(suffix=suffix),
            mating_events=mating_events if timescale == "month_since_start" else None,
            save_exploratory=(pulse_type == "all" and timescale == "month_since_start"),
        )

    session_data = np.load(resolve_activity_hist_npz(data_path, "session_pulse_rate_hz"))
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

    plot_circadian_panels_all_shapes()

    print(f"Saved figures to {save_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--all-shapes-panels":
        apply_presentation_style()
        plot_circadian_panels_all_shapes()
    else:
        main()
