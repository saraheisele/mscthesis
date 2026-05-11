# %%
from __future__ import annotations

import calendar
from datetime import datetime
from pathlib import Path
from typing import Iterable  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

#################################
############# CONFIG #############
#################################
# Set to True to analyze only double pulses
# Set to False to analyze all detected pulses
DOUBLE_PEAKS_ONLY = False

#################################
############# LOAD #############
#################################

# set filename suffix and save path based on configuration
suffix = "_dp" if DOUBLE_PEAKS_ONLY else "_all"
figures_subdir = "double_pulses" if DOUBLE_PEAKS_ONLY else "all_pulses"
hist_subdir = "double_pulses_hist" if DOUBLE_PEAKS_ONLY else "all_pulses_hist"

# load histogram dictionaries from .npz files
base_path = Path(
    "/home/eisele/wrk/mscthesis/data/intermediate/eels-mfn2021_dummy_activity_histograms"
)
data_path = base_path / hist_subdir
count_data = np.load(data_path / "berlin_dummypulses_count_hist_dict.npz")

histogram_dict = {k: count_data[k] for k in count_data.files}

rec_count_data = np.load(data_path / "berlin_dummypulses_rec_hist_dict.npz")

rec_hist_dict = {k: rec_count_data[k] for k in rec_count_data.files}

rec_time_data = np.load(data_path / "berlin_dummypulses_rec_time_hist_dict.npz")

rec_time_hist_dict = {k: rec_time_data[k] for k in rec_time_data.files}

#################################
############# PLOTS #############
#################################

save_path = Path(f"/home/eisele/wrk/mscthesis/figures/{figures_subdir}/")
save_path.mkdir(parents=True, exist_ok=True)


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Divide while returning NaN where the denominator is zero."""

    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    out = np.full(
        np.broadcast_shapes(numerator.shape, denominator.shape), np.nan, dtype=float
    )
    np.divide(numerator, denominator, out=out, where=denominator != 0)
    return out


def _month_start_ticks_for_year(year: int, n_bins: int) -> tuple[np.ndarray, list[str]]:
    """Return month-start tick positions and labels for a daily series."""

    month_lengths = [calendar.monthrange(year, month)[1] for month in range(1, 13)]
    month_starts = np.concatenate(([0], np.cumsum(month_lengths)[:-1]))
    month_labels = [datetime(year, month, 1).strftime("%b") for month in range(1, 13)]

    valid = month_starts < n_bins
    return month_starts[valid], [
        label for label, keep in zip(month_labels, valid) if keep
    ]


def _timescale_axis_specs(
    timescale: str, n_bins: int
) -> tuple[np.ndarray, list[str], str]:
    """Axis ticks/labels for the session-wise median plots."""

    if timescale == "minute":
        ticks = np.arange(0, n_bins, 60)
        labels = [f"{h:02d}:00" for h in range(len(ticks))]
        xlabel = "time of day"
    elif timescale == "hour":
        ticks = np.arange(n_bins)
        labels = [f"{h:02d}:00" for h in range(n_bins)]
        xlabel = "time of day"
    elif timescale == "day":
        ticks, labels = _month_start_ticks_for_year(2023, n_bins)
        xlabel = "month"
    elif timescale == "month":
        ticks = np.arange(n_bins)
        labels = [
            datetime(2000, month, 1).strftime("%b") for month in range(1, n_bins + 1)
        ]
        xlabel = "month"
    elif timescale == "year":
        ticks = np.arange(n_bins)
        labels = [str(year) for year in range(2023, 2023 + n_bins)]
        xlabel = "year"
    else:
        ticks = np.arange(n_bins)
        labels = [str(i) for i in ticks]
        xlabel = "bin index"

    return ticks, labels, xlabel


def _plot_binned_histogram(
    counts: np.ndarray,
    normalized_counts: np.ndarray,
    rate: np.ndarray,
    normalized_rate: np.ndarray,
    *,
    title: str,
    save_name: str,
    xticks: np.ndarray,
    xtick_labels: list[str],
    xlabel: str,
) -> None:
    """Render the repeated 2x2 histogram overview plots."""

    x = np.arange(len(rate))
    fig, ax = plt.subplots(2, 2, figsize=(20, 10), sharex=True)

    ax[0, 0].bar(x, counts)
    ax[1, 0].bar(x, normalized_counts, color="green")
    ax[0, 1].plot(x, rate)
    ax[1, 1].plot(x, normalized_rate, color="green")

    for a in ax.flat:
        a.set_xticks(xticks)
        a.set_xticklabels(xtick_labels, rotation=45)
        a.set_xlim(-0.5, len(rate) - 0.5)
        a.set_xlabel(xlabel)

    ax[0, 0].set_ylabel("pulse count")
    ax[1, 0].set_ylabel("normalized pulse count")
    ax[0, 1].set_ylabel("firing rate [Hz]")
    ax[1, 1].set_ylabel("normalized firing rate [Hz]")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path / save_name, dpi=300)
    plt.show()


# %%
## 24 hours - minute bins
firing_rate_minute = histogram_dict["minute"] / 60.0  # Hz
firing_rate_minute_normalized = _safe_divide(
    histogram_dict["minute"], rec_time_hist_dict["minute"]
)

minute_ticks = np.arange(0, len(firing_rate_minute), 60)
minute_labels = [f"{h:02d}:00" for h in range(len(minute_ticks))]
_plot_binned_histogram(
    histogram_dict["minute"],
    _safe_divide(histogram_dict["minute"], rec_time_hist_dict["minute"]),
    firing_rate_minute,
    firing_rate_minute_normalized,
    title="24-hour histogram (1-min bins)",
    save_name=f"24h_minute{suffix}.png",
    xticks=minute_ticks,
    xtick_labels=minute_labels,
    xlabel="time of day",
)

# %%
## 24 hours - hourly bins
firing_rate_hour = histogram_dict["hour"] / 60.0 / 60.0  # Hz
firing_rate_hour_normalized = _safe_divide(
    histogram_dict["hour"], rec_time_hist_dict["hour"]
)

hour_ticks = np.arange(len(firing_rate_hour))
hour_labels = [f"{h:02d}:00" for h in range(len(hour_ticks))]
_plot_binned_histogram(
    histogram_dict["hour"],
    _safe_divide(histogram_dict["hour"], rec_time_hist_dict["hour"]),
    firing_rate_hour,
    firing_rate_hour_normalized,
    title="24-hour histogram (hourly bins)",
    save_name=f"24h_hour{suffix}.png",
    xticks=hour_ticks,
    xtick_labels=hour_labels,
    xlabel="time of day",
)

# %%
## months - daily bins
firing_rate_day = histogram_dict["day"] / 24 / 60.0 / 60  # Hz
firing_rate_day_normalized = _safe_divide(
    histogram_dict["day"], rec_time_hist_dict["day"]
)

day_ticks, day_labels = _month_start_ticks_for_year(2023, len(firing_rate_day))
_plot_binned_histogram(
    histogram_dict["day"],
    _safe_divide(histogram_dict["day"], rec_time_hist_dict["day"]),
    firing_rate_day,
    firing_rate_day_normalized,
    title="monthly histogram (daily bins)",
    save_name=f"12month_day{suffix}.png",
    xticks=day_ticks,
    xtick_labels=day_labels,
    xlabel="month",
)

# %%
## months - monthly bins
firing_rate_month = histogram_dict["month"] / 30 / 24 / 60 / 60  # Hz
firing_rate_month_normalized = _safe_divide(
    histogram_dict["month"], rec_time_hist_dict["month"]
)

month_ticks = np.arange(len(firing_rate_month))
month_labels = [
    datetime(2000, m, 1).strftime("%b") for m in range(1, len(firing_rate_month) + 1)
]
_plot_binned_histogram(
    histogram_dict["month"],
    _safe_divide(histogram_dict["month"], rec_time_hist_dict["month"]),
    firing_rate_month,
    firing_rate_month_normalized,
    title="monthly histogram (monthly bins)",
    save_name=f"12month_month{suffix}.png",
    xticks=month_ticks,
    xtick_labels=month_labels,
    xlabel="month",
)

# %%
## years
firing_rate_year = histogram_dict["year"] / 365 / 24 / 60 / 60  # Hz
firing_rate_year_normalized = _safe_divide(
    histogram_dict["year"], rec_time_hist_dict["year"]
)

year_ticks = np.arange(len(firing_rate_year))
year_labels = [str(y) for y in range(2023, 2023 + len(histogram_dict["year"]))]
_plot_binned_histogram(
    histogram_dict["year"],
    _safe_divide(histogram_dict["year"], rec_time_hist_dict["year"]),
    firing_rate_year,
    firing_rate_year_normalized,
    title="yearly histogram (yearly bins)",
    save_name=f"years_year{suffix}.png",
    xticks=year_ticks,
    xtick_labels=year_labels,
    xlabel="year",
)

# %%
# normalized fr line plots per session, one per timescale
# The arrays in the .npz may not all use the same axis order.
# We normalize them so that the last axis is always the bin axis.


def _ensure_sessions_by_bins(
    arr: np.ndarray, expected_bins: int, timescale: str
) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2:
        raise ValueError(
            f"Expected a 2D array for '{timescale}', got shape {arr.shape}."
        )

    if arr.shape[1] == expected_bins:
        return arr
    if arr.shape[0] == expected_bins:
        return arr.T

    raise ValueError(
        f"Could not orient '{timescale}' array with shape {arr.shape} to match "
        f"expected bin count {expected_bins}. Check how the .npz was created."
    )


def _plot_session_firing_rates(timescale: str, arr: np.ndarray) -> None:
    expected_bins = len(histogram_dict[timescale])
    arr = _ensure_sessions_by_bins(arr, expected_bins, timescale)
    x = np.arange(arr.shape[1])
    ticks, labels, xlabel = _timescale_axis_specs(timescale, arr.shape[1])

    fig, ax = plt.subplots(figsize=(10, 4.5))

    for session in arr:
        if np.all(np.isnan(session)):
            continue
        ax.plot(x, session, alpha=0.15, color="tab:blue", linewidth=0.7)

    median = np.nanmedian(arr, axis=0)
    p_lo = np.nanpercentile(arr, 16, axis=0)
    p_hi = np.nanpercentile(arr, 84, axis=0)

    median_m = np.ma.masked_invalid(median)
    p_lo_m = np.ma.masked_invalid(p_lo)
    p_hi_m = np.ma.masked_invalid(p_hi)

    ax.plot(x, median_m, color="tab:red", linewidth=2.5, label="median")
    ax.fill_between(x, p_lo_m, p_hi_m, color="tab:red", alpha=0.25, label="16–84th pct")

    ax.set_title(timescale)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_xlim(-0.5, arr.shape[1] - 0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("firing rate (Hz)")
    ax.legend(loc="upper right", fontsize="small")
    fig.tight_layout()
    fig.savefig(save_path / f"{timescale}{suffix}.png", dpi=300)
    plt.show()


fr_data = np.load(data_path / "berlin_dummypulses_normalized_fr.npz")

for timescale in fr_data.files:
    _plot_session_firing_rates(timescale, fr_data[timescale])
# %%
