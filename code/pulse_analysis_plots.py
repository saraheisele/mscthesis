# %%
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from dateutil.relativedelta import relativedelta

#################################
############# CONFIG #############
#################################
PULSE_TYPES = {
    "all": {
        "label": "all pulses",
        "suffix": "_all",
        "figures_subdir": "all_pulses",
        "hist_subdir": "all_pulses_hist",
    },
    "double": {
        "label": "double pulses",
        "suffix": "_dp",
        "figures_subdir": "double_pulses",
        "hist_subdir": "double_pulses_hist",
    },
    "wide": {
        "label": "wide pulses",
        "suffix": "_wide",
        "figures_subdir": "wide_pulses",
        "hist_subdir": "wide_pulses_hist",
    },
    "fat": {
        "label": "fat pulses",
        "suffix": "_fat",
        "figures_subdir": "fat_pulses",
        "hist_subdir": "fat_pulses_hist",
    },
}


def select_pulse_type(default="all"):
    choices = ", ".join(PULSE_TYPES)
    selected = input(f"Pulse analysis type ({choices}) [{default}]: ").strip().lower()
    if not selected:
        return default
    if selected not in PULSE_TYPES:
        raise ValueError(
            f"Unknown pulse analysis type '{selected}'. Choose one of: {choices}."
        )
    return selected


pulse_type = select_pulse_type(default="all")
pulse_config = PULSE_TYPES[pulse_type]

#################################
############# LOAD #############
#################################

# set filename suffix and save path based on configuration
suffix = pulse_config["suffix"]
figures_subdir = pulse_config["figures_subdir"]
hist_subdir = pulse_config["hist_subdir"]

# load pulse rate dictionaries from .npz files
data_path = Path(
    f"/home/eisele/wrk/mscthesis/data/intermediate/eels-mfn2021_dummy_activity_histograms/{hist_subdir}"
)
pulse_rate_data = np.load(
    data_path / "berlin_dummypulses_pulse_rate_hz_hist_dict.npz"
)
pulse_rate_hist_dict = {k: pulse_rate_data[k] for k in pulse_rate_data.files}

metadata_path = data_path / "berlin_dummypulses_hist_metadata.npz"
if metadata_path.exists():
    metadata = np.load(metadata_path)
    first_month_year = int(metadata["first_month_year"])
    first_month_month = int(metadata["first_month_month"])
    first_year = int(metadata["first_year"])
else:
    first_month_year = 2023
    first_month_month = 1
    first_year = 2023

#################################
############# PLOTS #############
#################################
# TODO: cleanup ploting code, make functions for repeated code (e.g. x ticks and labels), make rcParams

### Save Path
save_path = Path(f"/home/eisele/wrk/mscthesis/data/processed/{figures_subdir}/")
save_path.mkdir(parents=True, exist_ok=True)


def start_axes_at_zero(axes, x_max=None):
    for axis in np.ravel(axes):
        if x_max is not None:
            axis.set_xlim(0, x_max)
        else:
            axis.set_xlim(left=0)
        axis.set_ylim(bottom=0)


def month_since_start_labels(n_months):
    first_month = datetime(first_month_year, first_month_month, 1)
    return [
        (first_month + relativedelta(months=i)).strftime("%b %Y")
        for i in range(n_months)
    ]


def plot_pulse_rate(
    timescale, title, xlabel, filename, tick_positions=None, tick_labels=None
):
    if timescale not in pulse_rate_hist_dict:
        return

    rate = pulse_rate_hist_dict[timescale]
    x = np.arange(len(rate))
    fig, ax = plt.subplots(figsize=(20, 6))

    ax.plot(x, rate, color="green")
    if tick_positions is not None and tick_labels is not None:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Pulse Rate (Hz)")
    start_axes_at_zero(ax, x_max=len(rate))
    fig.suptitle(title)

    plt.tight_layout()
    plt.savefig(save_path / filename, dpi=300)
    plt.show()


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


# %%
## 24 hours - minute bins
minute_ticks = np.arange(0, len(pulse_rate_hist_dict["minute"]), 60)
minute_tick_labels = [f"{h:02d}:00" for h in range(len(minute_ticks))]
plot_pulse_rate(
    "minute",
    "24-hour pulse rate histogram (1-min bins)",
    "time of day",
    f"24h_minute{suffix}.png",
    minute_ticks,
    minute_tick_labels,
)


# %%
## 24 hours - hourly bins
hour_ticks = np.arange(len(pulse_rate_hist_dict["hour"]))
hour_tick_labels = [f"{h:02d}:00" for h in hour_ticks]
plot_pulse_rate(
    "hour",
    "24-hour pulse rate histogram (hourly bins)",
    "time of day",
    f"24h_hour{suffix}.png",
    hour_ticks,
    hour_tick_labels,
)


# %%
## months - monthly bins
month_ticks = np.arange(len(pulse_rate_hist_dict["month"]))
month_tick_labels = [datetime(2000, m, 1).strftime("%b") for m in range(1, 13)]
plot_pulse_rate(
    "month",
    "monthly pulse rate histogram (monthly bins)",
    "month",
    f"12month_month{suffix}.png",
    month_ticks,
    month_tick_labels,
)


# %%
## months since recording start - monthly bins
if "month_since_start" in pulse_rate_hist_dict:
    month_since_start_ticks = np.arange(
        len(pulse_rate_hist_dict["month_since_start"])
    )
    month_since_start_tick_step = max(1, len(month_since_start_ticks) // 18)
    month_since_start_tick_idx = month_since_start_ticks[::month_since_start_tick_step]
    month_since_start_all_labels = month_since_start_labels(
        len(month_since_start_ticks)
    )
    plot_pulse_rate(
        "month_since_start",
        "monthly pulse rate histogram since recording start (monthly bins)",
        "month since recording start",
        f"months_since_start_month{suffix}.png",
        month_since_start_tick_idx,
        [month_since_start_all_labels[i] for i in month_since_start_tick_idx],
    )


# %%
## years
year_ticks = np.arange(len(pulse_rate_hist_dict["year"]))
year_tick_labels = [
    str(y) for y in range(first_year, first_year + len(pulse_rate_hist_dict["year"]))
]
plot_pulse_rate(
    "year",
    "yearly pulse rate histogram (yearly bins)",
    "year",
    f"years_year{suffix}.png",
    year_ticks,
    year_tick_labels,
)


data = np.load(data_path / "berlin_dummypulses_session_pulse_rate_hz.npz")
for timescale in data.files:
    if timescale == "day":
        continue

    arr = data[timescale]  # shape (n_sessions, n_bins)
    x = np.arange(arr.shape[1])

    # per-session thin lines
    plt.figure(figsize=(8, 4))
    for i in range(arr.shape[0]):
        # plt.plot doesnt make sense for higher timescales bc many rec sessions only contribute to 1 bin
        # plt.plot(x, arr[i], alpha=0.15, color="tab:blue", linewidth=0.7)

        # scatter
        valid = ~np.isnan(arr[i])
        plt.scatter(x[valid], arr[i][valid], alpha=0.2, s=10, color="tab:blue")

        # # boxplot
        # valid_per_bin = [arr[:, j][~np.isnan(arr[:, j])] for j in range(arr.shape[1])]
        # plt.boxplot(valid_per_bin)
        # TODO: maybe do violin plots/heatmaps instead

    # Summarize active sessions only. Keeping true zero-rate sessions in this
    # summary makes sparse pulse types collapse to a flat 0 Hz median.
    active_arr = np.where(arr > 0, arr, np.nan)
    median, p_lo, p_hi = nan_summary(active_arr)

    # mask invalid values so plotting skips all-NaN bins
    median_m = np.ma.masked_invalid(median)
    p_lo_m = np.ma.masked_invalid(p_lo)
    p_hi_m = np.ma.masked_invalid(p_hi)

    # thicker median line + shaded percentile band
    plt.plot(
        x,
        median_m,
        color="tab:red",
        linewidth=1.5,
        label="active-session median",
    )
    plt.fill_between(
        x,
        p_lo_m,
        p_hi_m,
        color="tab:red",
        alpha=0.25,
        label="active-session 16-84th pct",
    )

    plt.title(timescale)
    plt.xlabel("bin index")
    plt.ylabel("Pulse Rate (Hz)")
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.legend(loc="upper right", fontsize="small")
    plt.tight_layout()
    plt.savefig(save_path / f"{timescale}{suffix}.png", dpi=300)
    plt.show()


# TODO: clean up script, modularize, make functions
# %%
