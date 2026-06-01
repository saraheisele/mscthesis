# %%
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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

# load histogram dictionaries from .npz files
data_path = Path(
    f"/home/eisele/wrk/mscthesis/data/intermediate/eels-mfn2021_dummy_activity_histograms/{hist_subdir}"
)
count_data = np.load(data_path / "berlin_dummypulses_count_hist_dict.npz")
histogram_dict = {k: count_data[k] for k in count_data.files}

rec_count_data = np.load(data_path / "berlin_dummypulses_rec_hist_dict.npz")
rec_hist_dict = {k: rec_count_data[k] for k in rec_count_data.files}

rec_time_data = np.load(data_path / "berlin_dummypulses_rec_time_hist_dict.npz")
rec_time_hist_dict = {k: rec_time_data[k] for k in rec_time_data.files}

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


# %%
## 24 hours - minute bins
pulse_rate_minute = histogram_dict["minute"] / 60.0  # Hz
pulse_rate_minute_normalized = (
    histogram_dict["minute"] / rec_time_hist_dict["minute"]
)  # Hz, normalized by recording time per bin


x = np.arange(len(pulse_rate_minute))
fig, ax = plt.subplots(2, 1, figsize=(20, 10), sharex=True)

ax[0].plot(x, pulse_rate_minute)
ax[1].plot(x, pulse_rate_minute_normalized, color="green")

hour_ticks = np.arange(0, len(histogram_dict["minute"]), 60)
hour_labels = [f"{h:02d}:00" for h in range(hour_ticks.size)]
for a in ax:
    a.set_xticks(hour_ticks)
    a.set_xticklabels(hour_labels, rotation=45)
    a.set_xlabel("time of day")
start_axes_at_zero(ax, x_max=len(pulse_rate_minute))

ax[0].set_ylabel("Pulse Rate (Hz)")
ax[1].set_ylabel("Normalized Pulse Rate (Hz)")

fig.suptitle("24-hour histogram (1-min bins)")

plt.tight_layout()
plt.savefig(save_path / f"24h_minute{suffix}.png", dpi=300)
plt.show()


# %%
## 24 hours - hourly bins
pulse_rate_hour = histogram_dict["hour"] / 60.0 / 60.0  # Hz
pulse_rate_hour_normalized = (
    histogram_dict["hour"] / rec_time_hist_dict["hour"]
)  # Hz, normalized by recording time per bin


x = np.arange(len(pulse_rate_hour))
fig, ax = plt.subplots(2, 1, figsize=(20, 10), sharex=True)

ax[0].plot(x, pulse_rate_hour)
ax[1].plot(x, pulse_rate_hour_normalized, color="green")

hour_labels = [f"{h:02d}:00" for h in range(len(x))]
for a in ax:
    a.set_xticks(x)
    a.set_xticklabels(hour_labels, rotation=45)
    a.set_xlabel("time of day")
start_axes_at_zero(ax, x_max=len(pulse_rate_hour))

ax[0].set_ylabel("Pulse Rate (Hz)")
ax[1].set_ylabel("Normalized Pulse Rate (Hz)")

fig.suptitle("24-hour histogram (hourly bins)")

plt.tight_layout()
plt.savefig(save_path / f"24h_hour{suffix}.png", dpi=300)
plt.show()

# %%
## months - monthly bins
pulse_rate_month = histogram_dict["month"] / 30 / 24 / 60 / 60  # Hz
# TODO: use actual number of days per month instead of hardcoding 30 days for all months
pulse_rate_month_normalized = (
    histogram_dict["month"] / (rec_time_hist_dict["month"])
)  # Hz, normalized by recording time per bin

x = np.arange(len(pulse_rate_month))
fig, ax = plt.subplots(2, 1, figsize=(20, 10), sharex=True)

ax[0].plot(x, pulse_rate_month)
ax[1].plot(x, pulse_rate_month_normalized, color="green")

month_labels = [datetime(2000, m, 1).strftime("%b") for m in range(1, 13)]
for a in ax:
    a.set_xticks(x)
    a.set_xticklabels(month_labels, rotation=45)
    a.set_xlabel("month")
start_axes_at_zero(ax, x_max=len(pulse_rate_month))

ax[0].set_ylabel("Pulse Rate (Hz)")
ax[1].set_ylabel("Normalized Pulse Rate (Hz)")

fig.suptitle("monthly histogram (monthly bins)")

plt.tight_layout()
plt.savefig(save_path / f"12month_month{suffix}.png", dpi=300)
plt.show()

# %%
## years
pulse_rate_year = histogram_dict["year"] / 365 / 24 / 60 / 60  # Hz
pulse_rate_year_normalized = (
    histogram_dict["year"] / (rec_time_hist_dict["year"])
)  # Hz, normalized by recording time per bin

x = np.arange(len(pulse_rate_year))
fig, ax = plt.subplots(2, 1, figsize=(20, 10), sharex=True)

ax[0].plot(x, pulse_rate_year)
ax[1].plot(x, pulse_rate_year_normalized, color="green")

year_labels = [str(y) for y in range(2023, 2023 + len(histogram_dict["year"]))]
for a in ax:
    a.set_xticks(x)
    a.set_xticklabels(year_labels, rotation=45)
    a.set_xlabel("year")
start_axes_at_zero(ax, x_max=len(pulse_rate_year))

ax[0].set_ylabel("Pulse Rate (Hz)")
ax[1].set_ylabel("Normalized Pulse Rate (Hz)")

fig.suptitle("yearly histogram (yearly bins)")

plt.tight_layout()
plt.savefig(save_path / f"years_year{suffix}.png", dpi=300)
plt.show()


data = np.load(data_path / "berlin_dummypulses_normalized_fr.npz")
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

    # robust summary (choose percentiles you prefer)
    median = np.nanmedian(arr, axis=0)
    p_lo = np.nanpercentile(arr, 16, axis=0)  # e.g. 16th percentile
    p_hi = np.nanpercentile(arr, 84, axis=0)  # e.g. 84th percentile

    # mask invalid values so plotting skips all-NaN bins
    median_m = np.ma.masked_invalid(median)
    p_lo_m = np.ma.masked_invalid(p_lo)
    p_hi_m = np.ma.masked_invalid(p_hi)

    # thicker median line + shaded percentile band
    plt.plot(x, median_m, color="tab:red", linewidth=1.5, label="nan-median")
    plt.fill_between(
        x, p_lo_m, p_hi_m, color="tab:red", alpha=0.25, label="16–84th pct"
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
