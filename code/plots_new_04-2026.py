# %%
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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

# %%
## 24 hours - minute bins
firing_rate_minute = histogram_dict["minute"] / 60.0  # Hz
firing_rate_minute_normalized = (
    histogram_dict["minute"] / rec_time_hist_dict["minute"]
)  # Hz, normalized by recording time per bin


x = np.arange(len(firing_rate_minute))
fig, ax = plt.subplots(2, 2, figsize=(20, 10), sharex=True)

ax[0, 0].bar(x, histogram_dict["minute"])
ax[1, 0].bar(
    x, (histogram_dict["minute"] / rec_time_hist_dict["minute"]), color="green"
)

ax[0, 1].plot(x, firing_rate_minute)
ax[1, 1].plot(x, firing_rate_minute_normalized, color="green")

hour_ticks = np.arange(0, len(histogram_dict["minute"]), 60)
hour_labels = [f"{h:02d}:00" for h in range(hour_ticks.size)]
for a in ax.flat:
    a.set_xticks(hour_ticks)
    a.set_xticklabels(hour_labels, rotation=45)
    a.set_xlim(0, len(firing_rate_minute))
    a.set_xlabel("time of day")

ax[0, 0].set_ylabel("pulse count")
ax[1, 0].set_ylabel("normalized pulse count")
ax[0, 1].set_ylabel("firing rate [Hz]")
ax[1, 1].set_ylabel("normalized firing rate [Hz]")

fig.suptitle("24-hour histogram (1-min bins)")

plt.tight_layout()
plt.savefig(save_path / f"24h_minute{suffix}.png", dpi=300)
plt.show()


# %%
# ## 24 hours - minute bins - count & fr in same plot
# firing_rate_minute = histogram_dict["minute"] / 60.0  # Hz
# firing_rate_minute_normalized = (
#     histogram_dict["minute"] / rec_time_hist_dict["minute"]
# )  # Hz, normalized by recording time per bin


# x = np.arange(len(firing_rate_minute))
# fig, ax = plt.subplots(2, 1, figsize=(20, 10), sharex=True)

# ax[0].bar(x, histogram_dict["minute"])
# ax[1].bar(x, (histogram_dict["minute"] / rec_time_hist_dict["minute"]), color="green")

# ax[0].plot(x, firing_rate_minute)
# ax[1].plot(x, firing_rate_minute_normalized, color="green")

# hour_ticks = np.arange(0, len(histogram_dict["minute"]), 60)
# hour_labels = [f"{h:02d}:00" for h in range(hour_ticks.size)]
# for a in ax.flat:
#     a.set_xticks(hour_ticks)
#     a.set_xticklabels(hour_labels, rotation=45)
#     a.set_xlim(0, len(firing_rate_minute))
#     a.set_xlabel("time of day")

# ax[0].set_ylabel("pulse count")
# ax[1].set_ylabel("normalized pulse count")

# fig.suptitle("24-hour histogram (1-min bins)")

# plt.tight_layout()
# plt.savefig(save_path / "24h_minute_count+fr_all.png", dpi=300)
# plt.show()


# %%
## 24 hours - hourly bins
firing_rate_hour = histogram_dict["hour"] / 60.0 / 60.0  # Hz
firing_rate_hour_normalized = (
    histogram_dict["hour"] / rec_time_hist_dict["hour"]
)  # Hz, normalized by recording time per bin


x = np.arange(len(firing_rate_hour))
fig, ax = plt.subplots(2, 2, figsize=(20, 10), sharex=True)

ax[0, 0].bar(x, histogram_dict["hour"])
ax[1, 0].bar(x, (histogram_dict["hour"] / rec_time_hist_dict["hour"]), color="green")

ax[0, 1].plot(x, firing_rate_hour)
ax[1, 1].plot(x, firing_rate_hour_normalized, color="green")

hour_labels = [f"{h:02d}:00" for h in range(len(x))]
for a in ax.flat:
    a.set_xticks(x)
    a.set_xticklabels(hour_labels, rotation=45)
    a.set_xlim(0, len(firing_rate_hour))
    a.set_xlabel("time of day")

ax[0, 0].set_ylabel("pulse count")
ax[1, 0].set_ylabel("normalized pulse count")
ax[0, 1].set_ylabel("firing rate [Hz]")
ax[1, 1].set_ylabel("normalized firing rate [Hz]")

fig.suptitle("24-hour histogram (hourly bins)")

plt.tight_layout()
plt.savefig(save_path / f"24h_hour{suffix}.png", dpi=300)
plt.show()

# %%
## months - daily bins
# TODO: dont hardcode x tick every 30 days, use actual month starts
firing_rate_day = histogram_dict["day"] / 24 / 60.0 / 60  # Hz
firing_rate_day_normalized = (
    histogram_dict["day"] / (rec_time_hist_dict["day"])
)  # Hz, normalized by recording time per bin


x = np.arange(len(firing_rate_day))
fig, ax = plt.subplots(2, 2, figsize=(20, 10), sharex=True)

ax[0, 0].bar(x, histogram_dict["day"])
ax[1, 0].bar(x, (histogram_dict["day"] / rec_time_hist_dict["day"]), color="green")

ax[0, 1].plot(x, firing_rate_day)
ax[1, 1].plot(x, firing_rate_day_normalized, color="green")

month_ticks = np.arange(0, len(histogram_dict["day"]), 31)
month_labels = [datetime(2000, m, 1).strftime("%b") for m in range(1, 13)]
for a in ax.flat:
    a.set_xticks(month_ticks)
    a.set_xticklabels(month_labels, rotation=45)
    a.set_xlabel("month")

ax[0, 0].set_ylabel("pulse count")
ax[1, 0].set_ylabel("normalized pulse count")
ax[0, 1].set_ylabel("firing rate [Hz]")
ax[1, 1].set_ylabel("normalized firing rate [Hz]")

fig.suptitle("monthly histogram (daily bins)")

plt.tight_layout()
plt.savefig(save_path / f"12month_day{suffix}.png", dpi=300)
plt.show()


# %%
## months - monthly bins
firing_rate_month = histogram_dict["month"] / 30 / 24 / 60 / 60  # Hz
# TODO: use actual number of days per month instead of hardcoding 30 days for all months
firing_rate_month_normalized = (
    histogram_dict["month"] / (rec_time_hist_dict["month"])
)  # Hz, normalized by recording time per bin

x = np.arange(len(firing_rate_month))
fig, ax = plt.subplots(2, 2, figsize=(20, 10), sharex=True)

ax[0, 0].bar(x, histogram_dict["month"])
ax[1, 0].bar(x, (histogram_dict["month"] / rec_time_hist_dict["month"]), color="green")

ax[0, 1].plot(x, firing_rate_month)
ax[1, 1].plot(x, firing_rate_month_normalized, color="green")

month_labels = [datetime(2000, m, 1).strftime("%b") for m in range(1, 13)]
for a in ax.flat:
    a.set_xticks(x)
    a.set_xticklabels(month_labels, rotation=45)
    a.set_xlabel("month")

ax[0, 0].set_ylabel("pulse count")
ax[1, 0].set_ylabel("normalized pulse count")
ax[0, 1].set_ylabel("firing rate [Hz]")
ax[1, 1].set_ylabel("normalized firing rate [Hz]")

fig.suptitle("monthly histogram (monthly bins)")

plt.tight_layout()
plt.savefig(save_path / f"12month_month{suffix}.png", dpi=300)
plt.show()

# %%
## years
firing_rate_year = histogram_dict["year"] / 365 / 24 / 60 / 60  # Hz
firing_rate_year_normalized = (
    histogram_dict["year"] / (rec_time_hist_dict["year"])
)  # Hz, normalized by recording time per bin

x = np.arange(len(firing_rate_year))
fig, ax = plt.subplots(2, 2, figsize=(20, 10), sharex=True)

ax[0, 0].bar(x, histogram_dict["year"])
ax[1, 0].bar(x, (histogram_dict["year"] / rec_time_hist_dict["year"]), color="green")

ax[0, 1].plot(x, firing_rate_year)
ax[1, 1].plot(x, firing_rate_year_normalized, color="green")

year_labels = [str(y) for y in range(2023, 2023 + len(histogram_dict["year"]))]
for a in ax.flat:
    a.set_xticks(x)
    a.set_xticklabels(year_labels, rotation=45)
    a.set_xlabel("year")

ax[0, 0].set_ylabel("pulse count")
ax[1, 0].set_ylabel("normalized pulse count")
ax[0, 1].set_ylabel("firing rate [Hz]")
ax[1, 1].set_ylabel("normalized firing rate [Hz]")

fig.suptitle("yearly histogram (yearly bins)")

plt.tight_layout()
plt.savefig(save_path / f"years_year{suffix}.png", dpi=300)
plt.show()


# %%
# normalized fr line plots per session, one per timescale
# for timescale in data.files:  # e.g. 'minute','hour',...
#     arr = data[timescale]  # shape (n_sessions, n_bins)
#     plt.figure(figsize=(8, 4))
#     for i in range(arr.shape[0]):
#         plt.plot(arr[i], alpha=0.2, color="blue")
#     plt.title(timescale)
#     plt.xlabel("bin index")
#     plt.ylabel("firing rate (Hz)")
#     plt.show()

data = np.load(data_path / "berlin_dummypulses_normalized_fr.npz")
for timescale in data.files:
    arr = data[timescale]  # shape (n_sessions, n_bins)
    x = np.arange(arr.shape[1])

    # per-session thin lines
    plt.figure(figsize=(8, 4))
    for i in range(arr.shape[0]):
        plt.plot(x, arr[i], alpha=0.15, color="tab:blue", linewidth=0.7)

    # robust summary (choose percentiles you prefer)
    median = np.nanmedian(arr, axis=0)
    p_lo = np.nanpercentile(arr, 16, axis=0)  # e.g. 16th percentile
    p_hi = np.nanpercentile(arr, 84, axis=0)  # e.g. 84th percentile

    # mask invalid values so plotting skips all-NaN bins
    median_m = np.ma.masked_invalid(median)
    p_lo_m = np.ma.masked_invalid(p_lo)
    p_hi_m = np.ma.masked_invalid(p_hi)

    # thicker median line + shaded percentile band
    plt.plot(x, median_m, color="tab:red", linewidth=2.5, label="nan-median")
    plt.fill_between(
        x, p_lo_m, p_hi_m, color="tab:red", alpha=0.25, label="16–84th pct"
    )

    plt.title(timescale)
    plt.xlabel("bin index")
    plt.ylabel("firing rate (Hz)")
    plt.legend(loc="upper right", fontsize="small")
    plt.tight_layout()
    plt.savefig(save_path / f"{timescale}{suffix}.png", dpi=300)
    plt.show()
# %%
