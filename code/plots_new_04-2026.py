from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

#################################
############# LOAD #############
#################################

# load histogram dict from .npz file
data_path = Path(
    "/home/eisele/wrk/mscthesis/data/newdata/berlin_histogram_dict_2026.npz"
)
data = np.load(data_path)
histogram_dict = {k: data[k] for k in data.files}

#################################
############# PLOTS #############
#################################

## activity over time - 24h, minute bins
# x coordinates run 0…1439 (= minutes since midnight)
x = np.arange(len(histogram_dict["minute"]))

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(x, histogram_dict["minute"], width=1.0, align="edge", color="C0")

# label the x‑axis in hours every 60 minutes
hour_ticks = np.arange(0, len(histogram_dict["minute"]), 60)
hour_labels = [f"{h:02d}:00" for h in range(hour_ticks.size)]
ax.set_xticks(hour_ticks)
ax.set_xticklabels(hour_labels, rotation=45)

ax.set_xlim(0, len(histogram_dict["minute"]))
ax.set_xlabel("time of day")
ax.set_ylabel("pulses per minute")
ax.set_title("24‑h histogram (1‑min bins)")

plt.tight_layout()
plt.show()

# %%
## firing rate over time, 24h, minute bins
# firing rate in Hz for each minute: convert pulses per minute -> pulses per second
firing_rate = histogram_dict["minute"] / 60.0  # Hz

# plot firing rate (1-min bins)
x = np.arange(len(firing_rate))
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(x, firing_rate, color="C1")

# label the x‑axis in hours every 60 minutes
hour_ticks = np.arange(0, len(firing_rate), 60)
hour_labels = [f"{h:02d}:00" for h in range((hour_ticks.size))]
ax.set_xticks(hour_ticks)
ax.set_xticklabels(hour_labels, rotation=45)

ax.set_xlim(0, len(firing_rate))
ax.set_xlabel("time of day")
ax.set_ylabel("firing rate (Hz)")
ax.set_title("24‑h firing rate (1‑min bins)")

plt.tight_layout()
plt.show()

# %%
## activity over time - 24h, hour bins
x = np.arange(len(histogram_dict["hour"]))

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(x, histogram_dict["hour"], align="edge", color="C0")

# label the x‑axis in hours every 60 minutes
hour_labels = [f"{h:02d}:00" for h in range(len(x))]
ax.set_xticks(x)
ax.set_xticklabels(hour_labels, rotation=45)

ax.set_xlim(0, len(histogram_dict["hour"]))
ax.set_xlabel("time of day")
ax.set_ylabel("pulses per minute")
ax.set_title("24‑h histogram (1 hour bins)")

plt.tight_layout()
plt.show()

# %%
## activity over time, 12 months, daily bins
# TODO: dont hardcode x tick every 30 days, use actual month starts

x = np.arange(len(histogram_dict["day"]))
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(x, histogram_dict["day"], color="C0", align="center")

month_ticks = np.arange(0, len(histogram_dict["day"]), 31)
month_labels = [datetime(2000, m, 1).strftime("%b") for m in range(1, 13)]
ax.set_xticks(month_ticks)
ax.set_xticklabels(month_labels, rotation=45)

ax.set_xlabel("month")
ax.set_ylabel("pulses")
ax.set_title("12‑month histogram (daily bins)")

plt.tight_layout()
plt.show()

# %%
## activity over time, 12 months, monthly bins
# plot monthly histogram
x = np.arange(len(histogram_dict["month"]))
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(x, histogram_dict["month"], color="C0", align="center")

month_labels = [datetime(2000, m, 1).strftime("%b") for m in range(1, 13)]
ax.set_xticks(x)
ax.set_xticklabels(month_labels, rotation=45)

ax.set_xlabel("month")
ax.set_ylabel("pulses")
ax.set_title("12‑month histogram (monthly bins)")

plt.tight_layout()
plt.show()

# %%
## activity over time, years
# TODO: correct for amount of recordings per year (or per bin in general)!!
x = np.arange(len(histogram_dict["year"]))
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(x, histogram_dict["year"], color="C0", align="center")

year_labels = [str(y) for y in range(2019, 2019 + len(histogram_dict["year"]))]
ax.set_xticks(x)
ax.set_xticklabels(year_labels, rotation=45)

ax.set_xlabel("year")
ax.set_ylabel("pulses")
ax.set_title("yearly histogram (yearly bins)")

plt.tight_layout()
plt.show()

# %%
