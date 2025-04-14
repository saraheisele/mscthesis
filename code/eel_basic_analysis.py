"""
This file is for analyzing random stuff about the eel data from berlin.

- does one animal have bigger amplitude even when normalized
    >normalize (to what? max amplitude of each animal?)
    >get metrics of avg amplitude and frequency per animal
- do they react to each other
    >plot correlation of spikes
- how does spike pattern change during day
    >plot average spike pattern for each hour (of all spikes) -> histogram
    >threshold to distinguish between different spike types (low/high voltage)
    >plot different spike types for each hour
    >are there there patterns that dont relate to feeding or sleeping?

- how long do they sleep
    >check with videos if I can find correlation between behavior and spike pattern during day
    >is there any repeting pattern that could be related to sleep wake cycle
- does spiking activity correlate to external factors (check day rhythm plot with synced videos)
    >check for correlation of spike pattern with feeding
    >check for correlation of spike pattern with light
    >check for anycorrelation with factors that are stated in the metadata (word file)
- is positive pole of electric field really at head of animal
    >check in synced video if head is at electrode where positive spike was recorded
    >are we sure that positive spike equals positive pole??
- which animal spikes with higher frequency and amplitude
    >look at this in audian with synced video (!!)

TODO:
- filter if necessary
- detect spikes (threshold)
- normalize (max amplitude)
- verify with video which animal produces which spikes (is higher amplitude = male?)
- threshold to get two diff. spike sets, one per animal
- threshold to distinguish between low and high voltage spikes
- get avg amplitude and frequency per animal
- check correlation between animals
- plot spike frequency and amplitude over time (hourly bins to start with)
- plot spike pattern for each animal and each spike type over time

- once this is done check with videos if there are any correlations between behavior/external factors and spiking activity

- try patricks deep_peak_sieve (import missing modules)
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks
from rich.console import Console
from audioio.audioloader import AudioLoader

con = Console()

# %%
### Import data
con.log("Loading data")
# Define the path to the data (this is for one folder only, change path if want to look at other files)
wavpath = Path("/mnt/data1/eels-mfn2021/recordings2025-03-12/")

# Get all wav files of one folder and store them alphabetically in a list
wavfiles = sorted(list(wavpath.glob("*.wav")))

# # Only one file for testing
# wavfile = wavfiles[0]

# Load the wav file
data = AudioLoader(wavfiles)

# Make all spikes positive
data_abs = np.abs(data)


# %%
### Parameters
peak_height_threshold = 0.003
min_peak_distance_seconds = 0.005
min_peak_distance = int(np.ceil(min_peak_distance_seconds * data.rate))
sample_rate = data.rate  # Hz??


# %%
### Functions
# extract individual channels and peaks per channel
def detect_peaks(
    block_filtered: np.ndarray,
    channels_count: int,
    peak_height_threshold: float,
    min_peak_distance: int,
):
    """
    Run peak-finding for each channel in a filtered block.
    Returns a list of peak indices and a parallel list of channel indices.
    """
    con.log("Starting peak detection")
    peaks_list = [
        find_peaks(
            block_filtered[:, ch],
            prominence=peak_height_threshold,
            distance=min_peak_distance,
        )[0]
        for ch in range(channels_count)
    ]
    channels_list = [
        np.array([ch] * len(peaks_list[ch]), dtype=np.int32)
        for ch in range(channels_count)
    ]
    return peaks_list, channels_list


# call function to get peaks and channels
peaks_list, channels_list = detect_peaks(
    data_abs,
    data.channels,
    peak_height_threshold,
    min_peak_distance,
)


# %%
# TODO:
# plot histogram for all wav files of one folder

# find which peaks in peaks_list are in which minute (peaks_list contains indexes of data_abs)

# how many data points per minute (per hour for now)
points_per_min = sample_rate * 60 * 60

# preallocate structure to store number of spike for each minute
minute_1 = []
minute_2 = []
minute_3 = []
minute_4 = []
minute_5 = []
# array_list = [np.random.rand(10) for _ in range(5)]

# check that my peak to minute assignment is not bullshit (too uniform)
# make code prettier
# consider that peaks_list is a list of arrays, one for each channel!!
# in erstem channel alle peaks zu nehmen, in folgenden channels peaks nur nehmen falls sie mehr als bestimmte range entfernt sind von existierenden peaks
# verschiedene distanzen checken un die nehmen bei der am meisten peaks detected werden über alle channel
# --> erst checken ob alle peaks auf channel[0] auf allen channels vorhanden sind -> nein es gibt kürzere channel (vlt mit threshold spiele zum peaks detecten)

# alt approach w extend (faster) --> put this into loop
minute_1.extend(peaks_list[0][peaks_list[0] <= points_per_min])
minute_2.extend(
    peaks_list[0][
        (peaks_list[0] > points_per_min) & (peaks_list[0] <= 2 * points_per_min)
    ]
)
minute_3.extend(
    peaks_list[0][
        (peaks_list[0] > 2 * points_per_min) & (peaks_list[0] <= 3 * points_per_min)
    ]
)
minute_4.extend(
    peaks_list[0][
        (peaks_list[0] > 3 * points_per_min) & (peaks_list[0] <= 4 * points_per_min)
    ]
)
minute_5.extend(
    peaks_list[0][
        (peaks_list[0] > 4 * points_per_min) & (peaks_list[0] <= 5 * points_per_min)
    ]
)

# # approach with last channel for sanity check
# minute_1.extend(peaks_list[7][peaks_list[7] <= points_per_min])
# minute_2.extend(
#     peaks_list[7][
#         (peaks_list[7] > points_per_min) & (peaks_list[7] <= 2 * points_per_min)
#     ]
# )
# minute_3.extend(
#     peaks_list[7][
#         (peaks_list[7] > 2 * points_per_min) & (peaks_list[7] <= 3 * points_per_min)
#     ]
# )
# minute_4.extend(
#     peaks_list[7][
#         (peaks_list[7] > 3 * points_per_min) & (peaks_list[7] <= 4 * points_per_min)
#     ]
# )
# minute_5.extend(
#     peaks_list[7][
#         (peaks_list[7] > 4 * points_per_min) & (peaks_list[7] <= 5 * points_per_min)
#     ]
# )

# concatenate all minute lists into one
minute_bins = np.array(
    (
        minute_1,
        minute_2,
        minute_3,
        minute_4,
        minute_5,
    )
)

# calculate and store spike counts per minute
spike_counts = [
    len(minute_1),
    len(minute_2),
    len(minute_3),
    len(minute_4),
    len(minute_5),
]


# %%
### Plotting
# plot raw data and detected spikes
con.log("Plotting detected peaks")
fig, ax = plt.subplots(nrows=data.channels, ncols=1)
for channel in range(data.channels):
    ax[channel].vlines(
        peaks_list[channel][:10],
        ymin=0,
        ymax=np.max(data_abs[:, channel]),
        color="r",
    )
    ax[channel].plot(data_abs[:10000, channel])

    # fig.suptitle(f"Detected peaks in {wavfiles.name}")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")

plt.show()


# %%
# plot spike frequency and amplitude over time
fig, ax = plt.subplots()
ax.bar(
    range(minute_bins.shape[0]),
    spike_counts,
    width=1,
    edgecolor="black",
    align="edge",
)
ax.set_title("Number of Peaks per Minute")
ax.set_xlabel("Minute")
ax.set_ylabel("Number of Peaks")
plt.show()

# %%
