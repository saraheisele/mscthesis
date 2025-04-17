"""
This script loads .npz files created from Patricks deep_peak_sieve package.
.npz files are created in collect_peaks.py.
.npz files contain:
    "peaks" - Window of amplitude values for each detected peak
    "centers" - avg index of highest amplitude value of peaks on all 16 channels
    "channels" - contains True/False for all 16 channels for all peaks, depending if peak showed up on that channel
    "start_stop_idx" - start and stop index of the peak (to find peak in original data)
    "labels" - only exists if peaks were labeled in later step of deep_peak_sieve. -1 default; not labeled, 1 = spike, 0 = noise

This script calculates the number of detected peaks per minute/hour and plots the result as histogram.
"""
# TODO: implement case that there are no peaks detected for a file (then no file)
# TODO: only plot full bins not started ones
# TODO: add possibility to put in folder with folders that contains npz files
# TODO: add wav loading to function
# does it make more sense to write a function with the same purpose for different cases or to have one function that handles
# all three cases but then I have to implement if statements in each function

# TODO: plot amplitude over time?/for individuals?
# TODO: make main function
# TODO: google what M and U , .. at top of tab means and color code beige/yellow?

# %%
from rich.console import Console
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import math
from audioio.audioloader import AudioLoader
# from IPython import embed

# Initialize console for logging
con = Console()

# %%
### Import Data ###
# Import detected peaks
con.log("Loading detected peaks")

# Datapath to folder that contains npz files
datapath = Path(
    "/home/eisele/wrk/mscthesis/data/raw/eellogger_example_data_peaks/recordings2025-03-06/eellogger1-20250306T101402_peaks.npz"
)

# Import corresponding wav file
filename = str(datapath.name).split("_")[0] + ".wav"
parent = "_".join(str(datapath.parent.parent).split("_")[:-1])
wavpath = Path(datapath.parent.parent.parent / parent / datapath.parent.name / filename)

# Get all wav files of one folder and store them alphabetically in a list
# wavfiles = sorted(list(wavpath.glob("*.wav")))

# Only one file for testing
# wavfile = wavfiles[0]

# Load the wav file
wav_data = AudioLoader(wavpath)


# %%
### Parameters ###
# sampling rate
sample_rate = wav_data.rate  # Hz

# number of minutes per recording
samples_per_rec = wav_data.shape[0]
min_per_rec = samples_per_rec / sample_rate / 60

# assign wanted bin size in minutes
bin_size = 1


# %%
### Functions ###
# load data
def load_peaks(datapath):
    """
    Load all .npz files from the given directory.
    """
    # Check if the path exists
    if not datapath.exists():
        raise FileNotFoundError(f"Path {datapath} does not exist.")

    # Check if the path is a directory
    if datapath.is_file():
        # Check if the file is an npz file
        if datapath.suffix == ".npz":
            con.log(f"Path {datapath} is a single npz file.")
            # Load file
            data_file = []
            with np.load(datapath) as data:
                # Store objects in list for consistency with directory case
                data_file.append({key: data[key] for key in data.files})
                npz_path = [datapath]
                return data_file, npz_path
        else:
            raise FileNotFoundError(f"File {datapath} is not an npz file.")

    if datapath.is_dir():
        con.log(f"Path {datapath} is a directory.")
        # Get all npz files in this folder
        npz_path_list = sorted(datapath.glob("*.npz"))

        # Initialize list to store npz files (list of dictionaries)
        data_list = []

        # Load all npz files into list
        for file in npz_path_list:
            with np.load(file) as data:
                data_list.append({key: data[key] for key in data.files})

                # Print number of detected peaks per file
                con.log(f"# Peaks {file.name}: {data['peaks'].shape[0]}")

                # Warning if number of channels is not 16
                if data["channels"].shape[1] != 16:
                    con.log(
                        f"Warning: {file.name} has {data['channels'].shape[1]} channels, expected 16 channels."
                    )
        return data_list, npz_path_list


# Calculate how many peaks per time interval
def peaks_over_time(
    data: list,
    sample_rate: int,
    min_per_rec: int,
    bin_size: int,
):
    """
    Calculate number of peaks per minute.
    """

    # data points per minute (sample rate in Hz * 60 sec/min)
    points_per_min = sample_rate * 60

    # calculate data points per bin
    points_per_bin = points_per_min * bin_size

    # number of bins for all recordings
    num_bins = math.ceil(len(data) * min_per_rec / bin_size)

    # initialize structure to store number of peaks per minute
    peaks_per_bin = np.zeros(num_bins)

    # loop over dicts in data list
    for i, file in enumerate(data):
        # set offset for each file
        offset = i * points_per_min * min_per_rec
        # loop over all peaks in file
        for peak in file["centers"]:  # TODO: adjust for single files
            # assign peaks to minutes
            global_peak_idx = int((peak + offset) // points_per_bin)
            # add peak to corresponding minute
            peaks_per_bin[global_peak_idx] += 1
    return peaks_per_bin


# Extract time from filename
def get_timestamp(npz_files):
    # get last file name
    first_file = npz_files[0].name
    # get time stamp from file name
    datetime_str = first_file.split("-")[1].split("_")[0]
    # convert to datetime object
    start_time = datetime.strptime(datetime_str, "%Y%m%dT%H%M%S")
    return start_time


# Plot histogram of number of peaks per time bin
def plot_peaks_time(peaks, start_time, binsize):
    """
    Plot histogram of number of peaks per time bin with actual time on x-axis.
    """
    # Generate time labels for each minute bin
    # time_labels = [start_time + timedelta(minutes=i) for i in range(len(peaks))]
    time_labels = [
        start_time + timedelta(minutes=i * binsize) for i in range(len(peaks))
    ]

    # Format as "HH:MM"
    time_labels_str = [time.strftime("%H:%M") for time in time_labels]

    # Plot spike frequency over time
    fig, ax = plt.subplots()
    ax.bar(
        list(range(len(peaks))),
        peaks,
        width=1,
        edgecolor="black",
        align="edge",
    )
    ax.set_title(f"Number of Peaks per Bin ({binsize} min)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Number of Peaks")

    # Set x-ticks to actual time
    ax.set_xticks(list(range(len(peaks))))
    # Rotate for better readability
    ax.set_xticklabels(time_labels_str, rotation=45, ha="right")

    plt.tight_layout()
    plt.show()


# %%
### PLOTTING
# Plot first peaks to visualize and verify app. good detection
# fig, ax = plt.subplots()
# for i in range(10):
#     ax.plot(data["peaks"][i, :])

# plt.show()

# %%
### Main ###
# load data
peak_data, file_paths = load_peaks(datapath)

# calculate peaks per bin
bin_peaks = peaks_over_time(peak_data, sample_rate, min_per_rec, bin_size)

# get start time from first file
timestamp = get_timestamp(file_paths)

# plot time histogram
plot_peaks_time(bin_peaks, timestamp, bin_size)

# %%
