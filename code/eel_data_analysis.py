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
# TODO: implement case that there are no peaks detected for a file (then no npz file)
# TODO: only plot full bins not started ones
# TODO: plot amplitude over time?/for individuals?
# TODO: make main function

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

# Datapath to npz files (folder, subfolder or file)
# datapath = Path(
#     "/home/eisele/wrk/mscthesis/data/raw/eellogger_example_data_peaks/recordings2025-03-06/eellogger1-20250306T101402_peaks.npz"
# )

# datapath = Path(
#     "/home/eisele/wrk/mscthesis/data/raw/eellogger_example_data_peaks/recordings2025-03-06/"
# )

datapath = Path("/home/eisele/wrk/mscthesis/data/raw/eellogger_example_data_peaks/")


# %%
### Functions ###
# Import corresponding wav file (TODO: make this work for sup directories)
def load_wav(datapath):
    """
    Load the corresponding wav file for the given npz file/s.
    """
    # Print status
    con.log("Loading wav files")

    if datapath.is_file():
        # Get the filename from the npz file
        filename = str(datapath.name).split("_")[0] + ".wav"

        # Get the parent directory of the npz file
        parent = "_".join(str(datapath.parent.parent).split("_")[:-1])

        # Construct the path to the wav file
        wavpath = Path(
            datapath.parent.parent.parent / parent / datapath.parent.name / filename
        )

        # Load the wav file
        audio_data = AudioLoader(wavpath)
    elif datapath.is_dir():
        # Construct path to folder that contains wav files
        parent = "_".join(str(datapath.parent).split("_")[:-1])
        wavpath = Path(datapath.parent.parent / parent / datapath.name)

        # Get all wav files of one folder and store them alphabetically in a list
        wavfiles = sorted(list(wavpath.glob("*.wav")))

        # Only one file for testing (Remove when working!!)
        wavfile = wavfiles[0]

        # Load the wav file
        audio_data = AudioLoader(wavfile)

    ### Extract necessary parameters from wav file ###
    # sampling rate
    fs = audio_data.rate  # Hz

    # number of minutes per recording
    samples_per_rec = audio_data.shape[0]
    minutes_rec = samples_per_rec / fs / 60  # 60 s/min

    return audio_data, fs, minutes_rec


# load npz files
def load_peaks(datapath):
    """
    Load all .npz files from the given directory.
    """
    # Print status
    con.log("Loading detected peaks")

    # Check if the path exists
    if not datapath.exists():
        raise FileNotFoundError(f"Path {datapath} does not exist.")

    # Initiaize lists
    data_list = []
    npz_path_list = []

    # Check if the path is a directory
    if datapath.is_file():
        # Check if the file is an npz file
        if datapath.suffix == ".npz":
            con.log(f"Path {datapath} is a single npz file.")
            # Load file
            with np.load(datapath) as data:
                # Store objects in list for consistency with directory case
                data_list.append({key: data[key] for key in data.files})
                npz_path_list = npz_path_list.append(datapath)
        else:
            raise FileNotFoundError(f"File {datapath} is not an npz file.")
    # elif datapath.is_dir():
    #     con.log(f"Path {datapath} is a directory.")
    #     # Get all npz files in this folder
    #     npz_path_list = sorted(datapath.glob("*.npz"))

    #     # Initialize list to store npz files (list of dictionaries)
    #     data_list = []

    #     # Load all npz files into list
    #     for file in npz_path_list:
    #         with np.load(file) as data:
    #             data_list.append({key: data[key] for key in data.files})

    #             # Print number of detected peaks per file
    #             con.log(f"# Peaks {file.name}: {data['peaks'].shape[0]}")

    #             # Warning if number of channels is not 16
    #             if data["channels"].shape[1] != 16:
    #                 con.log(
    #                     f"Warning: {file.name} has {data['channels'].shape[1]} channels, expected 16 channels."
    #                 )

    elif datapath.is_dir():
        con.log(f"Searching for .npz files in {datapath} and its subdirectories.")
        # Recursively find all .npz files in the directory and subdirectories
        for file in datapath.rglob("*.npz"):
            with np.load(file) as data:
                data_list.append({key: data[key] for key in data.files})
                npz_path_list.append(file)
                # Log number of detected peaks per file
                con.log(f"# Peaks {file.name}: {data['peaks'].shape[0]}")
                # Warning if number of channels is not 16
                if data["channels"].shape[1] != 16:
                    con.log(
                        f"Warning: {file.name} has {data['channels'].shape[1]} channels, expected 16 channels."
                    )
    else:
        raise FileNotFoundError(f"Path {datapath} is neither a file nor a directory.")

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
# set bin size in minutes
binsize = 10
# set sample rate (Hz)
sample_rate = 48000
# minutes per recording
min_per_rec = 5

# load data
peak_data, file_paths = load_peaks(datapath)
# wav_data, sample_rate, min_per_rec = load_wav(datapath)

# calculate peaks per bin (insert binsize in minutes here)
bin_peaks = peaks_over_time(peak_data, sample_rate, min_per_rec, binsize)

# get start time from first file
timestamp = get_timestamp(file_paths)

# plot time histogram
plot_peaks_time(bin_peaks, timestamp, binsize)

# %%
