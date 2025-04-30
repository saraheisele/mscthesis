"""
This script loads .npz files created from Patricks deep_peak_sieve package.
.npz files are created in collect_peaks.py.
.npz files contain:
dict_keys(['labels', 'peaks', 'channels', 'amplitudes', 'centers', 'start_stop_index', 'rate', 'predicted_labels', 'predicted_probs'])
    "peaks" - Window of amplitude values for each detected peak
    "centers" - avg index of highest amplitude value of peaks on all 16 channels
    "channels" - contains True/False for all 16 channels for all peaks, depending if peak showed up on that channel
    "start_stop_idx" - start and stop index of the peak (to find peak in original data)
    "labels" - only exists if peaks were labeled in later step of deep_peak_sieve. -1 default; not labeled, 1 = spike, 0 = noise

peak_data contains as many entries/dicts as there are .npz files in the folder.

This script calculates the number of detected peaks per minute/hour and plots the result as histogram.
"""

# TODO: make code work for sup folders and single files
# TODO: modularize code
# TODO: make main function
# TODO: account for amount of recordings that contribute to each bin ?
# TODO: implement case that there are no peaks detected (no file created or empty file)
# ask patrick if i should not discard start and end bins but assign prozentual soviele peaks wie zeit in dieser bin war?

# TODO: plot amplitude over time?/for individuals?


# DONE:
# Friday
# add spike counts to correct bin for each file
# rewrote code to load npz files one by one, instead of creating data_list
# rewrote code to correctly assign peaks to bins (not just stack more bins at the end)
# discard peaks at start and end of rec where not whole length of bin is covered by data
# code works for single file
# code works for sup folder
# make wav file importing work for sup folders
# make extraction of rec length and sample rate work for several files

# Wednesday
# function geschrieben die checkt ob recordings subsequent sind und paths in recording sessions speichert
# todo: function in peaks_over_time einfügen

# %%
from rich.console import Console
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from audioio.audioloader import AudioLoader
from IPython import embed

# Initialize console for logging
con = Console()

# %%
### Import Data ###

## Datapath to npz files (folder, subfolder or file)
# datapath = Path(
#     "/home/eisele/wrk/mscthesis/data/raw/eellogger_example_data_peaks/recordings2025-03-06/eellogger1-20250306T101402_peaks.npz"
# )

# Path to folder
datapath = Path(
    "/home/eisele/wrk/mscthesis/data/raw/eellogger_example_data_peaks/recordings2025-03-06/"
)

## Path to sup folder (TODO: check if it works)
# datapath = Path("/home/eisele/wrk/mscthesis/data/raw/eellogger_example_data_peaks/")


# %%
### Functions ###
# Import corresponding wav file - currently this function is unnessecary bc minutes per rec is not relevant for new approach!
def load_wav(datapath):
    """
    Load the corresponding wav file for the given npz file/s and returns recording length in min.
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
        # # Construct path to folder that contains wav files # check that this works for normal dir
        # new_path_name = "_".join(str(datapath.name).split("_")[:-1])
        # wavpath = Path(datapath.parent / new_path_name)

        # For normal directory (TEST)
        parent = "_".join(str(datapath.parent).split("_")[:-1])
        wavpath = Path(datapath.parent.parent / parent / datapath.name)

        # Get all wav files of directoy/supdirectory and store them alphabetically in a list
        wavfiles = sorted(list(wavpath.rglob("*.wav")))

        # Initialize emtpy list to store number of minutes for each recording
        minutes_list = []

        for i, wav in enumerate(wavfiles):
            # Load each wav file
            audio_data = AudioLoader(wav)

            # Get sampling rate for this wav file
            fs = audio_data.rate

            # Get number of minutes per recording for this wav file
            samples_per_rec = audio_data.shape[0]
            minutes_rec = samples_per_rec / fs / 60

            # # Store in peak_data (TODO: Adjust this!!)
            # npz_data[i]["minutes"] = minutes_rec

            ### Test ###
            # save all minutes_rec in a list
            minutes_list.append(minutes_rec)

    return minutes_rec


# Load npz files
def load_peaks(datapath):
    """
    Load .npz files from the given path, directory or supdirectory.
    """
    # Print status
    con.log("Loading detected peaks")

    # Check if the path exists
    if not datapath.exists():
        raise FileNotFoundError(f"Path {datapath} does not exist.")

    # Initiaize list to store file paths
    npz_path_list = []

    # Check if the path is a directory, a file or a directory containing files
    if datapath.is_file():
        # Check if the file is an npz file
        if datapath.suffix == ".npz":
            con.log(f"Path {datapath} is a single npz file.")
            # Load file
            with np.load(datapath) as data:
                # Store objects in list for consistency with directory case
                npz_path_list = npz_path_list.append(datapath)
        else:
            raise FileNotFoundError(f"File {datapath} is not an npz file.")

    # Recursively find all .npz files in the directory and subdirectories
    elif datapath.is_dir():
        for file in datapath.rglob("*.npz"):
            with np.load(file) as data:
                npz_path_list.append(file)
                # Log number of detected peaks per file
                con.log(f"# Peaks {file.name}: {data['peaks'].shape[0]}")
                # Warning if number of channels is not 16
                if data["channels"].shape[1] != 16:
                    con.log(
                        f"Warning: {file.name} has {data['channels'].shape[1]} channels, expected 16 channels."
                    )
    else:
        raise FileNotFoundError(
            f"Path {datapath} is not a file, directory containing files or directory containing folders containing files."
        )
    return sorted(npz_path_list)


# Calculate how many peaks per time interval
def peaks_over_time(
    bin_size: int,
    datapath: Path,
    file_paths: list,
    min_per_rec: int = 5,  # minutes per recording
):
    """
    Calculate number of peaks per minute.
    """
    # number of bins for 1 day, depending on bin size
    num_bins = int(24 * 60 / bin_size)  # 24 hours * 60 minutes / bin size in minutes

    # Initialize array to store number of peaks per bin
    peaks_per_bin = np.full(num_bins, np.nan)

    # Iterate over each npz file independently to load it
    if datapath.is_file():
        with np.load(datapath) as data:
            # only use peaks that have been predicted as valid peaks
            valid_peaks = data["peaks"][data["predicted_labels"] == 1]

    # Recursively find all .npz files in the directory and subdirectories
    elif datapath.is_dir():
        for i, file in enumerate(datapath.rglob("*.npz")):
            # TODO: make this into a function ?
            with np.load(file) as data:
                # extract sample rate
                sample_rate = data["rate"]
                # --------------
                ###
                # get time of first and last file of recording
                start_time, end_time = get_timestamps(file_paths)

                # calculate the difference in seconds to next bin
                start_diff_seconds = timedelta(
                    minutes=bin_size - (start_time.minute % bin_size),
                    seconds=-start_time.second,
                ).total_seconds()  # just take timedelta directly
                end_diff_seconds = timedelta(
                    minutes=end_time.minute % bin_size, seconds=end_time.second
                ).total_seconds()

                # calculate how many indices to discard at start and end of each session(folder)
                start_diff_idx = int(start_diff_seconds * sample_rate)
                end_diff_idx = int(end_diff_seconds * sample_rate)

                # get correct start bin for this file
                if start_time.minute > 0:
                    # round up to the next bin if not already on a bin
                    start_bin = start_time.hour + 1
                else:
                    start_bin = start_time.hour

                # embed()
                # -------------
                ###
                # data points per minute (sample rate in Hz * 60 sec/min)
                points_per_min = sample_rate * 60

                # calculate data points per bin
                points_per_bin = points_per_min * bin_size

                ###
                # only take peaks that have been predicted as valid peaks
                valid_peaks = data["centers"][data["predicted_labels"] == 1]

                # loop over all peaks in file
                for peak in valid_peaks:
                    # discard peaks at start and end of rec where not whole length of bin is covered by data
                    if (i == 0 and peak <= start_diff_idx) or (
                        i == -1 and peak >= end_diff_idx
                    ):
                        continue
                    else:
                        # assign peaks to bins
                        global_peak_idx = int(peak // points_per_bin) + start_bin
                        # add peak to corresponding bin
                        peaks_per_bin[global_peak_idx] += 1

    return peaks_per_bin


def check_sessions(
    file_paths: list,
    max_difference: int = 6,  # min
):
    # extract timestamps from filepaths
    dt_list = []
    rec_sessions = {}
    for i, path in enumerate(file_paths):
        time_stamp = path.name.split("-")[1].split("_")[0]
        dt = datetime.strptime(time_stamp, "%Y%m%dT%H%M%S")

        # append dict for first session
        if i == 0:
            dt_list.append(dt.minute)
            rec_sessions.append({f"rec session {i}": path})

        # check if time stamps are consecutive
        if i > 0:
            # access last saved time stamp of current session to check if current time is subsequent
            if dt.minute > (dt_list[-1] + max_difference):
                rec_sessions[f"rec session {len(rec_sessions)}"] = None

            # append path to last session
            list(rec_sessions.keys())[-1] = path
            dt_list.append(dt.minute)
    return rec_sessions


# Extract time from filename (TODO: doesn't work for sup folders with this logic !!)
def get_timestamps(npz_files):
    """
    Get start and end time from the filenames of the first and last npz files in one recording directory.
    """
    # get first and last file names
    first_file = npz_files[0].name
    last_file = npz_files[-1].name

    # get time stamp from file name
    datetime_str_start = first_file.split("-")[1].split("_")[0]
    datetime_str_end = last_file.split("-")[1].split("_")[0]

    # convert to datetime object
    dt_start = datetime.strptime(datetime_str_start, "%Y%m%dT%H%M%S")
    dt_end = datetime.strptime(datetime_str_end, "%Y%m%dT%H%M%S")

    return dt_start, dt_end


# Plot histogram of number of peaks per time bin
def plot_peaks_time(bin_peaks, binsize):
    """
    Plot histogram of number of peaks per time bin with actual time on x-axis.
    """
    #### old code, working ####
    # # Generate time labels for each minute bin
    # time_labels = [
    #     start_time + timedelta(minutes=i * binsize) for i in range(len(peaks))
    # ]
    # # Format as "HH:MM"
    # time_labels_str = [time.strftime("%H:%M") for time in time_labels]
    # ---
    # Number of bins for 1 day, depending on bin size
    # time_labels = list(range(0, 24 * 60, binsize)/60)

    # ---
    # Plot spike frequency over time
    fig, ax = plt.subplots()
    ax.bar(
        list(range(len(bin_peaks))),
        bin_peaks,
        width=1,
        edgecolor="black",
        align="edge",
    )
    ax.set_title(f"Number of Peaks per {binsize} min over Time")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Number of Peaks")

    # Set x-ticks to actual time
    ax.set_xticks(list(range(len(bin_peaks))))
    # embed()
    # Rotate for better readability
    # ax.set_xticklabels(time_labels, rotation=45, ha="right")

    plt.tight_layout()
    plt.show()


# %%
### Main ###
# set bin size in minutes (TODO: prompt user to input number of minutes and only take value between 1 and 60)
binsize = 60  # min

# load data
file_paths = load_peaks(datapath)
# min_per_rec = load_wav(datapath)

# get start time from first file
# start_time = get_timestamps(file_paths)

# calculate peaks per bin (TODO: insert binsize in minutes as variable here ?)
bins = peaks_over_time(binsize, datapath, file_paths, min_per_rec=5)

embed()

# plot time histogram
plot_peaks_time(bins, binsize)

# %%
