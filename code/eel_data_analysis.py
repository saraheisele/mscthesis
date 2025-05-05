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

# Friday
# made it work that start and end bin are percentually calculated and not disregarded
# fixed mistakes
# started implementing session function in peaks over time

# %%
from rich.console import Console
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from audioio.audioloader import AudioLoader
# from IPython import embed

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


# %%
# Calculate how many peaks per time interval
def peaks_over_time(
    bin_size: int,
    datapath: Path,
    paths: dict,  # TODO: make that it works for dict
    # TODO: iterate over sessions in peaks_over_time
    # TODO: calculate end time +5 min??
):
    """
    Calculate number of peaks per minute.
    """
    # number of bins for 1 day, depending on bin size
    num_bins = int(24 * 60 / bin_size)  # 24 hours * 60 minutes / bin size in minutes

    # Initialize array to store number of peaks per bin
    peaks_per_bin = np.zeros(num_bins)

    # get time of first and last file of recording
    start_time, end_time = get_timestamps(paths)

    # calculate start and end bin (TODO: dont hardcode)
    start_bin = start_time.hour
    end_bin = end_time.hour

    # Iterate over each npz file independently to load it
    if datapath.is_file():
        with np.load(datapath) as data:
            # only use peaks that have been predicted as valid peaks
            valid_peaks = data["peaks"][data["predicted_labels"] == 1]
            # TODO: insert function from below

    # Recursively find all .npz files in the directory and subdirectories
    elif datapath.is_dir():
        for file in datapath.rglob("*.npz"):
            # TODO: make this into a function
            with np.load(file) as data:
                # only take peaks that have been predicted as valid peaks
                valid_peaks = data["centers"][data["predicted_labels"] == 1]

                # extract sample rate
                sample_rate = int(data["rate"])  # int

                # calculate data points per minute (sample rate in Hz * 60 sec/min)
                idx_per_min = sample_rate * 60

                # calculate data points per bin
                idx_per_bin = idx_per_min * bin_size

                # case that start and end of a recording session are in the same hour
                if start_time.hour == end_time.hour:
                    # calculate the difference of start and end time in seconds
                    diff_seconds = timedelta(
                        minutes=end_time.minute - start_time.minute,
                        seconds=end_time.second - start_time.second,
                    ).total_seconds()

                    # convert to number of indices in this time window
                    diff_idx = int(diff_seconds * sample_rate)

                    # calculate how many percent of the bin is covered by data
                    diff_percent = diff_idx / idx_per_bin

                    # loop over all peaks in file
                    for peak in valid_peaks:
                        # assign peaks to bin
                        global_peak_idx = int(peak // idx_per_bin) + start_bin
                        # add correct number of peaks to bin
                        peaks_per_bin[global_peak_idx] += diff_percent

                # case that start and end of a recording session are in different hours
                else:
                    # calculate the difference to next/last bin in seconds
                    start_diff_seconds = timedelta(
                        minutes=bin_size - (start_time.minute % bin_size),
                        seconds=-start_time.second,
                    ).total_seconds()
                    end_diff_seconds = timedelta(
                        minutes=end_time.minute % bin_size, seconds=end_time.second
                    ).total_seconds()

                    # calculate how many indices to discard at start and end of each session(folder)
                    start_diff_idx = int(start_diff_seconds * sample_rate)
                    end_diff_idx = int(end_diff_seconds * sample_rate)

                    # calculate the percentage of start_diff_idx relative to points_per_bin
                    start_diff_percent = start_diff_idx / idx_per_bin
                    end_diff_percent = end_diff_idx / idx_per_bin

                    # loop over all peaks in file
                    for peak in valid_peaks:
                        # assign peaks to bins
                        global_peak_idx = int(peak // idx_per_bin) + start_bin
                        # check if bin is at start or end of recording session, add correct number of peaks to bins
                        if global_peak_idx == start_bin:
                            peaks_per_bin[global_peak_idx] += start_diff_percent
                        elif global_peak_idx == end_bin:
                            peaks_per_bin[global_peak_idx] += end_diff_percent
                        else:
                            peaks_per_bin[global_peak_idx] += 1

    return peaks_per_bin


def check_sessions(
    file_paths: list,
    max_difference: int = 6,  # min
):
    """
    Check if recordings are subsequent and store seperate recording sessions in a dictionary.
    """
    # TODO: implement to check for missing files but not new rec session
    # extract timestamps from filepaths
    dt_list = []
    rec_sessions = {}
    for i, path in enumerate(file_paths):
        time_stamp = path.name.split("-")[1].split("_")[0]
        dt = datetime.strptime(time_stamp, "%Y%m%dT%H%M%S")

        # append dict for first session
        if i == 0:
            rec_sessions[f"rec session {i}"] = []
            dt_list.append(dt.minute)

        # check if time stamps are consecutive
        if i > 0:
            # access last saved time stamp of current session to check if current time is subsequent
            if dt.minute > (dt_list[-1] + max_difference):
                rec_sessions[f"rec session {len(rec_sessions)}"] = []

            # add curent time stamp to list
            dt_list.append(
                dt.minute
            )  # maybe this works in for loop lvl instead this level?

        # append path to last session
        last_key = list(rec_sessions.keys())[-1]
        rec_sessions[last_key].append(path)  # TODO: check if this works

    return rec_sessions


# Extract time from filename
def get_timestamps(npz_files):
    """
    Get start and end time from the filenames of the first and last npz files in one recording session (entry in dict).
    Stores start and end time of each session in a dict with the same keys.
    """
    # old code, working
    # # get first and last file names
    # first_file = npz_files[0].name
    # last_file = npz_files[-1].name

    # # get time stamp from file name
    # datetime_str_start = first_file.split("-")[1].split("_")[0]
    # datetime_str_end = last_file.split("-")[1].split("_")[0]

    # # convert to datetime object
    # dt_start = datetime.strptime(datetime_str_start, "%Y%m%dT%H%M%S")
    # dt_end = datetime.strptime(datetime_str_end, "%Y%m%dT%H%M%S")

    # ----
    # create dict to store timestamps of each session
    time_stamps = {}

    # Iterate over each key in the dictionary
    for key, file_list in npz_files.items():
        # Get first and last file names in the current session
        first_file = file_list[0].name
        last_file = file_list[-1].name

        # Extract timestamps from file names
        datetime_str_start = first_file.split("-")[1].split("_")[0]
        datetime_str_end = last_file.split("-")[1].split("_")[0]

        # Convert to datetime objects
        dt_start = datetime.strptime(datetime_str_start, "%Y%m%dT%H%M%S")
        dt_end = datetime.strptime(datetime_str_end, "%Y%m%dT%H%M%S")

        # store start and end time per session as tuple in dict
        time_stamps[key] = dt_start, dt_end

    return time_stamps


# Plot histogram of number of peaks per time bin
def plot_peaks_time(bin_peaks, binsize):
    """
    Plot histogram of number of peaks per time bin with actual time on x-axis.
    """
    #### old code, working #### TODO: dont hardcode x ticks
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
    ax.set_title(f"Number of Peaks over Time (binsize: {binsize} min)")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Number of Peaks")

    # Set x-ticks to actual time
    ax.set_xticks(list(range(len(bin_peaks))))
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

# sort file paths into recording sessions
session_paths = check_sessions(file_paths)

# calculate peaks per bin (TODO: insert binsize in minutes as variable here ?)
bins = peaks_over_time(binsize, datapath, session_paths)

# plot time histogram
plot_peaks_time(bins, binsize)

# %%
