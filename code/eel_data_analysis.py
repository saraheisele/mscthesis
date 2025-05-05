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

# TODO: make code work for single files
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

# Monday
# implemented session function in peaks over time, iterating over sessions instead of all paths now
# new implementations works for sup folders
# skip files if filename doesnt start with eellogger
# initiated offset and start idx variables to count up peak idx and assign correctly to subsequent bins (within and across recordings)

# next: make it work for no file (check with wav files if there should be a file)/empty npz file

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

# # Path to folder
# datapath = Path(
#     "/home/eisele/wrk/mscthesis/data/raw/eellogger_example_data_peaks/recordings2025-03-06/"
# )

# Path to sup folder
datapath = Path("/home/eisele/wrk/mscthesis/data/raw/eellogger_example_data_peaks/")


# %%
### Functions ###
# Import corresponding wav file - currently this function is unnessecary bc minutes per rec is not relevant for new approach!
def load_wav(datapath):
    """
    Load the corresponding wav file for the given npz file/s and returns recording length in min.
    """
    # Print status
    con.log("Loading wav files")

    # Initiaize list to store file paths globally
    # wav_path_list = []

    if datapath.is_file():
        # check if file starts with eellogger # TODO: necessary here?
        if datapath.name.startswith("eellogger") and datapath.suffix == ".npz":
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

            # add path to list for consistency with directory case
            wav_path_list = [wavpath]

    elif datapath.is_dir():
        # Construct path to sup folder TODO: change this so not necessary to hardcode if its a sup/normal directory
        new_path_name = "_".join(str(datapath.name).split("_")[:-1])
        wavpath = Path(datapath.parent / new_path_name)

        # # Construct path to normal directory
        # parent = "_".join(str(datapath.parent).split("_")[:-1])
        # wavpath = Path(datapath.parent.parent / parent / datapath.name)

        # Get all wav files of directoy/supdirectory and store them alphabetically in a list
        wav_path_list = sorted(list(wavpath.rglob("*.wav")))

        # Initialize emtpy list to store number of minutes for each recording
        minutes_list = []

        for i, wav in enumerate(wav_path_list):
            # check if wav file starts with eellogger
            if wav.name.startswith("eellogger"):
                # Load each wav file
                audio_data = AudioLoader(wav)

                # Get sampling rate for this wav file
                fs = audio_data.rate

                # Get number of minutes per recording for this wav file
                samples_per_rec = audio_data.shape[0]
                minutes_rec = samples_per_rec / fs / 60

                # # Store in peak_data
                # npz_data[i]["minutes"] = minutes_rec

                ### Test ###
                # save all minutes_rec in a list
                minutes_list.append(minutes_rec)

    return minutes_rec, wav_path_list


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
        if datapath.suffix == ".npz" and datapath.name.startswith("eellogger"):
            con.log(f"Path {datapath} is a single npz file.")
            # Store objects in list for consistency with directory case
            npz_path_list.append(datapath)
        else:
            raise FileNotFoundError(f"File {datapath} is not an npz file.")

    # Recursively find all .npz files in the directory and subdirectories
    elif datapath.is_dir():
        for file in datapath.rglob("*.npz"):
            # skip files that don't start with eellogger (wrong recording format)
            if file.name.startswith("eellogger"):
                npz_path_list.append(file)
                with np.load(file) as data:  # TODO: necessary?
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


# Construction####################
def faulty_files(wav_path_list, npz_path_list, sessions):
    """
    Check if there are any faulty files in the given lists of wav and npz files.
    """
    # Check if the number of wav and npz files is equal TODO: necessary?
    if len(wav_path_list) != len(npz_path_list):
        # Check if the wav and npz files have the same names
        for i, (wav_file, npz_file) in enumerate(zip(wav_path_list, npz_path_list)):
            if (
                wav_file.name != npz_file.name
            ):  # change condition to if no npz file with wav file name or npz file with wav file name is emtpy
                con.log(
                    f"Warning: Wav file {wav_file.name} does not match npz file {npz_file.name}."
                )
                # Insert NaN in place of the faulty npz file TODO: insert this into correct place in sessions
                npz_path_list[i] = np.nan

    return npz_path_list


# Construction####################


# %%
# Calculate how many peaks per time interval
def peaks_over_time(
    bin_size: int,
    paths: dict,
):
    """
    Calculate number of peaks per minute.
    """
    # number of bins for 1 day, depending on bin size
    num_bins = int(24 * 60 / bin_size)  # 24 hours * 60 minutes / bin size in minutes

    # Initialize array to store number of peaks per bin
    peaks_per_bin = np.zeros(num_bins)

    # Iterate over each rec session in the dictionary
    for key, file_list in paths.items():
        # get time of first and last timestamps of recording session
        start_time, end_time = get_timestamps(file_list)

        # calculate start and end bin (TODO: dont hardcode)
        start_bin = start_time.hour
        end_bin = end_time.hour

        # iterate over each filepath in the session
        for i, file in enumerate(file_list):
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
                    # calculate the t_diff of start and end time in seconds
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
                    # calculate the t_diff to next/last bin in seconds
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

                    # calculate offset to account for length of recorddings
                    min_per_rec = int(5)  # TODO: dont hardcode this
                    offset = i * idx_per_min * min_per_rec

                    # calculate at which idx of start bin this recording starts
                    start_idx = idx_per_bin - start_diff_idx

                    # loop over all peaks in file
                    for peak in valid_peaks:
                        # assign peaks to bins
                        global_peak_idx = (
                            int((start_idx + offset + peak) // idx_per_bin) + start_bin
                        )

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
):
    """
    Check if recordings are subsequent and store seperate recording sessions in a dictionary.
    # TODO: implement to check for missing files but not new rec session
    """
    # accepted time difference and tolerance between two files
    t_diff = timedelta(minutes=5)
    tolerance = timedelta(seconds=1)

    # extract timestamps from filepaths
    dt_list = []
    rec_sessions = {}

    for i, path in enumerate(file_paths):
        # get time stamp from file name
        time_stamp = path.name.split("-")[1].split("_")[0]
        # convert it to datetime object
        dt = datetime.strptime(time_stamp, "%Y%m%dT%H%M%S")

        # append dict for first session
        if i == 0:
            rec_sessions[f"rec session {i}"] = []

        # check if next file is not subsequent to last file
        elif i > 0 and abs((dt - dt_list[-1]) - t_diff) > tolerance:
            # create new session
            rec_sessions[f"rec session {len(rec_sessions)}"] = []

        # add curent time stamp to list
        dt_list.append(dt)

        # append path to last session
        last_key = list(rec_sessions.keys())[-1]
        rec_sessions[last_key].append(path)

    return rec_sessions


# Extract time from filename
def get_timestamps(npz_files):
    """
    Get start and end time from the filenames of the first and last npz files in one recording session.
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
    con.log("Plotting histogram")
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
_, wav_paths = load_wav(datapath)
npz_paths = load_peaks(datapath)

# sort file paths into recording sessions
session_paths = check_sessions(npz_paths)

# faulty files function
new_session_paths = faulty_files(wav_paths, npz_paths, session_paths)

# calculate peaks per bin (TODO: insert binsize in minutes as variable here ?)
bins = peaks_over_time(binsize, new_session_paths)

# plot time histogram
plot_peaks_time(bins, binsize)

# %%
