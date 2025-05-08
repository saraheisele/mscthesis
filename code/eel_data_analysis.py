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
# TODO: modularize and improve code
# TODO: progress bar
# TODO: make main function
# TODO: account for amount of recordings that contribute to each bin (determine certainty)
# TODO: peaks over years, months, temp, leitfähigkeit, individuum


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

# Tuesday:
# finished function that checks for faulty files
# implemented faulty_files functionj in check_sessions function
# wrote function to trim nans if they are at edges of list
# make it work for no file (check with wav files if there should be a file)/empty npz file

# Wednesday:
# implement faulty files in peaks_per_bin
# soft coded start and end bin
# initiated global variables
# changed logic for files that are present but no peaks detected (not treated as faulty files anymore)
# peaks_per_bin checks now if there are any valid peaks in the file
# check current code version
# try on big dataset

# Thursday:
# wav files have to match naming pattern to be included in wav path list
# files that dont start with eellogger (also eelgrid) are not in wav file list
# fixed check_sessions function to work for multiple subsequent None values
# killed remove nans at edges function bc its redundant with new check sessions logic
# empty wav files are now not included in wav file list
# if peak idx reaches maximum bin it is reset to continute in bin idx 0
#
# next: ask patrick how he includes wav files in his code?
# why no npz file for folder 20240913, 20240612, 20240517, 20240503, 20240502, 20240418, 20231201 ?
# what to do in this case:
# [12:03:49] Error loading eellogger1-20240503T051442_peaks.npz: tuple index out of range                                                                                                                                                        eel_data_analysis.py:228
#            Error loading eellogger1-20240503T051943_peaks.npz: tuple index out of range                                                                                                                                                        eel_data_analysis.py:228
#            Error loading eellogger1-20240503T052943_peaks.npz: tuple index out of range                                                                                                                                                        eel_data_analysis.py:228
# %%
from rich.console import Console
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from audioio.audioloader import AudioLoader
import re

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
datapath = Path("/mnt/data1/eels-mfn2021_peaks/")


# %%
### Functions ###
# Import corresponding wav file - currently this function is unnessecary bc minutes per rec is not relevant for new approach!
def load_wav(datapath):
    """
    Load the corresponding wav file for the given npz file/s and returns recording length in min.
    """
    # Print status
    con.log("Loading wav files")

    # Initialize list to store file paths globally
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

        # Match eellogger followed by any number, then -YYYYMMDDTHHMMSS
        pattern = re.compile(r"^eellogger\d+-\d{8}T\d{6}$")

        # Get all wav files of directory and store them alphabetically in a list
        wav_path_list = sorted(
            [
                f
                for f in wavpath.rglob("*.wav")
                # only store if normal stem and not empty
                if pattern.match(f.stem) and f.stat().st_size > 0
            ]
        )

        # Initialize emtpy list to store number of minutes for each recording
        minutes_list = []

        for i, wav in enumerate(wav_path_list):
            try:
                # Load each wav file
                audio_data = AudioLoader(wav)
            except Exception as e:
                con.log(f"Error loading: {e}")
                continue
            # Get sampling rate for this wav file
            fs = audio_data.rate

            # Get number of minutes per recording for this wav file
            samples_per_rec = audio_data.shape[0]
            minutes_rec = samples_per_rec / fs / 60

            # # Store in peak_data
            # npz_data[i]["minutes"] = minutes_rec

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

    else:
        raise FileNotFoundError(
            f"Path {datapath} is not a file, directory containing files or directory containing folders containing files."
        )

    return sorted(npz_path_list)


def faulty_files(wav_path_list, npz_path_list):
    """
    Check if there are any missing or empty files in the given lists of wav and npz files.
    """
    ## check if the wav and npz files have the same names, i.e. if npz file exists for each wav file
    for i, (wav_file, npz_file) in enumerate(zip(wav_path_list, npz_path_list)):
        # get the filenames without "peaks" and suffixes
        wav_name = wav_file.stem
        npz_name = npz_file.stem.split("_")[0]
        if wav_name != npz_name:
            con.log(f"Warning: No npz file that matches wav file {wav_file.name}!")
            # insert NaN in place of missing npz file
            npz_path_list.insert(i, None)

        ## check if existing npz file is empty
        # check for file size
        elif Path(npz_file).stat().st_size == 0:
            con.log(f"Warning: npz file {npz_file.name} is empty!")
            # insert NaN in place of empty npz file
            npz_path_list[i] = None

        # if file size isnt zero, check if npz file contains npy files (dict keys)
        else:
            try:
                with np.load(npz_file) as data:
                    # check if npz file contains any keys
                    if len(data.files) == 0:
                        con.log(f"Warning: npz file {npz_file.name} contains no keys!")
                        # insert NaN in place of empty npz file
                        npz_path_list[i] = None

                    # check if predicted_labels exists (crucial for analysis)
                    elif "predicted_labels" not in data.files:
                        con.log(
                            f"Warning: npz file {npz_file.name} does not contain predicted_labels key!"
                        )
                        # insert NaN in place of file without predicted_labels
                        npz_path_list[i] = None

                    # if keys exist, check if keys are empty
                    else:
                        for key in data.files:
                            if data[key].size == 0:
                                con.log(
                                    f"Warning: npz file {npz_file.name} contains empty key {key}!"
                                )
            except Exception as e:
                con.log(f"Error loading {npz_file.name}: {e}")
                # insert NaN in place of empty npz file
                npz_path_list[i] = None

    return npz_path_list


def check_sessions(file_paths: list):
    """
    Check if recordings are subsequent and store seperate recording sessions in a dictionary.
    """
    ## Pair each path with its extracted datetime or None
    # initiate empty list to store tuples of (datetime, path)
    dt_path_list = []

    for path in file_paths:
        # append Nones for currupted files to preserve order
        if path is None:
            dt_path_list.append((None, None))
        # valid files
        else:
            # get time stamp from file name
            time_stamp = path.name.split("-")[1].split("_")[0]
            dt = datetime.strptime(time_stamp, "%Y%m%dT%H%M%S")
            # add path together with its timestamp to list
            dt_path_list.append((dt, path))

    ##
    # accepted time difference and tolerance between two files
    t_diff = timedelta(minutes=5)
    tolerance = timedelta(seconds=1)

    # store recording sessions in a dictionary
    rec_sessions = {}
    current_session = []

    # set last valid timestamp to None to indicate first session
    last_valid_dt = None

    # count number of subsequent Nones
    none_counter = 0

    # iterate over paths and corresponding timestamps
    for dt, path in dt_path_list:
        # skip corrupted files
        if dt is None:
            # increase counter for each subsequent None
            none_counter += 1
            continue

        # at first valid entry, start first session
        if last_valid_dt is None:
            # add first path to session
            current_session = [path]

        # all other valid entries
        else:
            # check time difference to last valid timestamp
            time_gap = dt - last_valid_dt
            # calculate how big expected gap is based on how many files were skipped
            expected_gap = t_diff * (none_counter + 1)

            # if current rec is subsequent to the last valid (non-None) rec
            if abs(time_gap - expected_gap) <= tolerance:
                # add to the same session
                current_session.append(path)

            # if current rec not subsequent
            else:
                # add last session to dict
                session_key = f"rec session {len(rec_sessions)}"
                rec_sessions[session_key] = current_session
                # start a new session
                current_session = [path]

        # update last valid timestamp
        last_valid_dt = dt
        # reset none counter
        none_counter = 0

    # save the final session
    if current_session:
        session_key = f"rec session {len(rec_sessions)}"
        rec_sessions[session_key] = current_session

    return rec_sessions


# TODO: save this dictionary to a file for later use!!


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


# %%
# Calculate how many peaks per time interval
def peaks_over_time(
    bin_size: int,  # min
    time_line: int,  # min
    min_per_rec: int,  # min
    paths: dict,
):
    """
    Calculate number of peaks per minute.
    """
    # number of bins
    num_bins = int(time_line / bin_size)

    # Initialize array to store number of peaks per bin
    peaks_per_bin = np.zeros(num_bins)

    # initialize array to hold nan percentages
    nan_per_bin = np.zeros(num_bins)

    # Iterate over each rec session in the dictionary
    for key, file_list in paths.items():
        con.log(f"Processing {key}")
        # get time of first and last timestamps of recording session
        start_time, end_time = get_timestamps(file_list)

        # minutes from start of time axis to start time of recording session
        start_time_minutes = (
            timedelta(
                hours=start_time.hour,
                minutes=start_time.minute,
                seconds=start_time.second,
            ).total_seconds()
            / 60
        )
        end_time_minutes = (
            timedelta(
                hours=end_time.hour,
                minutes=end_time.minute,
                seconds=end_time.second,
            ).total_seconds()
            / 60
        )

        # calculate start and end bin
        start_bin = int(start_time_minutes // bin_size)
        end_bin = int(end_time_minutes // bin_size)

        # iterate over each filepath in the session
        for i, file in enumerate(file_list):
            # ------------ TODO: maybe implement better logic?
            # check if file is faulty
            if file is None:
                # extract sample rate from previous file
                with np.load(file_list[i - 1]) as data:
                    sample_rate = int(data["rate"])  # int

                # calculate data points per minute (sample rate in Hz * 60 sec/min)
                idx_per_min = sample_rate * 60

                # calculate data points per bin
                idx_per_bin = idx_per_min * bin_size

                # calculate how many indices are missing due to faulty file
                missing_idx = int(min_per_rec * idx_per_min)

                # calculate how many percent of the bin faulty file makes up
                nan_percent = missing_idx / idx_per_bin

                ## get correct bin of faulty files
                # extract start time of previous file
                prev_start_time = datetime.strptime(
                    file_list[i - 1].name.split("-")[1].split("_")[0], "%Y%m%dT%H%M%S"
                )

                # calculate start time of faulty file
                nan_start_time = (
                    timedelta(
                        hours=prev_start_time.hour,
                        minutes=prev_start_time.minute,
                        seconds=prev_start_time.second,
                    ).total_seconds()
                    / 60
                ) + timedelta(minutes=min_per_rec)

                # calculate start bin of faulty file
                correct_nan_bin = nan_start_time // bin_size

                ## store amount of missing nan percentage per bin
                nan_per_bin[correct_nan_bin] += nan_percent

            # ----------
            # non faulty files
            else:
                # load current file
                with np.load(file) as data:
                    # extract peaks that have been predicted as valid peaks
                    valid_peaks = data["centers"][data["predicted_labels"] == 1]

                    # check if there are any valid peaks
                    if len(valid_peaks) == 0:
                        con.log(f"Warning: No valid peaks in {file.name}!")
                        continue

                    # if valid peaks detected
                    else:
                        # ------- maybe put this whole block at start
                        # extract sample rate
                        sample_rate = int(data["rate"])  # int

                        # calculate data points per minute (sample rate in Hz * 60 sec/min)
                        idx_per_min = sample_rate * 60

                        # calculate data points per bin
                        idx_per_bin = idx_per_min * bin_size
                        # -------

                        # start and end of a recording session are in the same hour
                        if start_bin == end_bin:
                            # calculate the time difference of start and end time in seconds
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
                                # check if bin contains faulty files and add according correct percentage of peaks
                                peaks_per_bin[global_peak_idx] += (
                                    diff_percent - nan_per_bin[global_peak_idx]
                                )

                        # start and end of a recording session are in different hours
                        else:
                            # calculate the time difference to next/last bin in seconds
                            start_diff_seconds = timedelta(
                                minutes=bin_size - (start_time.minute % bin_size),
                                seconds=-start_time.second,
                            ).total_seconds()
                            end_diff_seconds = timedelta(
                                minutes=end_time.minute % bin_size,
                                seconds=end_time.second,
                            ).total_seconds()

                            # calculate how many indices at start and end of each session
                            start_diff_idx = int(start_diff_seconds * sample_rate)
                            end_diff_idx = int(end_diff_seconds * sample_rate)

                            # calculate the percentage of start_diff_idx relative to points_per_bin (in decimals)
                            start_diff_percent = start_diff_idx / idx_per_bin
                            end_diff_percent = end_diff_idx / idx_per_bin

                            # calculate offset to account for length of recordings
                            offset = i * idx_per_min * min_per_rec

                            # calculate at which idx of start bin this recording starts
                            start_idx = idx_per_bin - start_diff_idx

                            # loop over all peaks in file
                            try:
                                for peak in valid_peaks:
                                    # assign peaks to bins
                                    global_peak_idx = (
                                        int((start_idx + offset + peak) // idx_per_bin)
                                        + start_bin
                                    )

                                    # reset global peak idx if it exceeds number of bins so it goes into bin 0 at 24
                                    if global_peak_idx >= num_bins * 3:
                                        global_peak_idx = global_peak_idx - num_bins * 3
                                        con.log(3)
                                    elif global_peak_idx >= num_bins * 2:
                                        global_peak_idx = global_peak_idx - num_bins * 2
                                        con.log(2)
                                    elif global_peak_idx >= num_bins:
                                        global_peak_idx = global_peak_idx - num_bins

                                    # check if bin is at start or end of recording session, add correct number of peaks to bins
                                    # check if bin contains faulty files and add according correct percentage of peaks
                                    if global_peak_idx == start_bin:
                                        peaks_per_bin[global_peak_idx] += (
                                            start_diff_percent
                                            - nan_per_bin[global_peak_idx]
                                        )
                                    elif global_peak_idx == end_bin:
                                        peaks_per_bin[global_peak_idx] += (
                                            end_diff_percent
                                            - nan_per_bin[global_peak_idx]
                                        )
                                    # normal bin
                                    else:
                                        peaks_per_bin[global_peak_idx] += (
                                            1 - nan_per_bin[global_peak_idx]
                                        )
                            except IndexError:  # TODO: find actual solution for this
                                con.log(
                                    f"Warning: Peak index {peak} out of range for file {file.name}!!!!!!!!!"
                                )
                                continue

    return peaks_per_bin


# Plot histogram of number of peaks per time bin
def plot_peaks_time(bin_peaks, binsize):
    """
    Plot histogram of number of peaks per time bin with actual time on x-axis.
    """
    con.log("Plotting histogram")

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
# set time on x axis in minutes
timeline = 24 * 60  # min
# set recording length in minutes (TODO: maybe solve this with array of rec lengths from load wav function)
rec_length = 5  # min

# load data
_, wav_paths = load_wav(datapath)
npz_paths = load_peaks(datapath)

# faulty files function
npz_paths_new = faulty_files(wav_paths, npz_paths)

# sort file paths into recording sessions
session_paths = check_sessions(npz_paths_new)

# calculate peaks per bin
bins = peaks_over_time(binsize, timeline, rec_length, session_paths)

# plot time histogram
plot_peaks_time(bins, binsize)


# %%
