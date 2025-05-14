"""
This script loads preprocessed data from eel_data_preprocessing script
and calculates the number of detected peaks per minute/hour and plots the result as histogram.
"""

# TODO: make code work for single files ??
# TODO: modularize and improve code
# TODO: progress bar
# TODO: make main function

# TODO: save rec session dict as json file and rec session/min/counts as csv with pandas (check pic of tafel from patricks aufschrieb)

# TODO: count peaks per minute and then add them up to get hours/diff bin sizes, ignore peaks that are in bins that are not whole
# TODO: dont use counts but firing rate per minute per bin
# TODO: plot distribution of EODs per bin (hopefully uniform) infront of histogram of day cyle
# TODO: account for amount of recordings that contribute to each bin (determine certainty)

# TODO: make csv of metadata (automate this ?)
# TODO: peaks over years, months, temp, leitfähigkeit, individuum

# next: ask patrick how he includes wav files in his code?
# why no npz file for folder 20240913, 20240612, 20240517, 20240503, 20240502, 20240418, 20231201 ?
# what to do in this case:
# [12:03:49] Error loading eellogger1-20240503T051442_peaks.npz: tuple index out of range                                                                                                                                                        eel_data_analysis.py:228
#            Error loading eellogger1-20240503T051943_peaks.npz: tuple index out of range                                                                                                                                                        eel_data_analysis.py:228
#            Error loading eellogger1-20240503T052943_peaks.npz: tuple index out of range
# %%
from rich.console import Console
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Initialize console for logging
con = Console()


# %%
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


### put plot in seperate script
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
