"""
This script loads preprocessed data from eel_data_preprocessing script
and calculates the number of detected peaks per minute/hour and plots the result as histogram.
"""

# Monday:
# calculate firing rate per bin
# plot firing rate per bin
# new script for plots
# list that holds number of recs per minute
# plot histogram of number of recs per minute infront of eod count distribution
# plot eod counts of each minute infront of hourly count hist

# Friday:
# TODO: account for amount of recordings that contribute to each bin (determine certainty)
# TODO: modularize peaks_over_time function (for loop in extra function)
# make progress bar more simple (track)
# normalization in seperate function
# TODO: save session idx, path idx (from json), minute, EOD count per minute per rec in npz file
# TODO: use npz file to to plot sem error bars
# use arrays instead of lists
# TODO: make rec count y axis in minutes (second yaxis)

# TODO: make csv of metadata (automate this ?), to check for correlations
# TODO: peaks over years, months, temp, leitfähigkeit, individuum

# %%
from rich.console import Console
from rich.progress import track
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
import matplotlib.pyplot as plt
# from IPython import embed

# Initialize console for logging
con = Console()


# %%
def extract_data(load_path):
    """
    Extract data from json file and convert to list of paths.
    """
    # load preprocessed paths of recording sessions from json file
    with open(load_path, "r") as file:
        data = json.load(file)

    # extract all rec paths into one list and convert to Path objects
    all_paths = [Path(file) for file_list in data.values() for file in file_list]

    # extract session id for each recording from key names
    session_ids = [
        i for i, file_list in enumerate(data.values()) for file in file_list
    ]  # TODO: maybe change this to return session structure in all_paths and loop over both sturctures

    return all_paths, session_ids


# Extract time from filename
def get_timestamps(npz_file_path):
    """
    Get start and end time from the filenames of the first and last npz files in one recording session.
    """
    # get file name
    filename = npz_file_path.name

    # get time stamp from file name
    datetime_str_start = filename.split("-")[1].split("_")[0]

    # convert to datetime object
    dt_start = datetime.strptime(datetime_str_start, "%Y%m%dT%H%M%S")
    dt_end = dt_start + timedelta(minutes=5)

    return dt_start, dt_end


def load_npz(file):
    """
    Load npz file and extract data.
    """
    # load npz file
    with np.load(file) as data:
        # extract peaks that have been predicted as valid peaks
        valid_peaks = data["centers"][data["predicted_labels"] == 1]

        # extract sample rate
        sample_rate = int(data["rate"])  # int

    return valid_peaks, sample_rate


# Count how many peaks happen per minute
def peaks_over_time(
    paths: list,  # list of paths to npz files
    num_bins: int,  # min
    min_per_rec: int,  # min
):
    """
    Calculate number of peaks per minute.
    """
    # initialize array to store number of peaks
    eods_per_min = np.zeros(num_bins)

    # initialize array to store number of recordings
    rec_per_min = np.zeros(num_bins)

    ### iterate over each file path in path list
    # track progress with progress bar
    for file in track(paths):  # TODO: account for case (4:00 - 9:00 and 4:00 - 8:59)
        # TODO: save session idx, path idx (from json), minute, EOD count per minute per rec in npz file

        ## get start and end time of each rec
        start_time, _ = get_timestamps(file)

        # calculate start and end bin of rec
        start_bin = (start_time.hour * 60) + start_time.minute
        end_bin = start_bin + min_per_rec  # TODO: embed here to check end_time

        ## load current file
        valid_peaks, sample_rate = load_npz(file)

        # calculate data points per minute (sample rate in Hz * 60 sec/min)
        idx_per_min = sample_rate * 60

        # find start index within start bin
        start_idx = int(start_time.second * sample_rate)

        ### loop over all peaks in file
        for peak in valid_peaks:
            # assign peaks to bins
            global_peak_idx = int((start_idx + peak) // idx_per_min) + start_bin

            # reset global peak idx if it exceeds number of bins so it goes into bin 0 after max bin
            if global_peak_idx >= num_bins:
                global_peak_idx = global_peak_idx - num_bins

            ### add peaks into the correct bin
            # if start of rec isn't at beginning of a minute ignore peaks in start and end bins
            if start_time.second != 0:
                if global_peak_idx > start_bin and global_peak_idx < end_bin:
                    eods_per_min[global_peak_idx] += 1
            # if start of recording is at the beginning of a minute include whole recording length
            else:
                if global_peak_idx >= start_bin and global_peak_idx < end_bin:
                    eods_per_min[global_peak_idx] += 1

        ### increment number of recordings per bin # TODO: improve this
        if start_time.second != 0:
            rec_per_min[start_bin + 1 : end_bin - 1] += 1
        else:
            rec_per_min[start_bin : end_bin - 1] += 1

    return eods_per_min, rec_per_min


def normalize_eods(eods_per_min, rec_per_min):
    """
    Normalize eod count per minute by number of recordings per minute.
    Input: eods_per_min, rec_per_min (lists of eod counts and rec counts per minute)
    Output: normalized eod count per minute (array)
    """
    return np.array(eods_per_min) / np.array(rec_per_min)


# reshape and sum bins to get eod count per desired bin size (e.g. hour)
def bin_eods_recs(eods_per_min, rec_per_min, mins_per_timestep=int):
    """
    Bin eods per minute into desired timestep. Same for recs per minute
    """
    # reshape and sum bins to get hourly eod counts
    binned_eods = eods_per_min.reshape(-1, mins_per_timestep).sum(axis=1)

    # reshape and sum bins to get hourly rec counts
    binned_recs = rec_per_min.reshape(-1, mins_per_timestep).sum(axis=1)

    return binned_eods, binned_recs


# calculate firing rate per bin/timestep
def firing_rate_per_bin(eods_per_timestep, mins_per_timestep=int):
    """
    Calculates firing rate in Hz per bin.
    """
    return eods_per_timestep / (mins_per_timestep * 60)


# Plot histogram of number of peaks per time bin
def plot_eod_hist(binned_eods, recs_per_bin, eod_per_min, save_path=None):
    """
    Plot histogram of number of peaks per time bin with actual time on x-axis.
    """
    con.log("Plotting histogram.")

    # Create a time axis for minute counts
    minute_time_axis = np.linspace(
        0, len(binned_eods), len(eod_per_min), endpoint=False
    )

    # Plot spike frequency over time
    fig, axs = plt.subplots(3, 1, figsize=(15, 12))
    ax = axs[0]
    ax.bar(
        list(range(len(binned_eods))),
        binned_eods,
        width=1,
        edgecolor="black",
        label="EOD Count",
        align="edge",
    )
    ax.set_title("EOD count per Hour")
    ax.set_ylabel("Number of Peaks")
    ax.set_xticks(list(range(len(binned_eods))))

    ax = axs[1]
    ax.bar(
        minute_time_axis,
        eod_per_min,
        width=1 / 60,
        alpha=0.5,
        label="EOD Count per Minute",
        align="edge",
    )
    ax.set_title("EOD count per Minute")
    ax.set_ylabel("Number of Peaks")
    ax.set_xticks(list(range(len(binned_eods))))

    ax = axs[2]
    ax.bar(
        list(range(len(recs_per_bin))),
        recs_per_bin,
        width=1,
        alpha=0.5,
        label="Recording Count",
        align="edge",
    )
    ax.set_title("Recording Count over Time")
    ax.set_ylabel("Number of Recordings")
    ax.set_xlabel("Time [h]")
    ax.set_xticks(list(range(len(binned_eods))))

    plt.tight_layout()

    # Save figure
    if save_path:
        con.log(f"Saving figure to {save_path}.")
        fig.savefig(save_path, dpi=300)

    # Show figure
    plt.show()


# Plot firing rate per time bin
def plot_firing_rate(fr_per_bin, save_path=None):
    """
    Plot line plot of firing rate per time bin with actual time on x-axis.
    """
    con.log("Plotting firing rates.")

    # Plot firing rate over time
    fig, ax = plt.subplots()
    ax.plot(
        list(range(len(fr_per_bin))),
        fr_per_bin,
        marker="o",
        linestyle="-",
        label="Firing Rate",
    )
    ax.set_title("EOD Firing Rate over Time")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("EODs per Second [Hz]")
    ax.set_xticks(list(range(len(fr_per_bin))))
    plt.tight_layout()

    # Save figure
    if save_path:
        con.log(f"Saving figure to {save_path}.")
        fig.savefig(save_path, dpi=300)

    # Show figure
    plt.show()


# %% Main
def main():
    # extract data from json file
    data_paths = extract_data(
        load_path="/home/eisele/wrk/mscthesis/data/intermediate/eellogger_session_paths.json"
    )

    # calculate number of eods per minute
    eod_counts, rec_counts = peaks_over_time(
        data_paths, num_bins=24 * 60, min_per_rec=5
    )

    # normalize eod counts per minute by number of recordings per minute
    eod_counts_norm = normalize_eods(eod_counts, rec_counts)

    # bin eod counts per minute into desired timestep (e.g. hour)
    binned_eod_counts, binned_rec_counts = bin_eods_recs(
        eod_counts_norm, rec_counts, mins_per_timestep=60
    )

    # calculate firing rate in Hz for each bin
    firing_rates = firing_rate_per_bin(binned_eod_counts, mins_per_timestep=60)

    # plot eod count histogram
    plot_eod_hist(
        binned_eod_counts,
        binned_rec_counts,
        eod_counts_norm,
        save_path="/home/eisele/wrk/mscthesis/data/processed/eod_count_min_hour_rec_count.png",
    )

    # plot firing rate over time
    plot_firing_rate(
        firing_rates,
        save_path="/home/eisele/wrk/mscthesis/data/processed/eod_firing_rate_over_time.png",
    )


if __name__ == "__main__":
    main()


# %%
