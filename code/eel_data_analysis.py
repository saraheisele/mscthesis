"""
This script loads preprocessed data from eel_data_preprocessing script
and calculates the number of detected peaks per minute/hour and plots the result as histogram.
"""

# next
# TODO: dont use counts but firing rate per minute per bin (counts/time)
# TODO: save rec session/min/counts as csv with pandas (check pic of tafel from patricks aufschrieb)
# TODO: plot distribution of EODs per bin (hopefully uniform) infront of histogram of day cyle
# TODO: account for amount of recordings that contribute to each bin (determine certainty)

# TODO: ask if i should modularize peaks_over_time function further??
# TODO: make csv of metadata (automate this ?)
# TODO: peaks over years, months, temp, leitfähigkeit, individuum

# %%
from rich.console import Console
from rich.progress import Progress
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
from pathlib import Path
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

    # extract all paths into one list and convert to Path objects
    all_paths = [Path(file) for file_list in data.values() for file in file_list]

    return all_paths


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
    # initialize array to store number of peaks per bin
    peaks_per_bin = np.zeros(num_bins)

    # track overall progress in progress bar
    with Progress() as progress:
        task = progress.add_task("Processing...", total=len(paths))

        ### iterate over each file path in path list
        for file in paths:
            ## get start and end time of each rec
            start_time, _ = get_timestamps(file)

            # calculate start and end bin of rec
            start_bin = (start_time.hour * 60) + start_time.minute
            end_bin = start_bin + min_per_rec

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
                        peaks_per_bin[global_peak_idx] += 1
                # if start of recording is at the beginning of a minute include whole recording length
                else:
                    if global_peak_idx >= start_bin and global_peak_idx < end_bin:
                        peaks_per_bin[global_peak_idx] += 1

            # update progress bar
            progress.update(task, advance=1)

    return peaks_per_bin


### put plot in seperate script
# Plot histogram of number of peaks per time bin
def plot_peaks_time(bin_peaks, save_path=None, bins_per_timestep=int):
    """
    Plot histogram of number of peaks per time bin with actual time on x-axis.
    """
    con.log("Plotting histogram")

    # Reshape and sum bins to get hourly data
    hourly_peaks = bin_peaks.reshape(-1, bins_per_timestep).sum(axis=1)

    # Plot spike frequency over time
    fig, ax = plt.subplots()
    ax.bar(
        list(range(len(hourly_peaks))),
        hourly_peaks,
        width=1,
        edgecolor="black",
        align="edge",
    )
    ax.set_title("EOD count over Time")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Number of Peaks")
    ax.set_xticks(list(range(len(hourly_peaks))))
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

    # calculate peaks per bin
    binned_peaks = peaks_over_time(data_paths, num_bins=24 * 60, min_per_rec=5)

    # plot time histogram
    plot_peaks_time(
        binned_peaks,
        save_path="/home/eisele/wrk/mscthesis/data/processed/eod_count_over_time_hist.png",
        bins_per_timestep=60,
    )


if __name__ == "__main__":
    main()


# %%
