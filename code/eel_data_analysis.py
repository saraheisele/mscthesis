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
# make progress bar more simple (track)
# normalization in seperate function (account for amount of recordings that contribute to each bin)
# use arrays instead of lists
# change peaks_over_time function to store all necessary data in arrays
# save session idx, path idx (from json), minute, EOD count per minute per rec in npz file

# Monday:
# plans for brazil

# Tuesday:
# test which resistor is best for eelgrid data recordings
# ordered hardware for brazil
# restructured code to calculate sem using new variables from peaks_over_time function
# function that makes variables of len num_bins (e.g. 1440 for 24h) to store eod counts of each rec for each minute
# first binning minutes variable into hourly bins then calculate sem per hour
# got rid of normalizing step bc thats being done in calculating the std (mean) already (according to benda)
# downloaded data from peter (may, wetransfer)

# Wednesday:
# TODO: check if no normalization necessary if i do mean (bc that would be bdividing by n recs 2 time)??
# refine whats being saved
# TODO: split script in 2 halves
# use npz file to to plot sem error bars (determine certainty)
# TODO: ask if theres diff for sem with n of rec or n of min per bin??
# make rec count y axis in minutes
# clean up code
# TODO: write documentation for peaks_over_time function


# TODO: make csv of metadata (automate this w LLM ?)!
# TODO: check for correlations with external factors (e.g. temperature, salinity, time of day)
# TODO: peaks over years, months, temp, leitfähigkeit, individuum

# %%
from rich.console import Console
from rich.progress import track
import numpy as np
from datetime import datetime
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
    session_ids = [i for i, file_list in enumerate(data.values()) for file in file_list]

    return all_paths, session_ids


# Extract time from filename (called in peaks_over_time)
def get_timestamps(npz_file_path, min_per_rec=int):
    """
    Get start and end time from the filenames of the first and last npz files in one recording session.
    """
    # get file name
    filename = npz_file_path.name

    # get time stamp from file name
    datetime_str_start = filename.split("-")[1].split("_")[0]

    # convert to datetime object
    dt_start = datetime.strptime(datetime_str_start, "%Y%m%dT%H%M%S")
    # dt_end = dt_start + timedelta(minutes=min_per_rec)

    # calculate start and end bin of rec
    start_bin = (dt_start.hour * 60) + dt_start.minute
    end_bin = start_bin + min_per_rec - 1

    # if start of rec isn't at beginning of a minute ignore peaks in start and end bins
    if dt_start.second != 0:
        start_bin += 1

    return start_bin, end_bin, dt_start


# Loads one npz file created from patricks find pulses script (called in peaks_over_time)
def load_npz(file):
    """
    Load one npz file, corresponds to one recording.
    Extract valid peaks and sample rate of this recording.
    Input:
        - file: path to npz file
    Returns:
        - valid_peaks: np.array of valid peaks (pulses classified as pulse and not noise)
        - sample_rate: int, sample rate of the recording in Hz
    """
    # load npz file
    with np.load(file) as data:
        # extract peaks that have been predicted as valid peaks
        valid_peaks = data["centers"][data["predicted_labels"] == 1]

        # extract sample rate
        sample_rate = int(data["rate"])  # int

    return valid_peaks, sample_rate


# TODO: fix progress bar
# Count how many peaks happen per minute
def peaks_over_time(
    paths: list,  # list of paths to npz files
    sessions: list,  # list of session ids (same len as paths)
    num_bins: int,  # min
    min_per_rec: int,  # min
):
    """
    Calculate number of peaks per minute.
    """
    # initialize array to store total number of peaks per minute
    eods_per_min = np.zeros(num_bins)

    # initlialize array to store number of recordings per minute
    rec_per_min = np.zeros(num_bins)

    # initialize array to store recording path ids
    rec_path_ids = np.zeros(len(paths) * min_per_rec)

    # initialize array to store session ids
    session_ids = np.zeros(len(paths) * min_per_rec)

    # initialize array to store current minute idx
    minutes = np.zeros(len(paths) * min_per_rec)

    # initialize array to store EOD count per minute per rec
    eod_per_rec_and_min = np.zeros(len(paths) * min_per_rec)

    ### iterate over each file path in path list
    # track progress with progress bar
    for i, file in track(enumerate(paths)):
        ### get start and end bin of each rec
        # only bins that the recording covers in whole are considered, start and end bins are discarded in case rec starts/ends at a non-full minute
        start_bin, end_bin, start_time = get_timestamps(file)

        ### load current file
        valid_peaks, sample_rate = load_npz(file)

        # calculate data points per minute (sample rate in Hz * 60 sec/min)
        idx_per_min = sample_rate * 60

        # find start index within start bin
        start_idx = int(start_time.second * sample_rate)

        ### initialize counters for next for loop
        # counter for eod counts per minute per rec
        eod_counter = 0

        # counter for current idx in arrays to fill
        fill_idx = i * min_per_rec

        # store last global peak idx to check if new minute
        last_global_peak_idx = None

        ### loop over all peaks in file
        for peak in valid_peaks:
            # assign peaks to bins
            global_peak_idx = int((start_idx + peak) // idx_per_min) + start_bin

            # reset global peak idx if it exceeds number of bins so it goes into bin 0 after max bin
            if global_peak_idx >= num_bins:
                global_peak_idx = global_peak_idx - num_bins

            ### append arrays in every new minute/global_peak_idx
            if (
                last_global_peak_idx is not None
                and global_peak_idx != last_global_peak_idx
            ):
                # append arrays at current fill idx
                rec_path_ids[fill_idx] = i
                session_ids[fill_idx] = sessions[i]
                minutes[fill_idx] = last_global_peak_idx
                eod_per_rec_and_min[fill_idx] = eod_counter

                # increase counters
                fill_idx += 1
                eod_counter = 0

            ### add peaks into the correct bin
            if global_peak_idx >= start_bin and global_peak_idx <= end_bin:
                eods_per_min[global_peak_idx] += 1
                eod_counter += 1

            # change last peak idx variable
            last_global_peak_idx = global_peak_idx

        ### increment number of recordings per bin
        rec_per_min[start_bin:end_bin] += 1

    return (
        eods_per_min,
        rec_per_min,
        rec_path_ids,
        session_ids,
        minutes,
        eod_per_rec_and_min,
    )


# save relevant variables from this script to npz file for plotting in diff script
def save_intermediate(
    rec_path_ids,
    session_ids,
    minutes,
    eod_per_rec_and_min,
    save_path=None,
):
    """
    Save intermediate data to npz file.
    """
    if save_path:
        con.log(f"Saving intermediate data to npz file in {save_path}.")
        np.savez(
            save_path,
            rec_path_id=rec_path_ids,
            session_id=session_ids,
            minute=minutes,
            eods_per_rec_min=eod_per_rec_and_min,
        )


# store eod counts of each rec that controbuted to a minute into a list of lists for all minutes
def create_eods_per_min(eod_per_rec_and_min, minutes, num_bins):
    """
    Store the individual eod counts of each recording that contributed to a minute in a list.
    Input:
        - eod_per_rec_and_min (array of eod counts of all minutes one rec contributed to, sorted after rec no/idx),
        - minutes (array of minute indices, sorted same as eod_per_rec_and_min),
        - num_bins (number of bins)
    Output:
        - all_eods_per_min (list of length num_bins, each element is a list of eod counts for that minute)
    """
    # store eod counts of all recs per minute
    all_eods_per_min = []

    # iterate over all minutes and store a list of eod values at that minute in a list
    for i in range(num_bins):
        all_eods_per_min.append([eod_per_rec_and_min[minutes == i]])

    return all_eods_per_min


# sum eod counts and rec counts per minute
def sum_eods_rec_per_min(all_eods_per_min):
    """
    Sum eod counts and rec counts per minute.
    Input: all_eods_per_min (list of lists, each list contains eod counts for that minute)
    Output:
        - sum_eod_per_min (array of summed eod counts per minute),
        - rec_per_min (array of number of recordings per minute)
    """
    # sum eod counts across recs per minute
    sum_eod_per_min = np.array([sum(eods) for eods in all_eods_per_min])
    # store number of recordings per minute
    rec_per_min = np.array([len(eods) for eods in all_eods_per_min])

    return sum_eod_per_min, rec_per_min


# reshape bins to get eod count per desired bin size (e.g. hour)
def bin_eods_recs(sum_eod_per_min, sum_rec_per_min, mins_per_timestep):
    """
    Bin eods per minute into desired timestep. Same for recs per minute
    """
    # reshape and sum bins to get hourly eod counts
    binned_eods = sum_eod_per_min.reshape(-1, mins_per_timestep).sum(axis=1)
    # reshape and sum bins to get hourly rec counts
    binned_recs = sum_rec_per_min.reshape(-1, mins_per_timestep).sum(axis=1)

    return binned_eods, binned_recs


def sem_per_bin(binned_eods, binned_recs):
    """
    Calculate standard error of the mean (SEM) for eod counts per minute.
    Input: eods_per_min, recs_per_min (arrays of eod counts and rec counts per minute)
    Output: sem_eods_per_min (array of SEM values for each minute)
    """
    # initialize array to store SEM per bin
    sem = np.zeros_like(binned_eods)
    # TODO: binned recs is not actual amount of recordings per bin, rather number of minutes that contributed to this bin

    # calculate SEM for each bin
    for i in range(len(binned_eods)):
        if binned_recs[i] > 0:
            sem[i] = np.std(binned_eods[i]) / np.sqrt(binned_recs[i])
        # set SEM to 0 if no recordings for this minute
        else:
            con.log(f"No recordings for minute {i}, setting SEM to 0.")
            sem[i] = 0

    return sem


# calculate firing rate per bin/timestep
def rate_per_bin(eods_per_timestep, sem, mins_per_timestep):
    """
    Calculates firing rate in Hz per bin.
    """
    # convert count data and count sem to rate in Hz
    rate = eods_per_timestep / mins_per_timestep / 60
    rate_sem = sem / mins_per_timestep / 60
    return rate, rate_sem


# Plot histogram of number of peaks per time bin
def plot_eod_hist(binned_eods, recs_per_bin, minute_eods, save_path=None):
    """
    Plot histogram of number of peaks per time bin with actual time on x-axis.
    """
    con.log("Plotting histogram.")

    # Create a time axis for minute counts
    minute_time_axis = np.linspace(
        0, len(binned_eods), len(minute_eods), endpoint=False
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
        minute_eods,
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
    ax.set_ylabel("Number of Sampled Minutes")
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
def plot_firing_rate(fr_per_bin, rate_sem, save_path=None):
    """
    Plot firing rate per time bin with error bars.
    """
    con.log("Plotting firing rates with error bars.")

    # Plot firing rate over time with error bars
    fig, ax = plt.subplots()
    ax.errorbar(
        list(range(len(fr_per_bin))),
        fr_per_bin,
        yerr=rate_sem,
        fmt="o",
        capsize=5,
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
    # global variables to adjust timescale of analysis
    timescale_in_minutes = 24 * 60  # 24 hours in minutes
    minutes_per_bin = 60

    # extract data from json file
    data_paths, session_ids = extract_data(
        load_path="/home/eisele/wrk/mscthesis/data/intermediate/eellogger_session_paths.json"
    )

    # get path, session id, minute and eod count for that minute for each recording
    rec_paths, sessions, minute_idx, eod_counts = peaks_over_time(
        data_paths, session_ids, timescale_in_minutes, min_per_rec=5
    )

    # save data
    save_intermediate(
        rec_paths,
        sessions,
        minute_idx,
        eod_counts,
        save_path="/home/eisele/wrk/mscthesis/data/intermediate/intermediate_pulse_data.npz",
    )

    # ---- put in seperate script ----
    # store eod counts of all recs per minute in a list of lists
    eod_counts_per_min = create_eods_per_min(
        eod_counts, minute_idx, timescale_in_minutes
    )

    # sum up all individual eod counts and number of recs for each minute
    eod_per_min, rec_per_min = sum_eods_rec_per_min(eod_counts_per_min)

    # bin eod counts per minute into desired timestep (e.g. hour)
    binned_eod_counts, binned_rec_counts = bin_eods_recs(
        eod_per_min, rec_per_min, minutes_per_bin
    )

    # calcutate standard error of the mean (SEM) for eod counts per bin
    sem_eod_counts = sem_per_bin(binned_eod_counts, binned_rec_counts)

    # calculate firing rate in Hz for each bin
    firing_rates = rate_per_bin(binned_eod_counts, minutes_per_bin)

    # ------------ put in seperate script ------------
    # plot eod count histogram
    plot_eod_hist(
        binned_eod_counts,
        binned_rec_counts,
        eod_per_min,
        save_path="/home/eisele/wrk/mscthesis/data/processed/eod_count_min_hour_rec_count.png",
    )

    # plot firing rate over time
    plot_firing_rate(
        firing_rates,
        sem_eod_counts,
        save_path="/home/eisele/wrk/mscthesis/data/processed/eod_firing_rate_over_time.png",
    )


if __name__ == "__main__":
    main()
