# %%
from rich.console import Console
import numpy as np
# from IPython import embed

# Initialize console for logging
con = Console()


# %%
# Load data from previous script
def load_data(path):
    """
    Load data from a .npz file.
    Input: path (str) - path to the .npz file
    Output: eod_counts, minute_idx, rec_count_min (arrays of eod counts, minute indices, and recording counts per minute)
    """
    con.log(f"Loading data from {path}.")

    # load npz file
    with np.load(path, allow_pickle=True) as data:
        rec_ids = data["rec_path_id"]
        session_ids = data["session_id"]
        minutes = data["minute"]
        eod_counts = data["eods_per_rec_min"]
        rec_counts = data["recs_per_min"]

    return rec_ids, session_ids, minutes, eod_counts, rec_counts


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
        all_eods_per_min.append(eod_per_rec_and_min[minutes == i])

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
    rec_per_min = np.array(
        [len(eods) for eods in all_eods_per_min]
    )  # TODO: rename this array/check which rec count variable is correct!

    return sum_eod_per_min, rec_per_min


# reshape bins to get eod count per desired bin size (e.g. hour)
def bin_eods_recs(sum_eod_per_min, sum_rec_per_min, mins_per_timestep):
    """
    Bin eods per minute into desired timestep. Same for recs per minute
    """
    # reshape and sum bins to get hourly eod counts
    binned_eods = sum_eod_per_min.reshape(-1, mins_per_timestep).sum(axis=1)
    # reshape and sum bins to get hourly rec counts
    binned_recs = sum_rec_per_min.reshape(-1, mins_per_timestep).sum(
        axis=1
    )  # TODO: rename this array

    return binned_eods, binned_recs


def sem_per_bin(binned_eods, binned_recs):
    """
    Calculate standard error of the mean (SEM) for eod counts per minute.
    Input: eods_per_min, recs_per_min (arrays of eod counts and rec counts per minute)
    Output: sem_eods_per_min (array of SEM values for each minute)
    """
    # initialize array to store SEM per bin
    sem = np.zeros_like(binned_eods)

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


# save relevant variables from this script to npz file for plotting in diff script
def save_data(
    binned_eods,
    binned_recs,
    sum_eod_per_min,
    rate,
    sem,
    rate_sem,
    save_path=None,
):
    """
    Save intermediate data to npz file.
    """
    if save_path:
        con.log(f"Saving data to npz file in {save_path}.")
        np.savez(
            save_path,
            binned_eod_counts=binned_eods,
            binned_rec_counts=binned_recs,
            eod_per_min=sum_eod_per_min,
            pulse_rates=rate,
            sem_eod_count=sem,
            sem_eod_rate=rate_sem,
        )


# %%
def main():
    # load data from previous script
    rec_ids, session_ids, minute_idx, eod_counts, rec_count_min = load_data(
        path="/home/eisele/wrk/mscthesis/data/intermediate/intermediate_pulse_data.npz",
    )

    # global variables to adjust timescale of analysis
    timescale_in_minutes = 24 * 60  # 24 hours in minutes
    minutes_per_bin = 60

    # store eod counts of all recs per minute in a list of lists
    eod_counts_per_min = create_eods_per_min(
        eod_counts, minute_idx, timescale_in_minutes
    )

    # sum up all individual eod counts and number of recs for each minute
    eod_per_min, rec_per_min = sum_eods_rec_per_min(eod_counts_per_min)

    # bin eod counts per minute into desired timestep (e.g. hour)
    binned_eod_counts, binned_rec_counts = bin_eods_recs(
        eod_per_min, rec_count_min, minutes_per_bin
    )

    # calcutate standard error of the mean (SEM) for eod counts per bin
    sem_eod_counts = sem_per_bin(binned_eod_counts, binned_rec_counts)

    # calculate firing rate in Hz for each bin
    firing_rates, sem_eod_rate = rate_per_bin(
        binned_eod_counts, sem_eod_counts, minutes_per_bin
    )

    # save data for plotting in a separate script
    save_data(
        binned_eod_counts,
        binned_rec_counts,
        eod_per_min,
        firing_rates,
        sem_eod_counts,
        sem_eod_rate,
        save_path="/home/eisele/wrk/mscthesis/data/intermediate/pulse_data.npz",
    )


if __name__ == "__main__":
    main()
