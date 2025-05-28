"""
This script loads the data (which format?) thats being save in th eel_data_analysis script.
It plots the number of detected peaks per minute/hour as histogram.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console

con = Console()


# %%
# Load data for plotting
def load_data(load_path):
    """
    Load data from a .npz file.
    Input: path (str) - path to the .npz file
    Output: eod_counts, minute_idx, rec_count_min (arrays of eod counts, minute indices, and recording counts per minute)
    """
    con.log(f"Loading data from {load_path}.")

    # load npz file
    with np.load(load_path) as data:
        eod_counts = data["binned_eod_counts"]
        rec_counts = data["binned_rec_counts"]
        eod_per_min = data["eod_per_min"]
        eod_rates = data["pulse_rates"]
        count_sem = data["sem_eod_count"]
        rate_sem = data["sem_eod_rate"]

    return eod_counts, rec_counts, eod_per_min, eod_rates, count_sem, rate_sem


# Plot histogram of number of peaks per time bin
def plot_eod_hist(binned_eods, recs_per_bin, minute_eods, count_sem, save_path=None):
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
        yerr=count_sem,
        capsize=5,
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
        fmt="-o",
        capsize=5,
        label="Firing Rate",
    )
    ax.set_title("EOD Pulse Rate over Time")
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
    (
        binned_eod_counts,
        binned_rec_counts,
        eod_per_min,
        eod_rates,
        sem_eod_counts,
        sem_eod_rate,
    ) = load_data(
        load_path="/home/eisele/wrk/mscthesis/data/intermediate/pulse_data.npz"
    )

    # plot eod count histogram
    plot_eod_hist(
        binned_eod_counts,
        binned_rec_counts,
        eod_per_min,
        sem_eod_counts,
        save_path="/home/eisele/wrk/mscthesis/data/processed/eod_count_min_hour_rec_count.png",
    )

    # plot firing rate over time
    plot_firing_rate(
        eod_rates,
        sem_eod_rate,
        save_path="/home/eisele/wrk/mscthesis/data/processed/eod_firing_rate_over_time.png",
    )


if __name__ == "__main__":
    main()
