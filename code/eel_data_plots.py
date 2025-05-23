"""
This script loads the data (which format?) thats being save in th eel_data_analysis script.
It plots the number of detected peaks per minute/hour as histogram.
"""

# %%
import matplotlib.pyplot as plt
from rich.console import Console

con = Console()


# %%
# Load data
def load_data(load_path):
    return None


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
    binned_peaks = load_data(load_path="/home/eisele/wrk/mscthesis/data/intermediate/")

    # plot time histogram
    plot_peaks_time(
        binned_peaks,
        save_path="/home/eisele/wrk/mscthesis/data/processed/eod_count_over_time_hist.png",
        bins_per_timestep=60,
    )


if __name__ == "__main__":
    main()
