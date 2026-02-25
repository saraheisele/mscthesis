"""
This script loads preprocessed data from eel_data_preprocessing script
and calculates the number of detected peaks per minute/hour and plots the result as histogram.
"""

# TODO: check which binned recs variable is true/best and why errorbars are so small
# TODO: check if no normalization necessary if i do mean (bc that would be bdividing by n recs 2 time)??
# TODO: write documentation for peaks_over_time function
# TODO: write script that cleans up data from eelgrid and eellogger from sd cards in april
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
# from IPython import embed

# Initialize console for logging
con = Console()


# %%
# TODO: implement this cahnge paths function (the code reappears in extract data function)
def change_paths_in_json(load_path):
    """
    Change paths in a json file from old base path to new base path.
    Input:
        - load_path: Path to the json file
        - old_base: Old base path to be replaced
        - new_base: New base path to replace the old one
    """
    # load preprocessed paths of recording sessions from json file
    with open(load_path, "r") as file:
        data = json.load(file)

    # Define the old and new base paths
    old_base = "/mnt/data1/eels-mfn2021_peaks"
    new_base = "/home/efish/labdata/eels-mfn2021_peaks"

    # change paths from to new location of data
    for key in data:
        data[key] = [path.replace(old_base, new_base) for path in data[key]]

    # define path where json file is created
    path = Path(
        "/home/eisele/wrk/mscthesis/data/intermediate/eellogger_session_paths.json"
    )

    # Save the updated JSON
    with path.open("w") as f:
        json.dump(data, f)


def extract_data(load_path):
    """
    Extract data from json file and convert to list of paths.
    """
    # load preprocessed paths of recording sessions from json file
    with open(load_path, "r") as file:
        data = json.load(file)

    # Define the old and new base paths
    old_base = "/mnt/data1/eels-mfn2021_peaks"
    new_base = "/home/efish/labdata/eels-mfn2021_peaks"

    # Update paths
    for key in data:
        data[key] = [path.replace(old_base, new_base) for path in data[key]]

    # Save the updated JSON
    with open("your_file_updated.json", "w") as f:
        json.dump(data, f, indent=4)

    # extract all rec paths into one list and convert to Path objects
    all_paths = [Path(file) for file_list in data.values() for file in file_list]

    # extract session id for each recording from key names
    session_ids = [i for i, file_list in enumerate(data.values()) for file in file_list]

    return all_paths, session_ids


# Extract time, start and end bin from filename (called in peaks_over_time)
def get_timestamps(npz_file_path, min_per_rec):
    """
    Get start and end time and bins from the file path to npz file.
    Input:
        - npz_file_path: Path to the npz file
        - min_per_rec: int, number of minutes per recording
    Returns:
        - start_bin: int, start bin of the recording
        - end_bin: int, end bin of the recording
        - dt_start: datetime object, start time of the recording
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
    # initlialize array to store number of recordings per minute
    recs_per_min = np.zeros(num_bins)

    # initialize array to store recording path ids
    rec_path_ids = np.full(len(paths) * min_per_rec, None, dtype=object)

    # initialize array to store session ids
    session_ids = np.full(len(paths) * min_per_rec, None, dtype=object)

    # initialize array to store current minute idx
    minutes = np.full(len(paths) * min_per_rec, None, dtype=object)

    # initialize array to store EOD count per minute per rec
    eod_per_rec_and_min = np.full(len(paths) * min_per_rec, None, dtype=object)

    ### iterate over each file path in path list
    # track progress with progress bar
    for i, file in track(
        enumerate(paths), total=len(paths), description="Processing files..."
    ):
        ### get start and end bin of each rec
        # only bins that the recording covers in whole are considered, start and end bins are discarded in case rec starts/ends at a non-full minute
        start_bin, end_bin, start_time = get_timestamps(file, min_per_rec)

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
            true_startbin = (start_time.hour * 60) + start_time.minute
            global_peak_idx = int((start_idx + peak) // idx_per_min) + true_startbin

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

                # manage counters
                fill_idx += 1
                eod_counter = 0

            ### add peaks into the correct bin
            if global_peak_idx >= start_bin and global_peak_idx <= end_bin:
                eod_counter += 1

            # change last peak idx variable
            last_global_peak_idx = global_peak_idx

        ### increment number of recordings per bin
        recs_per_min[start_bin:end_bin] += 1

    return (
        rec_path_ids,
        session_ids,
        minutes,
        eod_per_rec_and_min,
        recs_per_min,
    )


# save relevant variables from this script to npz file for plotting in diff script
def save_intermediate(
    rec_path_ids,
    session_ids,
    minutes,
    eod_per_rec_and_min,
    recs_per_min,
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
            recs_per_min=recs_per_min,
        )


# %% Main
def main():
    # global variables to adjust timescale of analysis
    timescale_in_minutes = 24 * 60  # 24 hours in minutes

    # change_paths_in_json(
    #     load_path="/home/eisele/wrk/mscthesis/data/intermediate/eellogger_session_paths.json"
    # )
    # exit()

    # extract data from json file
    data_paths, session_ids = extract_data(
        load_path="/home/eisele/wrk/mscthesis/data/intermediate/eellogger_session_paths.json"
    )

    # get path, session id, minute and eod count for that minute for each recording
    rec_paths, sessions, minute_idx, eod_counts, rec_count_min = peaks_over_time(
        data_paths, session_ids, timescale_in_minutes, min_per_rec=5
    )

    # save data
    save_intermediate(
        rec_paths,
        sessions,
        minute_idx,
        eod_counts,
        rec_count_min,
        save_path="/home/eisele/wrk/mscthesis/data/intermediate/intermediate_pulse_data.npz",
    )


if __name__ == "__main__":
    main()
