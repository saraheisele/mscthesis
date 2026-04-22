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

The script consists of 4 steps:
1. Access and store paths of npz files and corresponding wav files in seperate sorted lists.
2. Check both lists for missing, empty or corrupt files and exclude them from further analysis.
3. Sort npz file paths into recording sessions based on the time stamps in the file names.
This step is necessary because sometimes there are several recording sessions within the same folder.
This step produces a dictionary that contains as many keys as rec sessions,
each key contains a list of npz file paths that belong to that session.
4. Save the dictionary with the session paths to a json file for later use.

The json file created in this script is loaded and used in the next step of this analysis (eel_data_analysis.py).
"""

# TODO: write documentation and comments for this script
# TODO: Maybe store fs, rec length and start time in npz file/dictionary

# %%
from rich.console import Console
from pathlib import Path
import numpy as np
import nixio
from datetime import datetime, timedelta, date
from IPython import embed

# Initialize console for logging
con = Console()


### Functions ###
def get_path_list(datapath):
    """
    Make a list fo all the paths of hdf5 files from the given file, directory or supdirectory.
    """
    # Print status
    con.log("Loading detected pulses from hdf5 files.")

    # Check if the path exists
    if not datapath.exists():
        raise FileNotFoundError(f"Path {datapath} does not exist.")

    # Initialize list to store file paths
    path_list = []

    # Check if the path is a directory, a file or a directory containing files
    if datapath.is_file():
        # Check if the file is an hdf5 file
        if datapath.suffix == ".h5":
            con.log(f"Path {datapath} is a single hdf5 file.")
            # Store objects in list for consistency with directory case
            path_list.append(datapath)
        else:
            raise FileNotFoundError(f"File {datapath} is not an hdf5 file.")

    # Recursively find all .h5 files in the directory and subdirectories
    elif datapath.is_dir():
        for file in datapath.rglob("*.h5"):
            path_list.append(file)

    else:
        raise FileNotFoundError(
            f"Path {datapath} is not a file, directory containing files or directory containing folders containing files."
        )

    return sorted(path_list)


def load_eods(file_paths):
    # Print status
    con.log("Loading hdf5 files.")

    pulse_center_list = []
    fs_list = []
    dt_list = []

    for fp in file_paths:
        file = nixio.File.open(str(fp), nixio.FileMode.ReadWrite)
        block = file.blocks["pulses"]
        # data_array_names = [da.name for da in block.data_arrays]

        data_array_names = [da.name for da in block.data_arrays]
        if "centers" not in data_array_names:
            con.log(f"File {fp} does not contain 'centers' data array. Skipping.")
            continue

        pulses_center_idx = block.data_arrays["centers"]
        pred_labels = block.data_arrays["predicted_labels"]

        con.log(
            f"Found total of {np.sum(pred_labels[:] == 1)} predicted pulses in file {fp}"
        )

        pulse_center_list.append(pulses_center_idx[pred_labels[:] == 1])

        # extract fs in Hz
        section = file.sections["pulses_metadata"]

        fs = section["metadata"]["samplerate"]
        starttime_str = section["metadata"]["metadata"]["INFO"]["DateTimeOriginal"]

        # add sampling rate of each wav to list
        fs_list.append(fs)

        # convert to datetime object
        dt_start = datetime.strptime(starttime_str, "%Y-%m-%dT%H:%M:%S")

        # add start time datetime object to list
        dt_list.append(dt_start)

    return pulse_center_list, fs_list, dt_list


# calculate number of pulses per time bin
def make_histogram(pulse_centers, sampling_rates, start_times):
    # create list to store unix timestamps of each pulse for later storage in hdf5 file
    timestamp_list = []

    # create histograms in a dict
    hist_sizes = {
        "minute": 24 * 60,
        "hour": 24,
        "day": 366,
        "month": 12,
        "year": (date.today().year - 2023) + 1,
    }
    # TODO: dont hardcode start year, but get it from start_times list - ask patrick how

    hist = {k: np.zeros(v, dtype=int) for k, v in hist_sizes.items()}

    # create dict to hold recording time per bin for normalization
    rec_time_hist = {k: np.zeros_like(v) for k, v in hist.items()}

    con.log("Calculating histogram of pulses per minute.")

    # iterate through each of the lists in pulse_centers (one per hdf5 file)
    for i, rec in enumerate(pulse_centers):
        # initiate new arrays for each for each h5 file/pulse list
        # to store pulse counts so they can also be used to increment the recording counter
        prelim_min = np.zeros(24 * 60, dtype=int)
        prelim_hour = np.zeros(24, dtype=int)
        prelim_day = np.zeros(366, dtype=int)
        prelim_month = np.zeros(12, dtype=int)
        prelim_year = np.zeros((date.today().year - 2023) + 1, dtype=int)

        # --- dict option ---
        preliminary_hist = {k: np.zeros_like(v) for k, v in hist.items()}

        # for each pulse list, iterate through the pulse indices
        for idx in rec:
            # get the time of the pulse in seconds
            pulse_time_sec = idx / sampling_rates[i]  # sampling rate in Hz

            # get absolute time of each pulse by adding start time of recording session
            pulse_time_abs = start_times[i] + timedelta(seconds=pulse_time_sec)

            # extract time components of pulses for histogramming
            minute = pulse_time_abs.hour * 60 + pulse_time_abs.minute
            hour = pulse_time_abs.hour
            day = pulse_time_abs.timetuple().tm_yday - 1  # day of year (0‑365)
            month = pulse_time_abs.month - 1  # month of year (0‑11)
            year = pulse_time_abs.year - 2023  # year since start of recordings

            # append counts for each time bin of the current pulse to the array of the current hdf5 file
            prelim_min[minute] += 1
            prelim_hour[hour] += 1
            prelim_day[day] += 1
            prelim_month[month] += 1
            prelim_year[year] += 1

            ## convert pulse to Unix timestamp and append to list for later storage in hdf5 file
            pulse_time_abs_unix = pulse_time_abs.timestamp()  # float64
            timestamp_list.append(pulse_time_abs_unix)

        ## append count arrays for i to dict of histograms
        hist["minute"] += prelim_min
        hist["hour"] += prelim_hour
        hist["day"] += prelim_day
        hist["month"] += prelim_month
        hist["year"] += prelim_year

        ## increment indices to which i contributed to dict of recording time
        rec_time_hist["minute"][prelim_min > 0] += 1
        rec_time_hist["hour"][prelim_hour > 0] += 1
        rec_time_hist["day"][prelim_day > 0] += 1
        rec_time_hist["month"][prelim_month > 0] += 1
        rec_time_hist["year"][prelim_year > 0] += 1

    con.log("Finished calculating histogram.")
    embed()
    return hist, timestamp_list


# def get_rec_time()


#################################
############# SAVE #############
#################################


# create a hdf5 file with nixio to later save the timestamp of each pulse in it
def open_nix_for_output(output_path: Path):
    nix_file = nixio.File.open(str(output_path), nixio.FileMode.Overwrite)
    nix_timestamps = nix_file.create_block(name="Timestamp", type_="datetime")

    return nix_file, nix_timestamps


# save the unix timestamp of each pulse in the earlier created nix_timestamp block of nix_file
def append_cluster_block(
    time_stamp_block, time_stamp_list: list, created: bool
) -> bool:
    con.log("Saving pulse timestamps to nix file.")

    if not time_stamp_list:
        return created

    if not created:
        time_stamp_block.create_data_array(
            "timestamps", "timestamps", data=time_stamp_list
        )

    for time in time_stamp_list:
        time_stamp_block.data_arrays["timestamps"].append(time)

    return True


def save_hist(histogram_dict, output_path: Path):
    con.log(f"Saving histogram dict to {output_path}.")
    # ensure values are numpy arrays
    clean = {k: np.asarray(v) for k, v in histogram_dict.items()}
    # save (hist is your dict of numpy arrays)
    np.savez_compressed(output_path, **clean)


# TODO: improve name of npz file
# TODO: should I save the histogram dict in nix file as well?


# %% Main
def main():
    # Path to data directory containing hdf5 files with detected pulses
    data_path = Path(
        "/home/eisele/wrk/mscthesis/data/newdata/eels-mfn2021_dummy_pulses_redetected/berlin_tank_site"
    )

    # path to output directory
    # save_path = Path("/home/eisele/wrk/mscthesis/data/newdata")
    # TODO: implement save path nicely in both save functions

    # make list containing all paths to hdf5 files in the given datapath
    path_list = get_path_list(data_path)

    # load hdf5 files from path list and extract pulse centers of pulses that were predicted as EODs
    pulse_centers, sampling_rates, start_times = load_eods(path_list)

    # calculate histogram of number of pulses per minute for 24‑h period (0…1439 minutes)
    histogram_dict, timestamps = make_histogram(
        pulse_centers, sampling_rates, start_times
    )

    # create a hdf5 file with nixio to later save the timestamp of each pulse in it
    nix_file, nix_block = open_nix_for_output(
        Path("/home/eisele/wrk/mscthesis/data/newdata/berlin_pulse_timestamps_2026.nix")
    )

    # save the unix timestamp of each pulse in the earlier created nix_timestamp block of nix_file
    created = append_cluster_block(nix_block, timestamps, created=False)  # noqa: F841

    # save histogram dict to .npz file for later use in plotting
    save_hist(
        histogram_dict,
        Path("/home/eisele/wrk/mscthesis/data/newdata/berlin_histogram_dict_2026.npz"),
    )


if __name__ == "__main__":
    main()

# %%
