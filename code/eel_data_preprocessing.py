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
# TODO: improve plots

# %%
from rich.console import Console
from pathlib import Path
import numpy as np
import nixio
from thunderlab.dataloader import DataLoader
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
# from IPython import embed

# from IPython import embed

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


# %%
def find_wav(file_paths):
    """
    Takes the sorted path_list from get_path_list function, that contains one or several hdf5 files.
    Builds the path to the corresponding wavfiles by iterating over h5 files.
    Stores the first wav file of that recording session in a list and returns the list of wav file paths.
    """
    # Print status
    con.log("Accessing corresponding wav files.")

    # initialize empty list to store wav file paths
    wav_path_list = []

    # iterate through hdf5 paths and build path to corresponding wav files
    for fp in file_paths:
        wav_name = "_".join(str(fp.name).split("_")[:-1])
        wavpath = Path(
            fp.parent.parent.parent.parent
            / "raw/trial_dataset_new/fielddata_trial/inpa2025_trial/inpa_enclosure"
            / wav_name
        )

        # get the first wav file of directory and store the path to it in list
        first_file = sorted(wavpath.rglob("*.wav"))[0]
        wav_path_list.append(first_file)

    return wav_path_list


# Extracts sampling rates in Hz from the first corresponding wav files of each hdf5 file and stores them in a list
def load_wav(wav_list):
    fs_list = []
    dt_list = []
    length_list = []

    # iterate through wav files
    for wav in wav_list:
        # load wav file
        try:
            audio_data = DataLoader(wav)
        except Exception as e:
            con.log(f"Error loading: {e}")
            continue

        ### get sampling rate in Hz for this wav file
        fs = audio_data.rate
        # add sampling rate of each wav to list
        fs_list.append(fs)

        ### get file name
        filename = wav.stem

        # get start time of recording from first wav file name (str)
        start_time = filename.split("-")[1]

        # convert to datetime object
        dt_start = datetime.strptime(start_time, "%Y%m%dT%H%M%S")

        # add start time datetime object to list
        dt_list.append(dt_start)

        ### get file length
        # get number of minutes per recording for this wav file
        samples_per_rec = audio_data.shape[0]
        minutes_rec = samples_per_rec / fs / 60

        # add recording length in minutes to list
        length_list.append(minutes_rec)

    return fs_list, dt_list, length_list


# calculate number of pulses per time bin
def make_histogram(pulse_centers, sampling_rates, start_times):
    # create list to hold histogram
    time_range = 24 * 60  # min
    histogram_data = np.zeros(time_range)
    # hist_monthly = np.zeros(12*31)
    timestamp_list = []

    # iterate through each of the lists in pulse_centers (one per hdf5 file)
    for i, rec in enumerate(pulse_centers):
        # for each pulse list, iterate through the pulse indices
        for idx in rec:
            # divide the pulse idx with the sampling rate of the corresponding recording session to get the time of the pulse in seconds
            pulse_time_sec = idx / sampling_rates[i]

            # get absolute time of pulse by adding start time of recording session
            pulse_time_abs = start_times[i] + timedelta(seconds=pulse_time_sec)

            # get number of total minutes
            minute = pulse_time_abs.hour * 60 + pulse_time_abs.minute

            # append
            histogram_data[minute] += 1

            # Convert pulse index to Unix timestamp (float64) for later storage in hdf5 file
            pulse_time_abs_unix = pulse_time_abs.timestamp()  # float64
            timestamp_list.append(pulse_time_abs_unix)

    return histogram_data, timestamp_list


def make_monthly_histogram(timestamps, n_months=12):
    """Aggregate Unix timestamps into consecutive monthly bins.

    timestamps: iterable of float (unix timestamps)
    n_months: number of consecutive months to aggregate starting at the month
              of the earliest timestamp

    Returns: (counts (np.array length n_months), month_starts (list of datetime))
    """
    if not timestamps:
        return np.zeros(n_months, dtype=int), []

    # convert to datetimes
    dates = [datetime.fromtimestamp(ts) for ts in timestamps]
    start = min(dates)
    # beginning of the first month
    start_month = datetime(start.year, start.month, 1)

    # build list of month starts
    month_starts = [start_month]
    for i in range(1, n_months):
        prev = month_starts[-1]
        year = prev.year + (prev.month // 12)
        month = (prev.month % 12) + 1
        month_starts.append(datetime(year, month, 1))

    counts = np.zeros(n_months, dtype=int)
    for dt in dates:
        months_diff = (dt.year - start_month.year) * 12 + (dt.month - start_month.month)
        if 0 <= months_diff < n_months:
            counts[months_diff] += 1

    return counts, month_starts


# create a hdf5 file with nixio to later save the timestamp of each pulse in it
def open_nix_for_output(output_path: Path):
    nix_file = nixio.File.open(str(output_path), nixio.FileMode.Overwrite)
    nix_timestamps = nix_file.create_block(name="Timestamp", type_="datetime")

    return nix_file, nix_timestamps


# save the unix timestamp of each pulse in the earlier created nix_timestamp block of nix_file
def append_cluster_block(
    time_stamp_block, time_stamp_list: list, created: bool
) -> bool:
    if not time_stamp_list:
        return created

    if not created:
        time_stamp_block.create_data_array(
            "timestamps", "timestamps", data=time_stamp_list
        )

    for time in time_stamp_list:
        time_stamp_block.data_arrays["timestamps"].append(time)

    return True


# %% Main
# def main():
# Path to sup folder
datapath = Path(
    "/home/eisele/wrk/mscthesis/data/newdata/eels-mfn2021_dummy_pulses_redetected/berlin_tank_site"
)

# make list containing all paths to hdf5 files in the given datapath
path_list = get_path_list(datapath)

# load hdf5 files from path list and extract pulse centers of pulses that were predicted as EODs
pulse_centers, sampling_rates, start_times = load_eods(path_list)

# # make list of the first wav file of each recording session corresponding to the hdf5 files in path_list
# wav_list = find_wav(path_list)

# # get sampling rates from the first wav file of each recording session
# sampling_rates, start_times, _ = load_wav(wav_list)

# extract sampling rates and start times from the first


# calculate histogram of number of pulses per minute for 24‑h period (0…1439 minutes)
histogram, timestamps = make_histogram(pulse_centers, sampling_rates, start_times)

# create a hdf5 file with nixio to later save the timestamp of each pulse in it
nix_file, nix_block = open_nix_for_output(
    Path(
        "/home/eisele/wrk/mscthesis/data/intermediate/inpa2025_peaks/pulse_timestamps.nix"
    )
)

# save the unix timestamp of each pulse in the earlier created nix_timestamp block of nix_file
created = append_cluster_block(nix_block, timestamps, created=False)

# if __name__ == "__main__":
#     main()

# %%
#################################
############# PLOTS #############
#################################

## activity over time - 24h, minute bins
# x coordinates run 0…1439 (= minutes since midnight)
x = np.arange(len(histogram))

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(x, histogram, width=1.0, align="edge", color="C0")

# label the x‑axis in hours every 60 minutes
hour_ticks = np.arange(0, 24 * 60 + 1, 60)
hour_labels = [f"{h:02d}:00" for h in range(25)]
ax.set_xticks(hour_ticks)
ax.set_xticklabels(hour_labels, rotation=45)

ax.set_xlim(0, 24 * 60)
ax.set_xlabel("time of day")
ax.set_ylabel("pulses per minute")
ax.set_title("24‑h histogram (1‑min bins)")

plt.tight_layout()
plt.show()

# %%
## firing rate over time, 24h, minute bins
# firing rate in Hz for each minute: convert pulses per minute -> pulses per second
firing_rate = histogram / 60.0  # Hz

# plot firing rate (1-min bins)
x = np.arange(len(firing_rate))
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(x, firing_rate, color="C1")

# label the x‑axis in hours every 60 minutes
hour_ticks = np.arange(0, 24 * 60 + 1, 60)
hour_labels = [f"{h:02d}:00" for h in range(25)]
ax.set_xticks(hour_ticks)
ax.set_xticklabels(hour_labels, rotation=45)

ax.set_xlim(0, 24 * 60)
ax.set_xlabel("time of day")
ax.set_ylabel("firing rate (Hz)")
ax.set_title("24‑h firing rate (1‑min bins)")

plt.tight_layout()
plt.show()

# %%
## activity over time - 24h, hour bins
x = np.arange(0, len(histogram), 60)
histogram_hourly = [sum(histogram[i : i + 60]) for i in range(0, len(histogram), 60)]

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(x, histogram_hourly, width=60, align="edge", color="C0")

# label the x‑axis in hours every 60 minutes
hour_ticks = np.arange(0, 24 * 60 + 1, 60)
hour_labels = [f"{h:02d}:00" for h in range(25)]
ax.set_xticks(hour_ticks)
ax.set_xticklabels(hour_labels, rotation=45)

ax.set_xlim(0, 24 * 60)
ax.set_xlabel("time of day")
ax.set_ylabel("pulses per minute")
ax.set_title("24‑h histogram (1‑min bins)")

plt.tight_layout()
plt.show()

# %%
## activity over time, 12 months, daily bins
# TODO: ?

# %%
## activity over time, 12 months, monthly bins

# aggregate timestamps into 12 consecutive monthly bins (starting at earliest timestamp)
monthly_counts, month_starts = make_monthly_histogram(timestamps, n_months=12)

# plot monthly histogram
if len(monthly_counts) > 0:
    x = np.arange(len(monthly_counts))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x, monthly_counts, color="C0", align="center")

    ax.set_xticks(x)
    ax.set_xticklabels([dt.strftime("%Y-%m") for dt in month_starts], rotation=45)

    ax.set_xlabel("month")
    ax.set_ylabel("pulses")
    ax.set_title("12‑month histogram (monthly bins)")

    plt.tight_layout()
    plt.show()

# %%
## activity over time, years
# TODO

# %%
