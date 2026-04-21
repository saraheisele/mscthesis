"""This script contains functions from my eel_data processing pipeline that I dont need anymore/right now."""

import numpy as np
from rich.console import Console
from rich.progress import Progress
from pathlib import Path
import re
from datetime import datetime, timedelta, date
import json
from thunderlab.dataloader import DataLoader


# Initialize console for logging
con = Console()


# Import corresponding wav file - currently this function is unnessecary bc minutes per rec is not relevant for new approach!
def load_wav(datapath):
    """
    Load the corresponding wav file for the given npz file/s and returns recording length in min.
    """
    # Print status
    con.log("Loading wav files.")

    # Initialize list to store file paths globally
    # wav_path_list = []

    if datapath.is_file():
        # check if file starts with eellogger # TODO: necessary here?
        if datapath.name.startswith("eellogger") and datapath.suffix == ".npz":
            # Get the filename from the npz file
            filename = str(datapath.name).split("_")[0] + ".wav"

            # Get the parent directory of the npz file
            parent = "_".join(str(datapath.parent.parent).split("_")[:-1])

            # Construct the path to the wav file
            wavpath = Path(
                datapath.parent.parent.parent / parent / datapath.parent.name / filename
            )

            # Load the wav file
            # audio_data = AudioLoader(wavpath)

            # add path to list for consistency with directory case
            wav_path_list = [wavpath]

    elif datapath.is_dir():
        # Construct path to sup folder TODO: change this so not necessary to hardcode if its a sup/normal directory
        new_path_name = "_".join(str(datapath.name).split("_")[:-1])
        wavpath = Path(datapath.parent / new_path_name)

        # # Construct path to normal directory
        # parent = "_".join(str(datapath.parent).split("_")[:-1])
        # wavpath = Path(datapath.parent.parent / parent / datapath.name)

        # Match eellogger followed by any number, then -YYYYMMDDTHHMMSS
        pattern = re.compile(r"^eellogger\d+-\d{8}T\d{6}$")

        # Get all wav files of directory and store them alphabetically in a list
        wav_path_list = sorted(
            [
                f
                for f in wavpath.rglob("*.wav")
                # only store if normal stem and not empty
                if pattern.match(f.stem) and f.stat().st_size > 0
            ]
        )

        # # Initialize emtpy list to store number of minutes for each recording - not used atm!!
        # minutes_list = []

        # for i, wav in enumerate(wav_path_list):
        #     try:
        #         # Load each wav file
        #         audio_data = AudioLoader(wav)
        #     except Exception as e:
        #         con.log(f"Error loading: {e}")
        #         continue
        #     # Get sampling rate for this wav file
        #     fs = audio_data.rate

        #     # Get number of minutes per recording for this wav file
        #     samples_per_rec = audio_data.shape[0]
        #     minutes_rec = samples_per_rec / fs / 60

        #     # # Store in peak_data
        #     # npz_data[i]["minutes"] = minutes_rec

        #     # save all minutes_rec in a list
        #     minutes_list.append(minutes_rec)

    return wav_path_list


# wav_paths = load_wav(datapath)


# Find files that are empty, missing or somehow corrupted
def faulty_files(wav_path_list, npz_path_list):
    """
    Check if there are any missing or empty files in the given lists of wav and npz files.
    """
    # print status
    con.log("Checking for empty/missing/corrupted files.")

    # add progress bar
    with Progress() as progress:
        task = progress.add_task(
            "Checking for faulty files...", total=len(npz_path_list)
        )

        ## check if the wav and npz files have the same names, i.e. if npz file exists for each wav file
        for i, (wav_file, npz_file) in enumerate(zip(wav_path_list, npz_path_list)):
            # get the filenames without "peaks" and suffixes
            wav_name = wav_file.stem
            npz_name = npz_file.stem.split("_")[0]
            if wav_name != npz_name:
                con.log(f"Warning: No npz file that matches wav file {wav_file.name}!")
                # insert NaN in place of missing npz file
                npz_path_list.insert(i, None)

            ## check if existing npz file is empty
            # check for file size
            elif Path(npz_file).stat().st_size == 0:
                con.log(f"Warning: npz file {npz_file.name} is empty!")
                # insert NaN in place of empty npz file
                npz_path_list[i] = None

            # if file size isnt zero, check if npz file contains npy files (dict keys)
            else:
                try:
                    with np.load(npz_file) as data:
                        # check if npz file contains any keys
                        if len(data.files) == 0:
                            con.log(
                                f"Warning: npz file {npz_file.name} contains no keys!"
                            )
                            # insert NaN in place of empty npz file
                            npz_path_list[i] = None

                        # check if predicted_labels exists (crucial for analysis)
                        elif "predicted_labels" not in data.files:
                            con.log(
                                f"Warning: npz file {npz_file.name} does not contain predicted_labels key!"
                            )
                            # insert NaN in place of file without predicted_labels
                            npz_path_list[i] = None

                        # if keys exist, check if keys are empty
                        else:
                            for key in data.files:
                                if data[key].size == 0:
                                    con.log(
                                        f"Warning: npz file {npz_file.name} contains empty key {key}!"
                                    )
                except Exception as e:
                    con.log(f"Error loading {npz_file.name}: {e}")
                    # insert NaN in place of empty npz file
                    npz_path_list[i] = None

            # update progress bar
            progress.update(task, advance=1)

    return npz_path_list


# Sort file paths into recording sessions
def check_sessions(file_paths: list):
    """
    Check if recordings are subsequent and store seperate recording sessions in a dictionary.
    """
    ## Pair each path with its extracted datetime or None
    # initiate empty list to store tuples of (datetime, path) --- TODO: make this for loop a seperate function
    dt_path_list = []

    for path in file_paths:
        # append Nones for currupted files to preserve order
        if path is None:
            dt_path_list.append((None, None))
        # valid files
        else:
            # get time stamp from file name
            time_stamp = path.name.split("-")[1].split("_")[0]
            dt = datetime.strptime(time_stamp, "%Y%m%dT%H%M%S")
            # add path together with its timestamp to list
            dt_path_list.append((dt, path))

    ##
    # accepted time difference and tolerance between two files
    t_diff = timedelta(minutes=5)
    tolerance = timedelta(seconds=1)

    # store recording sessions in a dictionary
    rec_sessions = {}
    current_session = []

    # set last valid timestamp to None to indicate first session
    last_valid_dt = None

    # count number of subsequent Nones
    none_counter = 0

    # iterate over paths and corresponding timestamps
    for dt, path in dt_path_list:
        # skip corrupted files
        if dt is None:
            # increase counter for each subsequent None
            none_counter += 1
            continue

        # at first valid entry, start first session
        if last_valid_dt is None:
            # add first path to session
            current_session = [path]

        # all other valid entries
        else:
            # check time difference to last valid timestamp
            time_gap = dt - last_valid_dt
            # calculate how big expected gap is based on how many files were skipped
            expected_gap = t_diff * (none_counter + 1)

            # if current rec is subsequent to the last valid (non-None) rec
            if abs(time_gap - expected_gap) <= tolerance:
                # add to the same session
                current_session.append(path)

            # if current rec not subsequent
            else:
                # add last session to dict
                session_key = f"rec session {len(rec_sessions)}"
                rec_sessions[session_key] = current_session
                # start a new session
                current_session = [path]

        # update last valid timestamp
        last_valid_dt = dt
        # reset none counter
        none_counter = 0

    # save the final session
    if current_session:
        session_key = f"rec session {len(rec_sessions)}"
        rec_sessions[session_key] = current_session

    return rec_sessions


# Save dictionary that contains lists of paths per rec session in json file
def save_session_paths(session_paths):
    """
    Save the session paths to a JSON file.
    """
    # define path where json file is created
    path = Path(
        "/home/eisele/wrk/mscthesis/data/intermediate/eellogger_session_paths.json"
    )

    # Convert Path objects to strings
    session_paths_str = {k: [str(p) for p in v] for k, v in session_paths.items()}

    # print statement before saving
    con.log(f"Saving session paths to {path}.")

    # open json and save dict in it
    with path.open("w") as file:
        json.dump(session_paths_str, file)


#################################################
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
def load_wav_new(wav_list):
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


#################################################

# create arrays to store pulse counts for different time bins
min_histogram = np.zeros(24 * 60)  # minute bins for 24h period
hour_histogram = np.zeros(24)  # hourly bins for 24h period
day_histogram = np.zeros(366)  # daily bins for 1 year
month_histogram = np.zeros(12)  # monthly bins for 1 year
year_histogram = np.zeros(date.today().year - 2023)  # years since start of dataset


#################################################


# TODO: fix this (it only returns counts for 2019-01)
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


# # aggregate timestamps into 12 consecutive monthly bins (starting at earliest timestamp)
# monthly_counts, month_starts = make_monthly_histogram(timestamps, n_months=12)

#################################################


# %%
### Import Data ###

## Datapath to npz files (folder, subfolder or file)
# datapath = Path(
#     "/home/eisele/wrk/mscthesis/data/raw/eellogger_example_data_peaks/recordings2025-03-06/eellogger1-20250306T101402_peaks.npz"
# )

# # Path to folder
# datapath = Path(
#     "/home/eisele/wrk/mscthesis/data/raw/eellogger_example_data_peaks/recordings2025-03-06/"
# )
