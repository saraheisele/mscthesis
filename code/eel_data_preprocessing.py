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

The script saves the data as follows:
(TODO: specify)
"""

# %%
from rich.console import Console
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta
from audioio.audioloader import AudioLoader
import re

# Initialize console for logging
con = Console()

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

# Path to sup folder
datapath = Path("/mnt/data1/eels-mfn2021_peaks/")


# %%
### Functions ###
# Import corresponding wav file - currently this function is unnessecary bc minutes per rec is not relevant for new approach!
def load_wav(datapath):
    """
    Load the corresponding wav file for the given npz file/s and returns recording length in min.
    """
    # Print status
    con.log("Loading wav files")

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
            audio_data = AudioLoader(wavpath)

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

        # Initialize emtpy list to store number of minutes for each recording - not used atm!!
        minutes_list = []

        for i, wav in enumerate(wav_path_list):
            try:
                # Load each wav file
                audio_data = AudioLoader(wav)
            except Exception as e:
                con.log(f"Error loading: {e}")
                continue
            # Get sampling rate for this wav file
            fs = audio_data.rate

            # Get number of minutes per recording for this wav file
            samples_per_rec = audio_data.shape[0]
            minutes_rec = samples_per_rec / fs / 60

            # # Store in peak_data
            # npz_data[i]["minutes"] = minutes_rec

            # save all minutes_rec in a list
            minutes_list.append(minutes_rec)

    return minutes_rec, wav_path_list


# Load npz files
def load_peaks(datapath):
    """
    Load .npz files from the given path, directory or supdirectory.
    """
    # Print status
    con.log("Loading detected peaks")

    # Check if the path exists
    if not datapath.exists():
        raise FileNotFoundError(f"Path {datapath} does not exist.")

    # Initiaize list to store file paths
    npz_path_list = []

    # Check if the path is a directory, a file or a directory containing files
    if datapath.is_file():
        # Check if the file is an npz file
        if datapath.suffix == ".npz" and datapath.name.startswith("eellogger"):
            con.log(f"Path {datapath} is a single npz file.")
            # Store objects in list for consistency with directory case
            npz_path_list.append(datapath)
        else:
            raise FileNotFoundError(f"File {datapath} is not an npz file.")

    # Recursively find all .npz files in the directory and subdirectories
    elif datapath.is_dir():
        for file in datapath.rglob("*.npz"):
            # skip files that don't start with eellogger (wrong recording format)
            if file.name.startswith("eellogger"):
                npz_path_list.append(file)

    else:
        raise FileNotFoundError(
            f"Path {datapath} is not a file, directory containing files or directory containing folders containing files."
        )

    return sorted(npz_path_list)


def faulty_files(wav_path_list, npz_path_list):
    """
    Check if there are any missing or empty files in the given lists of wav and npz files.
    """
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
                        con.log(f"Warning: npz file {npz_file.name} contains no keys!")
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

    return npz_path_list


def check_sessions(file_paths: list):
    """
    Check if recordings are subsequent and store seperate recording sessions in a dictionary.
    """
    ## Pair each path with its extracted datetime or None
    # initiate empty list to store tuples of (datetime, path)
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


# TODO: save this dictionary to a file for later use!!
