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

# TODO: get list of al hdf files and iterate through them
# TODO: load wav files and extract sample rates
# TODO: implement to kick hdf5 file out when it detects 0 predicted pulses (log: found # h5 files, # h5 files are being processed)

# %%
from rich.console import Console
from rich.progress import Progress
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta
import json
import nixio
from thunderlab.dataloader import DataLoader
# from IPython import embed
# from audioio.audioloader import AudioLoader

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


# %%


def load_eods(file_paths):
    # Print status
    con.log("Loading hdf5 files.")

    for fp in file_paths:
        file = nixio.File.open(str(fp), nixio.FileMode.ReadWrite)
        block = file.blocks["Average pulses"]
        # data_array_names = [da.name for da in block.data_arrays]

        pulses_center_idx = block.data_arrays["cluster_centers"]
        pred_labels = block.data_arrays["predicted_labels"]

        con.log(
            f"Found total of {np.sum(pred_labels[:] == 1)} predicted pulses in file {fp}"
        )

    return pulses_center_idx, pred_labels


# %%
def get_wav(file_paths):
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
        new_path_name = "_".join(str(fp.name).split("_")[:-1])
        wavpath = Path(
            fp.parent.parent.parent.parent
            / "raw/trial_dataset_new/fielddata_trial/inpa2025_trial/inpa_enclosure"
            / new_path_name
        )

        # get the first wav file of directory and store the path to it in list
        first_file = sorted(wavpath.rglob("*.wav"))[0]
        wav_path_list.append(first_file)

    return wav_path_list


# %%
# Extracts sampling rates in Hz from the first corresponding wav files of each hdf5 file and stores them in a list
def get_fs(wav_path_list):
    fs_list = []

    # iterate through wav files
    for wav in wav_path_list:
        try:
            audio_data = DataLoader(wav)
        except Exception as e:
            con.log(f"Error loading: {e}")
            continue
        # get sampling rate for this wav file
        fs = audio_data.rate
        # add sampling rate of each wav to list
        fs_list.append(fs)

    return fs_list


# %%
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


# %% Main
# def main():
#     # Path to sup folder
#     datapath = Path("/home/eisele/wrk/mscthesis/data/intermediate/inpa2025_peaks")

#     # make list containing all paths to hdf5 files in the given datapath
#     path_list = get_path_list(datapath)

#     # load hdf5 files from path list and extract pulse centers and predicted labels
#     pulses_center_idx, pred_labels = load_eods(path_list)


#     # # faulty files function
#     # path_list_new = faulty_files(path_list)

#     # # sort file paths into recording sessions
#     # session_paths = check_sessions(path_list_new)

#     # # save session paths to json file
#     # save_session_paths(session_paths)


# if __name__ == "__main__":
#     main()

# %%
