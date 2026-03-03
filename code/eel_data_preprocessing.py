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

# TODO: implement to kick hdf5 file out when it detects 0 predicted pulses (log: found # h5 files, # h5 files are being processed)
# TODO: Maybe store fs, rec length and start time in npz file/dictionary
# %%
from rich.console import Console
from pathlib import Path
import numpy as np
import nixio
from thunderlab.dataloader import DataLoader
from datetime import datetime
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


# %%
# Extracts sampling rates in Hz from the first corresponding wav files of each hdf5 file and stores them in a list
def load_wav(wav_path_list):
    fs_list = []
    dt_list = []
    length_list = []

    # iterate through wav files
    for wav in wav_path_list:
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
        filename = wav.name

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

    # %% Main
    # def main():
    # Path to sup folder
    datapath = Path("/home/eisele/wrk/mscthesis/data/intermediate/inpa2025_peaks")

    # make list containing all paths to hdf5 files in the given datapath
    path_list = get_path_list(datapath)

    # load hdf5 files from path list and extract pulse centers and predicted labels
    pulse_center_idc, pred_labels = load_eods(path_list)

    # make list of the first wav file of each recording session corresponding to the hdf5 files in path_list
    wav_list = find_wav(path_list)

    # get sampling rates from the first wav file of each recording session
    sampling_rates, start_times, rec_lengths = load_wav(wav_list)


# if __name__ == "__main__":
#     main()

# %%
