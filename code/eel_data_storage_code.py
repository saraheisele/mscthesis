"""This script contains functions from my eel_data processing pipeline that I dont need anymore/right now."""

from rich.console import Console
from pathlib import Path
import re


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
