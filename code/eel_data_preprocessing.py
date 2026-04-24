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
from dateutil.relativedelta import relativedelta
from IPython import embed

# Initialize console for logging
con = Console()


#################################
############# LOAD ##############
#################################


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
    con.log("Loading hdf5 files.")

    ## initialize empty lists to hold extracted data
    pulse_center_list = []
    fs_list = []
    dt_start_list = []
    dt_end_list = []
    duration_list = []

    # iterate through file paths, load hdf5 files
    for fp in file_paths:
        ## load hdf5 file with nixio
        file = nixio.File.open(str(fp), nixio.FileMode.ReadWrite)

        ## access data
        # pulses block contains the detected pulses and their metadata
        block = file.blocks["pulses"]
        # get the names of the data arrays in the block
        data_array_names = [da.name for da in block.data_arrays]

        # check if centers data array exists, if no pulses were detected, its not created and we can skip the file
        if "centers" not in data_array_names:
            con.log(f"File {fp} does not contain 'centers' data array. Skipping.")
            continue

        ## extract relevant data from h5 file
        # center index of each detected pulse in the original data
        pulses_center_idx = block.data_arrays["centers"]
        # predicted labels for each detected pulse, 1 = pulse, 0 = no pulse
        pred_labels = block.data_arrays["predicted_labels"]

        ## access metadata
        section = file.sections["pulses_metadata"]

        ## extract relevant metadata
        # sampling rate in Hz
        fs = section["metadata"]["samplerate"]
        # start time of recording session as string
        starttime_str = section["metadata"]["metadata"]["INFO"]["DateTimeOriginal"]
        # recording length in seconds
        duration = section["metadata"]["duration"]

        # convert start time string to datetime object
        dt_start = datetime.strptime(starttime_str, "%Y-%m-%dT%H:%M:%S")

        # add recording length to start time to get end time of recording session
        dt_end = dt_start + timedelta(seconds=duration)

        ## append lists
        pulse_center_list.append(pulses_center_idx[pred_labels[:] == 1])
        fs_list.append(fs)
        dt_start_list.append(dt_start)
        dt_end_list.append(dt_end)
        duration_list.append(duration)

    return pulse_center_list, fs_list, dt_start_list, dt_end_list, duration


###########################################
############# DATA PROCESSING #############
###########################################


# calculate number of pulses per time bin
def make_histogram(pulse_centers, sampling_rates, start_times):
    # create list to store unix timestamps of each pulse for later storage in hdf5 file
    timestamp_list = []  # TODO: make seperate function to store timestamps??

    # create histograms in a dict
    hist_sizes = {
        "minute": 24 * 60,
        "hour": 24,
        "day": 366,
        "month": 12,
        "year": (date.today().year - 2023) + 1,
    }
    # TODO: dont hardcode start year, but get it from start_times list - make extra function for this

    hist = {k: np.zeros(v, dtype=int) for k, v in hist_sizes.items()}

    # create dict to hold recording time (seconds) per bin for normalization
    # and a separate dict to keep the original "number of recordings contributing" count
    rec_hist = {k: np.zeros_like(v, dtype=int) for k, v in hist.items()}

    con.log("Calculating count histograms...")

    # iterate through each of the lists in pulse_centers (one per hdf5 file)
    for i, rec in enumerate(pulse_centers):
        # initiate new preliminary dict for each h5 file/pulse list/i
        # to store pulse counts so they can also be used to increment the recording counter
        hist_i = {k: np.zeros_like(v) for k, v in hist.items()}

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
            hist_i["minute"][minute] += 1
            hist_i["hour"][hour] += 1
            hist_i["day"][day] += 1
            hist_i["month"][month] += 1
            hist_i["year"][year] += 1

            ## convert pulse to Unix timestamp and append to list for later storage in hdf5 file
            pulse_time_abs_unix = pulse_time_abs.timestamp()  # float64
            timestamp_list.append(pulse_time_abs_unix)

        ## append global counters (counts of pulses and whether this recording contributed any pulses)
        for item in hist:
            hist[item] += hist_i[item]
            rec_hist[item][hist_i[item] > 0] += 1
            # TODO: do i even need rec count hist when i have rec time hist?

    con.log("Finished calculating histogram.")
    embed()
    return hist, rec_hist, timestamp_list


def rec_time_per_bin(start_times, end_times):
    con.log("Calculating recording time histograms...")

    # create dict to hold rec times per bin
    hist_sizes = {
        "minute": 24 * 60,
        "hour": 24,
        "day": 366,
        "month": 12,
        "year": (date.today().year - 2023) + 1,
    }
    # TODO: dont hardcode start year, but get it from start_times list - make extra function for this

    rec_time_hist = {k: np.zeros(v, dtype=float) for k, v in hist_sizes.items()}

    # iterate through recording sessions bzw. the respective  start times
    for i, st in enumerate(start_times):
        ## find starting bin for all timescales
        start_idx = {
            "minute": st.hour * 60 + st.minute,
            "hour": st.hour,
            "day": st.timetuple().tm_yday - 1,  # day of year (0‑365)
            "month": st.month - 1,  # month of year (0‑11)
            "year": st.year - 2023,  # year since start of recordings
        }

        ## find end bin for all timescales
        end_idx = {
            "minute": end_times[i].hour * 60 + end_times[i].minute,
            "hour": end_times[i].hour,
            "day": end_times[i].timetuple().tm_yday - 1,  # day of year (0‑365)
            "month": end_times[i].month - 1,  # month of year (0‑11)
            "year": end_times[i].year - 2023,  # year since start of recordings
        }

        ## find the start of the next bin for all timescales
        # using dateutil.relativdelta bc. it also does variable units like months and years unlike timedelta
        next_bins = {
            "minute": st.replace(second=0, microsecond=0) + relativedelta(minutes=1),
            "hour": st.replace(minute=0, second=0, microsecond=0)
            + relativedelta(hours=1),
            "day": st.replace(hour=0, minute=0, second=0, microsecond=0)
            + relativedelta(days=1),
            "month": st.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            + relativedelta(months=1),
            "year": st.replace(
                month=1, day=1, hour=0, minute=0, second=0, microsecond=0
            )
            + relativedelta(years=1),
        }

        # calculate time deltas to next bin for each scale in new dict
        start_dt = {k: (v - st).total_seconds() for k, v in next_bins.items()}

        ## check where rec session ends - start of the last bin for each timescale
        last_bins = {
            "minute": end_times[i].replace(second=0, microsecond=0),
            "hour": end_times[i].replace(minute=0, second=0, microsecond=0),
            "day": end_times[i].replace(hour=0, minute=0, second=0, microsecond=0),
            "month": end_times[i].replace(
                day=1, hour=0, minute=0, second=0, microsecond=0
            ),
            "year": end_times[i].replace(
                month=1, day=1, hour=0, minute=0, second=0, microsecond=0
            ),
        }

        # last bin timedelta to end of rec session (per-scale dict)
        end_dt = {k: (end_times[i] - v).total_seconds() for k, v in last_bins.items()}

        # prepare per-bin duration arrays for full bins (exclusive start/end)
        fixed_dur = {"minute": 60, "hour": 60 * 60, "day": 24 * 60 * 60}

        ## append rec time in sec to respective bins in rec_time_hist
        for item in rec_time_hist:
            # add partial seconds for the start and end bins
            rec_time_hist[item][start_idx[item]] += start_dt[item]
            rec_time_hist[item][end_idx[item]] += end_dt[item]

            # add full-bin durations for bins strictly between start and end
            if item in fixed_dur:
                dur = fixed_dur[item]
                # constant-duration full bins
                if start_idx[item] + 1 <= end_idx[item] - 1:
                    rec_time_hist[item][start_idx[item] + 1 : end_idx[item]] += dur
                else:
                    # wrap-around
                    if start_idx[item] + 1 < rec_time_hist[item].shape[0]:
                        rec_time_hist[item][start_idx[item] + 1 :] += dur
                    if end_idx[item] > 0:
                        rec_time_hist[item][: end_idx[item]] += dur
            else:  # no rec session spans more than one full month or year bin, so no need to add full-bin durations
                continue

    embed()
    return rec_time_hist


#################################
############# SAVE ##############
#################################


# create a hdf5 file with nixio to later save the timestamp of each pulse in it
def open_nix_for_output(output_path: Path):
    nix_file = nixio.File.open(
        str(
            output_path.with_name(
                output_path.stem + "berlin_dummypulses_timestamps.nix"
            )
        ),
        nixio.FileMode.Overwrite,
    )
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

    # TODO: do this in chunks
    # for time in time_stamp_list:

    time_stamp_block.data_arrays["timestamps"].append(time_stamp_list)

    return True


def save_histograms(count_hist, rec_hist, rec_time_hist, output_path: Path):
    con.log(f"Saving dictionaries to {output_path}.")
    # ensure values are numpy arrays
    clean_count_hist = {k: np.asarray(v) for k, v in count_hist.items()}
    clean_rec_hist = {k: np.asarray(v) for k, v in rec_hist.items()}
    clean_rec_time = {k: np.asarray(v) for k, v in rec_time_hist.items()}

    # save to seperate npz files
    np.savez_compressed(
        output_path.with_name(
            output_path.stem + "berlin_dummypulses_count_hist_dict.npz"
        ),
        **clean_count_hist,
    )
    np.savez_compressed(
        output_path.with_name(
            output_path.stem + "berlin_dummypulses_rec_hist_dict.npz"
        ),
        **clean_rec_hist,
    )
    np.savez_compressed(
        output_path.with_name(
            output_path.stem + "berlin_dummypulses_rec_time_hist_dict.npz"
        ),
        **clean_rec_time,
    )


# TODO: should I save the histogram dict in nix file as well?


#################################
############# MAIN ##############
#################################


# %%
def main():
    # path to directory containing hdf5 files with detected pulses
    data_path = Path(
        "/home/eisele/wrk/mscthesis/data/newdata/eels-mfn2021_dummy_pulses_redetected/berlin_tank_site"
    )

    # path to output directory
    save_path = Path("/home/eisele/wrk/mscthesis/data/newdata")

    # make list containing all paths to hdf5 files in the given datapath
    path_list = get_path_list(data_path)

    # load hdf5 files from path list and extract pulse centers of pulses that were predicted as EODs
    pulse_centers, sampling_rates, start_times, end_times, duration = load_eods(
        path_list
    )

    # calculate histogram of number of pulses per minute for 24‑h period (0…1439 minutes)
    count_histogram_dict, rec_count_hist_dict, timestamps = make_histogram(
        pulse_centers, sampling_rates, start_times
    )

    # calculate the recording time for each bin for all timescales
    rec_time_hist_dict = rec_time_per_bin(start_times, end_times)

    # # create a hdf5 file with nixio to later save the timestamp of each pulse in it
    # nix_file, nix_block = open_nix_for_output(Path(save_path))

    # # save the unix timestamp of each pulse in the earlier created nix_timestamp block of nix_file
    # created = append_cluster_block(nix_block, timestamps, created=False)  # noqa: F841

    # save histogram dictionaries to .npz file for later use in plotting
    save_histograms(
        count_histogram_dict,
        rec_count_hist_dict,
        rec_time_hist_dict,
        Path(save_path),
    )


if __name__ == "__main__":
    main()


# TODO: implement progress bar!
# %%
