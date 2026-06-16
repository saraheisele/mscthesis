"""Load predetected pulses from HDF5 and build multi-timescale activity histograms.

Analysis part: pulse activity preprocessing (Part 1 of Berlin activity analysis).
Dependencies: data_paths, pulse_config, h5_io; input .h5 files from deep_peak_sieve.

Generates pulse-count and pulse-rate .npz files at minute/hour/day/month/year scales,
normalized by recording effort. Run once per pulse type (all, double, wide, fat).
"""

from rich.console import Console
from pathlib import Path
import numpy as np
import nixio
import tqdm
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from data_paths import H5_DIR, activity_hist_dir
from h5_io import get_path_list
from pulse_config import PULSE_TYPES, select_pulse_type

# Initialize console for logging
con = Console()


#################################
############# LOAD ##############
#################################


def load_eods(file_paths, pulse_type="all"):
    """
    Load pulse detection data from HDF5 files with nixio.

    Extracts pulse centers, sampling rates, and recording times from .h5 files.
    Filters pulses based on model predictions and optional pulse type markers.

    Args:
        file_paths (list): List of Path objects to .h5 files
        pulse_type (str): Type of pulses to extract (from PULSE_TYPES keys)

    Returns:
        tuple: (pulse_center_list, fs_list, dt_start_list, dt_end_list, duration_list)
               - pulse_center_list: List of numpy arrays with pulse indices per file
               - fs_list: Sampling rates (Hz) per file
               - dt_start_list: Recording start times (datetime) per file
               - dt_end_list: Recording end times (datetime) per file
               - duration_list: Recording durations (seconds) per file
    """
    pulse_config = PULSE_TYPES[pulse_type]
    pulse_marker = pulse_config["array"]
    con.log(f"Loading hdf5 files. Pulse analysis type: {pulse_config['label']}")

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

        ## access data from "pulses" block
        block = file.blocks["pulses"]
        data_array_names = [da.name for da in block.data_arrays]

        # Skip files with no detected pulses (no 'centers' array means no detections)
        if "centers" not in data_array_names:
            con.log(f"File {fp} does not contain 'centers' data array. Skipping.")
            continue

        ## extract pulse data and model predictions
        pulses_center_idx = block.data_arrays["centers"]
        pred_labels = block.data_arrays["predicted_labels"]

        # Filter pulses based on model prediction and pulse type marker
        if pulse_marker is not None:
            if pulse_marker not in data_array_names:
                con.log(
                    f"File {fp} does not contain '{pulse_marker}' data array. "
                    "Using zero selected pulses for this file."
                )
                selected_centers = pulses_center_idx[:0]
            else:
                # Keep only pulses predicted positive AND marked as the selected pulse type
                marker = block.data_arrays[pulse_marker]
                mask = (pred_labels[:] == 1) & (marker[:] == 1)
                selected_centers = pulses_center_idx[mask]
        else:
            # For "all" pulses: keep only those predicted as positive
            selected_centers = pulses_center_idx[pred_labels[:] == 1]

        ## extract metadata from .h5 file
        section = file.sections["pulses_metadata"]
        fs = section["metadata"]["samplerate"]
        starttime_str = section["metadata"]["metadata"]["INFO"]["DateTimeOriginal"]
        duration = section["metadata"]["duration"]

        # Convert start time string to datetime and calculate end time
        dt_start = datetime.strptime(starttime_str, "%Y-%m-%dT%H:%M:%S")
        dt_end = dt_start + timedelta(seconds=duration)

        ## append to output lists
        pulse_center_list.append(selected_centers)
        fs_list.append(fs)
        dt_start_list.append(dt_start)
        dt_end_list.append(dt_end)
        duration_list.append(duration)

    return pulse_center_list, fs_list, dt_start_list, dt_end_list, duration_list


###########################################
############# DATA PROCESSING #############
###########################################
# TODO: maybe change make_histogram approach to session_fr approach (session wise and then just add up al the session lists to get total pulse counts)


def month_start(dt):
    """Return datetime object with time set to start of the month."""
    return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def month_index(dt, first_month):
    """
    Calculate the month index relative to a reference month.

    Args:
        dt (datetime): Target datetime
        first_month (datetime): Reference month

    Returns:
        int: Number of months between first_month and dt
    """
    return (dt.year - first_month.year) * 12 + (dt.month - first_month.month)


def histogram_time_bounds(start_times, end_times):
    """
    Calculate histogram bin counts and temporal bounds for all timescales.

    Args:
        start_times (list): Recording start times (datetime objects)
        end_times (list): Recording end times (datetime objects)

    Returns:
        dict: Contains first_month, first_year, month_count (bins), year_count (bins)
    """
    first_start = min(start_times)
    last_end = max(end_times)
    first_month = month_start(first_start)
    last_month = month_start(last_end)
    first_year = first_start.year

    return {
        "first_month": first_month,
        "first_year": first_year,
        "month_count": month_index(last_month, first_month) + 1,
        "year_count": last_end.year - first_year + 1,
    }


def make_histogram(pulse_centers, sampling_rates, start_times, end_times):
    """
    Generate activity histograms at multiple timescales.

    Creates histograms binning pulse counts at different temporal resolutions
    (minute, hour, day, month, year). Also converts pulse indices to Unix timestamps.

    Args:
        pulse_centers (list): List of pulse index arrays (one per recording file)
        sampling_rates (list): Sampling rates (Hz) for each file
        start_times (list): Recording start times (datetime) for each file
        end_times (list): Recording end times (datetime) for each file

    Returns:
        tuple: (hist, rec_hist, timestamp_list, session_counts)
               - hist: Dict of histograms (pulse counts per bin at each timescale)
               - rec_hist: Dict of counts (number of recordings contributing to each bin)
               - timestamp_list: List of Unix timestamps for all pulses
               - session_counts: Per-session histograms
    """
    # create list to store unix timestamps of each pulse for later storage in hdf5 file
    timestamp_list = []  # TODO: make seperate function to store timestamps??
    time_bounds = histogram_time_bounds(start_times, end_times)

    # create histograms in a dict
    hist_sizes = {
        "minute": 24 * 60,
        "hour": 24,
        "day": 366,
        "month": 12,
        "month_since_start": time_bounds["month_count"],
        "year": time_bounds["year_count"],
    }

    # histogram dict to hold pulse counts over all rec sessions for each time scale
    hist = {k: np.zeros(v, dtype=int) for k, v in hist_sizes.items()}

    # list to hold all dicts per rec session
    session_counts = []

    # create dict to hold recording time (seconds) per bin for normalization
    # and a separate dict to keep the original "number of recordings contributing" count
    rec_hist = {k: np.zeros_like(v, dtype=int) for k, v in hist.items()}

    con.log("Calculating count histograms...")

    # iterate through each of the lists in pulse_centers (one per hdf5 file)
    for i, rec in enumerate(tqdm.tqdm(pulse_centers, desc="Processing pulse centers")):
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
            month_since_start = month_index(pulse_time_abs, time_bounds["first_month"])
            year = pulse_time_abs.year - time_bounds["first_year"]

            # increment histogram bins for each timescale
            hist_i["minute"][minute] += 1
            hist_i["hour"][hour] += 1
            hist_i["day"][day] += 1
            hist_i["month"][month] += 1
            hist_i["month_since_start"][month_since_start] += 1
            hist_i["year"][year] += 1

            # store Unix timestamp for this pulse
            pulse_time_abs_unix = pulse_time_abs.timestamp()
            timestamp_list.append(pulse_time_abs_unix)

        ## accumulate global counters and track which recordings contributed to each bin
        for item in hist:
            hist[item] += hist_i[item]
            rec_hist[item][hist_i[item] > 0] += 1
        session_counts.append(hist_i)

    con.log("Finished calculating histogram.")
    return hist, rec_hist, timestamp_list, session_counts


def rec_time_per_bin(start_times, end_times):
    """
    Calculate total recording time per histogram bin.

    Handles partial bin coverage when recordings span multiple time bins.
    Tracks per-session recording times for later normalization.

    Args:
        start_times (list): Recording start times (datetime objects)
        end_times (list): Recording end times (datetime objects)

    Returns:
        tuple: (rec_time_hist, session_rec_times)
               - rec_time_hist: Dict of recording seconds per bin at each timescale
               - session_rec_times: Per-session recording time histograms
    """
    con.log("Calculating recording time histograms...")
    time_bounds = histogram_time_bounds(start_times, end_times)

    # create dict to hold rec times per bin
    hist_sizes = {
        "minute": 24 * 60,
        "hour": 24,
        "day": 366,
        "month": 12,
        "month_since_start": time_bounds["month_count"],
        "year": time_bounds["year_count"],
    }

    rec_time_hist = {k: np.zeros(v, dtype=float) for k, v in hist_sizes.items()}

    # list to hold all dicts per rec session
    session_rec_times = []

    for i, st in enumerate(tqdm.tqdm(start_times, desc="Processing recording times")):
        # build a per-session rec time dict to avoid mutating the global one
        rec_time_i = {
            k: np.zeros_like(v, dtype=float) for k, v in rec_time_hist.items()
        }

        cursor = st
        while cursor < end_times[i]:
            next_minute = cursor.replace(second=0, microsecond=0) + relativedelta(
                minutes=1
            )
            segment_end = min(next_minute, end_times[i])
            segment_seconds = (segment_end - cursor).total_seconds()

            rec_time_i["minute"][cursor.hour * 60 + cursor.minute] += segment_seconds
            rec_time_i["hour"][cursor.hour] += segment_seconds
            rec_time_i["day"][cursor.timetuple().tm_yday - 1] += segment_seconds
            rec_time_i["month"][cursor.month - 1] += segment_seconds
            rec_time_i["month_since_start"][
                month_index(cursor, time_bounds["first_month"])
            ] += segment_seconds
            rec_time_i["year"][cursor.year - time_bounds["first_year"]] += (
                segment_seconds
            )

            cursor = segment_end

        # add the per-session durations to the global histogram
        for item in rec_time_hist:
            rec_time_hist[item] += rec_time_i[item]

        # append the per-session copy to the list
        session_rec_times.append(rec_time_i)
    return rec_time_hist, session_rec_times


def pulse_rate_hz(count_hist, rec_time_hist):
    """
    Calculate pulse rates by normalizing pulse counts by recording time.

    Handles bins with no recording time (sets rate to NaN).

    Args:
        count_hist (dict): Pulse counts per bin at each timescale
        rec_time_hist (dict): Recording time per bin at each timescale

    Returns:
        dict: Pulse rates (Hz) per bin at each timescale
    """
    pulse_rates = {}
    for k in count_hist:
        counts = np.asarray(count_hist[k], dtype=float)
        rec_times = np.asarray(rec_time_hist[k], dtype=float)
        rate = np.full_like(counts, np.nan, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            np.divide(counts, rec_times, out=rate, where=rec_times != 0)
        pulse_rates[k] = rate
    return pulse_rates


def session_pulse_rate_hz(session_counts, session_rec_times):
    """
    Calculate per-session pulse rates.

    Args:
        session_counts (list): Per-session pulse count histograms
        session_rec_times (list): Per-session recording time histograms

    Returns:
        list: Per-session pulse rate histograms (one dict per session)
    """
    # list to store pulse rate histograms for each rec session
    session_rates = []
    for count_hist, rec_time_hist in zip(session_counts, session_rec_times):
        session_rates.append(pulse_rate_hz(count_hist, rec_time_hist))
    return session_rates


#################################
############# SAVE ##############
#################################


def save_histograms(count_hist, rec_hist, rec_time_hist, output_path: Path):
    """
    Save histogram dictionaries to compressed .npz files.

    Args:
        count_hist (dict): Pulse count histograms
        rec_hist (dict): Recording count histograms
        rec_time_hist (dict): Recording time histograms
        output_path (Path): Output directory or file path
    """
    con.log(f"Saving dictionaries to {output_path}.")
    # ensure values are numpy arrays
    clean_count_hist = {k: np.asarray(v) for k, v in count_hist.items()}
    clean_rec_hist = {k: np.asarray(v) for k, v in rec_hist.items()}
    clean_rec_time = {k: np.asarray(v) for k, v in rec_time_hist.items()}
    # Save files inside the output directory (create parent if necessary)
    parent_dir = output_path if output_path.is_dir() else output_path.parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    filename_count = "berlin_dummypulses_count_hist_dict.npz"
    filename_rec = "berlin_dummypulses_rec_hist_dict.npz"
    filename_time = "berlin_dummypulses_rec_time_hist_dict.npz"

    save_count = parent_dir / filename_count
    save_rec = parent_dir / filename_rec
    save_time = parent_dir / filename_time

    np.savez_compressed(save_count, **clean_count_hist)
    np.savez_compressed(save_rec, **clean_rec_hist)
    np.savez_compressed(save_time, **clean_rec_time)


def save_pulse_rate_histograms(pulse_rate_hist, output_path: Path):
    """
    Save pulse rate histograms to compressed .npz file.

    Args:
        pulse_rate_hist (dict): Pulse rates (Hz) per bin at each timescale
        output_path (Path): Output directory or file path
    """
    con.log(f"Saving pulse rate dictionaries to {output_path}.")
    clean_pulse_rates = {k: np.asarray(v) for k, v in pulse_rate_hist.items()}
    parent_dir = output_path if output_path.is_dir() else output_path.parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        parent_dir / "berlin_dummypulses_pulse_rate_hz_hist_dict.npz",
        **clean_pulse_rates,
    )


def save_histogram_metadata(start_times, end_times, output_path: Path):
    """
    Save histogram metadata needed for time axis formatting in plots.

    Args:
        start_times (list): Recording start times (datetime objects)
        end_times (list): Recording end times (datetime objects)
        output_path (Path): Output directory or file path
    """
    time_bounds = histogram_time_bounds(start_times, end_times)
    parent_dir = output_path if output_path.is_dir() else output_path.parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        parent_dir / "berlin_dummypulses_hist_metadata.npz",
        first_month_year=time_bounds["first_month"].year,
        first_month_month=time_bounds["first_month"].month,
        first_year=time_bounds["first_year"],
    )


def save_session_pulse_rate_hz(session_pulse_rate_hz_list, out_path):
    """
    Save per-session pulse rates to compressed .npz file.

    Stacks per-session rate arrays by timescale into 2D arrays (sessions × bins).

    Args:
        session_pulse_rate_hz_list (list): Per-session pulse rate dicts
        out_path (Path): Output directory or file path
    """
    timescales = list(session_pulse_rate_hz_list[0].keys())
    arrs = {
        k: np.vstack([sess[k] for sess in session_pulse_rate_hz_list])
        for k in timescales
    }
    # Save the file inside the out_path directory, not in the parent directory
    filename = "berlin_dummypulses_session_pulse_rate_hz.npz"
    save_file = (
        out_path / filename if out_path.is_dir() else out_path.with_name(filename)
    )
    np.savez_compressed(
        save_file,
        **arrs,
    )


#################################
############# MAIN ##############
#################################


def main():
    """
    Main workflow: Load pulse data, generate histograms, and save results.

    Prompts user for pulse type, loads all .h5 files, extracts pulses,
    generates multi-timescale activity histograms and pulse rates, then saves
    all outputs to .npz files.
    """
    pulse_type = select_pulse_type(default="all")

    # path to directory containing hdf5 files with detected pulses
    data_path = H5_DIR

    # path to output directory - adjust based on selected pulse type
    hist_subdir = PULSE_TYPES[pulse_type]["hist_subdir"]

    save_path = activity_hist_dir(hist_subdir)
    # Ensure the output directory exists
    save_path.mkdir(parents=True, exist_ok=True)

    # make list containing all paths to hdf5 files in the given datapath
    path_list = get_path_list(data_path)

    # load hdf5 files from path list and extract pulse centers of pulses that were predicted as EODs
    pulse_centers, sampling_rates, start_times, end_times, duration = load_eods(
        path_list, pulse_type=pulse_type
    )

    # calculate histogram of number of pulses per minute for 24‑h period (0…1439 minutes)
    count_histogram_dict, rec_count_hist_dict, timestamps, pulse_count_per_session = (
        make_histogram(pulse_centers, sampling_rates, start_times, end_times)
    )

    # calculate the recording time for each bin for all timescales
    rec_time_hist_dict, rec_time_per_session = rec_time_per_bin(start_times, end_times)

    # calculate pulse rates as pulse count / recording time in each bin
    pulse_rate_hist_dict = pulse_rate_hz(count_histogram_dict, rec_time_hist_dict)
    session_pulse_rate_hz_list = session_pulse_rate_hz(
        pulse_count_per_session, rec_time_per_session
    )

    save_histograms(
        count_histogram_dict,
        rec_count_hist_dict,
        rec_time_hist_dict,
        Path(save_path),
    )
    save_pulse_rate_histograms(pulse_rate_hist_dict, Path(save_path))
    save_histogram_metadata(start_times, end_times, Path(save_path))

    save_session_pulse_rate_hz(session_pulse_rate_hz_list, Path(save_path))


if __name__ == "__main__":
    main()
