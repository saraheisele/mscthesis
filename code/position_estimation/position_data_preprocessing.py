"""Extract eel head positions from predetected .h5 files and build spatial histograms.

Analysis part: position preprocessing (Part 5a of Berlin activity analysis).
Dependencies: data_paths, h5_io, eel_data_preprocessing, position_utils.

Mirrors the activity histogram pipeline but bins pulse head positions along the
16-electrode line at multiple timescales.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from path_setup import setup_script_paths

setup_script_paths(__file__, "activity_timescales")

import numpy as np
import nixio
import tqdm
from rich.console import Console

from data_paths import H5_DIR, position_hist_dir
from eel_data_preprocessing import (
    histogram_time_bounds,
    month_index,
    rec_time_per_bin,
    save_histogram_metadata,
)
from h5_io import get_path_list, get_pulse_block, open_h5
from position_utils import (
    DEFAULT_BRIGHT_DARK_BOUNDARY_M,
    N_ELECTRODES,
    default_electrode_positions_m,
    head_position_from_pulse,
    tank_zone,
)

con = Console()

POSITION_METHODS = ("peak_positive", "weighted_mean")
POSITION_BIN_EDGES_M = np.linspace(0.0, 3.75, N_ELECTRODES + 1)


def load_positions(file_paths, method: str = "peak_positive"):
    """Load per-pulse positions and timing metadata from h5 files."""
    electrode_positions_m = default_electrode_positions_m()
    position_lists = []
    time_sec_lists = []
    fs_list = []
    start_times = []
    end_times = []

    for fp in file_paths:
        handle = open_h5(fp, "r")
        if handle is None:
            continue

        try:
            block = get_pulse_block(handle)
            data_array_names = [da.name for da in block.data_arrays]
            if "centers" not in data_array_names:
                con.log(f"Skipping {fp.name}: no centers array.")
                continue

            centers = block.data_arrays["centers"][:]
            labels = block.data_arrays["predicted_labels"][:]
            raw_pulses = block.data_arrays["raw_pulses"][:]
            mask = labels == 1

            section = handle.sections["pulses_metadata"]
            fs = float(section["metadata"]["samplerate"])
            starttime_str = section["metadata"]["metadata"]["INFO"]["DateTimeOriginal"]
            duration = float(section["metadata"]["duration"])
        finally:
            handle.close()

        from datetime import datetime

        dt_start = datetime.strptime(starttime_str, "%Y-%m-%dT%H:%M:%S")
        dt_end = dt_start + timedelta(seconds=duration)

        positions = []
        pulse_times_sec = []
        for center_idx, pulse in zip(centers[mask], raw_pulses[mask]):
            head_m, _ = head_position_from_pulse(
                pulse, electrode_positions_m, method=method
            )
            positions.append(head_m)
            pulse_times_sec.append(float(center_idx) / fs)

        position_lists.append(np.asarray(positions, dtype=float))
        time_sec_lists.append(np.asarray(pulse_times_sec, dtype=float))
        fs_list.append(fs)
        start_times.append(dt_start)
        end_times.append(dt_end)

    return position_lists, time_sec_lists, fs_list, start_times, end_times


def make_position_histograms(
    position_lists,
    time_sec_lists,
    start_times,
    end_times,
    boundary_m: float = DEFAULT_BRIGHT_DARK_BOUNDARY_M,
):
    """Build 1D position summaries and 2D occupancy histograms."""
    time_bounds = histogram_time_bounds(start_times, end_times)
    hist_sizes = {
        "minute": 24 * 60,
        "hour": 24,
        "day": 366,
        "month": 12,
        "month_since_start": time_bounds["month_count"],
        "year": time_bounds["year_count"],
    }

    position_sum = {k: np.zeros(v, dtype=float) for k, v in hist_sizes.items()}
    position_count = {k: np.zeros(v, dtype=int) for k, v in hist_sizes.items()}
    bright_count = {k: np.zeros(v, dtype=int) for k, v in hist_sizes.items()}
    dark_count = {k: np.zeros(v, dtype=int) for k, v in hist_sizes.items()}

    occ_timescales = ("minute", "hour")
    occurrence = {
        ts: np.zeros((hist_sizes[ts], N_ELECTRODES), dtype=int) for ts in occ_timescales
    }

    session_position_sum = []
    session_position_count = []

    for positions, pulse_times_sec, dt_start in tqdm.tqdm(
        zip(position_lists, time_sec_lists, start_times),
        desc="Position histograms",
        total=len(position_lists),
    ):
        session_sum = {k: np.zeros(v, dtype=float) for k, v in hist_sizes.items()}
        session_count = {k: np.zeros(v, dtype=int) for k, v in hist_sizes.items()}

        for pos_m, pulse_time_sec in zip(positions, pulse_times_sec):
            pulse_time_abs = dt_start + timedelta(seconds=float(pulse_time_sec))
            minute = pulse_time_abs.hour * 60 + pulse_time_abs.minute
            hour = pulse_time_abs.hour
            day = pulse_time_abs.timetuple().tm_yday - 1
            month = pulse_time_abs.month - 1
            month_since_start = month_index(pulse_time_abs, time_bounds["first_month"])
            year = pulse_time_abs.year - time_bounds["first_year"]

            bins = {
                "minute": minute,
                "hour": hour,
                "day": day,
                "month": month,
                "month_since_start": month_since_start,
                "year": year,
            }

            for key, bin_idx in bins.items():
                position_sum[key][bin_idx] += pos_m
                position_count[key][bin_idx] += 1
                session_sum[key][bin_idx] += pos_m
                session_count[key][bin_idx] += 1
                if tank_zone(pos_m, boundary_m) == "bright":
                    bright_count[key][bin_idx] += 1
                else:
                    dark_count[key][bin_idx] += 1

            pos_bin = int(
                np.clip(np.digitize(pos_m, POSITION_BIN_EDGES_M) - 1, 0, N_ELECTRODES - 1)
            )
            occurrence["minute"][minute, pos_bin] += 1
            occurrence["hour"][hour, pos_bin] += 1

        session_position_sum.append(session_sum)
        session_position_count.append(session_count)

    return {
        "position_sum": position_sum,
        "position_count": position_count,
        "bright_count": bright_count,
        "dark_count": dark_count,
        "occurrence": occurrence,
        "session_position_sum": session_position_sum,
        "session_position_count": session_position_count,
    }


def mean_position_hist(position_sum, position_count):
    """Compute mean position per bin, NaN where no pulses."""
    mean_hist = {}
    for key in position_sum:
        sums = np.asarray(position_sum[key], dtype=float)
        counts = np.asarray(position_count[key], dtype=float)
        mean = np.full_like(sums, np.nan, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            np.divide(sums, counts, out=mean, where=counts > 0)
        mean_hist[key] = mean
    return mean_hist


def session_mean_position(session_sums, session_counts):
    """Per-session mean position histograms."""
    session_means = []
    for sum_hist, count_hist in zip(session_sums, session_counts):
        session_means.append(mean_position_hist(sum_hist, count_hist))
    return session_means


def save_position_histograms(results, output_path: Path, method: str):
    """Save position histogram dictionaries to compressed npz files."""
    output_path.mkdir(parents=True, exist_ok=True)
    prefix = f"berlin_position_{method}"

    np.savez_compressed(
        output_path / f"{prefix}_sum_hist_dict.npz",
        **{k: np.asarray(v) for k, v in results["position_sum"].items()},
    )
    np.savez_compressed(
        output_path / f"{prefix}_count_hist_dict.npz",
        **{k: np.asarray(v) for k, v in results["position_count"].items()},
    )
    np.savez_compressed(
        output_path / f"{prefix}_bright_count_hist_dict.npz",
        **{k: np.asarray(v) for k, v in results["bright_count"].items()},
    )
    np.savez_compressed(
        output_path / f"{prefix}_dark_count_hist_dict.npz",
        **{k: np.asarray(v) for k, v in results["dark_count"].items()},
    )
    np.savez_compressed(
        output_path / f"{prefix}_occurrence_hist.npz",
        **{k: np.asarray(v) for k, v in results["occurrence"].items()},
    )

    session_means = session_mean_position(
        results["session_position_sum"], results["session_position_count"]
    )
    timescales = list(session_means[0].keys())
    session_arrays = {
        ts: np.vstack([session[ts] for session in session_means]) for ts in timescales
    }
    np.savez_compressed(
        output_path / f"{prefix}_session_mean_position.npz",
        **session_arrays,
    )

    mean_hist = mean_position_hist(results["position_sum"], results["position_count"])
    np.savez_compressed(
        output_path / f"{prefix}_mean_position_hist_dict.npz",
        **{k: np.asarray(v) for k, v in mean_hist.items()},
    )


def select_position_method(default: str = "peak_positive") -> str:
    choices = ", ".join(POSITION_METHODS)
    selected = input(f"Position method ({choices}) [{default}]: ").strip()
    if not selected:
        return default
    if selected not in POSITION_METHODS:
        raise ValueError(f"Unknown position method '{selected}'. Choose: {choices}.")
    return selected


def main():
    method = select_position_method(default="peak_positive")
    save_path = position_hist_dir(method)
    save_path.mkdir(parents=True, exist_ok=True)

    path_list = get_path_list(H5_DIR)
    position_lists, time_sec_lists, _fs_list, start_times, end_times = load_positions(
        path_list, method=method
    )

    results = make_position_histograms(
        position_lists, time_sec_lists, start_times, end_times
    )
    rec_time_hist_dict, _session_rec_times = rec_time_per_bin(start_times, end_times)
    np.savez_compressed(
        save_path / f"berlin_position_{method}_rec_time_hist_dict.npz",
        **{k: np.asarray(v) for k, v in rec_time_hist_dict.items()},
    )

    save_position_histograms(results, save_path, method=method)
    save_histogram_metadata(start_times, end_times, save_path)
    con.log(f"Saved position histograms to {save_path}")


if __name__ == "__main__":
    main()
