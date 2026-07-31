"""Shared H5 collector for pulse half-width and temporal property analyses.

Analysis part: special-pulse metrics collection (shared by half-width
distributions and pulse-properties-over-time plots).
Dependencies: data_paths, h5_io, double_peaks_detection, prototype_pulse_plots,
pulse_shape_metrics.

Walks predetected pulse .h5 files once, labels each predicted-positive pulse as
normal / double / wide, and returns a DataFrame of timestamps and shape metrics
used by both distribution and temporal trend scripts.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import nixio
import numpy as np
import pandas as pd

from h5_io import get_path_list, get_pulse_block, load_marker_array, open_h5
from special_pulses.double_peaks_detection import (
    compute_half_max_width,
    expand_marker_to_all_pulses,
)
from special_pulses.prototype_pulse_plots import (
    PULSE_SHAPES,
    get_biggest_unclipped_waveform,
)
from special_pulses.pulse_shape_metrics import baseline_correct, double_pulse_metrics


def _full_marker(file_path, block, array_name, candidates, num_pulses):
    raw = load_marker_array(file_path, array_name, block)
    if raw is None:
        return np.zeros(num_pulses, dtype=np.int64)
    return expand_marker_to_all_pulses(raw, candidates, num_pulses, array_name)


def collect_pulse_property_records(data_path) -> pd.DataFrame:
    rows = []
    for file_path in get_path_list(Path(data_path)):
        file = open_h5(file_path, nixio.FileMode.ReadOnly)
        if file is None:
            continue
        try:
            block = get_pulse_block(file)
            names = [da.name for da in block.data_arrays]
            if "centers" not in names or "raw_pulses" not in names:
                continue

            fs = float(file.sections["pulses_metadata"]["metadata"]["samplerate"])
            start_str = file.sections["pulses_metadata"]["metadata"]["metadata"]["INFO"][
                "DateTimeOriginal"
            ]
            rec_start = datetime.strptime(start_str, "%Y-%m-%dT%H:%M:%S")
            centers = block.data_arrays["centers"][:]
            pred = block.data_arrays["predicted_labels"][:]
            raw = block.data_arrays["raw_pulses"]
            num_pulses = len(raw)
            candidates = np.where(pred == 1)[0]

            double_m = _full_marker(file_path, block, "is_double_peak", candidates, num_pulses)
            wide_m = _full_marker(file_path, block, "is_wide_pulse", candidates, num_pulses)

            for pulse_idx in candidates:
                pulse_time = rec_start + timedelta(seconds=float(centers[pulse_idx]) / fs)
                trace, _ = get_biggest_unclipped_waveform(raw[pulse_idx][:])
                corrected = baseline_correct(trace)

                if double_m[pulse_idx] == 1:
                    shape = "double"
                    dp = double_pulse_metrics(corrected, fs)
                    half_ms = dp["half_width_ms"]
                    peak_sep_ms = dp["peak_separation_ms"]
                    trough_ratio = dp["trough_depth_ratio"]
                elif wide_m[pulse_idx] == 1:
                    shape = "wide"
                    w, _ = compute_half_max_width(corrected, fs)
                    half_ms, peak_sep_ms, trough_ratio = w * 1000, np.nan, np.nan
                else:
                    shape = "normal"
                    w, _ = compute_half_max_width(corrected, fs)
                    half_ms, peak_sep_ms, trough_ratio = w * 1000, np.nan, np.nan

                rows.append(
                    {
                        "timestamp": pulse_time,
                        "pulse_shape": shape,
                        "half_width_ms": half_ms,
                        "peak_separation_ms": peak_sep_ms,
                        "trough_depth_ratio": trough_ratio,
                        "session": file_path.stem.replace("_pulses", ""),
                    }
                )
        finally:
            file.close()

    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def half_widths_by_shape(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Extract half_width_ms per pulse_shape from a property records DataFrame."""
    out: dict[str, np.ndarray] = {}
    for key in PULSE_SHAPES:
        if df.empty:
            out[key] = np.asarray([], dtype=float)
            continue
        vals = df.loc[df["pulse_shape"] == key, "half_width_ms"].dropna().to_numpy(dtype=float)
        out[key] = vals
    return out
