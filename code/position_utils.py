"""Shared geometry and I/O for eel head position along the 16-electrode line.

Analysis part: position analysis infrastructure (Part 5).
Dependencies: data_paths.

Electrode geometry: 16 electrodes spaced 25 cm apart along a 3.75 m line.
Channel 0 is in the bright tank area; channel 15 extends toward the dark area.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import nixio

from data_paths import ELECTRODE_LAYOUT_JSON, H5_DIR, LAB_DATA_DIR

N_ELECTRODES = 16
ELECTRODE_SPACING_M = 0.25
LINE_LENGTH_M = (N_ELECTRODES - 1) * ELECTRODE_SPACING_M
DEFAULT_BRIGHT_DARK_BOUNDARY_M = 2.5

WAV_TIME_RE = re.compile(r"(\d{8}T\d{6})")


@dataclass
class PulsePosition:
    """Position estimate for one detected pulse."""

    time_sec: float
    head_m: float
    head_channel: int
    method: str


def default_electrode_positions_m(layout_path: Path | None = None) -> np.ndarray:
    """Return electrode positions along the line in metres (bright → dark)."""
    layout_path = layout_path or ELECTRODE_LAYOUT_JSON
    if layout_path.exists():
        with open(layout_path) as handle:
            layout = json.load(handle)
        coords_cm = np.asarray(layout["coordinates"], dtype=float)
        return coords_cm[:, 0] / 100.0
    return np.arange(N_ELECTRODES, dtype=float) * ELECTRODE_SPACING_M


def channel_amplitudes(pulse_waveform: np.ndarray) -> np.ndarray:
    """Per-channel positive peak amplitudes (samples × channels)."""
    return np.max(pulse_waveform, axis=0)


def head_position_from_pulse(
    pulse_waveform: np.ndarray,
    electrode_positions_m: np.ndarray | None = None,
    method: str = "peak_positive",
) -> tuple[float, int]:
    """Estimate head position along the electrode line from one pulse snippet.

    Parameters
    ----------
    pulse_waveform
        Shape (num_samples, num_channels).
    electrode_positions_m
        Position of each electrode in metres. Defaults to evenly spaced line.
    method
        ``peak_positive``: electrode with largest positive peak (head location).
        ``weighted_mean``: amplitude-weighted mean position using positive peaks.

    Returns
    -------
    head_m, head_channel
        Head position in metres and index of the strongest channel.
    """
    if electrode_positions_m is None:
        electrode_positions_m = default_electrode_positions_m()

    amplitudes = np.asarray(channel_amplitudes(pulse_waveform), dtype=float).ravel()
    positions = np.asarray(electrode_positions_m, dtype=float).ravel()

    # Some sessions have fewer active channels than electrodes in the layout.
    n = min(len(amplitudes), len(positions))
    if n == 0:
        raise ValueError("Empty pulse waveform or electrode layout.")
    amplitudes = amplitudes[:n]
    positions = positions[:n]

    head_channel = int(np.argmax(amplitudes))

    if method == "peak_positive":
        return float(positions[head_channel]), head_channel

    if method == "weighted_mean":
        weights = np.clip(amplitudes, 0.0, None)
        if weights.sum() == 0:
            return float(positions[head_channel]), head_channel
        head_m = float(np.dot(positions, weights) / weights.sum())
        return head_m, head_channel

    raise ValueError(
        f"Unknown position method '{method}'. Use 'peak_positive' or 'weighted_mean'."
    )


def smooth_positions(positions_m: np.ndarray, window: int = 5) -> np.ndarray:
    """Moving-average smoothing along the pulse sequence."""
    if positions_m.size == 0:
        return positions_m
    window = max(1, min(window, positions_m.size))
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(positions_m, kernel, mode="same")


def movement_direction(positions_m: np.ndarray, window: int = 5) -> np.ndarray:
    """Signed movement direction (+1 bright→dark, -1 dark→bright) per pulse."""
    if positions_m.size < 2:
        return np.zeros_like(positions_m)
    smoothed = smooth_positions(positions_m, window=window)
    velocity = np.gradient(smoothed)
    direction = np.sign(velocity)
    direction[direction == 0] = np.nan
    return direction


def eel_body_endpoints(
    head_m: float,
    direction: float,
    body_length_m: float = 2.0,
) -> tuple[float, float]:
    """Return head and tail positions along the line for drawing the eel body."""
    if np.isnan(direction) or direction == 0:
        tail_m = max(0.0, head_m - body_length_m * 0.5)
    elif direction > 0:
        tail_m = head_m - body_length_m
    else:
        tail_m = head_m + body_length_m
    tail_m = float(np.clip(tail_m, 0.0, LINE_LENGTH_M))
    return float(head_m), tail_m


def tank_zone(position_m: float, boundary_m: float = DEFAULT_BRIGHT_DARK_BOUNDARY_M) -> str:
    """Classify position as bright or dark area."""
    return "bright" if position_m < boundary_m else "dark"


def parse_wav_timestamp(wav_path: Path) -> datetime:
    """Parse ``YYYYMMDDTHHMMSS`` timestamp from an eellogger wav filename."""
    match = WAV_TIME_RE.search(wav_path.stem)
    if not match:
        raise ValueError(f"Could not parse timestamp from wav filename: {wav_path.name}")
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S")


def session_name_from_wav(wav_path: Path) -> str:
    """Infer Berlin session folder name from a wav path."""
    for parent in wav_path.parents:
        if parent.name.startswith("recordings"):
            return parent.name
    raise ValueError(
        f"Could not infer session folder from wav path: {wav_path}. "
        "Place the wav inside a recordings_* folder or pass --h5 explicitly."
    )


def h5_path_for_session(session_name: str, h5_dir: Path | None = None) -> Path:
    """Return the predetected-pulses h5 file for a session folder name."""
    h5_dir = h5_dir or H5_DIR
    candidates = sorted(h5_dir.glob(f"{session_name}_pulses.h5"))
    if not candidates:
        raise FileNotFoundError(
            f"No h5 file matching {session_name}_pulses.h5 in {h5_dir}"
        )
    return candidates[0]


def wav_duration_seconds(wav_path: Path) -> float:
    """Estimate wav chunk duration from consecutive files in the session (default 5 min)."""
    session_dir = wav_path.parent
    wav_files = sorted(session_dir.glob("eellogger*.wav"))
    times = []
    for path in wav_files:
        try:
            times.append(parse_wav_timestamp(path))
        except ValueError:
            continue
    if len(times) < 2:
        return 300.0
    gaps = [(t2 - t1).total_seconds() for t1, t2 in zip(times[:-1], times[1:])]
    positive_gaps = [gap for gap in gaps if gap > 0]
    if not positive_gaps:
        return 300.0
    return float(np.median(positive_gaps))


def load_h5_metadata(h5_path: Path) -> tuple[float, datetime, float]:
    """Return samplerate, recording start time, and duration for one h5 file."""
    with nixio.File.open(str(h5_path), "r") as handle:
        section = handle.sections["pulses_metadata"]
        fs = float(section["metadata"]["samplerate"])
        starttime_str = section["metadata"]["metadata"]["INFO"]["DateTimeOriginal"]
        duration = float(section["metadata"]["duration"])
    dt_start = datetime.strptime(starttime_str, "%Y-%m-%dT%H:%M:%S")
    return fs, dt_start, duration


def load_pulses_for_wav(
    wav_path: Path,
    h5_path: Path | None = None,
    method: str = "peak_positive",
    electrode_positions_m: np.ndarray | None = None,
) -> tuple[list[PulsePosition], dict]:
    """Load pulse positions for the time window covered by one wav chunk."""
    wav_path = Path(wav_path)
    session_name = session_name_from_wav(wav_path)
    h5_path = h5_path or h5_path_for_session(session_name)
    wav_start = parse_wav_timestamp(wav_path)
    chunk_duration = wav_duration_seconds(wav_path)
    wav_end = wav_start + timedelta(seconds=chunk_duration)

    fs, h5_start, _duration = load_h5_metadata(h5_path)
    electrode_positions_m = electrode_positions_m or default_electrode_positions_m()

    with nixio.File.open(str(h5_path), "r") as handle:
        block = handle.blocks["pulses"]
        if "centers" not in [da.name for da in block.data_arrays]:
            return [], {
                "wav_path": str(wav_path),
                "h5_path": str(h5_path),
                "wav_start": wav_start,
                "wav_end": wav_end,
                "fs": fs,
            }

        centers = block.data_arrays["centers"][:]
        labels = block.data_arrays["predicted_labels"][:]
        raw_pulses = block.data_arrays["raw_pulses"][:]
        mask = labels == 1

    pulse_positions: list[PulsePosition] = []
    for center_idx, pulse in zip(centers[mask], raw_pulses[mask]):
        pulse_time = h5_start + timedelta(seconds=float(center_idx) / fs)
        if pulse_time < wav_start or pulse_time >= wav_end:
            continue
        head_m, head_channel = head_position_from_pulse(
            pulse, electrode_positions_m, method=method
        )
        pulse_positions.append(
            PulsePosition(
                time_sec=(pulse_time - wav_start).total_seconds(),
                head_m=head_m,
                head_channel=head_channel,
                method=method,
            )
        )

    meta = {
        "wav_path": str(wav_path),
        "h5_path": str(h5_path),
        "session_name": session_name,
        "wav_start": wav_start,
        "wav_end": wav_end,
        "fs": fs,
        "n_pulses": len(pulse_positions),
        "method": method,
    }
    return pulse_positions, meta


def find_entry_recordings(
    h5_dir: Path | None = None,
    method: str = "peak_positive",
    min_pulses: int = 20,
    edge_m: float = 0.5,
    late_fraction: float = 0.3,
) -> list[dict]:
    """Find wav-scale windows where the eel appears to enter from an edge.

    Heuristic: early pulses are near one line end and later pulses move toward
    the interior (useful for picking interesting animation candidates).
    """
    h5_dir = h5_dir or H5_DIR
    electrode_positions_m = default_electrode_positions_m()
    candidates: list[dict] = []

    for h5_path in sorted(h5_dir.glob("*_pulses.h5")):
        fs, h5_start, duration = load_h5_metadata(h5_path)
        with nixio.File.open(str(h5_path), "r") as handle:
            block = handle.blocks["pulses"]
            if "centers" not in [da.name for da in block.data_arrays]:
                continue
            centers = block.data_arrays["centers"][:]
            labels = block.data_arrays["predicted_labels"][:]
            raw_pulses = block.data_arrays["raw_pulses"][:]
            mask = labels == 1
            if mask.sum() < min_pulses:
                continue

        positions = []
        times_sec = []
        for center_idx, pulse in zip(centers[mask], raw_pulses[mask]):
            head_m, _ = head_position_from_pulse(
                pulse, electrode_positions_m, method=method
            )
            positions.append(head_m)
            times_sec.append(float(center_idx) / fs)

        positions_arr = np.asarray(positions)
        times_arr = np.asarray(times_sec)
        chunk_duration = 300.0
        n_chunks = int(np.ceil(duration / chunk_duration))

        for chunk_idx in range(n_chunks):
            chunk_start = chunk_idx * chunk_duration
            chunk_end = chunk_start + chunk_duration
            in_chunk = (times_arr >= chunk_start) & (times_arr < chunk_end)
            if in_chunk.sum() < min_pulses:
                continue

            chunk_pos = positions_arr[in_chunk]
            chunk_times = times_arr[in_chunk] - chunk_start
            early = chunk_times <= chunk_duration * (1 - late_fraction)
            late = chunk_times >= chunk_duration * late_fraction
            if early.sum() < 5 or late.sum() < 5:
                continue

            early_mean = float(np.mean(chunk_pos[early]))
            late_mean = float(np.mean(chunk_pos[late]))
            near_bright = early_mean <= edge_m and late_mean > early_mean + 0.3
            near_dark = early_mean >= LINE_LENGTH_M - edge_m and late_mean < early_mean - 0.3
            if not (near_bright or near_dark):
                continue

            chunk_start_dt = h5_start + timedelta(seconds=chunk_start)
            wav_name = f"eellogger1-{chunk_start_dt.strftime('%Y%m%dT%H%M%S')}.wav"
            session_name = h5_path.stem.replace("_pulses", "")
            wav_path = LAB_DATA_DIR / session_name / wav_name
            candidates.append(
                {
                    "session": session_name,
                    "wav_name": wav_name,
                    "wav_path": str(wav_path),
                    "chunk_start_sec": chunk_start,
                    "entry_from": "bright" if near_bright else "dark",
                    "early_mean_m": early_mean,
                    "late_mean_m": late_mean,
                    "n_pulses": int(in_chunk.sum()),
                }
            )

    return sorted(candidates, key=lambda item: item["n_pulses"], reverse=True)
