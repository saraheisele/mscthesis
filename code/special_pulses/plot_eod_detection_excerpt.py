"""Generate Methods detection excerpt and Results intro recording figure.

Uses presentation RC params. Reproduces the Berlin 1 s excerpt with mixed
low/high-amplitude EODs used in Materials & Methods and Results.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from path_setup import setup_script_paths

setup_script_paths(__file__)

import matplotlib.pyplot as plt
import nixio
import numpy as np
from scipy.io import wavfile

from presentation_style import (
    NON_PULSE_SHAPE_COLOR,
    apply_presentation_style,
    pulse_shape_color,
    save_thesis_figure,
)

WAV_PATH = Path(
    "/data2/labdata/eels-mfn2021/berlin_tank_site/recordings_2024-02-26/"
    "eellogger1-20240226T085915.wav"
)
H5_PATH = Path(
    "/home/efish/eelsmfn2021_eods/berlin_tank_site/recordings_2024-02-26_pulses.h5"
)
LATEX_DETECTION = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "latex_thesis"
    / "figures"
    / "detection"
)

# Fixed window chosen for readable pulse density + amplitude contrast.
T0_S = 286.6
WIN_S = 1.0
CUTOUT_SAMPLES = 289


def _load_pulse_times_rel_wav() -> tuple[np.ndarray, float]:
    with nixio.File.open(str(H5_PATH), "r") as handle:
        section = handle.sections["pulses_metadata"]
        fs_h5 = float(section["metadata"]["samplerate"])
        starttime_str = section["metadata"]["metadata"]["INFO"]["DateTimeOriginal"]
        h5_start = datetime.strptime(starttime_str, "%Y-%m-%dT%H:%M:%S")
        block = handle.blocks["pulses_eel_eod"]
        centers = block.data_arrays["centers"][:]
        labels = block.data_arrays["predicted_labels"][:]

    wav_start = datetime.strptime("20240226T085915", "%Y%m%dT%H%M%S")
    mask = labels == 1
    rel = np.array(
        [
            (h5_start + timedelta(seconds=float(c) / fs_h5) - wav_start).total_seconds()
            for c in centers[mask]
        ]
    )
    rel = rel[(rel >= 0) & (rel < 300)]
    return rel, fs_h5


def _polarity(trace: np.ndarray) -> float:
    return -1.0 if abs(np.min(trace)) > np.max(trace) else 1.0


def plot_detection_excerpt() -> Path:
    """Two-panel Methods figure: excerpt with detections + one waveform cutout."""
    apply_presentation_style(force=True)
    rel, fs_h5 = _load_pulse_times_rel_wav()
    t0, win = T0_S, WIN_S
    pulse_times = rel[(rel >= t0) & (rel < t0 + win)]

    fs, data = wavfile.read(str(WAV_PATH))
    i0, i1 = int(t0 * fs), int((t0 + win) * fs)
    seg = data[i0:i1].astype(np.float64)
    ch = int(np.argmax(np.max(np.abs(seg), axis=0)))
    pol = _polarity(seg[:, ch])
    trace = seg[:, ch] * pol
    marker = pulse_shape_color("double")

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10.5, 7.0),
        gridspec_kw={"height_ratios": [1.4, 1.0], "hspace": 0.35},
    )

    ax = axes[0]
    t_ms = (np.arange(i0, i1) / fs - t0) * 1000.0
    ax.plot(t_ms, trace, color="#222222", lw=0.9)
    ymin, ymax = np.percentile(trace, [0.2, 99.8])
    pad = 0.12 * (ymax - ymin + 1)
    ax.set_ylim(ymin - pad, ymax + pad)
    for pt in pulse_times:
        x = (pt - t0) * 1000.0
        ax.axvline(x, color=marker, lw=1.0, alpha=0.8, zorder=2)
        ax.plot(x, ymax + 0.35 * pad, marker="v", color=marker, ms=6, clip_on=False)
    ax.set_xlim(0, win * 1000)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.set_title("A  Recording excerpt with detected EOD times")

    amps = []
    for pt in pulse_times:
        s = int((pt - 0.003) * fs)
        e = int((pt + 0.003) * fs)
        amps.append(np.max(np.abs(data[s:e, ch])))
    pt = pulse_times[int(np.argmax(amps))]

    half_ms = 8.0
    zi0 = int((pt - half_ms / 1000) * fs)
    zi1 = int((pt + half_ms / 1000) * fs)
    zseg = data[zi0:zi1, ch].astype(np.float64) * pol
    zt = (np.arange(zi0, zi1) / fs - pt) * 1000.0

    ax2 = axes[1]
    ax2.plot(zt, zseg, color="#222222", lw=1.4)
    ax2.axvline(0, color=marker, lw=1.2, alpha=0.9, label="Detected pulse time")
    cut_half_ms = (CUTOUT_SAMPLES / 2) / fs_h5 * 1000.0
    ax2.axvspan(-cut_half_ms, cut_half_ms, color=marker, alpha=0.12, label="Waveform cutout")
    ax2.set_xlim(-half_ms, half_ms)
    ax2.set_xlabel("Time relative to detected peak (ms)")
    ax2.set_ylabel("Amplitude (a.u.)")
    ax2.set_title("B  Waveform cutout around one detected EOD")
    ax2.legend(loc="upper right", frameon=False)

    LATEX_DETECTION.mkdir(parents=True, exist_ok=True)
    out = LATEX_DETECTION / "eod_detection_excerpt.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(
        LATEX_DETECTION / "eod_detection_excerpt.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    save_thesis_figure("detection/eod_detection_excerpt.png", fig)
    plt.close(fig)
    print(f"Saved {out}")
    return out


def plot_results_intro_excerpt() -> Path:
    """Single-panel Results intro: mixed low/high amplitude EODs."""
    apply_presentation_style(force=True)
    rel, _ = _load_pulse_times_rel_wav()
    t0, win = T0_S, WIN_S
    pulse_times = rel[(rel >= t0) & (rel < t0 + win)]

    fs, data = wavfile.read(str(WAV_PATH))
    i0, i1 = int(t0 * fs), int((t0 + win) * fs)
    seg = data[i0:i1].astype(np.float64)
    ch = int(np.argmax(np.max(np.abs(seg), axis=0)))
    pol = _polarity(seg[:, ch])
    trace = seg[:, ch] * pol

    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    t_ms = (np.arange(i0, i1) / fs - t0) * 1000.0
    ax.plot(t_ms, trace, color="#222222", lw=0.9)
    ymin, ymax = np.percentile(trace, [0.2, 99.8])
    pad = 0.12 * (ymax - ymin + 1)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlim(0, win * 1000)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (a.u.)")
    # Mark a few low vs high peaks lightly for orientation (no detection claim).
    for pt in pulse_times:
        x = (pt - t0) * 1000.0
        local = trace[max(0, int((pt - t0) * fs) - 20) : int((pt - t0) * fs) + 20]
        if local.size:
            ax.plot(x, np.max(local), marker="o", ms=4, color=NON_PULSE_SHAPE_COLOR, alpha=0.7)

    LATEX_DETECTION.mkdir(parents=True, exist_ok=True)
    out = LATEX_DETECTION / "results_intro_recording_excerpt.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    save_thesis_figure("detection/results_intro_recording_excerpt.png", fig)
    plt.close(fig)
    print(f"Saved {out}")
    return out


def main() -> None:
    plot_detection_excerpt()
    plot_results_intro_excerpt()


if __name__ == "__main__":
    main()
