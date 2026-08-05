"""Plot one random raw pulse from early 2025 and early 2026 (strongest channel)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from path_setup import setup_script_paths

setup_script_paths(__file__)

import matplotlib.pyplot as plt
import nixio
import numpy as np

from data_paths import H5_DIR
from h5_io import get_pulse_block

FILES = {
    "2025-01-03": H5_DIR / "recordings_2025-01-03_pulses.h5",
    "2026-01-07": H5_DIR / "recordings_2026-01-07_brightarea_pulses.h5",
}


def strongest_channel_trace(pulse):
    pulse = np.asarray(pulse, dtype=float)
    ch = int(np.argmax(np.max(np.abs(pulse), axis=0)))
    return pulse[:, ch], ch


def load_random_trace(path, rng):
    nix_file = nixio.File.open(str(path), nixio.FileMode.ReadOnly)
    try:
        raw = get_pulse_block(nix_file).data_arrays["raw_pulses"]
        idx = int(rng.integers(0, len(raw)))
        trace, ch = strongest_channel_trace(raw[idx][:])
        return trace, idx, ch
    finally:
        nix_file.close()


def main():
    rng = np.random.default_rng()
    fig, axes = plt.subplots(2, 1, sharex=False, figsize=(8, 5))

    for ax, (label, path) in zip(axes, FILES.items()):
        trace, idx, ch = load_random_trace(path, rng)
        ax.plot(trace, color="k", lw=1)
        ax.set_title(f"{label}  pulse {idx}  ch {ch}")
        ax.set_ylabel("amplitude")

    axes[-1].set_xlabel("sample")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
