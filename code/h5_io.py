"""Shared I/O helpers for predetected pulse HDF5 files.

Analysis part: infrastructure (used by preprocessing, detection, and correlation scripts).
Dependencies: none.
"""

from pathlib import Path

import numpy as np
import nixio
from rich.console import Console

from data_paths import SPECIAL_PULSE_MARKERS_DIR

_console = Console()

PULSE_BLOCK_NAMES = ("pulses", "pulses_eel_eod")


def get_pulse_block(nix_file):
    """Return the eel pulse data block (legacy or full-dataset layout)."""
    for name in PULSE_BLOCK_NAMES:
        try:
            return nix_file.blocks[name]
        except KeyError:
            continue
    raise KeyError(
        f"No pulse block found; expected one of {PULSE_BLOCK_NAMES}"
    )


def marker_sidecar_path(h5_path, array_name: str) -> Path:
    return SPECIAL_PULSE_MARKERS_DIR / f"{Path(h5_path).stem}_{array_name}.npz"


def save_marker_sidecar(h5_path, array_name: str, values) -> Path:
    SPECIAL_PULSE_MARKERS_DIR.mkdir(parents=True, exist_ok=True)
    path = marker_sidecar_path(h5_path, array_name)
    np.savez_compressed(path, marker=np.asarray(values, dtype=np.int64))
    return path


def load_marker_array(h5_path, array_name: str, block):
    """Load a pulse marker from the h5 block or a sidecar .npz file."""
    data_array_names = [da.name for da in block.data_arrays]
    if array_name in data_array_names:
        return block.data_arrays[array_name][:]
    sidecar = marker_sidecar_path(h5_path, array_name)
    if sidecar.exists():
        return np.load(sidecar)["marker"]
    return None


def open_h5(path, mode=nixio.FileMode.ReadOnly):
    """Open an HDF5 file, or return None if it is locked or unreadable."""
    try:
        return nixio.File.open(str(path), mode)
    except Exception as exc:
        _console.log(
            f"[yellow]Skipping {path}: cannot open file "
            f"({type(exc).__name__}: {exc})[/yellow]"
        )
        return None


def open_h5_readwrite_or_readonly(path):
    """Open for writing when permitted, otherwise read-only for sidecar output."""
    file = open_h5(path, nixio.FileMode.ReadWrite)
    if file is not None:
        return file, "h5"
    file = open_h5(path, nixio.FileMode.ReadOnly)
    if file is not None:
        _console.log(
            f"[yellow]{path}: opened read-only; markers will be saved to sidecar files[/yellow]"
        )
        return file, "sidecar"
    return None, None


def get_path_list(datapath: Path) -> list[Path]:
    """Recursively find all .h5 files under datapath (file or directory)."""
    datapath = Path(datapath)
    _console.log("Loading detected pulses from hdf5 files.")

    if not datapath.exists():
        raise FileNotFoundError(f"Path {datapath} does not exist.")

    if datapath.is_file():
        if datapath.suffix != ".h5":
            raise FileNotFoundError(f"Path {datapath} is not an hdf5 file.")
        _console.log(f"Path {datapath} is a single hdf5 file.")
        return [datapath]

    if datapath.is_dir():
        path_list = sorted(datapath.rglob("*.h5"))
        _console.log(f"Found {len(path_list)} hdf5 files in {datapath}.")
        return path_list

    raise FileNotFoundError(f"Path {datapath} is neither a file nor a directory.")
