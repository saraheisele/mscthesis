"""Shared I/O helpers for predetected pulse HDF5 files.

Analysis part: infrastructure (used by preprocessing, detection, and correlation scripts).
Dependencies: none.
"""

from pathlib import Path

from rich.console import Console

_console = Console()


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
