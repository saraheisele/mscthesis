"""Bootstrap sys.path for scripts in code subpackages."""

from __future__ import annotations

import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent


def setup_script_paths(script_file: str | Path, *extra_packages: str) -> None:
    """Ensure imports from code root, the script's package, and optional siblings work."""
    script_dir = Path(script_file).resolve().parent
    paths = [CODE_DIR, script_dir]
    for name in extra_packages:
        paths.append(CODE_DIR / name)
    for path in paths:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
