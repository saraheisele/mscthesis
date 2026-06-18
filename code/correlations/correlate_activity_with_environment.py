"""Run all environmental and contextual correlation analyses.

Analysis part: orchestrator for Part 3 (environment, volleys, tank areas).
Dependencies: environment_correlation, volley_pulse_analysis, tank_area_pulse_analysis.

Runs the three sub-analyses in sequence. Each sub-module can also be executed
standalone for faster iteration on one analysis type.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from path_setup import setup_script_paths

setup_script_paths(__file__)

from data_paths import ENVIRONMENT_CORRELATION_DIR, H5_ROOT
from environment_correlation import main as run_environment_correlation
from tank_area_pulse_analysis import run_area_analysis
from volley_pulse_analysis import run_volley_analysis


def main():
    run_environment_correlation()

    h5_files = sorted(H5_ROOT.glob("**/*.h5"))
    run_volley_analysis(h5_files)
    run_area_analysis(h5_files)

    print(f"\n{'=' * 70}")
    print(f"All analyses complete. Figures saved to: {ENVIRONMENT_CORRELATION_DIR}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
