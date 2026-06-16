"""Central path configuration for the eel analysis pipeline.

Analysis part: infrastructure — all scripts import paths from here.
Dependencies: none.

Switch from the development test subset to the full dataset by changing
H5_DIR (and optionally ACTIVITY_HISTOGRAMS_DIR / POSITION_HISTOGRAMS_DIR) below.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = PROJECT_ROOT / "code"

#################################
############# INPUT #############
#################################

# Predetected pulse .h5 files (Berlin tank site)
H5_DIR = PROJECT_ROOT / "data/raw/eels-mfn2021_dummy_pulses_redetected/berlin_tank_site"
# Full dataset — uncomment and set when rerunning on all recordings:
# H5_DIR = Path("/data2/labdata/eels-mfn2021/berlin_tank_site/predetected_pulses")

# Parent directory for recursive **/*.h5 searches (e.g. volley analysis)
H5_ROOT = H5_DIR.parent

# Raw lab recordings: wav files and session Word documents (images/*.docx)
LAB_DATA_DIR = Path("/data2/labdata/eels-mfn2021/berlin_tank_site")

# Environmental sensor Excel files (temperature, conductivity)
EXCEL_DIR = (
    PROJECT_ROOT
    / "data/raw/eels-mfn2021_dummy_pulses_redetected/leitwerte_metadaten"
)

#################################
########## INTERMEDIATE #########
#################################

INTERMEDIATE_DIR = PROJECT_ROOT / "data/intermediate"

ACTIVITY_HISTOGRAMS_DIR = INTERMEDIATE_DIR / "eels-mfn2021_dummy_activity_histograms"
# Full dataset — optionally use a separate output folder:
# ACTIVITY_HISTOGRAMS_DIR = INTERMEDIATE_DIR / "eels-mfn2021_activity_histograms"

SPECIAL_PULSE_CLASSIFIER_DIR = INTERMEDIATE_DIR / "special_pulse_classifier"
SESSION_PATHS_JSON = INTERMEDIATE_DIR / "eellogger_session_paths.json"

# Legacy paths used by old_analysis_version scripts
LEGACY_PULSE_DATA_NPZ = INTERMEDIATE_DIR / "pulse_data.npz"
LEGACY_INTERMEDIATE_PULSE_DATA_NPZ = INTERMEDIATE_DIR / "intermediate_pulse_data.npz"
EXAMPLE_WAV_DIR = PROJECT_ROOT / "data/raw/eellogger_example_data/recordings2025-03-31-20250401"

#################################
########### PROCESSED ###########
#################################

PROCESSED_DIR = PROJECT_ROOT / "data/processed"
FEEDING_CORRELATION_DIR = PROCESSED_DIR / "feeding_correlation"
ENVIRONMENT_CORRELATION_DIR = PROCESSED_DIR / "environment_correlation"
PULSE_SHAPE_CORRELATION_DIR = PROCESSED_DIR / "pulse_shape_correlation"
POSITION_HISTOGRAMS_DIR = INTERMEDIATE_DIR / "eels-mfn2021_dummy_position_histograms"
POSITION_FIGURES_DIR = PROCESSED_DIR / "position_analysis"

# Full dataset — optionally use separate output folders:
# POSITION_HISTOGRAMS_DIR = INTERMEDIATE_DIR / "eels-mfn2021_position_histograms"

# Berlin tank electrode line layout (cm coordinates, bright → dark)
ELECTRODE_LAYOUT_JSON = LAB_DATA_DIR / "electrode_layout.json"


def activity_hist_dir(hist_subdir: str) -> Path:
    """Directory with preprocessed histogram .npz files for one pulse type."""
    return ACTIVITY_HISTOGRAMS_DIR / hist_subdir


def processed_figures_dir(figures_subdir: str) -> Path:
    """Output directory for pulse analysis figures."""
    return PROCESSED_DIR / figures_subdir


def position_hist_dir(hist_subdir: str = "") -> Path:
    """Directory with preprocessed position histogram .npz files."""
    if hist_subdir:
        return POSITION_HISTOGRAMS_DIR / hist_subdir
    return POSITION_HISTOGRAMS_DIR
