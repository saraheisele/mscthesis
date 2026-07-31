"""Central path configuration for the eel analysis pipeline.

Analysis part: infrastructure — all scripts import paths from here.
Dependencies: none.

Set USE_DUMMY_DATASET (or env EEL_USE_DUMMY_DATASET) to switch between the
development subset and the full Berlin tank dataset.

Dummy thesis figures → figures/
Full thesis figures  → figures/full/
Dummy processed plots → data/processed/dummy/
Full processed plots  → data/processed/
"""

from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = PROJECT_ROOT / "code"
THESIS_FIGURES_DIR = PROJECT_ROOT / "figures"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def active_thesis_figures_dir() -> Path:
    """Root directory for thesis figures for the active dataset."""
    if USE_DUMMY_DATASET:
        return THESIS_FIGURES_DIR
    return THESIS_FIGURES_DIR / "full"

#################################
######## DATASET SELECTION ######
#################################

# True  → dummy subset, outputs under data/processed/dummy/
# False → full dataset at /home/efish/eelsmfn2021_eods/berlin_tank_site
USE_DUMMY_DATASET = _env_bool("EEL_USE_DUMMY_DATASET", True)

#################################
############# INPUT #############
#################################

FULL_H5_DIR = Path("/home/efish/eelsmfn2021_eods/berlin_tank_site")
DUMMY_H5_DIR = (
    PROJECT_ROOT / "data/raw/eels-mfn2021_dummy_pulses_redetected/berlin_tank_site"
)
LAB_DATA_DIR = Path("/data2/labdata/eels-mfn2021/berlin_tank_site")

if USE_DUMMY_DATASET:
    H5_DIR = DUMMY_H5_DIR
else:
    H5_DIR = FULL_H5_DIR

H5_ROOT = H5_DIR  # alias used by some correlation scripts

# Shared lab metadata (temperature / conductivity Excel). Same folder for dummy
# and full modes — there is no separate full-dataset Excel tree in this repo.
EXCEL_DIR = (
    PROJECT_ROOT
    / "data/raw/eels-mfn2021_dummy_pulses_redetected/leitwerte_metadaten"
)

#################################
########## INTERMEDIATE #########
#################################

INTERMEDIATE_DIR = PROJECT_ROOT / "data/intermediate"

if USE_DUMMY_DATASET:
    ACTIVITY_HISTOGRAMS_DIR = INTERMEDIATE_DIR / "eels-mfn2021_dummy_activity_histograms"
    POSITION_HISTOGRAMS_DIR = INTERMEDIATE_DIR / "eels-mfn2021_dummy_position_histograms"
else:
    ACTIVITY_HISTOGRAMS_DIR = INTERMEDIATE_DIR / "eels-mfn2021_activity_histograms"
    POSITION_HISTOGRAMS_DIR = INTERMEDIATE_DIR / "eels-mfn2021_position_histograms"

SPECIAL_PULSE_CLASSIFIER_DIR = INTERMEDIATE_DIR / "special_pulse_classifier"
if USE_DUMMY_DATASET:
    SPECIAL_PULSE_MARKERS_DIR = INTERMEDIATE_DIR / "special_pulse_markers"
else:
    SPECIAL_PULSE_MARKERS_DIR = INTERMEDIATE_DIR / "special_pulse_markers_full"
SESSION_PATHS_JSON = INTERMEDIATE_DIR / "eellogger_session_paths.json"

LEGACY_PULSE_DATA_NPZ = INTERMEDIATE_DIR / "pulse_data.npz"
LEGACY_INTERMEDIATE_PULSE_DATA_NPZ = INTERMEDIATE_DIR / "intermediate_pulse_data.npz"
EXAMPLE_WAV_DIR = PROJECT_ROOT / "data/raw/eellogger_example_data/recordings2025-03-31-20250401"

#################################
########### PROCESSED ###########
#################################

PROCESSED_DIR = PROJECT_ROOT / "data/processed"

# Full-dataset results live directly under PROCESSED_DIR; dummy results are isolated.
if USE_DUMMY_DATASET:
    PROCESSED_DATASET_DIR = PROCESSED_DIR / "dummy"
else:
    PROCESSED_DATASET_DIR = PROCESSED_DIR

FEEDING_CORRELATION_DIR = PROCESSED_DATASET_DIR / "feeding_correlation"
ENVIRONMENT_CORRELATION_DIR = PROCESSED_DATASET_DIR / "environment_correlation"
PULSE_SHAPE_CORRELATION_DIR = PROCESSED_DATASET_DIR / "pulse_shape_correlation"
POSITION_FIGURES_DIR = PROCESSED_DATASET_DIR / "position_analysis"
PULSE_SHAPE_PROTOTYPES_DIR = PROCESSED_DATASET_DIR / "pulse_shape_prototypes"
HALF_WIDTH_DISTRIBUTIONS_DIR = PROCESSED_DATASET_DIR / "half_width_distributions"
MATING_CORRELATION_DIR = PROCESSED_DATASET_DIR / "mating_correlation"
PCA_SPACE_DIR = PROCESSED_DATASET_DIR / "pca_space"

ELECTRODE_LAYOUT_JSON = LAB_DATA_DIR / "electrode_layout.json"
EEL_SVG = CODE_DIR / "assets" / "eel.svg"

# Recording corrections applied during histogram preprocessing
DUAL_LINE_START_DATE = "2025-11-25"
PARTIAL_RECORDING_YEARS = (2023, 2026)


def activity_hist_dir(hist_subdir: str) -> Path:
    """Directory with preprocessed histogram .npz files for one pulse type."""
    return ACTIVITY_HISTOGRAMS_DIR / hist_subdir


def processed_figures_dir(figures_subdir: str) -> Path:
    """Output directory for pulse analysis figures."""
    return PROCESSED_DATASET_DIR / figures_subdir


def position_hist_dir(hist_subdir: str = "") -> Path:
    """Directory with preprocessed position histogram .npz files."""
    if hist_subdir:
        return POSITION_HISTOGRAMS_DIR / hist_subdir
    return POSITION_HISTOGRAMS_DIR
