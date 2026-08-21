"""Regenerate selected thesis figures after style/layout tweaks.

Uses cached counts / timeseries where possible. Prototype overlays need H5
access for a small gray-trace sample (medians come from saved NPZs).

Run with: EEL_USE_DUMMY_DATASET=0 python code/special_pulses/replot_thesis_pulse_figures.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from path_setup import setup_script_paths

setup_script_paths(__file__)

import pandas as pd
from rich.console import Console

from data_paths import (
    HALF_WIDTH_DISTRIBUTIONS_DIR,
    PROJECT_ROOT,
    PULSE_SHAPE_PROTOTYPES_DIR,
    USE_DUMMY_DATASET,
    active_thesis_figures_dir,
)
from presentation_style import apply_presentation_style
from special_pulses.half_width_distributions import plot_distributions_from_dataframe
from special_pulses.prototype_pulse_plots import replot_prototypes_from_cache
from special_pulses.pulse_shape_distribution import plot_pulse_shape_distribution

console = Console()

COUNTS_JSON = PULSE_SHAPE_PROTOTYPES_DIR / "pulse_shape_distribution_counts.json"
PROPERTIES_CSV = (
    PROJECT_ROOT / "data/processed/mating_correlation/pulse_properties_timeseries.csv"
)
LATEX_FIGURES = PROJECT_ROOT / "docs/latex_thesis/figures"


def _sync_pulse_shape_pngs() -> None:
    """Copy regenerated pulse_shapes PNGs into the LaTeX tree."""
    src_root = active_thesis_figures_dir() / "pulse_shapes"
    dst_root = LATEX_FIGURES / "pulse_shapes"
    dst_root.mkdir(parents=True, exist_ok=True)
    names = [
        "pulse_shape_distribution.png",
        "prototype_pulses_stacked.png",
        "prototype_normal_pulse.png",
        "prototype_wide_pulse.png",
        "prototype_double_pulse.png",
        "normal_pulses_aligned_to_double.png",
        "mean_pulse_shapes_panel.png",
        "half_width_distributions_combined.png",
        "half_width_distributions_overlay.png",
        "half_width_kde_all_shapes.png",
        "half_width_distributions_all_shapes.png",
    ]
    for name in names:
        src = src_root / name
        if src.exists():
            shutil.copy2(src, dst_root / name)
            console.log(f"Synced {name} → latex figures")


def main() -> None:
    apply_presentation_style()
    console.log(f"USE_DUMMY_DATASET={USE_DUMMY_DATASET}")
    console.log(f"Thesis figures → {active_thesis_figures_dir()}")

    if not COUNTS_JSON.exists():
        raise FileNotFoundError(COUNTS_JSON)
    with open(COUNTS_JSON) as handle:
        counts = json.load(handle)["overall"]
    plot_pulse_shape_distribution(
        counts,
        filename="pulse_shapes/pulse_shape_distribution.png",
        output_dir=PULSE_SHAPE_PROTOTYPES_DIR,
        y_unit="mio",
    )

    if not PROPERTIES_CSV.exists():
        raise FileNotFoundError(PROPERTIES_CSV)
    console.log(f"Loading {PROPERTIES_CSV} ...")
    df = pd.read_csv(PROPERTIES_CSV, usecols=["pulse_shape", "half_width_ms"])
    plot_distributions_from_dataframe(df, HALF_WIDTH_DISTRIBUTIONS_DIR)

    console.log("Replotting prototypes from cached medians + H5 overlay sample...")
    replot_prototypes_from_cache()

    _sync_pulse_shape_pngs()
    console.log("Done.")


if __name__ == "__main__":
    main()
