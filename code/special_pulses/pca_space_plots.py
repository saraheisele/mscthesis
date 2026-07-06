"""Save labeled-pulse PCA space plots to the processed output directory.

Analysis part: special-pulse visualization (routine pipeline step).
Dependencies: data_paths, double_peaks_detection, presentation_style.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from path_setup import setup_script_paths

setup_script_paths(__file__)

from rich.console import Console

from data_paths import PCA_SPACE_DIR, SPECIAL_PULSE_CLASSIFIER_DIR
from presentation_style import apply_presentation_style
from special_pulses.double_peaks_detection import (
    LABELING_PULSE_CLASSES,
    SPECIAL_PULSE_CLASSES,
    get_default_ml_paths,
    load_labeled_dataset,
    normalize_waveforms_for_pca,
    plot_labeled_pulses_pca_space,
)

console = Console()


def save_pca_space_plots(output_dir: Path | None = None) -> Path | None:
    """Create PCA scatter plots from the labeled pulse dataset, if available."""
    apply_presentation_style()
    output_dir = Path(output_dir) if output_dir else PCA_SPACE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    ml_paths = get_default_ml_paths()
    labels_path = ml_paths["labels"]
    if not labels_path.exists():
        console.log(
            f"[yellow]No labeled pulse dataset at {labels_path}; skipping PCA plots."
        )
        return None

    waveforms, labels, _ = load_labeled_dataset(labels_path)
    active_mask = np.isin(labels, list(LABELING_PULSE_CLASSES))
    waveforms = waveforms[active_mask]
    labels = labels[active_mask]
    if len(labels) < 2:
        console.log("[yellow]Not enough labeled pulses for PCA plots.")
        return None

    waveforms = normalize_waveforms_for_pca(waveforms)
    output_path = output_dir / "labeled_pulses_pca_space.png"
    plot_labeled_pulses_pca_space(
        waveforms,
        labels,
        output_path=output_path,
        show=False,
        class_names=SPECIAL_PULSE_CLASSES,
    )
    console.log(f"Saved PCA space plot to {output_path}")
    return output_path


def main():
    save_pca_space_plots()


if __name__ == "__main__":
    main()
