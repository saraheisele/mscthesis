"""Save RF-classified pulse PCA space plots to the processed output directory.

Analysis part: special-pulse visualization (routine pipeline step).
Dependencies: data_paths, double_peaks_detection, prototype_pulse_plots,
presentation_style.

Uses ``special_pulse_class`` written into each H5 by the classifier apply step
(first step of ``run_analysis.sh``). Hand-label NPZs
(``naturalistic_*_labels*.npz`` / legacy ``labeled_special_pulses*.npz``) are
only the small manually labeled train/test sets used to *train* the RF — not
the full-dataset class map.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from path_setup import setup_script_paths

setup_script_paths(__file__)

from rich.console import Console

from data_paths import H5_DIR, PCA_SPACE_DIR
from presentation_style import apply_presentation_style, save_thesis_figure
from special_pulses.double_peaks_detection import (
    LABELING_PULSE_CLASSES,
    SPECIAL_PULSE_CLASSES,
    normalize_waveforms_for_pca,
    plot_labeled_pulses_pca_space,
)
from special_pulses.prototype_pulse_plots import (
    CLASS_IDS,
    PULSE_SHAPES,
    collect_classifier_pulse_indices,
    load_all_waveforms,
)

console = Console()

# Cap per RF class for the PCA scatter (full dataset has millions of pulses).
PCA_SAMPLE_PER_CLASS = 5_000
PCA_RANDOM_SEED = 42


def collect_rf_waveforms_for_pca(
    data_path=H5_DIR,
    sample_per_class: int = PCA_SAMPLE_PER_CLASS,
    random_seed: int = PCA_RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Sample waveforms + RF class ids from H5 ``special_pulse_class`` markers.

    For doubles (valley alignment), candidates are collected until
    ``sample_per_class`` pulses **pass** the two-peak/valley morph filter —
    not merely until that many RF-class indices are held.
    """
    waveforms: list[np.ndarray] = []
    labels: list[int] = []

    for pulse_key, pulse_shape in PULSE_SHAPES.items():
        class_id = int(pulse_shape["class_id"])
        if class_id not in LABELING_PULSE_CLASSES:
            continue
        class_seed = random_seed + class_id
        align_mode = pulse_shape["align"]

        # Valley morph filter rejects most RF "doubles"; scan the full class
        # index list so load_all_waveforms can keep going until sample_per_class
        # valley-valid waveforms are found (or the class is exhausted).
        if align_mode == "valley":
            collect_cap = None
            console.log(
                f"PCA sample {pulse_key}: collecting all RF-class indices "
                f"(need {sample_per_class:,} with detectable valley/peaks)..."
            )
        else:
            collect_cap = int(sample_per_class * 2.0)

        entries, total = collect_classifier_pulse_indices(
            data_path,
            class_id,
            max_entries=collect_cap,
            random_seed=class_seed,
        )
        if not entries:
            console.log(
                f"[yellow]WARNING: no RF-classified {pulse_key} pulses found in H5."
            )
            continue
        console.log(
            f"PCA sample {pulse_key}: holding {len(entries):,} RF indices "
            f"(class total {total:,}); loading until {sample_per_class:,} "
            f"pass align={align_mode!r}"
        )
        _corrected, normalized, _fs = load_all_waveforms(
            entries,
            align_mode=align_mode,
            max_waveforms=sample_per_class,
            random_seed=class_seed,
        )
        if not normalized:
            console.log(
                f"[yellow]WARNING: could not load waveforms for RF class {pulse_key}."
            )
            continue
        if len(normalized) < sample_per_class:
            console.log(
                f"[yellow]WARNING: only {len(normalized):,}/{sample_per_class:,} "
                f"{pulse_key} pulses passed align={align_mode!r} "
                f"(exhausted {len(entries):,} RF candidates)."
            )
        for trace in normalized:
            waveforms.append(np.asarray(trace, dtype=float))
            labels.append(class_id)
        console.log(f"  kept {len(normalized):,} {pulse_key} waveforms for PCA")

    if len(waveforms) < 2:
        return None, None

    # Common length (load_all_waveforms already unifies on WAVEFORM_FS; pad/crop
    # defensively if classes still differ slightly).
    lengths = {len(w) for w in waveforms}
    if len(lengths) != 1:
        target_len = int(max(lengths))
        console.log(
            f"[yellow]PCA waveform lengths {lengths}; matching to {target_len}."
        )
        from special_pulses.prototype_pulse_plots import match_trace_length

        waveforms = [match_trace_length(w, target_len) for w in waveforms]

    return np.asarray(waveforms, dtype=float), np.asarray(labels, dtype=np.int64)


def save_pca_space_plots(output_dir: Path | None = None) -> Path | None:
    """Create PCA scatter plots from H5 RF class labels, if available."""
    apply_presentation_style()
    output_dir = Path(output_dir) if output_dir else PCA_SPACE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    console.log(
        f"PCA from H5 RF labels (special_pulse_class), "
        f"up to {PCA_SAMPLE_PER_CLASS:,} pulses/class..."
    )
    waveforms, labels = collect_rf_waveforms_for_pca()
    if waveforms is None or labels is None:
        console.log(
            "[yellow]WARNING: skipped labeled_pulses_pca_space.png — "
            "no RF-classified waveforms loaded from H5 "
            "(run classifier apply / pipeline step 1 first)."
        )
        return None

    active_mask = np.isin(labels, list(LABELING_PULSE_CLASSES))
    waveforms = waveforms[active_mask]
    labels = labels[active_mask]
    if len(labels) < 2:
        console.log(
            "[yellow]WARNING: skipped labeled_pulses_pca_space.png — "
            f"not enough RF-classified pulses ({len(labels)})."
        )
        return None

    unique, counts = np.unique(labels, return_counts=True)
    console.log(
        "PCA class counts: "
        + ", ".join(
            f"{SPECIAL_PULSE_CLASSES.get(int(u), u)}={int(c)}"
            for u, c in zip(unique, counts)
        )
    )

    waveforms = normalize_waveforms_for_pca(waveforms)
    output_path = output_dir / "labeled_pulses_pca_space.png"
    fig, _axes = plot_labeled_pulses_pca_space(
        waveforms,
        labels,
        output_path=None,
        show=False,
        class_names=SPECIAL_PULSE_CLASSES,
        legend_title="RF class",
        title="Robust PCA space of RF-classified pulses",
    )
    if fig is None:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    save_thesis_figure("pulse_shapes/labeled_pulses_pca_space.png", fig)
    console.log(f"Saved PCA space plot to {output_path}")
    plt.close(fig)
    return output_path


def main():
    save_pca_space_plots()


if __name__ == "__main__":
    main()
