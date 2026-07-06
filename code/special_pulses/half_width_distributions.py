"""Plot half-width distributions for all pulse shapes.

Analysis part: pulse shape metrics (outputs isolated via data_paths).
Dependencies: data_paths, h5_io, pulse_shape_metrics, prototype_pulse_plots.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from path_setup import setup_script_paths

setup_script_paths(__file__)

import matplotlib.pyplot as plt
import numpy as np
import nixio
from rich.console import Console

from data_paths import HALF_WIDTH_DISTRIBUTIONS_DIR, H5_DIR
from presentation_style import apply_presentation_style, pulse_shape_color
from h5_io import get_path_list, get_pulse_block, load_marker_array, open_h5
from special_pulses.double_peaks_detection import compute_half_max_width
from special_pulses.prototype_pulse_plots import (
    PULSE_SHAPES,
    baseline_correct,
    expand_marker,
    get_biggest_unclipped_waveform,
)
from special_pulses.pulse_shape_metrics import double_pulse_metrics

console = Console()
OUTPUT_DIR = HALF_WIDTH_DISTRIBUTIONS_DIR


def _full_marker(file_path, block, array_name, candidates, num_pulses):
    raw = load_marker_array(file_path, array_name, block)
    if raw is None:
        return np.zeros(num_pulses, dtype=np.int64)
    return expand_marker(raw, candidates, num_pulses)


def collect_half_widths(data_path) -> dict[str, np.ndarray]:
    """Collect half-width (ms) per pulse shape from all h5 files."""
    results = {key: [] for key in PULSE_SHAPES}

    for file_path in get_path_list(Path(data_path)):
        file = open_h5(file_path, nixio.FileMode.ReadOnly)
        if file is None:
            continue
        try:
            block = get_pulse_block(file)
            names = [da.name for da in block.data_arrays]
            if "raw_pulses" not in names or "predicted_labels" not in names:
                continue

            fs = float(file.sections["pulses_metadata"]["metadata"]["samplerate"])
            raw = block.data_arrays["raw_pulses"]
            pred = block.data_arrays["predicted_labels"][:]
            num_pulses = len(raw)
            candidates = np.where(pred == 1)[0]
            if len(candidates) == 0:
                continue

            double_m = _full_marker(file_path, block, "is_double_peak", candidates, num_pulses)
            wide_m = _full_marker(file_path, block, "is_wide_pulse", candidates, num_pulses)
            fat_m = _full_marker(file_path, block, "is_fat_pulse", candidates, num_pulses)

            for pulse_idx in candidates:
                trace, _ = get_biggest_unclipped_waveform(raw[pulse_idx][:])
                corrected = baseline_correct(trace)

                if double_m[pulse_idx] == 1:
                    dp = double_pulse_metrics(corrected, fs)
                    if not np.isnan(dp["half_width_ms"]):
                        results["double"].append(dp["half_width_ms"])
                elif wide_m[pulse_idx] == 1:
                    w, _ = compute_half_max_width(corrected, fs)
                    results["wide"].append(w * 1000)
                elif fat_m[pulse_idx] == 1:
                    w, _ = compute_half_max_width(corrected, fs)
                    results["fat"].append(w * 1000)
                else:
                    w, _ = compute_half_max_width(corrected, fs)
                    results["normal"].append(w * 1000)
        finally:
            file.close()

    return {k: np.asarray(v, dtype=float) for k, v in results.items()}


def plot_distributions(widths_by_shape: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {k: int(v.size) for k, v in widths_by_shape.items()}
    with open(output_dir / "half_width_pulse_counts.json", "w") as handle:
        json.dump(counts, handle, indent=2)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (key, shape) in zip(axes.ravel(), PULSE_SHAPES.items()):
        values = widths_by_shape[key]
        if values.size == 0:
            ax.set_title(f"{shape['label']} (n=0)")
            continue
        ax.hist(values, bins=50, color=shape["color"], alpha=0.75, edgecolor="white")
        ax.axvline(
            np.median(values),
            color="black",
            linestyle="--",
            label=f"median={np.median(values):.2f} ms",
        )
        ax.set_xlabel("Half width (ms)")
        ax.set_ylabel("Count")
        ax.set_title(f"{shape['label']} (n={values.size:,})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Half-width distributions by pulse shape", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "half_width_distributions_all_shapes.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    for key, shape in PULSE_SHAPES.items():
        values = widths_by_shape[key]
        if values.size == 0:
            continue
        ax.hist(
            values,
            bins=40,
            alpha=0.45,
            label=f"{shape['label']} (n={values.size:,})",
            color=shape["color"],
            density=True,
        )
    ax.set_xlabel("Half width (ms)")
    ax.set_ylabel("Density")
    ax.set_title("Overlaid half-width distributions (all pulse shapes)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "half_width_distributions_overlay.png", dpi=300)
    plt.close(fig)
    console.log(f"Saved half-width plots to {output_dir}")


def main(data_path=H5_DIR):
    apply_presentation_style()
    console.log("Collecting half-widths for all pulse shapes...")
    widths = collect_half_widths(data_path)
    for key, values in widths.items():
        console.log(f"  {PULSE_SHAPES[key]['label']}: {values.size} pulses")
    plot_distributions(widths, OUTPUT_DIR)


if __name__ == "__main__":
    main()
