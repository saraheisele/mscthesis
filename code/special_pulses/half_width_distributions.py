"""Plot half-width distributions for all pulse shapes.

Analysis part: pulse shape metrics (outputs isolated via data_paths).
Dependencies: data_paths, pulse_property_collect, prototype_pulse_plots.
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
from rich.console import Console

from data_paths import HALF_WIDTH_DISTRIBUTIONS_DIR, H5_DIR
from presentation_style import LEGEND_LOC, apply_presentation_style, save_thesis_figure
from special_pulses.prototype_pulse_plots import PULSE_SHAPES
from special_pulses.pulse_property_collect import (
    collect_pulse_property_records,
    half_widths_by_shape,
)

console = Console()
OUTPUT_DIR = HALF_WIDTH_DISTRIBUTIONS_DIR


def plot_distributions(widths_by_shape: dict, output_dir: Path):
    apply_presentation_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {k: int(v.size) for k, v in widths_by_shape.items()}
    with open(output_dir / "half_width_pulse_counts.json", "w") as handle:
        json.dump(counts, handle, indent=2)

    n_shapes = len(PULSE_SHAPES)
    fig, axes = plt.subplots(1, n_shapes, figsize=(4.5 * n_shapes, 5), sharex=True)
    xmax = 0.0
    for values in widths_by_shape.values():
        if values.size:
            xmax = max(xmax, float(np.max(values)))
    shared_xlim = (0.0, xmax * 1.02 if xmax > 0 else 1.0)

    for ax, (key, shape) in zip(np.atleast_1d(axes), PULSE_SHAPES.items()):
        values = widths_by_shape[key]
        if values.size == 0:
            ax.set_title(f"{shape['label']} (n=0)")
            ax.set_xlim(shared_xlim)
            continue
        ax.hist(values, bins=50, color=shape["color"], alpha=0.75, edgecolor="white")
        ax.axvline(
            np.median(values),
            color="black",
            linestyle="--",
            label=f"Median={np.median(values):.2f} ms",
        )
        ax.set_xlabel("Half width (ms)")
        ax.set_ylabel("Count")
        ax.set_title(f"{shape['label']} (n={values.size:,})")
        ax.set_xlim(shared_xlim)
        ax.legend(loc=LEGEND_LOC)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Half-width distributions by pulse shape")
    fig.savefig(output_dir / "half_width_distributions_all_shapes.png", dpi=300)
    save_thesis_figure("pulse_shapes/half_width_distributions_all_shapes.png", fig)
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
    ax.legend(loc=LEGEND_LOC)
    ax.grid(True, alpha=0.3)
    fig.savefig(output_dir / "half_width_distributions_overlay.png", dpi=300)
    save_thesis_figure("pulse_shapes/half_width_distributions_overlay.png", fig)
    plt.close(fig)
    console.log(f"Saved half-width plots to {output_dir}")


def main(data_path=H5_DIR):
    apply_presentation_style()
    console.log("Collecting half-widths for all pulse shapes...")
    df = collect_pulse_property_records(data_path)
    widths = half_widths_by_shape(df)
    for key, values in widths.items():
        console.log(f"  {PULSE_SHAPES[key]['label']}: {values.size} pulses")
    plot_distributions(widths, OUTPUT_DIR)


if __name__ == "__main__":
    main()
