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
from scipy import stats

from data_paths import HALF_WIDTH_DISTRIBUTIONS_DIR, H5_DIR
from presentation_style import (
    LEGEND_LOC,
    apply_presentation_style,
    pulse_shape_color,
    save_thesis_figure,
    shade_hex,
)
from special_pulses.prototype_pulse_plots import PULSE_SHAPE_DISPLAY_ORDER, PULSE_SHAPES
from special_pulses.pulse_property_collect import (
    collect_pulse_property_records,
    half_widths_by_shape,
)

console = Console()
OUTPUT_DIR = HALF_WIDTH_DISTRIBUTIONS_DIR

# Drop near-zero / spurious widths; thesis x-axis focuses on the main mass.
HALF_WIDTH_MIN_MS = 0.7
HALF_WIDTH_XLIM_MS = 3.5


def _filter_widths(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return values[values >= HALF_WIDTH_MIN_MS]


def plot_distributions(widths_by_shape: dict, output_dir: Path):
    apply_presentation_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    filtered = {k: _filter_widths(v) for k, v in widths_by_shape.items()}
    counts = {k: int(v.size) for k, v in filtered.items()}
    with open(output_dir / "half_width_pulse_counts.json", "w") as handle:
        json.dump(
            {
                "min_ms": HALF_WIDTH_MIN_MS,
                "xlim_ms": HALF_WIDTH_XLIM_MS,
                "counts": counts,
            },
            handle,
            indent=2,
        )

    shared_xlim = (0.0, HALF_WIDTH_XLIM_MS)

    # Per-shape panels (exploratory / archive).
    n_shapes = len(PULSE_SHAPES)
    fig, axes = plt.subplots(1, n_shapes, figsize=(4.5 * n_shapes, 5), sharex=True)
    for ax, (key, shape) in zip(np.atleast_1d(axes), PULSE_SHAPES.items()):
        values = filtered[key]
        if values.size == 0:
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
        ax.set_xlim(shared_xlim)
        ax.legend(loc=LEGEND_LOC)
        ax.grid(True, alpha=0.3)

    fig.savefig(output_dir / "half_width_distributions_all_shapes.png", dpi=300)
    save_thesis_figure("pulse_shapes/half_width_distributions_all_shapes.png", fig)
    plt.close(fig)

    # Overlaid densities (also left panel of combined thesis figure).
    fig, ax = plt.subplots(figsize=(12, 6))
    _plot_overlay_on_ax(ax, filtered)
    fig.savefig(output_dir / "half_width_distributions_overlay.png", dpi=300)
    save_thesis_figure("pulse_shapes/half_width_distributions_overlay.png", fig)
    plt.close(fig)

    # Pooled histogram + KDE (also right panel of combined thesis figure).
    all_values = np.concatenate(
        [filtered[k] for k in PULSE_SHAPE_DISPLAY_ORDER if filtered[k].size]
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_pooled_on_ax(ax, all_values)
    fig.savefig(output_dir / "half_width_kde_all_shapes.png", dpi=300)
    save_thesis_figure("pulse_shapes/half_width_kde_all_shapes.png", fig)
    plt.close(fig)

    # Thesis figure: per-shape overlay | pooled histogram side by side.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True, constrained_layout=True)
    _plot_overlay_on_ax(axes[0], filtered)
    _plot_pooled_on_ax(axes[1], all_values)
    axes[0].set_ylabel("Density")
    axes[1].set_ylabel("Density")
    fig.savefig(output_dir / "half_width_distributions_combined.png", dpi=300)
    save_thesis_figure("pulse_shapes/half_width_distributions_combined.png", fig)
    plt.close(fig)

    console.log(f"Saved half-width plots to {output_dir}")


def _plot_overlay_on_ax(ax, filtered: dict[str, np.ndarray]) -> None:
    for key in PULSE_SHAPE_DISPLAY_ORDER:
        shape = PULSE_SHAPES[key]
        values = filtered[key]
        if values.size == 0:
            continue
        ax.hist(
            values,
            bins=40,
            alpha=0.45,
            label=f"{shape['label']} (n={values.size:,})",
            color=shape["color"],
            density=True,
            range=(0.0, HALF_WIDTH_XLIM_MS),
        )
    ax.set_xlabel("Half width (ms)")
    ax.set_ylabel("Density")
    ax.set_xlim(0.0, HALF_WIDTH_XLIM_MS)
    ax.legend(loc=LEGEND_LOC)
    ax.grid(True, alpha=0.3)


def _plot_pooled_on_ax(ax, values: np.ndarray) -> None:
    if values.size == 0:
        ax.set_xlim(0.0, HALF_WIDTH_XLIM_MS)
        return

    hist_color = pulse_shape_color("all")
    kde_color = shade_hex(hist_color, 0.55)
    ax.hist(
        values,
        bins=80,
        density=True,
        color=hist_color,
        alpha=0.75,
        edgecolor="white",
        linewidth=0.3,
        range=(0.0, HALF_WIDTH_XLIM_MS),
    )
    if values.size >= 50:
        sample = (
            values
            if values.size <= 50_000
            else np.random.default_rng(0).choice(values, size=50_000, replace=False)
        )
        kde_x = np.linspace(0.0, HALF_WIDTH_XLIM_MS, 300)
        kde_y = stats.gaussian_kde(sample)(kde_x)
        ax.plot(kde_x, kde_y, color=kde_color, alpha=0.65, linewidth=2.2, label="KDE")
        ax.legend(loc=LEGEND_LOC)
    ax.set_xlabel("Half width (ms)")
    ax.set_ylabel("Density")
    ax.set_xlim(0.0, HALF_WIDTH_XLIM_MS)
    ax.grid(True, alpha=0.3)


def plot_distributions_from_dataframe(df, output_dir: Path | None = None) -> None:
    """Plot half-width figures from an existing property DataFrame."""
    output_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    widths = half_widths_by_shape(df)
    plot_distributions(widths, output_dir)


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
