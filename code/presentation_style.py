"""Presentation matplotlib style: thesis color palette and readable defaults.

Analysis part: shared visualization theme for thesis presentation figures.
Dependencies: matplotlib, data_paths.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

THESIS_COLORS = (
    "#4477AA",
    "#EE6677",
    "#228833",
    "#CCBB44",
    "#66CCEE",
    "#AA3377",
    "#BBBBBB",
)

THESIS_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "thesis_diverging",
    [THESIS_COLORS[0], THESIS_COLORS[6], THESIS_COLORS[1]],
)

PULSE_SHAPE_COLORS = {
    "normal": THESIS_COLORS[0],
    "double": THESIS_COLORS[1],
    "wide": THESIS_COLORS[2],
    "all": THESIS_COLORS[4],
}

DARK_AREA_FACE_COLOR = "#808080"
DARK_AREA_ALPHA = 0.28

LEGEND_LOC = "upper right"

PRESENTATION_RCPARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.labelsize": 16,
    "axes.labelweight": "bold",
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "legend.title_fontsize": 13,
    "lines.linewidth": 2.5,
    "lines.markersize": 8,
    "axes.linewidth": 1.5,
    "xtick.major.width": 1.3,
    "ytick.major.width": 1.3,
    "figure.dpi": 100,
    "figure.autolayout": True,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}

_style_applied = False


def pulse_shape_color(key: str) -> str:
    """Return the presentation color for a pulse-shape category key."""
    return PULSE_SHAPE_COLORS.get(key, THESIS_COLORS[5])


def classifier_label_colors(label_ids, class_names: dict) -> list[str]:
    """Map classifier label ids to consistent pulse-shape colors."""
    colors = []
    for label_id in label_ids:
        name = class_names.get(int(label_id), "")
        colors.append(pulse_shape_color(name if name in PULSE_SHAPE_COLORS else "all"))
    return colors


def apply_presentation_style(force: bool = False) -> None:
    """Apply shared rcParams for presentation-ready figures."""
    global _style_applied
    if _style_applied and not force:
        return
    plt.rcParams.update(PRESENTATION_RCPARAMS)
    _style_applied = True


def save_thesis_figure(
    filename: str,
    fig=None,
    *,
    dpi: int = 300,
    **kwargs,
) -> Path:
    """Save a figure to the thesis figures directory."""
    from data_paths import active_thesis_figures_dir

    figures_dir = active_thesis_figures_dir()
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    if fig is None:
        plt.savefig(path, dpi=dpi, **kwargs)
    else:
        fig.savefig(path, dpi=dpi, **kwargs)
    return path


def copy_thesis_asset(source: Path, filename: str | None = None) -> Path:
    """Copy a non-PNG asset (e.g. animation) into the thesis figures directory."""
    from data_paths import active_thesis_figures_dir

    figures_dir = active_thesis_figures_dir()
    figures_dir.mkdir(parents=True, exist_ok=True)
    dest = figures_dir / (filename or source.name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    return dest


def thesis_figure_path(filename: str) -> Path:
    """Return the absolute path for a thesis figure relative to the active root."""
    from data_paths import active_thesis_figures_dir

    path = active_thesis_figures_dir() / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_thesis_json(filename: str, payload) -> Path:
    """Write a JSON sidecar next to thesis figures (respects dummy vs full root)."""
    import json

    path = thesis_figure_path(filename)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return path


def shade_dark_region(
    ax,
    boundary_m: float,
    *,
    y_min: float | None = None,
    y_max: float | None = None,
    facecolor: str = DARK_AREA_FACE_COLOR,
    alpha: float = DARK_AREA_ALPHA,
    zorder: float = 0,
) -> None:
    """Fill the tank dark area (positions greater than the bright/dark boundary)."""
    if y_min is None or y_max is None:
        y_min, y_max = ax.get_ylim()
    x_right = max(ax.get_xlim()[1], boundary_m + 0.01)
    ax.fill_between(
        [boundary_m, x_right],
        y_min,
        y_max,
        facecolor=facecolor,
        alpha=alpha,
        zorder=zorder,
        linewidth=0,
    )


def shade_dark_region_above(
    ax,
    boundary_y: float,
    *,
    y_max: float | None = None,
    x_min: float | None = None,
    x_max: float | None = None,
    facecolor: str = DARK_AREA_FACE_COLOR,
    alpha: float = DARK_AREA_ALPHA,
    zorder: float = 0,
) -> None:
    """Fill the dark tank area on position-over-time plots (y at or above boundary_y)."""
    if x_min is None or x_max is None:
        x_min, x_max = ax.get_xlim()
    if y_max is None:
        y_max = ax.get_ylim()[1]
    ax.fill_between(
        [x_min, x_max],
        boundary_y,
        y_max,
        facecolor=facecolor,
        alpha=alpha,
        zorder=zorder,
        linewidth=0,
    )


def add_bright_dark_boundary(
    ax,
    boundary_m: float,
    *,
    label: str = "Bright/dark boundary",
    linestyle: str = "--",
    color: str = "#333333",
    linewidth: float = 2.0,
) -> None:
    ax.axvline(
        boundary_m,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        label=label,
        zorder=4,
    )


def add_bright_dark_boundary_horizontal(
    ax,
    boundary_y: float,
    *,
    label: str = "Bright/dark boundary",
    linestyle: str = "--",
    color: str = "#333333",
    linewidth: float = 2.0,
) -> None:
    ax.axhline(
        boundary_y,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        label=label,
        zorder=4,
    )


def shade_dark_electrodes(
    ax,
    start_electrode: int,
    *,
    n_electrodes: int | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    facecolor: str = DARK_AREA_FACE_COLOR,
    alpha: float = DARK_AREA_ALPHA,
    zorder: float = 0,
) -> None:
    """Fill the dark tank area on electrode-index plots (electrode start and above)."""
    if n_electrodes is None:
        n_electrodes = int(ax.get_xlim()[1]) + 1
    if y_min is None:
        y_min = ax.get_ylim()[0]
    if y_max is None:
        y_max = ax.get_ylim()[1]
    ax.fill_betweenx(
        [y_min, y_max],
        start_electrode - 0.5,
        n_electrodes - 0.5,
        facecolor=facecolor,
        alpha=alpha,
        zorder=zorder,
        linewidth=0,
    )


def add_dark_electrode_boundary(
    ax,
    start_electrode: int,
    *,
    label: str = "Bright/dark boundary",
    linestyle: str = "--",
    color: str = "#333333",
    linewidth: float = 2.0,
) -> None:
    ax.axvline(
        start_electrode - 0.5,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        label=label,
        zorder=4,
    )
