"""Presentation matplotlib style: Paulaner Spezi palette and readable defaults.

Analysis part: shared visualization theme for thesis presentation figures.
Dependencies: matplotlib; optional paulaner_colormaps for the Spezi palette.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

try:
    import paulaner_colormaps as _paulaner_colormaps

    SPEZI_CMAP = _paulaner_colormaps.spezi
except ImportError:
    SPEZI_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
        "spezi",
        ["#4b1e7d", "#e30083", "#e7011c", "#ec6500", "#f9b700"],
    )

_SPEZI_SAMPLE_POSITIONS = np.linspace(0.06, 0.94, 6)


def _sample_spezi_color(index: int) -> str:
    rgba = SPEZI_CMAP(_SPEZI_SAMPLE_POSITIONS[index % len(_SPEZI_SAMPLE_POSITIONS)])
    return mpl.colors.to_hex(rgba)


PULSE_SHAPE_COLORS = {
    "normal": _sample_spezi_color(0),
    "double": _sample_spezi_color(1),
    "wide": _sample_spezi_color(2),
    "fat": _sample_spezi_color(3),
    "all": _sample_spezi_color(4),
}

DARK_AREA_FACE_COLOR = "#808080"
DARK_AREA_ALPHA = 0.28

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
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}

_style_applied = False


def pulse_shape_color(key: str) -> str:
    """Return the presentation color for a pulse-shape category key."""
    return PULSE_SHAPE_COLORS.get(key, _sample_spezi_color(5))


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


def add_bright_dark_boundary(
    ax,
    boundary_m: float,
    *,
    label: str = "bright/dark boundary",
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
