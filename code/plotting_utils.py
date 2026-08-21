"""Shared matplotlib helpers for pulse and position histogram plots.

Analysis part: visualization utilities.
Dependencies: none (callers pass metadata such as first_year).
"""

from datetime import datetime

import numpy as np
from dateutil.relativedelta import relativedelta


def month_since_start_labels(n_bins: int, first_month_year: int, first_month_month: int):
    """Generate month labels for month_since_start timescale."""
    labels = []
    cursor = datetime(first_month_year, first_month_month, 1)
    for _ in range(n_bins):
        labels.append(cursor.strftime("%b %Y"))
        cursor += relativedelta(months=1)
    return labels


def format_x_axis(
    axis,
    timescale: str,
    n_bins: int,
    *,
    first_year: int = 2023,
    first_month_year: int = 2023,
    first_month_month: int = 1,
    data=None,
    thin_ticks: bool = False,
    tick_fontsize: float | None = None,
):
    """Apply timescale-specific tick positions, labels, and x-axis padding.

    If ``thin_ticks`` is True, keep every second tick (helps dense 24 h /
    months-since-start panels).
    """
    x = np.arange(n_bins)

    if timescale == "year" and data is not None:
        valid_indices = np.where(~np.isnan(data) & (data != 0))[0]
        if len(valid_indices) > 0:
            n_bins = valid_indices[-1] + 1
            x = np.arange(n_bins)

    if timescale == "minute":
        tick_positions = np.arange(0, n_bins, 60)
        tick_labels = [f"{h:02d}:00" for h in range(len(tick_positions))]
        xlabel = "Time of day"
    elif timescale == "hour":
        tick_positions = x
        tick_labels = [f"{h:02d}:00" for h in x]
        xlabel = "Time of day"
    elif timescale == "month":
        tick_positions = x
        tick_labels = [datetime(2000, m, 1).strftime("%b") for m in range(1, 13)]
        xlabel = "Month"
    elif timescale == "month_since_start":
        tick_step = max(1, n_bins // 18)
        tick_positions = x[::tick_step]
        all_labels = month_since_start_labels(n_bins, first_month_year, first_month_month)
        tick_labels = [all_labels[i] for i in tick_positions]
        xlabel = "Month since recording start"
    elif timescale == "year":
        tick_positions = x
        tick_labels = [str(y) for y in range(first_year, first_year + n_bins)]
        xlabel = "Year"
    else:
        tick_positions = x
        tick_labels = [str(i) for i in x]
        xlabel = "Bin index"

    if thin_ticks and len(tick_positions) > 2:
        tick_positions = tick_positions[::2]
        tick_labels = tick_labels[::2]

    axis.set_xticks(tick_positions)
    kwargs = {"rotation": 45, "ha": "right"}
    if tick_fontsize is not None:
        kwargs["fontsize"] = tick_fontsize
    axis.set_xticklabels(tick_labels, **kwargs)
    axis.set_xlim(-0.5, n_bins - 0.5)
    axis.set_xlabel(xlabel)
