"""Bar charts of RF pulse-shape prevalence over the processed dataset.

Analysis part: special-pulse summary figures.
Dependencies: data_paths, h5_io, mating_notes_utils, presentation_style.

Produces:
  - figures/.../pulse_shapes/pulse_shape_distribution.png
  - figures/.../workinprogress/pulse_shape_distribution_mating_window.png
    (±2 days around mating notes)
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from path_setup import setup_script_paths

setup_script_paths(__file__)

import matplotlib.pyplot as plt
import nixio
import numpy as np
import pandas as pd
from rich.console import Console

from correlations.mating_notes_utils import extract_mating_events
from data_paths import H5_DIR, PULSE_SHAPE_PROTOTYPES_DIR
from h5_io import get_path_list, get_pulse_block, load_marker_array, open_h5
from presentation_style import (
    apply_presentation_style,
    pulse_shape_color,
    save_thesis_figure,
    save_thesis_json,
)
from special_pulses.double_peaks_detection import (
    MULTICLASS_ARRAY_NAME,
    SPECIAL_PULSE_CLASSES,
)

console = Console()

SHAPE_ORDER = ("normal", "wide", "double")
MATING_WINDOW_DAYS = 2
OUTPUT_DIR = PULSE_SHAPE_PROTOTYPES_DIR


def _recording_start(file) -> datetime | None:
    try:
        start_str = file.sections["pulses_metadata"]["metadata"]["metadata"]["INFO"][
            "DateTimeOriginal"
        ]
        return datetime.strptime(start_str, "%Y-%m-%dT%H:%M:%S")
    except (KeyError, TypeError, ValueError):
        return None


def collect_pulse_shape_records(data_path=H5_DIR) -> pd.DataFrame:
    """Collect one row per RF-classified pulse: shape label + timestamp."""
    rows = []
    for file_path in get_path_list(Path(data_path)):
        file = open_h5(file_path, nixio.FileMode.ReadOnly)
        if file is None:
            continue
        try:
            block = get_pulse_block(file)
            names = [da.name for da in block.data_arrays]
            if "centers" not in names or "raw_pulses" not in names:
                continue

            classes = load_marker_array(file_path, MULTICLASS_ARRAY_NAME, block)
            if classes is None:
                continue

            fs = float(file.sections["pulses_metadata"]["metadata"]["samplerate"])
            rec_start = _recording_start(file)
            if rec_start is None:
                continue

            centers = block.data_arrays["centers"][:]
            num_pulses = len(block.data_arrays["raw_pulses"])
            if "predicted_labels" in names:
                candidates = np.where(block.data_arrays["predicted_labels"][:] == 1)[0]
            else:
                candidates = np.arange(num_pulses)

            for pulse_idx in candidates:
                class_id = int(classes[pulse_idx])
                shape = SPECIAL_PULSE_CLASSES.get(class_id)
                if shape not in SHAPE_ORDER:
                    continue
                pulse_time = rec_start + timedelta(
                    seconds=float(centers[pulse_idx]) / fs
                )
                rows.append(
                    {
                        "timestamp": pulse_time,
                        "pulse_shape": shape,
                        "session": file_path.stem.replace("_pulses", ""),
                    }
                )
        finally:
            file.close()

    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def count_shapes(df: pd.DataFrame) -> dict[str, int]:
    counts = {shape: 0 for shape in SHAPE_ORDER}
    if df.empty:
        return counts
    value_counts = df["pulse_shape"].value_counts()
    for shape in SHAPE_ORDER:
        counts[shape] = int(value_counts.get(shape, 0))
    return counts


def filter_mating_window(
    df: pd.DataFrame,
    mating_events: list[dict],
    window_days: float = MATING_WINDOW_DAYS,
) -> pd.DataFrame:
    if df.empty or not mating_events:
        return df.iloc[0:0].copy()

    window = timedelta(days=window_days)
    mask = np.zeros(len(df), dtype=bool)
    timestamps = df["timestamp"].to_numpy()
    for event in mating_events:
        start = np.datetime64(event["event_time"] - window)
        end = np.datetime64(event["event_time"] + window)
        mask |= (timestamps >= start) & (timestamps <= end)
    return df.loc[mask].copy()


def _draw_total_brace(ax, bars, total: int, ymax: float) -> None:
    """Horizontal curly brace above all bars; tip points up at the total-n label."""
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch

    x0 = float(bars[0].get_x())
    x1 = float(bars[-1].get_x() + bars[-1].get_width())
    xm = 0.5 * (x0 + x1)
    y = ymax * 0.88
    h = ymax * 0.05
    tip = ymax * 0.08
    curl = 0.12 * (x1 - x0)
    tip_w = 0.055 * (x1 - x0)

    verts = [
        (x0, y),
        (x0, y + 0.7 * h),
        (x0 + 0.5 * curl, y + h),
        (x0 + curl, y + h),
        (xm - tip_w, y + h),
        (xm, y + h + tip),
        (xm + tip_w, y + h),
        (x1 - curl, y + h),
        (x1 - 0.5 * curl, y + h),
        (x1, y + 0.7 * h),
        (x1, y),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.LINETO,
        MplPath.LINETO,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
    ]
    tip_y = y + h + tip
    ax.add_patch(
        PathPatch(
            MplPath(verts, codes),
            fill=False,
            lw=1.8,
            edgecolor="black",
            clip_on=False,
        )
    )
    ax.text(
        xm,
        tip_y + 0.012 * ymax,
        f"n = {total:,}",
        ha="center",
        va="bottom",
        clip_on=False,
    )


def plot_pulse_shape_distribution(
    counts: dict[str, int],
    *,
    filename: str,
    output_dir: Path,
    y_unit: str = "count",
) -> Path:
    apply_presentation_style()
    labels = [shape.capitalize() for shape in SHAPE_ORDER]
    values = [counts[shape] for shape in SHAPE_ORDER]
    colors = [pulse_shape_color(shape) for shape in SHAPE_ORDER]
    total = sum(values)

    if y_unit == "mio":
        plot_values = [v / 1e6 for v in values]
        ylabel = "Pulse count (Mio)"
        ymax = 10.0
        yticks = list(range(1, 11))
    else:
        plot_values = list(values)
        ylabel = "Pulse count"
        ymax = (max(plot_values) if any(plot_values) else 1.0) * 1.28
        yticks = None

    fig, ax = plt.subplots(figsize=(9, 6.4))
    bars = ax.bar(labels, plot_values, color=colors, edgecolor="black", linewidth=1.0)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Pulse shape")
    # Headroom for bar labels + total brace above the tallest bar.
    ax.set_ylim(0, ymax)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.grid(True, axis="y", alpha=0.3)

    for bar, value, plot_value in zip(bars, values, plot_values):
        pct = 100.0 * value / total if total else 0.0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            plot_value + 0.02 * ymax,
            f"{value:,} ({pct:.1f}%)",
            ha="center",
            va="bottom",
            clip_on=True,
        )

    if total:
        _draw_total_brace(ax, bars, total, ymax)

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / Path(filename).name
    fig.savefig(out, dpi=300)
    save_thesis_figure(filename, fig)
    plt.close(fig)
    console.log(f"Saved {out}")
    return out


def main(data_path=H5_DIR):
    apply_presentation_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    console.log("Collecting RF pulse-shape labels across dataset...")
    df = collect_pulse_shape_records(data_path)
    counts = count_shapes(df)
    console.log(
        "  Counts: "
        + ", ".join(f"{shape}={counts[shape]:,}" for shape in SHAPE_ORDER)
    )

    plot_pulse_shape_distribution(
        counts,
        filename="pulse_shapes/pulse_shape_distribution.png",
        output_dir=OUTPUT_DIR,
        y_unit="mio",
    )

    mating_events = extract_mating_events()
    mating_df = filter_mating_window(df, mating_events)
    mating_counts = count_shapes(mating_df)
    console.log(
        f"  Mating-window (±{MATING_WINDOW_DAYS} d) pulses: {len(mating_df):,} "
        f"around {len(mating_events)} notes"
    )
    plot_pulse_shape_distribution(
        mating_counts,
        filename="workinprogress/pulse_shape_distribution_mating_window.png",
        output_dir=OUTPUT_DIR,
        y_unit="count",
    )

    payload = {
        "overall": counts,
        "mating_window_days": MATING_WINDOW_DAYS,
        "mating_window": mating_counts,
        "n_mating_notes": len(mating_events),
    }
    with open(OUTPUT_DIR / "pulse_shape_distribution_counts.json", "w") as handle:
        json.dump(payload, handle, indent=2)
    save_thesis_json("workinprogress/pulse_shape_distribution_counts.json", payload)
    console.log(f"Wrote counts to {OUTPUT_DIR / 'pulse_shape_distribution_counts.json'}")


if __name__ == "__main__":
    main()
