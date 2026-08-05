"""Pulse property trends over time (KDE, half-width by day/hour, mating-note zoom).

Analysis part: special-pulse temporal analysis.
Dependencies: data_paths, pulse_property_collect, prototype_pulse_plots.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from path_setup import setup_script_paths

setup_script_paths(__file__)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter
from rich.console import Console
from scipy import stats

from correlations.mating_notes_utils import extract_mating_events
from data_paths import (
    HALF_WIDTH_DISTRIBUTIONS_DIR,
    H5_DIR,
    MATING_CORRELATION_DIR,
    PULSE_SHAPE_PROTOTYPES_DIR,
)
from presentation_style import (
    LEGEND_LOC,
    apply_presentation_style,
    pulse_shape_color,
    save_thesis_figure,
)
from special_pulses.prototype_pulse_plots import PULSE_SHAPES
from special_pulses.pulse_property_collect import collect_pulse_property_records

console = Console()
HALF_WIDTH_OUTPUT_DIR = HALF_WIDTH_DISTRIBUTIONS_DIR
MATING_OUTPUT_DIR = MATING_CORRELATION_DIR
MATING_MARKER_COLOR = "crimson"
MAX_IPI_S = 60.0  # ignore gaps longer than this when estimating instantaneous Hz


def plot_half_width_trend(df: pd.DataFrame, output_dir: Path):
    """Weekly median half-width over time, all pulse shapes pooled."""
    sub = df[df["half_width_ms"].notna()].sort_values("timestamp")
    if len(sub) < 20:
        return

    weekly = sub.set_index("timestamp")["half_width_ms"].resample("W")
    median = weekly.median()
    q25 = weekly.quantile(0.25)
    q75 = weekly.quantile(0.75)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(median.index, median.values, color="steelblue", linewidth=2, label="Weekly median")
    ax.fill_between(median.index, q25.values, q75.values, alpha=0.3, color="steelblue", label="Weekly IQR")
    ax.set_ylabel("Half width (ms)")
    ax.set_xlabel("Date")
    ax.set_title("Half-width trend over time (all pulses)")
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "kde_trend_half_width_ms.png", dpi=300)
    plt.close(fig)


def _mating_events_df(mating_notes: pd.DataFrame | list | None) -> pd.DataFrame:
    if mating_notes is None:
        return pd.DataFrame(columns=["event_time"])
    if isinstance(mating_notes, list):
        mating_notes = pd.DataFrame(mating_notes)
    if mating_notes.empty or "event_time" not in mating_notes.columns:
        return pd.DataFrame(columns=["event_time"])
    out = mating_notes.copy()
    out["event_time"] = pd.to_datetime(out["event_time"])
    return out


def _mark_mating_on_period_axis(
    ax,
    months: list,
    mating_notes: pd.DataFrame | list | None,
    *,
    y_frac: float = 0.97,
    label: str = "Mating note",
) -> None:
    """Mark months that contain mating notes (1-based boxplot x positions)."""
    mating = _mating_events_df(mating_notes)
    if mating.empty or not months:
        return

    mating_months = set(mating["event_time"].dt.to_period("M"))
    marked = False
    y_max = ax.get_ylim()[1]
    y_min = ax.get_ylim()[0]
    y = y_min + y_frac * (y_max - y_min)
    for idx, month in enumerate(months, start=1):
        if month not in mating_months:
            continue
        ax.axvline(idx, color=MATING_MARKER_COLOR, linestyle="--", alpha=0.55, linewidth=1.2)
        ax.scatter(
            [idx],
            [y],
            marker="v",
            color=MATING_MARKER_COLOR,
            s=45,
            zorder=6,
            label=label if not marked else None,
        )
        marked = True
    if marked:
        ax.legend(loc=LEGEND_LOC, fontsize=10)


def plot_half_width_monthly(
    df: pd.DataFrame,
    output_dir: Path,
    mating_notes: pd.DataFrame | list | None = None,
):
    """Monthly half-width distributions, all pulse shapes pooled.

    Uses boxplots of the per-pulse half-width distribution (percentiles), so
    unequal recording effort / sample size does not inflate the y-axis — only
    the distribution shape is shown. Sample sizes are annotated under each box.
    """
    sub = df[df["half_width_ms"].notna()].copy()
    if sub.empty:
        return

    sub["month"] = sub["timestamp"].dt.to_period("M")
    months = sorted(sub["month"].unique())
    if not months:
        return

    grouped = [sub.loc[sub["month"] == month, "half_width_ms"].values for month in months]
    counts = [len(vals) for vals in grouped]
    labels = [str(month) for month in months]

    # Equal box widths: distribution percentiles are already sample-size independent.
    fig, ax = plt.subplots(figsize=(max(12, len(months) * 0.45), 5.5))
    ax.boxplot(
        grouped,
        tick_labels=labels,
        showfliers=False,
        patch_artist=True,
        widths=0.55,
        boxprops={"facecolor": "steelblue", "alpha": 0.35},
        medianprops={"color": "black", "linewidth": 1.5},
    )
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Half width (ms)")
    ax.set_xlabel("Month")
    ax.set_title(
        "Monthly half-width distribution (all pulses)\n"
        "boxplot percentiles — normalized for sample size; n annotated"
    )
    ax.grid(True, axis="y", alpha=0.3)

    y0 = ax.get_ylim()[0]
    for idx, n in enumerate(counts, start=1):
        ax.text(idx, y0, f"n={n}", ha="center", va="bottom", fontsize=6, color="0.35")

    _mark_mating_on_period_axis(ax, months, mating_notes)
    plt.tight_layout()
    fig.savefig(output_dir / "half_width_monthly_heatmap.png", dpi=300)
    save_thesis_figure("exploratory_mating_corr/half_width_monthly_heatmap.png", fig)
    plt.close(fig)


def plot_double_pulse_frequency_monthly(
    df: pd.DataFrame,
    output_dir: Path,
    mating_notes: pd.DataFrame | list | None = None,
    *,
    first_month: pd.Period | None = None,
):
    """Monthly distribution of instantaneous double-pulse frequency (Hz).

    Frequency is 1/IPI between successive double pulses (IPI ≤ MAX_IPI_S).
    Same boxplot style as half_width_monthly_heatmap.png; x-axis spans every
    month since recording start (empty months kept for mating markers).
    """
    dbl = df[df["pulse_shape"] == "double"].sort_values("timestamp").copy()
    if dbl.empty:
        return

    dbl["ipi_s"] = dbl["timestamp"].diff().dt.total_seconds()
    valid = dbl[(dbl["ipi_s"] > 0) & (dbl["ipi_s"] <= MAX_IPI_S)].copy()
    if valid.empty:
        return
    valid["freq_hz"] = 1.0 / valid["ipi_s"]
    valid["month"] = valid["timestamp"].dt.to_period("M")

    data_start = df["timestamp"].dt.to_period("M").min() if not df.empty else valid["month"].min()
    data_end = df["timestamp"].dt.to_period("M").max() if not df.empty else valid["month"].max()
    if first_month is not None:
        data_start = first_month

    months = list(pd.period_range(data_start, data_end, freq="M"))
    grouped = []
    positions = []
    counts = []
    for idx, month in enumerate(months, start=1):
        vals = valid.loc[valid["month"] == month, "freq_hz"].values
        if vals.size == 0:
            continue
        grouped.append(vals)
        positions.append(idx)
        counts.append(len(vals))

    if not positions:
        return

    labels = [str(month) for month in months]
    color = pulse_shape_color("double")

    fig, ax = plt.subplots(figsize=(max(12, len(months) * 0.45), 5.5))
    ax.boxplot(
        grouped,
        positions=positions,
        tick_labels=[labels[p - 1] for p in positions],
        showfliers=False,
        patch_artist=True,
        widths=0.55,
        boxprops={"facecolor": color, "alpha": 0.35},
        medianprops={"color": "black", "linewidth": 1.5},
    )
    ax.set_xticks(range(1, len(months) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlim(0.5, len(months) + 0.5)
    ax.set_ylabel("Double-pulse frequency (Hz)")
    ax.set_xlabel("Month since recording start")
    ax.set_title(
        "Monthly double-pulse frequency distribution\n"
        "instantaneous 1/IPI (≤60 s); boxplot percentiles — normalized for sample size"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    y0 = ax.get_ylim()[0]
    for pos, n in zip(positions, counts):
        ax.text(pos, y0, f"n={n}", ha="center", va="bottom", fontsize=6, color="0.35")

    _mark_mating_on_period_axis(ax, months, mating_notes)
    plt.tight_layout()
    fig.savefig(output_dir / "double_pulse_frequency_monthly.png", dpi=300)
    save_thesis_figure("exploratory_mating_corr/double_pulse_frequency_monthly.png", fig)
    plt.close(fig)


def plot_half_width_distribution(df: pd.DataFrame, output_dir: Path):
    """Overall half-width distribution (histogram + KDE), all pulse shapes pooled."""
    apply_presentation_style()
    values = df["half_width_ms"].dropna().values
    if values.size == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    hist_color = pulse_shape_color("all")
    kde_color = pulse_shape_color("normal")
    ax.hist(values, bins=80, density=True, color=hist_color, alpha=0.75, edgecolor="white", linewidth=0.3)
    if values.size >= 50:
        sample = values if values.size <= 50_000 else np.random.default_rng(0).choice(values, size=50_000, replace=False)
        kde_x = np.linspace(sample.min(), sample.max(), 300)
        kde_y = stats.gaussian_kde(sample)(kde_x)
        ax.plot(kde_x, kde_y, color=kde_color, label="KDE")
        ax.legend(loc=LEGEND_LOC)
    ax.set_xlabel("Half width (ms)")
    ax.set_ylabel("Density")
    ax.set_title("Half-width across all pulse shapes")
    ax.grid(True, alpha=0.3)

    fig.savefig(output_dir / "half_width_kde_all_shapes.png", dpi=300)
    save_thesis_figure("pulse_shapes/half_width_kde_all_shapes.png", fig)
    plt.close(fig)


def plot_property_kde(df: pd.DataFrame, property_col: str, output_dir: Path):
    """Rolling KDE of a pulse property over time, per shape."""
    shapes = [k for k in PULSE_SHAPES if k in df["pulse_shape"].unique()]
    if not shapes:
        return

    fig, axes = plt.subplots(len(shapes), 1, figsize=(14, 3.5 * len(shapes)), sharex=True)
    if len(shapes) == 1:
        axes = [axes]

    for ax, shape in zip(axes, shapes):
        sub = df[(df["pulse_shape"] == shape) & df[property_col].notna()].sort_values("timestamp")
        if len(sub) < 20:
            ax.set_title(f"{PULSE_SHAPES[shape]['label']}: insufficient data")
            continue

        # Bin by week and plot median + rolling IQR
        sub = sub.set_index("timestamp")
        weekly = sub[property_col].resample("W").median()
        q25 = sub[property_col].resample("W").quantile(0.25)
        q75 = sub[property_col].resample("W").quantile(0.75)
        ax.plot(weekly.index, weekly.values, color=PULSE_SHAPES[shape]["color"], linewidth=2)
        ax.fill_between(weekly.index, q25.values, q75.values, alpha=0.25, color=PULSE_SHAPES[shape]["color"])
        ax.set_ylabel(property_col.replace("_", " "))
        ax.set_title(f"{PULSE_SHAPES[shape]['label']} — weekly median {property_col}")
        ax.grid(True, alpha=0.3)

    axes[-1].xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    fig.suptitle(f"Temporal trend: {property_col}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / f"kde_trend_{property_col}.png", dpi=300)
    plt.close(fig)


def find_double_pulse_spikes(df: pd.DataFrame, window_days: int = 14) -> list[tuple]:
    """Find periods with elevated double-pulse fraction."""
    if df.empty:
        return []
    daily = df.groupby(df["timestamp"].dt.date)["pulse_shape"].value_counts(normalize=True).unstack(fill_value=0)
    if "double" not in daily.columns:
        return []
    threshold = daily["double"].quantile(0.85)
    spikes = daily[daily["double"] >= threshold]
    periods = []
    for date, row in spikes.iterrows():
        periods.append((datetime.combine(date, datetime.min.time()), float(row["double"])))
    return periods[:10]


def plot_double_pulse_zoom(df: pd.DataFrame, mating_notes: pd.DataFrame, output_dir: Path):
    """Zoom into double-pulse increase periods with mating notes annotated."""
    if df.empty:
        return

    daily_rate = (
        df.assign(date=df["timestamp"].dt.date)
        .groupby("date")["pulse_shape"]
        .apply(lambda s: (s == "double").mean())
        .reset_index(name="double_fraction")
    )
    daily_rate["date"] = pd.to_datetime(daily_rate["date"])

    threshold = daily_rate["double_fraction"].quantile(0.85)
    highlight = daily_rate[daily_rate["double_fraction"] >= threshold]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily_rate["date"], daily_rate["double_fraction"], color=PULSE_SHAPES["double"]["color"])
    ax.scatter(
        highlight["date"],
        highlight["double_fraction"],
        color="red",
        s=40,
        zorder=5,
        label=f"Top 15% double fraction (≥{threshold:.3f})",
    )
    ax.set_ylabel("Fraction double pulses")
    ax.set_xlabel("Date")
    ax.set_title("Daily double-pulse fraction with high-activity periods")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if not mating_notes.empty:
        note_text = "\n".join(
            f"{row.session}: {row.note[:80]}..." for row in mating_notes.head(5).itertuples()
        )
        ax.text(
            0.02, 0.98, f"Mating-related notes:\n{note_text}",
            transform=ax.transAxes, fontsize=7, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    plt.tight_layout()
    fig.savefig(output_dir / "double_pulse_fraction_zoom_mating_notes.png", dpi=300)
    plt.close(fig)

    if len(highlight) > 0:
        zoom_start = highlight["date"].min() - timedelta(days=7)
        zoom_end = highlight["date"].max() + timedelta(days=7)
        fig, ax = plt.subplots(figsize=(12, 5))
        mask = (daily_rate["date"] >= zoom_start) & (daily_rate["date"] <= zoom_end)
        ax.plot(daily_rate.loc[mask, "date"], daily_rate.loc[mask, "double_fraction"], marker="o")
        ax.set_title("Zoom: elevated double-pulse periods")
        ax.set_ylabel("Double fraction")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(output_dir / "double_pulse_fraction_zoom.png", dpi=300)
        plt.close(fig)


def main(data_path=H5_DIR):
    apply_presentation_style()
    HALF_WIDTH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MATING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    console.log("Collecting per-pulse properties with timestamps...")
    df = collect_pulse_property_records(data_path)
    df.to_csv(MATING_OUTPUT_DIR / "pulse_properties_timeseries.csv", index=False)
    console.log(f"  {len(df):,} pulses with properties")

    mating_notes = pd.DataFrame(extract_mating_events())
    mating_notes.to_csv(MATING_OUTPUT_DIR / "mating_notes_from_docx.csv", index=False)

    if df["half_width_ms"].notna().any():
        plot_half_width_trend(df, HALF_WIDTH_OUTPUT_DIR)
        plot_half_width_monthly(df, HALF_WIDTH_OUTPUT_DIR, mating_notes)
        plot_half_width_distribution(df, HALF_WIDTH_OUTPUT_DIR)

    plot_double_pulse_frequency_monthly(df, HALF_WIDTH_OUTPUT_DIR, mating_notes)

    for prop in ("peak_separation_ms", "trough_depth_ratio"):
        if prop in df.columns and df[prop].notna().any():
            plot_property_kde(df, prop, PULSE_SHAPE_PROTOTYPES_DIR)

    plot_double_pulse_zoom(df, mating_notes, MATING_OUTPUT_DIR)

    console.log(
        f"Saved half-width plots to {HALF_WIDTH_OUTPUT_DIR}; "
        f"mating-related outputs to {MATING_OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
