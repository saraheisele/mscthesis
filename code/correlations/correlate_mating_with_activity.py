"""Correlate mating-related lab notes with pulse activity and half-width.

Analysis part: mating correlation.
Dependencies: data_paths, mating_notes_utils, correlate_activity_with_feeding helpers.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from path_setup import setup_script_paths

setup_script_paths(__file__)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from correlations.correlate_activity_with_feeding import (
    extract_eellogger_on_times,
    find_docx_for_session,
    infer_recording_start,
    load_pulses_by_type,
    minute_pulse_rates,
    session_name_from_h5,
)
from correlations.mating_notes_utils import extract_mating_events
from data_paths import H5_DIR, PULSE_PROPERTIES_DIR

OUTPUT_DIR = PULSE_PROPERTIES_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MATING_WINDOW_DAYS = 2
BASELINE_EXCLUDE_DAYS = 2
PULSE_TYPES = {
    "all": {"label": "All pulses", "array": None},
    "double": {"label": "Double pulses", "array": "is_double_peak"},
}


def add_mating_flags(df: pd.DataFrame, mating_events: list[dict]) -> pd.DataFrame:
    out = df.copy()
    out["mating_flag"] = 0
    out["days_to_mating"] = np.nan
    if not mating_events:
        return out

    for idx, row in out.iterrows():
        diffs_days = [
            (event["event_time"] - row["timestamp"]).total_seconds() / 86400.0
            for event in mating_events
        ]
        nearest = min(diffs_days, key=abs)
        out.at[idx, "days_to_mating"] = nearest
        if abs(nearest) <= MATING_WINDOW_DAYS:
            out.at[idx, "mating_flag"] = 1
    return out


def correlate_mating_flag(minute_df: pd.DataFrame) -> dict:
    if minute_df.empty or minute_df["mating_flag"].nunique() < 2:
        return {
            "n_minutes": len(minute_df),
            "n_mating_minutes": int(minute_df["mating_flag"].sum()),
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "mean_rate_mating_hz": np.nan,
            "mean_rate_nonmating_hz": np.nan,
            "rate_ratio_mating_over_nonmating": np.nan,
        }

    mating = minute_df["mating_flag"].astype(float)
    rate = minute_df["pulse_rate_hz"].astype(float)
    pearson_r, pearson_p = stats.pearsonr(mating, rate)

    mating_rates = minute_df.loc[minute_df["mating_flag"] == 1, "pulse_rate_hz"]
    nonmating_rates = minute_df.loc[
        (minute_df["mating_flag"] == 0)
        & (
            minute_df["days_to_mating"].isna()
            | (minute_df["days_to_mating"].abs() > BASELINE_EXCLUDE_DAYS)
        ),
        "pulse_rate_hz",
    ]
    mean_mating = mating_rates.mean() if len(mating_rates) else np.nan
    mean_non = nonmating_rates.mean() if len(nonmating_rates) else np.nan
    ratio = mean_mating / mean_non if mean_non and not np.isnan(mean_non) else np.nan

    return {
        "n_minutes": len(minute_df),
        "n_mating_minutes": int(minute_df["mating_flag"].sum()),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "mean_rate_mating_hz": mean_mating,
        "mean_rate_nonmating_hz": mean_non,
        "rate_ratio_mating_over_nonmating": ratio,
    }


def collect_minute_records(mating_events: list[dict]) -> pd.DataFrame:
    rows = []
    h5_files = sorted(H5_DIR.glob("*_pulses.h5"))
    for h5_path in h5_files:
        session_name = session_name_from_h5(h5_path)
        docx_path = find_docx_for_session(session_name)
        eellogger_on = (
            extract_eellogger_on_times(session_name, docx_path)
            if docx_path is not None
            else []
        )
        try:
            rec_start, duration, _ = infer_recording_start(
                h5_path, feeding_events=[], eellogger_on_times=eellogger_on
            )
        except OSError:
            continue

        for pulse_type, cfg in PULSE_TYPES.items():
            pulses_by_type, fs, _, _ = load_pulses_by_type(
                h5_path,
                feeding_events=[],
                eellogger_on_times=eellogger_on,
            )
            if not pulses_by_type or pulse_type not in pulses_by_type:
                continue

            minute_df = minute_pulse_rates(pulses_by_type[pulse_type], fs, rec_start, duration)
            minute_df = add_mating_flags(minute_df, mating_events)
            minute_df["pulse_type"] = pulse_type
            minute_df["session"] = session_name
            rows.append(minute_df)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    return out


def daily_rates(minute_df: pd.DataFrame) -> pd.DataFrame:
    minute_df = minute_df.copy()
    minute_df["date"] = minute_df["timestamp"].dt.normalize()
    daily = (
        minute_df.groupby("date", as_index=False)
        .agg(
            pulse_rate_hz=("pulse_rate_hz", "mean"),
            mating_flag=("mating_flag", "max"),
        )
        .sort_values("date")
    )
    return daily


def plot_pulse_rate_correlation(
    minute_df: pd.DataFrame,
    mating_events: list[dict],
    pulse_type: str,
    output_path: Path,
):
    cfg = PULSE_TYPES[pulse_type]
    subset = minute_df[minute_df["pulse_type"] == pulse_type]
    if subset.empty:
        return

    corr = correlate_mating_flag(subset)
    daily = daily_rates(subset)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [2, 1]})

    ax = axes[0]
    ax.plot(daily["date"], daily["pulse_rate_hz"], color="tab:blue", linewidth=1.2)
    for event in mating_events:
        ax.axvline(
            event["event_time"],
            color="crimson",
            linestyle="--",
            alpha=0.8,
            linewidth=1,
        )
    ax.set_ylabel("Daily mean pulse rate (Hz)")
    ax.set_title(f"{cfg['label']}: daily pulse rate with mating-note markers")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    labels = [
        f"Outside mating window\n(>{BASELINE_EXCLUDE_DAYS} days away)",
        f"±{MATING_WINDOW_DAYS} days around mating note",
    ]
    values = [corr["mean_rate_nonmating_hz"], corr["mean_rate_mating_hz"]]
    bars = ax.bar(labels, values, color=["tab:gray", "tab:red"], alpha=0.85)
    ax.set_ylabel("Mean pulse rate (Hz)")
    ax.set_title(
        f"Minute-level comparison  "
        f"(r={corr['pearson_r']:+.3f}, p={corr['pearson_p']:.3g}, "
        f"ratio={corr['rate_ratio_mating_over_nonmating']:.2f})"
    )
    ax.grid(True, axis="y", alpha=0.3)
    for bar, value in zip(bars, values):
        if not np.isnan(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def half_width_comparison(
    properties_csv: Path,
    mating_events: list[dict],
    output_path: Path,
) -> dict:
    if not properties_csv.exists():
        return {}

    df = pd.read_csv(properties_csv, parse_dates=["timestamp"])
    df = df[df["half_width_ms"].notna()].copy()
    if df.empty:
        return {}

    overall_mean = df["half_width_ms"].mean()
    window = timedelta(days=MATING_WINDOW_DAYS)
    peri_values = []
    for event in mating_events:
        start = event["event_time"] - window
        end = event["event_time"] + window
        mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
        if mask.any():
            peri_values.append(df.loc[mask, "half_width_ms"].mean())

    peri_mean = float(np.mean(peri_values)) if peri_values else np.nan
    ratio = peri_mean / overall_mean if overall_mean and not np.isnan(peri_mean) else np.nan

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Overall dataset", f"±{MATING_WINDOW_DAYS} days around mating note"]
    values = [overall_mean, peri_mean]
    colors = ["tab:blue", "tab:red"]
    bars = ax.bar(labels, values, color=colors, alpha=0.85)
    ax.set_ylabel("Mean pulse half width (ms)")
    ax.set_title(
        "Average half width around mating mentions vs overall\n"
        f"(ratio mating-window/overall = {ratio:.2f}, n_events={len(peri_values)})"
    )
    ax.grid(True, axis="y", alpha=0.3)
    for bar, value in zip(bars, values):
        if not np.isnan(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return {
        "overall_half_width_ms": overall_mean,
        "mating_window_half_width_ms": peri_mean,
        "half_width_ratio_mating_over_overall": ratio,
        "n_mating_events_with_data": len(peri_values),
    }


def main():
    mating_events = extract_mating_events()
    mating_df = pd.DataFrame(mating_events)
    mating_df.to_csv(OUTPUT_DIR / "mating_notes_from_docx.csv", index=False)
    print(f"Extracted {len(mating_events)} mating-related notes")

    minute_df = collect_minute_records(mating_events)
    if not minute_df.empty:
        minute_df.to_csv(OUTPUT_DIR / "minute_pulse_rates_with_mating_flags.csv", index=False)

    corr_rows = []
    for pulse_type in PULSE_TYPES:
        subset = minute_df[minute_df["pulse_type"] == pulse_type] if not minute_df.empty else pd.DataFrame()
        corr = correlate_mating_flag(subset)
        corr_rows.append({"pulse_type": pulse_type, **corr})
        plot_pulse_rate_correlation(
            minute_df,
            mating_events,
            pulse_type,
            OUTPUT_DIR / f"mating_{pulse_type}_pulse_rate_correlation.png",
        )
    pd.DataFrame(corr_rows).to_csv(OUTPUT_DIR / "mating_pulse_rate_correlation_summary.csv", index=False)

    hw_summary = half_width_comparison(
        OUTPUT_DIR / "pulse_properties_timeseries.csv",
        mating_events,
        OUTPUT_DIR / "mating_half_width_comparison.png",
    )
    if hw_summary:
        pd.DataFrame([hw_summary]).to_csv(
            OUTPUT_DIR / "mating_half_width_comparison_summary.csv", index=False
        )

    print(f"Saved mating correlation plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
