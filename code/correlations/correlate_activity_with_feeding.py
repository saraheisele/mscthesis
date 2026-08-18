"""Extract feeding times from session Word records and correlate with pulse activity.

Analysis part: feeding correlation (Part 4 of Berlin activity analysis).
Dependencies: data_paths, session_notes_utils, pulse_config; requires .h5 files and
session .docx logs in LAB_DATA_DIR.

Parses feeding timestamps from session Word documents, aligns them with predetected
pulse recordings, and tests whether pulse rates differ around feeding events.
"""

from __future__ import annotations

from pathlib import Path

from path_setup import setup_script_paths

setup_script_paths(__file__)

import re
import warnings
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats

warnings.filterwarnings("ignore")

from correlations.session_notes_utils import (
    DATE_INLINE_RE,
    FEEDING_FLAG_RADIUS_MIN,
    TIME_RE,
    docx_text,
    extract_eellogger_on_times,
    find_docx_for_session,
    infer_recording_start,
    parse_session_date,
    parse_time_on_date,
    session_name_from_h5,
)
from data_paths import (
    DUAL_LINE_START_DATE,
    FEEDING_CORRELATION_DIR,
    H5_DIR,
    LAB_DATA_DIR,
)
from h5_io import get_path_list, get_pulse_block, load_marker_array, open_h5
from presentation_style import apply_presentation_style, pulse_shape_color, save_thesis_figure
from pulse_config import PULSE_TYPES, PULSE_TYPE_DISPLAY_ORDER

OUTPUT_DIR = FEEDING_CORRELATION_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEEDING_LINE_RE = re.compile(
    r"(feed|food|strike|füt|fut|pr[äa]sent|present|fress)", re.IGNORECASE
)
NEGATIVE_CONTEXT_RE = re.compile(
    r"(expecting|before feeding|after feeding|not hungry|without feeding|"
    r"introduced objects before|going after any introduced|filter|cleaning|"
    r"eellogger|camera|video|temperature|temperatur|^T\s*=|^L\s*=|rain|"
    r"reference time|recording time|overnight|electrode|battery|synchron)",
    re.IGNORECASE,
)

PERI_WINDOW_MIN = 10
BASELINE_EXCLUDE_MIN = 5
DAY_MINUTES = 24 * 60
CLOCK_TIME_BOOTSTRAP_ITERATIONS = 10000
CLOCK_TIME_CI_LEVEL = 95
CLOCK_TIME_BOOTSTRAP_SEED = 0


def pulse_count_weight(dt_start: datetime) -> float:
    """Halve pulse counts from 25 Nov 2025, matching activity histogram dual-line weighting."""
    dual_line_start = datetime.strptime(DUAL_LINE_START_DATE, "%Y-%m-%d")
    if dt_start.date() >= dual_line_start.date():
        return 0.5
    return 1.0


def extract_feeding_events(session_name: str, docx_path: Path) -> list[dict]:
    text = docx_text(docx_path)
    default_date = parse_session_date(session_name)
    if default_date is None:
        return []

    current_date = default_date.replace(hour=0, minute=0, second=0, microsecond=0)
    events = []

    for line in text.splitlines():
        inline_date = DATE_INLINE_RE.search(line.replace(" ", ""))
        if inline_date:
            try:
                current_date = datetime.strptime(inline_date.group(1), "%Y%m%d")
            except ValueError:
                pass

        if not FEEDING_LINE_RE.search(line):
            continue
        if NEGATIVE_CONTEXT_RE.search(line) and not re.search(
            r"(feed|food|strike|fress)", line, re.IGNORECASE
        ):
            continue

        times = TIME_RE.findall(line)
        if not times:
            continue

        event_type = "feeding"
        lower = line.lower()
        if re.search(r"present|pr[äa]sent|provided|entered|entered", lower):
            event_type = "food_presentation"
        elif "strike" in lower:
            event_type = "strike"

        for time_str in times:
            event_dt = parse_time_on_date(time_str, current_date)
            if event_dt is None:
                continue
            events.append(
                {
                    "session": session_name,
                    "docx_file": docx_path.name,
                    "event_type": event_type,
                    "event_time": event_dt,
                    "source_line": line.strip(),
                }
            )

    dedup = {}
    for event in events:
        key = (event["session"], event["event_time"], event["event_type"], event["source_line"])
        dedup[key] = event
    return sorted(dedup.values(), key=lambda e: e["event_time"])


def load_pulses_by_type(
    h5_path: Path,
    feeding_events: list[dict] | None = None,
    eellogger_on_times: list[datetime] | None = None,
) -> tuple[dict[str, np.ndarray], float, datetime, float]:
    nix_file = open_h5(h5_path)
    if nix_file is None:
        return {}, 0.0, datetime.min, 0.0

    try:
        block = get_pulse_block(nix_file)
        array_names = {da.name for da in block.data_arrays}
        if "centers" not in array_names:
            return {}, 0.0, datetime.min, 0.0

        centers = block.data_arrays["centers"][:]
        if "predicted_labels" in array_names:
            mask = block.data_arrays["predicted_labels"][:] == 1
        else:
            mask = np.ones(len(centers), dtype=bool)

        pulses = {}
        for key, cfg in PULSE_TYPES.items():
            marker = cfg["array"]
            if marker is None:
                pulses[key] = centers[mask]
            else:
                marker_values = load_marker_array(h5_path, marker, block)
                if marker_values is None:
                    pulses[key] = centers[:0]
                else:
                    pulses[key] = centers[mask][marker_values[mask].astype(bool)]

        meta = nix_file.sections["pulses_metadata"]["metadata"]
        fs = float(meta["samplerate"])
        duration = float(meta["duration"])
        h5_start = datetime.strptime(
            meta["metadata"]["INFO"]["DateTimeOriginal"], "%Y-%m-%dT%H:%M:%S"
        )

        rec_start, _, _ = infer_recording_start(
            h5_path,
            feeding_events=feeding_events,
            eellogger_on_times=eellogger_on_times,
            duration=duration,
            h5_start=h5_start,
        )
        return pulses, fs, rec_start, duration
    finally:
        nix_file.close()


def minute_pulse_rates(
    pulse_samples: np.ndarray, fs: float, rec_start: datetime, duration: float
) -> pd.DataFrame:
    n_bins = int(np.ceil(duration / 60.0))
    counts = np.zeros(n_bins, dtype=float)
    if len(pulse_samples) > 0:
        seconds = pulse_samples / fs
        bin_idx = (seconds // 60).astype(int)
        bin_idx = bin_idx[(bin_idx >= 0) & (bin_idx < n_bins)]
        if len(bin_idx) > 0:
            counts += np.bincount(bin_idx, minlength=n_bins)[:n_bins]
    counts *= pulse_count_weight(rec_start)

    timestamps = [rec_start + timedelta(minutes=i) for i in range(n_bins)]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "pulse_count": counts,
            "pulse_rate_hz": counts / 60.0,
        }
    )


def events_in_recording(events: list[dict], rec_start: datetime, rec_end: datetime) -> list[dict]:
    matched = []
    for event in events:
        t = event["event_time"]
        if rec_start - timedelta(minutes=FEEDING_FLAG_RADIUS_MIN) <= t <= rec_end + timedelta(
            minutes=FEEDING_FLAG_RADIUS_MIN
        ):
            matched.append(event)
    return matched


def add_feeding_flags(minute_df: pd.DataFrame, events: list[dict]) -> pd.DataFrame:
    out = minute_df.copy()
    out["feeding_flag"] = 0
    out["minutes_to_feeding"] = np.nan
    if not events:
        return out

    for idx, row in out.iterrows():
        diffs = [
            (event["event_time"] - row["timestamp"]).total_seconds() / 60.0
            for event in events
        ]
        nearest = min(diffs, key=abs)
        out.at[idx, "minutes_to_feeding"] = nearest
        if abs(nearest) <= FEEDING_FLAG_RADIUS_MIN:
            out.at[idx, "feeding_flag"] = 1
    return out


def peri_event_trajectory(
    pulse_samples: np.ndarray, fs: float, rec_start: datetime, events: list[dict]
) -> np.ndarray | None:
    if len(events) == 0 or len(pulse_samples) == 0:
        return None

    window_s = PERI_WINDOW_MIN * 60
    bin_edges = np.arange(-window_s, window_s + 60, 60)
    trajectories = []

    for event in events:
        offset_s = (event["event_time"] - rec_start).total_seconds()
        rel_seconds = pulse_samples / fs - offset_s
        hist, _ = np.histogram(rel_seconds, bins=bin_edges)
        trajectories.append(hist.astype(float) * pulse_count_weight(rec_start) / 60.0)

    if not trajectories:
        return None
    return np.mean(np.vstack(trajectories), axis=0)


def sig_stars(p: float) -> str:
    if p is None or np.isnan(p):
        return "n.s."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _empty_feeding_corr() -> dict:
    return {
        "n_minutes": 0,
        "n_feeding_minutes": 0,
        "n_nonfeeding_minutes": 0,
        "pearson_r": np.nan,
        "pearson_p": np.nan,
        "spearman_r": np.nan,
        "spearman_p": np.nan,
        "mean_rate_feeding_hz": np.nan,
        "mean_rate_nonfeeding_hz": np.nan,
        "rate_ratio_feeding_over_nonfeeding": np.nan,
        "mannwhitney_u": np.nan,
        "mannwhitney_p": np.nan,
    }


def correlate_feeding_flag(minute_records: pd.DataFrame) -> dict:
    if minute_records.empty or minute_records["feeding_flag"].nunique() < 2:
        out = _empty_feeding_corr()
        out["n_minutes"] = len(minute_records)
        out["n_feeding_minutes"] = (
            int(minute_records["feeding_flag"].sum()) if not minute_records.empty else 0
        )
        return out

    feeding = minute_records["feeding_flag"].astype(float)
    rate = minute_records["pulse_rate_hz"].astype(float)
    pearson_r, pearson_p = stats.pearsonr(feeding, rate)
    spearman_r, spearman_p = stats.spearmanr(feeding, rate)

    feeding_rates = minute_records.loc[minute_records["feeding_flag"] == 1, "pulse_rate_hz"]
    nonfeeding_rates = minute_records.loc[
        (minute_records["feeding_flag"] == 0)
        & (
            minute_records["minutes_to_feeding"].isna()
            | (minute_records["minutes_to_feeding"].abs() > BASELINE_EXCLUDE_MIN)
        ),
        "pulse_rate_hz",
    ]

    mean_feed = feeding_rates.mean() if len(feeding_rates) else np.nan
    mean_non = nonfeeding_rates.mean() if len(nonfeeding_rates) else np.nan
    ratio = mean_feed / mean_non if mean_non and not np.isnan(mean_non) else np.nan

    if len(feeding_rates) > 0 and len(nonfeeding_rates) > 0:
        # Two-sided Mann–Whitney U on minute-level rates (feeding window vs baseline).
        mannwhitney_u, mannwhitney_p = stats.mannwhitneyu(
            feeding_rates.to_numpy(dtype=float),
            nonfeeding_rates.to_numpy(dtype=float),
            alternative="two-sided",
        )
    else:
        mannwhitney_u, mannwhitney_p = np.nan, np.nan

    return {
        "n_minutes": len(minute_records),
        "n_feeding_minutes": int(len(feeding_rates)),
        "n_nonfeeding_minutes": int(len(nonfeeding_rates)),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "mean_rate_feeding_hz": mean_feed,
        "mean_rate_nonfeeding_hz": mean_non,
        "rate_ratio_feeding_over_nonfeeding": ratio,
        "mannwhitney_u": float(mannwhitney_u),
        "mannwhitney_p": float(mannwhitney_p),
    }


def extract_all_feeding_events() -> pd.DataFrame:
    rows = []
    for session_dir in sorted(LAB_DATA_DIR.glob("recordings_*")):
        if not session_dir.is_dir():
            continue
        docx_path = find_docx_for_session(session_dir.name)
        if docx_path is None:
            continue
        for event in extract_feeding_events(session_dir.name, docx_path):
            rows.append(event)

    if not rows:
        return pd.DataFrame(
            columns=["session", "docx_file", "event_type", "event_time", "source_line"]
        )

    df = pd.DataFrame(rows)
    df["event_time"] = pd.to_datetime(df["event_time"])
    return df.sort_values(["session", "event_time"]).reset_index(drop=True)


def minutes_of_day(times: pd.Series) -> np.ndarray:
    t = pd.to_datetime(times)
    return (t.dt.hour * 60 + t.dt.minute + t.dt.second / 60.0).to_numpy(dtype=float)


def circular_mean_minutes(mins: np.ndarray) -> float:
    return float(stats.circmean(np.asarray(mins, dtype=float), high=DAY_MINUTES, low=0.0))


def format_clock(mins: float) -> str:
    m = float(mins) % DAY_MINUTES
    hour = int(m // 60)
    minute = int(round(m - hour * 60))
    if minute == 60:
        hour = (hour + 1) % 24
        minute = 0
    return f"{hour:02d}:{minute:02d}"


def bootstrap_circular_mean_ci(
    mins: np.ndarray,
    n_bootstrap: int = CLOCK_TIME_BOOTSTRAP_ITERATIONS,
    ci_level: int = CLOCK_TIME_CI_LEVEL,
    seed: int = CLOCK_TIME_BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Circular mean of clock minutes with a percentile CI, unwrapped around the point estimate."""
    rng = np.random.default_rng(seed)
    mins = np.asarray(mins, dtype=float)
    point = circular_mean_minutes(mins)
    n = len(mins)
    boots = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        boots[i] = circular_mean_minutes(rng.choice(mins, size=n, replace=True))
    wrapped = (boots - point + DAY_MINUTES / 2.0) % DAY_MINUTES - DAY_MINUTES / 2.0
    alpha = (100 - ci_level) / 2.0
    lo = (point + np.percentile(wrapped, alpha)) % DAY_MINUTES
    hi = (point + np.percentile(wrapped, 100.0 - alpha)) % DAY_MINUTES
    return point, lo, hi


def first_feeding_event_per_session(feeding_df: pd.DataFrame) -> pd.DataFrame:
    if feeding_df.empty:
        return feeding_df
    return (
        feeding_df.sort_values("event_time")
        .groupby("session", as_index=False)
        .first()
        .reset_index(drop=True)
    )


def feeding_clock_time_stats(feeding_df: pd.DataFrame) -> dict:
    """Circular mean clock time of the first logged feeding-related event per session."""
    first = first_feeding_event_per_session(feeding_df)
    if first.empty:
        return {
            "n_sessions": 0,
            "n_events": 0,
            "mean_minutes": np.nan,
            "ci_lo_minutes": np.nan,
            "ci_hi_minutes": np.nan,
            "mean_clock": "",
            "ci_lo_clock": "",
            "ci_hi_clock": "",
            "ci_level": CLOCK_TIME_CI_LEVEL,
            "selection": "first_event_per_session",
        }
    mins = minutes_of_day(first["event_time"])
    point, lo, hi = bootstrap_circular_mean_ci(mins)
    return {
        "n_sessions": int(len(first)),
        "n_events": int(len(feeding_df)),
        "mean_minutes": point,
        "ci_lo_minutes": lo,
        "ci_hi_minutes": hi,
        "mean_clock": format_clock(point),
        "ci_lo_clock": format_clock(lo),
        "ci_hi_clock": format_clock(hi),
        "ci_level": CLOCK_TIME_CI_LEVEL,
        "selection": "first_event_per_session",
    }


def write_feeding_clock_time_summary(stats: dict) -> Path:
    out = OUTPUT_DIR / "feeding_clock_time_summary.csv"
    pd.DataFrame([stats]).to_csv(out, index=False)
    return out


def print_feeding_clock_time(stats: dict) -> None:
    print("\n=== Feeding clock time (circular mean, first event per session) ===")
    if not stats["n_sessions"]:
        print("No feeding events extracted.")
        return
    print(
        f"mean={stats['mean_clock']}  "
        f"bootstrap {stats['ci_level']}% CI {stats['ci_lo_clock']}–{stats['ci_hi_clock']}  "
        f"(n={stats['n_sessions']} sessions, {stats['n_events']} extracted events)"
    )


def _pulse_h5_files() -> list[Path]:
    return [p for p in get_path_list(H5_DIR) if p.name.endswith("_pulses.h5")]


def run_analysis():
    apply_presentation_style()
    print("Extracting feeding times from Word documents...")
    feeding_df = extract_all_feeding_events()
    feeding_df.to_csv(OUTPUT_DIR / "feeding_events_extracted.csv", index=False)
    write_session_feeding_summary(feeding_df)
    clock_stats = feeding_clock_time_stats(feeding_df)
    write_feeding_clock_time_summary(clock_stats)
    n_sessions = feeding_df["session"].nunique() if not feeding_df.empty else 0
    print(f"Extracted {len(feeding_df)} feeding-related events from {n_sessions} sessions")

    session_events = {
        session: group.to_dict("records")
        for session, group in feeding_df.groupby("session")
    }
    session_eellogger_on = {}
    for session_dir in sorted(LAB_DATA_DIR.glob("recordings_*")):
        docx_path = find_docx_for_session(session_dir.name)
        if docx_path is None:
            continue
        session_eellogger_on[session_dir.name] = extract_eellogger_on_times(
            session_dir.name, docx_path
        )

    h5_files = _pulse_h5_files()
    print(f"Processing {len(h5_files)} h5 files...")

    minute_rows = []
    h5_summary_rows = []
    peri_curves = {key: [] for key in PULSE_TYPES}

    for h5_path in h5_files:
        session_name = session_name_from_h5(h5_path)
        events = session_events.get(session_name, [])
        eellogger_on = session_eellogger_on.get(session_name, [])
        rec_start, duration, start_method = infer_recording_start(
            h5_path, feeding_events=events, eellogger_on_times=eellogger_on
        )
        rec_end = rec_start + timedelta(seconds=duration)
        matched_events = events_in_recording(events, rec_start, rec_end)

        pulses_by_type, fs, _, _ = load_pulses_by_type(
            h5_path, feeding_events=events, eellogger_on_times=eellogger_on
        )
        if not pulses_by_type or "all" not in pulses_by_type:
            continue

        for pulse_type, pulse_samples in pulses_by_type.items():
            minute_df = minute_pulse_rates(pulse_samples, fs, rec_start, duration)
            minute_df = add_feeding_flags(minute_df, matched_events)
            minute_df["h5_file"] = h5_path.name
            minute_df["session"] = session_name
            minute_df["pulse_type"] = pulse_type
            minute_df["recording_start"] = rec_start
            minute_df["recording_end"] = rec_end
            minute_df["n_feeding_events_in_recording"] = len(matched_events)
            minute_rows.append(minute_df)

            if pulse_type == "all":
                corr = correlate_feeding_flag(minute_df)
                h5_summary_rows.append(
                    {
                        "h5_file": h5_path.name,
                        "session": session_name,
                        "recording_start": rec_start,
                        "recording_end": rec_end,
                        "start_time_method": start_method,
                        "n_feeding_events_in_recording": len(matched_events),
                        **corr,
                    }
                )

            curve = peri_event_trajectory(pulse_samples, fs, rec_start, matched_events)
            if curve is not None:
                peri_curves[pulse_type].append(curve)

    minute_records = pd.concat(minute_rows, ignore_index=True) if minute_rows else pd.DataFrame()
    minute_records.to_csv(OUTPUT_DIR / "minute_pulse_rates_with_feeding_flags.csv", index=False)

    h5_summary = pd.DataFrame(h5_summary_rows)
    h5_summary.to_csv(OUTPUT_DIR / "h5_feeding_correlation_summary.csv", index=False)

    corr_rows = []
    for pulse_type in PULSE_TYPES:
        subset = minute_records[minute_records["pulse_type"] == pulse_type]
        corr = correlate_feeding_flag(subset)
        corr_rows.append({"pulse_type": pulse_type, **corr})
    corr_summary = pd.DataFrame(corr_rows)
    corr_summary.to_csv(OUTPUT_DIR / "global_feeding_correlation_by_pulse_type.csv", index=False)

    matched_events_df = []
    for h5_path in h5_files:
        session_name = session_name_from_h5(h5_path)
        events = session_events.get(session_name, [])
        eellogger_on = session_eellogger_on.get(session_name, [])
        rec_start, duration, start_method = infer_recording_start(
            h5_path, feeding_events=events, eellogger_on_times=eellogger_on
        )
        rec_end = rec_start + timedelta(seconds=duration)
        events = session_events.get(session_name, [])
        for event in events_in_recording(events, rec_start, rec_end):
            matched_events_df.append(
                {
                    **event,
                    "h5_file": h5_path.name,
                    "recording_start": rec_start,
                    "recording_end": rec_end,
                    "seconds_from_recording_start": (
                        event["event_time"] - rec_start
                    ).total_seconds(),
                    "start_time_method": start_method,
                }
            )
    pd.DataFrame(matched_events_df).to_csv(
        OUTPUT_DIR / "feeding_events_matched_to_h5.csv", index=False
    )
    write_coverage_report(feeding_df, matched_events_df, h5_summary, h5_files)

    save_peri_curves(peri_curves)
    plot_peri_feeding_curves(peri_curves)
    plot_feeding_vs_nonfeeding_rates(corr_summary)
    print_summary(feeding_df, h5_summary, corr_summary)
    print_feeding_clock_time(clock_stats)
    print(f"Results saved to {OUTPUT_DIR}")


def rebuild_peri_curves_from_minute_csv(
    minute_csv: Path | None = None,
    matched_csv: Path | None = None,
) -> dict[str, list[np.ndarray]]:
    """Approximate event-aligned peri curves from minute-binned rates (no H5 rescan).

    Each matched feeding event contributes one trajectory of length
    ``2 * PERI_WINDOW_MIN`` by reading the nearest minute bin to
    ``event_time + relative_minute``.
    """
    minute_csv = minute_csv or (OUTPUT_DIR / "minute_pulse_rates_with_feeding_flags.csv")
    matched_csv = matched_csv or (OUTPUT_DIR / "feeding_events_matched_to_h5.csv")
    minute = pd.read_csv(minute_csv, parse_dates=["timestamp"])
    matched = pd.read_csv(matched_csv, parse_dates=["event_time"])
    if matched.empty or minute.empty:
        return {key: [] for key in PULSE_TYPES}

    x = np.arange(-PERI_WINDOW_MIN, PERI_WINDOW_MIN)
    peri_curves = {key: [] for key in PULSE_TYPES}
    max_offset_s = 45.0

    for event in matched.itertuples(index=False):
        h5_name = event.h5_file
        event_time = pd.Timestamp(event.event_time)
        for pulse_type in PULSE_TYPES:
            sub = minute[
                (minute["h5_file"] == h5_name) & (minute["pulse_type"] == pulse_type)
            ]
            if sub.empty:
                continue
            timestamps = sub["timestamp"].to_numpy()
            rates = sub["pulse_rate_hz"].to_numpy(dtype=float)
            curve = np.zeros(len(x), dtype=float)
            n_hit = 0
            for i, rel in enumerate(x):
                target = np.datetime64(event_time + pd.Timedelta(minutes=int(rel)))
                diffs_s = np.abs(
                    timestamps.astype("datetime64[s]") - target
                ).astype("timedelta64[s]").astype(float)
                j = int(np.argmin(diffs_s))
                if diffs_s[j] <= max_offset_s:
                    curve[i] = rates[j]
                    n_hit += 1
            if n_hit > 0:
                peri_curves[pulse_type].append(curve)
    return peri_curves


def replot_peri_feeding_from_saved() -> None:
    """Rebuild the peri-feeding panel figure from NPZ, else from minute CSV."""
    apply_presentation_style()
    peri_curves = load_peri_curves()
    if not any(peri_curves.values()):
        print("No NPZ peri stacks found; rebuilding from minute CSV...")
        peri_curves = rebuild_peri_curves_from_minute_csv()
        save_peri_curves(peri_curves)
    if not any(peri_curves.values()):
        raise FileNotFoundError(f"Could not build peri curves from {OUTPUT_DIR}")
    plot_peri_feeding_curves(peri_curves)
    print(f"Replotted peri-feeding trajectories to {OUTPUT_DIR}")


def write_session_feeding_summary(feeding_df: pd.DataFrame):
    if feeding_df.empty:
        return

    summary_rows = []
    for session, group in feeding_df.groupby("session"):
        times = "; ".join(
            f"{row.event_time.strftime('%Y-%m-%d %H:%M:%S')} ({row.event_type})"
            for row in group.itertuples()
        )
        summary_rows.append(
            {
                "session": session,
                "docx_file": group.iloc[0]["docx_file"],
                "n_feeding_events": len(group),
                "feeding_times": times,
            }
        )
    pd.DataFrame(summary_rows).sort_values("session").to_csv(
        OUTPUT_DIR / "feeding_times_by_session.csv", index=False
    )


def write_coverage_report(
    feeding_df: pd.DataFrame,
    matched_events: list[dict],
    h5_summary: pd.DataFrame,
    h5_files: list[Path] | None = None,
):
    matched_df = pd.DataFrame(matched_events)
    if h5_files is None:
        h5_files = _pulse_h5_files()
    rows = []
    for session in sorted(feeding_df["session"].unique()):
        n_events = int((feeding_df["session"] == session).sum())
        n_matched = (
            int((matched_df["session"] == session).sum()) if not matched_df.empty else 0
        )
        rows.append(
            {
                "session": session,
                "n_feeding_events_in_docx": n_events,
                "n_feeding_events_in_h5_window": n_matched,
                "n_h5_files_for_session": sum(
                    1 for p in h5_files if p.name.startswith(f"{session}")
                ),
                "feeding_covered_by_h5": n_matched > 0,
            }
        )
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "feeding_h5_coverage_report.csv", index=False)


def _peri_mean_sem(curves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.vstack(curves)
    mean_curve = np.mean(stacked, axis=0)
    # SEM across feeding-event trajectories (one curve per matched feeding event).
    sem = stats.sem(stacked, axis=0) if len(curves) > 1 else np.zeros_like(mean_curve)
    return mean_curve, sem


def save_peri_curves(peri_curves: dict[str, list[np.ndarray]]) -> Path | None:
    """Persist peri-event stacks so trajectories can be replotted without rescanning H5."""
    payload = {}
    for pulse_type, curves in peri_curves.items():
        if not curves:
            continue
        payload[pulse_type] = np.vstack(curves)
    if not payload:
        return None
    out = OUTPUT_DIR / "peri_feeding_pulse_rate_trajectories.npz"
    np.savez_compressed(out, **payload)
    return out


def load_peri_curves(path: Path | None = None) -> dict[str, list[np.ndarray]]:
    path = path or (OUTPUT_DIR / "peri_feeding_pulse_rate_trajectories.npz")
    if not path.exists():
        return {key: [] for key in PULSE_TYPES}
    with np.load(path) as data:
        return {
            key: [row for row in data[key]] if key in data.files else []
            for key in PULSE_TYPES
        }


def plot_peri_feeding_curves(peri_curves: dict[str, list[np.ndarray]]):
    """Plot all / wide / double peri-feeding means in separate panels (own y-scales)."""
    apply_presentation_style()
    pulse_order = [key for key in PULSE_TYPE_DISPLAY_ORDER if peri_curves.get(key)]
    if not pulse_order:
        return

    n = len(pulse_order)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3.0 * n), sharex=True)
    if n == 1:
        axes = [axes]
    x = np.arange(-PERI_WINDOW_MIN, PERI_WINDOW_MIN)

    for i, (ax, pulse_type) in enumerate(zip(axes, pulse_order)):
        cfg = PULSE_TYPES[pulse_type]
        color = pulse_shape_color(pulse_type)
        mean_curve, sem = _peri_mean_sem(peri_curves[pulse_type])
        ax.plot(x, mean_curve, color=color)
        ax.fill_between(
            x,
            mean_curve - sem,
            mean_curve + sem,
            alpha=0.2,
            color=color,
            linewidth=0,
        )
        ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_title(cfg["label"], fontsize=14)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.text(
                0.99,
                0.96,
                r"Shaded band: $\pm$ SEM across feeding events",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=plt.rcParams["legend.fontsize"],
            )

    axes[-1].set_xlabel("Minutes relative to feeding event")
    fig.supylabel("Mean pulse rate (Hz)", fontweight="bold")
    fig.suptitle("Average pulse activity around feeding events", y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "peri_feeding_pulse_rate_trajectories.png", dpi=300)
    save_thesis_figure("correlations/peri_feeding_pulse_rate_trajectories.png")
    plt.close()


def plot_feeding_vs_nonfeeding_rates(corr_summary: pd.DataFrame):
    if corr_summary.empty:
        return

    apply_presentation_style()

    labels = [PULSE_TYPES[row["pulse_type"]]["label"] for _, row in corr_summary.iterrows()]
    feeding = corr_summary["mean_rate_feeding_hz"].values
    nonfeeding = corr_summary["mean_rate_nonfeeding_hz"].values
    p_values = corr_summary.get(
        "mannwhitney_p", pd.Series([np.nan] * len(corr_summary))
    ).values

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5.5))
    pulse_types = corr_summary["pulse_type"].tolist()
    for i, pulse_type in enumerate(pulse_types):
        color = pulse_shape_color(pulse_type)
        ax.bar(
            x[i] - width / 2,
            nonfeeding[i],
            width,
            color=color,
            alpha=0.55,
        )
        ax.bar(
            x[i] + width / 2,
            feeding[i],
            width,
            color=color,
        )

    y_max = float(np.nanmax(np.concatenate([feeding, nonfeeding])))
    for i, p in enumerate(p_values):
        bar_top = max(feeding[i], nonfeeding[i])
        # Keep all / wide / double annotations readable despite different bar heights.
        y = bar_top + 0.04 * max(y_max, 1.0)
        ax.plot(
            [x[i] - width / 2, x[i] - width / 2, x[i] + width / 2, x[i] + width / 2],
            [y - 0.015 * y_max, y, y, y - 0.015 * y_max],
            color="black",
            linewidth=1.2,
        )
        ax.text(
            x[i],
            y + 0.01 * max(y_max, 1.0),
            sig_stars(float(p)),
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Mean pulse rate (Hz)")
    ax.set_title("Pulse rate during feeding windows vs baseline")
    ax.set_ylim(0, y_max * 1.22)
    ax.legend(
        handles=[
            Patch(
                facecolor="#c5c5c5",
                edgecolor="none",
                label=f"Non-feeding (>{BASELINE_EXCLUDE_MIN} min away)",
            ),
            Patch(
                facecolor="#555555",
                edgecolor="none",
                label=f"Feeding (±{FEEDING_FLAG_RADIUS_MIN} min)",
            ),
        ],
        title=(
            "Mann–Whitney U (two-sided):\n"
            "*** $p<0.001$, ** $p<0.01$, * $p<0.05$"
        ),
    )
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feeding_vs_nonfeeding_pulse_rates.png", dpi=300)
    save_thesis_figure("correlations/feeding_vs_nonfeeding_pulse_rates.png")
    plt.close()


def recompute_feeding_summary_from_minute_csv(
    minute_csv: Path | None = None,
) -> pd.DataFrame:
    """Rebuild global feeding-vs-baseline summary (incl. MWU) from saved minute rates."""
    minute_csv = minute_csv or (OUTPUT_DIR / "minute_pulse_rates_with_feeding_flags.csv")
    minute_records = pd.read_csv(minute_csv)
    corr_rows = []
    for pulse_type in PULSE_TYPES:
        subset = minute_records[minute_records["pulse_type"] == pulse_type]
        corr = correlate_feeding_flag(subset)
        corr_rows.append({"pulse_type": pulse_type, **corr})
    corr_summary = pd.DataFrame(corr_rows)
    corr_summary.to_csv(OUTPUT_DIR / "global_feeding_correlation_by_pulse_type.csv", index=False)
    return corr_summary


def replot_feeding_bars_from_saved() -> None:
    """Recompute MWU summary from minute CSV and redraw the feeding-vs-baseline bars."""
    apply_presentation_style()
    corr_summary = recompute_feeding_summary_from_minute_csv()
    plot_feeding_vs_nonfeeding_rates(corr_summary)
    print("\n=== Feeding vs baseline (Mann–Whitney U) ===")
    for _, row in corr_summary.iterrows():
        label = PULSE_TYPES[row["pulse_type"]]["label"]
        print(
            f"{label:16s}  feeding={row['mean_rate_feeding_hz']:.3f} Hz, "
            f"non-feeding={row['mean_rate_nonfeeding_hz']:.3f} Hz, "
            f"ratio={row['rate_ratio_feeding_over_nonfeeding']:.2f}, "
            f"MWU p={row['mannwhitney_p']:.4g} {sig_stars(row['mannwhitney_p'])}"
        )
    print(f"Replotted feeding bars to {OUTPUT_DIR}")


def summarize_feeding_clock_time_from_saved(events_csv: Path | None = None) -> dict:
    """Recompute circular mean feeding clock time from the extracted-events CSV."""
    events_csv = events_csv or (OUTPUT_DIR / "feeding_events_extracted.csv")
    feeding_df = pd.read_csv(events_csv, parse_dates=["event_time"])
    stats = feeding_clock_time_stats(feeding_df)
    write_feeding_clock_time_summary(stats)
    print_feeding_clock_time(stats)
    print(f"Wrote {OUTPUT_DIR / 'feeding_clock_time_summary.csv'}")
    return stats


def print_summary(feeding_df: pd.DataFrame, h5_summary: pd.DataFrame, corr_summary: pd.DataFrame):
    print("\n=== Feeding extraction summary ===")
    print(f"Sessions with docx records: {feeding_df['session'].nunique() if not feeding_df.empty else 0}")
    print(f"Total extracted events: {len(feeding_df)}")
    if not feeding_df.empty:
        print(feeding_df["event_type"].value_counts().to_string())

    print("\n=== H5 overlap summary ===")
    if not h5_summary.empty:
        with_events = h5_summary[h5_summary["n_feeding_events_in_recording"] > 0]
        print(f"H5 files with feeding events in recording window: {len(with_events)} / {len(h5_summary)}")
        print(
            f"Total feeding events matched to a recording: "
            f"{with_events['n_feeding_events_in_recording'].sum():.0f}"
        )

    print("\n=== Global feeding correlation (minute-level feeding flag vs pulse rate) ===")
    if not corr_summary.empty:
        for _, row in corr_summary.iterrows():
            label = PULSE_TYPES[row["pulse_type"]]["label"]
            mwu_p = row.get("mannwhitney_p", np.nan)
            print(
                f"{label:16s}  r={row['pearson_r']:+.3f} (p={row['pearson_p']:.4g}), "
                f"feeding={row['mean_rate_feeding_hz']:.3f} Hz, "
                f"non-feeding={row['mean_rate_nonfeeding_hz']:.3f} Hz, "
                f"ratio={row['rate_ratio_feeding_over_nonfeeding']:.2f}, "
                f"MWU p={mwu_p:.4g} {sig_stars(mwu_p)}"
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replot-peri",
        action="store_true",
        help="Only rebuild peri-feeding panels from saved NPZ (or minute CSV fallback).",
    )
    parser.add_argument(
        "--replot-bars",
        action="store_true",
        help="Recompute feeding-vs-baseline MWU from minute CSV and redraw bar plot.",
    )
    parser.add_argument(
        "--clock-time",
        action="store_true",
        help="Recompute mean feeding clock time ± bootstrap CI from extracted events CSV.",
    )
    args = parser.parse_args()
    if args.replot_peri:
        replot_peri_feeding_from_saved()
    elif args.replot_bars:
        replot_feeding_bars_from_saved()
    elif args.clock_time:
        summarize_feeding_clock_time_from_saved()
    else:
        run_analysis()
