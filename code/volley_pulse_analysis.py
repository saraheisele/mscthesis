"""Detect high-voltage volleys and test special-pulse enrichment around them.

Analysis part: volley / pulse-shape context analysis (Part 3b).
Dependencies: data_paths; requires predetected .h5 files with pulse markers.

Identifies rapid pulse bursts (volleys), then compares double/wide/fat pulse
proportions before, during, and after volleys against a clean baseline.
"""

from pathlib import Path

import numpy as np
import nixio
from scipy import stats
from scipy.signal import find_peaks

from data_paths import H5_ROOT

VOLLEY_MIN_PULSES = 5
VOLLEY_MAX_ISI_MS = 2.0
VOLLEY_FALLBACK_MIN_PULSES = 3
VOLLEY_FALLBACK_MAX_ISI_MS = 5.0
VOLLEY_CLUSTER_PREFILTER_MS = 15.0
VOLLEY_SUBPEAK_MERGE_MS = 0.5
VOLLEY_CONTEXT_WINDOW_S = 5.0


def load_raw_pulse_data(h5_file_path):
    """
    Load pulse centers and markers from predetected .h5 file.

    Uses the same nixio layout as eel_data_preprocessing.py:
    block ``pulses`` for arrays, section ``pulses_metadata`` for samplerate.

    Args:
        h5_file_path (Path): Path to .h5 file

    Returns:
        tuple: (pulse_centers, pulse_markers_dict, metadata)
               - pulse_centers: Array of pulse center sample indices (predicted positive only)
               - pulse_markers_dict: Dict with marker arrays (double, wide, fat)
               - metadata: (sampling_rate, duration, start_time)
    """
    try:
        with nixio.File.open(str(h5_file_path)) as nix_file:
            if "pulses" not in [b.name for b in nix_file.blocks]:
                return None, None, None

            block = nix_file.blocks["pulses"]
            data_array_names = [da.name for da in block.data_arrays]
            if "centers" not in data_array_names:
                return None, None, None

            centers = block.data_arrays["centers"][:]
            if "predicted_labels" in data_array_names:
                positive_mask = block.data_arrays["predicted_labels"][:] == 1
            else:
                positive_mask = np.ones(len(centers), dtype=bool)

            pulse_centers = centers[positive_mask]
            pulse_markers = {}
            marker_map = {
                "is_double_peak": "double",
                "is_wide_pulse": "wide",
                "is_fat_pulse": "fat",
            }
            for array_name, marker_key in marker_map.items():
                if array_name in data_array_names:
                    pulse_markers[marker_key] = (
                        block.data_arrays[array_name][:][positive_mask].astype(bool)
                    )

            meta_section = nix_file.sections["pulses_metadata"]["metadata"]
            sampling_rate = float(meta_section["samplerate"])
            duration = float(meta_section["duration"])
            start_time_str = meta_section["metadata"]["INFO"]["DateTimeOriginal"]

            return (
                pulse_centers,
                pulse_markers,
                (sampling_rate, duration, start_time_str),
            )
    except Exception as e:
        print(f"Error loading {h5_file_path}: {e}")
        return None, None, None


def _get_representative_waveform(pulse_waveform):
    """Strongest channel waveform (same idea as double_peaks_detection.py)."""
    channel_strength = np.max(np.abs(pulse_waveform), axis=0)
    best_channel = int(np.argmax(channel_strength))
    return pulse_waveform[:, best_channel]


def _merge_event_times(event_times, sampling_rate, tolerance_ms=VOLLEY_SUBPEAK_MERGE_MS):
    if len(event_times) == 0:
        return event_times
    event_times = np.sort(event_times)
    tolerance_samples = int(sampling_rate * tolerance_ms / 1000)
    merged = [int(event_times[0])]
    for sample in event_times[1:]:
        if int(sample) - merged[-1] > tolerance_samples:
            merged.append(int(sample))
    return np.array(merged, dtype=np.int64)


def _find_short_isi_clusters(pulse_centers, sampling_rate, max_isi_ms=VOLLEY_CLUSTER_PREFILTER_MS):
    """Groups of consecutive detected pulses with inter-center ISI <= max_isi_ms."""
    if len(pulse_centers) < 2:
        return []

    pulse_centers = np.sort(pulse_centers)
    position_indices = np.arange(len(pulse_centers))
    isis = np.diff(pulse_centers) / sampling_rate * 1000
    clusters = []
    current = [0]

    for i, isi in enumerate(isis):
        if isi <= max_isi_ms:
            current.append(i + 1)
        else:
            if len(current) >= 2:
                clusters.append(position_indices[current])
            current = [i + 1]

    if len(current) >= 2:
        clusters.append(position_indices[current])

    return clusters


def _cluster_waveform_event_times(
    raw_pulses, all_centers, cluster_positions, positive_indices, sampling_rate
):
    """
    Build a max-amplitude envelope across overlapping snippets in a short-ISI cluster,
    then detect sub-peaks on that envelope.
    """
    half_width = len(raw_pulses[int(positive_indices[0])][:]) // 2
    min_distance = max(1, int(sampling_rate * 0.0005))
    envelope = {}

    for position in cluster_positions:
        pulse_idx = int(positive_indices[position])
        try:
            signal = np.abs(_get_representative_waveform(raw_pulses[pulse_idx][:]))
        except KeyError:
            continue
        base_sample = int(all_centers[pulse_idx]) - half_width
        for offset, amplitude in enumerate(signal):
            sample = base_sample + offset
            envelope[sample] = max(envelope.get(sample, 0.0), amplitude)

    if not envelope:
        return np.array([], dtype=np.int64)

    samples = np.array(sorted(envelope.keys()))
    amplitudes = np.array([envelope[sample] for sample in samples])
    peak_max = amplitudes.max()
    peak_indices, _ = find_peaks(
        amplitudes, height=peak_max * 0.25, distance=min_distance
    )
    return samples[peak_indices]


def extract_waveform_event_times(h5_file_path):
    """
    Derive sub-pulse event times from raw waveform snippets in short-ISI clusters.

    For consecutive detected pulses closer than VOLLEY_CLUSTER_PREFILTER_MS, overlapping
    snippets are merged into one envelope and sub-peaks are detected on that envelope.
    """
    try:
        with nixio.File.open(str(h5_file_path)) as nix_file:
            block = nix_file.blocks["pulses"]
            data_array_names = [da.name for da in block.data_arrays]
            if "raw_pulses" not in data_array_names or "centers" not in data_array_names:
                return None, None

            raw_pulses = block.data_arrays["raw_pulses"]
            all_centers = block.data_arrays["centers"][:]
            if "predicted_labels" in data_array_names:
                positive_indices = np.where(block.data_arrays["predicted_labels"][:] == 1)[0]
            else:
                positive_indices = np.arange(len(all_centers))

            if len(positive_indices) == 0:
                return np.array([], dtype=np.int64), None

            sampling_rate = float(
                nix_file.sections["pulses_metadata"]["metadata"]["samplerate"]
            )
            positive_centers = all_centers[positive_indices]
            clusters = _find_short_isi_clusters(positive_centers, sampling_rate)
            event_times = []

            for cluster in clusters:
                event_times.extend(
                    _cluster_waveform_event_times(
                        raw_pulses,
                        all_centers,
                        cluster,
                        positive_indices,
                        sampling_rate,
                    )
                )

            return _merge_event_times(
                np.array(event_times, dtype=np.int64), sampling_rate
            ), sampling_rate
    except Exception as e:
        print(f"Error extracting waveform events from {h5_file_path}: {e}")
        return None, None


def detect_volleys(
    event_times,
    sampling_rate,
    min_pulses=VOLLEY_MIN_PULSES,
    max_isi_ms=VOLLEY_MAX_ISI_MS,
):
    """
    Detect volleys from a sorted timeline of event sample indices.

    A volley is a run of at least ``min_pulses`` consecutive events where every
    inter-event interval is <= ``max_isi_ms``.
    """
    if event_times is None or len(event_times) < min_pulses:
        return []

    event_times = np.sort(event_times)
    isis = np.diff(event_times) / sampling_rate * 1000

    volleys = []
    current_volley = [0]

    for i, isi in enumerate(isis):
        if isi <= max_isi_ms:
            current_volley.append(i + 1)
        else:
            if len(current_volley) >= min_pulses:
                event_indices = np.array(current_volley)
                volleys.append(
                    {
                        "event_indices": event_indices,
                        "start_sample": int(event_times[event_indices[0]]),
                        "end_sample": int(event_times[event_indices[-1]]),
                        "n_events": len(event_indices),
                        "start_time_s": event_times[event_indices[0]] / sampling_rate,
                        "end_time_s": event_times[event_indices[-1]] / sampling_rate,
                    }
                )
            current_volley = [i + 1]

    if len(current_volley) >= min_pulses:
        event_indices = np.array(current_volley)
        volleys.append(
            {
                "event_indices": event_indices,
                "start_sample": int(event_times[event_indices[0]]),
                "end_sample": int(event_times[event_indices[-1]]),
                "n_events": len(event_indices),
                "start_time_s": event_times[event_indices[0]] / sampling_rate,
                "end_time_s": event_times[event_indices[-1]] / sampling_rate,
            }
        )

    return volleys


def count_volleys_in_files(h5_files, method, min_pulses, max_isi_ms):
    """Count volleys across files for a given detection method."""
    total = 0
    for h5_file in h5_files:
        if method == "waveform":
            event_times, sampling_rate = extract_waveform_event_times(h5_file)
            if event_times is None:
                continue
        else:
            pulse_centers, _, metadata = load_raw_pulse_data(h5_file)
            if pulse_centers is None or len(pulse_centers) == 0:
                continue
            sampling_rate = metadata[0]
            event_times = pulse_centers

        total += len(
            detect_volleys(
                event_times,
                sampling_rate,
                min_pulses=min_pulses,
                max_isi_ms=max_isi_ms,
            )
        )
    return total


def resolve_volley_detection_strategy(h5_files):
    """
    Try detection strategies in order and return the first one that finds volleys.

    Step 1: waveform sub-peaks (>=5 events, ISI <= 2 ms)
    Step 2: pulse centers fallback (>=3 events, ISI <= 5 ms)
    """
    strategies = [
        {
            "step": 1,
            "method": "waveform",
            "min_pulses": VOLLEY_MIN_PULSES,
            "max_isi_ms": VOLLEY_MAX_ISI_MS,
            "label": (
                f"waveform sub-peaks (>={VOLLEY_MIN_PULSES} events, "
                f"ISI <= {VOLLEY_MAX_ISI_MS:g} ms)"
            ),
        },
        {
            "step": 2,
            "method": "centers",
            "min_pulses": VOLLEY_FALLBACK_MIN_PULSES,
            "max_isi_ms": VOLLEY_FALLBACK_MAX_ISI_MS,
            "label": (
                f"pulse centers fallback (>={VOLLEY_FALLBACK_MIN_PULSES} events, "
                f"ISI <= {VOLLEY_FALLBACK_MAX_ISI_MS:g} ms)"
            ),
        },
    ]

    print("\nVolley detection strategy search:")
    print("-" * 70)
    for strategy in strategies:
        count = count_volleys_in_files(
            h5_files,
            strategy["method"],
            strategy["min_pulses"],
            strategy["max_isi_ms"],
        )
        print(
            f"  Step {strategy['step']}: {strategy['label']} -> {count} volleys"
        )
        if count > 0:
            print(f"  Using step {strategy['step']} for pulse-type analysis.")
            return strategy

    print("  No volleys found with any strategy.")
    return None


def detect_file_volleys(h5_file, strategy):
    """Detect volleys in one file using the selected strategy."""
    if strategy["method"] == "waveform":
        event_times, sampling_rate = extract_waveform_event_times(h5_file)
        if event_times is None:
            return [], None
    else:
        pulse_centers, _, metadata = load_raw_pulse_data(h5_file)
        if pulse_centers is None or len(pulse_centers) == 0:
            return [], None
        sampling_rate = metadata[0]
        event_times = pulse_centers

    volleys = detect_volleys(
        event_times,
        sampling_rate,
        min_pulses=strategy["min_pulses"],
        max_isi_ms=strategy["max_isi_ms"],
    )
    return volleys, sampling_rate


def _merge_sample_intervals(intervals):
    """Merge overlapping (start, end) sample intervals."""
    if not intervals:
        return []

    intervals = sorted(intervals, key=lambda item: item[0])
    merged = [intervals[0]]

    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged


def _intervals_duration_s(intervals, sampling_rate):
    if not intervals:
        return 0.0
    return sum((end - start) / sampling_rate for start, end in intervals)


def _pulse_mask_outside_intervals(pulse_centers, intervals):
    mask = np.ones(len(pulse_centers), dtype=bool)
    for start, end in intervals:
        mask &= ~((pulse_centers >= start) & (pulse_centers <= end))
    return mask


def analyze_pulses_around_volleys(
    pulse_centers,
    pulse_markers,
    volleys,
    sampling_rate,
    recording_duration_s=None,
    window_s=VOLLEY_CONTEXT_WINDOW_S,
):
    """
    Analyze special pulse shapes around volleys with rate-corrected baselines.

    Baseline pulses and rates are computed outside +/- ``window_s`` around every
    volley. Before/after contexts use fixed-duration windows; significance tests
    compare each pulse type to the count expected from its clean baseline
    proportion given the total pulses actually observed in that context.
    """
    window_samples = int(window_s * sampling_rate)
    pulse_types = ["double", "wide", "fat"]

    if recording_duration_s is None:
        recording_duration_s = (
            pulse_centers[-1] / sampling_rate if len(pulse_centers) > 0 else 0.0
        )
    duration_samples = max(1, int(recording_duration_s * sampling_rate))

    neighborhood_intervals = _merge_sample_intervals(
        [
            (
                max(0, volley["start_sample"] - window_samples),
                min(duration_samples, volley["end_sample"] + window_samples),
            )
            for volley in volleys
        ]
    )
    baseline_mask = _pulse_mask_outside_intervals(
        pulse_centers, neighborhood_intervals
    )
    baseline_time_s = recording_duration_s - _intervals_duration_s(
        neighborhood_intervals, sampling_rate
    )

    stats_dict = {
        "total_volleys": len(volleys),
        "within": {ptype: 0 for ptype in pulse_types},
        "within_total": 0,
        "within_time_s": 0.0,
        "before": {ptype: 0 for ptype in pulse_types},
        "after": {ptype: 0 for ptype in pulse_types},
        "before_total": 0,
        "after_total": 0,
        "before_time_s": len(volleys) * window_s,
        "after_time_s": len(volleys) * window_s,
        "baseline": {ptype: 0 for ptype in pulse_types},
        "baseline_total": int(np.sum(baseline_mask)),
        "baseline_time_s": baseline_time_s,
    }

    for ptype in pulse_types:
        if ptype in pulse_markers:
            stats_dict["baseline"][ptype] = int(np.sum(pulse_markers[ptype][baseline_mask]))

    for volley in volleys:
        start_sample = volley["start_sample"]
        end_sample = volley["end_sample"]

        before_window = (start_sample - window_samples, start_sample)
        after_window = (end_sample, end_sample + window_samples)

        in_volley = (pulse_centers >= start_sample) & (pulse_centers <= end_sample)
        before_mask = (pulse_centers >= before_window[0]) & (
            pulse_centers < before_window[1]
        )
        after_mask = (pulse_centers >= after_window[0]) & (
            pulse_centers < after_window[1]
        )

        before_mask &= ~in_volley
        after_mask &= ~in_volley

        stats_dict["within_total"] += int(np.sum(in_volley))
        stats_dict["before_total"] += int(np.sum(before_mask))
        stats_dict["after_total"] += int(np.sum(after_mask))
        stats_dict["within_time_s"] += volley["end_time_s"] - volley["start_time_s"]

        for ptype in pulse_types:
            if ptype not in pulse_markers:
                continue
            marker = pulse_markers[ptype]
            stats_dict["within"][ptype] += int(np.sum(marker[in_volley]))
            stats_dict["before"][ptype] += int(np.sum(marker[before_mask]))
            stats_dict["after"][ptype] += int(np.sum(marker[after_mask]))

    return stats_dict


def compare_volley_pulse_type_rate_corrected(all_pulse_stats):
    """
    Test whether each pulse type deviates from its clean baseline rate.

    For each context window, the expected count of a pulse type is:

        expected = N_context * p_type_baseline

    where ``N_context`` is the total number of pulses observed in that window
    (thereby correcting for higher/lower overall pulse rate) and
    ``p_type_baseline`` comes from time outside all volley neighborhoods.
    """
    results = {}
    baseline_total = all_pulse_stats["baseline_total"]
    baseline_time_s = all_pulse_stats["baseline_time_s"]
    baseline_counts = all_pulse_stats["baseline"]
    baseline_overall_rate = (
        baseline_total / baseline_time_s if baseline_time_s > 0 else np.nan
    )

    for context in ["before", "after", "within"]:
        context_total = all_pulse_stats[f"{context}_total"]
        context_time_s = all_pulse_stats[f"{context}_time_s"]
        context_counts = all_pulse_stats[context]
        context_overall_rate = (
            context_total / context_time_s if context_time_s > 0 else np.nan
        )

        for ptype in ["double", "wide", "fat"]:
            baseline_n = baseline_counts[ptype]
            context_n = context_counts[ptype]
            baseline_prop = (
                baseline_n / baseline_total if baseline_total > 0 else np.nan
            )
            baseline_rate = (
                baseline_n / baseline_time_s if baseline_time_s > 0 else np.nan
            )
            context_rate = (
                context_n / context_time_s if context_time_s > 0 else np.nan
            )
            expected_count = (
                context_total * baseline_prop
                if context_total > 0 and not np.isnan(baseline_prop)
                else np.nan
            )
            expected_rate = (
                baseline_prop * context_overall_rate
                if not np.isnan(baseline_prop) and not np.isnan(context_overall_rate)
                else np.nan
            )

            if (
                context_total > 0
                and baseline_total > 0
                and not np.isnan(baseline_prop)
            ):
                p_value = stats.binomtest(
                    context_n, context_total, baseline_prop
                ).pvalue
            else:
                p_value = np.nan

            results[(context, ptype)] = {
                "baseline_rate_hz": baseline_rate,
                "context_rate_hz": context_rate,
                "expected_rate_hz": expected_rate,
                "baseline_prop_pct": 100 * baseline_prop
                if not np.isnan(baseline_prop)
                else np.nan,
                "observed_count": context_n,
                "expected_count": expected_count,
                "overall_rate_ratio": (
                    context_overall_rate / baseline_overall_rate
                    if baseline_overall_rate and not np.isnan(context_overall_rate)
                    else np.nan
                ),
                "fold_vs_expected": (
                    context_n / expected_count
                    if expected_count and expected_count > 0
                    else np.nan
                ),
                "p_value": p_value,
            }

    return results


def print_volley_analysis(volley_stats, strategy=None):
    """Print formatted volley analysis results."""
    print(f"\n{'=' * 70}")
    print("HIGH VOLTAGE VOLLEY ANALYSIS")
    print(f"{'=' * 70}")

    if strategy:
        print(f"\nDetection method: {strategy['label']}")

    print("\nVolley Detection Summary:")
    print(f"  Total volleys detected: {volley_stats['total_volleys']}")

    if volley_stats["total_volleys"] == 0:
        print("  No volleys detected in the data.")
        return

    baseline_total = volley_stats["baseline_total"]
    baseline_time_s = volley_stats["baseline_time_s"]
    baseline_rate = baseline_total / baseline_time_s if baseline_time_s > 0 else np.nan

    print(
        f"\nClean baseline (outside +/- {VOLLEY_CONTEXT_WINDOW_S:.0f}s of all volleys):"
    )
    print(
        f"  {baseline_total} pulses over {baseline_time_s:.1f} s "
        f"({baseline_rate:.2f} Hz overall)"
    )
    print("-" * 70)
    for pulse_type in ["double", "wide", "fat"]:
        count = volley_stats["baseline"][pulse_type]
        pct = 100 * count / baseline_total if baseline_total else 0
        rate = count / baseline_time_s if baseline_time_s > 0 else np.nan
        print(
            f"  {pulse_type.upper():6}: {count:5d} pulses ({pct:5.1f}%)"
            f" | {rate:.3f} Hz"
        )

    corrected = compare_volley_pulse_type_rate_corrected(volley_stats)
    contexts = [
        (
            "before",
            f"BEFORE volleys ({VOLLEY_CONTEXT_WINDOW_S:.0f}s window)",
            "before_total",
            "before_time_s",
        ),
        (
            "after",
            f"AFTER volleys ({VOLLEY_CONTEXT_WINDOW_S:.0f}s window)",
            "after_total",
            "after_time_s",
        ),
        (
            "within",
            "WITHIN volleys",
            "within_total",
            "within_time_s",
        ),
    ]

    print(
        f"\nRate-corrected enrichment (expected count = observed total pulses"
        f" x baseline proportion):"
    )

    for context_key, context_label, total_key, time_key in contexts:
        total = volley_stats[total_key]
        time_s = volley_stats[time_key]
        overall_rate = total / time_s if time_s > 0 else np.nan
        rate_ratio = corrected[(context_key, "double")]["overall_rate_ratio"]

        print(f"\n{context_label}:")
        print("-" * 70)
        print(
            f"  Pulses: {total} over {time_s:.1f} s ({overall_rate:.2f} Hz)"
        )
        if not np.isnan(rate_ratio):
            print(
                f"  Overall pulse rate vs clean baseline: {rate_ratio:.2f}x"
            )
        if total == 0:
            print("  No pulses in this context.")
            continue

        for pulse_type in ["double", "wide", "fat"]:
            result = corrected[(context_key, pulse_type)]
            sig = ""
            if not np.isnan(result["p_value"]) and result["p_value"] < 0.05:
                direction = (
                    "more"
                    if result["fold_vs_expected"] > 1
                    else "fewer"
                )
                sig = (
                    f" * significantly {direction} than expected"
                    f" (p={result['p_value']:.4f})"
                )
            print(
                f"  {pulse_type.upper():6}: observed {result['observed_count']:5d}"
                f" | expected {result['expected_count']:6.1f}"
                f" | fold {result['fold_vs_expected']:.2f}x"
                f" | rate {result['context_rate_hz']:.3f} vs"
                f" {result['baseline_rate_hz']:.3f} Hz baseline{sig}"
            )



def run_volley_analysis(h5_files=None):
    """Detect volleys across h5 files and print pulse-type context statistics."""
    if h5_files is None:
        h5_files = sorted(H5_ROOT.glob("**/*.h5"))

    if not h5_files:
        print("No .h5 files found. Skipping volley analysis.")
        return

    print(f"\n{'=' * 70}")
    print("HIGH-VOLTAGE VOLLEY ANALYSIS")
    print(f"  Pulse-type context window: +/- {VOLLEY_CONTEXT_WINDOW_S:.0f} s")
    print(f"{'=' * 70}")

    strategy = resolve_volley_detection_strategy(h5_files)
    all_pulse_stats = {
        "total_volleys": 0,
        "within": {"double": 0, "wide": 0, "fat": 0},
        "within_total": 0,
        "within_time_s": 0.0,
        "before": {"double": 0, "wide": 0, "fat": 0},
        "after": {"double": 0, "wide": 0, "fat": 0},
        "before_total": 0,
        "after_total": 0,
        "before_time_s": 0.0,
        "after_time_s": 0.0,
        "baseline": {"double": 0, "wide": 0, "fat": 0},
        "baseline_total": 0,
        "baseline_time_s": 0.0,
    }

    if strategy is not None:
        for h5_file in h5_files:
            pulse_centers, pulse_markers, metadata = load_raw_pulse_data(h5_file)
            if pulse_centers is None or len(pulse_centers) == 0:
                continue

            volleys, sampling_rate = detect_file_volleys(h5_file, strategy)
            if sampling_rate is None:
                sampling_rate = metadata[0]
            recording_duration_s = metadata[1] if metadata else None

            if len(volleys) > 0:
                print(
                    f"  {h5_file.name}: {len(pulse_centers)} pulses, "
                    f"{len(volleys)} volleys"
                )

            if volleys and pulse_markers:
                stats = analyze_pulses_around_volleys(
                    pulse_centers,
                    pulse_markers,
                    volleys,
                    sampling_rate,
                    recording_duration_s=recording_duration_s,
                )
                all_pulse_stats["total_volleys"] += stats["total_volleys"]
                all_pulse_stats["within_total"] += stats["within_total"]
                all_pulse_stats["within_time_s"] += stats["within_time_s"]
                all_pulse_stats["before_total"] += stats["before_total"]
                all_pulse_stats["after_total"] += stats["after_total"]
                all_pulse_stats["before_time_s"] += stats["before_time_s"]
                all_pulse_stats["after_time_s"] += stats["after_time_s"]
                all_pulse_stats["baseline_total"] += stats["baseline_total"]
                all_pulse_stats["baseline_time_s"] += stats["baseline_time_s"]
                for context in ["within", "before", "after", "baseline"]:
                    for ptype in ["double", "wide", "fat"]:
                        all_pulse_stats[context][ptype] += stats[context][ptype]

    print_volley_analysis(all_pulse_stats, strategy=strategy)


def main():
    run_volley_analysis()


if __name__ == "__main__":
    main()
