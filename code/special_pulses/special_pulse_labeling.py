"""Interactive special-pulse labeling UIs and plot helpers.

Used for building training/test label sets. Not required for production
apply of a trained classifier.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nixio
import numpy as np
from rich.console import Console

from h5_io import get_path_list, get_pulse_block, open_h5
from waveform_rule_metrics import get_representative_waveform

con = Console()


def _detection():
    """Return the already-loaded detection module (package or flat name)."""
    for name in (
        "special_pulses.double_peaks_detection",
        "double_peaks_detection",
    ):
        mod = sys.modules.get(name)
        if mod is not None and hasattr(mod, "RobustPCA"):
            return mod
    import double_peaks_detection as mod

    return mod


_dpd = _detection()
LABELING_CLASS_ARRAYS = _dpd.LABELING_CLASS_ARRAYS
LABELING_PULSE_CLASSES = _dpd.LABELING_PULSE_CLASSES
RobustPCA = _dpd.RobustPCA
expand_marker_to_all_pulses = _dpd.expand_marker_to_all_pulses
get_default_ml_paths = _dpd.get_default_ml_paths
get_first_available_array = _dpd.get_first_available_array
load_balanced_detector_labeled_pulses = _dpd.load_balanced_detector_labeled_pulses
normalize_waveforms_for_pca = _dpd.normalize_waveforms_for_pca
prepare_classifier_waveforms = _dpd.prepare_classifier_waveforms


LABEL_REF_FS = 24000.0


def normalize_channels_for_label_plot(pulse_data):
    """
    Baseline-correct all channels, flip each to positive dominant polarity,
    and normalize with one shared scale across channels.
    """
    pulse_data = np.asarray(pulse_data, dtype=float)
    corrected = pulse_data.copy()

    baseline_window = max(1, corrected.shape[0] // 5)
    baseline = np.median(corrected[:baseline_window, :], axis=0, keepdims=True)
    corrected -= baseline

    # Flip each channel so its dominant peak is positive.
    for channel_idx in range(corrected.shape[1]):
        trace = corrected[:, channel_idx]
        if abs(np.min(trace)) > np.max(trace):
            corrected[:, channel_idx] = -trace

    scale = np.max(np.abs(corrected))
    if scale == 0:
        scale = 1.0

    return corrected / scale


def apply_label_ref_timebase(records, waveforms=None, ref_fs=LABEL_REF_FS):
    """
    Reinterpret each pulse on a common reference sample rate for labeling/RF.

    Sample values are unchanged; native_fs is kept for provenance. Display and
    saved training features then use ref_fs so 48 kHz and 24 kHz snippets share
    the same plotted duration (×2 stretch for 48 kHz).
    """
    ref_fs = float(ref_fs)
    for record in records:
        native_fs = float(record["fs"])
        record["native_fs"] = native_fs
        record["fs"] = ref_fs
    if waveforms is not None:
        return records, np.asarray(waveforms, dtype=float)
    return records


def compute_label_plot_ylim(records):
    """
    Fixed y-limits for the labeling UI, shared across all pulses in the session.

    Lower bound is capped at -0.3 so polarity-flipped channels stay comparable.
    """
    ymin = 0.0
    ymax = 0.0
    for record in records:
        if "all_channels" in record:
            pulse_data = normalize_channels_for_label_plot(record["all_channels"])
        else:
            pulse_data = np.asarray(record.get("waveform", []), dtype=float)
            if pulse_data.size == 0:
                continue
            if abs(np.min(pulse_data)) > np.max(pulse_data):
                pulse_data = -pulse_data
            scale = np.max(np.abs(pulse_data)) or 1.0
            pulse_data = pulse_data / scale
        ymin = min(ymin, float(np.min(pulse_data)))
        ymax = max(ymax, float(np.max(pulse_data)))

    span = max(abs(ymin), abs(ymax), 1.0)
    margin = 0.05 * span
    lower = max(ymin - margin, -0.3)
    upper = ymax + margin
    return (lower, upper)


def pulse_peak_index(pulse_data, best_channel=None):
    """Return the sample index of the dominant absolute peak."""
    pulse_data = np.asarray(pulse_data, dtype=float)
    if pulse_data.ndim == 2:
        if best_channel is None:
            channel_strengths = np.max(np.abs(pulse_data), axis=0)
            best_channel = int(np.argmax(channel_strengths))
        trace = pulse_data[:, best_channel]
    else:
        trace = pulse_data
    return int(np.argmax(np.abs(trace)))


def peak_aligned_time_ms(n_samples, peak_idx, fs):
    """Time axis in ms with the pulse peak at t = 0."""
    return (np.arange(n_samples) - peak_idx) / fs * 1000


def compute_label_plot_half_window_ms(records, ref_fs=None):
    """Symmetric half-window (ms) for fixed peak-centered labeling axes."""
    half_windows = []
    for record in records:
        fs = float(ref_fs if ref_fs is not None else record["fs"])
        if "all_channels" in record:
            pulse_data = record["all_channels"]
            peak_idx = pulse_peak_index(
                pulse_data, best_channel=record.get("best_channel")
            )
            n_samples = pulse_data.shape[0]
        else:
            waveform = np.asarray(record["waveform"], dtype=float)
            peak_idx = pulse_peak_index(waveform)
            n_samples = len(waveform)

        half_samples = max(peak_idx, n_samples - 1 - peak_idx)
        half_windows.append(half_samples / fs * 1000)

    return max(half_windows) if half_windows else 5.0


def plot_labeling_pulse(
    ax,
    waveform,
    record,
    label_counts,
    current_idx,
    total,
    *,
    label_plot_half_window_ms,
    label_plot_ylim,
):
    ax.clear()
    fs = record["fs"]

    if "all_channels" in record:
        pulse_data = normalize_channels_for_label_plot(record["all_channels"])
        best_channel = record.get("best_channel")
        peak_idx = pulse_peak_index(pulse_data, best_channel=best_channel)
        time_axis = peak_aligned_time_ms(pulse_data.shape[0], peak_idx, fs)

        for channel_idx in range(pulse_data.shape[1]):
            is_best_channel = channel_idx == best_channel
            ax.plot(
                time_axis,
                pulse_data[:, channel_idx],
                linewidth=1.8 if is_best_channel else 0.9,
                alpha=0.95 if is_best_channel else 0.45,
                label=f"ch {channel_idx}" if is_best_channel else None,
            )
    else:
        plot_wave = np.asarray(waveform, dtype=float).copy()
        if abs(np.min(plot_wave)) > np.max(plot_wave):
            plot_wave *= -1
        peak_idx = pulse_peak_index(plot_wave)
        time_axis = peak_aligned_time_ms(len(plot_wave), peak_idx, fs)
        ax.plot(time_axis, plot_wave, linewidth=2.0, color="steelblue")

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.25, linestyle=":")
    ax.set_xlim(-label_plot_half_window_ms, label_plot_half_window_ms)
    ax.set_ylim(label_plot_ylim)
    ax.set_xlabel("Time from peak (ms)")
    ax.set_ylabel("Amplitude (normalized, +polarity)")
    ax.grid(True, alpha=0.3, linestyle="--")

    counts = " | ".join(
        f"{name}: {label_counts.get(label_id, 0)}"
        for label_id, name in LABELING_PULSE_CLASSES.items()
    )
    sampled_as = record.get("sampling_pool", "candidate")
    native_fs = record.get("native_fs", record["fs"])
    fs_note = (
        f"native {native_fs/1000:.0f} kHz→plot {record['fs']/1000:.0f} kHz"
        if abs(float(native_fs) - float(record["fs"])) > 1
        else f"{record['fs']/1000:.0f} kHz"
    )
    title = (
        f"Label pulse {current_idx + 1}/{total} | "
        f"{Path(record['file_path']).name}, pulse {record['pulse_idx']} | "
        f"sampled as: {sampled_as} | {fs_note}\n"
        "[0] normal  [1] wide  [2] double  [S] skip  [Q] finish | "
        f"{counts}"
    )
    ax.set_title(title, fontsize=11, fontweight="bold")


def _pulse_record_key(file_path, pulse_idx):
    return (str(file_path), int(pulse_idx))


def load_excluded_pulse_keys(labels_path):
    """Return set of (file_path, pulse_idx) already labeled (e.g. test set)."""
    labels_path = Path(labels_path)
    if not labels_path.exists():
        return set()
    data = np.load(labels_path, allow_pickle=False)
    records = data["records"]
    return {
        _pulse_record_key(rec["file_path"], rec["pulse_idx"])
        for rec in records
    }


def load_naturalistic_pulses_for_labeling(
    data_path,
    n_pulses=500,
    max_files=40,
    random_seed=42,
    exclude_keys=None,
):
    """
    Sample pulses at natural prevalence for labeling.

    Stratifies across files (≈equal draw per file) so the set is not dominated
    by one recording. Does not use old detector class pools — only
    predicted-positive (or all) pulses, drawn at random.

    exclude_keys : optional set of (file_path, pulse_idx) to skip (e.g. test set).
    """
    path_list = get_path_list(Path(data_path))
    if not path_list:
        raise FileNotFoundError(f"No H5 files under {data_path}")

    exclude_keys = exclude_keys or set()
    rng = np.random.default_rng(random_seed)
    file_order = rng.permutation(len(path_list))
    selected_files = [path_list[int(i)] for i in file_order[: min(max_files, len(path_list))]]

    pools_by_file = []
    n_excluded = 0
    for file_i, file_path in enumerate(selected_files, 1):
        if file_i == 1 or file_i % 20 == 0 or file_i == len(selected_files):
            con.log(
                f"  Scanning candidate indices [{file_i}/{len(selected_files)}] "
                f"{file_path.name}"
            )
        file = open_h5(file_path, nixio.FileMode.ReadOnly)
        if file is None:
            continue
        try:
            block = get_pulse_block(file)
            names = [da.name for da in block.data_arrays]
            if "raw_pulses" not in names:
                continue
            raw_pulses = block.data_arrays["raw_pulses"]
            num_pulses = len(raw_pulses)
            if "predicted_labels" in names:
                candidate_indices = np.where(
                    block.data_arrays["predicted_labels"][:] == 1
                )[0]
            else:
                candidate_indices = np.arange(num_pulses)
            if len(candidate_indices) == 0:
                continue
            fs = float(file.sections["pulses_metadata"]["metadata"]["samplerate"])
            pool = []
            for pulse_idx in candidate_indices:
                key = _pulse_record_key(file_path, pulse_idx)
                if key in exclude_keys:
                    n_excluded += 1
                    continue
                pool.append(
                    {
                        "file_path": str(file_path),
                        "pulse_idx": int(pulse_idx),
                        "fs": float(fs),
                        "sampling_pool": "naturalistic",
                    }
                )
            if pool:
                pools_by_file.append(pool)
        finally:
            file.close()

    if not pools_by_file:
        return np.empty((0, 0)), []

    # ≈equal draw per file, then top up if short
    n_files = len(pools_by_file)
    per_file = max(1, n_pulses // n_files)
    selected_records = []
    leftovers = []
    for pool in pools_by_file:
        take = min(per_file, len(pool))
        chosen = rng.choice(len(pool), size=take, replace=False)
        chosen_set = {int(j) for j in chosen}
        selected_records.extend(pool[int(i)] for i in chosen)
        leftovers.extend(pool[i] for i in range(len(pool)) if i not in chosen_set)

    if len(selected_records) < n_pulses and leftovers:
        need = min(n_pulses - len(selected_records), len(leftovers))
        extra = rng.choice(len(leftovers), size=need, replace=False)
        selected_records.extend(leftovers[int(i)] for i in extra)

    if len(selected_records) > n_pulses:
        keep = rng.choice(len(selected_records), size=n_pulses, replace=False)
        selected_records = [selected_records[int(i)] for i in keep]

    shuffle_order = rng.permutation(len(selected_records))
    selected_records = [selected_records[int(i)] for i in shuffle_order]

    waveforms = []
    records = []
    # Open each file once and fetch only the selected pulse indices.
    # Do NOT cache raw_pulses[:] — full arrays are multi-GB and thrash RAM.
    by_file = {}
    for record in selected_records:
        by_file.setdefault(record["file_path"], []).append(record)

    for file_i, (file_path, file_records) in enumerate(by_file.items(), 1):
        if file_i == 1 or file_i % 20 == 0 or file_i == len(by_file):
            con.log(
                f"  Loading selected waveforms [{file_i}/{len(by_file)}] "
                f"{Path(file_path).name} ({len(file_records)} pulses)"
            )
        nix_file = open_h5(file_path, nixio.FileMode.ReadOnly)
        if nix_file is None:
            continue
        try:
            raw_pulses = get_pulse_block(nix_file).data_arrays["raw_pulses"]
            for record in file_records:
                pulse_data = np.asarray(
                    raw_pulses[int(record["pulse_idx"])][:], dtype=float
                )
                trace, best_channel = get_representative_waveform(pulse_data)
                waveforms.append(trace)
                records.append(
                    {
                        "file_path": record["file_path"],
                        "pulse_idx": record["pulse_idx"],
                        "fs": record["fs"],
                        "best_channel": int(best_channel),
                        "sampling_pool": "naturalistic",
                        "all_channels": np.asarray(pulse_data, dtype=float),
                    }
                )
        finally:
            nix_file.close()

    con.log(
        f"  Naturalistic labeling sample: {len(records)} pulses from "
        f"{len({r['file_path'] for r in records})} files "
        f"(target n={n_pulses}, max_files={max_files}"
        + (f", excluded {n_excluded} already-labeled" if n_excluded else "")
        + ")"
    )
    return np.asarray(waveforms, dtype=float), records


def interactive_label_pulses(
    data_path,
    labels_path=None,
    pulses_per_type=300,
    random_seed=42,
    *,
    sampling="balanced",
    n_pulses=500,
    max_files=40,
    append_existing=True,
    exclude_keys=None,
    n_uncertain=100,
    n_wide_enrich=75,
    n_double_enrich=75,
):
    """
    Prompt the user to label pulses in a matplotlib window.

    Labels:
        0 = normal
        1 = wide
        2 = double

    sampling:
        "balanced" — old detector pools (training-style, class-balanced)
        "naturalistic" — random / file-stratified (test-set style)
        "enrichment" — RF-uncertain + rare-class candidates
    """
    ml_paths = get_default_ml_paths()
    labels_path = Path(labels_path) if labels_path else ml_paths["labels"]
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    con.log("Loading pulse candidates for manual labeling...")
    if sampling == "naturalistic":
        waveforms, records = load_naturalistic_pulses_for_labeling(
            data_path,
            n_pulses=n_pulses,
            max_files=max_files,
            random_seed=random_seed,
            exclude_keys=exclude_keys,
        )
    elif sampling == "enrichment":
        waveforms, records = load_enrichment_pulses_for_labeling(
            data_path,
            n_uncertain=n_uncertain,
            n_wide=n_wide_enrich,
            n_double=n_double_enrich,
            max_files=max_files,
            random_seed=random_seed,
            exclude_keys=exclude_keys,
        )
    elif sampling == "balanced":
        waveforms, records = load_balanced_detector_labeled_pulses(
            data_path, pulses_per_type=pulses_per_type, random_seed=random_seed
        )
    else:
        raise ValueError(f"Unknown sampling mode: {sampling}")

    if len(waveforms) == 0:
        con.log("No pulse candidates found.")
        return None

    use_ref_timebase = sampling in {"naturalistic", "enrichment"}
    if use_ref_timebase:
        records, waveforms = apply_label_ref_timebase(
            records, waveforms, ref_fs=LABEL_REF_FS
        )
        n_stretched = sum(
            1
            for r in records
            if abs(float(r.get("native_fs", r["fs"])) - LABEL_REF_FS) > 1
        )
        con.log(
            f"  Timebase normalized to {LABEL_REF_FS/1000:.0f} kHz for labeling/"
            f"features ({n_stretched}/{len(records)} pulses were higher-rate and "
            "are shown time-stretched × fs/ref_fs)."
        )

    waveforms = normalize_waveforms_for_pca(waveforms)
    labels = np.full(len(waveforms), -1, dtype=np.int64)
    label_plot_half_window_ms = compute_label_plot_half_window_ms(
        records, ref_fs=LABEL_REF_FS if use_ref_timebase else None
    )
    label_plot_ylim = compute_label_plot_ylim(records)

    con.log("\nLabeling instructions:")
    con.log(f"  Sampling mode: {sampling}")
    con.log(f"  Save path: {labels_path}")
    if use_ref_timebase:
        con.log(
            f"  Common plot/feature timebase: {LABEL_REF_FS/1000:.0f} kHz "
            "(48 kHz snippets stretched ×2 so all traces share the x-axis)"
        )
    con.log("  0 = normal/non-special pulse")
    con.log("  1 = wide pulse")
    con.log("  2 = double pulse")
    con.log("  S = skip current pulse")
    con.log("  Q = finish and save labels collected so far")
    con.log(
        f"  Fixed x-axis: ±{label_plot_half_window_ms:.2f} ms from peak "
        "(for width comparison)"
    )
    con.log(
        f"  Fixed y-axis: [{label_plot_ylim[0]:.2f}, {label_plot_ylim[1]:.2f}] "
        "(all channels flipped to +polarity)"
    )

    state = {"idx": 0, "quit": False}
    label_counts = {}

    fig, ax = plt.subplots(figsize=(12, 5))

    def advance():
        while state["idx"] < len(waveforms) and labels[state["idx"]] != -1:
            state["idx"] += 1

        if state["idx"] >= len(waveforms):
            state["quit"] = True
            plt.close(fig)
            return

        plot_labeling_pulse(
            ax,
            waveforms[state["idx"]],
            records[state["idx"]],
            label_counts,
            state["idx"],
            len(waveforms),
            label_plot_half_window_ms=label_plot_half_window_ms,
            label_plot_ylim=label_plot_ylim,
        )
        fig.canvas.draw_idle()

    def on_key_press(event):
        if event.key is None:
            return

        key = event.key.lower()
        if key in {"0", "1", "2"}:
            label = int(key)
            labels[state["idx"]] = label
            label_counts[label] = label_counts.get(label, 0) + 1
            con.log(
                f"  Labeled pulse {state['idx'] + 1}/{len(waveforms)} as "
                f"{LABELING_PULSE_CLASSES[label]}"
            )
            state["idx"] += 1
            advance()
        elif key == "s":
            con.log(f"  Skipped pulse {state['idx'] + 1}/{len(waveforms)}")
            state["idx"] += 1
            advance()
        elif key == "q":
            state["quit"] = True
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key_press)
    advance()
    plt.show(block=True)

    labeled_mask = labels != -1
    if not np.any(labeled_mask):
        con.log("No labels collected.")
        return None

    def _as_pool_records(record_list):
        return np.array(
            [
                (
                    r["file_path"],
                    r["pulse_idx"],
                    r["fs"],
                    r["best_channel"],
                    str(r.get("sampling_pool", "naturalistic")),
                )
                for r in record_list
            ],
            dtype=[
                ("file_path", "U512"),
                ("pulse_idx", "i8"),
                ("fs", "f8"),
                ("best_channel", "i8"),
                ("sampling_pool", "U64"),
            ],
        )

    def _upgrade_records(rec_arr, default_pool="naturalistic"):
        """Ensure records include sampling_pool (older npz files lack it)."""
        rec_arr = np.asarray(rec_arr)
        names = set(rec_arr.dtype.names or ())
        if "sampling_pool" in names:
            return rec_arr.astype(
                [
                    ("file_path", "U512"),
                    ("pulse_idx", "i8"),
                    ("fs", "f8"),
                    ("best_channel", "i8"),
                    ("sampling_pool", "U64"),
                ],
                copy=False,
            )
        return np.array(
            [
                (
                    r["file_path"],
                    r["pulse_idx"],
                    r["fs"],
                    r["best_channel"],
                    default_pool,
                )
                for r in rec_arr
            ],
            dtype=[
                ("file_path", "U512"),
                ("pulse_idx", "i8"),
                ("fs", "f8"),
                ("best_channel", "i8"),
                ("sampling_pool", "U64"),
            ],
        )

    labeled_idx = np.where(labeled_mask)[0]
    save_waveforms = waveforms[labeled_mask]
    save_labels = labels[labeled_mask]
    save_records = _as_pool_records([records[i] for i in labeled_idx])

    if append_existing and labels_path.exists():
        existing = np.load(labels_path, allow_pickle=False)
        existing_records = _upgrade_records(
            existing["records"], default_pool="naturalistic"
        )
        save_waveforms = np.vstack([existing["waveforms"], save_waveforms])
        save_labels = np.concatenate([existing["labels"], save_labels])
        save_records = np.concatenate([existing_records, save_records])

    np.savez_compressed(
        labels_path,
        waveforms=save_waveforms,
        labels=save_labels,
        records=save_records,
    )
    con.log(
        f"Saved {np.sum(labeled_mask)} new labels "
        f"({len(save_labels)} total) to {labels_path}"
    )
    unique, counts = np.unique(save_labels, return_counts=True)
    con.log(
        "Label counts: "
        + ", ".join(
            f"{LABELING_PULSE_CLASSES.get(int(u), u)}={int(c)}"
            for u, c in zip(unique, counts)
        )
    )

    return labels_path


def interactive_label_naturalistic_test_set(
    data_path,
    labels_path=None,
    n_pulses=500,
    max_files=40,
    random_seed=42,
    append_existing=False,
    exclude_train_set=True,
):
    """
    Label a naturalistic held-out test set (random / file-stratified sampling).

    Saved separately from the training label file so it is not used for fitting.
    """
    ml_paths = get_default_ml_paths()
    labels_path = (
        Path(labels_path) if labels_path else ml_paths["naturalistic_test_labels"]
    )
    exclude_keys = set()
    if exclude_train_set:
        exclude_keys |= load_excluded_pulse_keys(ml_paths["naturalistic_train_labels"])
        con.log(
            f"Excluding {len(exclude_keys)} pulses already in naturalistic train set."
        )

    con.log("\n" + "=" * 60)
    con.log("NATURALISTIC TEST-SET LABELING")
    con.log("=" * 60)
    con.log(
        "This set is for evaluating the thresholded RF only — "
        "it will not be used to train the classifier."
    )
    con.log(f"Target size: ~{n_pulses} pulses across up to {max_files} files")
    con.log(f"Output: {labels_path}")
    con.log("=" * 60)

    return interactive_label_pulses(
        data_path,
        labels_path=labels_path,
        random_seed=random_seed,
        sampling="naturalistic",
        n_pulses=n_pulses,
        max_files=max_files,
        append_existing=append_existing,
        exclude_keys=exclude_keys,
    )


def interactive_label_naturalistic_train_set(
    data_path,
    labels_path=None,
    n_pulses=1200,
    max_files=80,
    random_seed=7,
    append_existing=False,
    exclude_test_set=True,
):
    """
    Label a naturalistic training set (file-stratified random sampling).

    Excludes pulses already in the frozen naturalistic test set by default.
    Saved separately from the old detector-balanced training labels.
    """
    ml_paths = get_default_ml_paths()
    labels_path = (
        Path(labels_path) if labels_path else ml_paths["naturalistic_train_labels"]
    )
    exclude_keys = set()
    if exclude_test_set:
        exclude_keys = load_excluded_pulse_keys(ml_paths["naturalistic_test_labels"])
        con.log(
            f"Excluding {len(exclude_keys)} pulses already in naturalistic test set."
        )

    con.log("\n" + "=" * 60)
    con.log("NATURALISTIC TRAINING-SET LABELING")
    con.log("=" * 60)
    con.log(
        "Sampling: random among predicted-positive pulses, stratified across "
        "many H5 files (≈equal count per file). NOT balanced by old detector "
        "class pools — prevalence should match the data."
    )
    con.log(
        "Your frozen naturalistic test set is excluded so there is no train/test "
        "leakage. After labeling, we will train on this file and evaluate on the "
        "test set."
    )
    con.log(f"Target size: ~{n_pulses} pulses across up to {max_files} files")
    con.log(f"Output: {labels_path}")
    con.log("=" * 60)

    return interactive_label_pulses(
        data_path,
        labels_path=labels_path,
        random_seed=random_seed,
        sampling="naturalistic",
        n_pulses=n_pulses,
        max_files=max_files,
        append_existing=append_existing,
        exclude_keys=exclude_keys,
    )


def _load_classifier_for_enrichment(model_path=None):
    ml_paths = get_default_ml_paths()
    model_path = Path(model_path) if model_path else ml_paths["model"]
    if not model_path.exists():
        raise FileNotFoundError(f"Classifier not found at {model_path}")

    import __main__
    setattr(__main__, "RobustPCA", RobustPCA)

    with model_path.open("rb") as f:
        model_data = pickle.load(f)
    return model_data["classifier"], model_data


def load_enrichment_pulses_for_labeling(
    data_path,
    n_uncertain=100,
    n_wide=75,
    n_double=75,
    max_files=60,
    random_seed=11,
    exclude_keys=None,
    model_path=None,
    uncertain_max_proba=0.55,
    uncertain_margin=0.15,
    score_pool_per_file=80,
):
    """
    Build an active-learning + rare-class enrichment labeling sample.

    Pools (excluding already-labeled keys):
      - uncertain: RF max-proba low or top-2 classes nearly tied
      - wide: old detector wide marker and/or high RF P(wide)
      - double: old detector double marker and/or high RF P(double)
    """
    exclude_keys = set(exclude_keys or set())
    path_list = get_path_list(Path(data_path))
    if not path_list:
        raise FileNotFoundError(f"No H5 files under {data_path}")

    rng = np.random.default_rng(random_seed)
    classifier, _ = _load_classifier_for_enrichment(model_path)
    class_ids = [int(c) for c in classifier.classes_]
    id_to_col = {c: i for i, c in enumerate(class_ids)}

    file_order = rng.permutation(len(path_list))
    selected_files = [
        path_list[int(i)] for i in file_order[: min(max_files, len(path_list))]
    ]

    uncertain_pool = []
    wide_pool = []
    double_pool = []

    for file_idx, file_path in enumerate(selected_files, 1):
        con.log(
            f"  Scoring enrichment candidates [{file_idx}/{len(selected_files)}] "
            f"{file_path.name}"
        )
        file = open_h5(file_path, nixio.FileMode.ReadOnly)
        if file is None:
            continue
        try:
            block = get_pulse_block(file)
            names = [da.name for da in block.data_arrays]
            if "raw_pulses" not in names:
                continue

            raw_pulses = block.data_arrays["raw_pulses"]
            num_pulses = len(raw_pulses)
            if "predicted_labels" in names:
                candidate_indices = np.where(
                    block.data_arrays["predicted_labels"][:] == 1
                )[0]
            else:
                candidate_indices = np.arange(num_pulses)
            if len(candidate_indices) == 0:
                continue

            fs = float(file.sections["pulses_metadata"]["metadata"]["samplerate"])

            wide_marker = None
            double_marker = None
            wide_arr, _ = get_first_available_array(
                block, names, LABELING_CLASS_ARRAYS.get(1, ("is_wide_pulse",))
            )
            double_arr, _ = get_first_available_array(
                block, names, LABELING_CLASS_ARRAYS.get(2, ("is_double_peak",))
            )
            if wide_arr is not None:
                wide_marker = expand_marker_to_all_pulses(
                    wide_arr, candidate_indices, num_pulses, "is_wide_pulse"
                )
            if double_arr is not None:
                double_marker = expand_marker_to_all_pulses(
                    double_arr, candidate_indices, num_pulses, "is_double_peak"
                )

            # Detector enrichment candidates (cheap).
            for pulse_idx in candidate_indices:
                key = _pulse_record_key(file_path, pulse_idx)
                if key in exclude_keys:
                    continue
                rec = {
                    "file_path": str(file_path),
                    "pulse_idx": int(pulse_idx),
                    "fs": float(fs),
                    "sampling_pool": "enrichment",
                }
                if double_marker is not None and double_marker[pulse_idx] == 1:
                    double_pool.append({**rec, "sampling_pool": "enrich_double_det"})
                elif wide_marker is not None and wide_marker[pulse_idx] == 1:
                    wide_pool.append({**rec, "sampling_pool": "enrich_wide_det"})

            # RF scoring on a random subset per file for uncertain / soft rare.
            score_n = min(score_pool_per_file, len(candidate_indices))
            score_idx = rng.choice(candidate_indices, size=score_n, replace=False)
            traces = []
            meta = []
            for pulse_idx in score_idx:
                key = _pulse_record_key(file_path, pulse_idx)
                if key in exclude_keys:
                    continue
                pulse_data = np.asarray(raw_pulses[int(pulse_idx)][:], dtype=float)
                trace, best_channel = get_representative_waveform(pulse_data)
                traces.append(trace)
                meta.append(
                    {
                        "file_path": str(file_path),
                        "pulse_idx": int(pulse_idx),
                        "fs": float(fs),
                        "best_channel": int(best_channel),
                        "all_channels": pulse_data,
                    }
                )
            if not traces:
                continue

            X = prepare_classifier_waveforms(np.asarray(traces, dtype=float), classifier)
            proba = classifier.predict_proba(X)
            for i, row in enumerate(proba):
                p_sorted = np.sort(row)[::-1]
                max_p = float(p_sorted[0])
                margin = float(p_sorted[0] - p_sorted[1]) if len(p_sorted) > 1 else 1.0
                p_wide = float(row[id_to_col[1]]) if 1 in id_to_col else 0.0
                p_double = float(row[id_to_col[2]]) if 2 in id_to_col else 0.0
                pred = int(class_ids[int(np.argmax(row))])
                base = {
                    "file_path": meta[i]["file_path"],
                    "pulse_idx": meta[i]["pulse_idx"],
                    "fs": meta[i]["fs"],
                    "best_channel": meta[i]["best_channel"],
                    "all_channels": meta[i]["all_channels"],
                    "rf_max_proba": max_p,
                    "rf_margin": margin,
                    "rf_p_wide": p_wide,
                    "rf_p_double": p_double,
                    "rf_pred": pred,
                }
                if max_p < uncertain_max_proba or margin < uncertain_margin:
                    uncertain_pool.append(
                        {**base, "sampling_pool": "enrich_uncertain"}
                    )
                if pred == 2 or p_double >= 0.25:
                    double_pool.append({**base, "sampling_pool": "enrich_double_rf"})
                elif pred == 1 or p_wide >= 0.30:
                    wide_pool.append({**base, "sampling_pool": "enrich_wide_rf"})
        finally:
            file.close()

    def _dedupe(pool):
        seen = set()
        out = []
        for rec in pool:
            key = _pulse_record_key(rec["file_path"], rec["pulse_idx"])
            if key in seen or key in exclude_keys:
                continue
            seen.add(key)
            out.append(rec)
        return out

    uncertain_pool = _dedupe(uncertain_pool)
    wide_pool = _dedupe(wide_pool)
    double_pool = _dedupe(double_pool)

    # Prefer most uncertain / highest rare-class score when oversampled.
    uncertain_pool.sort(
        key=lambda r: (r.get("rf_max_proba", 1.0), r.get("rf_margin", 1.0))
    )
    wide_pool.sort(key=lambda r: -float(r.get("rf_p_wide", 0.5)))
    double_pool.sort(key=lambda r: -float(r.get("rf_p_double", 0.5)))

    def _take(pool, n, preferred_prefixes=None):
        if not pool or n <= 0:
            return []
        preferred_prefixes = preferred_prefixes or ()
        preferred = []
        other = []
        for r in pool:
            pool_name = str(r.get("sampling_pool", ""))
            if any(pool_name.startswith(p) for p in preferred_prefixes):
                preferred.append(r)
            else:
                other.append(r)
        chosen = []
        for src in (preferred, other):
            need = n - len(chosen)
            if need <= 0:
                break
            if len(src) <= need:
                chosen.extend(src)
            else:
                top = src[: max(need, len(src) // 2)]
                pick = rng.choice(len(top), size=need, replace=False)
                chosen.extend(top[int(i)] for i in pick)
        return chosen

    selected = []
    selected.extend(_take(uncertain_pool, n_uncertain))
    used = {_pulse_record_key(r["file_path"], r["pulse_idx"]) for r in selected}
    wide_pool = [
        r
        for r in wide_pool
        if _pulse_record_key(r["file_path"], r["pulse_idx"]) not in used
    ]
    selected.extend(_take(wide_pool, n_wide, preferred_prefixes=("enrich_wide",)))
    used = {_pulse_record_key(r["file_path"], r["pulse_idx"]) for r in selected}
    double_pool = [
        r
        for r in double_pool
        if _pulse_record_key(r["file_path"], r["pulse_idx"]) not in used
    ]
    selected.extend(_take(double_pool, n_double, preferred_prefixes=("enrich_double",)))

    if not selected:
        return np.empty((0, 0)), []

    shuffle_order = rng.permutation(len(selected))
    selected = [selected[int(i)] for i in shuffle_order]

    waveforms = []
    records = []
    # Open each file once and fetch only selected pulse indices.
    # Do NOT load raw_pulses[:] — full arrays can be multi-GB and thrash memory.
    by_file = {}
    for record in selected:
        by_file.setdefault(record["file_path"], []).append(record)

    for file_path, file_records in by_file.items():
        # Prefer already-scored RF candidates (waveforms in memory).
        need_load = [r for r in file_records if "all_channels" not in r]
        raw_pulses = None
        nix_file = None
        if need_load:
            nix_file = open_h5(file_path, nixio.FileMode.ReadOnly)
            if nix_file is None:
                con.log(f"  Skipping unreadable file during enrichment load: {file_path}")
                continue
            try:
                raw_pulses = get_pulse_block(nix_file).data_arrays["raw_pulses"]
            except Exception as exc:
                con.log(f"  Skipping {Path(file_path).name}: {exc}")
                nix_file.close()
                continue

        try:
            for record in file_records:
                if "all_channels" in record:
                    pulse_data = np.asarray(record["all_channels"], dtype=float)
                    if "best_channel" in record:
                        best_channel = int(record["best_channel"])
                        trace = pulse_data[:, best_channel].copy()
                        if abs(np.min(trace)) > np.max(trace):
                            trace *= -1
                    else:
                        trace, best_channel = get_representative_waveform(pulse_data)
                else:
                    if raw_pulses is None:
                        continue
                    pulse_data = np.asarray(
                        raw_pulses[int(record["pulse_idx"])][:], dtype=float
                    )
                    trace, best_channel = get_representative_waveform(pulse_data)

                waveforms.append(trace)
                records.append(
                    {
                        "file_path": record["file_path"],
                        "pulse_idx": record["pulse_idx"],
                        "fs": record["fs"],
                        "best_channel": int(best_channel),
                        "sampling_pool": record.get("sampling_pool", "enrichment"),
                        "all_channels": np.asarray(pulse_data, dtype=float),
                    }
                )
        finally:
            if nix_file is not None:
                nix_file.close()

    pool_counts = {}
    for r in records:
        pool_counts[r["sampling_pool"]] = pool_counts.get(r["sampling_pool"], 0) + 1
    con.log(
        f"  Enrichment labeling sample: {len(records)} pulses "
        f"from {len({r['file_path'] for r in records})} files | pools={pool_counts}"
    )
    con.log(
        f"  Pool sizes before downsample: uncertain={len(uncertain_pool)}, "
        f"wide={len(wide_pool)}, double={len(double_pool)}"
    )
    return np.asarray(waveforms, dtype=float), records


def interactive_label_enrichment_train_set(
    data_path,
    labels_path=None,
    n_uncertain=100,
    n_wide=75,
    n_double=75,
    max_files=60,
    random_seed=11,
    append_existing=True,
):
    """
    Label uncertain + rare-class enrichment pulses; append to naturalistic train set.
    """
    ml_paths = get_default_ml_paths()
    labels_path = (
        Path(labels_path) if labels_path else ml_paths["naturalistic_train_labels"]
    )
    exclude_keys = load_excluded_pulse_keys(ml_paths["naturalistic_test_labels"])
    exclude_keys |= load_excluded_pulse_keys(labels_path)
    con.log(
        f"Excluding {len(exclude_keys)} already-labeled train/test pulses from enrichment."
    )

    n_total = n_uncertain + n_wide + n_double
    con.log("\n" + "=" * 60)
    con.log("ENRICHMENT TRAINING-SET LABELING")
    con.log("=" * 60)
    con.log(
        "Mix of (1) RF-uncertain pulses and (2) rare-class candidates "
        "(detector markers + elevated RF P(wide)/P(double))."
    )
    con.log(
        f"Targets: uncertain≈{n_uncertain}, wide≈{n_wide}, double≈{n_double} "
        f"(~{n_total} total). Labels append to the naturalistic train set."
    )
    con.log(f"Output: {labels_path}")
    con.log("=" * 60)

    return interactive_label_pulses(
        data_path,
        labels_path=labels_path,
        random_seed=random_seed,
        sampling="enrichment",
        n_pulses=n_total,
        max_files=max_files,
        append_existing=append_existing,
        exclude_keys=exclude_keys,
        n_uncertain=n_uncertain,
        n_wide_enrich=n_wide,
        n_double_enrich=n_double,
    )

