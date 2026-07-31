"""Detect and label special pulse shapes (double, wide) in predetected .h5 files.

Analysis part: pipeline step 1 — special-pulse ML classifier (train / apply).
Dependencies: data_paths, h5_io; writes marker arrays / sidecars for input .h5 files.

Architecture
------------
- ``waveform_rule_metrics`` — rule-based detectors and width/shape metrics
  (legacy / metrics / tuning; ML is the production shape decision).
- ``special_pulse_labeling`` — interactive labeling UIs and plot helpers.
- ``special_pulse_benchmark`` — classifier benchmark and decision-policy tuning.
- This module — production classifier core (RobustPCA, train/apply/predict),
  shared constants, CLI ``main()``, and backward-compatible re-exports.

Default workflow: train/apply a PCA + random forest classifier on manually labeled
examples. Rule-based per-shape detectors remain available via --mode rule-based.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from path_setup import setup_script_paths

setup_script_paths(__file__)

import matplotlib.pyplot as plt
import nixio
import numpy as np
from rich.console import Console
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_paths import H5_DIR, PCA_SPACE_DIR, SPECIAL_PULSE_CLASSIFIER_DIR
from h5_io import (
    get_path_list,
    get_pulse_block,
    load_marker_array,
    open_h5,
    open_h5_readwrite_or_readonly,
    save_marker_sidecar,
)
from presentation_style import LEGEND_LOC, apply_presentation_style, classifier_label_colors
from waveform_rule_metrics import (
    MIN_AMPLITUDE_THRESHOLD,
    check_pulse_shape_gaussian_exponential,
    compute_half_max_width,
    detect_double_pulse,
    detect_pulse as _detect_pulse_by_mode,
    detect_wide_pulse,
    get_representative_waveform,
)

# Initialize console for logging
con = Console()

#################################
############# MODE ##############
#################################

DETECTION_MODE = "double"
# options:
# "double"
# "wide"

MODE_CONFIG = {
    "double": {
        "array_name": "is_double_peak",
        "display_name": "double peak",
    },
    "wide": {
        "array_name": "is_wide_pulse",
        "display_name": "wide pulse",
    },
}

ARRAY_NAME = MODE_CONFIG[DETECTION_MODE]["array_name"]
DISPLAY_NAME = MODE_CONFIG[DETECTION_MODE]["display_name"]


def detect_pulse(pulse_waveform, sample_rate):
    """Unified detector wrapper using DETECTION_MODE."""
    return _detect_pulse_by_mode(pulse_waveform, sample_rate, mode=DETECTION_MODE)


SPECIAL_PULSE_CLASSES = {
    0: "normal",
    1: "wide",
    2: "double",
}

LABELING_PULSE_CLASSES = {
    0: "normal",
    1: "wide",
    2: "double",
}

SPECIAL_CLASS_ARRAYS = {
    1: "is_wide_pulse",
    2: "is_double_peak",
}

LABELING_CLASS_ARRAYS = {
    1: ("is_wide_pulse",),
    2: ("is_double_peak",),
}

MULTICLASS_ARRAY_NAME = "special_pulse_class"

# Max PCA dimensions for classifier pipelines (RobustPCA step).
PCA_MAX_CLASSIFIER_COMPONENTS = 20
# Max PCA dimensions computed for exploratory scatter plots.
PCA_MAX_PLOT_COMPONENTS = 10

# Production decision defaults for rare special pulses.
# Retuned on naturalistic_test_labels.npz (n=486): min_proba wide/double = 0.40
# slightly beats hard predict() on that test set (macro F1 0.724 vs 0.711).
# No rule-based gate: ML is the sole shape decision.
DEFAULT_NATURAL_PRIOR = {
    0: 0.92,  # normal
    1: 0.075,  # wide
    2: 0.005,  # double
}
DEFAULT_MIN_PROBA = {
    1: 0.40,  # wide
    2: 0.40,  # double
}
DEFAULT_DECISION_CONFIG = {
    "use_prior_reweight": False,
    "natural_prior": DEFAULT_NATURAL_PRIOR,
    "min_proba": DEFAULT_MIN_PROBA,
    "default_class": 0,
    "rule_gate_double": False,
    "rf_class_weight": None,
}


class RobustPCA(BaseEstimator, TransformerMixin):
    """
    PCA fit on inlier samples identified by MinCovDet (robust covariance).

    Outliers are excluded from the fit but still projected at transform time.
    """

    def __init__(self, n_components=2, random_state=42, support_fraction=None):
        self.n_components = n_components
        self.random_state = random_state
        self.support_fraction = support_fraction

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        n_components = min(self.n_components, n_samples - 1, n_features)
        n_components = max(1, n_components)
        self.n_components_ = n_components

        fit_mask = self._robust_inlier_mask(X)
        self.n_inliers_ = int(np.sum(fit_mask))
        self.n_outliers_ = int(np.sum(~fit_mask))

        self.pca_ = PCA(n_components=n_components, random_state=self.random_state)
        self.pca_.fit(X[fit_mask])
        return self

    def _robust_inlier_mask(self, X):
        n_samples, n_features = X.shape
        if n_samples < 3:
            return np.ones(n_samples, dtype=bool)

        # MinCovDet is expensive in high dimensions; detect outliers in a
        # compact PCA subspace, then fit the final PCA on those inliers.
        n_pre = min(50, n_samples - 1, n_features)
        if n_features > n_pre:
            pre_pca = PCA(n_components=n_pre, random_state=self.random_state)
            X_for_mcd = pre_pca.fit_transform(X)
        else:
            X_for_mcd = X

        support_fraction = self.support_fraction
        if support_fraction is None:
            support_fraction = min(0.75, (n_samples - 1) / n_samples)

        mcd = MinCovDet(
            support_fraction=support_fraction,
            random_state=self.random_state,
        )
        mcd.fit(X_for_mcd)
        return mcd.support_

    def transform(self, X):
        return self.pca_.transform(np.asarray(X, dtype=float))

    @property
    def explained_variance_ratio_(self):
        return self.pca_.explained_variance_ratio_

    @property
    def components_(self):
        return self.pca_.components_

def detect_special_pulses_in_file(file_path):
    """
    Detect the selected special pulse type in all pulses of a single h5 file
    and add the results as a data array.

    Parameters:
    -----------
    file_path : Path or str
        Path to the h5 file to process

    Returns:
    --------
    dict
        Statistics about the file: number of pulses and detected pulses
    """
    con.log(f"Processing: {Path(file_path).name}")

    # Open h5 file with read/write mode
    file, write_mode = open_h5_readwrite_or_readonly(file_path)
    if file is None:
        return {
            "file": Path(file_path).name,
            "status": "skipped",
            "reason": "locked_or_unreadable",
        }

    try:
        # Access pulses block
        block = get_pulse_block(file)
        data_array_names = [da.name for da in block.data_arrays]

        # Check if required data arrays exist
        if "raw_pulses" not in data_array_names:
            con.log("  ⚠ File does not contain 'raw_pulses' array. Skipping.")
            return {
                "file": Path(file_path).name,
                "status": "skipped",
                "reason": "no raw_pulses",
            }

        if "predicted_labels" not in data_array_names:
            con.log("  ⚠ File does not contain 'predicted_labels' array. Skipping.")
            return {
                "file": Path(file_path).name,
                "status": "skipped",
                "reason": "no predicted_labels",
            }

        # Load pulse data
        raw_pulses = block.data_arrays["raw_pulses"]
        predicted_labels = block.data_arrays["predicted_labels"]

        # Get sample rate from metadata
        fs = file.sections["pulses_metadata"]["metadata"]["samplerate"]
        ### TODO: CURRENTLY AT STEP 8 OF CHATS IMPLEMENTATION
        num_pulses = len(raw_pulses)
        con.log(f"  Found {num_pulses} pulses.")

        # Previous detector arrays are used to keep rule-based categories exclusive
        # when running the modes in order: double, then wide.
        predicted = predicted_labels[:]
        candidate_indices = np.where(predicted == 1)[0]

        if DETECTION_MODE == "wide":
            double_marker = load_marker_array(file_path, "is_double_peak", block)
            if double_marker is not None:
                is_double_peak = expand_marker_to_all_pulses(
                    double_marker,
                    candidate_indices,
                    num_pulses,
                    "is_double_peak",
                )
            else:
                is_double_peak = np.zeros(num_pulses, dtype=np.int64)
        else:
            is_double_peak = np.zeros(num_pulses, dtype=np.int64)

        # Analyze each pulse for the selected pulse type
        is_detection_array = np.zeros(num_pulses, dtype=np.int64)
        positive_count = 0
        negative_count = 0

        # detection loop
        for i, pulse in enumerate(raw_pulses):
            # TODO: ask patrick if all pulses in the h5 files have a 1 for predicted_labels!!
            # Only analyze predicted positive pulses
            if predicted[i] != 1:
                continue

            # Wide pulses cannot also be double peaks
            if DETECTION_MODE == "wide" and is_double_peak[i] == 1:
                negative_count += 1
                continue

            is_positive, _ = detect_pulse(pulse[:], fs)

            if is_positive:
                is_detection_array[i] = 1
                positive_count += 1
            else:
                negative_count += 1

            if (i + 1) % max(1, num_pulses // 10) == 0:
                con.log(f"  Processed {i + 1}/{num_pulses} pulses...")

        # Create or overwrite the "is_detection" data array in the h5 file
        if write_mode == "h5":
            if ARRAY_NAME in data_array_names:
                con.log(f"  Updating existing '{ARRAY_NAME}' array...")
                block.data_arrays[ARRAY_NAME][:] = is_detection_array
            else:
                con.log(f"  Creating new '{ARRAY_NAME}' array...")
                block.create_data_array(
                    ARRAY_NAME,
                    ARRAY_NAME,
                    data=is_detection_array,
                )
        else:
            sidecar = save_marker_sidecar(file_path, ARRAY_NAME, is_detection_array)
            con.log(f"  Saved '{ARRAY_NAME}' markers to {sidecar.name}")

        # Log summary
        con.log(
            f"  ✓ Completed: {positive_count} {DISPLAY_NAME}s detected, {negative_count} rejected"
        )

        return {
            "file": Path(file_path).name,
            "status": "completed",
            "num_pulses": num_pulses,
            "num_detected": positive_count,
            "num_rejected": negative_count,
            "ratio": positive_count / num_pulses if num_pulses > 0 else 0,
        }

    finally:
        file.close()


def process_all_h5_files(data_path):
    """
    Process all h5 files in a directory to detect the selected pulse type.

    Parameters:
    -----------
    data_path : Path or str
        Path to directory containing h5 files

    Returns:
    --------
    list
        List of dictionaries with statistics for each file
    """
    data_path = Path(data_path)
    path_list = get_path_list(data_path)

    if not path_list:
        con.log("No h5 files found.")
        return []

    results = []
    con.log(f"\n{'=' * 60}")
    con.log(f"Processing {len(path_list)} h5 files for {DISPLAY_NAME} detection")
    con.log(f"{'=' * 60}\n")

    for i, fp in enumerate(path_list, 1):
        con.log(f"[{i}/{len(path_list)}]")
        result = detect_special_pulses_in_file(fp)
        results.append(result)
        con.log()

    # Print summary statistics
    con.log(f"\n{'=' * 60}")
    con.log("SUMMARY")
    con.log(f"{'=' * 60}")

    # get total numbers across all files
    total_pulses = sum(r.get("num_pulses", 0) for r in results)
    total_detected = sum(r.get("num_detected", 0) for r in results)
    total_rejected = sum(r.get("num_rejected", 0) for r in results)

    # summary logging
    con.log(f"Total files processed: {len(results)}")
    con.log(f"Total pulses analyzed: {total_pulses}")
    con.log(f"Total {DISPLAY_NAME}s: {total_detected}")
    con.log(f"Total rejected: {total_rejected}")
    con.log(
        f"{DISPLAY_NAME.capitalize()} ratio: {total_detected / total_pulses if total_pulses > 0 else 0:.2%}"
    )

    return results


############################################
############# SUPERVISED LEARNING ##########
############################################


def get_default_ml_paths():
    base_path = SPECIAL_PULSE_CLASSIFIER_DIR
    pca_dir = PCA_SPACE_DIR
    pca_dir.mkdir(parents=True, exist_ok=True)
    return {
        "base": base_path,
        "labels": base_path / "labeled_special_pulses.npz",
        "naturalistic_test_labels": base_path / "naturalistic_test_labels.npz",
        "naturalistic_train_labels": base_path / "naturalistic_train_labels.npz",
        "model": base_path / "special_pulse_rf_pca.pkl",
        "pca_plot": pca_dir / "labeled_pulses_pca_space.png",
    }


def get_first_available_array(block, data_array_names, array_names):
    for array_name in array_names:
        if array_name in data_array_names:
            return block.data_arrays[array_name][:], array_name
    return None, None


def expand_marker_to_all_pulses(marker, candidate_indices, num_pulses, array_name):
    """
    Return a full-length marker array, accepting both full-pulse arrays and arrays
    written only for predicted-positive pulses.
    """
    marker = np.asarray(marker, dtype=np.int64)

    if len(marker) == num_pulses:
        return marker

    if len(marker) == len(candidate_indices):
        full_marker = np.zeros(num_pulses, dtype=np.int64)
        full_marker[candidate_indices] = marker
        return full_marker

    raise ValueError(
        f"Array '{array_name}' has length {len(marker)}, but expected either "
        f"{num_pulses} pulses or {len(candidate_indices)} predicted-positive pulses."
    )


def load_balanced_detector_labeled_pulses(
    data_path,
    pulses_per_type=300,
    random_seed=42,
):
    """
    Load a balanced manual-labeling set using old detector output arrays.

    Sampling pools:
        normal: predicted-positive pulses with wide == 0 and double == 0
        wide: predicted-positive pulses with wide == 1
        double: predicted-positive pulses with double == 1
    """
    path_list = get_path_list(Path(data_path))
    rng = np.random.default_rng(random_seed)
    pools = {label_id: [] for label_id in LABELING_PULSE_CLASSES}

    for file_idx, file_path in enumerate(path_list, 1):
        con.log(f"  Loading candidates [{file_idx}/{len(path_list)}] {file_path.name}")
        file = open_h5(file_path, nixio.FileMode.ReadOnly)
        if file is None:
            continue

        try:
            block = get_pulse_block(file)
            data_array_names = [da.name for da in block.data_arrays]

            if "raw_pulses" not in data_array_names:
                con.log("    No raw_pulses array. Skipping.")
                continue

            raw_pulses = block.data_arrays["raw_pulses"]
            num_pulses = len(raw_pulses)

            if "predicted_labels" in data_array_names:
                predicted_labels = block.data_arrays["predicted_labels"][:]
                candidate_indices = np.where(predicted_labels == 1)[0]
            else:
                candidate_indices = np.arange(num_pulses)

            if len(candidate_indices) == 0:
                continue

            detector_markers = {}
            missing = []
            for label_id in LABELING_CLASS_ARRAYS:
                marker, array_name = get_first_available_array(
                    block, data_array_names, LABELING_CLASS_ARRAYS[label_id]
                )
                if marker is None:
                    missing.append(LABELING_CLASS_ARRAYS[label_id][0])
                    continue

                detector_markers[label_id] = expand_marker_to_all_pulses(
                    marker, candidate_indices, num_pulses, array_name
                )

            if missing:
                con.log(f"    Missing {', '.join(missing)}. Skipping.")
                continue

            wide_marker = detector_markers[1]
            double_marker = detector_markers[2]

            fs = file.sections["pulses_metadata"]["metadata"]["samplerate"]

            candidate_mask = np.zeros(num_pulses, dtype=bool)
            candidate_mask[candidate_indices] = True
            masks = {
                0: candidate_mask & (wide_marker == 0) & (double_marker == 0),
                1: candidate_mask & (wide_marker == 1),
                2: candidate_mask & (double_marker == 1),
            }

            for label_id, mask in masks.items():
                for pulse_idx in np.where(mask)[0]:
                    pools[label_id].append(
                        {
                            "file_path": str(file_path),
                            "pulse_idx": int(pulse_idx),
                            "fs": float(fs),
                            "sampling_pool": LABELING_PULSE_CLASSES[label_id],
                        }
                    )

        finally:
            file.close()

    selected_records = []
    for label_id, class_name in LABELING_PULSE_CLASSES.items():
        pool = pools[label_id]
        if not pool:
            con.log(f"  No {class_name} candidates found.")
            continue

        sample_size = min(pulses_per_type, len(pool))
        if len(pool) < pulses_per_type:
            con.log(
                f"  Only {len(pool)} {class_name} candidates available; "
                f"using all of them."
            )
        else:
            con.log(f"  Sampling {sample_size} {class_name} candidates.")

        selected_indices = rng.choice(len(pool), size=sample_size, replace=False)
        selected_records.extend(pool[int(i)] for i in selected_indices)

    if not selected_records:
        return np.empty((0, 0)), []

    shuffle_order = rng.permutation(len(selected_records))
    selected_records = [selected_records[int(i)] for i in shuffle_order]

    waveforms = []
    records = []
    waveform_cache = {}
    for record in selected_records:
        file_path = record["file_path"]
        if file_path not in waveform_cache:
            file = open_h5(file_path, nixio.FileMode.ReadOnly)
            if file is None:
                continue
            try:
                block = get_pulse_block(file)
                waveform_cache[file_path] = block.data_arrays["raw_pulses"][:]
            finally:
                file.close()

        pulse_data = waveform_cache[file_path][record["pulse_idx"]]
        trace, best_channel = get_representative_waveform(pulse_data)
        waveforms.append(trace)
        records.append(
            {
                "file_path": record["file_path"],
                "pulse_idx": record["pulse_idx"],
                "fs": record["fs"],
                "best_channel": int(best_channel),
                "sampling_pool": record["sampling_pool"],
                "all_channels": np.asarray(pulse_data, dtype=float),
            }
        )

    return np.asarray(waveforms, dtype=float), records


def normalize_waveforms_for_pca(waveforms):
    """
    Baseline-correct and amplitude-normalize waveforms before PCA.
    """
    waveforms = np.asarray(waveforms, dtype=float)
    corrected = waveforms.copy()

    baseline_window = max(1, corrected.shape[1] // 5)
    baseline = np.median(corrected[:, :baseline_window], axis=1, keepdims=True)
    corrected -= baseline

    scale = np.max(np.abs(corrected), axis=1, keepdims=True)
    scale[scale == 0] = 1.0
    corrected /= scale

    return corrected


def resample_waveforms(waveforms: np.ndarray, target_length: int) -> np.ndarray:
    """Resample 1D waveforms to the length expected by the trained classifier."""
    waveforms = np.asarray(waveforms, dtype=float)
    if waveforms.ndim == 1:
        waveforms = waveforms[np.newaxis, :]
    if waveforms.shape[1] == target_length:
        return waveforms

    source_x = np.linspace(0.0, 1.0, waveforms.shape[1])
    target_x = np.linspace(0.0, 1.0, target_length)
    return np.vstack([np.interp(target_x, source_x, row) for row in waveforms])


def classifier_waveform_length(classifier) -> int:
    """Return the waveform feature length a fitted classifier pipeline expects."""
    return int(classifier.named_steps["scaler"].n_features_in_)


def prepare_classifier_waveforms(waveforms: np.ndarray, classifier) -> np.ndarray:
    """Resample and normalize waveforms for classifier prediction."""
    target_length = classifier_waveform_length(classifier)
    waveforms = resample_waveforms(waveforms, target_length)
    return normalize_waveforms_for_pca(waveforms)


def class_prior_from_labels(labels, class_ids=None):
    """Empirical class frequencies as a dict {class_id: prior}."""
    labels = np.asarray(labels, dtype=np.int64)
    if class_ids is None:
        class_ids = sorted(SPECIAL_PULSE_CLASSES)
    class_ids = [int(c) for c in class_ids]
    counts = {c: int(np.sum(labels == c)) for c in class_ids}
    total = max(sum(counts.values()), 1)
    return {c: counts[c] / total for c in class_ids}


def reweight_class_probabilities(proba, class_ids, train_prior, natural_prior):
    """
    Bayes-style prior correction: p'(c|x) ∝ p(c|x) * π_nat(c) / π_train(c).

    RF predict_proba reflects the training label mix. Multiplying by the ratio of
    natural to train priors shifts mass toward the production prevalence.
    """
    proba = np.asarray(proba, dtype=float)
    class_ids = [int(c) for c in class_ids]
    weights = np.array(
        [
            float(natural_prior.get(c, 0.0)) / max(float(train_prior.get(c, 1e-12)), 1e-12)
            for c in class_ids
        ],
        dtype=float,
    )
    adjusted = proba * weights[np.newaxis, :]
    row_sums = adjusted.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return adjusted / row_sums


def decide_classes_from_proba(
    proba,
    class_ids,
    *,
    min_proba=None,
    default_class=0,
):
    """
    Map class probabilities to labels with high bars for rare classes.

    Rare classes (keys in min_proba) are only assigned if their probability meets
    the threshold and they are the strongest rare candidate; otherwise default_class.
    """
    proba = np.asarray(proba, dtype=float)
    class_ids = np.asarray(class_ids, dtype=np.int64)
    min_proba = {int(k): float(v) for k, v in (min_proba or {}).items()}
    default_class = int(default_class)

    id_to_col = {int(c): i for i, c in enumerate(class_ids)}
    preds = np.full(len(proba), default_class, dtype=np.int64)

    rare_classes = [c for c in sorted(min_proba) if c in id_to_col]
    if not rare_classes:
        return class_ids[np.argmax(proba, axis=1)]

    for i, row in enumerate(proba):
        best_rare = None
        best_p = -1.0
        for class_id in rare_classes:
            p = float(row[id_to_col[class_id]])
            if p >= min_proba[class_id] and p > best_p:
                best_rare = class_id
                best_p = p
        if best_rare is not None:
            preds[i] = best_rare
        else:
            # Among non-rare / default: prefer argmax, but never assign a rare
            # class that failed its threshold.
            eligible = [
                c
                for c in class_ids
                if c not in min_proba or float(row[id_to_col[int(c)]]) >= min_proba[int(c)]
            ]
            if not eligible:
                preds[i] = default_class
            else:
                preds[i] = max(eligible, key=lambda c: float(row[id_to_col[int(c)]]))
    return preds


def predict_special_pulse_classes(
    classifier,
    waveforms,
    *,
    decision_config=None,
    train_prior=None,
    pulse_waveforms_raw=None,
    sample_rates=None,
):
    """
    Predict class IDs using predict_proba + optional prior/threshold/rule gate.

    Parameters
    ----------
    classifier : fitted sklearn Pipeline
    waveforms : array (n, T) already prepared for the classifier
    decision_config : dict, optional
    train_prior : dict, optional
    pulse_waveforms_raw : list/array of (T, C) multi-channel pulses for rule gate
    sample_rates : array of sample rates for rule gate
    """
    cfg = {**DEFAULT_DECISION_CONFIG, **(decision_config or {})}
    class_ids = np.asarray(classifier.classes_, dtype=np.int64)
    proba = classifier.predict_proba(waveforms)

    if cfg.get("use_prior_reweight", False):
        if train_prior is None:
            raise ValueError("train_prior is required when use_prior_reweight=True")
        proba = reweight_class_probabilities(
            proba,
            class_ids,
            train_prior,
            cfg.get("natural_prior", DEFAULT_NATURAL_PRIOR),
        )

    preds = decide_classes_from_proba(
        proba,
        class_ids,
        min_proba=cfg.get("min_proba", DEFAULT_MIN_PROBA),
        default_class=cfg.get("default_class", 0),
    )

    if cfg.get("rule_gate_double", False) and pulse_waveforms_raw is not None:
        if sample_rates is None:
            raise ValueError("sample_rates required when rule_gate_double=True")
        for i, pred in enumerate(preds):
            if pred != 2:
                continue
            is_double, _ = detect_double_pulse(
                np.asarray(pulse_waveforms_raw[i], dtype=float),
                float(sample_rates[i]),
            )
            if not is_double:
                preds[i] = 0

    return preds, proba


def _make_pca_estimator(n_components, random_state=42):
    return RobustPCA(n_components=n_components, random_state=random_state)


def _get_pca_n_components(X_train):
    X_train = np.asarray(X_train)
    n_components = min(
        PCA_MAX_CLASSIFIER_COMPONENTS,
        X_train.shape[0] - 1,
        X_train.shape[1],
    )
    return max(1, n_components)


def _get_pca_n_components_for_plot(X):
    X = np.asarray(X)
    n_components = min(
        PCA_MAX_PLOT_COMPONENTS,
        X.shape[0] - 1,
        X.shape[1],
    )
    return max(1, n_components)


def _select_meaningful_pc_pairs(explained_variance_ratio, max_pairs=4):
    """
    Choose PC scatter-plot axes beyond PC1/PC2.

    Always includes (PC1, PC2). Adds pairs that involve the next strongest
    components while both axes retain at least 2% explained variance.
    """
    explained = np.asarray(explained_variance_ratio, dtype=float)
    n_components = len(explained)
    if n_components < 2:
        return [(0, 0)]

    candidate_pairs = [(0, 1)]
    if n_components >= 3:
        candidate_pairs.extend([(0, 2), (1, 2)])
    if n_components >= 4:
        candidate_pairs.extend([(0, 3), (2, 3)])
    if n_components >= 5:
        candidate_pairs.append((1, 3))

    seen = set()
    selected = []
    min_variance = 0.02
    for i, j in candidate_pairs:
        if i >= n_components or j >= n_components:
            continue
        if explained[i] < min_variance or explained[j] < min_variance:
            continue
        pair = (min(i, j), max(i, j))
        if pair in seen:
            continue
        seen.add(pair)
        selected.append(pair)
        if len(selected) >= max_pairs:
            break

    return selected or [(0, 1)]


def _fit_pca_projection(waveforms, n_components):
    pca_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", _make_pca_estimator(n_components)),
        ]
    )
    projected = pca_pipeline.fit_transform(waveforms)
    explained = pca_pipeline.named_steps["pca"].explained_variance_ratio_
    return projected, explained, pca_pipeline


def _scatter_labeled_pca_pairs(
    ax,
    projected,
    labels,
    pc_x,
    pc_y,
    explained,
    labels_sorted,
    colors,
    split_name=None,
    split_styles=None,
):
    split_styles = split_styles or {
        "default": {"s": 42, "alpha": 0.75, "linewidths": 0.3, "zorder": 1},
    }
    style_key = split_name if split_name in split_styles else "default"
    style = split_styles[style_key]

    for color, label_id in zip(colors, labels_sorted):
        mask = labels == label_id
        if not np.any(mask):
            continue
        class_name = SPECIAL_PULSE_CLASSES.get(int(label_id), f"class {label_id}")
        legend_label = f"{class_name} (n={int(np.sum(mask))})"
        if split_name is not None:
            legend_label = f"{class_name} ({split_name}, n={int(np.sum(mask))})"

        ax.scatter(
            projected[mask, pc_x],
            projected[mask, pc_y],
            s=style["s"] * 1.4,
            alpha=style["alpha"],
            edgecolors="black",
            linewidths=style["linewidths"] + 0.4,
            color=color,
            zorder=style["zorder"],
            label=legend_label,
        )

    pc_x_var = explained[pc_x] * 100 if pc_x < len(explained) else 0.0
    pc_y_var = explained[pc_y] * 100 if pc_y < len(explained) else 0.0
    ax.set_xlabel(f"PC{pc_x + 1} ({pc_x_var:.1f}% variance)")
    ax.set_ylabel(f"PC{pc_y + 1} ({pc_y_var:.1f}% variance)")
    ax.grid(True, alpha=0.25, linestyle="--")


# Common timebase for labeling / naturalistic train features.
# Many H5 files are 48 kHz with ~10 ms snippets; others are 24 kHz with ~20 ms.
# For comparable shapes while labeling (and for RF features), we plot and store
# all snippets on this reference rate: equal sample counts → equal plotted duration


def _rate_dict(preds, class_ids=(0, 1, 2)):
    preds = np.asarray(preds, dtype=np.int64)
    n = max(len(preds), 1)
    return {
        SPECIAL_PULSE_CLASSES[c]: {
            "count": int(np.sum(preds == c)),
            "rate": float(np.mean(preds == c)) if len(preds) else 0.0,
        }
        for c in class_ids
    }


def _binary_prf(y_true, y_pred, positive_label):
    y_true = np.asarray(y_true) == positive_label
    y_pred = np.asarray(y_pred) == positive_label
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "support": int(np.sum(y_true)),
    }


def evaluate_naturalistic_test_set(
    labels_path=None,
    model_path=None,
):
    """
    Score the saved production classifier on the naturalistic test labels.
    """
    ml_paths = get_default_ml_paths()
    labels_path = (
        Path(labels_path) if labels_path else ml_paths["naturalistic_test_labels"]
    )
    model_path = Path(model_path) if model_path else ml_paths["model"]

    if not labels_path.exists():
        raise FileNotFoundError(
            f"Naturalistic test labels not found at {labels_path}. "
            "Run --mode label-naturalistic-test first."
        )
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    import __main__
    setattr(__main__, "RobustPCA", RobustPCA)

    waveforms, labels, _ = load_labeled_dataset(labels_path)
    with model_path.open("rb") as f:
        model_data = pickle.load(f)

    classifier = model_data["classifier"]
    decision_config = model_data.get("decision_config", DEFAULT_DECISION_CONFIG)
    train_prior = model_data.get("train_prior", {0: 0.42, 1: 0.20, 2: 0.38})

    X = prepare_classifier_waveforms(waveforms, classifier)
    y_hard = classifier.predict(X)
    y_policy, _ = predict_special_pulse_classes(
        classifier,
        X,
        decision_config=decision_config,
        train_prior=train_prior,
    )

    labels_sorted = sorted(np.unique(labels))
    target_names = [SPECIAL_PULSE_CLASSES[int(i)] for i in labels_sorted]

    con.log("\n" + "=" * 60)
    con.log("NATURALISTIC TEST EVALUATION")
    con.log("=" * 60)
    con.log(f"Labels: {labels_path} (n={len(labels)})")
    con.log(f"Model:  {model_path}")
    con.log(f"Decision config: {decision_config}")
    true_prior = class_prior_from_labels(labels)
    con.log(f"True label rates: {true_prior}")

    con.log("\n--- Hard predict() ---")
    print(
        classification_report(
            labels,
            y_hard,
            labels=labels_sorted,
            target_names=target_names,
            zero_division=0,
        )
    )
    print(confusion_matrix(labels, y_hard, labels=labels_sorted))

    con.log("\n--- Production decision policy ---")
    print(
        classification_report(
            labels,
            y_policy,
            labels=labels_sorted,
            target_names=target_names,
            zero_division=0,
        )
    )
    print(confusion_matrix(labels, y_policy, labels=labels_sorted))

    pred_rates = class_prior_from_labels(y_policy)
    con.log(f"Predicted rates (policy): {pred_rates}")

    out_dir = ml_paths["base"] / "naturalistic_test_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "naturalistic_test_metrics.json"
    summary = {
        "labels_path": str(labels_path),
        "model_path": str(model_path),
        "n": int(len(labels)),
        "true_prior": true_prior,
        "decision_config": decision_config,
        "hard_predict": _compute_multiclass_metrics(labels, y_hard, labels_sorted),
        "policy": _compute_multiclass_metrics(labels, y_policy, labels_sorted),
        "policy_double": _binary_prf(labels, y_policy, 2),
        "policy_wide": _binary_prf(labels, y_policy, 1),
        "predicted_rates_policy": pred_rates,
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    con.log(f"Saved {out_json}")
    return summary


def load_labeled_dataset(labels_path):
    labels_path = Path(labels_path)
    if not labels_path.exists():
        raise FileNotFoundError(f"Label dataset does not exist: {labels_path}")

    data = np.load(labels_path, allow_pickle=False)
    return data["waveforms"], data["labels"], data["records"]


def plot_labeled_pulses_pca_space(
    waveforms,
    labels,
    output_path=None,
    show=True,
    class_names=None,
):
    """
    Plot robust-PCA projections of manually labeled pulse waveforms.

    Computes up to PCA_MAX_PLOT_COMPONENTS components and renders the most
    informative PC pairs (PC1/PC2 plus additional high-variance axes).
    """
    apply_presentation_style()
    if len(labels) < 2:
        con.log("Need at least two labeled pulses to plot PCA space.")
        return None, None

    class_names = class_names or SPECIAL_PULSE_CLASSES
    labels_sorted = sorted(np.unique(labels))
    n_components = _get_pca_n_components_for_plot(waveforms)
    if n_components < 1:
        con.log("No waveform features available to plot PCA space.")
        return None, None

    projected, explained, pca_pipeline = _fit_pca_projection(waveforms, n_components)
    pca_step = pca_pipeline.named_steps["pca"]
    con.log(
        f"Robust PCA plot: {n_components} components "
        f"({pca_step.n_inliers_} inliers, {pca_step.n_outliers_} outliers excluded from fit)"
    )

    pc_pairs = _select_meaningful_pc_pairs(explained)
    n_panels = len(pc_pairs)
    n_cols = min(2, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7.0 * n_cols, 6.0 * n_rows),
        squeeze=False,
    )
    colors = classifier_label_colors(labels_sorted, class_names)

    for panel_idx, (pc_x, pc_y) in enumerate(pc_pairs):
        row_idx, col_idx = divmod(panel_idx, n_cols)
        ax = axes[row_idx, col_idx]
        _scatter_labeled_pca_pairs(
            ax,
            projected,
            labels,
            pc_x,
            pc_y,
            explained,
            labels_sorted,
            colors,
        )
        if panel_idx == 0:
            ax.legend(title="Manual label", frameon=True, loc=LEGEND_LOC)

    for panel_idx in range(n_panels, n_rows * n_cols):
        row_idx, col_idx = divmod(panel_idx, n_cols)
        axes[row_idx, col_idx].set_axis_off()

    fig.suptitle(
        "Robust PCA Space of Manually Labeled Pulses",
        y=1.02,
    )
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        con.log(f"Saved labeled-pulse PCA plot to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes


def _prepare_classifier_train_test_split(waveforms, labels, records=None):
    """
    Shared stratified train/test split for classifier training and benchmarking.
    """
    waveforms = np.asarray(waveforms, dtype=float)
    labels = np.asarray(labels, dtype=np.int64)
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    if len(unique_labels) < 2:
        raise ValueError("Need at least two labeled classes to train a classifier.")

    min_class_count = int(np.min(label_counts))
    num_classes = len(unique_labels)
    enough_for_stratify = min_class_count >= 2 and len(labels) >= 2 * num_classes
    stratify = labels if enough_for_stratify else None

    if stratify is not None:
        test_count = max(num_classes, int(np.ceil(0.25 * len(labels))))
        test_size = test_count / len(labels)
    else:
        test_size = 0.25 if len(labels) >= 8 else 0.5

    split_indices = np.arange(len(waveforms))
    train_idx, test_idx = train_test_split(
        split_indices,
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )

    labels_sorted = sorted(unique_labels)
    target_names = [SPECIAL_PULSE_CLASSES[int(i)] for i in labels_sorted]
    result = {
        "X_train": waveforms[train_idx],
        "X_test": waveforms[test_idx],
        "y_train": labels[train_idx],
        "y_test": labels[test_idx],
        "labels_sorted": labels_sorted,
        "target_names": target_names,
    }
    if records is not None:
        records = np.asarray(records)
        result["test_records"] = records[test_idx]
    return result


def _compute_multiclass_metrics(y_test, y_pred, labels_sorted):
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=labels_sorted,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_test,
            y_pred,
            labels=labels_sorted,
            average="weighted",
            zero_division=0,
        )
    )
    return {
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
    }

def train_special_pulse_classifier(
    labels_path=None,
    model_path=None,
    show_pca_plot=False,
    decision_config=None,
):
    """
    Train a robust PCA + random forest multiclass classifier and print precision/recall/f1.

    Saves decision_config + train_prior alongside the pipeline so deploy can use
    predict_proba with prior reweighting and rare-class thresholds.
    """
    ml_paths = get_default_ml_paths()
    if labels_path is None:
        # Prefer naturalistic training labels when available (new workflow).
        if ml_paths["naturalistic_train_labels"].exists():
            labels_path = ml_paths["naturalistic_train_labels"]
            con.log(f"Using naturalistic training labels: {labels_path}")
        else:
            labels_path = ml_paths["labels"]
    else:
        labels_path = Path(labels_path)
    model_path = Path(model_path) if model_path else ml_paths["model"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    decision_config = {**DEFAULT_DECISION_CONFIG, **(decision_config or {})}

    waveforms, labels, _ = load_labeled_dataset(labels_path)
    active_label_mask = np.isin(labels, list(LABELING_PULSE_CLASSES))
    if not np.all(active_label_mask):
        ignored_count = int(np.sum(~active_label_mask))
        con.log(f"Ignoring {ignored_count} labels outside normal/wide/double.")
        waveforms = waveforms[active_label_mask]
        labels = labels[active_label_mask]
    if len(labels) == 0:
        raise ValueError(
            "No normal/wide/double labels available to train a classifier."
        )

    unique_labels, label_counts = np.unique(labels, return_counts=True)
    if len(unique_labels) < 2:
        raise ValueError("Need at least two labeled classes to train a classifier.")

    plot_labeled_pulses_pca_space(
        waveforms,
        labels,
        output_path=ml_paths["pca_plot"],
        show=show_pca_plot,
    )

    split = _prepare_classifier_train_test_split(waveforms, labels)
    X_train = split["X_train"]
    X_test = split["X_test"]
    y_train = split["y_train"]
    y_test = split["y_test"]
    labels_sorted = split["labels_sorted"]
    target_names = split["target_names"]
    unique_labels = labels_sorted
    train_prior = class_prior_from_labels(y_train, class_ids=labels_sorted)

    n_components = _get_pca_n_components(X_train)
    rf_class_weight = decision_config.get("rf_class_weight", None)

    classifier = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", _make_pca_estimator(n_components)),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight=rf_class_weight,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    classifier.fit(X_train, y_train)
    y_pred_hard = classifier.predict(X_test)
    y_pred, _ = predict_special_pulse_classes(
        classifier,
        X_test,
        decision_config=decision_config,
        train_prior=train_prior,
    )

    con.log("\nClassifier performance on held-out labeled pulses (hard predict):")
    print(
        classification_report(
            y_test,
            y_pred_hard,
            labels=labels_sorted,
            target_names=target_names,
            zero_division=0,
        )
    )
    con.log("Confusion matrix (hard predict):")
    print(confusion_matrix(y_test, y_pred_hard, labels=labels_sorted))

    con.log(
        "\nHeld-out metrics with production decision policy "
        f"(prior_reweight={decision_config.get('use_prior_reweight')}, "
        f"min_proba={decision_config.get('min_proba')}):"
    )
    print(
        classification_report(
            y_test,
            y_pred,
            labels=labels_sorted,
            target_names=target_names,
            zero_division=0,
        )
    )
    con.log("Confusion matrix (decision policy):")
    print(confusion_matrix(y_test, y_pred, labels=labels_sorted))

    metrics = _compute_multiclass_metrics(y_test, y_pred, labels_sorted)
    macro_precision = metrics["macro_precision"]
    macro_recall = metrics["macro_recall"]
    macro_f1 = metrics["macro_f1"]
    weighted_precision = metrics["weighted_precision"]
    weighted_recall = metrics["weighted_recall"]
    weighted_f1 = metrics["weighted_f1"]

    with model_path.open("wb") as f:
        pickle.dump(
            {
                "classifier": classifier,
                "classes": SPECIAL_PULSE_CLASSES,
                "array_names": SPECIAL_CLASS_ARRAYS,
                "multiclass_array": MULTICLASS_ARRAY_NAME,
                "waveform_length": int(X_train.shape[1]),
                "train_prior": train_prior,
                "decision_config": decision_config,
            },
            f,
        )

    con.log(f"Saved trained classifier to {model_path}")
    con.log(f"Train prior: {train_prior}")
    con.log(f"Decision config: {decision_config}")
    con.log("\nFinal held-out classifier metrics (decision policy):")
    print(
        f"macro    precision={macro_precision:.3f} "
        f"recall={macro_recall:.3f} f1={macro_f1:.3f}"
    )
    print(
        f"weighted precision={weighted_precision:.3f} "
        f"recall={weighted_recall:.3f} f1={weighted_f1:.3f}"
    )
    return model_path


def write_or_create_data_array(block, array_name, values):
    data_array_names = [da.name for da in block.data_arrays]
    values = np.asarray(values, dtype=np.int64)

    if array_name in data_array_names:
        block.data_arrays[array_name][:] = values
    else:
        block.create_data_array(array_name, array_name, data=values)


def predict_special_pulses_in_file(
    file_path,
    classifier,
    *,
    decision_config=None,
    train_prior=None,
):
    file, write_mode = open_h5_readwrite_or_readonly(file_path)
    if file is None:
        return {"status": "skipped", "reason": "locked_or_unreadable"}

    try:
        block = get_pulse_block(file)
        data_array_names = [da.name for da in block.data_arrays]

        if "raw_pulses" not in data_array_names:
            con.log(f"  {Path(file_path).name}: no raw_pulses array. Skipping.")
            return {"status": "skipped", "reason": "no raw_pulses"}

        raw_pulses = block.data_arrays["raw_pulses"]
        num_pulses = len(raw_pulses)
        fs = float(file.sections["pulses_metadata"]["metadata"]["samplerate"])

        if "predicted_labels" in data_array_names:
            predicted_labels = block.data_arrays["predicted_labels"][:]
            candidate_indices = np.where(predicted_labels == 1)[0]
        else:
            candidate_indices = np.arange(num_pulses)

        predicted_classes = np.zeros(num_pulses, dtype=np.int64)

        if len(candidate_indices) > 0:
            waveforms = []
            raw_for_gate = []
            for pulse_idx in candidate_indices:
                pulse_data = raw_pulses[int(pulse_idx)][:]
                raw_for_gate.append(pulse_data)
                trace, _ = get_representative_waveform(pulse_data)
                waveforms.append(trace)

            waveforms = prepare_classifier_waveforms(
                np.asarray(waveforms, dtype=float), classifier
            )
            cfg = decision_config or DEFAULT_DECISION_CONFIG
            sample_rates = np.full(len(candidate_indices), fs, dtype=float)
            preds, _ = predict_special_pulse_classes(
                classifier,
                waveforms,
                decision_config=cfg,
                train_prior=train_prior,
                pulse_waveforms_raw=(
                    raw_for_gate if cfg.get("rule_gate_double") else None
                ),
                sample_rates=(
                    sample_rates if cfg.get("rule_gate_double") else None
                ),
            )
            predicted_classes[candidate_indices] = preds

        marker_arrays = {
            MULTICLASS_ARRAY_NAME: predicted_classes,
            **{
                array_name: (predicted_classes == label_id).astype(np.int64)
                for label_id, array_name in SPECIAL_CLASS_ARRAYS.items()
            },
        }

        if write_mode == "h5":
            for array_name, values in marker_arrays.items():
                write_or_create_data_array(block, array_name, values)
        else:
            for array_name, values in marker_arrays.items():
                sidecar = save_marker_sidecar(file_path, array_name, values)
                con.log(f"  Saved '{array_name}' markers to {sidecar.name}")

        counts = {
            SPECIAL_PULSE_CLASSES[label_id]: int(np.sum(predicted_classes == label_id))
            for label_id in SPECIAL_PULSE_CLASSES
        }

        con.log(f"  {Path(file_path).name}: {counts}")
        return {"status": "completed", "counts": counts, "write_mode": write_mode}

    finally:
        file.close()


def apply_special_pulse_classifier(data_path, model_path=None):
    ml_paths = get_default_ml_paths()
    model_path = Path(model_path) if model_path else ml_paths["model"]

    # Compatibility for older pickles: during training, RobustPCA may have been
    # pickled under "__main__" (e.g. when the training script was run directly).
    # When loading from another script, that class may not exist in __main__,
    # causing: "Can't get attribute 'RobustPCA' on <module '__main__' ...>".
    import __main__
    setattr(__main__, "RobustPCA", RobustPCA)

    with model_path.open("rb") as f:
        model_data = pickle.load(f)

    classifier = model_data["classifier"]
    decision_config = model_data.get("decision_config", DEFAULT_DECISION_CONFIG)
    train_prior = model_data.get("train_prior")
    if train_prior is None:
        # Older pickles: approximate train prior from balanced label mix.
        train_prior = {0: 0.42, 1: 0.20, 2: 0.38}
        con.log(
            "Model pickle has no train_prior; using approximate balanced priors "
            f"{train_prior}"
        )

    path_list = get_path_list(Path(data_path))

    results = []
    for file_idx, file_path in enumerate(path_list, 1):
        con.log(f"Predicting [{file_idx}/{len(path_list)}] {file_path.name}")
        results.append(
            predict_special_pulses_in_file(
                file_path,
                classifier,
                decision_config=decision_config,
                train_prior=train_prior,
            )
        )

    return results


_LABELING_EXPORTS = {
    "LABEL_REF_FS",
    "apply_label_ref_timebase",
    "compute_label_plot_half_window_ms",
    "compute_label_plot_ylim",
    "interactive_label_enrichment_train_set",
    "interactive_label_naturalistic_test_set",
    "interactive_label_naturalistic_train_set",
    "interactive_label_pulses",
    "load_enrichment_pulses_for_labeling",
    "load_excluded_pulse_keys",
    "load_naturalistic_pulses_for_labeling",
    "normalize_channels_for_label_plot",
    "peak_aligned_time_ms",
    "plot_labeling_pulse",
    "pulse_peak_index",
}

_BENCHMARK_EXPORTS = {
    "benchmark_pulse_classifiers",
    "plot_benchmark_pca_scatter",
    "plot_benchmark_waveform_sanity",
    "sample_naturalistic_pulses",
    "tune_classifier_decisions",
}


def __getattr__(name):
    """Lazy re-exports to avoid circular imports with labeling/benchmark modules."""
    if name in _LABELING_EXPORTS:
        try:
            from . import special_pulse_labeling as _labeling
        except ImportError:
            import special_pulse_labeling as _labeling

        return getattr(_labeling, name)
    if name in _BENCHMARK_EXPORTS:
        try:
            from . import special_pulse_benchmark as _benchmark
        except ImportError:
            import special_pulse_benchmark as _benchmark

        return getattr(_benchmark, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def run_supervised_detection(
    data_path,
    *,
    labels_path=None,
    model_path=None,
    label=False,
    pulses_per_type=300,
    train=False,
    retrain=False,
    apply=True,
    auto_train=False,
):
    """
    Run the supervised special-pulse workflow without interactive prompts.

    By default applies an existing trained classifier. When auto_train=True, trains
    automatically if no model exists but labeled examples are available.
    """
    try:
        from .special_pulse_labeling import interactive_label_pulses
    except ImportError:
        from special_pulse_labeling import interactive_label_pulses

    ml_paths = get_default_ml_paths()
    if labels_path is None:
        if ml_paths["naturalistic_train_labels"].exists():
            labels_path = ml_paths["naturalistic_train_labels"]
        else:
            labels_path = ml_paths["labels"]
    else:
        labels_path = Path(labels_path)
    model_path = Path(model_path) if model_path else ml_paths["model"]

    if label:
        interactive_label_pulses(
            data_path,
            labels_path=labels_path,
            pulses_per_type=pulses_per_type,
        )

    should_train = train or retrain
    if (
        auto_train
        and not should_train
        and apply
        and not model_path.exists()
        and labels_path.exists()
    ):
        should_train = True
        con.log(f"No model at {model_path}; training from {labels_path}")

    if should_train:
        if not labels_path.exists():
            raise FileNotFoundError(
                f"Labeled pulses not found at {labels_path}. "
                "Run with --label first to create training data."
            )
        train_special_pulse_classifier(
            labels_path=labels_path,
            model_path=model_path,
        )

    if apply:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained classifier not found at {model_path}. "
                "Run with --label and --train, or provide an existing model."
            )
        return apply_special_pulse_classifier(data_path, model_path=model_path)

    return None


def supervised_learning_workflow(data_path):
    """
    Interactive end-to-end workflow:
    1. optionally label pulses,
    2. train/evaluate PCA + random forest classifier,
    3. apply predictions to h5 files.
    """
    ml_paths = get_default_ml_paths()

    con.log("\n" + "=" * 60)
    con.log("SUPERVISED SPECIAL PULSE CLASSIFIER")
    con.log("=" * 60)
    con.log(f"Label dataset: {ml_paths['labels']}")
    con.log(f"Model file:     {ml_paths['model']}")
    con.log("=" * 60)

    should_label = input("Label pulses now? [y/N]: ").strip().lower() == "y"
    pulses_per_type = 300
    if should_label:
        pulses_per_type_raw = input(
            "Pulses to sample per type for labeling [300]: "
        ).strip()
        pulses_per_type = int(pulses_per_type_raw) if pulses_per_type_raw else 300

    should_train = (
        input("Train classifier from labeled pulses? [Y/n]: ").strip().lower()
    )
    should_apply = input("Apply classifier to h5 files now? [Y/n]: ").strip().lower()

    run_supervised_detection(
        data_path,
        labels_path=ml_paths["labels"],
        model_path=ml_paths["model"],
        label=should_label,
        pulses_per_type=pulses_per_type,
        train=should_train != "n",
        apply=should_apply != "n",
    )


def main():
    global DETECTION_MODE, ARRAY_NAME, DISPLAY_NAME

    parser = argparse.ArgumentParser(
        description="Detect special pulse shapes (double, wide) in .h5 files.",
    )
    parser.add_argument(
        "--mode",
        choices=(
            "supervised",
            "rule-based",
            "benchmark",
            "tune-decisions",
            "label-naturalistic-test",
            "label-naturalistic-train",
            "label-enrichment-train",
            "eval-naturalistic-test",
        ),
        default="supervised",
        help="Detection mode (default: supervised ML classifier)",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=H5_DIR,
        help=f"Directory with .h5 files (default: {H5_DIR})",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Prompt for each step of the supervised workflow",
    )
    parser.add_argument(
        "--label",
        action="store_true",
        help="Interactively label pulse examples for training",
    )
    parser.add_argument(
        "--pulses-per-type",
        type=int,
        default=300,
        help="Pulses to sample per class during balanced labeling (default: 300)",
    )
    parser.add_argument(
        "--n-pulses",
        type=int,
        default=500,
        help="Pulses to sample for naturalistic labeling (default: 500; train mode uses 1200 if unset via mode)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=40,
        help="Max H5 files to draw from for naturalistic sampling (default: 40)",
    )
    parser.add_argument(
        "--append-test-labels",
        action="store_true",
        help="Append to existing naturalistic test labels instead of overwriting",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train classifier from labeled examples",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain classifier even if a model file already exists",
    )
    parser.add_argument(
        "--no-apply",
        action="store_true",
        help="Skip applying the classifier to .h5 files",
    )
    parser.add_argument(
        "--detection-mode",
        choices=tuple(MODE_CONFIG),
        default=DETECTION_MODE,
        help="Pulse type for rule-based detection (default: %(default)s)",
    )
    args = parser.parse_args()

    if args.mode == "supervised":
        if args.interactive:
            supervised_learning_workflow(args.data_path)
            return

        run_supervised_detection(
            args.data_path,
            label=args.label,
            pulses_per_type=args.pulses_per_type,
            train=args.train,
            retrain=args.retrain,
            apply=not args.no_apply,
            auto_train=True,
        )
        return

    if args.mode == "rule-based":
        DETECTION_MODE = args.detection_mode
        ARRAY_NAME = MODE_CONFIG[DETECTION_MODE]["array_name"]
        DISPLAY_NAME = MODE_CONFIG[DETECTION_MODE]["display_name"]
        process_all_h5_files(args.data_path)
        return

    if args.mode == "tune-decisions":
        try:
            from .special_pulse_benchmark import tune_classifier_decisions
        except ImportError:
            from special_pulse_benchmark import tune_classifier_decisions

        tune_classifier_decisions(data_path=args.data_path)
        return

    if args.mode == "label-naturalistic-test":
        try:
            from .special_pulse_labeling import interactive_label_naturalistic_test_set
        except ImportError:
            from special_pulse_labeling import interactive_label_naturalistic_test_set

        interactive_label_naturalistic_test_set(
            args.data_path,
            n_pulses=args.n_pulses,
            max_files=args.max_files,
            append_existing=args.append_test_labels,
        )
        return

    if args.mode == "label-naturalistic-train":
        try:
            from .special_pulse_labeling import interactive_label_naturalistic_train_set
        except ImportError:
            from special_pulse_labeling import interactive_label_naturalistic_train_set

        n_train = args.n_pulses if args.n_pulses != 500 else 1200
        max_files = args.max_files if args.max_files != 40 else 80
        interactive_label_naturalistic_train_set(
            args.data_path,
            n_pulses=n_train,
            max_files=max_files,
            append_existing=args.append_test_labels,
        )
        return

    if args.mode == "label-enrichment-train":
        try:
            from .special_pulse_labeling import interactive_label_enrichment_train_set
        except ImportError:
            from special_pulse_labeling import interactive_label_enrichment_train_set

        interactive_label_enrichment_train_set(
            args.data_path,
            max_files=max(args.max_files, 60),
            append_existing=True,
        )
        return

    if args.mode == "eval-naturalistic-test":
        evaluate_naturalistic_test_set()
        return

    try:
        from .special_pulse_benchmark import benchmark_pulse_classifiers
    except ImportError:
        from special_pulse_benchmark import benchmark_pulse_classifiers

    benchmark_pulse_classifiers()


if __name__ == "__main__":
    main()
