"""Benchmark and decision-policy tuning for special-pulse classifiers.

Compares model/feature choices and decision thresholds; not part of the
default production apply path.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nixio
import numpy as np
from rich.console import Console
from rich.table import Table
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data_paths import H5_DIR, PCA_SPACE_DIR
from h5_io import get_path_list, get_pulse_block, open_h5
from waveform_rule_metrics import (
    detect_double_pulse,
    detect_wide_pulse,
    get_representative_waveform,
)

con = Console()


def _detection():
    """Return the already-loaded detection module (package or flat name)."""
    for name in (
        "special_pulses.double_peaks_detection",
        "double_peaks_detection",
    ):
        mod = sys.modules.get(name)
        if mod is not None and hasattr(mod, "load_labeled_dataset"):
            return mod
    import double_peaks_detection as mod

    return mod


_dpd = _detection()
DEFAULT_NATURAL_PRIOR = _dpd.DEFAULT_NATURAL_PRIOR
LABELING_PULSE_CLASSES = _dpd.LABELING_PULSE_CLASSES
SPECIAL_PULSE_CLASSES = _dpd.SPECIAL_PULSE_CLASSES
_compute_multiclass_metrics = _dpd._compute_multiclass_metrics
_fit_pca_projection = _dpd._fit_pca_projection
_get_pca_n_components = _dpd._get_pca_n_components
_get_pca_n_components_for_plot = _dpd._get_pca_n_components_for_plot
_make_pca_estimator = _dpd._make_pca_estimator
_prepare_classifier_train_test_split = _dpd._prepare_classifier_train_test_split
_scatter_labeled_pca_pairs = _dpd._scatter_labeled_pca_pairs
class_prior_from_labels = _dpd.class_prior_from_labels
get_default_ml_paths = _dpd.get_default_ml_paths
load_labeled_dataset = _dpd.load_labeled_dataset
prepare_classifier_waveforms = _dpd.prepare_classifier_waveforms
predict_special_pulse_classes = _dpd.predict_special_pulse_classes
reweight_class_probabilities = _dpd.reweight_class_probabilities


def _build_benchmark_classifier_pipelines():
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "svc_rbf": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "svc",
                    SVC(
                        kernel="rbf",
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=5),
    }


def _wrap_benchmark_pipeline(estimator, feature_space, n_components):
    if isinstance(estimator, Pipeline):
        if feature_space == "pca":
            return Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("pca", _make_pca_estimator(n_components)),
                    ("classifier", estimator.named_steps["svc"]),
                ]
            )
        return estimator

    steps = [("scaler", StandardScaler())]
    if feature_space == "pca":
        steps.append(("pca", _make_pca_estimator(n_components)))
    steps.append(("classifier", estimator))
    return Pipeline(steps=steps)


def plot_benchmark_pca_scatter(
    X_train,
    y_train,
    X_test,
    y_test,
    output_path,
    title,
    fit_on="train",
    show_splits=("train", "test"),
):
    """
    Plot labeled pulses in robust-PCA space across multiple PC pairs.

    fit_on:
        "train" — fit StandardScaler + RobustPCA on the training split only
        (matches the classifier pipeline).
        "full" — fit on train + test combined (exploratory view of full labels).
    show_splits:
        Which splits to draw, e.g. ("train", "test") or ("train", "test") for both.
    """
    show_splits = tuple(show_splits)
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_train = np.asarray(y_train, dtype=np.int64)
    y_test = np.asarray(y_test, dtype=np.int64)

    if fit_on == "train":
        fit_waveforms = X_train
    elif fit_on in {"full", "all"}:
        fit_waveforms = np.vstack([X_train, X_test])
    else:
        raise ValueError("fit_on must be 'train' or 'full'")

    n_components = _get_pca_n_components_for_plot(fit_waveforms)
    projected, explained, pca_pipeline = _fit_pca_projection(
        fit_waveforms, n_components
    )
    pca_step = pca_pipeline.named_steps["pca"]

    split_data = {}
    if "train" in show_splits:
        projected_train = pca_pipeline.transform(X_train)
        split_data["train"] = (projected_train, y_train)
    if "test" in show_splits:
        projected_test = pca_pipeline.transform(X_test)
        split_data["test"] = (projected_test, y_test)

    all_labels = [labels for _, labels in split_data.values()]
    labels_sorted = sorted(np.unique(np.concatenate(all_labels)))
    pc_pairs = _select_meaningful_pc_pairs(explained)
    n_panels = len(pc_pairs)
    n_cols = min(2, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.5 * n_cols, 5.5 * n_rows),
        squeeze=False,
    )
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(labels_sorted), 1)))
    split_styles = {
        "train": {"s": 28, "alpha": 0.35, "linewidths": 0.0, "zorder": 1},
        "test": {"s": 46, "alpha": 0.9, "linewidths": 0.35, "zorder": 2},
    }

    for panel_idx, (pc_x, pc_y) in enumerate(pc_pairs):
        row_idx, col_idx = divmod(panel_idx, n_cols)
        ax = axes[row_idx, col_idx]
        for split_name, (projected, labels) in split_data.items():
            _scatter_labeled_pca_pairs(
                ax,
                projected,
                labels,
                pc_x,
                pc_y,
                explained,
                labels_sorted,
                colors,
                split_name=split_name,
                split_styles=split_styles,
            )
        if panel_idx == 0:
            ax.legend(title="True label (split)", frameon=True, fontsize=7, loc="best")

    for panel_idx in range(n_panels, n_rows * n_cols):
        row_idx, col_idx = divmod(panel_idx, n_cols)
        axes[row_idx, col_idx].set_axis_off()

    fit_note = "fit on train" if fit_on == "train" else "fit on full dataset"
    fig.suptitle(
        (
            f"{title}\n({fit_note}; {n_components} components, "
            f"{pca_step.n_inliers_} inliers, {pca_step.n_outliers_} outliers excluded)"
        ),
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_benchmark_waveform_sanity(
    X_test,
    y_test,
    y_pred,
    test_records,
    output_path,
    title,
    examples_per_type=2,
):
    labels_sorted = sorted(np.unique(y_test))
    n_rows = len(labels_sorted)
    n_cols = 2
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(11, 2.8 * n_rows),
        squeeze=False,
    )

    for row_idx, label_id in enumerate(labels_sorted):
        class_name = SPECIAL_PULSE_CLASSES.get(int(label_id), f"class {label_id}")
        class_mask = y_test == label_id
        class_indices = np.where(class_mask)[0]

        correct_indices = class_indices[y_pred[class_indices] == label_id]
        wrong_indices = class_indices[y_pred[class_indices] != label_id]

        selections = [
            ("correct", correct_indices),
            ("misclassified", wrong_indices),
        ]

        for col_idx, (example_type, candidate_indices) in enumerate(selections):
            ax = axes[row_idx, col_idx]
            if len(candidate_indices) == 0:
                ax.text(
                    0.5,
                    0.5,
                    f"No {example_type} examples",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_axis_off()
                continue

            pick_count = min(examples_per_type, len(candidate_indices))
            picked = candidate_indices[:pick_count]

            for example_idx in picked:
                waveform = X_test[example_idx]
                record = test_records[example_idx]
                fs = float(record["fs"]) if record is not None else 1.0
                time_axis = np.arange(len(waveform)) / fs * 1000
                pred_name = SPECIAL_PULSE_CLASSES.get(
                    int(y_pred[example_idx]), f"class {y_pred[example_idx]}"
                )
                ax.plot(
                    time_axis,
                    waveform,
                    alpha=0.85,
                    linewidth=1.4,
                    label=f"pred={pred_name}",
                )

            ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Amplitude")
            ax.grid(True, alpha=0.25, linestyle="--")
            ax.legend(fontsize=8, loc="upper right")
            ax.set_title(
                f"{class_name}: {example_type} (true={class_name})",
                fontsize=10,
                fontweight="bold",
            )

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _print_benchmark_comparison_table(results):
    table = Table(title="Pulse classifier benchmark (held-out test set)")
    table.add_column("Run", style="cyan")
    table.add_column("Classifier")
    table.add_column("Features")
    table.add_column("Macro P/R/F1")
    table.add_column("Weighted P/R/F1")
    table.add_column("Best?", justify="center")

    best_macro_f1 = max(row["macro_f1"] for row in results)
    best_rows = [row for row in results if row["macro_f1"] == best_macro_f1]

    for row in results:
        is_best = row in best_rows
        table.add_row(
            row["run_id"],
            row["classifier"],
            row["feature_space"],
            (
                f"{row['macro_precision']:.3f} / "
                f"{row['macro_recall']:.3f} / "
                f"{row['macro_f1']:.3f}"
            ),
            (
                f"{row['weighted_precision']:.3f} / "
                f"{row['weighted_recall']:.3f} / "
                f"{row['weighted_f1']:.3f}"
            ),
            "★" if is_best else "",
        )

    con.print(table)
    return best_rows


def _summarize_benchmark_findings(results, best_rows):
    pca_runs = [row for row in results if row["feature_space"] == "pca"]
    raw_runs = [row for row in results if row["feature_space"] == "raw"]
    best_pca = max(pca_runs, key=lambda row: row["macro_f1"])
    best_raw = max(raw_runs, key=lambda row: row["macro_f1"])
    pca_vs_raw_gap = best_pca["macro_f1"] - best_raw["macro_f1"]

    classifier_best = {}
    for row in results:
        current = classifier_best.get(row["classifier"])
        if current is None or row["macro_f1"] > current["macro_f1"]:
            classifier_best[row["classifier"]] = row

    ranked_classifiers = sorted(
        classifier_best.values(), key=lambda row: row["macro_f1"], reverse=True
    )
    top = ranked_classifiers[0]
    second = ranked_classifiers[1] if len(ranked_classifiers) > 1 else None
    classifier_gap = top["macro_f1"] - second["macro_f1"] if second else 0.0

    con.log("\nBenchmark summary:")
    con.log(
        f"  Best overall: {best_rows[0]['classifier']} + {best_rows[0]['feature_space']} "
        f"(macro F1={best_rows[0]['macro_f1']:.3f})"
    )
    if len(best_rows) > 1:
        tied = ", ".join(
            f"{row['classifier']}+{row['feature_space']}" for row in best_rows
        )
        con.log(f"  Tied best runs: {tied}")

    if abs(pca_vs_raw_gap) < 0.02:
        pca_verdict = "marginal difference"
    elif pca_vs_raw_gap > 0:
        pca_verdict = f"PCA slightly better by {pca_vs_raw_gap:.3f} macro F1"
    else:
        pca_verdict = f"raw waveforms slightly better by {-pca_vs_raw_gap:.3f} macro F1"
    con.log(f"  PCA vs raw: {pca_verdict}")

    if classifier_gap < 0.02:
        classifier_verdict = "marginal difference between top classifiers"
    else:
        classifier_verdict = (
            f"{top['classifier']} leads by {classifier_gap:.3f} macro F1 "
            f"over {second['classifier']}"
        )
    con.log(f"  Classifier spread: {classifier_verdict}")


def benchmark_pulse_classifiers(labels_path=None):
    """
    Compare multiclass classifiers on PCA features vs raw waveforms.

    Saves metrics and plots under SPECIAL_PULSE_CLASSIFIER_DIR/benchmark/.
    """
    ml_paths = get_default_ml_paths()
    labels_path = Path(labels_path) if labels_path else ml_paths["labels"]
    benchmark_dir = ml_paths["base"] / "benchmark"
    pca_plot_dir = PCA_SPACE_DIR / "benchmark"
    waveform_plot_dir = benchmark_dir / "waveforms"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    if not labels_path.exists():
        con.log(f"No labeled dataset found at {labels_path}.")
        con.log("Run the supervised labeling workflow first (main menu option 2).")
        return None

    waveforms, labels, records = load_labeled_dataset(labels_path)
    active_label_mask = np.isin(labels, list(LABELING_PULSE_CLASSES))
    if not np.all(active_label_mask):
        ignored_count = int(np.sum(~active_label_mask))
        con.log(f"Ignoring {ignored_count} labels outside normal/wide/double.")
        waveforms = waveforms[active_label_mask]
        labels = labels[active_label_mask]
        records = records[active_label_mask]

    if len(labels) == 0:
        con.log("No normal/wide/double labels available for benchmarking.")
        return None

    split = _prepare_classifier_train_test_split(waveforms, labels, records=records)
    X_train = split["X_train"]
    X_test = split["X_test"]
    y_train = split["y_train"]
    y_test = split["y_test"]
    labels_sorted = split["labels_sorted"]
    test_records = split["test_records"]
    n_components = _get_pca_n_components(X_train)

    con.log("\n" + "=" * 60)
    con.log("PULSE CLASSIFIER BENCHMARK")
    con.log("=" * 60)
    con.log(f"Labels: {labels_path}")
    con.log(f"Train/test split: {len(y_train)} / {len(y_test)} pulses")
    con.log(f"PCA components (when used): {n_components}")
    con.log(f"Output directory: {benchmark_dir}")
    con.log("=" * 60)

    pca_plot_train_test_path = pca_plot_dir / "dataset_pca_train_test.png"
    pca_plot_full_fit_path = pca_plot_dir / "dataset_pca_full_fit.png"
    con.log("Saving shared PCA scatter plots (classifier-independent)...")
    plot_benchmark_pca_scatter(
        X_train,
        y_train,
        X_test,
        y_test,
        pca_plot_train_test_path,
        title="Labeled pulses in PCA space",
        fit_on="train",
        show_splits=("train", "test"),
    )
    plot_benchmark_pca_scatter(
        X_train,
        y_train,
        X_test,
        y_test,
        pca_plot_full_fit_path,
        title="Labeled pulses in PCA space",
        fit_on="full",
        show_splits=("train", "test"),
    )
    con.log(
        "PCA plots are identical across classifier runs because they visualize "
        "waveform geometry, not model predictions."
    )

    classifier_estimators = _build_benchmark_classifier_pipelines()
    feature_spaces = ("pca", "raw")
    results = []

    for classifier_name, estimator in classifier_estimators.items():
        for feature_space in feature_spaces:
            run_id = f"{classifier_name}__{feature_space}"
            con.log(f"Training {classifier_name} on {feature_space} features...")

            pipeline = _wrap_benchmark_pipeline(
                estimator, feature_space, n_components
            )
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            metrics = _compute_multiclass_metrics(y_test, y_pred, labels_sorted)

            waveform_plot_path = waveform_plot_dir / f"{run_id}_waveforms.png"
            plot_benchmark_waveform_sanity(
                X_test,
                y_test,
                y_pred,
                test_records,
                waveform_plot_path,
                title=(
                    f"Test-set waveform examples | {classifier_name} | "
                    f"{feature_space} features"
                ),
            )

            run_result = {
                "run_id": run_id,
                "classifier": classifier_name,
                "feature_space": feature_space,
                **metrics,
                "pca_plot_train_test": str(pca_plot_train_test_path),
                "pca_plot_full_fit": str(pca_plot_full_fit_path),
                "waveform_plot": str(waveform_plot_path),
            }
            results.append(run_result)

            con.log(
                f"  macro F1={metrics['macro_f1']:.3f}, "
                f"weighted F1={metrics['weighted_f1']:.3f}"
            )

    best_rows = _print_benchmark_comparison_table(results)
    _summarize_benchmark_findings(results, best_rows)

    metrics_json_path = benchmark_dir / "benchmark_metrics.json"
    metrics_csv_path = benchmark_dir / "benchmark_metrics.csv"
    best_macro_f1 = max(row["macro_f1"] for row in results)

    with metrics_json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "labels_path": str(labels_path),
                "train_size": int(len(y_train)),
                "test_size": int(len(y_test)),
                "pca_n_components": int(n_components),
                "pca_plot_train_test": str(pca_plot_train_test_path),
                "pca_plot_full_fit": str(pca_plot_full_fit_path),
                "best_macro_f1": float(best_macro_f1),
                "best_runs": [
                    {
                        "run_id": row["run_id"],
                        "classifier": row["classifier"],
                        "feature_space": row["feature_space"],
                        "macro_f1": row["macro_f1"],
                    }
                    for row in best_rows
                ],
                "results": results,
            },
            f,
            indent=2,
        )

    fieldnames = [
        "run_id",
        "classifier",
        "feature_space",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "is_best",
        "pca_plot_train_test",
        "pca_plot_full_fit",
        "waveform_plot",
    ]
    with metrics_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    **{key: row[key] for key in fieldnames if key != "is_best"},
                    "is_best": row["macro_f1"] == best_macro_f1,
                }
            )

    con.log(f"Saved benchmark metrics to {metrics_json_path}")
    con.log(f"Saved benchmark metrics to {metrics_csv_path}")
    con.log(f"Saved PCA plots to {pca_plot_dir}")
    con.log(f"Saved waveform sanity plots to {waveform_plot_dir}")

    return {
        "benchmark_dir": benchmark_dir,
        "results": results,
        "best_runs": best_rows,
    }


def sample_naturalistic_pulses(
    data_path,
    n_pulses=400,
    max_files=8,
    random_seed=42,
):
    """
    Draw a small random sample of pulses from H5 files (natural prevalence).

    Returns prepared 1D waveforms, multi-channel raw pulses, sample rates, and
    rule-based double/wide flags for the same sample.
    """
    path_list = get_path_list(Path(data_path))
    if not path_list:
        raise FileNotFoundError(f"No H5 files under {data_path}")

    rng = np.random.default_rng(random_seed)
    file_order = rng.permutation(len(path_list))
    selected_files = [path_list[int(i)] for i in file_order[:max_files]]

    pool = []
    for file_path in selected_files:
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
                candidate_indices = np.where(block.data_arrays["predicted_labels"][:] == 1)[0]
            else:
                candidate_indices = np.arange(num_pulses)
            fs = float(file.sections["pulses_metadata"]["metadata"]["samplerate"])
            for pulse_idx in candidate_indices:
                pool.append((str(file_path), int(pulse_idx), fs))
        finally:
            file.close()

    if not pool:
        raise RuntimeError("No candidate pulses found for naturalistic sampling.")

    sample_size = min(n_pulses, len(pool))
    chosen = [pool[int(i)] for i in rng.choice(len(pool), size=sample_size, replace=False)]

    # Group by file for efficient loading
    by_file = {}
    for file_path, pulse_idx, fs in chosen:
        by_file.setdefault(file_path, []).append((pulse_idx, fs))

    traces = []
    raw_list = []
    sample_rates = []
    rule_double = []
    rule_wide = []

    for file_path, items in by_file.items():
        file = open_h5(file_path, nixio.FileMode.ReadOnly)
        if file is None:
            continue
        try:
            block = get_pulse_block(file)
            raw_pulses = block.data_arrays["raw_pulses"]
            for pulse_idx, fs in items:
                pulse_data = np.asarray(raw_pulses[pulse_idx][:], dtype=float)
                trace, _ = get_representative_waveform(pulse_data)
                is_double, _ = detect_double_pulse(pulse_data, fs)
                is_wide, _ = detect_wide_pulse(pulse_data, fs)
                traces.append(trace)
                raw_list.append(pulse_data)
                sample_rates.append(fs)
                rule_double.append(bool(is_double))
                rule_wide.append(bool(is_wide))
        finally:
            file.close()

    return {
        "traces": np.asarray(traces, dtype=float),
        "raw_pulses": raw_list,
        "sample_rates": np.asarray(sample_rates, dtype=float),
        "rule_double": np.asarray(rule_double, dtype=bool),
        "rule_wide": np.asarray(rule_wide, dtype=bool),
        "n_pool": len(pool),
        "n_files": len(by_file),
    }


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


def tune_classifier_decisions(
    data_path=None,
    labels_path=None,
    n_natural=400,
    max_files=8,
    random_seed=42,
):
    """
    Compare decision policies without writing markers.

    Evaluates on:
      1) stratified holdout of the (biased) labeled set — optimistic P/R
      2) small naturalistic unlabeled sample — predicted rates vs rule-based

    Does not claim true naturalistic precision without manual labels.
    """
    ml_paths = get_default_ml_paths()
    labels_path = Path(labels_path) if labels_path else ml_paths["labels"]
    data_path = Path(data_path) if data_path else H5_DIR
    out_dir = ml_paths["base"] / "decision_tuning"
    out_dir.mkdir(parents=True, exist_ok=True)

    waveforms, labels, _ = load_labeled_dataset(labels_path)
    active = np.isin(labels, list(LABELING_PULSE_CLASSES))
    waveforms, labels = waveforms[active], labels[active]

    split = _prepare_classifier_train_test_split(waveforms, labels)
    X_train, X_test = split["X_train"], split["X_test"]
    y_train, y_test = split["y_train"], split["y_test"]
    labels_sorted = split["labels_sorted"]
    train_prior = class_prior_from_labels(y_train, class_ids=labels_sorted)
    train_prior = {int(k): float(v) for k, v in train_prior.items()}
    n_components = _get_pca_n_components(X_train)

    con.log("\n" + "=" * 60)
    con.log("DECISION POLICY TUNING (no marker writes)")
    con.log("=" * 60)
    con.log(f"Labeled holdout: train={len(y_train)} test={len(y_test)}")
    con.log(f"Train prior: {train_prior}")

    con.log("Sampling naturalistic pulses...")
    natural = sample_naturalistic_pulses(
        data_path,
        n_pulses=n_natural,
        max_files=max_files,
        random_seed=random_seed,
    )
    rule_preds = np.zeros(len(natural["traces"]), dtype=np.int64)
    rule_preds[natural["rule_wide"]] = 1
    # Prefer double over wide if both fire
    rule_preds[natural["rule_double"]] = 2
    rule_rates = _rate_dict(rule_preds)
    # Empirical prior from rule-based on this sample (floored to avoid zeros)
    rule_prior = {
        0: max(rule_rates["normal"]["rate"], 0.5),
        1: max(rule_rates["wide"]["rate"], 0.005),
        2: max(rule_rates["double"]["rate"], 0.002),
    }
    # Renormalize
    s = sum(rule_prior.values())
    rule_prior = {k: v / s for k, v in rule_prior.items()}

    con.log(
        f"Naturalistic sample: n={len(natural['traces'])} from "
        f"{natural['n_files']} files (pool={natural['n_pool']})"
    )
    con.log(
        "Rule-based rates on sample: "
        + ", ".join(f"{k}={v['rate']:.3%}" for k, v in rule_rates.items())
    )
    con.log(f"Assumed natural prior (default): {DEFAULT_NATURAL_PRIOR}")
    con.log(f"Rule-estimated prior: {rule_prior}")

    policies = {
        "A_hard_predict_balanced_weights": {
            "rf_class_weight": "balanced",
            "use_prior_reweight": False,
            "min_proba": {},
            "rule_gate_double": False,
            "use_hard_predict": True,
        },
        "B_hard_predict_no_class_weight": {
            "rf_class_weight": None,
            "use_prior_reweight": False,
            "min_proba": {},
            "rule_gate_double": False,
            "use_hard_predict": True,
        },
        "C_threshold_t050": {
            "rf_class_weight": None,
            "use_prior_reweight": False,
            "min_proba": {1: 0.50, 2: 0.50},
            "rule_gate_double": False,
            "use_hard_predict": False,
        },
        "D_threshold_t070": {
            "rf_class_weight": None,
            "use_prior_reweight": False,
            "min_proba": {1: 0.60, 2: 0.70},
            "rule_gate_double": False,
            "use_hard_predict": False,
        },
        "E_threshold_t085": {
            "rf_class_weight": None,
            "use_prior_reweight": False,
            "min_proba": {1: 0.65, 2: 0.85},
            "rule_gate_double": False,
            "use_hard_predict": False,
        },
        "F_threshold_t090": {
            "rf_class_weight": None,
            "use_prior_reweight": False,
            "min_proba": {1: 0.70, 2: 0.90},
            "rule_gate_double": False,
            "use_hard_predict": False,
        },
        "G_mild_prior_argmax": {
            "rf_class_weight": None,
            "use_prior_reweight": True,
            "natural_prior": {0: 0.90, 1: 0.08, 2: 0.02},
            "min_proba": {},
            "rule_gate_double": False,
            "use_hard_predict": False,
            "argmax_after_reweight": True,
        },
        "H_rule_prior_argmax": {
            "rf_class_weight": None,
            "use_prior_reweight": True,
            "natural_prior": rule_prior,
            "min_proba": {},
            "rule_gate_double": False,
            "use_hard_predict": False,
            "argmax_after_reweight": True,
        },
        "I_threshold_t070_rule_gate": {
            "rf_class_weight": None,
            "use_prior_reweight": False,
            "min_proba": {1: 0.60, 2: 0.70},
            "rule_gate_double": True,
            "use_hard_predict": False,
        },
        "J_threshold_t050_rule_gate": {
            "rf_class_weight": None,
            "use_prior_reweight": False,
            "min_proba": {1: 0.50, 2: 0.50},
            "rule_gate_double": True,
            "use_hard_predict": False,
        },
        "K_mild_prior_plus_t060": {
            "rf_class_weight": None,
            "use_prior_reweight": True,
            "natural_prior": {0: 0.90, 1: 0.08, 2: 0.02},
            "min_proba": {1: 0.40, 2: 0.60},
            "rule_gate_double": False,
            "use_hard_predict": False,
        },
    }

    # Fit one RF per class_weight setting and reuse
    fitted = {}
    for weight_key in ("balanced", None):
        pipe = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("pca", _make_pca_estimator(n_components)),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight=weight_key,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        pipe.fit(X_train, y_train)
        fitted[weight_key] = pipe

    X_nat = prepare_classifier_waveforms(natural["traces"], fitted[None])

    results = []
    for name, pol in policies.items():
        clf = fitted[pol["rf_class_weight"]]
        decision_config = {
            "use_prior_reweight": pol.get("use_prior_reweight", False),
            "natural_prior": {
                int(k): float(v)
                for k, v in pol.get("natural_prior", DEFAULT_NATURAL_PRIOR).items()
            },
            "min_proba": {int(k): float(v) for k, v in pol.get("min_proba", {}).items()},
            "default_class": 0,
            "rule_gate_double": pol.get("rule_gate_double", False),
            "rf_class_weight": pol["rf_class_weight"],
        }

        if pol.get("use_hard_predict"):
            y_hold = clf.predict(X_test)
            y_nat = clf.predict(X_nat)
        elif pol.get("argmax_after_reweight"):
            proba = clf.predict_proba(X_test)
            if decision_config["use_prior_reweight"]:
                proba = reweight_class_probabilities(
                    proba,
                    clf.classes_,
                    train_prior,
                    decision_config["natural_prior"],
                )
            y_hold = np.asarray(clf.classes_, dtype=np.int64)[np.argmax(proba, axis=1)]
            proba_n = clf.predict_proba(X_nat)
            if decision_config["use_prior_reweight"]:
                proba_n = reweight_class_probabilities(
                    proba_n,
                    clf.classes_,
                    train_prior,
                    decision_config["natural_prior"],
                )
            y_nat = np.asarray(clf.classes_, dtype=np.int64)[np.argmax(proba_n, axis=1)]
        else:
            y_hold, _ = predict_special_pulse_classes(
                clf,
                X_test,
                decision_config=decision_config,
                train_prior=train_prior,
            )
            y_nat, _ = predict_special_pulse_classes(
                clf,
                X_nat,
                decision_config=decision_config,
                train_prior=train_prior,
                pulse_waveforms_raw=(
                    natural["raw_pulses"]
                    if decision_config["rule_gate_double"]
                    else None
                ),
                sample_rates=(
                    natural["sample_rates"]
                    if decision_config["rule_gate_double"]
                    else None
                ),
            )

        hold_metrics = _compute_multiclass_metrics(y_test, y_hold, labels_sorted)
        double_hold = _binary_prf(y_test, y_hold, 2)
        wide_hold = _binary_prf(y_test, y_hold, 1)
        nat_rates = _rate_dict(y_nat)

        row = {
            "policy": name,
            "decision_config": decision_config,
            "holdout_macro_f1": hold_metrics["macro_f1"],
            "holdout_weighted_f1": hold_metrics["weighted_f1"],
            "holdout_double": double_hold,
            "holdout_wide": wide_hold,
            "natural_rates": nat_rates,
            "natural_double_rate": nat_rates["double"]["rate"],
            "natural_wide_rate": nat_rates["wide"]["rate"],
            "natural_normal_rate": nat_rates["normal"]["rate"],
            "rule_double_rate": rule_rates["double"]["rate"],
            "rule_wide_rate": rule_rates["wide"]["rate"],
        }
        results.append(row)

        con.log(
            f"{name}: hold macroF1={hold_metrics['macro_f1']:.3f} "
            f"double P/R={double_hold['precision']:.2f}/{double_hold['recall']:.2f} | "
            f"nat rates N/W/D="
            f"{nat_rates['normal']['rate']:.3%}/"
            f"{nat_rates['wide']['rate']:.3%}/"
            f"{nat_rates['double']['rate']:.3%}"
        )

    # Prefer policies whose naturalistic double rate is near rule-based (and ≪ 5%),
    # then maximize holdout double precision, then macro F1.
    def rank_key(r):
        double_ok = r["natural_double_rate"] < 0.05
        return (
            0 if double_ok else 1,
            abs(r["natural_double_rate"] - r["rule_double_rate"]),
            abs(r["natural_wide_rate"] - r["rule_wide_rate"]),
            -r["holdout_double"]["precision"],
            -r["holdout_double"]["f1"],
            -r["holdout_macro_f1"],
        )

    ranked = sorted(results, key=rank_key)
    best = ranked[0]

    summary = {
        "labels_path": str(labels_path),
        "data_path": str(data_path),
        "train_prior": train_prior,
        "natural_prior_assumed": DEFAULT_NATURAL_PRIOR,
        "rule_estimated_prior": rule_prior,
        "naturalistic_n": int(len(natural["traces"])),
        "rule_based_rates": rule_rates,
        "note": (
            "Holdout P/R is on a balanced labeled set and overestimates rare-class "
            "performance in production. Naturalistic rates have no ground truth; "
            "compare to rule-based rates and biological expectation (doubles ≪ few %)."
        ),
        "results": results,
        "recommended_policy": best["policy"],
        "recommended_decision_config": best["decision_config"],
    }

    out_json = out_dir / "decision_tuning_results.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    table = Table(title="Decision policy comparison")
    table.add_column("Policy")
    table.add_column("Hold macroF1", justify="right")
    table.add_column("Dbl P/R", justify="right")
    table.add_column("Nat double%", justify="right")
    table.add_column("Nat wide%", justify="right")
    table.add_column("Nat normal%", justify="right")
    for r in results:
        marker = " *" if r["policy"] == best["policy"] else ""
        table.add_row(
            r["policy"] + marker,
            f"{r['holdout_macro_f1']:.3f}",
            f"{r['holdout_double']['precision']:.2f}/{r['holdout_double']['recall']:.2f}",
            f"{100 * r['natural_double_rate']:.2f}",
            f"{100 * r['natural_wide_rate']:.2f}",
            f"{100 * r['natural_normal_rate']:.2f}",
        )
    con.print(table)
    con.log(
        f"Rule-based on same sample: double={100 * rule_rates['double']['rate']:.2f}%, "
        f"wide={100 * rule_rates['wide']['rate']:.2f}%"
    )
    con.log(f"Recommended (rate-aware): {best['policy']}")
    con.log(f"Saved {out_json}")
    return summary

