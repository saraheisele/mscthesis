"""Correlate special-pulse activity histograms with the all-pulses reference.

Analysis part: pulse-shape correlation (Part 1c of Berlin activity analysis).
Dependencies: data_paths, pulse_config; requires eel_data_preprocessing.py output.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from path_setup import setup_script_paths

setup_script_paths(__file__)

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from data_paths import ACTIVITY_HISTOGRAMS_DIR, PULSE_SHAPE_CORRELATION_DIR
from pulse_config import SPECIAL_PULSE_TYPES, TIMESCALES

BASE = ACTIVITY_HISTOGRAMS_DIR
OUTPUT_DIR = PULSE_SHAPE_CORRELATION_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PULSE_TYPES = SPECIAL_PULSE_TYPES


def load_hist(subdir):
    path = BASE / subdir / "berlin_dummypulses_pulse_rate_hz_hist_dict.npz"
    data = np.load(path)
    return {key: data[key] for key in data.files}


def sig_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def correlate_pair(x, y):
    mask = ~(np.isnan(x) | np.isnan(y))
    xv, yv = x[mask], y[mask]
    n = len(xv)
    if n < 3:
        return {
            "n": n,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_r": np.nan,
            "spearman_p": np.nan,
        }
    pearson_r, pearson_p = stats.pearsonr(xv, yv)
    spearman_r, spearman_p = stats.spearmanr(xv, yv)
    return {
        "n": n,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
    }


def cross_corr_peak(x, y):
    mask = ~(np.isnan(x) | np.isnan(y))
    xv, yv = x[mask].astype(float), y[mask].astype(float)
    if len(xv) < 3:
        return np.nan, np.nan
    xv = xv - xv.mean()
    yv = yv - yv.mean()
    if xv.std() == 0 or yv.std() == 0:
        return np.nan, np.nan
    c = np.correlate(xv, yv, mode="full")
    c /= np.sqrt(np.sum(xv**2) * np.sum(yv**2))
    lags = np.arange(-len(xv) + 1, len(xv))
    peak_idx = np.argmax(c)
    return c[peak_idx], lags[peak_idx]


def run_analysis():
    all_hist = load_hist("all_pulses_hist")
    shape_hists = {key: load_hist(cfg["hist_subdir"]) for key, cfg in PULSE_TYPES.items()}

    results = []
    for timescale in TIMESCALES:
        ref = all_hist[timescale]
        for pulse_type, cfg in PULSE_TYPES.items():
            shape = shape_hists[pulse_type][timescale]
            corr = correlate_pair(ref, shape)
            xcorr_peak, xcorr_lag = cross_corr_peak(ref, shape)
            results.append(
                {
                    "timescale": timescale,
                    "pulse_type": pulse_type,
                    "label": cfg["label"],
                    "xcorr_peak": xcorr_peak,
                    "xcorr_lag": xcorr_lag,
                    **corr,
                }
            )
    return results


def print_summary(results):
    n_tests = len(results)
    alpha_bonf = 0.05 / n_tests

    print("=" * 90)
    print("PULSE SHAPE vs ALL PULSES — HISTOGRAM CORRELATION")
    print("=" * 90)
    print(
        "Pearson & Spearman on pulse-rate (Hz) bins at zero lag; NaN bins excluded.\n"
    )

    for timescale in TIMESCALES:
        print(f"\n{'─' * 90}")
        print(f"TIMESCALE: {timescale}")
        print(f"{'─' * 90}")
        print(
            f"{'Pulse type':<16} {'n':>5} {'Pearson r':>10} {'p':>10} "
            f"{'Spearman ρ':>11} {'p':>10} {'X-corr peak':>12} {'lag':>6}"
        )
        for row in results:
            if row["timescale"] != timescale:
                continue
            print(
                f"{row['label']:<16} {row['n']:>5} {row['pearson_r']:>10.3f} "
                f"{row['pearson_p']:>10.4f}{sig_stars(row['pearson_p']):<3} "
                f"{row['spearman_r']:>11.3f} {row['spearman_p']:>10.4f}"
                f"{sig_stars(row['spearman_p']):<3} "
                f"{row['xcorr_peak']:>12.3f} {row['xcorr_lag']:>6}"
            )

    sig_pearson = [r for r in results if r["pearson_p"] < 0.05]
    sig_bonf = [r for r in results if r["pearson_p"] < alpha_bonf]
    print(f"\nSignificant Pearson (p < 0.05): {len(sig_pearson)}/{n_tests}")
    print(f"Bonferroni-significant Pearson (α = {alpha_bonf:.4f}): {len(sig_bonf)}/{n_tests}")


def plot_correlation_heatmap(results):
    timescales = TIMESCALES
    pulse_types = list(PULSE_TYPES.keys())
    labels = [PULSE_TYPES[p]["label"] for p in pulse_types]

    pearson = np.full((len(pulse_types), len(timescales)), np.nan)
    spearman = np.full_like(pearson, np.nan)
    for row in results:
        i = pulse_types.index(row["pulse_type"])
        j = timescales.index(row["timescale"])
        pearson[i, j] = row["pearson_r"]
        spearman[i, j] = row["spearman_r"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, matrix, title in zip(
        axes,
        [pearson, spearman],
        ["Pearson r", "Spearman ρ"],
    ):
        im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(timescales)))
        ax.set_xticklabels(timescales, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_title(title)
        for i in range(len(labels)):
            for j in range(len(timescales)):
                val = matrix[i, j]
                if np.isnan(val):
                    continue
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Pulse shape vs all-pulses histogram correlation by timescale")
    plt.tight_layout()
    out = OUTPUT_DIR / "pulse_shape_correlation_heatmap.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")


def main():
    results = run_analysis()
    print_summary(results)
    plot_correlation_heatmap(results)


if __name__ == "__main__":
    main()
