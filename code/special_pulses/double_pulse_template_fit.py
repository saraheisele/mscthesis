"""Fit double pulses as sums of two scaled, width-adjusted normal pulse templates.

Analysis part: pulse shape modelling (extends prototype_pulse_plots overlay).
Dependencies: prototype_pulse_plots, data_paths, presentation_style.

Model: y(t) = a1*T_w(t-t1) + a2*T_w(t-t2), with free peak positions, shared width
scale w, and fixed zero baseline. Double pulses are sampled from the RF classifier
(special_pulse_class), not rule-based is_double_peak markers.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from path_setup import setup_script_paths

setup_script_paths(__file__)

import matplotlib.pyplot as plt
import nixio
import numpy as np
from rich.console import Console
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from data_paths import H5_DIR, PULSE_SHAPE_PROTOTYPES_DIR
from presentation_style import (
    LEGEND_LOC,
    THESIS_COLORS,
    apply_presentation_style,
    pulse_shape_color,
    save_thesis_figure,
)
from double_peaks_detection import apply_special_pulse_classifier
from h5_io import get_pulse_block, open_h5
from special_pulses.prototype_pulse_plots import (
    CLASS_IDS,
    align_waveforms,
    baseline_correct,
    collect_classifier_pulse_indices,
    double_peak_indices,
    double_valley_index,
    get_biggest_unclipped_waveform,
    load_all_waveforms,
    normalize_trace,
    shift_waveform,
)

console = Console()

OUTPUT_DIR = PULSE_SHAPE_PROTOTYPES_DIR.parent / "double_pulse_template_fit"
SAMPLE_SIZE = 500
TEMPLATE_SAMPLE_SIZE = 5000
RANDOM_SEED = 42
WIDTH_BOUNDS = (0.25, 1.5)

NORMAL_COLOR = pulse_shape_color("normal")
DOUBLE_COLOR = pulse_shape_color("double")
REFERENCE_COLOR = THESIS_COLORS[6]


@dataclass
class PulseData:
    waveform: np.ndarray
    peak1: float
    peak2: float
    separation_ms: float


@dataclass
class FitResult:
    model: str
    a1: float
    a2: float
    t1_samples: float
    t2_samples: float
    delay_samples: float
    delay_ms: float
    amplitude_ratio: float
    width_scale: float
    r2: float
    rmse: float
    success: bool


def scaled_shifted_template(
    template: np.ndarray, peak_pos: float, width_factor: float
) -> np.ndarray:
    """Place template peak at peak_pos; width_factor < 1 narrows, > 1 widens."""
    peak_idx = float(np.argmax(template))
    x = np.arange(len(template), dtype=float)
    source_x = peak_idx + (x - peak_pos) / width_factor
    return np.interp(x, source_x, template, left=0.0, right=0.0)


def build_normal_template(
    data_path,
    template_sample_size: int = TEMPLATE_SAMPLE_SIZE,
    random_seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, float]:
    entries = collect_classifier_pulse_indices(data_path, CLASS_IDS["normal"])
    total = len(entries)
    if total > template_sample_size:
        rng = np.random.default_rng(random_seed)
        pick = rng.choice(total, size=template_sample_size, replace=False)
        entries = [entries[i] for i in pick]
        console.log(
            f"Template built from {template_sample_size:,} / {total:,} sampled normal pulses"
        )
    corrected, normalized, fs = load_all_waveforms(entries, align_mode="maximum")
    if not normalized:
        raise RuntimeError("No normal pulses found for template construction.")
    aligned = align_waveforms(normalized, corrected, "maximum", fs)
    console.log(f"Normal template: n={len(aligned):,}, length={len(aligned[0])} samples")
    template = np.mean(aligned, axis=0)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUTPUT_DIR / "normal_template.npz", template=template, fs=fs)
    return template, fs


def load_cached_template() -> tuple[np.ndarray, float] | None:
    cache = OUTPUT_DIR / "normal_template.npz"
    if not cache.exists():
        return None
    data = np.load(cache)
    return data["template"], float(data["fs"])


def _peak_guess_on_aligned(waveform: np.ndarray, fs: float, center: int) -> tuple[float, float]:
    """Fallback peak positions on an aligned waveform (for fit initialization only)."""
    prominence = 0.05 * np.max(waveform)
    peaks, _ = find_peaks(waveform, prominence=prominence)
    if len(peaks) >= 2:
        top2 = peaks[np.argsort(waveform[peaks])[-2:]]
        return float(min(top2)), float(max(top2))
    half_sep = 0.00075 * fs
    return float(center - half_sep), float(center + half_sep)


def prepare_rf_double_pulse(
    corrected: np.ndarray, normalized: np.ndarray, fs: float
) -> PulseData:
    """Valley-align RF-labelled doubles; keep all pulses (no rule-based rejection)."""
    center = len(normalized) // 2
    peaks_corr = double_peak_indices(corrected, fs)

    if peaks_corr is not None:
        valley = double_valley_index(corrected, fs)
        align_shift = center - valley
        t1 = float(peaks_corr[0] + align_shift)
        t2 = float(peaks_corr[1] + align_shift)
    else:
        align_shift = center - int(np.argmax(corrected))
        waveform_tmp = shift_waveform(normalized, align_shift)
        t1, t2 = _peak_guess_on_aligned(waveform_tmp, fs, center)

    waveform = shift_waveform(normalized, align_shift)
    if t2 <= t1:
        t1, t2 = t2, t1
    separation_ms = (t2 - t1) / fs * 1000
    return PulseData(waveform=waveform, peak1=t1, peak2=t2, separation_ms=separation_ms)


def load_rf_double_sample(
    data_path, sample_size: int, random_seed: int
) -> tuple[list[PulseData], float]:
    """Random sample of RF-classified double pulses (no is_double_peak filter)."""
    entries = collect_classifier_pulse_indices(data_path, CLASS_IDS["double"])
    rng = np.random.default_rng(random_seed)
    shuffled = list(entries)
    rng.shuffle(shuffled)

    pulses: list[PulseData] = []
    fs = None
    open_files: dict = {}

    try:
        for file_path, pulse_idx in shuffled:
            if len(pulses) >= sample_size:
                break

            file_key = str(file_path)
            if file_key not in open_files:
                file = open_h5(file_path, nixio.FileMode.ReadOnly)
                if file is None:
                    continue
                block = get_pulse_block(file)
                open_files[file_key] = (file, block)

            file, block = open_files[file_key]
            pulse_data = block.data_arrays["raw_pulses"][pulse_idx][:]
            if fs is None:
                fs = float(file.sections["pulses_metadata"]["metadata"]["samplerate"])

            trace, _ = get_biggest_unclipped_waveform(pulse_data)
            corrected = baseline_correct(trace)
            normalized = normalize_trace(corrected)
            pulses.append(prepare_rf_double_pulse(corrected, normalized, fs))
    finally:
        for file, _ in open_files.values():
            file.close()

    if not pulses:
        raise RuntimeError("No RF-classified double pulses found.")
    return pulses, fs


def fit_metrics(y: np.ndarray, yhat: np.ndarray) -> tuple[float, float]:
    residual = y - yhat
    rmse = float(np.sqrt(np.mean(residual**2)))
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return r2, rmse


def _failed_result(model: str) -> FitResult:
    return FitResult(
        model=model,
        a1=np.nan,
        a2=np.nan,
        t1_samples=np.nan,
        t2_samples=np.nan,
        delay_samples=np.nan,
        delay_ms=np.nan,
        amplitude_ratio=np.nan,
        width_scale=np.nan,
        r2=np.nan,
        rmse=np.nan,
        success=False,
    )


def _order_peaks(a1: float, a2: float, t1: float, t2: float) -> tuple[float, float, float, float]:
    if t2 < t1:
        return a2, a1, t2, t1
    return a1, a2, t1, t2


def _make_result(
    model: str,
    a1: float,
    a2: float,
    t1: float,
    t2: float,
    width_scale: float,
    y: np.ndarray,
    yhat: np.ndarray,
    fs: float,
) -> FitResult:
    a1, a2, t1, t2 = _order_peaks(a1, a2, t1, t2)
    r2, rmse = fit_metrics(y, yhat)
    delay = t2 - t1
    return FitResult(
        model=model,
        a1=float(a1),
        a2=float(a2),
        t1_samples=float(t1),
        t2_samples=float(t2),
        delay_samples=float(delay),
        delay_ms=float(delay / fs * 1000),
        amplitude_ratio=float(a2 / a1) if a1 > 1e-6 else np.nan,
        width_scale=float(width_scale),
        r2=r2,
        rmse=rmse,
        success=np.isfinite(r2),
    )


def fit_free_peaks(
    y: np.ndarray,
    template: np.ndarray,
    t1_init: float,
    t2_init: float,
    fs: float,
    *,
    fit_width: bool,
) -> FitResult:
    """Free peak positions with fixed baseline; optional shared width scale."""
    n = len(y)
    min_sep = 0.0004 * fs
    p1i = int(np.clip(round(t1_init), 0, n - 1))
    p2i = int(np.clip(round(t2_init), 0, n - 1))
    a1_0 = float(max(y[p1i], 0.2))
    a2_0 = float(max(y[p2i], 0.2))

    if fit_width:
        def model(_x, a1, a2, t1, t2, width_scale):
            return (
                a1 * scaled_shifted_template(template, t1, width_scale)
                + a2 * scaled_shifted_template(template, t2, width_scale)
            )

        try:
            popt, _ = curve_fit(
                model,
                np.arange(n),
                y,
                p0=[a1_0, a2_0, t1_init, t2_init, 0.65],
                bounds=(
                    [0.01, 0.01, 0.0, 0.0, WIDTH_BOUNDS[0]],
                    [2.5, 2.5, float(n - 1), float(n - 1), WIDTH_BOUNDS[1]],
                ),
                maxfev=20_000,
            )
            a1, a2, t1, t2, width_scale = popt
            if t2 - t1 < min_sep:
                return _failed_result("free_width")
            yhat = model(np.arange(n), a1, a2, t1, t2, width_scale)
            return _make_result("free_width", a1, a2, t1, t2, width_scale, y, yhat, fs)
        except Exception:
            return _failed_result("free_width")

    def model_fixed(_x, a1, a2, t1, t2):
        return (
            a1 * scaled_shifted_template(template, t1, 1.0)
            + a2 * scaled_shifted_template(template, t2, 1.0)
        )

    try:
        popt, _ = curve_fit(
            model_fixed,
            np.arange(n),
            y,
            p0=[a1_0, a2_0, t1_init, t2_init],
            bounds=(
                [0.01, 0.01, 0.0, 0.0],
                [2.5, 2.5, float(n - 1), float(n - 1)],
            ),
            maxfev=20_000,
        )
        a1, a2, t1, t2 = popt
        if t2 - t1 < min_sep:
            return _failed_result("free_fixed_width")
        yhat = model_fixed(np.arange(n), a1, a2, t1, t2)
        return _make_result("free_fixed_width", a1, a2, t1, t2, 1.0, y, yhat, fs)
    except Exception:
        return _failed_result("free_fixed_width")


def fit_all_pulses(
    pulses: list[PulseData], template: np.ndarray, fs: float
) -> tuple[list[FitResult], list[FitResult]]:
    fixed_width = []
    scaled_width = []
    for pulse in pulses:
        fixed_width.append(
            fit_free_peaks(
                pulse.waveform, template, pulse.peak1, pulse.peak2, fs, fit_width=False
            )
        )
        scaled_width.append(
            fit_free_peaks(
                pulse.waveform, template, pulse.peak1, pulse.peak2, fs, fit_width=True
            )
        )
    return fixed_width, scaled_width


def reconstruct_components(
    result: FitResult, template: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    width = 1.0 if not np.isfinite(result.width_scale) else result.width_scale
    comp1 = result.a1 * scaled_shifted_template(template, result.t1_samples, width)
    comp2 = result.a2 * scaled_shifted_template(template, result.t2_samples, width)
    return comp1, comp2


def reconstruct_fit(result: FitResult, template: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    comp1, comp2 = reconstruct_components(result, template)
    return comp1 + comp2, comp1, comp2


def reconstruct_fit_convolution(
    result: FitResult, template: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine the two template components by discrete convolution (same length)."""
    comp1, comp2 = reconstruct_components(result, template)
    yhat = np.convolve(comp1, comp2, mode="same")
    return yhat, comp1, comp2


def aggregate_fit_params(results: list[FitResult], stat: str) -> FitResult:
    ok = [r for r in results if r.success and np.isfinite(r.r2)]
    if not ok:
        return _failed_result(stat)
    reducer = np.median if stat == "median" else np.mean

    def _agg(key: str) -> float:
        return float(reducer([getattr(r, key) for r in ok]))

    return FitResult(
        model=stat,
        a1=_agg("a1"),
        a2=_agg("a2"),
        t1_samples=_agg("t1_samples"),
        t2_samples=_agg("t2_samples"),
        delay_samples=_agg("delay_samples"),
        delay_ms=_agg("delay_ms"),
        amplitude_ratio=_agg("amplitude_ratio"),
        width_scale=_agg("width_scale"),
        r2=np.nan,
        rmse=np.nan,
        success=True,
    )


def save_aggregate_fit_params(
    mean_fit: FitResult,
    median_fit: FitResult,
    n_pulses: int,
    fs: float,
    output_dir: Path,
) -> Path:
    payload = {
        "pulse_source": "rf_classifier_special_pulse_class",
        "n_pulses": n_pulses,
        "sample_rate_hz": fs,
        "mean_fit": {
            **asdict(mean_fit),
            "t1_ms": mean_fit.t1_samples / fs * 1000,
            "t2_ms": mean_fit.t2_samples / fs * 1000,
        },
        "median_fit": {
            **asdict(median_fit),
            "t1_ms": median_fit.t1_samples / fs * 1000,
            "t2_ms": median_fit.t2_samples / fs * 1000,
        },
    }
    out = output_dir / "double_pulse_template_fit_params.json"
    with open(out, "w") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)

    from data_paths import THESIS_FIGURES_DIR

    thesis_out = THESIS_FIGURES_DIR / "pulse_shapes" / "double_pulse_template_fit_params.json"
    thesis_out.parent.mkdir(parents=True, exist_ok=True)
    with open(thesis_out, "w") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
    console.log(f"Saved fit parameters to {out}")
    return out


def plot_aggregate_fit(
    pulses: list[PulseData],
    width_results: list[FitResult],
    template: np.ndarray,
    fs: float,
    output_dir: Path,
):
    """Plot mean aggregate template fit and its two components."""
    mean_fit = aggregate_fit_params(width_results, "mean")
    median_fit = aggregate_fit_params(width_results, "median")
    mean_y = np.mean([p.waveform for p in pulses], axis=0)

    mean_yhat, mean_c1, mean_c2 = reconstruct_fit(mean_fit, template)
    median_yhat, _, _ = reconstruct_fit(median_fit, template)
    mean_fit.r2, mean_fit.rmse = fit_metrics(mean_y, mean_yhat)
    median_fit.r2, median_fit.rmse = fit_metrics(mean_y, median_yhat)

    save_aggregate_fit_params(mean_fit, median_fit, len(pulses), fs, output_dir)

    apply_presentation_style()
    time_ms = np.arange(len(template)) / fs * 1000
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        time_ms,
        mean_yhat,
        color=DOUBLE_COLOR,
        label=(
            f"Mean fit ($\\Delta t$={mean_fit.delay_ms:.2f} ms, "
            f"$w$={mean_fit.width_scale:.2f}, $R^2$={mean_fit.r2:.3f})"
        ),
        zorder=4,
    )
    ax.plot(
        time_ms,
        mean_c1,
        color=NORMAL_COLOR,
        linestyle="--",
        label=f"Mean comp. 1 ($a_1$={mean_fit.a1:.2f})",
        zorder=3,
    )
    ax.plot(
        time_ms,
        mean_c2,
        color=NORMAL_COLOR,
        linestyle=":",
        label=f"Mean comp. 2 ($a_2$={mean_fit.a2:.2f})",
        zorder=3,
    )

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Normalized amplitude")
    ax.set_title(
        f"Mean width-scaled template fit (n={len(pulses)} RF-classified doubles)"
    )
    ax.legend(loc=LEGEND_LOC)
    ax.set_ylim(-0.12, 1.12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = output_dir / "double_pulse_template_fit_example.png"
    fig.savefig(out, dpi=300)
    save_thesis_figure("pulse_shapes/double_pulse_template_fit_example.png", fig)
    plt.close(fig)
    console.log(f"Saved {out}")


def plot_parameter_distributions(
    pulses: list[PulseData],
    fixed_results: list[FitResult],
    width_results: list[FitResult],
    output_dir: Path,
):
    fixed_ok = [r for r in fixed_results if r.success and np.isfinite(r.r2)]
    width_ok = [r for r in width_results if r.success and np.isfinite(r.r2)]

    apply_presentation_style()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    bins = 30

    def _hist(ax, values_fixed, values_width, xlabel, title, xmax=None):
        if values_fixed:
            ax.hist(values_fixed, bins=bins, alpha=0.55, color=NORMAL_COLOR, label="$w=1$", density=True)
        if values_width:
            ax.hist(values_width, bins=bins, alpha=0.55, color=DOUBLE_COLOR, label="Free $w$", density=True)
        if xmax is not None:
            ax.set_xlim(0, xmax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend(loc=LEGEND_LOC)
        ax.grid(True, alpha=0.3)

    _hist(
        axes[0, 0],
        [r.delay_ms for r in fixed_ok],
        [r.delay_ms for r in width_ok],
        "Peak separation (ms)",
        "Inter-peak delay",
        xmax=2.5,
    )
    _hist(
        axes[0, 1],
        [r.width_scale for r in width_ok],
        [],
        "Width scale $w$",
        "Fitted width scale (free-$w$ model)",
        xmax=WIDTH_BOUNDS[1],
    )
    axes[0, 1].axvline(1.0, color=REFERENCE_COLOR, linestyle="--", label="$w=1$")
    axes[0, 1].legend(loc=LEGEND_LOC)
    _hist(axes[0, 2], [r.r2 for r in fixed_ok], [r.r2 for r in width_ok], r"$R^2$", "Goodness of fit")
    ratios_f = [r.amplitude_ratio for r in fixed_ok if np.isfinite(r.amplitude_ratio) and r.amplitude_ratio < 8]
    ratios_w = [r.amplitude_ratio for r in width_ok if np.isfinite(r.amplitude_ratio) and r.amplitude_ratio < 8]
    _hist(axes[1, 0], ratios_f, ratios_w, r"$a_2/a_1$", "Amplitude ratio", xmax=5)
    _hist(axes[1, 1], [r.a1 for r in fixed_ok], [r.a1 for r in width_ok], r"$a_1$", "First component scale", xmax=2)
    _hist(axes[1, 2], [r.rmse for r in fixed_ok], [r.rmse for r in width_ok], "RMSE", "Residual magnitude", xmax=0.3)

    fig.suptitle(f"Template-fit parameter distributions (n={len(width_ok)} doubles)")
    plt.tight_layout()
    out = output_dir / "double_pulse_template_fit_distributions.png"
    fig.savefig(out, dpi=300)
    save_thesis_figure("pulse_shapes/double_pulse_template_fit_distributions.png", fig)
    plt.close(fig)
    console.log(f"Saved {out}")


def _json_default(obj):
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def summarize_results(
    pulses: list[PulseData], fixed_results: list[FitResult], width_results: list[FitResult]
) -> dict:
    def _stats(results: list[FitResult], key: str) -> dict:
        values = np.array([getattr(r, key) for r in results if r.success and np.isfinite(getattr(r, key))])
        if values.size == 0:
            return {"n": 0}
        return {
            "n": int(values.size),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
        }

    detected = np.array([p.separation_ms for p in pulses], dtype=float)
    return {
        "n_total": len(pulses),
        "detected_separation_ms": {
            "mean": float(np.mean(detected)),
            "median": float(np.median(detected)),
            "std": float(np.std(detected)),
        },
        "fixed_width": {
            "n_success": int(sum(r.success for r in fixed_results)),
            "delay_ms": _stats(fixed_results, "delay_ms"),
            "amplitude_ratio": _stats(fixed_results, "amplitude_ratio"),
            "width_scale": _stats(fixed_results, "width_scale"),
            "r2": _stats(fixed_results, "r2"),
            "rmse": _stats(fixed_results, "rmse"),
        },
        "free_width": {
            "n_success": int(sum(r.success for r in width_results)),
            "delay_ms": _stats(width_results, "delay_ms"),
            "amplitude_ratio": _stats(width_results, "amplitude_ratio"),
            "width_scale": _stats(width_results, "width_scale"),
            "r2": _stats(width_results, "r2"),
            "rmse": _stats(width_results, "rmse"),
        },
    }


def main(
    data_path=H5_DIR,
    sample_size=SAMPLE_SIZE,
    random_seed=RANDOM_SEED,
    apply_classifier=False,
):
    apply_presentation_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    console.log("Building mean normal pulse template and fitting double pulses...")

    if apply_classifier:
        console.log("Applying Random Forest classifier to h5 files...")
        apply_special_pulse_classifier(data_path)

    cached = load_cached_template()
    if cached is not None:
        template, fs = cached
        console.log(f"Loaded cached normal template ({len(template)} samples)")
    else:
        console.log("Collecting normal pulses for template...")
        template, fs = build_normal_template(data_path)

    console.log("Collecting RF-classified double pulses for fitting...")
    pulses, fs = load_rf_double_sample(data_path, sample_size, random_seed)
    console.log(
        f"Fitting {len(pulses)} doubles (fs={fs:.0f} Hz); "
        f"median detected separation={np.median([p.separation_ms for p in pulses]):.3f} ms"
    )

    fixed_results, width_results = fit_all_pulses(pulses, template, fs)
    plot_aggregate_fit(pulses, width_results, template, fs, OUTPUT_DIR)
    plot_parameter_distributions(pulses, fixed_results, width_results, OUTPUT_DIR)

    summary = summarize_results(pulses, fixed_results, width_results)
    with open(OUTPUT_DIR / "double_pulse_template_fit_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2, default=_json_default)

    rows = []
    for i, (pulse, fixed_r, width_r) in enumerate(zip(pulses, fixed_results, width_results)):
        rows.append(
            {
                "index": i,
                "detected_separation_ms": pulse.separation_ms,
                "fixed_width": asdict(fixed_r),
                "free_width": asdict(width_r),
            }
        )
    with open(OUTPUT_DIR / "double_pulse_template_fit_results.json", "w") as handle:
        json.dump(rows, handle, indent=2, default=_json_default)

    console.log(
        f"Fixed width: median R²={summary['fixed_width']['r2'].get('median', float('nan')):.3f}, "
        f"median Δt={summary['fixed_width']['delay_ms'].get('median', float('nan')):.3f} ms"
    )
    console.log(
        f"Free width: median R²={summary['free_width']['r2'].get('median', float('nan')):.3f}, "
        f"median w={summary['free_width']['width_scale'].get('median', float('nan')):.3f}, "
        f"median Δt={summary['free_width']['delay_ms'].get('median', float('nan')):.3f} ms"
    )
    console.log(f"Summary written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
