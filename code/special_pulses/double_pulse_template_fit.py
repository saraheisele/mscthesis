"""Fit the prototype double pulse as a sum of two normal-pulse templates.

Analysis part: pulse shape modelling (extends prototype_pulse_plots overlay).
Dependencies: prototype_pulse_plots, data_paths, presentation_style.

Pipeline default
----------------
Fit the **prototype median** double waveform with **model A** (native-width
normal template, free times & amplitudes). Thesis figure:
``figures/pulse_shapes/double_pulse_template_fit_example.png``.

Why median (not mean): peak-separation and shape residuals are mildly
right-skewed; the median better represents a typical double without being
pulled by rare wide/odder events.

Why model A (not freer models): most honest answer to “can a double be two
normals in quick succession?” — same normal shape, only timing and gain;
no extra width freedom that invites overfitting.

Why direct fit (not mean-of-params): averaging per-pulse fit parameters is
not the same as fitting the aggregate waveform, and answers the biological
question less cleanly. Direct fit to the prototype is preferred.

Archived (kept in this file, not called from ``main``)
------------------------------------------------------
- Per-pulse fitting + mean-of-params example / parameter distributions
- Sample direct mean/median fits (``run_direct_mean_target_fits``) — used to
  compare mean-of-params vs direct fit on a morph-filtered RF sample
- Model hierarchy A–F panels (``run_mean_model_hierarchy``) — used to choose
  default model A
- Prototype-mean direct fits — median is the canonical prototype now

Model: y(t) = a1*T_w(t-t1) + a2*T_w(t-t2). Doubles for archived sample paths
are selected via special_pulse_class, then morph-filtered (H5 labels never
overwritten).
"""

from __future__ import annotations

import argparse
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

from data_paths import H5_DIR, PULSE_SHAPE_PROTOTYPES_DIR
from presentation_style import (
    LEGEND_LOC,
    THESIS_COLORS,
    apply_presentation_style,
    pulse_shape_color,
    save_thesis_figure,
    save_thesis_json,
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

# Canonical pipeline target / model (see module docstring).
PROTOTYPE_STAT = "median"  # "mean" kept in loaders / archived helpers only
DEFAULT_MODEL = "A_w1_free_t"
MODEL_KEYS = (
    "A_w1_free_t",
    "B_fixed_dt_w1",
    "C_fixed_dt_shared_w",
    "D_fixed_dt_indep_w",
    "E_free_shared_w",
    "F_free_indep_w",
)

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
    width_scale_2: float = np.nan  # second-component width; nan → use width_scale


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
    required_fs: float | None = None,
) -> tuple[np.ndarray, float]:
    entries = collect_classifier_pulse_indices(data_path, CLASS_IDS["normal"])
    total = len(entries)
    # Oversample when filtering by rate so we still reach template_sample_size.
    load_cap = None if required_fs is None else template_sample_size
    if total > template_sample_size and required_fs is None:
        rng = np.random.default_rng(random_seed)
        pick = rng.choice(total, size=template_sample_size, replace=False)
        entries = [entries[i] for i in pick]
        console.log(
            f"Template built from {template_sample_size:,} / {total:,} sampled normal pulses"
        )
    corrected, normalized, fs = load_all_waveforms(
        entries,
        align_mode="maximum",
        max_waveforms=load_cap,
        random_seed=random_seed,
        required_fs=required_fs,
    )
    if not normalized:
        raise RuntimeError("No normal pulses found for template construction.")
    if required_fs is not None and len(normalized) < template_sample_size:
        console.log(
            f"[yellow]Template used {len(normalized):,} normals at {fs:.0f} Hz "
            f"(requested {template_sample_size:,})"
        )
    elif required_fs is not None:
        console.log(
            f"Template built from {len(normalized):,} / {total:,} normal pulses "
            f"at {fs:.0f} Hz"
        )
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


def prepare_rf_double_pulse(
    corrected: np.ndarray, normalized: np.ndarray, fs: float
) -> PulseData | None:
    """Valley-align an RF-labelled double with detectable two-peak morphology.

    Returns None when morphological peak-finding fails. This is an analysis-only
    filter and does not change H5 labels.
    """
    peaks_corr = double_peak_indices(corrected, fs)
    if peaks_corr is None:
        return None

    center = len(normalized) // 2
    valley = double_valley_index(corrected, fs)
    align_shift = center - valley
    t1 = float(peaks_corr[0] + align_shift)
    t2 = float(peaks_corr[1] + align_shift)
    waveform = shift_waveform(normalized, align_shift)
    if t2 <= t1:
        t1, t2 = t2, t1
    separation_ms = (t2 - t1) / fs * 1000
    return PulseData(waveform=waveform, peak1=t1, peak2=t2, separation_ms=separation_ms)


def load_rf_double_sample(
    data_path,
    sample_size: int,
    random_seed: int,
    required_fs: float | None = None,
) -> tuple[list[PulseData], float]:
    """Random sample of RF doubles that also show two-peak morphology.

    If ``required_fs`` is set, only pulses from files at that samplerate are used
    (dummy data mixes 24 kHz and 48 kHz recordings).
    """
    entries = collect_classifier_pulse_indices(data_path, CLASS_IDS["double"])
    rng = np.random.default_rng(random_seed)
    shuffled = list(entries)
    rng.shuffle(shuffled)

    pulses: list[PulseData] = []
    fs = None
    open_files: dict = {}
    n_seen = 0
    n_skipped_morphology = 0
    n_skipped_rate = 0

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
                file_fs = float(
                    file.sections["pulses_metadata"]["metadata"]["samplerate"]
                )
                if required_fs is not None and abs(file_fs - required_fs) >= 1e-6:
                    file.close()
                    open_files[file_key] = None
                else:
                    open_files[file_key] = (file, block, file_fs)

            cached = open_files[file_key]
            if cached is None:
                n_skipped_rate += 1
                continue
            file, block, file_fs = cached
            pulse_data = block.data_arrays["raw_pulses"][pulse_idx][:]
            if fs is None:
                fs = file_fs

            n_seen += 1
            trace, _ = get_biggest_unclipped_waveform(pulse_data)
            corrected = baseline_correct(trace)
            normalized = normalize_trace(corrected)
            prepared = prepare_rf_double_pulse(corrected, normalized, fs)
            if prepared is None:
                n_skipped_morphology += 1
                continue
            pulses.append(prepared)
    finally:
        for cached in open_files.values():
            if cached is not None:
                cached[0].close()

    if not pulses:
        raise RuntimeError(
            "No RF-classified doubles with detectable two-peak morphology found."
        )
    rate_note = (
        f", skipped {n_skipped_rate} at other sample rates"
        if required_fs is not None
        else ""
    )
    console.log(
        f"Morphology filter: kept {len(pulses)} / {n_seen} inspected RF doubles "
        f"(skipped {n_skipped_morphology} without two detectable peaks{rate_note})"
    )
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
        width_scale_2=np.nan,
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
    width_scale_2: float = np.nan,
) -> FitResult:
    if t2 < t1:
        a1, a2, t1, t2 = a2, a1, t2, t1
        if np.isfinite(width_scale_2):
            width_scale, width_scale_2 = width_scale_2, width_scale
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
        width_scale_2=float(width_scale_2),
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


def fit_fixed_delta_t(
    y: np.ndarray,
    template: np.ndarray,
    t1: float,
    t2: float,
    fs: float,
    *,
    fit_width: bool,
    independent_widths: bool = False,
) -> FitResult:
    """Keep peak times fixed at detected positions; free amplitudes (± widths)."""
    n = len(y)
    p1i = int(np.clip(round(t1), 0, n - 1))
    p2i = int(np.clip(round(t2), 0, n - 1))
    a1_0 = float(max(y[p1i], 0.2))
    a2_0 = float(max(y[p2i], 0.2))

    if independent_widths:
        def model(_x, a1, a2, w1, w2):
            return (
                a1 * scaled_shifted_template(template, t1, w1)
                + a2 * scaled_shifted_template(template, t2, w2)
            )

        try:
            popt, _ = curve_fit(
                model,
                np.arange(n),
                y,
                p0=[a1_0, a2_0, 0.65, 0.65],
                bounds=(
                    [0.01, 0.01, WIDTH_BOUNDS[0], WIDTH_BOUNDS[0]],
                    [2.5, 2.5, WIDTH_BOUNDS[1], WIDTH_BOUNDS[1]],
                ),
                maxfev=20_000,
            )
            a1, a2, w1, w2 = popt
            yhat = model(np.arange(n), a1, a2, w1, w2)
            return _make_result(
                "fixed_dt_indep_w", a1, a2, t1, t2, w1, y, yhat, fs, width_scale_2=w2
            )
        except Exception:
            return _failed_result("fixed_dt_indep_w")

    if fit_width:
        def model(_x, a1, a2, width_scale):
            return (
                a1 * scaled_shifted_template(template, t1, width_scale)
                + a2 * scaled_shifted_template(template, t2, width_scale)
            )

        try:
            popt, _ = curve_fit(
                model,
                np.arange(n),
                y,
                p0=[a1_0, a2_0, 0.65],
                bounds=(
                    [0.01, 0.01, WIDTH_BOUNDS[0]],
                    [2.5, 2.5, WIDTH_BOUNDS[1]],
                ),
                maxfev=20_000,
            )
            a1, a2, width_scale = popt
            yhat = model(np.arange(n), a1, a2, width_scale)
            return _make_result("fixed_dt_shared_w", a1, a2, t1, t2, width_scale, y, yhat, fs)
        except Exception:
            return _failed_result("fixed_dt_shared_w")

    def model_fixed(_x, a1, a2):
        return (
            a1 * scaled_shifted_template(template, t1, 1.0)
            + a2 * scaled_shifted_template(template, t2, 1.0)
        )

    try:
        popt, _ = curve_fit(
            model_fixed,
            np.arange(n),
            y,
            p0=[a1_0, a2_0],
            bounds=([0.01, 0.01], [2.5, 2.5]),
            maxfev=20_000,
        )
        a1, a2 = popt
        yhat = model_fixed(np.arange(n), a1, a2)
        return _make_result("fixed_dt_w1", a1, a2, t1, t2, 1.0, y, yhat, fs)
    except Exception:
        return _failed_result("fixed_dt_w1")


def fit_independent_widths(
    y: np.ndarray,
    template: np.ndarray,
    t1_init: float,
    t2_init: float,
    fs: float,
) -> FitResult:
    """Free amplitudes, peak times, and independent width scales."""
    n = len(y)
    min_sep = 0.0004 * fs
    p1i = int(np.clip(round(t1_init), 0, n - 1))
    p2i = int(np.clip(round(t2_init), 0, n - 1))
    a1_0 = float(max(y[p1i], 0.2))
    a2_0 = float(max(y[p2i], 0.2))

    def model(_x, a1, a2, t1, t2, w1, w2):
        return (
            a1 * scaled_shifted_template(template, t1, w1)
            + a2 * scaled_shifted_template(template, t2, w2)
        )

    try:
        popt, _ = curve_fit(
            model,
            np.arange(n),
            y,
            p0=[a1_0, a2_0, t1_init, t2_init, 0.65, 0.65],
            bounds=(
                [0.01, 0.01, 0.0, 0.0, WIDTH_BOUNDS[0], WIDTH_BOUNDS[0]],
                [2.5, 2.5, float(n - 1), float(n - 1), WIDTH_BOUNDS[1], WIDTH_BOUNDS[1]],
            ),
            maxfev=30_000,
        )
        a1, a2, t1, t2, w1, w2 = popt
        if t2 - t1 < min_sep:
            return _failed_result("free_indep_w")
        yhat = model(np.arange(n), a1, a2, t1, t2, w1, w2)
        return _make_result(
            "free_indep_w", a1, a2, t1, t2, w1, y, yhat, fs, width_scale_2=w2
        )
    except Exception:
        return _failed_result("free_indep_w")


def fit_two_normals_model(
    y: np.ndarray,
    template: np.ndarray,
    fs: float,
    model_key: str = DEFAULT_MODEL,
) -> FitResult:
    """Fit one of the A–F two-normal models to a single target waveform."""
    if model_key not in MODEL_KEYS:
        raise ValueError(f"Unknown model {model_key!r}; choose from {MODEL_KEYS}")
    peaks = double_peak_indices(y, fs)
    if peaks is None:
        center = len(y) // 2
        half = 0.001 * fs
        t1, t2 = float(center - half), float(center + half)
    else:
        t1, t2 = float(peaks[0]), float(peaks[1])

    dispatch = {
        "A_w1_free_t": lambda: fit_free_peaks(
            y, template, t1, t2, fs, fit_width=False
        ),
        "B_fixed_dt_w1": lambda: fit_fixed_delta_t(
            y, template, t1, t2, fs, fit_width=False
        ),
        "C_fixed_dt_shared_w": lambda: fit_fixed_delta_t(
            y, template, t1, t2, fs, fit_width=True
        ),
        "D_fixed_dt_indep_w": lambda: fit_fixed_delta_t(
            y, template, t1, t2, fs, fit_width=True, independent_widths=True
        ),
        "E_free_shared_w": lambda: fit_free_peaks(
            y, template, t1, t2, fs, fit_width=True
        ),
        "F_free_indep_w": lambda: fit_independent_widths(y, template, t1, t2, fs),
    }
    result = dispatch[model_key]()
    if result.success:
        result.model = model_key
    return result


# ---------------------------------------------------------------------------
# Archived exploratory helpers (not called from main / pipeline).
# Kept for reproducibility of the model-selection and mean-of-params checks.
# ---------------------------------------------------------------------------


def fit_all_pulses(
    pulses: list[PulseData], template: np.ndarray, fs: float
) -> tuple[list[FitResult], list[FitResult]]:
    """ARCHIVED: per-pulse fits used by mean-of-params / distributions."""
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
    w1 = 1.0 if not np.isfinite(result.width_scale) else result.width_scale
    w2 = w1 if not np.isfinite(result.width_scale_2) else result.width_scale_2
    comp1 = result.a1 * scaled_shifted_template(template, result.t1_samples, w1)
    comp2 = result.a2 * scaled_shifted_template(template, result.t2_samples, w2)
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
        width_scale_2=_agg("width_scale_2"),
    )


def save_aggregate_fit_params(
    mean_fit: FitResult,
    median_fit: FitResult,
    n_pulses: int,
    fs: float,
    output_dir: Path,
) -> Path:
    """ARCHIVED: mean-of-params JSON (no longer written by pipeline ``main``)."""
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
    console.log(f"Saved fit parameters to {out}")
    return out


def save_sample_mean_waveforms(
    pulses: list[PulseData], fs: float, output_dir: Path, *, seed: int
) -> Path:
    """ARCHIVED: cache valley-aligned mean/median of the morph-filtered fit sample."""
    waveforms = np.asarray([p.waveform for p in pulses], dtype=float)
    mean_y = np.mean(waveforms, axis=0)
    median_y = np.median(waveforms, axis=0)
    seps = np.asarray([p.separation_ms for p in pulses], dtype=float)
    out = output_dir / "double_pulse_fit_sample_mean.npz"
    np.savez(
        out,
        mean=mean_y,
        median=median_y,
        fs=fs,
        n=len(pulses),
        seed=seed,
        separations_ms=seps,
    )
    console.log(f"Saved sample mean waveforms to {out}")
    return out


def fit_waveform_two_templates(
    y: np.ndarray,
    template: np.ndarray,
    fs: float,
    *,
    fit_width: bool = True,
    label: str = "mean",
) -> FitResult:
    """ARCHIVED helper: free-peak fit with optional shared width (E-like / A-like)."""
    peaks = double_peak_indices(y, fs)
    if peaks is None:
        center = len(y) // 2
        half = 0.001 * fs
        t1_init, t2_init = float(center - half), float(center + half)
    else:
        t1_init, t2_init = float(peaks[0]), float(peaks[1])
    result = fit_free_peaks(y, template, t1_init, t2_init, fs, fit_width=fit_width)
    if result.success:
        result.model = f"direct_{label}_{'free_w' if fit_width else 'fixed_w'}"
    return result


def plot_direct_target_fit(
    y: np.ndarray,
    result: FitResult,
    template: np.ndarray,
    fs: float,
    output_dir: Path,
    *,
    target_name: str,
    n_pulses: int,
    filename: str,
    write_thesis_wip: bool = False,
):
    """ARCHIVED: overlay observed target, direct two-template fit, residual.

    Formerly wrote thesis copies under ``figures/workinprogress/``. Kept for
    manual re-runs; pipeline ``main`` does not call this.
    """
    yhat, c1, c2 = reconstruct_fit(result, template)
    residual = y - yhat
    time_ms = np.arange(len(y)) / fs * 1000
    peaks = double_peak_indices(y, fs)

    apply_presentation_style()
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax, ax_res = axes

    ax.plot(
        time_ms,
        y,
        color=DOUBLE_COLOR,
        linewidth=2.2,
        label=f"Observed {target_name}",
        zorder=4,
    )
    ax.plot(
        time_ms,
        yhat,
        color=REFERENCE_COLOR,
        linewidth=1.8,
        label=(
            f"Fit ({result.model}: $\\Delta t$={result.delay_ms:.2f} ms, "
            f"$w$={result.width_scale:.2f}, $R^2$={result.r2:.3f})"
        ),
        zorder=5,
    )
    ax.plot(
        time_ms,
        c1,
        color=NORMAL_COLOR,
        linestyle="--",
        label=f"Comp. 1 ($a_1$={result.a1:.2f})",
        zorder=3,
    )
    ax.plot(
        time_ms,
        c2,
        color=NORMAL_COLOR,
        linestyle=":",
        label=f"Comp. 2 ($a_2$={result.a2:.2f})",
        zorder=3,
    )
    if peaks is not None:
        ax.scatter(
            peaks / fs * 1000,
            y[peaks],
            color=DOUBLE_COLOR,
            s=70,
            marker="*",
            zorder=6,
            label=(
                f"Observed peaks "
                f"($\\Delta t$={(peaks[1] - peaks[0]) / fs * 1000:.2f} ms)"
            ),
        )
    ax.set_ylabel("Normalized amplitude")
    ax.set_title(
        f"Direct two-normal fit to {target_name} double "
        f"(n={n_pulses} morph-filtered RF doubles)"
    )
    ax.legend(loc=LEGEND_LOC, fontsize=8)
    ax.set_ylim(-0.15, 1.15)
    ax.grid(True, alpha=0.3)

    ax_res.axhline(0.0, color="0.5", linewidth=1)
    ax_res.plot(time_ms, residual, color=DOUBLE_COLOR, linewidth=1.5)
    ax_res.set_xlabel("Time (ms)")
    ax_res.set_ylabel("Residual")
    ax_res.set_title(f"Residual (RMSE={result.rmse:.3f})")
    ax_res.grid(True, alpha=0.3)

    plt.tight_layout()
    out = output_dir / filename
    fig.savefig(out, dpi=300)
    if write_thesis_wip:
        save_thesis_figure(f"workinprogress/{filename}", fig)
    plt.close(fig)
    console.log(f"Saved {out}")


def run_direct_mean_target_fits(
    pulses: list[PulseData],
    template: np.ndarray,
    fs: float,
    output_dir: Path,
    *,
    seed: int = RANDOM_SEED,
) -> dict[str, FitResult]:
    """ARCHIVED: direct fit to sample mean/median (not mean-of-params).

    Used to show that fitting the aggregate waveform answers the biological
    question better than averaging per-pulse parameters. Not run by pipeline
    ``main``; prefer the prototype-median Model A example instead.
    """
    save_sample_mean_waveforms(pulses, fs, output_dir, seed=seed)
    waveforms = np.asarray([p.waveform for p in pulses], dtype=float)
    targets = {
        "mean": np.mean(waveforms, axis=0),
        "median": np.median(waveforms, axis=0),
    }
    results: dict[str, FitResult] = {}
    for name, y in targets.items():
        result = fit_waveform_two_templates(
            y, template, fs, fit_width=True, label=name
        )
        results[name] = result
        console.log(
            f"Direct {name} fit: success={result.success}, "
            f"R²={result.r2:.3f}, Δt={result.delay_ms:.3f} ms, "
            f"w={result.width_scale:.3f}, a1={result.a1:.3f}, a2={result.a2:.3f}"
        )
        plot_direct_target_fit(
            y,
            result,
            template,
            fs,
            output_dir,
            target_name=name,
            n_pulses=len(pulses),
            filename=f"double_pulse_direct_{name}_fit.png",
            write_thesis_wip=False,
        )

    payload = {
        "n_pulses": len(pulses),
        "sample_rate_hz": fs,
        "seed": seed,
        "mean_fit": asdict(results["mean"]),
        "median_fit": asdict(results["median"]),
    }
    params_out = output_dir / "double_pulse_direct_mean_fit_params.json"
    with open(params_out, "w") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
    console.log(f"Saved {params_out}")
    return results


def run_mean_model_hierarchy(
    y: np.ndarray,
    template: np.ndarray,
    fs: float,
    output_dir: Path,
    *,
    n_pulses: int,
    target_name: str = "mean",
) -> dict[str, FitResult]:
    """ARCHIVED: compare constrained two-normal models A–F on one target.

    Used to choose default model A. Not produced by pipeline ``main``.
    """
    peaks = double_peak_indices(y, fs)
    if peaks is None:
        raise RuntimeError(f"No double peaks detected on {target_name} target.")
    t1, t2 = float(peaks[0]), float(peaks[1])
    observed_dt_ms = (t2 - t1) / fs * 1000

    results = {key: fit_two_normals_model(y, template, fs, key) for key in MODEL_KEYS}
    for key, result in results.items():
        w2 = (
            result.width_scale_2
            if np.isfinite(result.width_scale_2)
            else result.width_scale
        )
        console.log(
            f"{key}: success={result.success}, R²={result.r2:.3f}, "
            f"Δt={result.delay_ms:.3f} ms, w1={result.width_scale:.3f}, "
            f"w2={w2:.3f}, a1={result.a1:.3f}, a2={result.a2:.3f}"
        )

    apply_presentation_style()
    time_ms = np.arange(len(y)) / fs * 1000
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    for ax, key in zip(axes.ravel(), MODEL_KEYS):
        result = results[key]
        ax.plot(time_ms, y, color=DOUBLE_COLOR, lw=2, label="Observed", zorder=4)
        if result.success:
            yhat, c1, c2 = reconstruct_fit(result, template)
            w2 = (
                result.width_scale_2
                if np.isfinite(result.width_scale_2)
                else result.width_scale
            )
            ax.plot(
                time_ms,
                yhat,
                color=REFERENCE_COLOR,
                lw=1.8,
                label=f"Fit $R^2$={result.r2:.3f}",
                zorder=5,
            )
            ax.plot(time_ms, c1, color=NORMAL_COLOR, ls="--", alpha=0.85, label="C1")
            ax.plot(time_ms, c2, color=NORMAL_COLOR, ls=":", alpha=0.85, label="C2")
            ax.set_title(
                f"{key}\n"
                f"$\\Delta t$={result.delay_ms:.2f} ms, "
                f"$w$=({result.width_scale:.2f},{w2:.2f})"
            )
        else:
            ax.set_title(f"{key}\n(failed)")
        ax.scatter(
            [t1 / fs * 1000, t2 / fs * 1000],
            [y[int(round(t1))], y[int(round(t2))]],
            color=DOUBLE_COLOR,
            s=40,
            marker="*",
            zorder=6,
        )
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.15, 1.15)
        ax.legend(loc=LEGEND_LOC, fontsize=7)

    for ax in axes[1, :]:
        ax.set_xlabel("Time (ms)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Normalized amplitude")
    fig.suptitle(
        f"Model hierarchy on {target_name} double "
        f"(n={n_pulses}; observed $\\Delta t$={observed_dt_ms:.2f} ms)"
    )
    plt.tight_layout()
    out = output_dir / f"double_pulse_model_hierarchy_{target_name}.png"
    fig.savefig(out, dpi=300)
    # Thesis WIP copies intentionally disabled in the pipeline.
    plt.close(fig)
    console.log(f"Saved {out}")

    payload = {
        "target": target_name,
        "n_pulses": n_pulses,
        "observed_delta_t_ms": observed_dt_ms,
        "models": {k: asdict(v) for k, v in results.items()},
    }
    params_out = output_dir / f"double_pulse_model_hierarchy_{target_name}.json"
    with open(params_out, "w") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
    return results


def plot_aggregate_fit(
    pulses: list[PulseData],
    width_results: list[FitResult],
    template: np.ndarray,
    fs: float,
    output_dir: Path,
):
    """ARCHIVED: mean-of-params reconstruction (superseded by prototype direct fit)."""
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
            f"Mean-of-params fit ($\\Delta t$={mean_fit.delay_ms:.2f} ms, "
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
        f"Mean-of-params template fit "
        f"(n={len(pulses)} RF doubles with two-peak morphology)"
    )
    ax.legend(loc=LEGEND_LOC)
    ax.set_ylim(-0.12, 1.12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = output_dir / "double_pulse_template_fit_example_mean_of_params.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    console.log(f"Saved {out}")


def plot_prototype_template_fit_example(
    y: np.ndarray,
    result: FitResult,
    template: np.ndarray,
    fs: float,
    output_dir: Path,
    *,
    n_pulses: int,
    prototype_stat: str = PROTOTYPE_STAT,
):
    """Thesis example: direct two-normal fit to the prototype double waveform."""
    yhat, c1, c2 = reconstruct_fit(result, template)
    time_ms = np.arange(len(y)) / fs * 1000
    peaks = double_peak_indices(y, fs)
    w2 = (
        result.width_scale_2
        if np.isfinite(result.width_scale_2)
        else result.width_scale
    )

    apply_presentation_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        time_ms,
        y,
        color=DOUBLE_COLOR,
        linewidth=2.4,
        label=f"Prototype {prototype_stat} double",
        zorder=4,
    )
    ax.plot(
        time_ms,
        yhat,
        color=REFERENCE_COLOR,
        linewidth=2.0,
        label=(
            f"Model {result.model} fit "
            f"($\\Delta t$={result.delay_ms:.2f} ms, "
            f"$w$=({result.width_scale:.2f},{w2:.2f}), $R^2$={result.r2:.3f})"
        ),
        zorder=5,
    )
    ax.plot(
        time_ms,
        c1,
        color=NORMAL_COLOR,
        linestyle="--",
        label=f"Normal comp. 1 ($a_1$={result.a1:.2f})",
        zorder=3,
    )
    ax.plot(
        time_ms,
        c2,
        color=NORMAL_COLOR,
        linestyle=":",
        label=f"Normal comp. 2 ($a_2$={result.a2:.2f})",
        zorder=3,
    )
    if peaks is not None:
        ax.scatter(
            peaks / fs * 1000,
            y[peaks],
            color=DOUBLE_COLOR,
            s=70,
            marker="*",
            zorder=6,
            label=(
                f"Observed peaks "
                f"($\\Delta t$={(peaks[1] - peaks[0]) / fs * 1000:.2f} ms)"
            ),
        )
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Normalized amplitude")
    ax.set_title(
        f"Two-normal template fit to prototype {prototype_stat} double "
        f"(n={n_pulses} in prototype; model {result.model})"
    )
    ax.legend(loc=LEGEND_LOC, fontsize=8)
    ax.set_ylim(-0.12, 1.12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = output_dir / "double_pulse_template_fit_example.png"
    fig.savefig(out, dpi=300)
    save_thesis_figure("pulse_shapes/double_pulse_template_fit_example.png", fig)
    plt.close(fig)
    console.log(f"Saved {out}")


def save_prototype_fit_params(
    result: FitResult,
    *,
    fs: float,
    n_pulses: int,
    prototype_stat: str,
    model_key: str,
    output_dir: Path,
) -> Path:
    payload = {
        "pulse_source": "prototype_double_npz",
        "prototype_stat": prototype_stat,
        "model": model_key,
        "n_pulses": n_pulses,
        "sample_rate_hz": fs,
        "fit": {
            **asdict(result),
            "t1_ms": result.t1_samples / fs * 1000,
            "t2_ms": result.t2_samples / fs * 1000,
        },
        "notes": {
            "why_direct_fit": (
                "Direct fit to the prototype waveform; not mean-of-params."
            ),
            "why_median": (
                "Median prototype preferred over mean "
                "(right-skewed separations / outliers)."
            ),
            "why_model_A": (
                "Native-width normals with free times/amplitudes; most honest "
                "two-normal hypothesis without extra width freedom."
            ),
        },
    }
    out = output_dir / "double_pulse_template_fit_params.json"
    with open(out, "w") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
    save_thesis_json("pulse_shapes/double_pulse_template_fit_params.json", payload)
    console.log(f"Saved fit parameters to {out}")
    return out


def plot_parameter_distributions(
    pulses: list[PulseData],
    fixed_results: list[FitResult],
    width_results: list[FitResult],
    output_dir: Path,
):
    """ARCHIVED: per-pulse parameter histograms (not written by pipeline ``main``)."""
    fixed_ok = [r for r in fixed_results if r.success and np.isfinite(r.r2)]
    width_ok = [r for r in width_results if r.success and np.isfinite(r.r2)]

    apply_presentation_style()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    bins = 30

    def _hist(ax, values_fixed, values_width, xlabel, title, xmax=None):
        if values_fixed:
            ax.hist(
                values_fixed,
                bins=bins,
                alpha=0.55,
                color=NORMAL_COLOR,
                label="$w=1$",
                density=True,
            )
        if values_width:
            ax.hist(
                values_width,
                bins=bins,
                alpha=0.55,
                color=DOUBLE_COLOR,
                label="Free $w$",
                density=True,
            )
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
    _hist(
        axes[0, 2],
        [r.r2 for r in fixed_ok],
        [r.r2 for r in width_ok],
        r"$R^2$",
        "Goodness of fit",
    )
    ratios_f = [
        r.amplitude_ratio
        for r in fixed_ok
        if np.isfinite(r.amplitude_ratio) and r.amplitude_ratio < 8
    ]
    ratios_w = [
        r.amplitude_ratio
        for r in width_ok
        if np.isfinite(r.amplitude_ratio) and r.amplitude_ratio < 8
    ]
    _hist(axes[1, 0], ratios_f, ratios_w, r"$a_2/a_1$", "Amplitude ratio", xmax=5)
    _hist(
        axes[1, 1],
        [r.a1 for r in fixed_ok],
        [r.a1 for r in width_ok],
        r"$a_1$",
        "First component scale",
        xmax=2,
    )
    _hist(
        axes[1, 2],
        [r.rmse for r in fixed_ok],
        [r.rmse for r in width_ok],
        "RMSE",
        "Residual magnitude",
        xmax=0.3,
    )

    fig.suptitle(f"Template-fit parameter distributions (n={len(width_ok)} doubles)")
    plt.tight_layout()
    out = output_dir / "double_pulse_template_fit_distributions.png"
    fig.savefig(out, dpi=300)
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
    """ARCHIVED: summary stats over per-pulse fits."""
    def _stats(results, key):
        vals = [getattr(r, key) for r in results if r.success and np.isfinite(getattr(r, key))]
        if not vals:
            return {}
        arr = np.asarray(vals, dtype=float)
        return {
            "n": int(arr.size),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
        }

    detected = [p.separation_ms for p in pulses]
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
    random_seed=RANDOM_SEED,
    apply_classifier=False,
    model_key: str = DEFAULT_MODEL,
    prototype_stat: str = PROTOTYPE_STAT,
):
    """Pipeline entry: direct Model-A (default) fit to prototype median double.

    Archived exploratory paths (sample direct mean/median, model hierarchy,
    mean-of-params, prototype-mean fits) remain as helpers above but are not
    invoked here.
    """
    apply_presentation_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    console.log(
        f"Fitting prototype {prototype_stat} double with model {model_key} "
        f"(direct fit; not mean-of-params)..."
    )

    if apply_classifier:
        console.log("Applying Random Forest classifier to h5 files...")
        apply_special_pulse_classifier(data_path)

    from special_pulses.prototype_pulse_plots import load_prototype_mean_waveforms

    proto = load_prototype_mean_waveforms("double")
    if proto is None:
        raise RuntimeError(
            "Prototype double mean/median npz not found. "
            "Run special_pulses/prototype_pulse_plots.py first."
        )
    if prototype_stat not in ("mean", "median"):
        raise ValueError("prototype_stat must be 'mean' or 'median'")
    if prototype_stat != PROTOTYPE_STAT:
        console.log(
            f"[yellow]Using non-default prototype_stat={prototype_stat!r} "
            f"(pipeline default is {PROTOTYPE_STAT!r})"
        )
    if model_key != DEFAULT_MODEL:
        console.log(
            f"[yellow]Using non-default model={model_key!r} "
            f"(pipeline default is {DEFAULT_MODEL!r})"
        )

    required_fs = float(proto["fs"])
    console.log(
        f"Restricting normal template to {required_fs:.0f} Hz "
        f"(matching prototype double)"
    )

    cached = load_cached_template()
    if cached is not None and abs(cached[1] - required_fs) < 1e-6:
        template, fs = cached
        console.log(
            f"Loaded cached normal template ({len(template)} samples, fs={fs:.0f})"
        )
    else:
        if cached is not None:
            console.log(
                f"[yellow]Cached template fs={cached[1]:.0f} mismatches "
                f"required {required_fs:.0f}; rebuilding"
            )
        console.log("Collecting normal pulses for template...")
        template, fs = build_normal_template(
            data_path, required_fs=required_fs, random_seed=random_seed
        )

    y = np.asarray(proto[prototype_stat], dtype=float)
    if len(y) != len(template):
        raise RuntimeError(
            f"Prototype length {len(y)} != template length {len(template)}"
        )
    if abs(proto["fs"] - fs) >= 1e-6:
        raise RuntimeError(
            f"Prototype fs={proto['fs']} mismatches template fs={fs}"
        )

    result = fit_two_normals_model(y, template, fs, model_key=model_key)
    console.log(
        f"Prototype {prototype_stat} / {model_key}: success={result.success}, "
        f"R²={result.r2:.3f}, Δt={result.delay_ms:.3f} ms, "
        f"w={result.width_scale:.3f}, a1={result.a1:.3f}, a2={result.a2:.3f}"
    )
    plot_prototype_template_fit_example(
        y,
        result,
        template,
        fs,
        OUTPUT_DIR,
        n_pulses=int(proto["n_in_mean"]),
        prototype_stat=prototype_stat,
    )
    save_prototype_fit_params(
        result,
        fs=fs,
        n_pulses=int(proto["n_in_mean"]),
        prototype_stat=prototype_stat,
        model_key=model_key,
        output_dir=OUTPUT_DIR,
    )
    console.log(
        "Archived paths not run: sample direct mean/median, model hierarchy A–F, "
        "mean-of-params example, prototype-mean fits."
    )
    console.log(f"Done. Outputs under {OUTPUT_DIR} and figures/pulse_shapes/")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Fit the prototype double pulse with two normal templates. "
            f"Default: prototype {PROTOTYPE_STAT} + model {DEFAULT_MODEL}."
        )
    )
    parser.add_argument(
        "--model",
        choices=MODEL_KEYS,
        default=DEFAULT_MODEL,
        help="Two-normal model key (default: A_w1_free_t).",
    )
    parser.add_argument(
        "--prototype-stat",
        choices=("median", "mean"),
        default=PROTOTYPE_STAT,
        help="Which prototype aggregate to fit (default: median).",
    )
    parser.add_argument(
        "--apply-classifier",
        action="store_true",
        help="Re-apply RF labels before fitting (default: use existing labels).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(
        apply_classifier=args.apply_classifier,
        model_key=args.model,
        prototype_stat=args.prototype_stat,
    )
