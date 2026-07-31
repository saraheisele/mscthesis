"""Archived exploratory double-pulse template-fit helpers.

Analysis part: NOT in the main pipeline. Manual re-runs only.

Pipeline default lives in ``special_pulses/double_pulse_template_fit.py``
(direct Model-A fit to the prototype median double). These helpers were used
during model selection (hierarchy A–F, mean-of-params vs direct fit) and are
kept for sensitivity checks.

Run from ``code/`` after building a normal template and pulse sample via the
main module helpers.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from path_setup import setup_script_paths

setup_script_paths(__file__, "special_pulses")

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console

from presentation_style import (
    LEGEND_LOC,
    apply_presentation_style,
    save_thesis_figure,
)
from special_pulses.double_pulse_template_fit import (
    DOUBLE_COLOR,
    MODEL_KEYS,
    NORMAL_COLOR,
    PulseData,
    RANDOM_SEED,
    REFERENCE_COLOR,
    FitResult,
    _json_default,
    aggregate_fit_params,
    fit_metrics,
    fit_two_normals_model,
    fit_waveform_two_templates,
    reconstruct_fit,
    save_aggregate_fit_params,
    save_sample_mean_waveforms,
)
from special_pulses.prototype_pulse_plots import double_peak_indices

console = Console()


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

