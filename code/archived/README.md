# Archived analysis scripts

Scripts and helpers kept for reference / manual re-runs, **not** part of
`./run_analysis.sh`.

| Path | Why archived |
|------|----------------|
| `old_analysis_version/` | Pre-pipeline prototypes (old mount paths, early NPZ workflow) |
| `double_peaks_visualization.py` | Interactive QA for special-pulse markers; overlaps thesis half-width plots |
| `half_width_threshold_waveforms.py` | WIP valley-split waveform panels (formerly a runner step) |
| `double_pulse_template_fit_exploratory.py` | Model hierarchy A–F, mean-of-params, sample direct fits used during model selection |

To re-run exploratory template fits, build a template/sample via
`special_pulses/double_pulse_template_fit.py` helpers, then import from
`archived.double_pulse_template_fit_exploratory`.
