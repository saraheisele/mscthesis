# Berlin Eel Analysis Pipeline

Python scripts for electric-organ-discharge (EOD) activity from the MfN Berlin
tank recordings. Paths: `data_paths.py`. Presentation styling:
`presentation_style.py`.

## Quick start

```bash
cd /path/to/mscthesis   # or a git worktree of this repo
pip install -r requirements.txt
cd code

./run_analysis.sh dummy   # development subset → figures/, data/processed/dummy/
./run_analysis.sh full    # full Berlin dataset → figures/full/, data/processed/
```

Or run individual scripts from `code/` (so local imports resolve):

```bash
python activity_timescales/eel_data_preprocessing.py   # prompted: all|double|wide
python activity_timescales/pulse_analysis_plots.py
```

### Dataset mode

| Mode | How | Thesis figures | Processed outputs |
|------|-----|----------------|-------------------|
| **dummy** | `./run_analysis.sh dummy` (or `EEL_USE_DUMMY_DATASET=true`) | `figures/` | `data/processed/dummy/` |
| **full** | `./run_analysis.sh full` (or `EEL_USE_DUMMY_DATASET=false`) | `figures/full/` | `data/processed/` |

Logs: `data/intermediate/pipeline_logs/pipeline_dummy.log` or `pipeline_full.log`.

The runner resolves `ROOT` from its own location, so the same script works in
the main checkout and in a parallel git worktree.

## Pipeline overview

```
H5 pulses (+ optional labeling)
        │
        ▼
[1] special_pulses/double_peaks_detection.py     ML markers (double / wide)
        │
        ▼
[2] activity_timescales/eel_data_preprocessing   multi-timescale rate NPZs
[3] activity_timescales/pulse_analysis_plots     circadian / timescale panels
[4] correlations/correlate_pulse_shapes          special vs all-pulse rates
        │
        ▼
[5] special_pulses/half_width_distributions      half-width hists (shared collector)
[6] special_pulses/prototype_pulse_plots         mean/median prototypes (npz)
[7] special_pulses/pulse_shape_distribution      shape prevalence
[8] special_pulses/double_pulse_template_fit     Model A on prototype median
[9] special_pulses/pca_space_plots               labeled PCA scatter
[10] special_pulses/pulse_properties_over_time   temporal half-width / IPI
[11] correlations/correlate_mating_with_activity mating notes ↔ rates
        │
        ▼
[12] correlations/correlate_activity_with_environment
        ├─ environment_correlation.py           temp / conductivity
        ├─ volley_pulse_analysis.py             high-voltage volleys
        └─ tank_area_pulse_analysis.py          bright vs dark areas
[13] correlations/correlate_activity_with_feeding
        │
        ▼
[14] position_estimation/position_data_preprocessing
[15] position_estimation/position_analysis_plots
        │
        └── (optional) animate_eel_position.py  needs a wav path
```

Shared utilities (not numbered steps): `data_paths`, `h5_io`, `pulse_config`,
`path_setup`, `plotting_utils`, `presentation_style`, `pulse_shape_metrics`,
`pulse_property_collect`, `session_notes_utils`, `mating_notes_utils`,
`position_utils`, `eelplotting`.

## Thesis figures layout

Dummy runs write under `figures/`; full-dataset runs write the same relative
paths under `figures/full/`. Thesis-relevant plots are dual-written: processed
diagnostics under `data/processed/...`, presentation copies via
`save_thesis_figure()` / `copy_thesis_asset()`.

```
figures/                         # dummy (default); full → figures/full/
├── pulse_shapes/
│   ├── half_width_kde_all_shapes.png
│   ├── half_width_distributions_overlay.png
│   ├── half_width_distributions_all_shapes.png
│   ├── labeled_pulses_pca_space.png
│   ├── normal_pulses_aligned_to_double.png
│   ├── mean_pulse_shapes_panel.png
│   ├── prototype_{double,normal,wide}_pulse.png
│   ├── double_pulse_template_fit_example.png
│   └── pulse_shape_distribution.png
├── activity_timescales/
│   └── circadian_panels_{all_shapes,all,dp,wide}.png
├── correlations/
│   ├── correlations_all_pulse_shapes.png
│   ├── lag_corr_all_pulse_shapes.png
│   ├── feeding_vs_nonfeeding_pulse_rates.png
│   └── peri_feeding_pulse_rate_trajectories.png
├── position_estimation/
│   ├── overall_position_distribution_peak.png
│   ├── position_panels_peak.png
│   └── eel_position_animation.{mp4,gif}   # optional; not in run_analysis.sh
└── workinprogress/
    └── pulse_shape_distribution_mating_window.png
```

**Produced by `./run_analysis.sh`:** everything above except
`eel_position_animation.*`. Double-pulse model-hierarchy / sample-fit panels
live under `archived/double_pulse_template_fit_exploratory.py` and are **not**
written by the pipeline.

Exploratory mating overlays also land under `figures/exploratory_mating_corr/`
from `pulse_analysis_plots.py` and `pulse_properties_over_time.py`.

## Directory layout

| Subdirectory | Topic | Scripts |
|--------------|-------|---------|
| *(root)* | Paths, style, shared I/O | `data_paths.py`, `presentation_style.py`, `pulse_config.py`, `h5_io.py`, `plotting_utils.py`, `path_setup.py`, `eelplotting.py`, `run_analysis.sh` |
| `activity_timescales/` | Pulse-rate histograms | `eel_data_preprocessing.py`, `pulse_analysis_plots.py` |
| `special_pulses/` | Classification & shapes | see script reference below |
| `correlations/` | Environment, feeding, mating, shapes | orchestrators + helpers |
| `position_estimation/` | Location along electrode line | preprocessing, plots, optional animation |
| `archived/` | Legacy / WIP / exploratory | see [`archived/README.md`](archived/README.md) |

## Recommended run order

Prefer `./run_analysis.sh {dummy|full}`. Manual iteration uses the same order
as the pipeline overview above.

### Classifier first-time setup

```bash
python special_pulses/double_peaks_detection.py --label --train
python special_pulses/double_peaks_detection.py   # apply saved model (pipeline default)
```

### Double-pulse template fit — modelling choices

`double_pulse_template_fit.py` asks whether a double pulse can be produced as
**two normal pulses in quick succession**.

| Choice | Decision | Why |
|--------|----------|-----|
| Aggregate | **Direct fit** to the prototype waveform | Averaging per-pulse parameters ≠ fitting the typical shape |
| Prototype summary | **Median** (not mean) | Mild right-skew; median is more typical |
| Model | **A** (`A_w1_free_t`) by default | Same native-width normal template; free times & amplitudes only |

```bash
python special_pulses/double_pulse_template_fit.py
python special_pulses/double_pulse_template_fit.py --model E_free_shared_w
python special_pulses/double_pulse_template_fit.py --prototype-stat mean   # non-default
```

### Optional position animation

```bash
python position_estimation/animate_eel_position.py /path/to/eellogger.wav
python position_estimation/animate_eel_position.py --list-entry-recordings
```

## Data paths

Controlled by `EEL_USE_DUMMY_DATASET` in `data_paths.py` (default `true`; the
runner sets it explicitly):

| | Dummy | Full |
|--|-------|------|
| `H5_DIR` | `data/raw/.../berlin_tank_site` | `/home/efish/eelsmfn2021_eods/berlin_tank_site` |
| Thesis figures | `figures/` | `figures/full/` |
| Processed | `data/processed/dummy/` | `data/processed/` |
| Activity histograms | `data/intermediate/eels-mfn2021_dummy_activity_histograms/` | `.../eels-mfn2021_activity_histograms/` |

`EXCEL_DIR` (leitwerte) and `LAB_DATA_DIR` (session notes / electrode layout)
are **shared** across dummy and full modes.

Histogram NPZ basenames still use the historical prefix `berlin_dummypulses_*`
for both modes (name only; contents follow the active dataset).

## Script reference

| Script | Purpose | Pipeline |
|--------|---------|----------|
| `run_analysis.sh` | Unified runner (`dummy` \| `full`) | entry |
| `data_paths.py` | Central path configuration | utility |
| `presentation_style.py` | Thesis palette, `save_thesis_figure` | utility |
| `pulse_config.py` | Pulse-type labels, suffixes, marker names | utility |
| `h5_io.py` | Recursive `.h5` discovery / open / markers | utility |
| `plotting_utils.py` | Shared timescale axis formatting | utility |
| `path_setup.py` | Import bootstrap for subdirectory scripts | utility |
| `eelplotting.py` | SVG eel drawing (animation only) | optional |
| `activity_timescales/eel_data_preprocessing.py` | Pulse-rate histogram NPZs | [2] |
| `activity_timescales/pulse_analysis_plots.py` | Circadian / timescale panels | [3] |
| `special_pulses/double_peaks_detection.py` | RF special-pulse classifier (+ facade) | [1] |
| `special_pulses/waveform_rule_metrics.py` | Rule-based width/shape metrics | utility (used by [1]/metrics) |
| `special_pulses/special_pulse_labeling.py` | Interactive labeling UIs | train/label only |
| `special_pulses/special_pulse_benchmark.py` | Classifier benchmark / decision tuning | train only |
| `special_pulses/pulse_shape_metrics.py` | Shared waveform helpers | utility |
| `special_pulses/pulse_property_collect.py` | Shared half-width / property H5 walk | utility ([5]/[10]) |
| `special_pulses/half_width_distributions.py` | Half-width distributions by shape | [5] |
| `special_pulses/prototype_pulse_plots.py` | Prototype mean/median waveforms | [6] |
| `special_pulses/pulse_shape_distribution.py` | Shape prevalence bar charts | [7] |
| `special_pulses/double_pulse_template_fit.py` | Direct two-normal fit (median, model A) | [8] |
| `special_pulses/pca_space_plots.py` | Labeled-pulse PCA scatter | [9] |
| `special_pulses/pulse_properties_over_time.py` | Half-width / property trends | [10] |
| `correlations/correlate_pulse_shapes.py` | Cross-correlate shapes with all-pulses | [4] |
| `correlations/correlate_mating_with_activity.py` | Mating notes ↔ rates (all + double) | [11] |
| `correlations/correlate_activity_with_environment.py` | Env / volley / tank orchestrator | [12] |
| `correlations/environment_correlation.py` | Temperature & conductivity | via [12] |
| `correlations/volley_pulse_analysis.py` | High-voltage volleys | via [12] |
| `correlations/tank_area_pulse_analysis.py` | Bright vs dark tank areas | via [12] |
| `correlations/correlate_activity_with_feeding.py` | Feeding-time correlation | [13] |
| `correlations/session_notes_utils.py` | Session docx / wav time alignment | utility |
| `correlations/mating_notes_utils.py` | Mating-note extraction | utility |
| `position_estimation/position_data_preprocessing.py` | Position histograms | [14] |
| `position_estimation/position_analysis_plots.py` | Occupancy / mean position | [15] |
| `position_estimation/position_utils.py` | Geometry / layout helpers | utility |
| `position_estimation/animate_eel_position.py` | Movement animation for one wav | optional |

## Cleanup notes (code-cleanup branch)

- Unused / WIP scripts moved to `archived/` (see that README).
- Half-width H5 walks unified in `pulse_property_collect.py`.
- Waveform helpers (`baseline_correct`, …) live in `pulse_shape_metrics.py`.
- Session-note parsing extracted to `session_notes_utils.py`.
- Special-pulse monolith split: rule metrics / labeling / benchmark modules;
  `double_peaks_detection.py` remains the production classifier + CLI facade.
- Feeding/mating H5 discovery is recursive (`get_path_list`); mating still
  analyses **all + double** only (wide intentionally omitted).
- Position plots use shared `plotting_utils.format_x_axis`.
- Template-fit exploratory runners moved to
  `archived/double_pulse_template_fit_exploratory.py`.
