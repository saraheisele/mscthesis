# Berlin Eel Analysis Pipeline

Python scripts for analyzing electric-organ-discharge (EOD) activity from the MfN Berlin tank recordings. Paths are configured in `data_paths.py`; presentation styling in `presentation_style.py`.

## Quick start

```bash
cd /home/eisele/wrk/mscthesis
pip install -r requirements.txt
cd code

# Full analysis (classifier → histograms → thesis figures + processed dual-writes)
./run_analysis.sh dummy   # development subset
./run_analysis.sh full    # full Berlin tank dataset
```

Or run individual scripts interactively (from `code/` so local imports resolve):

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

## Thesis figures layout

Dummy runs write under `figures/`; full-dataset runs write the same relative paths under `figures/full/` (`EEL_USE_DUMMY_DATASET`). Thesis-relevant plots are dual-written: processed diagnostics under `data/processed/...`, and presentation copies via `save_thesis_figure()` / `copy_thesis_asset()`.

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
│   └── eel_position_animation.{mp4,gif}   # optional; needs wav (not in run_analysis.sh)
├── workinprogress/
│   ├── pulse_shape_distribution_mating_window.png
│   └── half-width threshold panels from half_width_threshold_waveforms.py
└── full/                        # same structure for the full Berlin dataset
```

**Produced by `./run_analysis.sh`:** everything above except `eel_position_animation.{mp4,gif}` (run `position_estimation/animate_eel_position.py` with a wav path). Double-pulse model-hierarchy / sample-fit panels are archived helpers in `double_pulse_template_fit.py` and are **not** written by the pipeline.

Exploratory mating overlays (not thesis-canonical) also land under `figures/exploratory_mating_corr/` from `pulse_analysis_plots.py` and `pulse_properties_over_time.py`.

## Directory layout

| Subdirectory | Topic | Scripts |
|--------------|-------|---------|
| *(root)* | Paths, style, shared I/O | `data_paths.py`, `presentation_style.py`, `pulse_config.py`, `h5_io.py`, `plotting_utils.py`, `path_setup.py`, `run_analysis.sh` |
| `activity_timescales/` | Pulse-rate histograms | `eel_data_preprocessing.py`, `pulse_analysis_plots.py` |
| `special_pulses/` | Classification & shapes | `double_peaks_detection.py`, `double_peaks_visualization.py`, `prototype_pulse_plots.py`, `pulse_shape_distribution.py`, `half_width_distributions.py`, `pulse_properties_over_time.py`, `double_pulse_template_fit.py`, `pca_space_plots.py`, `half_width_threshold_waveforms.py` |
| `correlations/` | Environment, feeding, mating, shapes | `environment_correlation.py`, `volley_pulse_analysis.py`, `tank_area_pulse_analysis.py`, `correlate_activity_with_environment.py`, `correlate_activity_with_feeding.py`, `correlate_pulse_shapes.py`, `correlate_mating_with_activity.py`, `mating_notes_utils.py` |
| `position_estimation/` | Location along electrode line | `position_utils.py`, `position_data_preprocessing.py`, `position_analysis_plots.py`, `animate_eel_position.py` |

Legacy scripts are kept in `old_analysis_version/` for reference only.

## Recommended run order

Prefer `./run_analysis.sh {dummy|full}`, which runs the steps below in order. For manual iteration:

### Part 1 — Pulse activity

1. **`special_pulses/double_peaks_detection.py`**  
   Applies the supervised PCA + random forest classifier (`is_double_peak`, `is_wide_pulse`, `special_pulse_class`).  
   First run: `python special_pulses/double_peaks_detection.py --label --train`  
   Later: `python special_pulses/double_peaks_detection.py` (reuses the saved model).

2. **`activity_timescales/eel_data_preprocessing.py`** — multi-timescale pulse-rate `.npz` histograms (`all`, `double`, `wide`).

3. **`activity_timescales/pulse_analysis_plots.py`** — circadian / timescale panels (thesis + processed).

4. **`correlations/correlate_pulse_shapes.py`** — special-pulse vs all-pulses histogram correlations.

### Part 2 — Pulse shapes

5. **`special_pulses/half_width_distributions.py`**
6. **`special_pulses/prototype_pulse_plots.py`** — builds prototype mean/median waveforms (npz); the **median** double is the canonical prototype used downstream
7. **`special_pulses/pulse_shape_distribution.py`** — overall shape prevalence (+ mating-window WIP plot)
8. **`special_pulses/double_pulse_template_fit.py`** — direct two-normal fit of the **prototype median** double (default **model A**); thesis figure `figures/pulse_shapes/double_pulse_template_fit_example.png`
9. **`special_pulses/pca_space_plots.py`**
10. **`special_pulses/pulse_properties_over_time.py`**
11. **`correlations/correlate_mating_with_activity.py`**
12. **`special_pulses/half_width_threshold_waveforms.py`** *(WIP)*

#### Double-pulse template fit — modelling choices

`double_pulse_template_fit.py` asks whether a double pulse can be produced as **two normal pulses in quick succession**.

| Choice | Decision | Why |
|--------|----------|-----|
| Aggregate | **Direct fit** to the prototype waveform | Averaging per-pulse fit parameters (“mean-of-params”) is not the same as fitting the typical shape and answers the biological question less cleanly. |
| Prototype summary | **Median** (not mean) | Peak separations / shapes are mildly right-skewed; the median is closer to a typical double and less pulled by rare outliers. Mean remains stored in the prototype `.npz` and in archived helpers. |
| Model | **A** (`A_w1_free_t`) by default | Same native-width normal template, free times & amplitudes only — the most honest, least overfitting-prone statement of the two-normal hypothesis. Models B–F remain available via `--model` for sensitivity checks. |

**Archived (code kept, not run by the pipeline):** sample direct mean/median fits (`run_direct_mean_target_fits` — used to compare mean-of-params vs direct fit), model-hierarchy A–F panels (`run_mean_model_hierarchy` — used to choose model A), mean-of-params example / parameter distributions, and prototype-**mean** fit figures. Manual re-runs can still call those helpers; they no longer write `figures/workinprogress/` thesis copies.

```bash
python special_pulses/double_pulse_template_fit.py
python special_pulses/double_pulse_template_fit.py --model E_free_shared_w
python special_pulses/double_pulse_template_fit.py --prototype-stat mean   # non-default
```

### Part 3 — Environment / context

13. **`correlations/correlate_activity_with_environment.py`** — orchestrates:
    - `environment_correlation.py` — temperature & conductivity
    - `volley_pulse_analysis.py` — high-voltage volleys
    - `tank_area_pulse_analysis.py` — bright vs dark tank areas

### Part 4 — Feeding

14. **`correlations/correlate_activity_with_feeding.py`**

### Part 5 — Position

15. **`position_estimation/position_data_preprocessing.py`**
16. **`position_estimation/position_analysis_plots.py`**
17. **`position_estimation/animate_eel_position.py`** — optional; needs a wav path (writes mp4 + gif thesis assets)

```bash
python position_estimation/animate_eel_position.py /path/to/eellogger.wav
python position_estimation/animate_eel_position.py --list-entry-recordings
```

## Data paths

Controlled by `EEL_USE_DUMMY_DATASET` in `data_paths.py` (default `true`; the runner sets it explicitly):

| | Dummy | Full |
|--|-------|------|
| `H5_DIR` | `data/raw/.../berlin_tank_site` | `/home/efish/eelsmfn2021_eods/berlin_tank_site` |
| Thesis figures | `figures/` | `figures/full/` |
| Processed | `data/processed/dummy/` | `data/processed/` |
| Activity histograms | `data/intermediate/eels-mfn2021_dummy_activity_histograms/` | `.../eels-mfn2021_activity_histograms/` |

`EXCEL_DIR` and `LAB_DATA_DIR` (session notes / electrode layout) are shared.

## Script reference

| Script | Purpose |
|--------|---------|
| `run_analysis.sh` | Unified pipeline runner (`dummy` \| `full`) |
| `data_paths.py` | Central path configuration |
| `presentation_style.py` | Thesis color palette, rcParams, `save_thesis_figure` |
| `pulse_config.py` | Pulse-type labels, suffixes, h5 marker names |
| `h5_io.py` | Recursive `.h5` discovery |
| `plotting_utils.py` | Shared matplotlib axis formatting |
| `path_setup.py` | Import bootstrap for subdirectory scripts |
| `activity_timescales/eel_data_preprocessing.py` | Pulse-rate histogram `.npz` files |
| `activity_timescales/pulse_analysis_plots.py` | Pulse-rate / circadian panels |
| `special_pulses/double_peaks_detection.py` | RF special-pulse classifier |
| `special_pulses/prototype_pulse_plots.py` | Prototype waveform overlays |
| `special_pulses/pulse_shape_distribution.py` | Shape prevalence bar charts |
| `special_pulses/half_width_distributions.py` | Half-width distributions by shape |
| `special_pulses/double_pulse_template_fit.py` | Direct two-normal fit to prototype median double (model A default) |
| `special_pulses/pca_space_plots.py` | Labeled-pulse PCA scatter |
| `special_pulses/pulse_properties_over_time.py` | Half-width / property trends |
| `correlations/correlate_pulse_shapes.py` | Cross-correlate shapes with all-pulses |
| `correlations/correlate_activity_with_environment.py` | Environment / volley / tank orchestrator |
| `correlations/correlate_activity_with_feeding.py` | Feeding-time correlation |
| `correlations/correlate_mating_with_activity.py` | Mating-note correlation |
| `position_estimation/position_analysis_plots.py` | Position and occupancy plots |
| `position_estimation/animate_eel_position.py` | Movement animation for one wav chunk |
