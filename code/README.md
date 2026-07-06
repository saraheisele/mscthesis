# Berlin Eel Analysis Pipeline

Python scripts for analyzing electric-organ-discharge (EOD) activity from the MfN Berlin tank recordings. All paths are configured in `data_paths.py`.

## Quick start

```bash
cd /home/eisele/wrk/mscthesis
pip install -r requirements.txt
cd code
python activity_timescales/eel_data_preprocessing.py      # interactive: choose pulse type
python activity_timescales/pulse_analysis_plots.py        # interactive: choose pulse type
```

Run scripts from the `code/` directory so local imports resolve correctly.

## Directory layout

| Subdirectory | Topic | Scripts |
|--------------|-------|---------|
| *(root)* | Paths, shared config, I/O | `data_paths.py`, `pulse_config.py`, `h5_io.py`, `plotting_utils.py`, `path_setup.py` |
| `activity_timescales/` | Pulse-rate histograms | `eel_data_preprocessing.py`, `pulse_analysis_plots.py` |
| `special_pulses/` | Double / wide / fat detection | `double_peaks_detection.py`, `double_peaks_visualization.py`, `prototype_pulse_plots.py` |
| `correlations/` | Environment, feeding, pulse-shape links | `environment_correlation.py`, `volley_pulse_analysis.py`, `tank_area_pulse_analysis.py`, `correlate_activity_with_environment.py`, `correlate_activity_with_feeding.py`, `correlate_pulse_shapes.py` |
| `position_estimation/` | Eel location along electrode line | `position_utils.py`, `position_data_preprocessing.py`, `position_analysis_plots.py`, `animate_eel_position.py` |

Legacy scripts are kept in `old_analysis_version/` for reference only.

## Recommended run order

### Part 1 — Pulse activity (all pulse types)

1. **`special_pulses/double_peaks_detection.py`** *(optional but needed for double/wide/fat)*  
   Applies the supervised PCA + random forest classifier to annotate `.h5` files with
   `is_double_peak`, `is_wide_pulse`, `is_fat_pulse`, and `special_pulse_class`.  
   First run: `python special_pulses/double_peaks_detection.py --label --train`  
   Later runs: `python special_pulses/double_peaks_detection.py` (reuses the saved model).  
   Rule-based detection: `--mode rule-based --detection-mode double|wide|fat`.

2. **`activity_timescales/eel_data_preprocessing.py`**  
   Builds multi-timescale pulse-rate `.npz` histograms.  
   Run four times: `all`, `double`, `wide`, `fat` (prompted at start).  
   **Output:** `data/intermediate/eels-mfn2021_dummy_activity_histograms/`

3. **`activity_timescales/pulse_analysis_plots.py`**  
   Global and session-wise pulse-rate figures.  
   **Output:** `data/processed/{all,double,wide,fat}_pulses/`

4. **`correlations/correlate_pulse_shapes.py`**  
   Correlates special-pulse rates with overall activity.  
   **Output:** `data/processed/pulse_shape_correlation/`

### Part 2 — Special-pulse QA

5. **`special_pulses/double_peaks_visualization.py`** — detection grids and interactive verification  
6. **`special_pulses/prototype_pulse_plots.py`** — aligned prototype waveforms per shape  

### Part 3 — Environmental and contextual analyses

7. **`correlations/correlate_activity_with_environment.py`** — runs all three sub-analyses below, or run individually:
   - **`correlations/environment_correlation.py`** — temperature & conductivity vs pulse rate  
   - **`correlations/volley_pulse_analysis.py`** — high-voltage volley context analysis  
   - **`correlations/tank_area_pulse_analysis.py`** — bright vs dark tank area comparison  
   **Input:** Excel files in `data/raw/eels-mfn2021_dummy_pulses_redetected/leitwerte_metadaten/`  
   **Output:** `data/processed/environment_correlation/`

### Part 4 — Feeding correlation

8. **`correlations/correlate_activity_with_feeding.py`**  
   Parses session `.docx` logs from `/data2/labdata/eels-mfn2021/berlin_tank_site/`.  
   **Output:** `data/processed/feeding_correlation/`

### Part 5 — Position analysis

9. **`position_estimation/position_data_preprocessing.py`** — spatial histograms from `.h5` pulse positions  
10. **`position_estimation/position_analysis_plots.py`** — mean position, occupancy, bright/dark fraction plots  
11. **`position_estimation/animate_eel_position.py`** — animation for a single wav chunk  

   ```bash
   python position_estimation/animate_eel_position.py /path/to/eellogger.wav
   python position_estimation/animate_eel_position.py --list-entry-recordings
   ```

## Data paths

Edit `data_paths.py` to switch between the development subset and full dataset:

| Variable | Default (dev subset) | Full dataset |
|----------|---------------------|--------------|
| `H5_DIR` | `data/raw/.../berlin_tank_site` | `/data2/labdata/eels-mfn2021/berlin_tank_site/predetected_pulses` |
| `EXCEL_DIR` | `data/raw/.../leitwerte_metadaten` | same location |
| `LAB_DATA_DIR` | `/data2/labdata/eels-mfn2021/berlin_tank_site` | — |

## Script reference

| Script | Purpose |
|--------|---------|
| `data_paths.py` | Central path configuration |
| `pulse_config.py` | Pulse-type labels, suffixes, h5 marker array names |
| `h5_io.py` | Recursive `.h5` file discovery |
| `plotting_utils.py` | Shared matplotlib axis formatting |
| `path_setup.py` | Import bootstrap for scripts in subdirectories |
| `activity_timescales/eel_data_preprocessing.py` | Build pulse-rate histogram `.npz` files |
| `activity_timescales/pulse_analysis_plots.py` | Plot pulse-rate time series |
| `correlations/correlate_pulse_shapes.py` | Cross-correlate pulse shapes with all-pulses |
| `special_pulses/double_peaks_detection.py` | Supervised ML special-pulse classifier (rule-based fallback) |
| `special_pulses/double_peaks_visualization.py` | Visual QA of detections |
| `special_pulses/prototype_pulse_plots.py` | Prototype waveform overlays |
| `correlations/environment_correlation.py` | Temperature/conductivity correlation |
| `correlations/volley_pulse_analysis.py` | Volley detection and pulse-type enrichment |
| `correlations/tank_area_pulse_analysis.py` | Bright vs dark area pulse composition |
| `correlations/correlate_activity_with_environment.py` | Orchestrator for Part 3 |
| `correlations/correlate_activity_with_feeding.py` | Feeding-time correlation |
| `position_estimation/position_utils.py` | Electrode geometry and position estimation |
| `position_estimation/position_data_preprocessing.py` | Spatial histogram generation |
| `position_estimation/position_analysis_plots.py` | Position and occupancy plots |
| `position_estimation/animate_eel_position.py` | Movement animation for one wav chunk |
