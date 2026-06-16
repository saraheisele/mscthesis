# Berlin Eel Analysis Pipeline

Python scripts for analyzing electric-organ-discharge (EOD) activity from the MfN Berlin tank recordings. All paths are configured in `data_paths.py`.

## Quick start

```bash
cd /home/eisele/wrk/mscthesis
pip install -r requirements.txt
cd code
python eel_data_preprocessing.py      # interactive: choose pulse type
python pulse_analysis_plots.py        # interactive: choose pulse type
```

Run scripts from the `code/` directory so local imports resolve correctly.

## Analysis parts

| Part | Topic | Scripts |
|------|-------|---------|
| **Infrastructure** | Paths, shared config, I/O | `data_paths.py`, `pulse_config.py`, `h5_io.py`, `plotting_utils.py` |
| **1 — Activity** | Pulse-rate histograms | `eel_data_preprocessing.py` → `pulse_analysis_plots.py`, `correlate_pulse_shapes.py` |
| **2 — Special pulses** | Double / wide / fat detection | `double_peaks_detection.py` → `double_peaks_visualization.py`, `prototype_pulse_plots.py` |
| **3 — Environment** | Temperature, conductivity, volleys, tank areas | `environment_correlation.py`, `volley_pulse_analysis.py`, `tank_area_pulse_analysis.py` (or `correlate_activity_with_environment.py` for all) |
| **4 — Feeding** | Feeding-time correlation | `correlate_activity_with_feeding.py` |
| **5 — Position** | Eel location along electrode line | `position_utils.py` → `position_data_preprocessing.py` → `position_analysis_plots.py`, `animate_eel_position.py` |

Legacy scripts are kept in `old_analysis_version/` for reference only.

## Recommended run order

### Part 1 — Pulse activity (all pulse types)

1. **`double_peaks_detection.py`** *(optional but needed for double/wide/fat)*  
   Annotates `.h5` files with `is_double_peak`, `is_wide_pulse`, `is_fat_pulse` markers.  
   Run once per detection mode (edit `DETECTION_MODE` at top of script).

2. **`eel_data_preprocessing.py`**  
   Builds multi-timescale pulse-rate `.npz` histograms.  
   Run four times: `all`, `double`, `wide`, `fat` (prompted at start).  
   **Output:** `data/intermediate/eels-mfn2021_dummy_activity_histograms/`

3. **`pulse_analysis_plots.py`**  
   Global and session-wise pulse-rate figures.  
   **Output:** `data/processed/{all,double,wide,fat}_pulses/`

4. **`correlate_pulse_shapes.py`**  
   Correlates special-pulse rates with overall activity.  
   **Output:** `data/processed/pulse_shape_correlation/`

### Part 2 — Special-pulse QA

5. **`double_peaks_visualization.py`** — detection grids and interactive verification  
6. **`prototype_pulse_plots.py`** — aligned prototype waveforms per shape  

### Part 3 — Environmental and contextual analyses

7. **`correlate_activity_with_environment.py`** — runs all three sub-analyses below, or run individually:
   - **`environment_correlation.py`** — temperature & conductivity vs pulse rate  
   - **`volley_pulse_analysis.py`** — high-voltage volley context analysis  
   - **`tank_area_pulse_analysis.py`** — bright vs dark tank area comparison  
   **Input:** Excel files in `data/raw/eels-mfn2021_dummy_pulses_redetected/leitwerte_metadaten/`  
   **Output:** `data/processed/environment_correlation/`

### Part 4 — Feeding correlation

8. **`correlate_activity_with_feeding.py`**  
   Parses session `.docx` logs from `/data2/labdata/eels-mfn2021/berlin_tank_site/`.  
   **Output:** `data/processed/feeding_correlation/`

### Part 5 — Position analysis

9. **`position_data_preprocessing.py`** — spatial histograms from `.h5` pulse positions  
10. **`position_analysis_plots.py`** — mean position, occupancy, bright/dark fraction plots  
11. **`animate_eel_position.py`** — animation for a single wav chunk  

   ```bash
   python animate_eel_position.py /path/to/eellogger.wav
   python animate_eel_position.py --list-entry-recordings
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
| `eel_data_preprocessing.py` | Build pulse-rate histogram `.npz` files |
| `pulse_analysis_plots.py` | Plot pulse-rate time series |
| `correlate_pulse_shapes.py` | Cross-correlate pulse shapes with all-pulses |
| `double_peaks_detection.py` | Rule-based special-pulse detection + ML classifier |
| `double_peaks_visualization.py` | Visual QA of detections |
| `prototype_pulse_plots.py` | Prototype waveform overlays |
| `environment_correlation.py` | Temperature/conductivity correlation |
| `volley_pulse_analysis.py` | Volley detection and pulse-type enrichment |
| `tank_area_pulse_analysis.py` | Bright vs dark area pulse composition |
| `correlate_activity_with_environment.py` | Orchestrator for Part 3 |
| `correlate_activity_with_feeding.py` | Feeding-time correlation |
| `position_utils.py` | Electrode geometry and position estimation |
| `position_data_preprocessing.py` | Spatial histogram generation |
| `position_analysis_plots.py` | Position and occupancy plots |
| `animate_eel_position.py` | Movement animation for one wav chunk |
