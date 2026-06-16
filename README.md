# Masters Thesis — Sarah Eisele

Electric eel (*Electrophorus electricus*) behaviour analysis using Berlin tank grid recordings.

## Installation

```bash
pip install -r requirements.txt
```

Some dependencies (`audian`, `audioio`, `thunderfish`, `thunderlab`) are installed from the bendalab GitHub repositories. A full install requires network access and git.

## Project structure

```
mscthesis/
├── code/                  Analysis pipeline (see code/README.md)
├── data/
│   ├── raw/               Predetected pulses, environmental Excel files
│   ├── intermediate/      Preprocessed histograms (.npz)
│   └── processed/         Figures and correlation outputs
├── requirements.txt
└── README.md
```

## Analysis pipeline

All analysis scripts live in [`code/`](code/). See [`code/README.md`](code/README.md) for the full script reference, data paths, and recommended run order.

**Typical workflow:**

1. Run special-pulse detection (`double_peaks_detection.py`)
2. Build activity histograms (`eel_data_preprocessing.py`) for each pulse type
3. Generate plots and correlations (`pulse_analysis_plots.py`, `correlate_pulse_shapes.py`, …)
4. Run environmental, feeding, and position analyses as needed

Paths default to a development subset under `data/raw/eels-mfn2021_dummy_pulses_redetected/`. Switch to the full dataset in `code/data_paths.py`.

## External tools

- **Video/audio sync:** `/home/wrk/videosync/src/sync_video_2.py`
- **Raw lab data:** `/data2/labdata/eels-mfn2021/berlin_tank_site/`

## Timeline

| Month | Focus |
|-------|-------|
| März–April | Einarbeitung, Berlin recordings, audio/video sync |
| Mai–Juni | Pulse detection, data analysis, thesis writing (Intro/MatMet) |
| Juli–August | Hardware for Brazil, results outline, abstract |
| September | Field work in Brazil |
| Oktober–November | Data analysis, results, discussion, submission |
