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
│   ├── intermediate/      Preprocessed histograms (.npz), classifier artifacts
│   └── processed/         Diagnostic plots / tables (dummy under processed/dummy/)
├── figures/               Thesis figures (dummy); full-dataset copies under figures/full/
├── requirements.txt
└── README.md
```

## Analysis pipeline

All analysis scripts live in [`code/`](code/). See [`code/README.md`](code/README.md) for the full script reference, figure layout, and recommended run order.

**Typical workflow** — run the unified pipeline from `code/`:

```bash
cd code
./run_analysis.sh dummy   # development subset → figures/, data/processed/dummy/
./run_analysis.sh full    # full Berlin dataset → figures/full/, data/processed/
```

Paths are selected via `EEL_USE_DUMMY_DATASET` (set by the runner) in `code/data_paths.py`. Individual scripts can still be run interactively for iteration.

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
