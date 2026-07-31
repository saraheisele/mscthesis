# Masters Thesis — Sarah Eisele

Electric eel (*Electrophorus electricus*) behaviour analysis using Berlin tank
grid recordings.

## Installation

```bash
pip install -r requirements.txt
```

Some dependencies (`audian`, `audioio`, `thunderfish`, `thunderlab`) are
installed from the bendalab GitHub repositories. A full install requires
network access and git.

## Project structure

```
mscthesis/
├── code/                  Analysis pipeline (see code/README.md)
│   ├── archived/          Legacy / WIP / exploratory (not in runner)
│   ├── activity_timescales/
│   ├── special_pulses/
│   ├── correlations/
│   └── position_estimation/
├── data/
│   ├── raw/               Predetected pulses, environmental Excel files
│   ├── intermediate/      Preprocessed histograms (.npz), classifier artifacts
│   └── processed/         Diagnostic plots / tables (dummy under processed/dummy/)
├── figures/               Thesis figures (dummy); full-dataset copies under figures/full/
├── requirements.txt
└── README.md
```

## Analysis pipeline

All analysis scripts live in [`code/`](code/). See [`code/README.md`](code/README.md)
for the full script reference, figure layout, and recommended run order.

**End-to-end flow**

1. Classify special pulses (double / wide) on predetected H5 files  
2. Build multi-timescale activity histograms and circadian panels  
3. Pulse-shape metrics (half-width, prototypes, template fit, PCA, trends)  
4. Correlate activity with mating notes, environment/volleys/tank area, feeding  
5. Estimate and plot position along the electrode line  

**Typical workflow** — run the unified pipeline from `code/`:

```bash
cd code
./run_analysis.sh dummy   # development subset → figures/, data/processed/dummy/
./run_analysis.sh full    # full Berlin dataset → figures/full/, data/processed/
```

Paths are selected via `EEL_USE_DUMMY_DATASET` (set by the runner) in
`code/data_paths.py`. Individual scripts can still be run interactively for
iteration.

### Parallel worktrees

To refactor or clean code without blocking thesis analysis on `master`:

```bash
git worktree add -b code-cleanup ../mscthesis-cleanup master
# work in ../mscthesis-cleanup; keep editing ../mscthesis on master
```

`run_analysis.sh` resolves the project root from its own path, so either
checkout works.

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
