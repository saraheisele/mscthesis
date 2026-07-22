#!/usr/bin/env bash
# Full-dataset pipeline — thesis figures only, saved under figures/full/.
set -euo pipefail

ROOT="/home/eisele/wrk/mscthesis"
CODE="$ROOT/code"
LOGDIR="$ROOT/data/intermediate/pipeline_logs"
mkdir -p "$LOGDIR"

export MPLBACKEND=Agg
export EEL_USE_DUMMY_DATASET=false
cd "$CODE"

LOGFILE="$LOGDIR/pipeline_full_thesis.log"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"; }

run_preprocessing() {
  local pt="$1"
  log "=== Activity preprocessing ($pt) ==="
  printf '%s\n' "$pt" | python3 activity_timescales/eel_data_preprocessing.py
  log "=== Done: activity preprocessing ($pt) ==="
}

run_plots() {
  local pt="$1"
  log "=== Circadian panels ($pt) ==="
  printf '%s\n' "$pt" | python3 activity_timescales/pulse_analysis_plots.py
  log "=== Done: circadian panels ($pt) ==="
}

log "Full-dataset thesis pipeline started (figures -> figures/full/)"

log "=== Apply special-pulse classifier to full H5 files ==="
python3 special_pulses/double_peaks_detection.py
log "=== Done: classifier application ==="

for pt in all double wide; do
  run_preprocessing "$pt"
done

for pt in all double wide; do
  run_plots "$pt"
done

log "=== Half-width distributions ==="
python3 special_pulses/half_width_distributions.py
log "=== Done: half-width distributions ==="

log "=== Prototype pulse plots ==="
python3 special_pulses/prototype_pulse_plots.py
log "=== Done: prototype pulse plots ==="

log "=== Double-pulse template fit ==="
python3 special_pulses/double_pulse_template_fit.py
log "=== Done: double-pulse template fit ==="

log "=== PCA space plot ==="
python3 special_pulses/pca_space_plots.py
log "=== Done: PCA space plot ==="

log "=== Half-width KDE (pulse properties) ==="
python3 special_pulses/pulse_properties_over_time.py
log "=== Done: half-width KDE ==="

log "=== Environment correlations ==="
python3 correlations/environment_correlation.py
log "=== Done: environment correlations ==="

log "=== Feeding correlations ==="
python3 correlations/correlate_activity_with_feeding.py
log "=== Done: feeding correlations ==="

log "=== Position preprocessing ==="
printf '\n' | python3 position_estimation/position_data_preprocessing.py
log "=== Done: position preprocessing ==="

log "=== Position thesis plots ==="
printf '\n' | python3 position_estimation/position_analysis_plots.py
log "=== Done: position thesis plots ==="

log "Full-dataset thesis pipeline finished successfully."
