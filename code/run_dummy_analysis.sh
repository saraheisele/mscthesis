#!/usr/bin/env bash
# Dummy-dataset analysis pipeline — outputs go to data/processed/dummy/ (never overwrites full dataset).
set -euo pipefail

ROOT="/home/eisele/wrk/mscthesis"
CODE="$ROOT/code"
LOGDIR="$ROOT/data/intermediate/pipeline_logs"
mkdir -p "$LOGDIR"

export MPLBACKEND=Agg
cd "$CODE"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGDIR/pipeline_dummy.log"; }

run_ml_detection() {
  log "=== Part 1a: supervised special pulse classification ==="
  python3 special_pulses/double_peaks_detection.py
  log "=== Done: supervised special pulse classification ==="
}

run_preprocessing() {
  local pt="$1"
  log "=== Part 1b: activity preprocessing ($pt) ==="
  printf '%s\n' "$pt" | python3 activity_timescales/eel_data_preprocessing.py
  log "=== Done: activity preprocessing ($pt) ==="
}

run_plots() {
  local pt="$1"
  log "=== Part 1c: pulse analysis plots ($pt) ==="
  printf '%s\n' "$pt" | python3 activity_timescales/pulse_analysis_plots.py
  log "=== Done: pulse analysis plots ($pt) ==="
}

log "Dummy pipeline started (USE_DUMMY_DATASET=true, outputs under data/processed/dummy/)"

run_ml_detection

for pt in all double wide; do
  run_preprocessing "$pt"
done

for pt in all double wide; do
  run_plots "$pt"
done

log "=== Part 1d: pulse shape correlation ==="
python3 correlations/correlate_pulse_shapes.py
log "=== Done: pulse shape correlation ==="

log "=== Part 2a: half-width distributions ==="
python3 special_pulses/half_width_distributions.py
log "=== Done: half-width distributions ==="

log "=== Part 2b: prototype pulse plots ==="
python3 special_pulses/prototype_pulse_plots.py
log "=== Done: prototype pulse plots ==="

log "=== Part 2e: PCA space plots ==="
python3 special_pulses/pca_space_plots.py
log "=== Done: PCA space plots ==="

log "=== Part 2c: pulse properties over time ==="
python3 special_pulses/pulse_properties_over_time.py
log "=== Done: pulse properties over time ==="

log "=== Part 2d: mating correlation ==="
python3 correlations/correlate_mating_with_activity.py
log "=== Done: mating correlation ==="

log "=== Part 3: environment / volley / tank area ==="
python3 correlations/correlate_activity_with_environment.py
log "=== Done: environment / volley / tank area ==="

log "=== Part 4: feeding correlation ==="
python3 correlations/correlate_activity_with_feeding.py
log "=== Done: feeding correlation ==="

log "=== Part 5a: position preprocessing ==="
printf '\n' | python3 position_estimation/position_data_preprocessing.py
log "=== Done: position preprocessing ==="

log "=== Part 5b: position analysis plots ==="
printf '\n' | python3 position_estimation/position_analysis_plots.py
log "=== Done: position analysis plots ==="

log "Dummy pipeline finished successfully."
