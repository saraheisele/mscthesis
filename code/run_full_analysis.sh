#!/usr/bin/env bash
# Full Berlin eel analysis pipeline — logs each part to data/intermediate/pipeline_logs/
set -euo pipefail

ROOT="/home/eisele/wrk/mscthesis"
CODE="$ROOT/code"
LOGDIR="$ROOT/data/intermediate/pipeline_logs"
mkdir -p "$LOGDIR"

export MPLBACKEND=Agg
cd "$CODE"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGDIR/pipeline.log"; }

run_detection() {
  local mode="$1"
  log "=== Part 1a: special pulse detection ($mode) ==="
  python3 - <<PY
import sys
from pathlib import Path
sys.path.insert(0, ".")
from path_setup import setup_script_paths
setup_script_paths("special_pulses/double_peaks_detection.py")
import special_pulses.double_peaks_detection as det
from data_paths import H5_DIR

det.DETECTION_MODE = "$mode"
det.ARRAY_NAME = det.MODE_CONFIG["$mode"]["array_name"]
det.DISPLAY_NAME = det.MODE_CONFIG["$mode"]["display_name"]
det.process_all_h5_files(H5_DIR)
PY
  log "=== Done: special pulse detection ($mode) ==="
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

log "Pipeline started (H5_DIR=/home/efish/eelsmfn2021_eods/berlin_tank_site)"

for mode in double wide fat; do
  run_detection "$mode"
done

for pt in all double wide fat; do
  run_preprocessing "$pt"
done

for pt in all double wide fat; do
  run_plots "$pt"
done

log "=== Part 1d: pulse shape correlation ==="
python3 correlations/correlate_pulse_shapes.py
log "=== Done: pulse shape correlation ==="

log "=== Part 2: prototype pulse plots ==="
python3 special_pulses/prototype_pulse_plots.py
log "=== Done: prototype pulse plots ==="

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

log "Pipeline finished successfully."
