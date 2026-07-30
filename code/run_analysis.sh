#!/usr/bin/env bash
# Unified Berlin eel analysis pipeline.
#
# Usage:
#   ./run_analysis.sh dummy   # development subset → figures/, data/processed/dummy/
#   ./run_analysis.sh full    # full dataset       → figures/full/, data/processed/
#
# Both modes run the same analysis steps (classifier → histograms → thesis figures
# + processed dual-writes). Only EEL_USE_DUMMY_DATASET and the log file change.
set -euo pipefail

ROOT="/home/eisele/wrk/mscthesis"
CODE="$ROOT/code"
LOGDIR="$ROOT/data/intermediate/pipeline_logs"
mkdir -p "$LOGDIR"

MODE="${1:-}"
case "$MODE" in
  dummy)
    export EEL_USE_DUMMY_DATASET=true
    LOGFILE="$LOGDIR/pipeline_dummy.log"
    FIGURES_HINT="figures/"
    PROCESSED_HINT="data/processed/dummy/"
    ;;
  full)
    export EEL_USE_DUMMY_DATASET=false
    LOGFILE="$LOGDIR/pipeline_full.log"
    FIGURES_HINT="figures/full/"
    PROCESSED_HINT="data/processed/"
    ;;
  *)
    echo "Usage: $0 {dummy|full}" >&2
    echo "  dummy  — development H5 subset; figures under figures/" >&2
    echo "  full   — full Berlin tank dataset; figures under figures/full/" >&2
    exit 1
    ;;
esac

export MPLBACKEND=Agg
cd "$CODE"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"; }

run_step() {
  local label="$1"
  shift
  log "=== $label ==="
  "$@"
  log "=== Done: $label ==="
}

run_pulse_type_loop() {
  local label="$1"
  local script="$2"
  local pt
  for pt in all double wide; do
    log "=== $label ($pt) ==="
    printf '%s\n' "$pt" | python3 "$script"
    log "=== Done: $label ($pt) ==="
  done
}

run_stdin_step() {
  local label="$1"
  local stdin_payload="$2"
  local script="$3"
  log "=== $label ==="
  printf '%s' "$stdin_payload" | python3 "$script"
  log "=== Done: $label ==="
}

log "Pipeline started (mode=$MODE, figures -> $FIGURES_HINT, processed -> $PROCESSED_HINT)"

run_step "supervised special pulse classification" \
  python3 special_pulses/double_peaks_detection.py

run_pulse_type_loop "activity preprocessing" \
  activity_timescales/eel_data_preprocessing.py

run_pulse_type_loop "pulse analysis plots" \
  activity_timescales/pulse_analysis_plots.py

run_step "pulse shape correlation" \
  python3 correlations/correlate_pulse_shapes.py

run_step "half-width distributions" \
  python3 special_pulses/half_width_distributions.py

run_step "prototype pulse plots" \
  python3 special_pulses/prototype_pulse_plots.py

run_step "pulse shape distribution" \
  python3 special_pulses/pulse_shape_distribution.py

run_step "double-pulse template fit (prototype median, model A)" \
  python3 special_pulses/double_pulse_template_fit.py

run_step "PCA space plots" \
  python3 special_pulses/pca_space_plots.py

run_step "pulse properties over time" \
  python3 special_pulses/pulse_properties_over_time.py

run_step "mating correlation" \
  python3 correlations/correlate_mating_with_activity.py

run_step "half-width threshold WIP" \
  python3 special_pulses/half_width_threshold_waveforms.py

run_step "environment / volley / tank area" \
  python3 correlations/correlate_activity_with_environment.py

run_step "feeding correlation" \
  python3 correlations/correlate_activity_with_feeding.py

run_stdin_step "position preprocessing" $'\n' \
  position_estimation/position_data_preprocessing.py

run_stdin_step "position analysis plots" $'\n' \
  position_estimation/position_analysis_plots.py

log "Pipeline finished successfully (mode=$MODE)."
