#!/usr/bin/env bash
# Snapshot pipeline health for the agent 2h monitor. Read-only except updating
# monitor_state.json timestamps / last_check fields.
set -euo pipefail

LOGDIR=/home/eisele/wrk/mscthesis/data/intermediate/pipeline_logs
STATE="$LOGDIR/monitor_state.json"
NOW=$(date -Is)
STALE_LOG_SECS=${STALE_LOG_SECS:-7200}   # no log growth for 2h → suspect
STALE_CPU_SECS=${STALE_CPU_SECS:-3600}   # 0% CPU for 1h while claiming to run

phase=$(python3 -c "import json; print(json.load(open('$STATE')).get('phase','full'))" 2>/dev/null || echo full)
if [[ "$phase" == "dummy" ]]; then
  MODE=dummy
  PIDFILE="$LOGDIR/pipeline_dummy.pid"
  LOG="$LOGDIR/pipeline_dummy.log"
  STDOUT="$LOGDIR/pipeline_dummy_stdout.log"
else
  MODE=full
  PIDFILE="$LOGDIR/pipeline_full.pid"
  LOG="$LOGDIR/pipeline_full.log"
  STDOUT="$LOGDIR/pipeline_full_stdout.log"
fi

PID=$(cat "$PIDFILE" 2>/dev/null || echo "")
alive=0
stat="missing"
etime="-"
pcpu="-"
pmem="-"
cmd="-"
if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
  alive=1
  read -r etime stat pcpu pmem cmd < <(ps -p "$PID" -o etime=,stat=,pcpu=,pmem=,cmd= --no-headers)
fi

# Child python (first matching)
child_line=$(ps --ppid "$PID" -o pid=,etime=,stat=,pcpu=,pmem=,cmd= --no-headers 2>/dev/null | head -1 || true)

log_mtime=$(stat -c %Y "$LOG" 2>/dev/null || echo 0)
stdout_mtime=$(stat -c %Y "$STDOUT" 2>/dev/null || echo 0)
now_epoch=$(date +%s)
log_age=$(( now_epoch - log_mtime ))
stdout_age=$(( now_epoch - stdout_mtime ))
newest_age=$log_age
(( stdout_age < newest_age )) && newest_age=$stdout_age

last_start=$(grep -E '=== ' "$LOG" 2>/dev/null | grep -v 'Done:' | tail -1 || true)
last_done=$(grep '=== Done:' "$LOG" 2>/dev/null | tail -1 || true)
finished=$(grep -c 'Pipeline finished successfully' "$LOG" 2>/dev/null || true)
traceback=$(grep -c -E 'Traceback \(most recent call last\)|^ERROR|Error:|Exception:' "$STDOUT" 2>/dev/null || true)
tail_stdout=$(tail -n 8 "$STDOUT" 2>/dev/null || true)
tail_log=$(tail -n 8 "$LOG" 2>/dev/null || true)

verdict="RUNNING"
reason="ok"
if (( finished > 0 )) && (( alive == 0 )); then
  verdict="FINISHED"
  reason="pipeline finished successfully; process exited"
elif (( alive == 0 )); then
  verdict="DEAD"
  reason="pid $PID not running and no success marker"
elif (( traceback > 0 )) && grep -q -E 'Traceback \(most recent call last\)' "$STDOUT" 2>/dev/null; then
  # Only flag if traceback is near the end (last 80 lines)
  if tail -n 80 "$STDOUT" | grep -q 'Traceback (most recent call last)'; then
    verdict="ERROR"
    reason="traceback near end of stdout"
  fi
elif (( newest_age > STALE_LOG_SECS )); then
  verdict="STALE"
  reason="no log/stdout mtime update for ${newest_age}s (>${STALE_LOG_SECS}s)"
elif [[ "$pcpu" != "-" ]] && awk -v c="$pcpu" 'BEGIN{exit !(c+0 < 0.1)}' && (( newest_age > STALE_CPU_SECS )); then
  verdict="STALE_IDLE"
  reason="pid alive but pcpu=${pcpu}% and no I/O for ${newest_age}s"
fi

cat <<EOF
=== PIPELINE HEALTH $(date -Is) ===
mode=$MODE phase=$phase
pid=$PID alive=$alive etime=$etime stat=$stat pcpu=$pcpu pmem=$pmem
cmd=$cmd
child: ${child_line:-none}
log_age_s=$log_age stdout_age_s=$stdout_age newest_age_s=$newest_age
last_start: $last_start
last_done:  $last_done
finished_markers=$finished traceback_hits=$traceback
VERDICT=$verdict
REASON=$reason

--- log tail ---
$tail_log

--- stdout tail ---
$tail_stdout
EOF

python3 - <<PY
import json
from pathlib import Path
p = Path("$STATE")
d = json.loads(p.read_text()) if p.exists() else {}
d["last_check"] = "$NOW"
d["last_verdict"] = "$verdict"
d["last_reason"] = """$reason"""
d["phase"] = "$phase"
if "$MODE" == "full":
    d["full_status"] = {"FINISHED":"done","DEAD":"failed","ERROR":"failed","STALE":"stale","STALE_IDLE":"stale"}.get("$verdict","running")
else:
    d["dummy_status"] = {"FINISHED":"done","DEAD":"failed","ERROR":"failed","STALE":"stale","STALE_IDLE":"stale"}.get("$verdict","running")
p.write_text(json.dumps(d, indent=2) + "\n")
PY

# Nonzero exit for DEAD/ERROR so callers can branch; STALE still 0 (agent decides)
case "$verdict" in
  DEAD|ERROR) exit 2 ;;
  *) exit 0 ;;
esac
