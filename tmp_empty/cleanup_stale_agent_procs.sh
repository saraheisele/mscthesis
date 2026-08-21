#!/usr/bin/env bash
# Kill orphaned Cursor agent sandboxes / leftover pipeline watchers.
# Run in a normal (non-agent) terminal:
#   bash /home/eisele/wrk/mscthesis/tmp_empty/cleanup_stale_agent_procs.sh
set -uo pipefail

ME=$$
echo "Running as $(id -un) (pid $ME)"

before=$(pgrep -c -f '/cursorsandbox --policy' || true)
echo "cursorsandbox before: ${before:-0}"

# Collect PIDs of orphaned sandboxes (anything matching the helper binary)
mapfile -t SB < <(pgrep -f '/cursorsandbox --policy' || true)

echo "Sending SIGKILL to ${#SB[@]} cursorsandbox PIDs..."
fail=0
if ((${#SB[@]} > 0)); then
  # Prefer killall if available (clearest intent)
  if command -v killall >/dev/null 2>&1; then
    killall -9 cursorsandbox 2>&1 || true
  fi
  # Also kill by absolute path match in case argv0 differs
  printf '%s\n' "${SB[@]}" | xargs -r -n 64 kill -9 2>&1 || true
  sleep 1
  for pid in "${SB[@]}"; do
    if ps -p "$pid" >/dev/null 2>&1; then
      err=$(kill -9 "$pid" 2>&1) || true
      if ps -p "$pid" >/dev/null 2>&1; then
        echo "FAILED kill -9 $pid (${err:-still alive})"
        fail=$((fail + 1))
        # show why
        ps -p "$pid" -o pid,user,uid,stat,wchan:30,cmd= | cat || true
      fi
    fi
  done
fi
sleep 1

# Leftover long sleeps from agent heartbeats / health loops
mapfile -t SLEEPS < <(ps -eo pid=,args= | awk '/\/bin\/sleep (7200|900)$|^sleep (7200|900)$/ {print $1}')
for pid in "${SLEEPS[@]:-}"; do
  [[ -z "${pid:-}" || "$pid" == "$ME" ]] && continue
  if ps -p "$pid" >/dev/null 2>&1; then
    echo "KILL sleep $pid"
    kill -9 "$pid" 2>/dev/null || true
  fi
done

# Stuck strace from Jul-31 inspect (if any remain)
pkill -9 -f 'strace -p 207101' 2>/dev/null || true

after=$(pgrep -c -f '/cursorsandbox --policy' || true)
alive_sample=$(pgrep -af '/cursorsandbox --policy' | head -3 || true)
loops=$(pgrep -c -f 'AGENT_LOOP' || true)

echo
echo "cursorsandbox after: ${after:-0} (was ${before:-0})"
echo "AGENT_LOOP left: ${loops:-0}"
echo "kill failures: $fail"
if [[ -n "${alive_sample}" ]]; then
  echo "sample still-alive:"
  echo "$alive_sample"
fi
if [[ "${after:-0}" -gt 0 ]]; then
  echo
  echo "If count did not drop, try from the same terminal:"
  echo "  killall -9 cursorsandbox"
  echo "  # or inspect one PID:"
  echo "  pid=\$(pgrep -n -f '/cursorsandbox --policy'); ps -p \$pid -o pid,user,stat,wchan,cmd; kill -9 \$pid; echo exit=\$?"
fi
echo "Done."
