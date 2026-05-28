#!/usr/bin/env bash
# Memory watchdog: tail free RAM and SIGTERM/SIGKILL a target PID if it gets
# dangerously low. Intended to babysit lerobot-train so we never repeat the
# DGX-Spark OOM lockup.
#
# Usage:
#   bash scripts/mem_watchdog.sh <pid> <min_free_gb> <log_file>
#
# Polls /proc/meminfo every 2s. Reports MemAvailable to log. If
# MemAvailable < min_free_gb for 2 consecutive samples, sends SIGTERM to the
# target PID, waits 10s, then SIGKILL if still alive.

set -euo pipefail

PID=${1:?pid required}
MIN_GB=${2:-20}
LOG=${3:-/tmp/mem_watchdog.log}

MIN_KB=$((MIN_GB * 1024 * 1024))
below_count=0

echo "[$(date -Iseconds)] watchdog start  pid=$PID  min_free=${MIN_GB} GiB" > "$LOG"

while kill -0 "$PID" 2>/dev/null; do
  avail_kb=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
  avail_gb=$(awk -v k="$avail_kb" 'BEGIN{printf "%.1f", k/1024/1024}')
  printf "[%s] MemAvailable=%s GiB\n" "$(date -Iseconds)" "$avail_gb" >> "$LOG"

  if [[ "$avail_kb" -lt "$MIN_KB" ]]; then
    below_count=$((below_count + 1))
    echo "  -> below threshold (count=$below_count)" >> "$LOG"
    if [[ "$below_count" -ge 2 ]]; then
      echo "[$(date -Iseconds)] KILLING pid=$PID (MemAvailable=${avail_gb} GiB < ${MIN_GB} GiB)" >> "$LOG"
      kill -TERM "$PID" 2>/dev/null || true
      sleep 10
      kill -KILL "$PID" 2>/dev/null || true
      echo "[$(date -Iseconds)] watchdog exit" >> "$LOG"
      exit 0
    fi
  else
    below_count=0
  fi
  sleep 2
done

echo "[$(date -Iseconds)] target pid $PID gone, watchdog exit" >> "$LOG"
