#!/usr/bin/env bash
# run_exp.sh <config.json> <label>
#
# Runs the diskann-benchmark binary on <config.json> under an RSS watchdog that
# kills the run if resident memory exceeds RSS_LIMIT_GB (protects this WSL — see
# the project's WSL-memory-safety rule). Captures full stdout to
# experiments/logs/<label>.log, tracks peak RSS, and appends a summary line to
# experiments/RESULTS.md. Per-L recall/QPS are read from the log at report time.
#
# The benchmark is a single multi-threaded process, so /proc/<pid>/status VmRSS
# is the whole working set — no process-tree summation needed.
set -uo pipefail

if [ "${1:-}" = "--help" ] || [ $# -lt 2 ]; then
  echo "usage: run_exp.sh <config.json> <label>"
  exit 0
fi

CONFIG="$1"
LABEL="$2"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/target/release/diskann-benchmark"
LOGDIR="$ROOT/experiments/logs"
RES="$ROOT/experiments/RESULTS.md"
RSS_LIMIT_GB="${RSS_LIMIT_GB:-26}"
RSS_LIMIT_KB=$(( RSS_LIMIT_GB * 1024 * 1024 ))

mkdir -p "$LOGDIR"
LOG="$LOGDIR/$LABEL.log"
[ -x "$BIN" ] || { echo "[run_exp] binary missing: $BIN"; exit 1; }
[ -f "$CONFIG" ] || { echo "[run_exp] config missing: $CONFIG"; exit 1; }

echo "[run_exp] $LABEL  config=$CONFIG  limit=${RSS_LIMIT_GB}GB  start=$(date -Is)" | tee "$LOG"

# Launch the benchmark from the worktree root (so relative search_directories resolve).
OUTJSON="$LOGDIR/$LABEL.out.json"
( cd "$ROOT" && "$BIN" run --input-file "$CONFIG" --output-file "$OUTJSON" ) >>"$LOG" 2>&1 &
BPID=$!

PEAKFILE="$(mktemp)"
echo 0 > "$PEAKFILE"
# Watchdog: sample VmRSS every 1s, track peak, kill if over the limit.
(
  max=0
  while kill -0 "$BPID" 2>/dev/null; do
    rss=$(awk '/^VmRSS:/{print $2}' "/proc/$BPID/status" 2>/dev/null)
    if [ -n "${rss:-}" ]; then
      [ "$rss" -gt "$max" ] && { max="$rss"; echo "$max" > "$PEAKFILE"; }
      if [ "$rss" -gt "$RSS_LIMIT_KB" ]; then
        echo "[run_exp] !!! RSS $((rss/1024/1024))GB > ${RSS_LIMIT_GB}GB — KILLING $LABEL" | tee -a "$LOG"
        kill -9 "$BPID" 2>/dev/null
        break
      fi
    fi
    sleep 1
  done
) &
WPID=$!

wait "$BPID"; RC=$?
kill "$WPID" 2>/dev/null; wait "$WPID" 2>/dev/null

PEAKKB=$(cat "$PEAKFILE" 2>/dev/null || echo 0); rm -f "$PEAKFILE"
PEAKGB=$(awk "BEGIN{printf \"%.2f\", ${PEAKKB:-0}/1048576}")
BUILD=$(grep -oE "Build time: [0-9.]+s" "$LOG" | tail -1)
AVGDEG=$(grep -oE "avg_degree[ =:]+[0-9.]+" "$LOG" | tail -1)

echo "[run_exp] $LABEL done rc=$RC  ${BUILD:-<no-build-line>}  ${AVGDEG:-}  peakRSS=${PEAKGB}GB  end=$(date -Is)" | tee -a "$LOG"

{
  echo ""
  echo "### $LABEL — rc=$RC, ${BUILD:-build:?}, peakRSS=${PEAKGB}GB, $(date -Is)"
  echo '```'
  # The search-stats table: header line containing "QPS" and "Recall", plus rows.
  grep -E "QPS|Recall|^\s*[0-9]+ " "$LOG" | grep -vE "Recall at,|recall_at" | tail -30
  echo '```'
} >> "$RES"

exit $RC
