#!/usr/bin/env bash
# sweep.sh <label1:config1.json> <label2:config2.json> ...
#
# Runs a sequence of experiment configs back-to-back via run_exp.sh, with NO
# concurrent compilation (call only when cargo is idle — build times are otherwise
# contaminated). Warms the dataset into page cache first so the first run isn't
# penalized by cold disk reads, making build times comparable across runs.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "[sweep] warming page cache (datasets/enron_fp16.bin) ..."
cat "$ROOT/datasets/enron_fp16.bin" > /dev/null 2>&1 || true

for pair in "$@"; do
  label="${pair%%:*}"
  cfg="${pair#*:}"
  echo "[sweep] === $label  ($cfg)  $(date -Is) ==="
  "$ROOT/scripts/run_exp.sh" "$cfg" "$label"
  echo "[sweep] === $label done rc=$?  $(date -Is) ==="
done
echo "[sweep] ALL DONE $(date -Is)"
