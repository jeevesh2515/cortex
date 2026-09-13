#!/bin/sh
set -eu
MODEL=${1:-opencode/muse-spark-1.2-contributor-free}
OUT=$(mktemp)
trap 'rm -f "$OUT"' EXIT
if ! timeout 40 opencode run --pure --dir /tmp --model "$MODEL" --format json 'Return exactly READY and nothing else. Do not access files or use tools.' >"$OUT" 2>&1; then
  echo "NOT_READY model=$MODEL"; cat "$OUT"; exit 1
fi
if grep -q '"type":"error"' "$OUT" || ! grep -q 'READY' "$OUT"; then
  echo "NOT_READY model=$MODEL"; cat "$OUT"; exit 1
fi
echo "READY model=$MODEL"
