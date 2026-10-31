#!/bin/bash
# Ablations: isolate the effect of non-IID shards, int8 transport, and churn.
set -e
W="$(cd "$(dirname "$0")" && pwd)"
OUT="${BB_THREAD_STORAGE:-/tmp}"
COMMON="--peers 6 --rounds 25 --inner-steps 40 --chars-per-peer 40000 --batch-size 16"

run () { echo "=== $1 ==="; "$W/.venv/bin/python" -m dptrain.run $COMMON ${@:2} \
           --out "$OUT/report_$1.json" 2>&1 | grep -v "^OMP"; }

cd "$W"
run main
run iid        --iid
run fp32       --no-compress
run nochurn    --drop-prob 0 --leave-prob 0 --join-prob 0
echo "ALL DONE"
