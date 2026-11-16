#!/usr/bin/env bash
# Run an IREE pass pipeline in process with the collector installed, then fold
# the trace into a single table.
#
#   ./run.sh <model.mlir> [collect|order|all]
#
# Environment:
#   HARNESS  path to the driver binary (default: <repo>/build/iree-harness/iree-harness)
#   OUT      output directory (default: <repo>/build/run)
#   PIPELINE override the pipeline list, comma separated
#   TARGETS  override the target flags
#
# The reference run for this repository used a 39.6 MB fp16 ONNX model; any
# MLIR file that IREE can consume works. See README.md for the numbers.
set -u

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
HARNESS=${HARNESS:-$REPO/build/iree-harness/iree-harness}
OUT=${OUT:-$REPO/build/run}
STEP=${2:-all}

MODEL=${1:-}
[ -n "$MODEL" ] || { echo "usage: $0 <model.mlir> [collect|order|all]" >&2; exit 2; }
[ -f "$MODEL" ] || { echo "no such file: $MODEL" >&2; exit 2; }

# The five IREE pipelines, in order. Each one is marked in the trace so the
# table can say which stage a step belongs to.
PIPELINE=${PIPELINE:-iree-global-optimization-transformation-pipeline,iree-dispatch-creation-pipeline,iree-flow-transformation-pipeline,iree-stream-transformation-pipeline,iree-hal-transformation-pipeline}

if [ -z "${TARGETS+x}" ]; then
  TARGETS=(--iree-hal-target-device=local
           --iree-hal-local-target-device-backends=llvm-cpu
           --iree-llvmcpu-target-cpu=generic)
fi

TRACE=$OUT/trace.json
CSV=$OUT/pass-order.csv

case $STEP in
  collect)
    [ -x "$HARNESS" ] || { echo "no driver at $HARNESS (see examples/iree/README in the main README)" >&2; exit 2; }
    mkdir -p "$OUT"
    # The collector stays on; --mlir-disable-threading keeps the trace readable
    # and the timings comparable between runs.
    COMPILE_ANALYSIS=$TRACE \
      "$HARNESS" --mlir-disable-threading "$MODEL" \
      --pipeline="$PIPELINE" "${TARGETS[@]}" > /dev/null || true
    [ -s "$TRACE" ] || { echo "no trace written to $TRACE" >&2; exit 1; }
    echo "trace: $TRACE ($(du -h "$TRACE" | cut -f1))" ;;
  order)
    python3 "$REPO/tools/pass-order.py" --trace "$TRACE" --csv "$CSV" ;;
  all)
    "$0" "$MODEL" collect && "$0" "$MODEL" order ;;
  *)
    echo "unknown step: $STEP" >&2; exit 2 ;;
esac
