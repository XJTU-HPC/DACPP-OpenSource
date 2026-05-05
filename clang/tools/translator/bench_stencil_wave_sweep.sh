#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

RUNS="${SWEEP_BENCH_RUNS:-3}"
MPI_RANKS="${SWEEP_BENCH_RANKS:-4}"
TIME_STEPS="${SWEEP_BENCH_TIME_STEPS:-200}"
SIZES="${SWEEP_BENCH_SIZES:-512 1024 1536}"
OUT_DIR="${SWEEP_BENCH_OUT_DIR:-/Volumes/QUQ/working/mpi_tmp/stencil_wave_sweep}"
RESULTS="$OUT_DIR/results.tsv"

mkdir -p "$OUT_DIR"
printf 'case\tnx\tny\ttime_steps\tranks\truns\ttranslated_s\tcoarse_s\tratio\n' > "$RESULTS"

extract_summary() {
    local log="$1"
    python3 - "$log" <<'PY'
from pathlib import Path
import re
import sys

text = Path(sys.argv[1]).read_text(errors="ignore")
translated = re.findall(r"translated median: ([0-9]+(?:\.[0-9]+)?)s", text)
coarse = re.findall(r"coarse MPI\+SYCL median: ([0-9]+(?:\.[0-9]+)?)s", text)
ratio = re.findall(r"ratio translated/coarse: ([0-9]+(?:\.[0-9]+)?)x", text)
if not translated or not coarse or not ratio:
    raise SystemExit("missing benchmark summary")
print(translated[-1], coarse[-1], ratio[-1])
PY
}

run_case() {
    local name="$1"
    local size="$2"
    local log="$OUT_DIR/${name}_${size}.log"

    echo "========================================================"
    echo "  Sweep: $name ${size}x${size} steps=$TIME_STEPS ranks=$MPI_RANKS runs=$RUNS"
    echo "========================================================"

    if [[ "$name" == "stencil" ]]; then
        STENCIL_BENCH_NX="$size" \
        STENCIL_BENCH_NY="$size" \
        STENCIL_BENCH_TIME_STEPS="$TIME_STEPS" \
        STENCIL_BENCH_RANKS="$MPI_RANKS" \
        STENCIL_BENCH_RUNS="$RUNS" \
            bash "$SCRIPT_DIR/bench_stencil_mpi.sh" > "$log" 2>&1
    else
        WAVE_BENCH_NX="$size" \
        WAVE_BENCH_NY="$size" \
        WAVE_BENCH_TIME_STEPS="$TIME_STEPS" \
        WAVE_BENCH_RANKS="$MPI_RANKS" \
        WAVE_BENCH_RUNS="$RUNS" \
            bash "$SCRIPT_DIR/bench_wave_mpi.sh" > "$log" 2>&1
    fi
    local rc=$?
    tail -n 8 "$log"
    if [[ "$rc" -ne 0 ]]; then
        echo "[FAIL] $name ${size}x${size}; see $log"
        return "$rc"
    fi

    local translated coarse ratio
    read -r translated coarse ratio < <(extract_summary "$log")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "$size" "$size" "$TIME_STEPS" "$MPI_RANKS" "$RUNS" \
        "$translated" "$coarse" "$ratio" >> "$RESULTS"
}

failed=0
for size in $SIZES; do
    run_case stencil "$size" || failed=1
    run_case wave "$size" || failed=1
done

echo
echo "Results: $RESULTS"
column -t -s $'\t' "$RESULTS" 2>/dev/null || cat "$RESULTS"
exit "$failed"
