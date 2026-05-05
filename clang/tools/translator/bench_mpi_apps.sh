#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
source "$SCRIPT_DIR/env.sh"

TESTS_DIR="$SCRIPT_DIR/tests"
TMP_DIR="${APP_BENCH_TMP_DIR:-/Volumes/QUQ/working/mpi_tmp/app_bench}"
MPI_RANKS="${APP_BENCH_RANKS:-4}"
RUNS="${APP_BENCH_RUNS:-3}"
SLOW_RATIO="${APP_BENCH_SLOW_RATIO:-10}"

APP_TESTS=(
    "MDP1.0"
    "liuliang1.0"
    "stencil1.0"
    "waveEquation1.0"
    "jacobi1.0"
    "oddeven0.1"
)

if [[ $# -gt 0 ]]; then
    APP_TESTS=("$@")
fi

rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"

pick_dac_source() {
    local source_dir="$1"
    find "$source_dir" -maxdepth 1 -type f -name "*.dac.cpp" \
        ! -name "*.mpi.dac.cpp" \
        ! -name "*.retranslated.dac.cpp" \
        ! -name "*.large_dac.cpp" \
        | sort | head -n 1
}

generated_cpp_path_for() {
    local source_file="$1"
    local base_name
    base_name="$(basename "$source_file")"
    if [[ "$base_name" == *.dac.cpp ]]; then
        printf '%s/%s\n' "$(dirname "$source_file")" "${base_name%.cpp}_sycl_buffer.cpp"
        return
    fi
    printf '%s/%s.dac_sycl_buffer.cpp\n' "$(dirname "$source_file")" "${base_name%.*}"
}

mpi_dac_path_for() {
    local source_file="$1"
    local dir_name
    local base_name
    dir_name="$(dirname "$source_file")"
    base_name="$(basename "$source_file")"
    if [[ "$base_name" == *.dac.cpp ]]; then
        printf '%s/%s.mpi.dac.cpp\n' "$dir_name" "${base_name%.dac.cpp}"
        return
    fi
    printf '%s/%s.mpi.dac.cpp\n' "$dir_name" "${base_name%.*}"
}

clean_output() {
    local input="$1"
    local output="$2"
    grep -v "AdaptiveCpp Warning" "$input" \
    | grep -v "This application uses SYCL buffers" \
    | grep -v "SYCL2020 USM model" \
    | grep -v "AdaptiveCpp performance guide" \
    | grep -v "https://github.com/AdaptiveCpp" \
    | grep -v '^[[:space:]]*$' \
    > "$output" || true
}

run_in_env() {
    local log="$1"; shift
    bash -lc "source '$SCRIPT_DIR/env.sh' && $*" > "$log" 2>&1
}

run_timed() {
    local log="$1"; shift
    python3 - "$log" "$@" <<'PY'
import subprocess
import sys
import time

log = sys.argv[1]
cmd = sys.argv[2:]
start = time.perf_counter()
with open(log, "wb") as out:
    proc = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT)
elapsed = time.perf_counter() - start
print(f"{elapsed:.6f}")
sys.exit(proc.returncode)
PY
}

median() {
    python3 - "$@" <<'PY'
import sys
vals = sorted(float(x) for x in sys.argv[1:])
if not vals:
    print("nan")
elif len(vals) % 2:
    print(f"{vals[len(vals)//2]:.6f}")
else:
    mid = len(vals) // 2
    print(f"{(vals[mid - 1] + vals[mid]) / 2:.6f}")
PY
}

ratio_of() {
    python3 - "$1" "$2" <<'PY'
import sys
num = float(sys.argv[1])
den = float(sys.argv[2])
print("inf" if den == 0 else f"{num / den:.2f}")
PY
}

is_slow_ratio() {
    python3 - "$1" "$2" <<'PY'
import sys
ratio = float(sys.argv[1]) if sys.argv[1] != "inf" else float("inf")
threshold = float(sys.argv[2])
sys.exit(0 if ratio >= threshold else 1)
PY
}

RESULTS="$TMP_DIR/results.tsv"
printf 'test\tbaseline_median_s\tmpi_median_s\tratio\tstatus\n' > "$RESULTS"

export DYLD_LIBRARY_PATH="$ACPP_ROOT/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"

for test_name in "${APP_TESTS[@]}"; do
    echo "========================================================"
    echo "  Bench: $test_name"
    echo "========================================================"

    dac_file="$(pick_dac_source "$TESTS_DIR/$test_name")"
    if [[ -z "$dac_file" ]]; then
        echo "[SKIP] No .dac.cpp found for $test_name"
        printf '%s\t%s\t%s\t%s\t%s\n' "$test_name" "-" "-" "-" "skip-no-source" >> "$RESULTS"
        continue
    fi

    work_dir="$TMP_DIR/$test_name"
    mkdir -p "$work_dir"

    base_dac="$work_dir/$(basename "$dac_file")"
    cp "$dac_file" "$base_dac"
    base_sycl="$(generated_cpp_path_for "$base_dac")"
    base_bin="$work_dir/base_bin"

    mpi_dac="$(mpi_dac_path_for "$base_dac")"
    cp "$dac_file" "$mpi_dac"
    mpi_sycl="$(generated_cpp_path_for "$mpi_dac")"
    mpi_bin="$work_dir/mpi_bin"

    echo "  [1/4] build baseline"
    if ! run_in_env "$work_dir/build_base.log" "dacpp '$base_dac' --mode=buffer && acpp-compile '$base_sycl' '$base_bin'"; then
        echo "[FAIL] baseline build failed"
        head -n 20 "$work_dir/build_base.log"
        printf '%s\t%s\t%s\t%s\t%s\n' "$test_name" "-" "-" "-" "fail-build-baseline" >> "$RESULTS"
        continue
    fi

    echo "  [2/4] build MPI"
    if ! run_in_env "$work_dir/build_mpi.log" "dacpp '$mpi_dac' --mode=buffer --mpi && acpp-compile '$mpi_sycl' '$mpi_bin'"; then
        echo "[FAIL] MPI build failed"
        head -n 20 "$work_dir/build_mpi.log"
        printf '%s\t%s\t%s\t%s\t%s\n' "$test_name" "-" "-" "-" "fail-build-mpi" >> "$RESULTS"
        continue
    fi
    if grep -Fq "<->" "$mpi_sycl"; then
        echo "[FAIL] generated MPI SYCL still contains <->"
        grep -Fn "<->" "$mpi_sycl" | head -n 5
        printf '%s\t%s\t%s\t%s\t%s\n' "$test_name" "-" "-" "-" "fail-residual-dac-expr" >> "$RESULTS"
        continue
    fi

    echo "  [3/4] run samples: runs=$RUNS np=$MPI_RANKS"
    base_times=()
    mpi_times=()
    compare_ok=1
    for run in $(seq 1 "$RUNS"); do
        base_log="$work_dir/base_run_${run}.out"
        mpi_log="$work_dir/mpi_run_${run}.out"

        base_time="$(run_timed "$base_log" "$base_bin")" || {
            echo "[FAIL] baseline run $run failed"
            head -n 20 "$base_log"
            compare_ok=0
            break
        }
        mpi_time="$(run_timed "$mpi_log" mpirun -np "$MPI_RANKS" "$mpi_bin")" || {
            echo "[FAIL] MPI run $run failed"
            head -n 20 "$mpi_log"
            compare_ok=0
            break
        }

        clean_output "$base_log" "$work_dir/base_run_${run}.clean"
        clean_output "$mpi_log" "$work_dir/mpi_run_${run}.clean"
        if ! diff -q "$work_dir/base_run_${run}.clean" "$work_dir/mpi_run_${run}.clean" > /dev/null 2>&1; then
            echo "[FAIL] output mismatch on run $run"
            diff "$work_dir/base_run_${run}.clean" "$work_dir/mpi_run_${run}.clean" | head -n 30
            compare_ok=0
            break
        fi

        base_times+=("$base_time")
        mpi_times+=("$mpi_time")
        echo "    run $run: baseline=${base_time}s mpi=${mpi_time}s"
    done

    if [[ "$compare_ok" != "1" ]]; then
        printf '%s\t%s\t%s\t%s\t%s\n' "$test_name" "-" "-" "-" "fail-run-or-compare" >> "$RESULTS"
        continue
    fi

    base_median="$(median "${base_times[@]}")"
    mpi_median="$(median "${mpi_times[@]}")"
    ratio="$(ratio_of "$mpi_median" "$base_median")"
    status="ok"
    if is_slow_ratio "$ratio" "$SLOW_RATIO"; then
        status="slow>=${SLOW_RATIO}x"
    fi

    echo "  [4/4] result: baseline=${base_median}s mpi=${mpi_median}s ratio=${ratio}x status=${status}"
    printf '%s\t%s\t%s\t%s\t%s\n' "$test_name" "$base_median" "$mpi_median" "$ratio" "$status" >> "$RESULTS"
done

echo
echo "Results: $RESULTS"
column -t -s $'\t' "$RESULTS" 2>/dev/null || cat "$RESULTS"
