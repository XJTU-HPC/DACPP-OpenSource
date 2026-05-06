#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
source "$SCRIPT_DIR/env.sh"

TESTS_DIR="$SCRIPT_DIR/tests"
TMP_DIR="${APP_BENCH_TMP_DIR:-/Volumes/QUQ/working/mpi_tmp/app_bench}"
MPI_RANKS="${APP_BENCH_RANKS:-4}"
RUNS="${APP_BENCH_RUNS:-3}"
SLOW_RATIO="${APP_BENCH_SLOW_RATIO:-10}"
TARGET_SECONDS="${APP_BENCH_TARGET_SECONDS:-7}"
MAX_BATCH="${APP_BENCH_MAX_BATCH:-1000}"
USE_LARGE_CASES="${APP_BENCH_USE_LARGE:-0}"
RUN_TIMEOUT_SECONDS="${APP_BENCH_RUN_TIMEOUT_SECONDS:-60}"

APP_TESTS=(
    "DFT1.0"
    "FOuLa1.0"
    "MDP1.0"
    "decay1.0"
    "gradientSum"
    "imageAdjustment1.0"
    "jacobi1.0"
    "liuliang1.0"
    "mandel1.0"
    "matMul1.0"
    "oddeven0.1"
    "stencil1.0"
    "vectorAddCombo"
    "waveEquation1.0"
)

if [[ $# -gt 0 ]]; then
    APP_TESTS=("$@")
fi

rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"

pick_dac_source() {
    local source_dir="$1"
    local preferred_src fallback_src

    preferred_src="$(find "$source_dir" -maxdepth 1 -type f -name "*.dac.cpp" \
        ! -name "*.mpi.dac.cpp" \
        ! -name "*.retranslated.dac.cpp" \
        ! -name "*.large_dac.cpp" \
        | sort | head -n 1)"
    fallback_src="$(find "$source_dir" -maxdepth 1 -type f -name "*.large_dac.cpp" \
        | sort | head -n 1)"

    if [[ "$USE_LARGE_CASES" == "1" ]]; then
        if [[ -n "$fallback_src" ]]; then
            printf '%s\n' "$fallback_src"
            return
        fi
        printf '%s\n' "$preferred_src"
        return
    fi

    if [[ -n "$preferred_src" ]]; then
        printf '%s\n' "$preferred_src"
        return
    fi
    printf '%s\n' "$fallback_src"
}

find_mpi_standard_sycl() {
    local source_dir="$1"
    find "$source_dir" -maxdepth 1 -type f -name "*.MPI_StandardSycl.cpp" \
        | sort | head -n 1
}

generated_cpp_path_for() {
    local source_file="$1"
    local base_name
    base_name="$(basename "$source_file")"
    if [[ "$base_name" == *.large_dac.cpp ]]; then
        printf '%s/%s\n' "$(dirname "$source_file")" "${base_name%.cpp}_sycl_buffer.cpp"
        return
    fi
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
import os

log = sys.argv[1]
cmd = sys.argv[2:]
timeout_s = float(os.environ.get("APP_BENCH_RUN_TIMEOUT_SECONDS", "0") or 0)
start = time.perf_counter()
with open(log, "wb") as out:
    try:
        proc = subprocess.run(
            cmd,
            stdout=out,
            stderr=subprocess.STDOUT,
            timeout=timeout_s if timeout_s > 0 else None,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start
        out.write(f"[TIMEOUT] command exceeded {timeout_s:.1f} seconds\n".encode())
        print(f"{elapsed:.6f}")
        sys.exit(124)
elapsed = time.perf_counter() - start
print(f"{elapsed:.6f}")
sys.exit(proc.returncode)
PY
}

run_batched_timed() {
    local log="$1"
    local count="$2"
    shift 2
    python3 - "$log" "$count" "$@" <<'PY'
import os
import subprocess
import sys
import time

log = sys.argv[1]
count = int(sys.argv[2])
cmd = sys.argv[3:]
capture = os.environ.get("APP_BENCH_CAPTURE_BATCH_OUTPUT", "0") == "1"
timeout_s = float(os.environ.get("APP_BENCH_RUN_TIMEOUT_SECONDS", "0") or 0)

start = time.perf_counter()
with open(log, "wb") as out:
    for idx in range(count):
        if capture:
            out.write(f"===== batch iteration {idx + 1}/{count} =====\n".encode())
            try:
                proc = subprocess.run(
                    cmd,
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    timeout=timeout_s if timeout_s > 0 else None,
                )
            except subprocess.TimeoutExpired:
                elapsed = time.perf_counter() - start
                out.write(
                    f"[TIMEOUT] command exceeded {timeout_s:.1f} seconds at batch iteration {idx + 1}/{count}\n".encode()
                )
                print(f"{elapsed:.6f}")
                sys.exit(124)
        else:
            try:
                proc = subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=timeout_s if timeout_s > 0 else None,
                )
            except subprocess.TimeoutExpired:
                elapsed = time.perf_counter() - start
                out.write(
                    f"[TIMEOUT] command exceeded {timeout_s:.1f} seconds at batch iteration {idx + 1}/{count}\n".encode()
                )
                print(f"{elapsed:.6f}")
                sys.exit(124)
        if proc.returncode != 0:
            out.write(
                f"command failed at batch iteration {idx + 1}/{count}: {proc.returncode}\n".encode()
            )
            elapsed = time.perf_counter() - start
            print(f"{elapsed:.6f}")
            sys.exit(proc.returncode)

elapsed = time.perf_counter() - start
print(f"{elapsed:.6f}")
PY
}

batch_count_for() {
    python3 - "$1" "$TARGET_SECONDS" "$MAX_BATCH" <<'PY'
import math
import sys

single = max(float(sys.argv[1]), 1e-6)
target = max(float(sys.argv[2]), 0.001)
max_batch = max(int(sys.argv[3]), 1)
count = max(1, math.ceil(target / single))
print(min(count, max_batch))
PY
}

per_run_time() {
    python3 - "$1" "$2" <<'PY'
import sys
elapsed = float(sys.argv[1])
count = max(int(sys.argv[2]), 1)
print(f"{elapsed / count:.6f}")
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
printf 'test\tbaseline_median_s\tmpi_median_s\tratio\tbaseline_batch\tmpi_batch\tref_median_s\tref_ratio\tref_batch\tstatus\n' > "$RESULTS"

export DYLD_LIBRARY_PATH="$ACPP_ROOT/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"

echo "Benchmark config: target=${TARGET_SECONDS}s runs=${RUNS} np=${MPI_RANKS} use_large=${USE_LARGE_CASES} max_batch=${MAX_BATCH} timeout=${RUN_TIMEOUT_SECONDS}s"
echo "Batch timing suppresses program output after the single-run correctness check."
export APP_BENCH_RUN_TIMEOUT_SECONDS="$RUN_TIMEOUT_SECONDS"

for test_name in "${APP_TESTS[@]}"; do
    echo "========================================================"
    echo "  Bench: $test_name"
    echo "========================================================"

    work_dir="$TMP_DIR/$test_name"
    mkdir -p "$work_dir"

    # ---- Find sources ----
    dac_file="$(pick_dac_source "$TESTS_DIR/$test_name")"
    ref_file="$(find_mpi_standard_sycl "$TESTS_DIR/$test_name")"

    if [[ -z "$dac_file" ]]; then
        echo "[SKIP] No .dac.cpp found for $test_name"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$test_name" "-" "-" "-" "-" "-" "-" "-" "-" "skip-no-source" >> "$RESULTS"
        continue
    fi
    echo "  source: $(basename "$dac_file")"

    # ---- Step 1: Build baseline ----
    base_dac="$work_dir/$(basename "$dac_file")"
    cp "$dac_file" "$base_dac"
    base_sycl="$(generated_cpp_path_for "$base_dac")"
    base_bin="$work_dir/base_bin"

    echo "  [1/5] build baseline"
    if ! run_in_env "$work_dir/build_base.log" "dacpp '$base_dac' --mode=buffer && acpp-compile '$base_sycl' '$base_bin'"; then
        echo "[FAIL] baseline build failed"
        head -n 20 "$work_dir/build_base.log"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$test_name" "-" "-" "-" "-" "-" "-" "-" "-" "fail-build-baseline" >> "$RESULTS"
        continue
    fi

    # ---- Step 2: Build MPI translated ----
    mpi_dac="$(mpi_dac_path_for "$base_dac")"
    cp "$dac_file" "$mpi_dac"
    mpi_sycl="$(generated_cpp_path_for "$mpi_dac")"
    mpi_bin="$work_dir/mpi_bin"

    echo "  [2/5] build MPI translated"
    if ! run_in_env "$work_dir/build_mpi.log" "dacpp '$mpi_dac' --mode=buffer --mpi && acpp-compile '$mpi_sycl' '$mpi_bin'"; then
        echo "[FAIL] MPI build failed"
        head -n 20 "$work_dir/build_mpi.log"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$test_name" "-" "-" "-" "-" "-" "-" "-" "-" "fail-build-mpi" >> "$RESULTS"
        continue
    fi
    if grep -Fq "<->" "$mpi_sycl"; then
        echo "[FAIL] generated MPI SYCL still contains <->"
        grep -Fn "<->" "$mpi_sycl" | head -n 5
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$test_name" "-" "-" "-" "-" "-" "-" "-" "-" "fail-residual-dac-expr" >> "$RESULTS"
        continue
    fi

    # ---- Step 3: Build MPI_StandardSycl reference (if exists) ----
    ref_bin=""
    if [[ -n "$ref_file" ]]; then
        ref_src="$work_dir/$(basename "$ref_file")"
        cp "$ref_file" "$ref_src"
        ref_bin="$work_dir/ref_bin"

        echo "  [3/5] build MPI_StandardSycl reference"
        if ! run_in_env "$work_dir/build_ref.log" "acpp-compile '$ref_src' '$ref_bin'"; then
            echo "[WARN] MPI_StandardSycl build failed, skipping reference"
            ref_bin=""
        fi
    else
        echo "  [3/5] no MPI_StandardSycl reference, skip"
    fi

    # ---- Step 4: Single-run correctness check and calibrated benchmark ----
    echo "  [4/5] run correctness check, then calibrated samples: runs=$RUNS np=$MPI_RANKS target=${TARGET_SECONDS}s"
    base_times=()
    mpi_times=()
    ref_times=()
    base_batch_count=""
    mpi_batch_count=""
    ref_batch_count=""
    compare_ok=1

    base_check_log="$work_dir/base_check.out"
    mpi_check_log="$work_dir/mpi_check.out"
    base_check_time="$(run_timed "$base_check_log" "$base_bin")" || {
        echo "[FAIL] baseline correctness run failed"
        head -n 20 "$base_check_log"
        compare_ok=0
    }
    if [[ "$compare_ok" == "1" ]]; then
        mpi_check_time="$(run_timed "$mpi_check_log" mpirun -np "$MPI_RANKS" "$mpi_bin")" || {
            echo "[FAIL] MPI correctness run failed"
            head -n 20 "$mpi_check_log"
            compare_ok=0
        }
    fi
    if [[ "$compare_ok" == "1" ]]; then
        clean_output "$base_check_log" "$work_dir/base_check.clean"
        clean_output "$mpi_check_log" "$work_dir/mpi_check.clean"
        if ! diff -q "$work_dir/base_check.clean" "$work_dir/mpi_check.clean" > /dev/null 2>&1; then
            echo "[FAIL] output mismatch on correctness check"
            diff "$work_dir/base_check.clean" "$work_dir/mpi_check.clean" | head -n 30
            compare_ok=0
        fi
    fi
    if [[ "$compare_ok" != "1" ]]; then
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$test_name" "-" "-" "-" "-" "-" "-" "-" "-" "fail-run-or-compare" >> "$RESULTS"
        continue
    fi

    base_probe_log="$work_dir/base_probe.log"
    mpi_probe_log="$work_dir/mpi_probe.log"
    base_probe_time="$(run_batched_timed "$base_probe_log" 1 "$base_bin")" || {
        echo "[FAIL] baseline probe run failed"
        head -n 20 "$base_probe_log"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$test_name" "-" "-" "-" "-" "-" "-" "-" "-" "fail-baseline-probe" >> "$RESULTS"
        continue
    }
    mpi_probe_time="$(run_batched_timed "$mpi_probe_log" 1 mpirun -np "$MPI_RANKS" "$mpi_bin")" || {
        echo "[FAIL] MPI probe run failed"
        head -n 20 "$mpi_probe_log"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$test_name" "-" "-" "-" "-" "-" "-" "-" "-" "fail-mpi-probe" >> "$RESULTS"
        continue
    }

    base_batch_count="$(batch_count_for "$base_probe_time")"
    mpi_batch_count="$(batch_count_for "$mpi_probe_time")"
    if [[ -n "$ref_bin" ]]; then
        ref_check_log="$work_dir/ref_check.out"
        ref_check_time="$(run_timed "$ref_check_log" mpirun -np "$MPI_RANKS" "$ref_bin")" || {
            echo "[WARN] reference correctness run failed, skipping reference"
            ref_bin=""
        }
        if [[ -n "$ref_bin" ]]; then
            ref_probe_log="$work_dir/ref_probe.log"
            ref_probe_time="$(run_batched_timed "$ref_probe_log" 1 mpirun -np "$MPI_RANKS" "$ref_bin")" || {
                echo "[WARN] reference probe run failed, skipping reference"
                ref_bin=""
            }
        fi
        if [[ -n "$ref_bin" ]]; then
            ref_batch_count="$(batch_count_for "$ref_probe_time")"
        fi
    fi

    echo "    correctness: baseline=${base_check_time}s mpi=${mpi_check_time}s"
    echo "    calibration: baseline_probe=${base_probe_time}s batch=${base_batch_count}; mpi_probe=${mpi_probe_time}s batch=${mpi_batch_count}${ref_batch_count:+; ref_probe=${ref_probe_time}s batch=${ref_batch_count}}"

    for run in $(seq 1 "$RUNS"); do
        base_log="$work_dir/base_batch_${run}.log"
        mpi_log="$work_dir/mpi_batch_${run}.log"

        base_elapsed="$(run_batched_timed "$base_log" "$base_batch_count" "$base_bin")" || {
            echo "[FAIL] baseline batch run $run failed"
            head -n 20 "$base_log"
            compare_ok=0
            break
        }
        mpi_elapsed="$(run_batched_timed "$mpi_log" "$mpi_batch_count" mpirun -np "$MPI_RANKS" "$mpi_bin")" || {
            echo "[FAIL] MPI batch run $run failed"
            head -n 20 "$mpi_log"
            compare_ok=0
            break
        }

        base_time="$(per_run_time "$base_elapsed" "$base_batch_count")"
        mpi_time="$(per_run_time "$mpi_elapsed" "$mpi_batch_count")"

        base_times+=("$base_time")
        mpi_times+=("$mpi_time")

        if [[ -n "$ref_bin" ]]; then
            ref_log="$work_dir/ref_batch_${run}.log"
            ref_elapsed="$(run_batched_timed "$ref_log" "$ref_batch_count" mpirun -np "$MPI_RANKS" "$ref_bin")" || {
                echo "[WARN] reference batch run $run failed"
                ref_bin=""
                break
            }
            ref_time="$(per_run_time "$ref_elapsed" "$ref_batch_count")"
            ref_times+=("$ref_time")
        fi

        echo "    run $run: baseline=${base_time}s (${base_elapsed}s/${base_batch_count}) mpi=${mpi_time}s (${mpi_elapsed}s/${mpi_batch_count})${ref_times[run-1]:+ ref=${ref_times[run-1]}s}"
    done

    if [[ "$compare_ok" != "1" ]]; then
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$test_name" "-" "-" "-" "-" "-" "-" "-" "-" "fail-run-or-compare" >> "$RESULTS"
        continue
    fi

    # ---- Step 5: Summarize ----
    base_median="$(median "${base_times[@]}")"
    mpi_median="$(median "${mpi_times[@]}")"
    ratio="$(ratio_of "$mpi_median" "$base_median")"
    status="ok"

    ref_median_str="-"
    ref_ratio_str="-"
    ref_batch_str="-"
    if [[ ${#ref_times[@]} -gt 0 ]]; then
        ref_median="$(median "${ref_times[@]}")"
        ref_ratio="$(ratio_of "$mpi_median" "$ref_median")"
        ref_median_str="$ref_median"
        ref_ratio_str="${ref_ratio}x"
        ref_batch_str="$ref_batch_count"
    fi

    if is_slow_ratio "$ratio" "$SLOW_RATIO"; then
        status="slow>=${SLOW_RATIO}x"
    fi

    if [[ "$ref_median_str" != "-" ]]; then
        echo "  [5/5] result: baseline=${base_median}s mpi=${mpi_median}s ratio=${ratio}x ref=${ref_median_str}s ref_ratio=${ref_ratio_str} status=${status}"
    else
        echo "  [5/5] result: baseline=${base_median}s mpi=${mpi_median}s ratio=${ratio}x status=${status}"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$test_name" "$base_median" "$mpi_median" "$ratio" "$base_batch_count" "$mpi_batch_count" "$ref_median_str" "$ref_ratio_str" "$ref_batch_str" "$status" >> "$RESULTS"
done

echo
echo "Results: $RESULTS"
column -t -s $'\t' "$RESULTS" 2>/dev/null || cat "$RESULTS"
