#!/usr/bin/env bash
# Benchmark: DACPP-translated code vs Standard SYCL reference
# Runs each program multiple times, reports average wall-clock time.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
source "$SCRIPT_DIR/env.sh"

BENCH_DIR="/Volumes/QUQ/working/bench_tmp"
WARMUP_RUNS=2
BENCH_RUNS=5
TIMEOUT_SEC=300

# Test cases that have both large_dac.cpp and large_StandardSycl.cpp
BENCH_TESTS=(
    "stencil1.0"
    "waveEquation1.0"
    "matMul1.0"
    "jacobi1.0"
    "MDP1.0"
    "DFT1.0"
    "FOuLa1.0"
    "decay1.0"
    "gradientSum"
    "liuliang1.0"
    "mandel1.0"
    "oddeven0.1"
)

# Allow passing test names as arguments
if [[ $# -gt 0 ]]; then
    BENCH_TESTS=("$@")
fi

rm -rf "$BENCH_DIR"
mkdir -p "$BENCH_DIR"

# ── helpers ──────────────────────────────────────────────

adapt_standard_reference() {
    local input_file="$1"
    local output_file="$2"
    python3 - "$input_file" "$output_file" <<'PY'
from pathlib import Path
import re
import sys

src = Path(sys.argv[1]).read_text()

if "#include <CL/sycl.hpp>" in src and "namespace sycl = cl::sycl;" not in src:
    if "using namespace sycl;" in src:
        src = src.replace(
            "using namespace sycl;",
            "namespace sycl = cl::sycl;\nusing namespace sycl;",
            1,
        )
    else:
        src = src.replace(
            "#include <CL/sycl.hpp>",
            "#include <CL/sycl.hpp>\nnamespace sycl = cl::sycl;",
            1,
        )

src = src.replace("gpu_selector_v", "default_selector_v")
src = re.sub(r"(?<![\w:])queue\s+q\b", "sycl::queue q", src)

Path(sys.argv[2]).write_text(src)
PY
}

# Returns median time in seconds (float)
run_timed() {
    local work_dir="$1"
    local binary="$2"
    local runs="$3"
    local results=()

    for _ in $(seq 1 "$runs"); do
        local elapsed
        elapsed=$(python3 - "$work_dir" "$binary" "$TIMEOUT_SEC" <<'PY'
import os, subprocess, sys, time

work_dir, binary, timeout_s = sys.argv[1:]
env = os.environ.copy()
acpp_root = env.get("ACPP_ROOT", "")
dyld = env.get("DYLD_LIBRARY_PATH", "")
if acpp_root:
    lib_path = os.path.join(acpp_root, "lib")
    env["DYLD_LIBRARY_PATH"] = lib_path if not dyld else f"{lib_path}:{dyld}"

start = time.perf_counter()
try:
    subprocess.run(
        [binary], cwd=work_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env=env, timeout=int(timeout_s), check=False,
    )
except subprocess.TimeoutExpired:
    print("TIMEOUT")
    sys.exit(0)
elapsed = time.perf_counter() - start
print(f"{elapsed:.6f}")
PY
        )
        if [[ "$elapsed" == "TIMEOUT" ]]; then
            echo "TIMEOUT"
            return
        fi
        results+=("$elapsed")
    done

    # Compute median
    python3 - "${results[@]}" <<'PY'
import sys
vals = sorted(float(x) for x in sys.argv[1:])
n = len(vals)
if n % 2 == 1:
    print(f"{vals[n//2]:.6f}")
else:
    print(f"{(vals[n//2 - 1] + vals[n//2]) / 2:.6f}")
PY
}

# ── main loop ────────────────────────────────────────────

printf "%-24s  %12s  %12s  %8s\n" "Test" "StandardSycl" "DACPP" "Ratio"
echo "-----------------------------------------------------------------------"

TOTAL=0
PASS=0
FAIL=0

for test_name in "${BENCH_TESTS[@]}"; do
    TOTAL=$((TOTAL + 1))
    source_dir="$SCRIPT_DIR/tests/$test_name"
    if [[ ! -d "$source_dir" ]]; then
        printf "%-24s  [SKIP: not found]\n" "$test_name"
        continue
    fi

    work_dir="$BENCH_DIR/$test_name"
    mkdir -p "$work_dir"
    cp -R "$source_dir"/. "$work_dir"/

    # ── 1. Find source files ──
    large_dac="$(find "$work_dir" -maxdepth 1 -name "*.large_dac.cpp" | sort | head -1)"
    large_std="$(find "$work_dir" -maxdepth 1 -name "*.large_StandardSycl.cpp" | sort | head -1)"

    if [[ -z "$large_dac" || -z "$large_std" ]]; then
        printf "%-24s  [SKIP: missing large files]\n" "$test_name"
        continue
    fi

    # ── 2. Translate DACPP ──
    if ! dacpp "$large_dac" --mode=buffer > "$work_dir/translate.log" 2>&1; then
        printf "%-24s  [FAIL: translate]\n" "$test_name"
        FAIL=$((FAIL + 1))
        continue
    fi

    generated_cpp="${large_dac%.cpp}_sycl_buffer.cpp"
    if [[ ! -f "$generated_cpp" ]]; then
        printf "%-24s  [FAIL: no generated file]\n" "$test_name"
        FAIL=$((FAIL + 1))
        continue
    fi

    # ── 3. Compile both ──
    std_bin="$work_dir/std_bin"
    dacpp_bin="$work_dir/dacpp_bin"

    adapt_standard_reference "$large_std" "$work_dir/adapted_std.cpp"

    if ! acpp-compile "$work_dir/adapted_std.cpp" "$std_bin" > "$work_dir/std_compile.log" 2>&1; then
        printf "%-24s  [FAIL: std compile]\n" "$test_name"
        cat "$work_dir/std_compile.log" | head -5
        FAIL=$((FAIL + 1))
        continue
    fi

    if ! acpp-compile "$generated_cpp" "$dacpp_bin" > "$work_dir/dacpp_compile.log" 2>&1; then
        printf "%-24s  [FAIL: dacpp compile]\n" "$test_name"
        cat "$work_dir/dacpp_compile.log" | head -5
        FAIL=$((FAIL + 1))
        continue
    fi

    # ── 4. Warmup ──
    run_timed "$work_dir" "$std_bin" "$WARMUP_RUNS" > /dev/null 2>&1
    run_timed "$work_dir" "$dacpp_bin" "$WARMUP_RUNS" > /dev/null 2>&1

    # ── 5. Benchmark ──
    std_time=$(run_timed "$work_dir" "$std_bin" "$BENCH_RUNS")
    dacpp_time=$(run_timed "$work_dir" "$dacpp_bin" "$BENCH_RUNS")

    if [[ "$std_time" == "TIMEOUT" || "$dacpp_time" == "TIMEOUT" ]]; then
        printf "%-24s  [TIMEOUT]\n" "$test_name"
        FAIL=$((FAIL + 1))
        continue
    fi

    # ── 6. Compute ratio ──
    ratio=$(python3 - "$std_time" "$dacpp_time" <<'PY'
import sys
std_t = float(sys.argv[1])
dacpp_t = float(sys.argv[2])
if std_t > 0:
    r = dacpp_t / std_t
    print(f"{r:.3f}")
else:
    print("N/A")
PY
    )

    printf "%-24s  %10.4f s  %10.4f s  %7s\n" "$test_name" "$std_time" "$dacpp_time" "${ratio}x"
    PASS=$((PASS + 1))
done

echo "-----------------------------------------------------------------------"
echo "Done: $PASS / $TOTAL benchmarked ($FAIL failed)"
