#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
source "$SCRIPT_DIR/env.sh"

TESTS_DIR="$SCRIPT_DIR/tests"
TMP_DIR="/Volumes/QUQ/working/local_tmp"

LOCAL_TESTS=(
    "matMul1.0"
    "FOuLa1.0"
    "decay1.0"
    "DFT1.0"
    "liuliang1.0"
    "mandel1.0"
    "gradientSum"
    "jacobi1.0"
)

# Extra cases that are currently slower, lack a local reference, or expose
# known output differences in the current translator. Enable them explicitly
# with:
#   INCLUDE_EXTENDED_LOCAL_TESTS=1 bash test_local.sh
# or pass test names directly:
#   LOCAL_TEST_TIMEOUT_SEC=300 bash test_local.sh MDP1.0
EXTENDED_LOCAL_TESTS=(
    "imageAdjustment1.0"
    "oddeven0.1"
    "stencil1.0"
    "waveEquation1.0"
    "MDP1.0"
    "vectorAddCombo"
)

if [[ "${INCLUDE_EXTENDED_LOCAL_TESTS:-0}" == "1" ]]; then
    LOCAL_TESTS+=("${EXTENDED_LOCAL_TESTS[@]}")
fi

# Default to a fast local smoke suite.
LOCAL_TEST_TIMEOUT_SEC="${LOCAL_TEST_TIMEOUT_SEC:-120}"

if [[ $# -gt 0 ]]; then
    LOCAL_TESTS=("$@")
fi

rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"

normalize_output() {
    local input_file="$1"
    local output_file="$2"
    python3 - "$input_file" "$output_file" <<'PY'
from pathlib import Path
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])

lines = []
for line in src.read_text(errors="ignore").splitlines():
    if "AdaptiveCpp Warning" in line:
        continue
    if "https://github.com/AdaptiveCpp" in line:
        continue
    lines.append(line.rstrip())

text = "\n".join(lines)
if text:
    text += "\n"
dst.write_text(text)
PY
}

pick_dac_source() {
    find "$1" -maxdepth 1 -type f -name "*.dac.cpp" \
        ! -name "*.mpi.dac.cpp" \
        ! -name "*.retranslated.dac.cpp" \
        ! -name "*.large_dac.cpp" \
        | sort | head -n 1
}

pick_standard_reference() {
    find "$1" -maxdepth 1 -type f -name "*.StandardSycl.cpp" \
        ! -name "*.MPI_StandardSycl.cpp" \
        ! -name "*.large_StandardSycl.cpp" \
        | sort | head -n 1
}

pick_serial_reference() {
    find "$1" -maxdepth 1 -type f -name "*.serial.cpp" \
        ! -name "*.large_serial.cpp" \
        | sort | head -n 1
}

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

compile_host_reference() {
    local source_file="$1"
    local output_bin="$2"
    local host_cxx="${CXX:-c++}"
    "$host_cxx" -O2 -std=c++17 \
        "$source_file" \
        "${COMMON_CXX_FLAGS[@]}" \
        "${INCLUDE_DIRS[@]/#/-I}" \
        -o "$output_bin"
}

run_binary_in_dir() {
    local work_dir="$1"
    local binary_path="$2"
    local output_file="$3"
    python3 - "$work_dir" "$binary_path" "$output_file" "$LOCAL_TEST_TIMEOUT_SEC" <<'PY'
import os
import subprocess
import sys

work_dir, binary_path, output_file, timeout_s = sys.argv[1:]
env = os.environ.copy()
acpp_root = env.get("ACPP_ROOT", "")
dyld = env.get("DYLD_LIBRARY_PATH", "")
if acpp_root:
    lib_path = os.path.join(acpp_root, "lib")
    env["DYLD_LIBRARY_PATH"] = lib_path if not dyld else f"{lib_path}:{dyld}"

with open(output_file, "w") as fp:
    try:
        completed = subprocess.run(
            [binary_path],
            cwd=work_dir,
            stdout=fp,
            stderr=subprocess.STDOUT,
            env=env,
            timeout=int(timeout_s),
            check=False,
        )
    except subprocess.TimeoutExpired:
        fp.write(f"[TIMEOUT] Execution exceeded {timeout_s} seconds.\n")
        sys.exit(124)

sys.exit(completed.returncode)
PY
}

REFERENCE_KIND=""
REFERENCE_PATH=""

prepare_reference() {
    local source_dir="$1"
    local work_dir="$2"

    local standard_src
    standard_src="$(pick_standard_reference "$source_dir")"
    if [[ -n "$standard_src" ]]; then
        local adapted_src="$work_dir/reference.StandardSycl.local.cpp"
        local reference_bin="$work_dir/reference_std_bin"
        adapt_standard_reference "$standard_src" "$adapted_src"
        if acpp-compile "$adapted_src" "$reference_bin" > "$work_dir/reference_std_compile.log" 2>&1; then
            if run_binary_in_dir "$work_dir" "$reference_bin" "$work_dir/reference_std.out"; then
                normalize_output "$work_dir/reference_std.out" "$work_dir/reference.clean"
                REFERENCE_KIND="StandardSycl"
                REFERENCE_PATH="$work_dir/reference.clean"
                return 0
            fi
        fi
    fi

    local serial_src
    serial_src="$(pick_serial_reference "$source_dir")"
    if [[ -n "$serial_src" ]]; then
        local reference_bin="$work_dir/reference_serial_bin"
        if compile_host_reference "$serial_src" "$reference_bin" > "$work_dir/reference_serial_compile.log" 2>&1; then
            if run_binary_in_dir "$work_dir" "$reference_bin" "$work_dir/reference_serial.out"; then
                normalize_output "$work_dir/reference_serial.out" "$work_dir/reference.clean"
                REFERENCE_KIND="serial"
                REFERENCE_PATH="$work_dir/reference.clean"
                return 0
            fi
        fi
    fi

    if [[ -f "$source_dir/result.out" ]]; then
        normalize_output "$source_dir/result.out" "$work_dir/reference.clean"
        REFERENCE_KIND="result.out"
        REFERENCE_PATH="$work_dir/reference.clean"
        return 0
    fi

    return 1
}

TOTAL=0
PASSED=0
FAILED=0
SKIPPED=0

for test_name in "${LOCAL_TESTS[@]}"; do
    echo "========================================================"
    echo "  Test: $test_name"
    echo "========================================================"
    TOTAL=$((TOTAL + 1))

    source_dir="$TESTS_DIR/$test_name"
    if [[ ! -d "$source_dir" ]]; then
        echo "[SKIP] Test directory not found."
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    work_dir="$TMP_DIR/$test_name"
    mkdir -p "$work_dir"
    cp -R "$source_dir"/. "$work_dir"/

    dac_file="$(pick_dac_source "$work_dir")"
    if [[ -z "$dac_file" ]]; then
        echo "[SKIP] No primary .dac.cpp found."
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    generated_cpp="${dac_file%.dac.cpp}.dac_sycl_buffer.cpp"
    generated_bin="$work_dir/generated_bin"

    echo "  [Step 1] Translate --mode=buffer"
    if ! dacpp "$dac_file" --mode=buffer > "$work_dir/translate.log" 2>&1; then
        echo "[FAIL] Translation failed."
        head -n 20 "$work_dir/translate.log"
        FAILED=$((FAILED + 1))
        continue
    fi

    if [[ ! -f "$generated_cpp" ]]; then
        echo "[FAIL] Generated file not found: $(basename "$generated_cpp")"
        FAILED=$((FAILED + 1))
        continue
    fi

    echo "  [Step 2] Compile generated code"
    if ! acpp-compile "$generated_cpp" "$generated_bin" > "$work_dir/generated_compile.log" 2>&1; then
        echo "[FAIL] Generated code compilation failed."
        head -n 20 "$work_dir/generated_compile.log"
        FAILED=$((FAILED + 1))
        continue
    fi

    echo "  [Step 3] Run generated code"
    if ! run_binary_in_dir "$work_dir" "$generated_bin" "$work_dir/generated.out"; then
        echo "[FAIL] Generated program execution failed."
        head -n 20 "$work_dir/generated.out"
        FAILED=$((FAILED + 1))
        continue
    fi
    normalize_output "$work_dir/generated.out" "$work_dir/generated.clean"

    echo "  [Step 4] Prepare local reference"
    REFERENCE_KIND=""
    REFERENCE_PATH=""
    if ! prepare_reference "$source_dir" "$work_dir"; then
        echo "[SKIP] No runnable local reference found."
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo "  [Step 5] Compare against $REFERENCE_KIND"
    if diff -u "$REFERENCE_PATH" "$work_dir/generated.clean" > "$work_dir/diff.log" 2>&1; then
        echo "[PASS] $test_name: generated output matches $REFERENCE_KIND."
        PASSED=$((PASSED + 1))
    else
        echo "[FAIL] $test_name: generated output differs from $REFERENCE_KIND."
        echo "  --- reference (first 15 lines) ---"
        head -n 15 "$REFERENCE_PATH"
        echo "  --- generated (first 15 lines) ---"
        head -n 15 "$work_dir/generated.clean"
        echo "  --- diff (first 30 lines) ---"
        head -n 30 "$work_dir/diff.log"
        FAILED=$((FAILED + 1))
    fi
done

echo
echo "========================================================"
echo "  Summary: $TOTAL tests | $PASSED passed | $FAILED failed | $SKIPPED skipped"
echo "========================================================"

if [[ $FAILED -ne 0 ]]; then
    exit 1
fi
