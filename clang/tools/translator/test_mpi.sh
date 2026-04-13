#!/usr/bin/env bash

# Setup environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
source "$SCRIPT_DIR/env.sh"

TESTS_DIR="$SCRIPT_DIR/tests"
TMP_DIR="/tmp/dacpp_mpi_tests"

rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"

NON_STENCIL_TESTS=(
    "matMul1.0"
    "FOuLa1.0"
    "decay1.0"
    "DFT1.0"
    "liuliang1.0"
    "MDP1.0"
    "mandel1.0"
    "imageAdjustment1.0"
    "vectorAddCombo"
    "gradientSum"
)

# ── 过滤 AdaptiveCpp 运行时警告 ──────────────────────────────────────────────
clean_output() {
    grep -v "AdaptiveCpp Warning" "$1" \
    | grep -v "https://github.com/AdaptiveCpp" \
    > "${1}.clean"
}

run_in_env() {
    local log="$1"; shift
    bash -lc "source '$SCRIPT_DIR/env.sh' && $*" > "$log" 2>&1
}

TOTAL=0; PASSED=0; FAILED=0; SKIPPED=0

for test_name in "${NON_STENCIL_TESTS[@]}"; do
    echo "========================================================"
    echo "  Test: $test_name"
    echo "========================================================"
    TOTAL=$((TOTAL + 1))

    dac_file=$(find "$TESTS_DIR/$test_name" -maxdepth 1 -type f -name "*.dac.cpp" | head -n 1)
    if [[ -z "$dac_file" ]]; then
        echo "[WARN] No .dac.cpp found, skipping."
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    work_dir="$TMP_DIR/$test_name"
    mkdir -p "$work_dir"

    # Copy files
    base_dac="$work_dir/$(basename "$dac_file")"
    cp "$dac_file" "$base_dac"
    base_sycl="${base_dac%.dac.cpp}.dac_sycl_buffer.cpp"
    base_bin="$work_dir/base_bin"

    # Step 1: Translate Baseline (Non-MPI, mode=buffer)
    echo "  [Step 1] Translate --mode=buffer (baseline, no MPI) and compile"
    if ! run_in_env "$work_dir/step1.log" "dacpp '$base_dac' --mode=buffer && acpp-compile '$base_sycl' '$base_bin'"; then
        echo "[FAIL] Baseline translate/compile failed."
        head -n 20 "$work_dir/step1.log"
        FAILED=$((FAILED + 1))
        continue
    fi

    # Run Baseline
    DYLD_LIBRARY_PATH="$ACPP_ROOT/lib" "$base_bin" > "$work_dir/base.out" 2>&1
    if [[ $? -ne 0 ]]; then
        echo "[FAIL] Baseline execution failed."
        head -n 20 "$work_dir/base.out"
        FAILED=$((FAILED + 1))
        continue
    fi
    clean_output "$work_dir/base.out"

    # Step 2: Translate MPI
    echo "  [Step 2] Translate --mode=usm --mpi and compile"
    mpi_dac="$work_dir/$(basename "${dac_file%.dac.cpp}").mpi.dac.cpp"
    cp "$dac_file" "$mpi_dac"
    mpi_sycl="${mpi_dac%.dac.cpp}.dac_sycl_usm.cpp"
    mpi_bin="$work_dir/mpi_bin"

    if ! run_in_env "$work_dir/step2.log" "dacpp '$mpi_dac' --mode=usm --mpi && acpp-compile '$mpi_sycl' '$mpi_bin'"; then
        echo "[FAIL] MPI translate/compile failed."
        head -n 20 "$work_dir/step2.log"
        FAILED=$((FAILED + 1))
        continue
    fi

    # Step 3: Run MPI and Compare
    echo "  [Step 3] Run MPI (mpirun -np 2) and compare"
    DYLD_LIBRARY_PATH="$ACPP_ROOT/lib" mpirun -np 2 "$mpi_bin" > "$work_dir/mpi.out" 2>&1
    if [[ $? -ne 0 ]]; then
        echo "[FAIL] MPI execution failed."
        head -n 20 "$work_dir/mpi.out"
        FAILED=$((FAILED + 1))
        continue
    fi
    clean_output "$work_dir/mpi.out"

    if diff -q "$work_dir/base.out.clean" "$work_dir/mpi.out.clean" > /dev/null 2>&1; then
        echo "[PASS] $test_name: MPI output matches baseline."
        PASSED=$((PASSED + 1))
    else
        echo "[FAIL] $test_name: MPI output differs from baseline!"
        echo "  --- baseline (first 15 lines) ---"
        head -n 15 "$work_dir/base.out.clean"
        echo "  --- mpi (2 ranks, first 15 lines) ---"
        head -n 15 "$work_dir/mpi.out.clean"
        echo "  --- diff (first 30 lines) ---"
        diff "$work_dir/base.out.clean" "$work_dir/mpi.out.clean" | head -n 30
        FAILED=$((FAILED + 1))
    fi
done

echo
echo "========================================================"
echo "  Summary: $TOTAL tests | $PASSED passed | $FAILED failed | $SKIPPED skipped"
echo "========================================================"
