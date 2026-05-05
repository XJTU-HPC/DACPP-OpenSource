#!/usr/bin/env bash

set -uo pipefail

# Setup environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
source "$SCRIPT_DIR/env.sh"

TESTS_DIR="$SCRIPT_DIR/tests"
TMP_DIR="/Volumes/QUQ/working/mpi_tmp"

rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"

MPI_TESTS=(
    "mpiDenseCoverSibling1.0"
    "matMul1.0"
    "FOuLa1.0"
    "decay1.0"
    "DFT1.0"
    "liuliang1.0"
    "MDP1.0"
    "jacobi1.0"
    "mandel1.0"
    "imageAdjustment1.0"
    "vectorAddCombo"
    "gradientSum"
    "mpiBroadcastRootOnlyCout"
    "mpiBroadcastTensor2Array"
    "mpiBroadcastUnknownFunction"
    "mpiBroadcastAliasRead"
    "mpiDistributedStencil1D"
    "mpiDistributedStencilNoBridge1D"
    "mpiDistributedStencilSteady1D"
    "mpiDistributedStencilAstRoute1D"
    "mpiDistributedStencilAstRouteFallback1D"
    "mpiDistributedStencil2DRowBlock"
    "mpiPhaseCWriteThenRead1D"
    "mpiPhaseCHalo1D"
    "mpiPhaseCHaloWide1D"
    "stencil1.0"
    "waveEquation1.0"
)

USE_LARGE_CASES=0
POSITIONAL_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --large)
            USE_LARGE_CASES=1
            ;;
        -h|--help)
            echo "Usage: bash test_mpi.sh [--large] [test_name ...]"
            exit 0
            ;;
        *)
            POSITIONAL_ARGS+=("$arg")
            ;;
    esac
done

if [[ ${#POSITIONAL_ARGS[@]} -gt 0 ]]; then
    MPI_TESTS=("${POSITIONAL_ARGS[@]}")
fi

pick_dac_source() {
    local source_dir="$1"
    local preferred_src fallback_src

    fallback_src="$(find "$source_dir" -maxdepth 1 -type f -name "*.large_dac.cpp" | sort | head -n 1)"
    preferred_src="$(find "$source_dir" -maxdepth 1 -type f -name "*.dac.cpp" \
        ! -name "*.mpi.dac.cpp" \
        ! -name "*.retranslated.dac.cpp" \
        ! -name "*.large_dac.cpp" \
        | sort | head -n 1)"

    if [[ "$USE_LARGE_CASES" == "1" ]]; then
        preferred_src="$fallback_src"
        fallback_src="$(find "$source_dir" -maxdepth 1 -type f -name "*.dac.cpp" \
            ! -name "*.mpi.dac.cpp" \
            ! -name "*.retranslated.dac.cpp" \
            ! -name "*.large_dac.cpp" \
            | sort | head -n 1)"
    fi

    if [[ -n "$preferred_src" ]]; then
        printf '%s\n' "$preferred_src"
        return
    fi

    if [[ -n "$fallback_src" ]]; then
        printf '%s\n' "$fallback_src"
    fi
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

# ── 过滤 AdaptiveCpp 运行时警告 ──────────────────────────────────────────────
clean_output() {
    grep -v "AdaptiveCpp Warning" "$1" \
    | grep -v "This application uses SYCL buffers" \
    | grep -v "SYCL2020 USM model" \
    | grep -v "AdaptiveCpp performance guide" \
    | grep -v "https://github.com/AdaptiveCpp" \
    | grep -v '^[[:space:]]*$' \
    > "${1}.clean"
}

run_in_env() {
    local log="$1"; shift
    bash -lc "source '$SCRIPT_DIR/env.sh' && $*" > "$log" 2>&1
}

check_mpi_expectations() {
    local expect_file="$1"
    local log_file="$2"
    local sycl_file="$3"
    local failed=0

    [[ -f "$expect_file" ]] || return 0

    while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
        local line kind pattern target_file
        line="${raw_line#"${raw_line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        [[ -z "$line" || "$line" == \#* ]] && continue

        kind="${line%%:*}"
        pattern="${line#*:}"
        case "$kind" in
            LOG_CONTAINS)
                target_file="$log_file"
                if ! grep -Fq "$pattern" "$target_file"; then
                    echo "[FAIL] Expected log to contain: $pattern"
                    failed=1
                fi
                ;;
            LOG_NOT_CONTAINS)
                target_file="$log_file"
                if grep -Fq "$pattern" "$target_file"; then
                    echo "[FAIL] Expected log not to contain: $pattern"
                    failed=1
                fi
                ;;
            SYCL_CONTAINS)
                target_file="$sycl_file"
                if ! grep -Fq "$pattern" "$target_file"; then
                    echo "[FAIL] Expected generated SYCL to contain: $pattern"
                    failed=1
                fi
                ;;
            SYCL_NOT_CONTAINS)
                target_file="$sycl_file"
                if grep -Fq "$pattern" "$target_file"; then
                    echo "[FAIL] Expected generated SYCL not to contain: $pattern"
                    failed=1
                fi
                ;;
            *)
                echo "[FAIL] Unknown expectation kind in $expect_file: $kind"
                failed=1
                ;;
        esac
    done < "$expect_file"

    return "$failed"
}

TOTAL=0; PASSED=0; FAILED=0; SKIPPED=0

for test_name in "${MPI_TESTS[@]}"; do
    echo "========================================================"
    echo "  Test: $test_name"
    echo "========================================================"
    TOTAL=$((TOTAL + 1))

    dac_file="$(pick_dac_source "$TESTS_DIR/$test_name")"
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
    base_sycl="$(generated_cpp_path_for "$base_dac")"
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
    echo "  [Step 2] Translate --mode=buffer --mpi and compile"
    mpi_dac="$(mpi_dac_path_for "$base_dac")"
    cp "$dac_file" "$mpi_dac"
    mpi_sycl="$(generated_cpp_path_for "$mpi_dac")"
    mpi_bin="$work_dir/mpi_bin"

    if ! run_in_env "$work_dir/step2.log" "dacpp '$mpi_dac' --mode=buffer --mpi && acpp-compile '$mpi_sycl' '$mpi_bin'"; then
        echo "[FAIL] MPI translate/compile failed."
        head -n 20 "$work_dir/step2.log"
        FAILED=$((FAILED + 1))
        continue
    fi

    expect_file="$TESTS_DIR/$test_name/mpi_expect.txt"
    if ! check_mpi_expectations "$expect_file" "$work_dir/step2.log" "$mpi_sycl"; then
        echo "[FAIL] MPI generated structure expectations failed."
        FAILED=$((FAILED + 1))
        continue
    fi

    # Step 3: Run MPI and Compare
    echo "  [Step 3] Run MPI (mpirun -np 4) and compare"
    DYLD_LIBRARY_PATH="$ACPP_ROOT/lib" mpirun -np 4 "$mpi_bin" > "$work_dir/mpi.out" 2>&1
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
