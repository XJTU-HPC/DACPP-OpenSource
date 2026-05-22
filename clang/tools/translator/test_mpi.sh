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
    "mpiOrReadWriteAccumulate1D"
    "mpiOrReadWriteAccumulate2D"
    "oddeven0.1"
    "vectorAddCombo"
    "mpiLoopReadWriteReject1D"
    "mpiLoopAliasReject1D"
    "mpiLoopVariantInputReject1D"
    "mpiLoopChainReject1D"
    "mpiLoopStencilInvariantReader1D"
    "mpiLoopStencilAliasRefresh1D"
    "mpiLoopStencilReferenceAliasRefresh1D"
    "mpiLoopStencilResidentHalo1D"
    "mpiLoopStencilResidentHaloEmptyRank1D"
    "mpiLoopStencilRightBoundaryFullSync1D"
    "mpiLoopStencilScalarReject2D"
    "mpiLoopStencilCountGuard2D"
    "mpiLoopStencilOrderReject2D"
    "mpiOwnerLoopWrongSliceReject1D"
    "mpiOwnerLoopVariantScalarReject1D"
    "mpiOwnerLoopMissingWritebackReject1D"
    "mpiOwnerLoopExtraMutationReject1D"
    "mpiOwnerLoopMultipleWriterReject1D"
    "mpiOwnerLoopScalarPayloadExprReject1D"
    "mpiOwnerLoopScalarShellArgReject1D"
    "mpiOwnerLoopLoopBoundReject1D"
    "mpiOwnerLoopWriterRangeReject1D"
    "mpiOwnerLoopWritebackIndexReject1D"
    "mpiFixedBlockOverlapReject1D"
    "mpiFixedBlockPayloadReject1D"
    "mpiFixedBlockMatrixSingleSplitReject1D"
    "mpiFixedBlockAllRanksFunctionRead1D"
    "mpiFixedBlockRootOnlyCout1D"
    "mpiFixedBlockPhaseExchangeOffsetReject1D"
    "mpiFixedBlockPhaseExchangeMissingBoundaryReject1D"
    "mpiFixedBlockPhaseExchangeNonAdjacentReject1D"
    "mpiFixedBlockPhaseExchangeRank3Run1D"
    "mpiFixedBlockPhaseExchangeWrongArgsReject1D"
    "mpiFixedBlockPhaseExchangeRemovedStmtUseReject1D"
    "mpiFixedBlockPhaseExchangeUnexpectedWriteReject1D"
    "mpiFixedBlockPhaseExchangeRuntimeMismatchGuard1D"
    "mpiFixedBlockPhaseExchangePostOutputUseReject1D"
    "mpiFixedBlockPhaseExchangePostOutputAliasReject1D"
    "mpiFixedBlockPhaseExchangeOddNReject1D"
    "gradientSum"
    "mpiBroadcastRootOnlyCout"
    "mpiBroadcastTensor2Array"
    "mpiBroadcastUnknownFunction"
    "mpiBroadcastAliasRead"
    "mpiDistributedStencil1D"
    "mpiDistributedStencilNoBridge1D"
    "mpiDistributedStencilSteady1D"
    "mpiOrStencilRefreshPolicy1D"
    "mpiDistributedStencilAstRoute1D"
    "mpiDistributedStencilAstRouteFallback1D"
    "mpiDistributedStencil2DRowBlock"
    "mpiOrStencilBoundaryStride2D"
    "mpiPhaseCWriteThenRead1D"
    "mpiPhaseCHalo1D"
    "mpiPhaseCHaloWide1D"
    "mpiMixedStencilORPhaseC"
    "spatialStencil2DOneStep"
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

is_structure_only_test() {
    local expect_file="$1"

    [[ -f "$expect_file" ]] || return 1

    while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
        local line kind value
        line="${raw_line#"${raw_line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        [[ -z "$line" || "$line" == \#* ]] && continue

        kind="${line%%:*}"
        value="${line#*:}"
        value="${value#"${value%%[![:space:]]*}"}"
        value="${value%"${value##*[![:space:]]}"}"

        if [[ "$kind" == "MPI_STRUCTURE_ONLY" ]]; then
            case "$value" in
                1|true|TRUE|yes|YES)
                    return 0
                    ;;
            esac
        fi
    done < "$expect_file"

    return 1
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
            MPI_STRUCTURE_ONLY)
                ;;
            MPI_RANKS)
                ;;
            *)
                echo "[FAIL] Unknown expectation kind in $expect_file: $kind"
                failed=1
                ;;
        esac
    done < "$expect_file"

    return "$failed"
}

mpi_ranks_for_test() {
    local expect_file="$1"
    local ranks="${MPI_TEST_RANKS:-4}"

    [[ -f "$expect_file" ]] || {
        printf '%s\n' "$ranks"
        return
    }

    while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
        local line kind value
        line="${raw_line#"${raw_line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        [[ -z "$line" || "$line" == \#* ]] && continue

        kind="${line%%:*}"
        value="${line#*:}"
        value="${value#"${value%%[![:space:]]*}"}"
        value="${value%"${value##*[![:space:]]}"}"

        if [[ "$kind" == "MPI_RANKS" && "$value" =~ ^[1-9][0-9]*$ ]]; then
            ranks="$value"
        fi
    done < "$expect_file"

    printf '%s\n' "$ranks"
}

TOTAL=0; PASSED=0; FAILED=0; SKIPPED=0

for test_name in "${MPI_TESTS[@]}"; do
    echo "========================================================"
    echo "  Test: $test_name"
    echo "========================================================"
    TOTAL=$((TOTAL + 1))

    expect_file="$TESTS_DIR/$test_name/mpi_expect.txt"
    if [[ "$USE_LARGE_CASES" == "1" && -f "$TESTS_DIR/$test_name/mpi_expect_large.txt" ]]; then
        expect_file="$TESTS_DIR/$test_name/mpi_expect_large.txt"
    fi
    structure_only=0
    if is_structure_only_test "$expect_file"; then
        structure_only=1
    fi
    mpi_ranks="$(mpi_ranks_for_test "$expect_file")"

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

    if [[ "$structure_only" == "1" ]]; then
        echo "  [Step 1] Structure-only test; skipping baseline translate/compile/run"
    else
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
    fi

    # Step 2: Translate MPI
    if [[ "$structure_only" == "1" ]]; then
        echo "  [Step 2] Translate --mode=buffer --mpi (structure-only)"
    else
        echo "  [Step 2] Translate --mode=buffer --mpi and compile"
    fi
    mpi_dac="$(mpi_dac_path_for "$base_dac")"
    cp "$dac_file" "$mpi_dac"
    mpi_sycl="$(generated_cpp_path_for "$mpi_dac")"
    mpi_bin="$work_dir/mpi_bin"

    if [[ "$structure_only" == "1" ]]; then
        step2_cmd="dacpp '$mpi_dac' --mode=buffer --mpi"
    else
        step2_cmd="dacpp '$mpi_dac' --mode=buffer --mpi && acpp-compile '$mpi_sycl' '$mpi_bin'"
    fi
    if ! run_in_env "$work_dir/step2.log" "$step2_cmd"; then
        echo "[FAIL] MPI translate/compile failed."
        head -n 20 "$work_dir/step2.log"
        FAILED=$((FAILED + 1))
        continue
    fi
    if grep -Fq "<->" "$mpi_sycl"; then
        echo "[FAIL] Generated MPI SYCL still contains a DACPP <-> expression."
        grep -Fn "<->" "$mpi_sycl" | head -n 5
        FAILED=$((FAILED + 1))
        continue
    fi

    if ! check_mpi_expectations "$expect_file" "$work_dir/step2.log" "$mpi_sycl"; then
        echo "[FAIL] MPI generated structure expectations failed."
        FAILED=$((FAILED + 1))
        continue
    fi

    if [[ "$structure_only" == "1" ]]; then
        echo "[PASS] $test_name: MPI structure expectations passed."
        PASSED=$((PASSED + 1))
        continue
    fi

    # Step 3: Run MPI and Compare
    echo "  [Step 3] Run MPI (mpirun -np $mpi_ranks) and compare"
    DYLD_LIBRARY_PATH="$ACPP_ROOT/lib" mpirun -np "$mpi_ranks" "$mpi_bin" > "$work_dir/mpi.out" 2>&1
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
