#!/usr/bin/env bash

# Check icpx availability
if ! command -v icpx &> /dev/null; then
    ONEAPI_SETVARS_CANDIDATES=()
    if [[ -n "${ONEAPI_SETVARS:-}" ]]; then
        ONEAPI_SETVARS_CANDIDATES+=("$ONEAPI_SETVARS")
    fi
    if [[ -n "${ONEAPI_ROOT:-}" ]]; then
        ONEAPI_SETVARS_CANDIDATES+=("$ONEAPI_ROOT/setvars.sh")
    fi
    if [[ -n "${INTEL_ONEAPI_ROOT:-}" ]]; then
        ONEAPI_SETVARS_CANDIDATES+=("$INTEL_ONEAPI_ROOT/setvars.sh")
    fi

    if [[ ${#ONEAPI_SETVARS_CANDIDATES[@]} -gt 0 ]]; then
        for oneapi_setvars in "${ONEAPI_SETVARS_CANDIDATES[@]}"; do
            if [[ -f "$oneapi_setvars" ]]; then
                echo "Loading Intel oneAPI environment..."
                source "$oneapi_setvars" &> /dev/null
                break
            fi
        done
    fi
fi

# Get the directory of this script
if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
    SCRIPT_PATH=$(realpath "${BASH_SOURCE[0]}")
elif [[ -n "${ZSH_VERSION:-}" ]]; then
    SCRIPT_PATH=$(realpath "${(%):-%x}")
else
    SCRIPT_PATH=$(realpath "$0")
fi
WORK_DIR=$(dirname "$SCRIPT_PATH")
HOST_OS=$(uname -s)
DACPP_TMP_ROOT="${DACPP_TMP_ROOT:-${TMPDIR:-/tmp}}"
DACPP_TMP_ROOT="${DACPP_TMP_ROOT%/}"

if [[ -z "$DACPP_TMP_ROOT" ]]; then
    DACPP_TMP_ROOT="/"
fi

if [[ -z "${ACPP_ROOT:-}" && -n "${ADAPTIVECPP_ROOT:-}" ]]; then
    ACPP_ROOT="$ADAPTIVECPP_ROOT"
fi

resolve_brew_prefix() {
    local formula="$1"
    if command -v brew &> /dev/null; then
        brew --prefix "$formula" 2>/dev/null || true
    fi
}

resolve_openmpi_prefix() {
    local prefix
    prefix=$(resolve_brew_prefix open-mpi)
    if [[ -n "$prefix" ]]; then
        echo "$prefix"
        return
    fi

    local mpicxx_path
    mpicxx_path=$(type -P mpicxx 2>/dev/null || command -v mpicxx 2>/dev/null || true)
    if [[ -n "$mpicxx_path" ]]; then
        cd "$(dirname "$mpicxx_path")/.." && pwd
    fi
}

resolve_libomp_prefix() {
    resolve_brew_prefix libomp
}

resolve_tool() {
    type -P "$1" 2>/dev/null || command -v "$1" 2>/dev/null || true
}

OPENMPI_PREFIX=$(resolve_openmpi_prefix)
LIBOMP_PREFIX=$(resolve_libomp_prefix)
DACPP_ICPX="${ICPX:-$(resolve_tool icpx)}"
DACPP_MPICXX="${MPICXX:-$(resolve_tool mpicxx)}"

if [[ -z "${I_MPI_CXX:-}" && -n "$DACPP_ICPX" ]]; then
    export I_MPI_CXX="$DACPP_ICPX"
fi

if [[ -z "${OMPI_CXX:-}" && -n "$DACPP_ICPX" ]]; then
    export OMPI_CXX="$DACPP_ICPX"
fi

ACPP_COMPILE_FLAGS=()
ACPP_LINK_FLAGS=()

if [[ -n "${ACPP_ROOT:-}" ]]; then
    ACPP_COMPILE_FLAGS+=(
        -I"$ACPP_ROOT/include/AdaptiveCpp"
    )
fi

if [[ -n "$LIBOMP_PREFIX" ]]; then
    ACPP_COMPILE_FLAGS+=(
        -I"$LIBOMP_PREFIX/include"
    )
fi

if [[ -n "$OPENMPI_PREFIX" ]]; then
    ACPP_COMPILE_FLAGS+=(
        -I"$OPENMPI_PREFIX/include"
    )
    ACPP_LINK_FLAGS+=(
        -L"$OPENMPI_PREFIX/lib"
        -lmpi
    )
fi

COMMON_CXX_FLAGS=()
if [[ "$HOST_OS" == "Darwin" ]]; then
    SDKROOT=$(xcrun --show-sdk-path 2>/dev/null)
    CLANG_RESOURCE_DIR=$(clang -print-resource-dir 2>/dev/null)
    CXX_INCLUDE_DIR="$SDKROOT/usr/include/c++/v1"

    if [[ -n "$SDKROOT" ]]; then
        COMMON_CXX_FLAGS+=(
            -stdlib=libc++
            -isysroot
            "$SDKROOT"
        )
    fi

    if [[ -n "$CLANG_RESOURCE_DIR" ]]; then
        COMMON_CXX_FLAGS+=(
            -resource-dir
            "$CLANG_RESOURCE_DIR"
        )
    fi

    if [[ -d "$CXX_INCLUDE_DIR" ]]; then
        COMMON_CXX_FLAGS+=(
            -cxx-isystem
            "$CXX_INCLUDE_DIR"
        )
    fi
fi

dacpp_args_enable_mpi() {
    local arg
    for arg in "$@"; do
        case "$arg" in
            --)
                break
                ;;
            --mpi|-mpi)
                return 0
                ;;
            --mpi=*|-mpi=*)
                local value="${arg#*=}"
                case "$value" in
                    0|false|False|FALSE|off|OFF)
                        ;;
                    *)
                        return 0
                        ;;
                esac
                ;;
        esac
    done
    return 1
}

# dacpp test.cpp
dacpp() {

    TRANSLATOR="${DACPP_TRANSLATOR:-$WORK_DIR/../../../build/bin/translator/translator}"

    # Verify that the translator exists and is executable
    if [[ ! -x "$TRANSLATOR" ]]; then
        echo "No such file: $TRANSLATOR" >&2
        return 1
    fi

    EXTRA_ARGS=(
        -extra-arg=-std=c++17
    )

    INCLUDE_ARGS=(
        -I"$WORK_DIR"/std_lib/include
        -I"$WORK_DIR"/dacppLib/include
    )

    TRANSLATOR_ARGS=()
    USER_COMPILE_ARGS=()
    local after_user_separator=0
    local arg
    for arg in "$@"; do
        if [[ "$after_user_separator" == "0" && "$arg" == "--" ]]; then
            after_user_separator=1
            continue
        fi

        if [[ "$after_user_separator" == "1" ]]; then
            USER_COMPILE_ARGS+=("$arg")
        else
            TRANSLATOR_ARGS+=("$arg")
        fi
    done

    if [[ "$HOST_OS" == "Darwin" ]]; then
        if [[ ${#COMMON_CXX_FLAGS[@]} -gt 0 ]]; then
            for flag in "${COMMON_CXX_FLAGS[@]}"; do
                EXTRA_ARGS+=("-extra-arg=$flag")
            done
        fi

        INCLUDE_ARGS=(
            -I"$WORK_DIR"/dacppLib/include
            -idirafter "$WORK_DIR"/std_lib/include
        )
    fi

    if ! dacpp_args_enable_mpi "$@" && [[ -d "$WORK_DIR/single_overlay" ]]; then
        INCLUDE_ARGS=(
            -I"$WORK_DIR"/single_overlay/dacppLib/include
            -I"$WORK_DIR"/single_overlay/dpcppLib/include
            -I"$WORK_DIR"/single_overlay/rewriter/include
            "${INCLUDE_ARGS[@]}"
        )
    fi

    "$TRANSLATOR" "${TRANSLATOR_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" -- \
    ${USER_COMPILE_ARGS[@]+"${USER_COMPILE_ARGS[@]}"} \
    "${INCLUDE_ARGS[@]}"
}

INCLUDE_DIRS=(
    "$WORK_DIR/dpcppLib/include/"
    "$WORK_DIR/dacppLib/include/"
    "$WORK_DIR/rewriter/include/"
)

SINGLE_INCLUDE_DIRS=(
    "$WORK_DIR/single_overlay/dpcppLib/include/"
    "$WORK_DIR/single_overlay/dacppLib/include/"
    "$WORK_DIR/single_overlay/rewriter/include/"
)

# SRC_FILES=(
#     "$WORK_DIR/rewriter/lib/dacInfo.cpp"
# )

generated_cpp_for() {
    local input="$1"
    local base="${input%.*}"
    local candidates=(
        "${base}_sycl_buffer.cpp"
        "${base}_sycl_usm.cpp"
        "${base}_sycl_buffer_mpi.cpp"
    )
    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -f "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    printf '%s\n' "${base}_sycl_buffer.cpp"
}

default_output_for_generated() {
    local generated="$1"
    local file_name
    file_name=$(basename "${generated%.*}")
    printf '%s\n' "$DACPP_TMP_ROOT/$file_name"
}

acpp-compile() {
    if [[ -z "${ACPP_ROOT:-}" || ! -x "$ACPP_ROOT/bin/acpp" ]]; then
        echo "AdaptiveCpp compiler not found. Set ACPP_ROOT to a valid install prefix." >&2
        return 127
    fi

    if [[ $# -lt 1 || $# -gt 2 ]]; then
        echo "usage: acpp-compile <generated.cpp> [output_bin]" >&2
        return 2
    fi

    local input_cpp="$1"
    local output_bin="${2:-$(default_output_for_generated "$input_cpp")}"
    local compile_include_dirs=("${INCLUDE_DIRS[@]}")

    if [[ -d "$WORK_DIR/single_overlay" ]] &&
       ! grep -Fq '"MPIPlanner.h"' "$input_cpp" 2>/dev/null; then
        compile_include_dirs=("${SINGLE_INCLUDE_DIRS[@]}" "${INCLUDE_DIRS[@]}")
    fi

    DYLD_LIBRARY_PATH="$ACPP_ROOT/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
    "$ACPP_ROOT/bin/acpp" \
        -O2 -std=c++17 \
        "$input_cpp" \
        ${COMMON_CXX_FLAGS[@]+"${COMMON_CXX_FLAGS[@]}"} \
        ${ACPP_COMPILE_FLAGS[@]+"${ACPP_COMPILE_FLAGS[@]}"} \
        "${compile_include_dirs[@]/#/-I}" \
        ${ACPP_LINK_FLAGS[@]+"${ACPP_LINK_FLAGS[@]}"} \
        -o "$output_bin"
}

dacpp_prepare_compile_include_dirs() {
    local input_cpp="$1"
    DACPP_COMPILE_INCLUDE_DIRS=("${INCLUDE_DIRS[@]}")

    if [[ -d "$WORK_DIR/single_overlay" ]] &&
       ! grep -Fq '"MPIPlanner.h"' "$input_cpp" 2>/dev/null; then
        DACPP_COMPILE_INCLUDE_DIRS=("${SINGLE_INCLUDE_DIRS[@]}" "${INCLUDE_DIRS[@]}")
    fi
}

oneapi-compile() {
    if [[ -z "$DACPP_ICPX" || ! -x "$DACPP_ICPX" ]]; then
        echo "icpx not found. Set ONEAPI_SETVARS, ONEAPI_ROOT, INTEL_ONEAPI_ROOT, or ICPX." >&2
        return 127
    fi

    if [[ $# -lt 1 || $# -gt 2 ]]; then
        echo "usage: oneapi-compile <generated.cpp> [output_bin]" >&2
        return 2
    fi

    local input_cpp="$1"
    local output_bin="${2:-$(default_output_for_generated "$input_cpp")}"
    dacpp_prepare_compile_include_dirs "$input_cpp"

    "$DACPP_ICPX" \
        -O2 -std=c++17 -fsycl \
        "$input_cpp" \
        ${COMMON_CXX_FLAGS[@]+"${COMMON_CXX_FLAGS[@]}"} \
        "${DACPP_COMPILE_INCLUDE_DIRS[@]/#/-I}" \
        ${ACPP_LINK_FLAGS[@]+"${ACPP_LINK_FLAGS[@]}"} \
        -o "$output_bin"
}

mpicxx-compile() {
    if [[ -z "$DACPP_MPICXX" || ! -x "$DACPP_MPICXX" ]]; then
        echo "mpicxx not found. Set MPICXX or load an MPI environment." >&2
        return 127
    fi

    if [[ $# -lt 1 || $# -gt 2 ]]; then
        echo "usage: mpicxx-compile <generated.cpp> [output_bin]" >&2
        return 2
    fi

    local input_cpp="$1"
    local output_bin="${2:-$(default_output_for_generated "$input_cpp")}"
    dacpp_prepare_compile_include_dirs "$input_cpp"

    "$DACPP_MPICXX" \
        -O2 -std=c++17 -fsycl \
        "$input_cpp" \
        ${COMMON_CXX_FLAGS[@]+"${COMMON_CXX_FLAGS[@]}"} \
        "${DACPP_COMPILE_INCLUDE_DIRS[@]/#/-I}" \
        ${ACPP_LINK_FLAGS[@]+"${ACPP_LINK_FLAGS[@]}"} \
        -o "$output_bin"
}

dacpp_detect_sycl_compiler() {
    case "${DACPP_SYCL_COMPILER:-auto}" in
        adaptivecpp|acpp)
            printf '%s\n' adaptivecpp
            ;;
        oneapi|icpx)
            printf '%s\n' icpx
            ;;
        mpi|mpicxx)
            printf '%s\n' mpicxx
            ;;
        auto|"")
            if [[ -n "${ACPP_ROOT:-}" && -x "$ACPP_ROOT/bin/acpp" ]]; then
                printf '%s\n' adaptivecpp
            elif [[ -n "$DACPP_ICPX" && -x "$DACPP_ICPX" ]]; then
                printf '%s\n' icpx
            elif [[ -n "$DACPP_MPICXX" && -x "$DACPP_MPICXX" ]]; then
                printf '%s\n' mpicxx
            else
                echo "No SYCL compiler found. Set ACPP_ROOT/ADAPTIVECPP_ROOT, load oneAPI, or set DACPP_SYCL_COMPILER." >&2
                return 127
            fi
            ;;
        *)
            echo "Unsupported DACPP_SYCL_COMPILER='${DACPP_SYCL_COMPILER}' (use auto, adaptivecpp, icpx, or mpicxx)." >&2
            return 2
            ;;
    esac
}

dacpp-compile() {
    local compiler
    compiler=$(dacpp_detect_sycl_compiler) || return $?

    case "$compiler" in
        adaptivecpp)
            acpp-compile "$@"
            ;;
        icpx)
            oneapi-compile "$@"
            ;;
        mpicxx)
            mpicxx-compile "$@"
            ;;
    esac
}
