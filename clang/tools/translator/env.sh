#!/usr/bin/env bash

# Check icpx availability
if ! command -v icpx &> /dev/null; then
    for oneapi_setvars in \
        /home/tools/intel/oneapi/setvars.sh \
        /opt/intel/oneapi/setvars.sh
    do
        if [[ -f "$oneapi_setvars" ]]; then
            echo "Loading Intel oneAPI environment..."
            source "$oneapi_setvars" &> /dev/null
            break
        fi
    done
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
DEFAULT_ACPP_ROOT=/Volumes/QUQ/working/sycl-install

if [[ -z "${ACPP_ROOT:-}" && -d "$DEFAULT_ACPP_ROOT" ]]; then
    ACPP_ROOT="$DEFAULT_ACPP_ROOT"
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

OPENMPI_PREFIX=$(resolve_openmpi_prefix)
LIBOMP_PREFIX=$(resolve_libomp_prefix)

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

# dacpp test.cpp
dacpp() {

    TRANSLATOR="$WORK_DIR"/../../../build/bin/translator/translator

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

    "$TRANSLATOR" "$@" \
    "${EXTRA_ARGS[@]}" -- \
    "${INCLUDE_ARGS[@]}"
}

INCLUDE_DIRS=(
    "$WORK_DIR/dpcppLib/include/"
    "$WORK_DIR/dacppLib/include/"
    "$WORK_DIR/rewriter/include/"
)

# SRC_FILES=(
#     "$WORK_DIR/rewriter/lib/dacInfo.cpp"
# )

resolve_tool() {
    type -P "$1" 2>/dev/null || command -v "$1" 2>/dev/null || true
}

generated_cpp_for() {
    local input="$1"
    local base="${input%.*}"
    printf '%s\n' "${base}_sycl_usm.cpp"
}

default_output_for_generated() {
    local generated="$1"
    local file_name
    file_name=$(basename "${generated%.*}")
    printf '%s\n' "/tmp/$file_name"
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

    DYLD_LIBRARY_PATH="$ACPP_ROOT/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
    "$ACPP_ROOT/bin/acpp" \
        -O2 -std=c++17 \
        "$input_cpp" \
        "${COMMON_CXX_FLAGS[@]}" \
        "${ACPP_COMPILE_FLAGS[@]}" \
        "${INCLUDE_DIRS[@]/#/-I}" \
        "${ACPP_LINK_FLAGS[@]}" \
        -o "$output_bin"
}

dacpp-translate-and-build() {
    if [[ $# -lt 1 || $# -gt 2 ]]; then
        echo "usage: dacpp-translate-and-build <source.dac.cpp> [output_bin]" >&2
        return 2
    fi

    local input_cpp="$1"
    local output_bin="$2"

    dacpp "$input_cpp" --mpi || return $?

    local generated_cpp
    generated_cpp=$(generated_cpp_for "$input_cpp")

    if [[ -z "$output_bin" ]]; then
        acpp-compile "$generated_cpp"
    else
        acpp-compile "$generated_cpp" "$output_bin"
    fi
}

ICPX=$(resolve_tool icpx)
MPICXX=$(resolve_tool mpicxx)
MPIICPC=$(resolve_tool mpiicpc)

if [[ -z "$MPIICPC" ]]; then
    MPIICPC="$MPICXX"
fi

# icpx -fsycl -fsycl-targets=nvptx64-nvidia-cuda --cuda-path=/data/cuda/cuda-11.8 -o test test.cpp
icpx() {
    if [[ -z "$ICPX" ]]; then
        echo "icpx not found in PATH" >&2
        return 127
    fi

    "$ICPX" "$@" \
    "${COMMON_CXX_FLAGS[@]}" \
    "${INCLUDE_DIRS[@]/#/-I}"

}

# icpx-cpu -o test test.cpp
icpx-cpu() {
    if [[ -z "$ICPX" ]]; then
        echo "icpx not found in PATH" >&2
        return 127
    fi

    "$ICPX" -fsycl \
     "$@" \
    "${COMMON_CXX_FLAGS[@]}" \
    "${INCLUDE_DIRS[@]/#/-I}"
    
}

# icpx-gpu -o test test.cpp
icpx-gpu() {
    if [[ -z "$ICPX" ]]; then
        echo "icpx not found in PATH" >&2
        return 127
    fi

    "$ICPX" -fsycl \
    -fsycl-targets=nvptx64-nvidia-cuda \
    --cuda-path=/home/tools/cuda/cuda-12.2 \
     "$@" \
    "${COMMON_CXX_FLAGS[@]}" \
    "${INCLUDE_DIRS[@]/#/-I}"
    
}

export I_MPI_CXX=icpx

#mpicxx -fsycl -o test test.cpp
mpicxx () {
    if [[ -z "$MPICXX" ]]; then
        echo "mpicxx not found in PATH" >&2
        return 127
    fi
    "$MPICXX" "$@" \
    "${COMMON_CXX_FLAGS[@]}" \
    "${INCLUDE_DIRS[@]/#/-I}"
}

# mpicxx-gpu -o test test.cpp
mpicxx-gpu() {
    if [[ -z "$MPICXX" ]]; then
        echo "mpicxx not found in PATH" >&2
        return 127
    fi

    "$MPICXX" -fsycl \
    -fsycl-targets=nvptx64-nvidia-cuda \
    --cuda-path=/home/tools/cuda/cuda-12.2 \
     "$@" \
    "${COMMON_CXX_FLAGS[@]}" \
    "${INCLUDE_DIRS[@]/#/-I}"
    
}

# mpiicpc -fsycl -o test test.cpp
mpiicpc () {
    if [[ -z "$MPIICPC" ]]; then
        echo "mpiicpc/mpicxx not found in PATH" >&2
        return 127
    fi
    "$MPIICPC" "$@" \
    "${COMMON_CXX_FLAGS[@]}" \
    "${INCLUDE_DIRS[@]/#/-I}"
}

# mpiicpc-gpu -o test test.cpp
mpiicpc-gpu() {
    if [[ -z "$MPIICPC" ]]; then
        echo "mpiicpc/mpicxx not found in PATH" >&2
        return 127
    fi

    "$MPIICPC" -fsycl \
    -fsycl-targets=nvptx64-nvidia-cuda \
    --cuda-path=/home/tools/cuda/cuda-12.2 \
     "$@" \
    "${COMMON_CXX_FLAGS[@]}" \
    "${INCLUDE_DIRS[@]/#/-I}"
    
}
