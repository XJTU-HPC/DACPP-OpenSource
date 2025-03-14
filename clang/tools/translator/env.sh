#!/usr/bin/env bash

# Check icpx availability
if ! which icpx &> /dev/null; then
    echo "Loading Intel oneAPI environment..."
    source /data/zhouyh/share/intel/oneapi/setvars.sh intel64 &> /dev/null
fi

# Get the directory of this script
SCRIPT_PATH=$(realpath "${BASH_SOURCE[0]}")
WORK_DIR=$(dirname "$SCRIPT_PATH")

# dacpp test.cpp
dacpp() {

    TRANSLATOR="$WORK_DIR"/../../../build/bin/translator/translator

    # Verify that the translator exists and is executable
    if [[ ! -x "$TRANSLATOR" ]]; then
        echo "No such file: $TRANSLATOR" >&2
        return 1
    fi

    "$TRANSLATOR" "$@" \
    -extra-arg=-std=c++17 -- \
    -I"$WORK_DIR"/std_lib/include \
    -I"$WORK_DIR"/dacppLib/include 
}

INCLUDE_DIRS=(
    "$WORK_DIR/dpcppLib/include/"
    "$WORK_DIR/dacppLib/include/"
    "$WORK_DIR/rewriter/include/"
)

# SRC_FILES=(
#     "$WORK_DIR/rewriter/lib/dacInfo.cpp"
# )

ICPX=$(which icpx)
MPICXX=$(which mpicxx)
MPIICPC=$(which mpiicpc)

# icpx -fsycl -fsycl-targets=nvptx64-nvidia-cuda --cuda-path=/data/cuda/cuda-11.8 -o test test.cpp
icpx() {

    "$ICPX" "$@" \
    "${INCLUDE_DIRS[@]/#/-I}"

}

# icpx-cpu -o test test.cpp
icpx-cpu() {

    "$ICPX" -fsycl \
     "$@" \
    "${INCLUDE_DIRS[@]/#/-I}"
    
}

# icpx-gpu -o test test.cpp
icpx-gpu() {

    "$ICPX" -fsycl \
    -fsycl-targets=nvptx64-nvidia-cuda \
    --cuda-path=/data/cuda/cuda-11.8 \
     "$@" \
    "${INCLUDE_DIRS[@]/#/-I}"
    
}

export I_MPI_CXX=icpx

#mpicxx -fsycl -o test test.cpp
mpicxx () {
    "$MPICXX" "$@" \
    "${INCLUDE_DIRS[@]/#/-I}"
}

# mpicxx-gpu -o test test.cpp
mpicxx-gpu() {

    "$MPICXX" -fsycl \
    -fsycl-targets=nvptx64-nvidia-cuda \
    --cuda-path=/data/cuda/cuda-11.8 \
     "$@" \
    "${INCLUDE_DIRS[@]/#/-I}"
    
}

# mpiicpc -fsycl -o test test.cpp
mpiicpc () {
    "$MPIICPC" "$@" \
    "${INCLUDE_DIRS[@]/#/-I}"
}

# mpiicpc-gpu -o test test.cpp
mpiicpc-gpu() {

    "$MPIICPC" -fsycl \
    -fsycl-targets=nvptx64-nvidia-cuda \
    --cuda-path=/data/cuda/cuda-11.8 \
     "$@" \
    "${INCLUDE_DIRS[@]/#/-I}"
    
}