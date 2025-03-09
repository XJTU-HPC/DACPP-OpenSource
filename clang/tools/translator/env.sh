#!/usr/bin/env bash

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