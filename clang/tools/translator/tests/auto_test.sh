# !/usr/bin/env bash

exec 2>/dev/null

# Delete all temporary files
rm -rf ./tmp
mkdir ./tmp

# Edit examples here
examples=(
    "matMul1.0"
    # "block_mat_mul"
    "waveEquation1.0"
    "stencil1.0"
    "jacobi1.0"
    "FOuLa1.0"
    "decay1.0"
    "DFT1.0"
    "imageAdjustment1.0"
    "liuliang1.0"
    "MDP1.0"
    "mandel1.0"
    "oddeven0.1"
)


WORK_DIR=../../../../clang/tools/translator

INCLUDE_DIRS=(
    "$WORK_DIR/dpcppLib/include/"
    "$WORK_DIR/dacppLib/include/"
    "$WORK_DIR/rewriter/include/"
    "$WORK_DIR/std_lib/include/"
)

# SRC_FILES=(
#     "$WORK_DIR/rewriter/lib/dacInfo.cpp"
# )

echo "------------------------------------------------------------------------------------------"
echo "DACPP to SYCL transpilation test"
echo

# DACPP to SYCL transpilation test
for dir in ${examples[@]}; do
    dacpp_file=$(find "./$dir/" -type f -name "*.dac.cpp" | head -n 1)
    if [ -z "$dacpp_file" ]; then
        echo "Example $dir: DACPP source file not found"
        continue
    fi
    mkdir -p "./tmp/$dir"
    cp "$dacpp_file" "./tmp/$dir"
    ../../../../build/bin/translator/translator "./tmp/$dacpp_file" \
    -extra-arg=-std=c++17 --  \
    "${INCLUDE_DIRS[@]/#/-I}" \
    >/dev/null
    sycl_file=$(find "./tmp/$dir" -type f -name "*.dac_sycl.cpp")
    if [ -z "$sycl_file" ]; then
        echo "Example $dir: DACPP to SYCL transpilation failed"
    else
        echo "Example $dir: DACPP to SYCL transpilation succeeded"
    fi
done

echo "------------------------------------------------------------------------------------------"
echo "Generated SYCL files compilation test"
echo

# Generated SYCL files compilation test
for dir in ${examples[@]}; do
    sycl_file=$(find "./tmp/$dir" -type f -name "*.dac_sycl.cpp")
    if [ -z "$sycl_file" ]; then
        continue
    fi
    icpx -fsycl -fsycl-targets=nvptx64-nvidia-cuda \
    "$sycl_file" \
    "${INCLUDE_DIRS[@]/#/-I}" \
    -o "./tmp/$dir/$dir" 
    exe_file=$(find "./tmp/$dir/" -type f -name "$dir")
    if [ -z "$exe_file" ]; then
        echo "Example $dir: SYCL compilation failed"
    else
        echo "Example $dir: SYCL compilation succeeded"
    fi
done

echo "------------------------------------------------------------------------------------------"
echo "SYCL files ouput result test"
echo

# SYCL files ouput result test
for dir in ${examples[@]}; do
    exe_file=$(find "./tmp/$dir/" -type f -name "$dir")
    if [ -z "$exe_file" ]; then
        continue
    fi
    exe_file="${exe_file#./tmp/$dir}"
    "./tmp/$dir/$exe_file" > "./tmp/$dir/$exe_file.out" 
    perl -pi -e 'chomp if eof' "./tmp/$dir/$exe_file.out"
    std_res=$(find "./$dir/" -type f -name "*.out" | head -n 1)
    if diff -y --suppress-common-lines "./tmp/$dir/$exe_file.out" "$std_res"; then
        echo "Example $dir: execution test succeeded"
    else
        echo
        echo "Example $dir: execution test failed with some different lines listed above"
    fi
done

echo "------------------------------------------------------------------------------------------"

# rm -rf ./tmp