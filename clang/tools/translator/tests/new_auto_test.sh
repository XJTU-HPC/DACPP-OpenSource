# !/usr/bin/env bash

# Check icpx availability
if ! which icpx &> /dev/null; then
    echo "Loading Intel oneAPI environment..."
    source /data/qinian/share/intel/oneapi2025/setvars.sh intel64 &> /dev/null
fi

source ../env.sh

exec 2>/dev/null

# Delete all temporary files
rm -rf ./tmp
mkdir ./tmp

# Edit examples here
examples=(
    "matMul1.0"
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
    # "block_mat_mul"
)


WORK_DIR=../../../../clang/tools/translator

INCLUDE_DIRS=(
    "$WORK_DIR/dpcppLib/include/"
    "$WORK_DIR/dacppLib/include/"
    "$WORK_DIR/rewriter/include/"
    "$WORK_DIR/std_lib/include/"
)


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
    dacpp "./tmp/$dacpp_file" >/dev/null
    sycl_file=$(find "./tmp/$dir" -type f -name "*.dac_sycl.cpp")
    if [ -z "$sycl_file" ]; then
        echo "Example $dir: DACPP to SYCL transpilation failed"
    else
        echo "Example $dir: DACPP to SYCL transpilation succeeded"
    fi
done

echo "------------------------------------------------------------------------------------------"
echo "Compile standard sycl files"
echo

# Compile standard sycl files
for dir in ${examples[@]}; do
    std_sycl_file=$(find "./$dir/" -type f -name "*.StandardSycl.cpp" | head -n 1)
    if [ -z "$std_sycl_file" ]; then
        std_file=$(find "./$dir/" -type f -name "*.serial.cpp" | head -n 1)
        if [ "$std_file" ]; then
            echo "Example $dir: Standard SYCL file does not exist but serial C++ file exists"
            g++ "$std_file" -o "./tmp/$dir/std_$dir"
            exe_file=$(find "./tmp/$dir/" -type f -name "std_$dir")
            if [ -z "$exe_file" ]; then
                echo "Example $dir: Serial C++ file compilation failed"
            else
                exe_file="${exe_file#./tmp/$dir}"
                "./tmp/$dir/$exe_file" > "./tmp/$dir/$exe_file.std.out"
            fi
        else
            echo "Example $dir: No standard SYCL file or serial C++ file found"
        fi
    else
        icpx-gpu "$std_sycl_file" -o "./tmp/$dir/std_$dir"
        # icpx-cpu "$std_sycl_file" -o "./tmp/$dir/std_$dir"
        exe_file=$(find "./tmp/$dir/" -type f -name "std_$dir")
        if [ -z "$exe_file" ]; then
            echo "Example $dir: Standard SYCL file compilation failed"
        else
            exe_file="${exe_file#./tmp/$dir}"
            "./tmp/$dir/$exe_file" > "./tmp/$dir/$exe_file.std.out"
        fi
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
    icpx-gpu "$sycl_file" -o "./tmp/$dir/$dir"
    # icpx-cpu "$sycl_file" -o "./tmp/$dir/$dir"
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
    std_res=$(find "./tmp/$dir/" -type f -name "*.std.out" | head -n 1)
    if diff -y --suppress-common-lines "./tmp/$dir/$exe_file.out" "$std_res"; then
        echo "Example $dir: execution test succeeded"
    else
        echo
        echo "Example $dir: execution test failed with some different lines listed above"
    fi
done

echo "------------------------------------------------------------------------------------------"

# rm -rf ./tmp