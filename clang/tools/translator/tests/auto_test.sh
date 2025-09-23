# !/usr/bin/env bash

exec 2>/dev/null

# 参数解析
MODE="usm"
for arg in "$@"; do
  case $arg in
    --usm) MODE="usm";;
    --buffer) MODE="buffer";;
    --usm_time)MODE="usm_time";;
  esac
done

SCALE="small"
for arg2 in "$@";do
    case $arg2 in
        --small) SCALE="small";;
        --large) SCALE="large";;
    esac
done

# Get the directory of this script
SCRIPT_PATH=$(realpath "${BASH_SOURCE[0]}")
TEST_DIR=$(dirname "$SCRIPT_PATH")
source $TEST_DIR/../env.sh

# Delete all temporary files
TMP_DIR="$TEST_DIR/tmp"
rm -rf $TMP_DIR
mkdir $TMP_DIR

# Edit examples here
examples=(
    "matMul1.0"
    "waveEquation1.0"
    "stencil1.0"
    "jacobi1.0"
    "FOuLa1.0"
    "decay1.0"
    "DFT1.0"
    #"imageAdjustment1.0"
    "liuliang1.0"
    "MDP1.0"
    "mandel1.0"
    "oddeven0.1"
    #"block_mat_mul"
)

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
    if [ "$SCALE" != "large" ]; then
        dacpp_file=$(find "$TEST_DIR/$dir/" -type f -name "*.dac.cpp" | head -n 1)
    else
        dacpp_file=$(find "$TEST_DIR/$dir/" -type f -name "*.large_dac.cpp" | head -n 1)
    fi
    if [ -z "$dacpp_file" ]; then
        echo "Example $dir: DACPP source file not found"
        continue
    fi
    mkdir -p "$TMP_DIR/$dir"
    cp "$dacpp_file" "$TMP_DIR/$dir"
    if [ "$SCALE" != "large" ]; then
        new_dacpp_file=$(find "$TMP_DIR/$dir/" -type f -name "*.dac.cpp" | head -n 1)
    else
        new_dacpp_file=$(find "$TMP_DIR/$dir/" -type f -name "*.large_dac.cpp" | head -n 1)
    fi
#    dacpp "$new_dacpp_file" >/dev/null
#    sycl_file=$(find "$TMP_DIR/$dir" -type f -name "*.dac_sycl.cpp")

    dacpp "$new_dacpp_file" --mode=$MODE >/dev/null
    sycl_file=$(find "$TMP_DIR/$dir" -type f -name "*_sycl_${MODE}.cpp")
    if [ -z "$sycl_file" ]; then
        echo "Example $dir: DACPP to SYCL transpilation failed"
    else
        echo "Example $dir: DACPP to SYCL transpilation succeeded"
    fi
done

echo "------------------------------------------------------------------------------------------"
echo "Compile standard sycl files"
echo
if [ "$MODE" != "usm_time" ]; then
# Compile standard sycl files
  for dir in ${examples[@]}; do
    if [ "$SCALE" != "large" ]; then
        std_sycl_file=$(find "$TEST_DIR/$dir/" -type f -name "*.StandardSycl.cpp" | head -n 1)
    else
        std_sycl_file=$(find "$TEST_DIR/$dir/" -type f -name "*.large_StandardSycl.cpp" | head -n 1)
    fi
    if [ -z "$std_sycl_file" ]; then
        if [ "$SCALE" != "large" ]; then
            std_file=$(find "$TEST_DIR/$dir/" -type f -name "*.serial.cpp" | head -n 1)
        else
            std_file=$(find "$TEST_DIR/$dir/" -type f -name "*.large_serial.cpp" | head -n 1)
        fi
        if [ "$std_file" ]; then
            echo "Example $dir: Standard SYCL file does not exist but serial C++ file exists"
            g++ "$std_file" -o "$TMP_DIR/$dir/std_$dir"
            exe_file=$(find "$TMP_DIR/$dir/" -type f -name "std_$dir")
            if [ -z "$exe_file" ]; then
                echo "Example $dir: Serial C++ file compilation failed"
            else
                exe_file="${exe_file#$TMP_DIR/$dir}"
                "$TMP_DIR/$dir/$exe_file" > "$TMP_DIR/$dir/$exe_file.std.out"
            fi
        else
            echo "Example $dir: No standard SYCL file or serial C++ file found"
        fi
    else
        icpx-gpu "$std_sycl_file" -o "$TMP_DIR/$dir/std_$dir"
        # icpx-cpu "$std_sycl_file" -o "$TMP_DIR/$dir/std_$dir"
        exe_file=$(find "$TMP_DIR/$dir/" -type f -name "std_$dir")
        if [ -z "$exe_file" ]; then
            echo "Example $dir: Standard SYCL file compilation failed"
        else
            exe_file="${exe_file#$TMP_DIR/$dir}"
            "$TMP_DIR/$dir/$exe_file" > "$TMP_DIR/$dir/$exe_file.std.out"
        fi
    fi 
done
else  echo "Skip standard SYCL compilation in usm_time mode"
fi

echo "------------------------------------------------------------------------------------------"
echo "Generated SYCL files compilation test"
echo

# Generated SYCL files compilation test
for dir in ${examples[@]}; do
#    sycl_file=$(find "$TMP_DIR/$dir" -type f -name "*.dac_sycl.cpp")
    sycl_file=$(find "$TMP_DIR/$dir" -type f -name "*_sycl_${MODE}.cpp")
    if [ -z "$sycl_file" ]; then
        continue
    fi
    icpx-gpu -fp-model strict -fno-fast-math "$sycl_file" -o "$TMP_DIR/$dir/$dir"
    # icpx-cpu "$sycl_file" -o "$TMP_DIR/$dir/$dir"
    # mpiicpc -fsycl "$sycl_file" -o "$TMP_DIR/$dir/$dir"
    # mpiicpc-gpu -fsycl "$sycl_file" -o "$TMP_DIR/$dir/$dir"
    exe_file=$(find "$TMP_DIR/$dir/" -type f -name "$dir")
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
    exe_file=$(find "$TMP_DIR/$dir/" -type f -name "$dir")
    if [ -z "$exe_file" ]; then
        continue
    fi
    exe_file="${exe_file#$TMP_DIR/$dir}"

    # 运行并保存结果
    "$TMP_DIR/$dir/$exe_file" > "$TMP_DIR/$dir/$exe_file.out"
    # mpirun -np 2 "$TMP_DIR/$dir/$exe_file" > "$TMP_DIR/$dir/$exe_file.out"

    if [ "$MODE" = "usm_time" ]; then
        # usm_time 模式：只保存结果
        echo "Example $dir: execution finished, result saved to $TMP_DIR/$dir/$exe_file.out"
    else
        # usm / buffer 模式：保存结果 + diff 对比
        std_res=$(find "$TMP_DIR/$dir/" -type f -name "*.std.out" | head -n 1)
        if [ -z "$std_res" ]; then
            echo "Example $dir: no standard result found for diff"
        elif diff -y --suppress-common-lines "$TMP_DIR/$dir/$exe_file.out" "$std_res"; then
            echo "Example $dir: execution test succeeded"
        else
            echo
            echo "Example $dir: execution test failed with some different lines listed above"
        fi
    fi
done


echo "------------------------------------------------------------------------------------------"

# rm -rf $TMP_DIR