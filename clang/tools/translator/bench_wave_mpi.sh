#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
source "$SCRIPT_DIR/env.sh"

BENCH_DIR="/Volumes/QUQ/working/mpi_tmp/wave_bench"
NX_VALUE="${WAVE_BENCH_NX:-1024}"
NY_VALUE="${WAVE_BENCH_NY:-1024}"
TIME_STEPS_VALUE="${WAVE_BENCH_TIME_STEPS:-200}"
MPI_RANKS="${WAVE_BENCH_RANKS:-4}"
RUNS="${WAVE_BENCH_RUNS:-3}"
RUN_TIMEOUT_SEC="${WAVE_BENCH_TIMEOUT_SEC:-600}"
POST_TIME_GRACE_SEC="${WAVE_BENCH_POST_TIME_GRACE_SEC:-2}"

ACTIVE_RUNNER=""
ACTIVE_BIN=""

rm -rf "$BENCH_DIR"
mkdir -p "$BENCH_DIR"

copy_and_scale_dac() {
    local input="$1"
    local output="$2"
    python3 - "$input" "$output" "$NX_VALUE" "$NY_VALUE" "$TIME_STEPS_VALUE" <<'PY'
from pathlib import Path
import re
import sys

inp, out, nx, ny, steps = sys.argv[1:]
src = Path(inp).read_text()
src = re.sub(r"const int NX\s*=\s*\d+\s*;", f"const int NX = {nx};", src, count=1)
src = re.sub(r"const int NY\s*=\s*\d+\s*;", f"const int NY = {ny};", src, count=1)
src = re.sub(r"const int TIME_STEPS\s*=\s*\d+\s*;", f"const int TIME_STEPS = {steps};", src, count=1)
Path(out).write_text(src)
PY
}

copy_and_scale_coarse() {
    local input="$1"
    local output="$2"
    python3 - "$input" "$output" "$NX_VALUE" "$NY_VALUE" "$TIME_STEPS_VALUE" <<'PY'
from pathlib import Path
import re
import sys

inp, out, nx, ny, steps = sys.argv[1:]
src = Path(inp).read_text()
src = re.sub(r"#define WAVE_NX\s+\d+", f"#define WAVE_NX {nx}", src, count=1)
src = re.sub(r"#define WAVE_NY\s+\d+", f"#define WAVE_NY {ny}", src, count=1)
src = re.sub(r"#define WAVE_TIME_STEPS\s+\d+", f"#define WAVE_TIME_STEPS {steps}", src, count=1)
Path(out).write_text(src)
PY
}

disable_final_print() {
    local path="$1"
    python3 - "$path" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
src = path.read_text()
src = re.sub(r"\n\s*matCur\.print\(\);\s*\n", "\n", src, count=1)
path.write_text(src)
PY
}

disable_coarse_result_dump() {
    local path="$1"
    python3 - "$path" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
src = path.read_text()
updated = re.sub(
    r'\n\s*std::cout << "\{";\n[\s\S]*?\n\s*std::cout << "\}" << std::endl;\s*\n',
    "\n",
    src,
    count=1,
)
path.write_text(updated)
PY
}

instrument_translated_loop() {
    local input="$1"
    python3 - "$input" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
src = path.read_text()
ctx_re = re.compile(
    r"(?P<indent>[ \t]*)__dacpp_mpi_or_[A-Za-z0-9_]+_ctx\s+__dacpp_mpi_or_ctx_[A-Za-z0-9_]+\s*;\n"
)
init_re = re.compile(
    r"(?P<indent>[ \t]*)__dacpp_mpi_or_[A-Za-z0-9_]+_init\s*\([^;]+;\n"
)
run_re = re.compile(
    r"(?P<indent>[ \t]*)__dacpp_mpi_or_[A-Za-z0-9_]+_run\s*\([^;]+;\n"
)

ctx = ctx_re.search(src)
if not ctx:
    raise SystemExit("failed to find translated ctx declaration")
init = init_re.search(src, ctx.end())
if not init:
    raise SystemExit("failed to find translated init call")
run = run_re.search(src, init.end())
if not run:
    raise SystemExit("failed to find translated run call")

insert_before_run = (
    f"{run.group('indent')}MPI_Barrier(MPI_COMM_WORLD);\n"
    f"{run.group('indent')}double __dacpp_bench_start = MPI_Wtime();\n"
)
src = src[:run.start()] + insert_before_run + src[run.start():]
run_end = run.end() + len(insert_before_run)
insert_after_run = (
    f"{run.group('indent')}MPI_Barrier(MPI_COMM_WORLD);\n"
    f"{run.group('indent')}double __dacpp_bench_local = MPI_Wtime() - __dacpp_bench_start;\n"
    f"{run.group('indent')}double __dacpp_bench_elapsed = 0.0;\n"
    f"{run.group('indent')}MPI_Reduce(&__dacpp_bench_local, &__dacpp_bench_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);\n"
    f"{run.group('indent')}if (mpi_rank == 0) std::cout << \"time_sec=\" << __dacpp_bench_elapsed << std::endl;\n"
)
src = src[:run_end] + insert_after_run + src[run_end:]
path.write_text(src)
PY
}

median() {
    python3 - "$@" <<'PY'
import sys
vals = sorted(float(x) for x in sys.argv[1:])
n = len(vals)
if n % 2:
    print(f"{vals[n//2]:.6f}")
else:
    print(f"{(vals[n//2-1] + vals[n//2]) / 2:.6f}")
PY
}

average() {
    python3 - "$@" <<'PY'
import sys
vals = [float(x) for x in sys.argv[1:]]
print(f"{sum(vals) / len(vals):.6f}")
PY
}

extract_time() {
    local log="$1"
    python3 - "$log" <<'PY'
from pathlib import Path
import re
import sys

text = Path(sys.argv[1]).read_text(errors="ignore")
matches = re.findall(r"time_sec=([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)", text)
if not matches:
    matches = re.findall(r"\[MPI_StandardSycl\]\[wave\] seconds=([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)", text)
if not matches:
    raise SystemExit("missing wave timing")
print(matches[-1])
PY
}

terminate_runner_tree() {
    local runner="$1"
    local bin="$2"

    if [[ -n "$runner" ]] && kill -0 "$runner" 2>/dev/null; then
        pkill -P "$runner" || true
        kill "$runner" || true
        sleep 1
    fi

    if [[ -n "$runner" ]]; then
        pkill -P "$runner" || true
        pkill -9 -P "$runner" || true
        kill -9 "$runner" || true
    fi

    [[ -n "$bin" ]] && pkill -f "$bin" || true
}

cleanup_active_runner() {
    if [[ -n "$ACTIVE_RUNNER" ]]; then
        terminate_runner_tree "$ACTIVE_RUNNER" "$ACTIVE_BIN"
        ACTIVE_RUNNER=""
        ACTIVE_BIN=""
    fi
}

trap cleanup_active_runner EXIT INT TERM

run_sample_once() {
    local label="$1"
    local bin="$2"
    local run="$3"
    local log="$BENCH_DIR/${label}_${run}.log"
    local t=""
    local deadline=$((SECONDS + RUN_TIMEOUT_SEC))

    rm -f "$log"
    pkill -f "$bin" || true

    DYLD_LIBRARY_PATH="$ACPP_ROOT/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
        mpirun -np "$MPI_RANKS" "$bin" > "$log" 2>&1 &
    ACTIVE_RUNNER="$!"
    ACTIVE_BIN="$bin"

    while true; do
        if [[ -f "$log" ]] && t="$(extract_time "$log" 2>/dev/null)"; then
            break
        fi
        if ! kill -0 "$ACTIVE_RUNNER" 2>/dev/null; then
            break
        fi
        if (( SECONDS >= deadline )); then
            echo "[FAIL] timeout waiting for $label run $run timing"
            [[ -f "$log" ]] && cat "$log"
            terminate_runner_tree "$ACTIVE_RUNNER" "$ACTIVE_BIN"
            ACTIVE_RUNNER=""
            ACTIVE_BIN=""
            exit 1
        fi
        sleep 1
    done

    if [[ -z "$t" ]]; then
        if [[ -f "$log" ]] && t="$(extract_time "$log" 2>/dev/null)"; then
            :
        else
            echo "[FAIL] missing timing for $label run $run"
            [[ -f "$log" ]] && cat "$log"
            terminate_runner_tree "$ACTIVE_RUNNER" "$ACTIVE_BIN"
            ACTIVE_RUNNER=""
            ACTIVE_BIN=""
            exit 1
        fi
    fi

    local grace_deadline=$((SECONDS + POST_TIME_GRACE_SEC))
    while kill -0 "$ACTIVE_RUNNER" 2>/dev/null && (( SECONDS < grace_deadline )); do
        sleep 1
    done

    if kill -0 "$ACTIVE_RUNNER" 2>/dev/null; then
        terminate_runner_tree "$ACTIVE_RUNNER" "$ACTIVE_BIN"
    fi

    wait "$ACTIVE_RUNNER" 2>/dev/null || true
    ACTIVE_RUNNER=""
    ACTIVE_BIN=""

    if [[ "$label" == "translated" ]]; then
        translated_times+=("$t")
    else
        coarse_times+=("$t")
    fi
    echo "  $label run $run: ${t}s"
}

DAC_SRC="$BENCH_DIR/wave.bench.dac.cpp"
MPI_DAC_SRC="$BENCH_DIR/wave.bench.mpi.dac.cpp"
TRANSLATED_CPP="$BENCH_DIR/wave.bench.mpi.dac_sycl_buffer.cpp"
TRANSLATED_BIN="$BENCH_DIR/translated_mpi_bin"
COARSE_SRC="$SCRIPT_DIR/tests/waveEquation1.0/waveEquation.MPI_StandardSycl.cpp"
COARSE_BENCH_SRC="$BENCH_DIR/waveEquation.MPI_StandardSycl.cpp"
COARSE_BIN="$BENCH_DIR/coarse_mpi_sycl_bin"

copy_and_scale_dac "$SCRIPT_DIR/tests/waveEquation1.0/waveEquation.dac.cpp" "$DAC_SRC"
cp "$DAC_SRC" "$MPI_DAC_SRC"
copy_and_scale_coarse "$COARSE_SRC" "$COARSE_BENCH_SRC"
disable_final_print "$DAC_SRC"
disable_final_print "$MPI_DAC_SRC"
disable_final_print "$COARSE_BENCH_SRC"
disable_coarse_result_dump "$COARSE_BENCH_SRC"

echo "[1/4] translate DACPP MPI wave"
dacpp "$MPI_DAC_SRC" --mode=buffer --mpi > "$BENCH_DIR/translate.log" 2>&1 || {
    cat "$BENCH_DIR/translate.log"
    exit 1
}
instrument_translated_loop "$TRANSLATED_CPP"

echo "[2/4] compile translated MPI wave"
acpp-compile "$TRANSLATED_CPP" "$TRANSLATED_BIN" > "$BENCH_DIR/translated_compile.log" 2>&1 || {
    cat "$BENCH_DIR/translated_compile.log"
    exit 1
}

echo "[3/4] compile coarse MPI+SYCL wave"
acpp-compile "$COARSE_BENCH_SRC" "$COARSE_BIN" > "$BENCH_DIR/coarse_compile.log" 2>&1 || {
    cat "$BENCH_DIR/coarse_compile.log"
    exit 1
}
chmod +x "$TRANSLATED_BIN" "$COARSE_BIN"

echo "[4/4] run benchmark: NX=$NX_VALUE NY=$NY_VALUE TIME_STEPS=$TIME_STEPS_VALUE np=$MPI_RANKS runs=$RUNS"
translated_times=()
coarse_times=()
for run in $(seq 1 "$RUNS"); do
    if (( run % 2 == 1 )); then
        run_sample_once coarse "$COARSE_BIN" "$run"
        run_sample_once translated "$TRANSLATED_BIN" "$run"
    else
        run_sample_once translated "$TRANSLATED_BIN" "$run"
        run_sample_once coarse "$COARSE_BIN" "$run"
    fi
done

translated_median="$(median "${translated_times[@]}")"
coarse_median="$(median "${coarse_times[@]}")"
translated_avg="$(average "${translated_times[@]}")"
coarse_avg="$(average "${coarse_times[@]}")"
avg_ratio="$(python3 - "$translated_avg" "$coarse_avg" <<'PY'
import sys
a = float(sys.argv[1])
b = float(sys.argv[2])
print(f"{a / b:.2f}" if b else "inf")
PY
)"

echo
echo "translated avg: ${translated_avg}s"
echo "coarse MPI+SYCL avg: ${coarse_avg}s"
echo "ratio translated/coarse (avg): ${avg_ratio}x"
echo "translated median: ${translated_median}s"
echo "coarse MPI+SYCL median: ${coarse_median}s"
echo "bench dir: $BENCH_DIR"
