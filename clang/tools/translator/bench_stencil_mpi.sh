#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
source "$SCRIPT_DIR/env.sh"

BENCH_DIR="/Volumes/QUQ/working/mpi_tmp/stencil1_bench"
NX_VALUE="${STENCIL_BENCH_NX:-1024}"
NY_VALUE="${STENCIL_BENCH_NY:-1024}"
TIME_STEPS_VALUE="${STENCIL_BENCH_TIME_STEPS:-200}"
MPI_RANKS="${STENCIL_BENCH_RANKS:-4}"
RUNS="${STENCIL_BENCH_RUNS:-3}"

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
src = re.sub(r"#define STENCIL_NX\s+\d+", f"#define STENCIL_NX {nx}", src, count=1)
src = re.sub(r"#define STENCIL_NY\s+\d+", f"#define STENCIL_NY {ny}", src, count=1)
src = re.sub(r"#define STENCIL_TIME_STEPS\s+\d+", f"#define STENCIL_TIME_STEPS {steps}", src, count=1)
Path(out).write_text(src)
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

extract_time() {
    local log="$1"
    python3 - "$log" <<'PY'
from pathlib import Path
import re
import sys
text = Path(sys.argv[1]).read_text(errors="ignore")
matches = re.findall(r"time_sec=([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)", text)
if matches:
    print(matches[-1])
    raise SystemExit(0)
matches = re.findall(r"\[MPI_StandardSycl\]\[stencil\]\[(?:naive|row-halo)\] seconds=([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)", text)
if matches:
    print(matches[-1])
    raise SystemExit(0)
raise SystemExit("missing time_sec")
PY
}

run_samples() {
    local label="$1"
    local bin="$2"
    for run in $(seq 1 "$RUNS"); do
        local log="$BENCH_DIR/${label}_${run}.log"
        DYLD_LIBRARY_PATH="$ACPP_ROOT/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
            mpirun -np "$MPI_RANKS" "$bin" > "$log" 2>&1
        local t
        t="$(extract_time "$log")" || {
            echo "[FAIL] missing timing for $label run $run"
            cat "$log"
            exit 1
        }
        if [[ "$label" == "translated" ]]; then
            translated_times+=("$t")
        else
            coarse_times+=("$t")
        fi
        echo "  $label run $run: ${t}s"
    done
}

DAC_SRC="$BENCH_DIR/stencil.bench.dac.cpp"
MPI_DAC_SRC="$BENCH_DIR/stencil.bench.mpi.dac.cpp"
TRANSLATED_CPP="$BENCH_DIR/stencil.bench.mpi.dac_sycl_buffer.cpp"
TRANSLATED_BIN="$BENCH_DIR/translated_mpi_bin"
COARSE_SRC="$SCRIPT_DIR/tests/stencil1.0/stencil.MPI_StandardSycl.cpp"
COARSE_BENCH_SRC="$BENCH_DIR/stencil.MPI_StandardSycl.cpp"
COARSE_BIN="$BENCH_DIR/coarse_mpi_sycl_bin"

copy_and_scale_dac "$SCRIPT_DIR/tests/stencil1.0/stencil.dac.cpp" "$DAC_SRC"
cp "$DAC_SRC" "$MPI_DAC_SRC"
copy_and_scale_coarse "$COARSE_SRC" "$COARSE_BENCH_SRC"

echo "[1/4] translate DACPP MPI stencil"
dacpp "$MPI_DAC_SRC" --mode=buffer --mpi > "$BENCH_DIR/translate.log" 2>&1 || {
    cat "$BENCH_DIR/translate.log"
    exit 1
}
instrument_translated_loop "$TRANSLATED_CPP"

echo "[2/4] compile translated MPI stencil"
acpp-compile "$TRANSLATED_CPP" "$TRANSLATED_BIN" > "$BENCH_DIR/translated_compile.log" 2>&1 || {
    cat "$BENCH_DIR/translated_compile.log"
    exit 1
}

echo "[3/4] compile coarse MPI+SYCL stencil"
acpp-compile "$COARSE_BENCH_SRC" "$COARSE_BIN" > "$BENCH_DIR/coarse_compile.log" 2>&1 || {
    cat "$BENCH_DIR/coarse_compile.log"
    exit 1
}

echo "[4/4] run benchmark: NX=$NX_VALUE NY=$NY_VALUE TIME_STEPS=$TIME_STEPS_VALUE np=$MPI_RANKS runs=$RUNS"
translated_times=()
coarse_times=()
run_samples translated "$TRANSLATED_BIN"
run_samples coarse "$COARSE_BIN"

translated_median="$(median "${translated_times[@]}")"
coarse_median="$(median "${coarse_times[@]}")"
ratio="$(python3 - "$translated_median" "$coarse_median" <<'PY'
import sys
a = float(sys.argv[1])
b = float(sys.argv[2])
print(f"{a / b:.2f}" if b else "inf")
PY
)"

echo
echo "translated median: ${translated_median}s"
echo "coarse MPI+SYCL median: ${coarse_median}s"
echo "ratio translated/coarse: ${ratio}x"
echo "bench dir: $BENCH_DIR"
