#!/usr/bin/env python3
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TESTS_DIR = SCRIPT_DIR / "tests"
ENV_SH = SCRIPT_DIR / "env.sh"
TMP_DIR = Path(os.environ.get("MPI_ONLY_BENCH_TMP_DIR", "/Volumes/QUQ/working/mpi_tmp/mpi_only_requested"))
RANKS = int(os.environ.get("MPI_ONLY_BENCH_RANKS", "4"))
TIMEOUT = float(os.environ.get("MPI_ONLY_BENCH_TIMEOUT_SECONDS", "1800"))

DEFAULT_MATRIX_N = 2048
DEFAULT_TIME_STEPS = 300
DEFAULT_VECTOR_ELEMENTS = DEFAULT_MATRIX_N * DEFAULT_MATRIX_N

# Manually enlarged cases whose previous run had at least one side below 2s.
CASE_CONFIG = {
    "DFT1.0": {"n": 4096},
    "FOuLa1.0": {"n": 8192, "steps": 600},
    "MDP1.0": {"n": 8192, "steps": 600},
    "decay1.0": {"n": 8192, "steps": 600},
    "gradientSum": {"rows": 8192, "cols": 4096},
    "imageAdjustment1.0": {"n": 4096},
    "jacobi1.0": {"n": 4096, "steps": 300},
    "liuliang1.0": {"n": 8192, "steps": 1000},
    "mandel1.0": {"n": 4096},
    "oddeven0.1": {"n": 4096},
    "stencil1.0": {"n": 2048, "steps": 600},
    "vectorAddCombo": {"elements": 8388608},
    "waveEquation1.0": {"n": 2048, "steps": 600},
}


def case_n(case):
    return int(CASE_CONFIG.get(case, {}).get("n", DEFAULT_MATRIX_N))


def case_rows(case):
    return int(CASE_CONFIG.get(case, {}).get("rows", case_n(case)))


def case_cols(case):
    return int(CASE_CONFIG.get(case, {}).get("cols", case_n(case)))


def case_steps(case):
    return int(CASE_CONFIG.get(case, {}).get("steps", DEFAULT_TIME_STEPS))


def case_elements(case):
    return int(CASE_CONFIG.get(case, {}).get("elements", case_rows(case) * case_cols(case)))


def scale_label(case):
    if case == "DFT1.0":
        return f"N={case_n(case)}"
    if case == "FOuLa1.0":
        return f"m={case_n(case)}, n={case_steps(case)}"
    if case == "MDP1.0":
        return f"N={case_n(case)}, T={case_steps(case)}"
    if case == "decay1.0":
        return f"numIsotopes={case_n(case)}, steps={case_steps(case)}"
    if case == "gradientSum":
        return f"{case_rows(case)}x{case_cols(case)}"
    if case == "jacobi1.0":
        return f"N={case_n(case)}, iter={case_steps(case)}"
    if case == "liuliang1.0":
        return f"WIDTH={case_n(case)}, steps={case_steps(case)}"
    if case == "mandel1.0":
        return f"{case_n(case)}x{case_n(case)}, max_iter=1000"
    if case == "matMul1.0":
        return f"{case_n(case)}x{case_n(case)}"
    if case == "oddeven0.1":
        return f"N={case_n(case)}"
    if case in {"stencil1.0", "waveEquation1.0"}:
        return f"{case_n(case)}x{case_n(case)}, steps={case_steps(case)}"
    if case == "vectorAddCombo":
        return f"N={case_elements(case)}"
    return f"{case_n(case)}x{case_n(case)}"


CASES = [
    ("DFT1.0", "DFT.large_dac.cpp", "DFT.MPI_StandardSycl.cpp", scale_label("DFT1.0")),
    ("FOuLa1.0", "FOuLa.large_dac.cpp", "FOuLa.MPI_StandardSycl.cpp", scale_label("FOuLa1.0")),
    ("MDP1.0", "mdp.large_dac.cpp", "mdp.MPI_StandardSycl.cpp", scale_label("MDP1.0")),
    ("decay1.0", "decay_chain.large_dac.cpp", "decay_chain.MPI_StandardSycl.cpp", scale_label("decay1.0")),
    ("gradientSum", "gradientSum.large_dac.cpp", "gradientSum.MPI_StandardSycl.cpp", scale_label("gradientSum")),
    ("imageAdjustment1.0", "imageAdjustment.large_dac.cpp", "imageAdjustment.MPI_StandardSycl.cpp", scale_label("imageAdjustment1.0")),
    ("jacobi1.0", "jacobi.large_dac.cpp", "jacobi.MPI_StandardSycl.cpp", scale_label("jacobi1.0")),
    ("liuliang1.0", "liuliang.large_dac.cpp", "liuliang.MPI_StandardSycl.cpp", scale_label("liuliang1.0")),
    ("mandel1.0", "mandel.large_dac.cpp", "mandel.MPI_StandardSycl.cpp", scale_label("mandel1.0")),
    ("matMul1.0", "matMul.large_dac.cpp", "matMul.MPI_StandardSycl.cpp", scale_label("matMul1.0")),
    ("oddeven0.1", "oddEven.large_dac.cpp", "oddEven.MPI_StandardSycl.cpp", scale_label("oddeven0.1")),
    ("stencil1.0", "stencil.large_dac.cpp", "stencil.MPI_StandardSycl.cpp", scale_label("stencil1.0")),
    ("vectorAddCombo", "vectorAddCombo.large_dac.cpp", "vectorAddCombo.MPI_StandardSycl.cpp", scale_label("vectorAddCombo")),
    ("waveEquation1.0", "waveEquation.large_dac.cpp", "waveEquation.MPI_StandardSycl.cpp", scale_label("waveEquation1.0")),
]


def q(path):
    return shlex.quote(str(path))


def run_bash(command, log_path, timeout=None):
    full = f"source {q(ENV_SH)} && {command}"
    with open(log_path, "wb") as log:
        return subprocess.run(
            ["bash", "-lc", full],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )


def timed_mpirun(binary, log_path):
    full = f"source {q(ENV_SH)} && mpirun -np {RANKS} {q(binary)}"
    start = time.perf_counter()
    with open(log_path, "wb") as log:
        try:
            proc = subprocess.run(
                ["bash", "-lc", full],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - start
            log.write(f"\n[TIMEOUT] exceeded {TIMEOUT:.1f}s\n".encode())
            return None, "timeout"
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        return elapsed, f"run-failed-{proc.returncode}"
    return elapsed, "ok"


def write(path, text):
    path.write_text(text, encoding="utf-8")


def replace(pattern, repl, text, flags=0):
    new_text, count = re.subn(pattern, repl, text, flags=flags)
    if count == 0:
        raise RuntimeError(f"pattern not found: {pattern}")
    return new_text


def patch_dac(case, text):
    if case == "DFT1.0":
        text = replace(r"const int N = \d+;", f"const int N = {case_n(case)};", text)
        text = text.replace("output_tensor.print();", "std::cout << output_tensor[0] << std::endl;")
    elif case == "FOuLa1.0":
        text = replace(r"int n = \d+;", f"int n = {case_steps(case)};", text)
        text = replace(r"int m = \d+;", f"int m = {case_n(case)};", text)
        text = text.replace("u_tensor[1].print();", "std::cout << u_tensor[1][0] << std::endl;")
    elif case == "MDP1.0":
        text = replace(r"const int N = \d+;", f"const int N = {case_n(case)};", text)
        text = replace(r"const int T = \d+;", f"const int T = {case_steps(case)};", text)
    elif case == "decay1.0":
        text = replace(r"const double dt = [0-9.]+;", "const double dt = 0.1;", text)
        text = replace(r"const double T = [0-9.]+;", f"const double T = {case_steps(case) / 10.0:.1f};", text)
        text = text.replace("while(t_tensor[0] <= T)", "while(t_tensor[0] < T)")
        text = replace(r"const size_t numIsotopes = \d+;", f"const size_t numIsotopes = {case_n(case)};", text)
        text = text.replace("A_tensor[1].print();", "std::cout << A_tensor[1][0] << std::endl;")
    elif case == "gradientSum":
        text = replace(r"const int NUM_NEURONS = \d+;", f"const int NUM_NEURONS = {case_rows(case)};", text)
        text = replace(r"const int INPUT_SIZE\s+= \d+;", f"const int INPUT_SIZE  = {case_cols(case)};", text)
    elif case == "imageAdjustment1.0":
        text = replace(r"const int width = \d+;", f"const int width = {case_n(case)};", text)
        text = replace(r"const int height = \d+;", f"const int height = {case_n(case)};", text)
        text = text.replace("image_tensor3.print();", "std::cout << image_tensor3[0][0] << std::endl;")
    elif case == "jacobi1.0":
        text = replace(r"const int N = \d+;", f"const int N = {case_n(case)};", text)
        text = replace(r"const int max_iter = \d+;", f"const int max_iter = {case_steps(case)};", text)
    elif case == "liuliang1.0":
        text = replace(r"const int WIDTH = \d+;", f"const int WIDTH = {case_n(case)};", text)
        text = replace(r"const double TIME_STEPS = \d+;", f"const double TIME_STEPS = {case_steps(case)};", text)
    elif case == "mandel1.0":
        text = replace(r"const int row_count = \d+, col_count = \d+, max_iterations = \d+;",
                       f"const int row_count = {case_n(case)}, col_count = {case_n(case)}, max_iterations = 1000;", text)
    elif case == "matMul1.0":
        text = text.replace("using namespace std;\n", f"using namespace std;\n\nconst int MAT_M = {case_n(case)};\nconst int MAT_K = {case_n(case)};\nconst int MAT_N = {case_n(case)};\n")
        text = replace(r"for \(int i = 0; i < 5; i\+\+\)", "for (int i = 0; i < MAT_K; i++)", text)
        text = replace(
            r"std::vector<int> dataA\{.*?\};\s*dacpp::Matrix<int> matA\(\{4, 5\}, dataA\);",
            "std::vector<int> dataA(MAT_M * MAT_K);\n"
            "    for (int i = 0; i < MAT_M; ++i) for (int k = 0; k < MAT_K; ++k) dataA[i * MAT_K + k] = (i + k) % 97;\n"
            "    dacpp::Matrix<int> matA({MAT_M, MAT_K}, dataA);",
            text,
            flags=re.S,
        )
        text = replace(
            r"std::vector<int> dataB\{.*?\};\s*dacpp::Matrix<int> matB\(\{5, 4\}, dataB\);",
            "std::vector<int> dataB(MAT_K * MAT_N);\n"
            "    for (int k = 0; k < MAT_K; ++k) for (int j = 0; j < MAT_N; ++j) dataB[k * MAT_N + j] = (k + j) % 89;\n"
            "    dacpp::Matrix<int> matB({MAT_K, MAT_N}, dataB);",
            text,
            flags=re.S,
        )
        text = replace(
            r"std::vector<int> dataC\{.*?\};\s*dacpp::Matrix<int> matC\(\{4, 4\}, dataC\);",
            "std::vector<int> dataC(MAT_M * MAT_N, 0);\n"
            "    dacpp::Matrix<int> matC({MAT_M, MAT_N}, dataC);",
            text,
            flags=re.S,
        )
        text = text.replace("matC.print();", "std::cout << matC[0][0] << std::endl;")
    elif case == "oddeven0.1":
        text = replace(r"const int N = \d+;", f"const int N = {case_n(case)};", text)
        text = text.replace("array_tensor.print();", "std::cout << array_tensor[0] << std::endl;")
    elif case == "stencil1.0":
        text = replace(r"const int NX = \d+;", f"const int NX = {case_n(case)};", text)
        text = replace(r"const int NY = \d+;", f"const int NY = {case_n(case)};", text)
        text = replace(r"const int TIME_STEPS = \d+;", f"const int TIME_STEPS = {case_steps(case)};", text)
        text = text.replace("matIn[0].print();", "std::cout << matIn[0][0] << std::endl;")
    elif case == "vectorAddCombo":
        text = replace(r"const int N = \d+;", f"const int N = {case_elements(case)};", text)
    elif case == "waveEquation1.0":
        text = replace(r"const int NX = \d+;", f"const int NX = {case_n(case)};", text)
        text = replace(r"const int NY = \d+;", f"const int NY = {case_n(case)};", text)
        text = replace(r"const int TIME_STEPS = \d+;", f"const int TIME_STEPS = {case_steps(case)};", text)
        text = text.replace("matCur.print();", "std::cout << matCur[0][0] << std::endl;")
    return text


def patch_standard(case, text):
    if case == "DFT1.0":
        text = replace(r"#define DFT_N \d+", f"#define DFT_N {case_n(case)}", text)
    elif case == "FOuLa1.0":
        text = replace(r"const int n = \d+;", f"const int n = {case_steps(case)};", text)
        text = replace(r"const int m = \d+;", f"const int m = {case_n(case)};", text)
    elif case == "MDP1.0":
        text = replace(r"#define MDP_N \d+", f"#define MDP_N {case_n(case)}", text)
        text = replace(r"#define MDP_T \d+", f"#define MDP_T {case_steps(case)}", text)
    elif case == "decay1.0":
        text = replace(r"const double dt = [0-9.]+;", "const double dt = 0.1;", text)
        text = replace(r"const double T = [0-9.]+;", f"const double T = {case_steps(case) / 10.0:.1f};", text)
        text = replace(r"const size_t numIsotopes = \d+;", f"const size_t numIsotopes = {case_n(case)};", text)
        text = replace(r"while \(t <= T\)", "while (t < T)", text)
    elif case == "gradientSum":
        text = text.replace("using namespace sycl;", "namespace sycl = cl::sycl;\nusing namespace cl::sycl;")
        text = replace(r"constexpr size_t NUM_NEURONS = \d+;", f"constexpr size_t NUM_NEURONS = {case_rows(case)};", text)
        text = replace(r"constexpr size_t INPUT_SIZE\s+= \d+;", f"constexpr size_t INPUT_SIZE  = {case_cols(case)};", text)
    elif case == "imageAdjustment1.0":
        text = replace(r"constexpr int width = \d+;", f"constexpr int width = {case_n(case)};", text)
        text = replace(r"constexpr int height = \d+;", f"constexpr int height = {case_n(case)};", text)
        text = replace(
            r"// Print as 2D grid\s+for \(int i = 0; i < height; \+\+i\) \{.*?\n        \}",
            "std::cout << global_image[0] << std::endl;",
            text,
            flags=re.S,
        )
    elif case == "jacobi1.0":
        text = replace(r"#define JACOBI_N \d+", f"#define JACOBI_N {case_n(case)}", text)
        text = replace(r"#define JACOBI_MAX_ITER \d+", f"#define JACOBI_MAX_ITER {case_steps(case)}", text)
    elif case == "liuliang1.0":
        text = replace(r"#define LIULIANG_WIDTH \d+", f"#define LIULIANG_WIDTH {case_n(case)}", text)
        text = replace(r"#define LIULIANG_STEPS \d+", f"#define LIULIANG_STEPS {case_steps(case)}", text)
    elif case == "mandel1.0":
        text = replace(r"#define MANDEL_N \d+", f"#define MANDEL_N {case_n(case)}", text)
    elif case == "matMul1.0":
        text = replace(r"constexpr int M = \d+, K = \d+, N = \d+;", f"constexpr int M = {case_n(case)}, K = {case_n(case)}, N = {case_n(case)};", text)
        text = replace(
            r"std::vector<int> dataA\{.*?\};",
            "std::vector<int> dataA(M * K);\n"
            "    for (int i = 0; i < M; ++i) for (int k = 0; k < K; ++k) dataA[i * K + k] = (i + k) % 97;",
            text,
            flags=re.S,
        )
        text = replace(
            r"std::vector<int> dataB\{.*?\};",
            "std::vector<int> dataB(K * N);\n"
            "    for (int k = 0; k < K; ++k) for (int j = 0; j < N; ++j) dataB[k * N + j] = (k + j) % 89;",
            text,
            flags=re.S,
        )
        text = replace(
            r"if \(rank == 0\) \{\s+std::cout << \"\{\";.*?\n    \}",
            "if (rank == 0) {\n        std::cout << global_result[0] << std::endl;\n    }",
            text,
            flags=re.S,
        )
    elif case == "oddeven0.1":
        text = replace(r"#define ODDEVEN_N \d+", f"#define ODDEVEN_N {case_n(case)}", text)
    elif case == "stencil1.0":
        text = replace(r"#define STENCIL_NX \d+", f"#define STENCIL_NX {case_n(case)}", text)
        text = replace(r"#define STENCIL_NY \d+", f"#define STENCIL_NY {case_n(case)}", text)
        text = replace(r"#define STENCIL_TIME_STEPS \d+", f"#define STENCIL_TIME_STEPS {case_steps(case)}", text)
    elif case == "vectorAddCombo":
        text = replace(r"#define VADD_N \d+", f"#define VADD_N {case_elements(case)}", text)
        text = replace(
            r"std::vector<float> a\{.*?\};\s+std::vector<float> b\{.*?\};\s+std::vector<float> c\{.*?\};",
            "std::vector<float> a(N), b(N), c(N);\n"
            "    for (int i = 0; i < N; ++i) {\n"
            "        a[i] = static_cast<float>(i);\n"
            "        b[i] = static_cast<float>(i * 2);\n"
            "        c[i] = static_cast<float>(1000 + i);\n"
            "    }",
            text,
            flags=re.S,
        )
        text = replace(
            r"if \(rank == 0\) \{\s+for \(float v : global_out\) \{.*?\n    \}",
            "if (rank == 0) {\n        std::cout << global_out[0] << std::endl;\n    }",
            text,
            flags=re.S,
        )
    elif case == "waveEquation1.0":
        text = replace(r"#define WAVE_NX \d+", f"#define WAVE_NX {case_n(case)}", text)
        text = replace(r"#define WAVE_NY \d+", f"#define WAVE_NY {case_n(case)}", text)
        text = replace(r"#define WAVE_TIME_STEPS \d+", f"#define WAVE_TIME_STEPS {case_steps(case)}", text)
        text = replace(
            r"if \(rank == 0\) \{\s+std::cerr << \"\[MPI_StandardSycl\]\[wave\] seconds=\" << max_seconds.*?\n    \}",
            "if (rank == 0) {\n        std::cerr << \"[MPI_StandardSycl][wave] seconds=\" << max_seconds << std::endl;\n        std::cout << global_out[0] << std::endl;\n    }",
            text,
            flags=re.S,
        )
    return text


def generated_path_for(src):
    return src.with_name(src.name[:-4] + "_sycl_buffer.cpp")


def main():
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    results = TMP_DIR / "results.tsv"
    selected = set(sys.argv[1:])
    cases = [case for case in CASES if not selected or case[0] in selected]
    if not results.exists() or not selected:
        write(results, "test\tscale\tstandard_mpi_s\tdac_translated_mpi_s\tstatus\n")

    print(f"tmp={TMP_DIR}")
    print(f"np={RANKS} default_matrix={DEFAULT_MATRIX_N} default_time_steps={DEFAULT_TIME_STEPS} timeout={TIMEOUT:.0f}s")

    for case, dac_name, std_name, scale in cases:
        print("=" * 70, flush=True)
        print(f"case: {case} scale: {scale}", flush=True)
        work = TMP_DIR / case
        work.mkdir(parents=True, exist_ok=True)

        dac_src = TESTS_DIR / case / dac_name
        std_src = TESTS_DIR / case / std_name
        dac_work = work / "case.mpi.dac.cpp"
        std_work = work / "case.MPI_StandardSycl.cpp"
        std_bin = work / "standard_bin"
        dac_bin = work / "dac_mpi_bin"

        try:
            write(dac_work, patch_dac(case, dac_src.read_text(encoding="utf-8")))
            write(std_work, patch_standard(case, std_src.read_text(encoding="utf-8")))
        except Exception as exc:
            status = f"patch-failed:{exc}"
            print(status, flush=True)
            with open(results, "a", encoding="utf-8") as out:
                out.write(f"{case}\t{scale}\t-\t-\t{status}\n")
            continue

        std_status = "ok"
        dac_status = "ok"

        print("  build standard MPI+SYCL", flush=True)
        try:
            proc = run_bash(f"acpp-compile {q(std_work)} {q(std_bin)}", work / "build_standard.log", timeout=TIMEOUT)
            if proc.returncode != 0:
                std_status = f"build-standard-failed-{proc.returncode}"
        except subprocess.TimeoutExpired:
            std_status = "build-standard-timeout"

        print("  translate/build DAC MPI", flush=True)
        if std_status == "ok":
            try:
                proc = run_bash(f"dacpp {q(dac_work)} --mode=buffer --mpi", work / "translate_dac.log", timeout=TIMEOUT)
                if proc.returncode != 0:
                    dac_status = f"translate-dac-failed-{proc.returncode}"
                else:
                    gen = generated_path_for(dac_work)
                    proc = run_bash(f"acpp-compile {q(gen)} {q(dac_bin)}", work / "build_dac.log", timeout=TIMEOUT)
                    if proc.returncode != 0:
                        dac_status = f"build-dac-failed-{proc.returncode}"
            except subprocess.TimeoutExpired:
                dac_status = "translate-or-build-dac-timeout"

        std_time = None
        dac_time = None
        if std_status == "ok" and dac_status == "ok":
            print("  run standard MPI+SYCL", flush=True)
            std_time, std_status = timed_mpirun(std_bin, work / "run_standard.log")
            print(f"    standard: {std_status} {std_time if std_time is not None else '-'}", flush=True)
            print("  run DAC translated MPI", flush=True)
            dac_time, dac_status = timed_mpirun(dac_bin, work / "run_dac.log")
            print(f"    dac: {dac_status} {dac_time if dac_time is not None else '-'}", flush=True)

        status = "ok" if std_status == "ok" and dac_status == "ok" else f"standard={std_status};dac={dac_status}"
        std_str = "-" if std_time is None else f"{std_time:.6f}"
        dac_str = "-" if dac_time is None else f"{dac_time:.6f}"
        with open(results, "a", encoding="utf-8") as out:
            out.write(f"{case}\t{scale}\t{std_str}\t{dac_str}\t{status}\n")

    print("=" * 70)
    print(f"results={results}")
    print(results.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
