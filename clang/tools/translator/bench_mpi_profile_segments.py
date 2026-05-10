#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import bench_mpi_only_requested as bench


SCRIPT_DIR = Path(__file__).resolve().parent
ENV_SH = SCRIPT_DIR / "env.sh"
DEFAULT_TMP = Path(
    os.environ.get(
        "MPI_PROFILE_BENCH_TMP_DIR",
        str(Path("/tmp") / "dacpp_mpi_profile_phase1_4"),
    )
)
DEFAULT_RANKS = os.environ.get("MPI_PROFILE_BENCH_RANKS", "4")
TIMEOUT = float(os.environ.get("MPI_PROFILE_BENCH_TIMEOUT_SECONDS", "1800"))
DEFAULT_TRIALS = int(os.environ.get("MPI_PROFILE_BENCH_TRIALS", "3"))

PROFILE_RE = re.compile(
    r"^DACPP_MPI_PROFILE\t(?P<label>[^\t]+)\t(?P<segment>[^\t]+)"
    r"\tcalls=(?P<calls>\d+)\tmax_ms=(?P<max>[0-9.]+)"
    r"\tsum_ms=(?P<sum>[0-9.]+)"
)
COLLECT_RE = re.compile(
    r"^\[DACPP\]\[PROFILE\]\[(?P<label>[^\]]+)\] "
    r"collect_positions_for_item (?P<metric>[^:]+): (?P<value>.*)$"
)
ADAPTIVECPP_FILTERS = (
    "AdaptiveCpp Warning",
    "This application uses SYCL buffers",
    "SYCL2020 USM model",
    "AdaptiveCpp performance guide",
    "https://github.com/AdaptiveCpp",
)


def q(path):
    return bench.q(path)


def run_bash(command, log_path, timeout=None):
    full = f"source {q(ENV_SH)} && {command}"
    with open(log_path, "wb") as log:
        return subprocess.run(
            ["bash", "-lc", full],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )


def run_mpirun(binary, ranks, env, stdout_path, stderr_path):
    full = f"source {q(ENV_SH)} && mpirun -np {ranks} {q(binary)}"
    run_env = os.environ.copy()
    run_env.update(env)
    start = time.perf_counter()
    try:
        with open(stdout_path, "wb") as stdout, open(stderr_path, "wb") as stderr:
            proc = subprocess.run(
                ["bash", "-lc", full],
                stdout=stdout,
                stderr=stderr,
                timeout=TIMEOUT,
                env=run_env,
            )
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start
        with open(stderr_path, "ab") as stderr:
            stderr.write(f"\n[TIMEOUT] exceeded {TIMEOUT:.1f}s\n".encode())
        return elapsed, "timeout"

    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        return elapsed, f"run-failed-{proc.returncode}"
    return elapsed, "ok"


def write_text(path, text):
    path.write_text(text, encoding="utf-8")


def clean_output_text(path):
    if not path.exists():
        return ""
    lines = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if any(marker in raw for marker in ADAPTIVECPP_FILTERS):
            continue
        if not raw.strip():
            continue
        lines.append(raw.rstrip())
    return "\n".join(lines)


def median_or_dash(values):
    ok = [value for value in values if value is not None]
    if not ok:
        return "-"
    return f"{statistics.median(ok):.6f}"


def mean_or_dash(values):
    ok = [value for value in values if value is not None]
    if not ok:
        return "-"
    return f"{statistics.mean(ok):.6f}"


def seconds_or_dash(value):
    if value is None:
        return "-"
    return f"{value:.6f}"


def status_join(statuses):
    bad = [status for status in statuses if status != "ok"]
    return "ok" if not bad else ";".join(bad)


def parse_ranks(value):
    ranks = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        rank = int(item)
        if rank <= 0:
            raise ValueError("rank counts must be positive")
        ranks.append(rank)
    if not ranks:
        raise ValueError("at least one rank count is required")
    return ranks


def selected_cases(names):
    selected = set(names)
    return [case for case in bench.CASES if not selected or case[0] in selected]


def generated_path_for(src):
    return bench.generated_path_for(src)


def git_output(args):
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=SCRIPT_DIR,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return ""
    return proc.stdout.strip()


def write_git_artifacts(tmp_dir):
    repo_root = SCRIPT_DIR.parent.parent.parent
    artifacts = (
        ("git_status.txt", ["status", "--short"]),
        ("git_diff_stat.txt", ["diff", "--stat", "--", "clang/tools/translator"]),
        ("git_diff.patch", ["diff", "--", "clang/tools/translator"]),
        ("git_untracked.txt", ["ls-files", "--others", "--exclude-standard", "--", "clang/tools/translator"]),
    )
    for name, args in artifacts:
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            write_text(tmp_dir / name, proc.stdout)
        except OSError as exc:
            write_text(tmp_dir / name, f"git artifact failed: {exc}\n")

    snapshots = tmp_dir / "untracked_snapshots"
    try:
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "--", "clang/tools/translator"],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        ).stdout.splitlines()
        for rel in untracked:
            src = repo_root / rel
            if not src.is_file():
                continue
            dst = snapshots / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    except OSError as exc:
        snapshots.mkdir(parents=True, exist_ok=True)
        write_text(snapshots / "ERROR.txt", f"untracked snapshot failed: {exc}\n")


def append_metadata(tmp_dir, args, ranks, cases):
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "tmp_dir": str(tmp_dir),
        "timeout_seconds": TIMEOUT,
        "trials": args.trials,
        "ranks": ranks,
        "cases": [case[0] for case in cases],
        "git_branch": git_output(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_head": git_output(["rev-parse", "HEAD"]),
        "git_status_short": git_output(["status", "--short"]),
        "command": " ".join(sys.argv),
    }
    write_text(tmp_dir / "metadata.json", json.dumps(metadata, indent=2) + "\n")
    write_git_artifacts(tmp_dir)


def ensure_headers(tmp_dir):
    write_text(
        tmp_dir / "results.tsv",
        "case\tscale\tranks\ttrial\tstandard_s\tdac_wall_s\tdac_profile_s"
        "\tprofile_stdout_matches_wall\tstatus\n",
    )
    write_text(
        tmp_dir / "summary.tsv",
        "case\tscale\tranks\tstandard_median_s\tstandard_mean_s"
        "\tdac_wall_median_s\tdac_wall_mean_s\tdac_profile_median_s"
        "\tdac_profile_mean_s\tprofile_stdout_matches_wall\tstatus\n",
    )
    write_text(
        tmp_dir / "profile_raw.tsv",
        "case\tscale\tranks\ttrial\tlabel\tsegment\tcalls\tmax_ms\tsum_ms\n",
    )
    write_text(
        tmp_dir / "profile_summary.tsv",
        "case\tscale\tranks\tlabel\tsegment\tmedian_calls\tmedian_max_ms"
        "\tmedian_sum_ms\ttrials\n",
    )
    write_text(
        tmp_dir / "collect_positions_profile.tsv",
        "case\tscale\tranks\ttrial\tlabel\tmetric\tvalue\n",
    )


def append(path, line):
    with open(path, "a", encoding="utf-8") as out:
        out.write(line)


def parse_profile_log(case, scale, ranks, trial, stderr_path, tmp_dir):
    profile_rows = []
    collect_rows = []
    for line in stderr_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = PROFILE_RE.match(line)
        if match:
            row = {
                "case": case,
                "scale": scale,
                "ranks": ranks,
                "trial": trial,
                "label": match.group("label"),
                "segment": match.group("segment"),
                "calls": int(match.group("calls")),
                "max_ms": float(match.group("max")),
                "sum_ms": float(match.group("sum")),
            }
            profile_rows.append(row)
            append(
                tmp_dir / "profile_raw.tsv",
                f"{case}\t{scale}\t{ranks}\t{trial}\t{row['label']}"
                f"\t{row['segment']}\t{row['calls']}\t{row['max_ms']:.6f}"
                f"\t{row['sum_ms']:.6f}\n",
            )
            continue
        match = COLLECT_RE.match(line)
        if match:
            collect_rows.append(
                {
                    "case": case,
                    "scale": scale,
                    "ranks": ranks,
                    "trial": trial,
                    "label": match.group("label"),
                    "metric": match.group("metric"),
                    "value": match.group("value"),
                }
            )
            append(
                tmp_dir / "collect_positions_profile.tsv",
                f"{case}\t{scale}\t{ranks}\t{trial}\t{match.group('label')}"
                f"\t{match.group('metric')}\t{match.group('value')}\n",
            )
    return profile_rows, collect_rows


def summarize_profiles(tmp_dir, rows):
    grouped = {}
    for row in rows:
        key = (row["case"], row["scale"], row["ranks"], row["label"], row["segment"])
        grouped.setdefault(key, []).append(row)
    for key in sorted(grouped):
        case, scale, ranks, label, segment = key
        values = grouped[key]
        calls = statistics.median([row["calls"] for row in values])
        max_ms = statistics.median([row["max_ms"] for row in values])
        sum_ms = statistics.median([row["sum_ms"] for row in values])
        append(
            tmp_dir / "profile_summary.tsv",
            f"{case}\t{scale}\t{ranks}\t{label}\t{segment}"
            f"\t{calls:.0f}\t{max_ms:.6f}\t{sum_ms:.6f}\t{len(values)}\n",
        )


def copy_snapshot(src, dst):
    if src.exists() and src.resolve() != dst.resolve():
        shutil.copy2(src, dst)


def prepare_case(case, dac_name, std_name, work):
    dac_src = bench.TESTS_DIR / case / dac_name
    std_src = bench.TESTS_DIR / case / std_name
    dac_work = work / "case.mpi.dac.cpp"
    std_work = work / "case.MPI_StandardSycl.cpp"
    write_text(dac_work, bench.patch_dac(case, dac_src.read_text(encoding="utf-8")))
    write_text(std_work, bench.patch_standard(case, std_src.read_text(encoding="utf-8")))
    return dac_work, std_work


def build_case(case, dac_work, std_work, work):
    std_bin = work / "standard_bin"
    dac_bin = work / "dac_mpi_bin"
    std_proc = run_bash(
        f"acpp-compile {q(std_work)} {q(std_bin)}",
        work / "build_standard.log",
        timeout=TIMEOUT,
    )
    if std_proc.returncode != 0:
        return std_bin, dac_bin, f"build-standard-failed-{std_proc.returncode}"

    translate_proc = run_bash(
        f"dacpp {q(dac_work)} --mode=buffer --mpi",
        work / "translate_dac.log",
        timeout=TIMEOUT,
    )
    if translate_proc.returncode != 0:
        return std_bin, dac_bin, f"translate-dac-failed-{translate_proc.returncode}"

    gen = generated_path_for(dac_work)
    copy_snapshot(gen, work / "case.mpi.dac_sycl_buffer.cpp")
    build_proc = run_bash(
        f"acpp-compile {q(gen)} {q(dac_bin)}",
        work / "build_dac.log",
        timeout=TIMEOUT,
    )
    if build_proc.returncode != 0:
        return std_bin, dac_bin, f"build-dac-failed-{build_proc.returncode}"
    return std_bin, dac_bin, "ok"


def run_case(tmp_dir, case_tuple, ranks_list, trials):
    case, dac_name, std_name, scale = case_tuple
    case_dir = tmp_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)

    try:
        dac_work, std_work = prepare_case(case, dac_name, std_name, case_dir)
    except Exception as exc:
        status = f"patch-failed:{exc}"
        for ranks in ranks_list:
            append(
                tmp_dir / "summary.tsv",
                f"{case}\t{scale}\t{ranks}\t-\t-\t-\t-\t-\t-\t-\t{status}\n",
            )
        return [], status

    std_bin, dac_bin, build_status = build_case(case, dac_work, std_work, case_dir)
    if build_status != "ok":
        for ranks in ranks_list:
            append(
                tmp_dir / "summary.tsv",
                f"{case}\t{scale}\t{ranks}\t-\t-\t-\t-\t-\t-\t-\t{build_status}\n",
            )
        return [], build_status

    all_profile_rows = []
    runtime_failures = []
    for ranks in ranks_list:
        rank_dir = case_dir / f"np{ranks}"
        rank_dir.mkdir(parents=True, exist_ok=True)
        standard_times = []
        dac_wall_times = []
        dac_profile_times = []
        trial_statuses = []
        stdout_matches = []

        for trial in range(1, trials + 1):
            trial_dir = rank_dir / f"trial{trial}"
            trial_dir.mkdir(parents=True, exist_ok=True)

            standard_s, standard_status = run_mpirun(
                std_bin,
                ranks,
                {},
                trial_dir / "standard.stdout",
                trial_dir / "standard.stderr",
            )
            dac_wall_s, dac_wall_status = run_mpirun(
                dac_bin,
                ranks,
                {},
                trial_dir / "dac_wall.stdout",
                trial_dir / "dac_wall.stderr",
            )
            dac_profile_s, dac_profile_status = run_mpirun(
                dac_bin,
                ranks,
                {"DACPP_MPI_PROFILE": "1"},
                trial_dir / "dac_profile.stdout",
                trial_dir / "dac_profile.stderr",
            )

            standard_times.append(standard_s if standard_status == "ok" else None)
            dac_wall_times.append(dac_wall_s if dac_wall_status == "ok" else None)
            dac_profile_times.append(dac_profile_s if dac_profile_status == "ok" else None)
            trial_status = status_join([standard_status, dac_wall_status, dac_profile_status])
            trial_statuses.append(trial_status)
            if trial_status != "ok":
                runtime_failures.append(f"np{ranks}/trial{trial}:{trial_status}")

            match = "-"
            if dac_wall_status == "ok" and dac_profile_status == "ok":
                match = (
                    "yes"
                    if clean_output_text(trial_dir / "dac_wall.stdout")
                    == clean_output_text(trial_dir / "dac_profile.stdout")
                    else "no"
                )
            stdout_matches.append(match)

            if dac_profile_status == "ok":
                rows, _ = parse_profile_log(
                    case,
                    scale,
                    ranks,
                    trial,
                    trial_dir / "dac_profile.stderr",
                    tmp_dir,
                )
                all_profile_rows.extend(rows)

            append(
                tmp_dir / "results.tsv",
                f"{case}\t{scale}\t{ranks}\t{trial}"
                f"\t{seconds_or_dash(standard_times[-1])}"
                f"\t{seconds_or_dash(dac_wall_times[-1])}"
                f"\t{seconds_or_dash(dac_profile_times[-1])}"
                f"\t{match}\t{trial_status}\n",
            )

        combined_status = status_join(trial_statuses)
        combined_match = "yes" if stdout_matches and all(m == "yes" for m in stdout_matches) else "no"
        append(
            tmp_dir / "summary.tsv",
            f"{case}\t{scale}\t{ranks}"
            f"\t{median_or_dash(standard_times)}\t{mean_or_dash(standard_times)}"
            f"\t{median_or_dash(dac_wall_times)}\t{mean_or_dash(dac_wall_times)}"
            f"\t{median_or_dash(dac_profile_times)}\t{mean_or_dash(dac_profile_times)}"
            f"\t{combined_match}\t{combined_status}\n",
        )

    if runtime_failures:
        return all_profile_rows, "runtime-failed:" + ",".join(runtime_failures)
    return all_profile_rows, "ok"


def main():
    parser = argparse.ArgumentParser(
        description="Collect DACPP MPI segmented profile benchmark artifacts."
    )
    parser.add_argument("cases", nargs="*", help="case names from bench_mpi_only_requested.py")
    parser.add_argument(
        "--tmp-dir",
        default=str(DEFAULT_TMP),
        help="artifact directory (default: %(default)s)",
    )
    parser.add_argument(
        "--ranks",
        default=DEFAULT_RANKS,
        help="comma-separated mpirun ranks, e.g. 1,2,4,8",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
        help="trials per case/rank (default: %(default)s)",
    )
    args = parser.parse_args()

    if args.trials <= 0:
        raise SystemExit("--trials must be positive")

    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    ranks = parse_ranks(args.ranks)
    cases = selected_cases(args.cases)
    ensure_headers(tmp_dir)
    append_metadata(tmp_dir, args, ranks, cases)

    print(f"tmp={tmp_dir}")
    print(f"ranks={','.join(str(rank) for rank in ranks)} trials={args.trials}")
    print(f"cases={','.join(case[0] for case in cases)}")

    all_profile_rows = []
    failures = []
    for case_tuple in cases:
        case = case_tuple[0]
        print("=" * 70, flush=True)
        print(f"case: {case}", flush=True)
        rows, status = run_case(tmp_dir, case_tuple, ranks, args.trials)
        all_profile_rows.extend(rows)
        if status != "ok":
            failures.append(f"{case}:{status}")
        print(f"  status={status}", flush=True)

    summarize_profiles(tmp_dir, all_profile_rows)
    print("=" * 70)
    print(f"results={tmp_dir / 'summary.tsv'}")
    print((tmp_dir / "summary.tsv").read_text(encoding="utf-8"))
    if failures:
        raise SystemExit("failures: " + ", ".join(failures))


if __name__ == "__main__":
    main()
