# Progress: MPI Translator Non-Specialized Performance Profiling

## 2026-05-10
- Read current-status, benchmark-summary, and performance-optimization TODO docs.
- Confirmed branch/head: `tqc-2` at `96118823800ca66c45049e9bb879fa866bea9fd7`.
- Read existing profiling runtime and legacy/Phase-C codegen. Existing profiler only covers `collect_positions_for_item`; Phase-C has a local non-taxonomy breakdown.
- Added shared segmented profiling runtime in `dpcppLib/include/mpi/common/Profile.h`; profiling is still default-off and only active when `DACPP_MPI_PROFILE=1`.
- Instrumented legacy AccessPattern wrappers, ordinary OR wrappers, loop-lowered direct/stencil paths, 2D resident/full-sync paths, P5 FixedBlock phase exchange, and FOuLa owner-loop specialization.
- Updated representative structure checks: `FOuLa1.0`, `MDP1.0`, `stencil1.0`, `oddeven0.1`, and legacy fallback `mpiFixedBlockMatrixSingleSplitReject1D`.
- Validation passed: `cmake --build build --target translator -j8`.
- Validation passed: `bash test_mpi.sh FOuLa1.0 MDP1.0 stencil1.0 oddeven0.1 mpiFixedBlockMatrixSingleSplitReject1D vectorAddCombo liuliang1.0 waveEquation1.0 decay1.0`.
- Profile smoke passed with `DACPP_MPI_PROFILE=1` for `vectorAddCombo`, `FOuLa1.0`, `MDP1.0`, `stencil1.0`, `oddeven0.1`, `decay1.0`, and compiled legacy fallback `mpiFixedBlockMatrixSingleSplitReject1D`.
- Validation passed: `git diff --check -- clang/tools/translator`.
- Added Phase 1.4 companion driver `bench_mpi_profile_segments.py`. It reuses the enlarged benchmark case patching from `bench_mpi_only_requested.py`, stores generated source snapshots and per-trial stdout/stderr, writes `results.tsv`, `summary.tsv`, `profile_raw.tsv`, `profile_summary.tsv`, `collect_positions_profile.tsv`, `metadata.json`, and git status/diff artifacts.
- Smoke-tested the new driver with `vectorAddCombo` at `--ranks 2 --trials 1`; artifact path: `/Volumes/QUQ/working/mpi_tmp/phase1_4_smoke`.
- Ran focused Phase 1.4 benchmark/profile collection:
  `python3 clang/tools/translator/bench_mpi_profile_segments.py --tmp-dir /Volumes/QUQ/working/mpi_tmp/phase1_4_profile_focus --ranks 1,2,4,8 --trials 3 vectorAddCombo FOuLa1.0 MDP1.0 stencil1.0 oddeven0.1 decay1.0`.
- Focused benchmark/profile collection passed for all six cases and all rank/trial combinations. For every run, DAC profile-on stdout matched DAC profile-off stdout.
- Collected legacy fallback smoke artifact at `/Volumes/QUQ/working/mpi_tmp/phase1_4_legacy_profile_smoke`; it emitted segmented `init/scatter/pack/kernel/gather/materialize/final_sync` rows plus the existing `collect_positions_for_item` profile and profile-on/off stdout matched.
- Final validation passed: `cmake --build build --target translator -j8`.
- Final validation passed: `bash test_mpi.sh FOuLa1.0 MDP1.0 stencil1.0 oddeven0.1 mpiFixedBlockMatrixSingleSplitReject1D vectorAddCombo liuliang1.0 waveEquation1.0 decay1.0`.
- Final validation passed: `python3 -m py_compile clang/tools/translator/bench_mpi_profile_segments.py`.
- Final script smoke passed: `python3 clang/tools/translator/bench_mpi_profile_segments.py --tmp-dir /Volumes/QUQ/working/mpi_tmp/phase1_4_smoke_final --ranks 1 --trials 1 decay1.0`; it also verified `git_untracked.txt` and `untracked_snapshots/` artifact capture.
- Final validation passed: `git diff --check -- clang/tools/translator`.
- Addressed Phase 1 code-review findings:
  - fixed resident-halo materialize/gather segment overlap by recording non-overlapping materialize sub-intervals around writer/direct-reader/reader host tensor updates;
  - removed the extra zero-work `Materialize` sample from loop-lowered direct `materialize()` when `loopLowerMaterializeEveryRun=true`;
  - added profile structure guards to `decay1.0`, `liuliang1.0`, and `waveEquation1.0`;
  - changed the profile benchmark driver's default artifact path to `/tmp/dacpp_mpi_profile_phase1_4`, still overrideable with `MPI_PROFILE_BENCH_TMP_DIR` or `--tmp-dir`.
- Review-fix validation passed: `cmake --build build --target translator -j8`.
- Review-fix validation passed: `bash test_mpi.sh FOuLa1.0 MDP1.0 stencil1.0 oddeven0.1 mpiFixedBlockMatrixSingleSplitReject1D vectorAddCombo liuliang1.0 waveEquation1.0 decay1.0`.
- Review-fix smoke passed: `/Volumes/QUQ/working/mpi_tmp/review_fix_decay_profile/profile_summary.tsv` shows `decay1.0` 4-rank `materialize` calls at 2400 instead of the previous inflated 2404.
- Review-fix smoke passed: running `bench_mpi_profile_segments.py` without `--tmp-dir` wrote artifacts under `/tmp/dacpp_mpi_profile_phase1_4`.
