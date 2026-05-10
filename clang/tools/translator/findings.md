# Findings: MPI Translator Non-Specialized Performance Profiling

## Context Read
- Canonical docs live under `clang/tools/translator/docs/...`, not repo-root `docs/...`.
- Current model is shell-derived/operator-resident first, then Phase-C stencil, then legacy AccessPattern fallback.
- `vectorAddCombo` is currently `Contiguous1D` OR chain, not legacy AccessPattern.
- FOuLa currently uses a guarded owner-loop specialization and should remain a regression checkpoint, not a reason to relax generic guards.

## Existing Profiling
- `dpcppLib/include/mpi/common/Profile.h` already gates profiling with `DACPP_MPI_PROFILE` and records `collect_positions_for_item`.
- Legacy wrapper code already emits a wrapper total and calls `reportCollectPositionsProfile`, but it does not emit standardized segments.
- Phase-C stencil code has local timing buckets (`input`, `dist_setup`, `kernel`, etc.) and a custom breakdown line; this should be folded toward the shared taxonomy rather than inventing another format.

## Profiling Taxonomy
- Required segment names from the TODO: `init`, `scatter`, `pack`, `kernel`, `halo`, `gather`, `bcast`, `materialize`, `final_sync`.
- Use wrapper/function labels so repeated DAC expressions can be distinguished.
- Output should be parseable and disabled by default.

## Implemented Profiling Shape
- Shared runtime API: `dacpp::mpi::SegmentedProfile`, `ProfileSegment`, `profileSegmentStart`, `recordProfileSegment`, and `reportSegmentedProfile`.
- Output format is one TSV-like line per non-empty segment, gated by `DACPP_MPI_PROFILE=1`:
  `DACPP_MPI_PROFILE\t<label>\t<segment>\tcalls=<n>\tmax_ms=<ms>\tsum_ms=<ms>`.
- Ordinary non-loop OR wrappers report once per wrapper call.
- Loop-lowered OR/stencil/FixedBlock paths accumulate in the generated context and report once at materialize/loop exit. Segment call counts therefore aggregate `rank x step` events for loop body segments.
- Legacy AccessPattern wrappers report once per wrapper and keep the existing `collect_positions_for_item` profile lines.
- Profiling output is on `stderr`, so normal stdout comparison remains unchanged.

## Runtime Smoke Evidence
- `vectorAddCombo` with `DACPP_MPI_PROFILE=1`: ordinary OR chain emitted init/scatter/kernel/gather/bcast/materialize rows and stdout stayed unchanged.
- `FOuLa1.0`: owner-loop emitted init/scatter/kernel/halo/gather/bcast/materialize rows.
- `MDP1.0` and `stencil1.0`: loop-resident halo paths emitted one aggregated table per materialize label, with kernel/halo call counts reflecting ranks times loop steps.
- `oddeven0.1`: P5 phase exchange emitted init/scatter/kernel/gather/bcast/materialize rows.
- `mpiFixedBlockMatrixSingleSplitReject1D`: legacy fallback emitted init/scatter/pack/kernel/gather/materialize/final_sync plus existing collect-position rows.

## Phase 1.4 Benchmark Reporting
- New companion driver: `bench_mpi_profile_segments.py`.
- Artifact format:
  - `results.tsv`: per case/rank/trial wall times for hand MPI+SYCL, DAC profile-off, and DAC profile-on.
  - `summary.tsv`: median/mean wall times.
  - `profile_raw.tsv`: parsed `DACPP_MPI_PROFILE` rows.
  - `profile_summary.tsv`: median calls/max_ms/sum_ms by case/rank/label/segment.
  - `collect_positions_profile.tsv`: legacy `collect_positions_for_item` profile rows.
  - per-trial `standard.stdout/stderr`, `dac_wall.stdout/stderr`, and `dac_profile.stdout/stderr`.
  - generated source snapshots plus `metadata.json`, `git_status.txt`, `git_diff_stat.txt`, and `git_diff.patch`.
- Main focused artifact path:
  `/Volumes/QUQ/working/mpi_tmp/phase1_4_profile_focus`.
- Focused command:
  `python3 clang/tools/translator/bench_mpi_profile_segments.py --tmp-dir /Volumes/QUQ/working/mpi_tmp/phase1_4_profile_focus --ranks 1,2,4,8 --trials 3 vectorAddCombo FOuLa1.0 MDP1.0 stencil1.0 oddeven0.1 decay1.0`.
- The driver is a benchmark/profile driver, like `bench_mpi_only_requested.py`, not the correctness oracle. It verifies profile-on DAC stdout equals profile-off DAC stdout. Baseline-vs-DAC correctness stays under `test_mpi.sh`.

## Phase 1.4 Focused Readout
- 4-rank wall medians:
  - `FOuLa1.0`: hand 0.269842s, DAC 0.373362s, DAC/hand 1.38x.
  - `MDP1.0`: hand 0.269820s, DAC 0.269225s, DAC/hand 1.00x.
  - `decay1.0`: hand 0.330189s, DAC 0.380968s, DAC/hand 1.15x.
  - `oddeven0.1`: hand 1.195697s, DAC 0.187462s, DAC/hand 0.16x.
  - `stencil1.0`: hand 1.009845s, DAC 0.954323s, DAC/hand 0.95x.
  - `vectorAddCombo`: hand 0.202725s, DAC 0.267805s, DAC/hand 1.32x.
- 4-rank segment sum_ms aggregates are attribution signals summed across ranks, not wall-time totals:
  - `vectorAddCombo`: scatter 47.954ms, materialize 38.883ms, bcast 19.832ms, kernel 16.179ms, gather 10.269ms.
  - `decay1.0`: kernel 591.297ms, gather 198.918ms, bcast 92.985ms, materialize 2.477ms.
  - `stencil1.0`: kernel 2318.965ms, halo 514.394ms, scatter 40.424ms, materialize 20.006ms, gather 18.564ms.
  - `MDP1.0`: kernel 417.994ms, halo 37.296ms; other measured segments are below 1ms aggregate.
  - `FOuLa1.0`: kernel 482.170ms, halo 61.911ms, bcast 56.898ms, materialize 51.093ms.
  - `oddeven0.1`: kernel dominates; communication/materialize segments are sub-millisecond to low single-digit aggregate.
- `vectorAddCombo` remains current OR `Contiguous1D`, not legacy. Its measured residual cost points more toward scatter/final materialize/bcast than legacy pack.
- `decay1.0` shows gather+bcast per-step cost, making it a strong Phase 3 materialization/residency inventory candidate.
- Resident halo cases are mostly kernel+halo dominated in this focused run; avoid widening stencil/FOuLa surface based on these numbers alone.
- Legacy fallback smoke artifact:
  `/Volumes/QUQ/working/mpi_tmp/phase1_4_legacy_profile_smoke`.
  It confirms the legacy AccessPattern path emits `pack` and `final_sync` segments while preserving existing `collect_positions_for_item` diagnostics.

## Code Review Fixes Applied
- Resident-halo materialize profiling now avoids gather/materialize overlap:
  - 1D final materialize records writer `array2Tensor`, reader `tensor2Array`, and reader `array2Tensor` as separate non-overlapping `Materialize` samples around the reader `MPI_Gatherv`.
  - 2D final materialize records writer, optional direct-reader, and final reader host updates as separate non-overlapping `Materialize` samples around any `MPI_Gatherv` calls.
- Loop-lowered direct `materialize()` for `loopLowerMaterializeEveryRun=true` now only reports the accumulated profile. The real per-run materialize work is still recorded inside `emitMaterializeOutput`; the empty final report path no longer increments `Materialize` calls.
- `decay1.0`, `liuliang1.0`, and `waveEquation1.0` now assert generated `SegmentedProfile __or_profile`, `ProfileSegment::Materialize`, and `reportSegmentedProfile(...)`.
- `bench_mpi_profile_segments.py` now defaults to `/tmp/dacpp_mpi_profile_phase1_4` instead of a machine-specific `/Volumes/QUQ/...` path. Existing env/CLI overrides remain supported.
- Review-fix smoke for `decay1.0` at 4 ranks showed `materialize` calls as 2400 after the fix, matching `rank x step` instead of the previous inflated count.
