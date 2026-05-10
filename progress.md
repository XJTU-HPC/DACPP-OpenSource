# Progress Log

## 2026-05-10 P5 Session

- Confirmed branch/worktree with `git status --short --branch`: `## tqc-2...origin/tqc-2`.
- Read current P4.6 closeout docs, shell-derived status doc, and tracking files.
- Read benchmark docs and current closeout TSV; kept old wrapper baseline separate from
  current closeout baseline. `oddeven0.1` has only the old wrapper record so far:
  hand-written 1.598514s, DAC wrapper 4.529443s.
- Searched the requested P5 terms. `FixedBlock` currently appears as enum/string metadata
  and test negative expectations, but there is no implemented accepted FixedBlock codegen
  path.
- Read `oddeven0.1`: both call sites use `dacpp::split S1(2,2)` and `ODDEVEN` has one
  READ fixed-block input plus one WRITE fixed-block output.
- Current key finding: OR analysis currently treats `RegularSplit` as stencil-window
  access and rejects writable regular split params, which is why `oddeven0.1` remains on
  legacy `AccessPattern` / `PackPlan`.
- Implemented the first P5 `FixedBlock` slice:
  - `FixedBlockPartitionAnalysis.cpp`
  - `FixedBlockCodegen.cpp`
  - `fixed_block_count_1d` runtime helper
  - OR plan/codegen/residency wiring
- Updated `oddeven0.1` expectations to require `FixedBlock` OR wrappers and absence of
  `AccessPattern` / `PackPlan` / stencil wrappers.
- Added `mpiFixedBlockOverlapReject1D` to pin `split(3,2)` fallback.
- Verification completed:
  - `cmake --build build --target translator -j8`
  - `bash clang/tools/translator/test_mpi.sh oddeven0.1 mpiFixedBlockOverlapReject1D`
  - `bash clang/tools/translator/test_mpi.sh MDP1.0 liuliang1.0 stencil1.0 waveEquation1.0`
  - `git diff --check`
- Focused benchmark completed:
  - `/Volumes/QUQ/working/mpi_tmp/p5_fixedblock_oddeven/results.tsv`
  - `oddeven0.1`, N=4096, standard 1.814108s, DAC P5 7.835445s, ok
  - Interpretation: current P5 standalone wrapper is a correctness/structure bridge;
    performance needs loop-resident phase exchange to avoid per-call full materialize and
    broadcast.

## 2026-05-10 P5 Review Fixes

- Tightened `FixedBlockPartitionAnalysis.cpp` so the P5 accepted slice rejects params
  with payload/void dimensions and shell params with more than one split before codegen
  can select `LocalLayoutKind::FixedBlock`.
- Tightened the same gate again after second review so a single split on a non-flat
  wrapper tensor, such as `Matrix` `input[{S1}]`, also falls back. Accepted P5 params now
  require both a 1D wrapper tensor and a 1D calc view.
- Updated `FixedBlockCodegen.cpp` so materialized `MPI_Bcast` counts use
  `narrow_mpi_count_or_abort(...)`; byte-transport product handling is kept in the local
  helper even though the first slice still rejects byte-transport element types.
- Added `mpiFixedBlockPayloadReject1D` to pin fallback for `input[{S1}][{}]` /
  `output[{S1}][{}]` style payload shapes, and extended `oddeven0.1` expectations to
  assert checked broadcast count generation.
- Added `mpiFixedBlockMatrixSingleSplitReject1D` to pin fallback for Matrix params that
  use only one split bracket but still require a 2D local view.
- Fixed the shell-derived status doc so the constraint is no P5 `FixedBlock` beyond the
  current proven first slice, rather than a stale blanket ban.
- Review-fix verification completed:
  - `cmake --build build --target translator -j8`
  - `bash clang/tools/translator/test_mpi.sh oddeven0.1 mpiFixedBlockOverlapReject1D mpiFixedBlockPayloadReject1D mpiFixedBlockMatrixSingleSplitReject1D`
  - `bash clang/tools/translator/test_mpi.sh MDP1.0 liuliang1.0 stencil1.0 waveEquation1.0`
  - `git diff --check`
- Re-ran focused benchmark after `test_mpi.sh` cleaned the shared tmp directory:
  - `/Volumes/QUQ/working/mpi_tmp/p5_fixedblock_oddeven/results.tsv`
  - `oddeven0.1`, N=4096, standard 1.814108s, DAC P5 7.835445s, ok

## 2026-05-10

- Revalidated the benchmark source split:
  - old wrapper baseline:
    `clang/tools/translator/docs/benchmarks/mpi_sycl_efficiency_benchmark_2026-05-06.md`
  - current closeout benchmark:
    `/Volumes/QUQ/working/mpi_tmp/p46_final_close/results.tsv`
- Confirmed the current worktree is still on branch `tqc-2` and preserved all unrelated
  user changes.
- Rewrote the two canonical P4.6 documents into current-state closeout form instead of the
  earlier step-by-step narrative.
- Removed stale and repetitive phase-history prose from the canonical docs and replaced it
  with:
  - current accepted boundary
  - landed implementation shape
  - benchmark comparison
  - `FOuLa1.0` blocker ownership
  - next-step handoff
- Reduced `task_plan.md`, `findings.md`, and `progress.md` to the active closeout state so
  future handoff reads like one coherent snapshot.

## Verification Already Completed For This Code State

```bash
cmake --build build --target translator -j8
bash clang/tools/translator/test_mpi.sh MDP1.0 liuliang1.0
bash clang/tools/translator/test_mpi.sh mpiLoopStencilResidentHalo1D mpiLoopStencilResidentHaloEmptyRank1D mpiLoopStencilRightBoundaryFullSync1D
bash clang/tools/translator/test_mpi.sh MDP1.0 liuliang1.0 stencil1.0 waveEquation1.0
bash clang/tools/translator/test_mpi.sh mpiLoopStencilScalarReject2D mpiLoopStencilCountGuard2D mpiLoopStencilOrderReject2D
git diff --check
```

Focused benchmark already captured:

```bash
MPI_ONLY_BENCH_TMP_DIR=/Volumes/QUQ/working/mpi_tmp/p46_final_close \
MPI_ONLY_BENCH_RANKS=4 \
MPI_ONLY_BENCH_TIMEOUT_SECONDS=1800 \
python3 clang/tools/translator/bench_mpi_only_requested.py \
  MDP1.0 liuliang1.0 stencil1.0 waveEquation1.0 FOuLa1.0
```
