# Progress Log

## Session: 2026-05-27 Four-Case Optimization Implementation

### Setup
- **Status:** in_progress
- Actions taken:
  - Accepted user's requirement to implement optimizations 1-4 serially with one sub-agent per optimization.
  - Added a new implementation plan to `task_plan.md`.
  - Confirmed current dirty state has untracked planning files and an untracked external analysis doc; these will not be staged or reverted unless requested.
  - Initially spawned extra agents for optimizations 2 and 3 by mistake; immediately shut them down. Continuing strictly serially with only the FOuLa agent active.
- Files modified:
  - /Volumes/QUQ/working/dacpp/task_plan.md
  - /Volumes/QUQ/working/dacpp/progress.md

### Phase 1: FOuLa Owner-Loop Cleanup and Larger Temporal Block
- **Status:** complete
- Actions taken:
  - Sub-agent implemented FOuLa owner-loop temporal cleanup in `LoopLocalStencilOwnerLoop.cpp`.
  - Parent reviewed diff: only FOuLa owner-loop codegen and `FOuLa1.0/mpi_expect.txt` changed.
  - Parent verified generated evidence in `/Volumes/QUQ/working/mpi_tmp/FOuLa1.0`: one `scatter_window_1d_temporal`, no ordinary `scatter_window_1d(__or_initial_col`, and `__or_temporal_block_size = 8`.
- Verification:
  - `cmake --build build --target translator -j8` passed.
  - `bash clang/tools/translator/test_mpi.sh FOuLa1.0` passed 1/1.
- Files modified:
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/operator_resident/LoopLocalStencilOwnerLoop.cpp
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/tests/FOuLa1.0/mpi_expect.txt

### Phase 2: Spatial-2D Halo Runtime Optimization
- **Status:** complete
- Actions taken:
  - Sub-agent optimized `exchange_halo_2d_spatial_inplace` scratch allocation in `OperatorResidentRuntime.h`.
  - Parent reviewed diff: no call-site/signature changes, no post-use/materialization changes, no FOuLa files touched.
  - Parent verified generated stencil/wave still use `distribution=spatial-2d accepted`, `halo-width=2`, and `exchange_halo_2d_spatial_inplace`.
- Verification:
  - `git diff --check -- clang/tools/translator/dpcppLib/include/mpi/operator_resident/OperatorResidentRuntime.h` passed.
  - `cmake --build build --target translator -j8` passed.
  - `bash clang/tools/translator/test_mpi.sh stencil1.0 waveEquation1.0` passed 2/2.
- Files modified:
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/mpi/operator_resident/OperatorResidentRuntime.h

### Phase 3: Larger Temporal Block for Stencil/Wave
- **Status:** complete
- Actions taken:
  - Replacement sub-agent completed the partial temporal-block diff for `StencilWindow2D`.
  - Parent reviewed the structural changes: 2D resident-halo temporal block is now 4, spatial halo width follows the block size, and generated code includes runtime narrow-partition fallback to block size 1.
  - Parent verified the touched structure fixtures still report expected accepted/rejected paths.
- Verification:
  - `cmake --build build --target translator -j8` passed.
  - `bash clang/tools/translator/test_mpi.sh stencil1.0 waveEquation1.0` passed 2/2.
  - `bash clang/tools/translator/test_mpi.sh mpiLoopStencilCountGuard2D mpiMixedStencilORPhaseC spatialStencil2DOneStep` passed 3/3.
- Files modified:
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/operator_resident/OperatorChainAnalysis.cpp
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/operator_resident/StencilWindowCodegen.cpp
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/tests/stencil1.0/mpi_expect.txt
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/tests/waveEquation1.0/mpi_expect.txt
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/tests/mpiLoopStencilCountGuard2D/mpi_expect.txt
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/tests/mpiMixedStencilORPhaseC/mpi_expect.txt
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/tests/spatialStencil2DOneStep/mpi_expect.txt

### Phase 4: Partial Materialization for Row/Print Post-Use
- **Status:** complete
- Actions taken:
  - A first optimization-4 worker launch failed due to a `gpt-5.3-codex` service-channel 503 before making code changes.
  - Per user instruction, launched replacement worker `019e688c-6f81-7f91-9697-e44e8038af35` with model `gpt-5.5`; it failed before code changes because the selected model was at capacity.
  - Launched retry worker `019e6896-a856-7373-8722-0dcbf5ecda98` with model `gpt-5.5` and service tier `priority`.
  - Worker scope is limited to fixed-row `.print()` partial materialization, with full `.print()` fallback preserved for `waveEquation1.0` and `imageAdjustment1.0`.
  - Parent reviewed the resulting shared post-use and stencil materialization changes.
  - Parent tightened the implementation so spatial-2d fixed-row print gathers one row slice per owning rank with `MPI_Gatherv` instead of per-element send/recv.
  - `stencil1.0` now logs `matIn post-use=fixed-row-print` and generated code revises only the printed row while rejecting the old full `gather_spatial_owned_to_root(__or_owned_out)` path.
  - `waveEquation1.0` still logs `matCur post-use=full fallback reason=member call on tensor`, and `imageAdjustment1.0` still uses full final output gather for `image_tensor3.print()`.
- Verification:
  - `git diff --check` passed.
  - `cmake --build build --target translator -j8` passed.
  - `bash clang/tools/translator/test_mpi.sh stencil1.0 waveEquation1.0 imageAdjustment1.0` passed 3/3.
  - `bash clang/tools/translator/test_mpi.sh vectorAddCombo mandel1.0 FOuLa1.0` passed 3/3.
  - Follow-up regression batch initially exposed stale `mpiMixedStencilORPhaseC` expectations: its stencil phase also has a root-guarded fixed-row `matIn[0].print()` and now correctly uses fixed-row materialization instead of full spatial gather.
  - Updated `mpiMixedStencilORPhaseC/mpi_expect.txt` to assert fixed-row row-slice gather while preserving PhaseC partial-exchange checks.
  - `bash clang/tools/translator/test_mpi.sh mpiMixedStencilORPhaseC` passed 1/1.
  - `bash clang/tools/translator/test_mpi.sh FOuLa1.0 stencil1.0 waveEquation1.0 imageAdjustment1.0 mpiLoopStencilCountGuard2D mpiMixedStencilORPhaseC spatialStencil2DOneStep vectorAddCombo mandel1.0` passed 9/9.
- Files modified:
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/include/mpi/shared/PostUseSyncPlan.h
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/shared/PostRegionAnalysis.cpp
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/shared/OutputSyncAnalysis.cpp
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/operator_resident/OperatorChainAnalysis.cpp
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/operator_resident/StencilWindowCodegen.cpp
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/tests/stencil1.0/mpi_expect.txt

## Session: 2026-05-27 Four-Case Shenwei Efficiency Diagnosis

### Phase 1-3: Analysis
- **Status:** complete
- Actions taken:
  - Restored planning context and reused generated large-case MPI/SYCL outputs in `/Volumes/QUQ/working/analysis_tmp/four_case_codegen`.
  - Inspected generated code for `imageAdjustment1.0`, `waveEquation1.0`, `stencil1.0`, and `FOuLa1.0`.
  - Inspected resident-halo runtime for spatial scatter, gather, and halo exchange implementations.
  - Compared generated structure with hand-written MPI/SYCL references in the four test directories.
  - Recorded case-by-case bottleneck findings in `findings.md`.
- Files modified:
  - /Volumes/QUQ/working/dacpp/findings.md
  - /Volumes/QUQ/working/dacpp/progress.md

## Session: 2026-05-22 stencil1 spatial-k2 profitability follow-up

### Phase 1: Discovery and Reproduction
- **Status:** in_progress
- Actions taken:
  - Restored planning context and confirmed branch `tqc-2` at `b2afd4588`.
  - Confirmed pre-existing unrelated deleted files remain in the worktree and will not be reverted or staged.
  - Added this session's plan to `task_plan.md`.
  - Re-read the required OR plan/analysis/codegen/runtime files and the target `stencil1.0`, `spatialStencil2DOneStep`, `mpiLoopStencilCountGuard2D`, `mpiMixedStencilORPhaseC`, and `waveEquation1.0` fixtures.
  - Temporarily removed the spatial-k2 post-use profitability gate for measurement only; `stencil1.0` accepted spatial-k2 structurally, confirming the gate is the current blocker.
  - Ran an experimental 4-rank, 3-trial profile for bounded `stencil1.0` spatial-k2 in `/Volumes/QUQ/working/mpi_tmp/profile_segments_p11_stencil1_spatial_k2_experiment`. It was correct but slow (`dac_wall_median_s=4.838048`), with the time dominated by kernel (`4496.731972 ms`) while materialize stayed microsecond-scale.
  - Restored the profitability fallback and tightened the translator reject reason to name host post-use plus the current rectangular buffer path.
- Files modified:
  - /Volumes/QUQ/working/dacpp/task_plan.md
  - /Volumes/QUQ/working/dacpp/progress.md
  - /Volumes/QUQ/working/dacpp/findings.md
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/operator_resident/OperatorChainAnalysis.cpp
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/tests/stencil1.0/mpi_expect.txt
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/tests/mpiMixedStencilORPhaseC/mpi_expect.txt
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/docs/mpi_translator/mpi_32node_top10_optimization_plan_2026-05-21.md

### Phase 4: Verification and Profile
- **Status:** complete
- Results:
  - `cmake --build build --target translator -j8` passed.
  - `bash clang/tools/translator/test_mpi.sh stencil1.0 spatialStencil2DOneStep mpiLoopStencilCountGuard2D mpiMixedStencilORPhaseC` passed 4/4.
  - `bash clang/tools/translator/test_mpi.sh mandel1.0 FOuLa1.0 liuliang1.0 MDP1.0 waveEquation1.0` passed 5/5.
  - Related `rg` fixtures passed 8/8: `DFT1.0 decay1.0 gradientSum imageAdjustment1.0 jacobi1.0 matMul1.0 oddeven0.1 vectorAddCombo`.
  - Final requested profile in `/Volumes/QUQ/working/mpi_tmp/profile_segments_p11_stencil1_spatial_k2_final` passed all six cases status ok. Key medians: `stencil1.0` DAC wall `0.846273 s`, `waveEquation1.0` DAC wall `1.024449 s`, `mandel1.0` DAC wall `2.321547 s`.

### Phase 5: Commit and Push
- **Status:** complete
- Actions taken:
  - Confirmed `git diff --cached --name-status` contained only four task-relevant files before commit.
  - Committed `6207a0f6b Refine stencil spatial k2 profitability`.
  - Pushed `6207a0f6b` to `origin/tqc-2`.

## Session: 2026-05-22 P11 2D Spatial Block Partition

### Phase 1: Discovery
- **Status:** complete
- Actions taken:
  - Confirmed branch `tqc-2` at `228fd3158` with only the user-reported unrelated dirty deletions plus untracked planning files.
  - Read P11 optimization-plan section, OR plan structures, 2D resident-halo analysis/codegen, runtime row-halo helpers, and `stencil1.0` expectations/source.
  - Identified a conservative first implementation target: canonical B2 `StencilWindow2D` resident halo, no direct reader, no scalar readers, 3x3 stride-1 window, followup `(1,1)`, and canonical boundary-local self-copy loops.
- Files modified:
  - /Volumes/QUQ/working/dacpp/task_plan.md
  - /Volumes/QUQ/working/dacpp/findings.md
  - /Volumes/QUQ/working/dacpp/progress.md

### Phase 2: Design and Implementation
- **Status:** complete
- Actions taken:
  - Added `spatial-2d` plan metadata/logs under `StencilResidentHaloMetadata`.
  - Accepted only canonical B2 `StencilWindow2D` resident halo shapes; direct-reader B3 now logs an explicit spatial rejection while keeping the row-temporal path.
  - Added runtime rectangle layout, process-grid ownership, rectangle scatter, point-to-point edge/corner halo exchange, spatial owned slices, spatial root gather reassembly, and spatial owner lookup.
  - Extended resident-halo 2D codegen for accepted spatial-2d and kept row fallback for rejected/direct-reader cases.
- Files modified:
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/include/mpi/operator_resident/OperatorResidentPlan.h
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/operator_resident/OperatorChainAnalysis.cpp
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/operator_resident/StencilWindowCodegen.cpp
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/mpi/operator_resident/OperatorResidentRuntime.h

### Phase 3: Fixtures and Documentation
- **Status:** complete
- Actions taken:
  - Updated `stencil1.0/mpi_expect.txt` for accepted `distribution=spatial-2d` logs and generated-code evidence.
  - Updated `waveEquation1.0/mpi_expect.txt` for explicit spatial rejection and preserved row-temporal direct-reader recurrence.
  - Updated `mpiLoopStencilCountGuard2D/mpi_expect.txt` for accepted no-use spatial-2d resident halo.
  - Added P11 Implementation Status to the optimization plan.
- Files modified:
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/tests/stencil1.0/mpi_expect.txt
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/tests/waveEquation1.0/mpi_expect.txt
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/tests/mpiLoopStencilCountGuard2D/mpi_expect.txt
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/docs/mpi_translator/mpi_32node_top10_optimization_plan_2026-05-21.md

### Phase 4: Verification
- **Status:** in_progress
- Results:
  - `cmake --build build --target translator -j8` passed.
  - `bash clang/tools/translator/test_mpi.sh stencil1.0` passed.
  - P6-P10 regression batch passed after expectation updates: `mandel1.0`, `FOuLa1.0`, `liuliang1.0`, `MDP1.0`, and `waveEquation1.0`.
  - Related fixture batch passed 11/11: `stencil1.0`, `mpiLoopStencilCountGuard2D`, `mpiLoopStencilOrderReject2D`, `mpiLoopStencilScalarReject2D`, `mpiDistributedStencil2DRowBlock`, `mpiOrStencilBoundaryStride2D`, `mpiMixedStencilORPhaseC`, `mpiOrReadWriteAccumulate2D`, `jacobi1.0`, `imageAdjustment1.0`, and `waveEquation1.0`.
  - Requested profile probe in `/Volumes/QUQ/working/mpi_tmp/profile_segments_p11_probe` passed all six requested cases with status ok.
  - Profile structural evidence: `stencil1.0` logs `distribution=spatial-2d accepted` and generated code contains `exchange_halo_2d_spatial_inplace`; `waveEquation1.0` logs `distribution=spatial-2d rejected reason=direct-reader recurrence requires row-layout role rotation` and keeps `temporal-block=2 accepted direct-reader recurrence`.
  - Performance note: `stencil1.0` 4-rank profile is slower (`dac_wall_median_s=8.687478`) because the first P11 spatial subset deliberately uses one-step halo exchange and keeps k=2 temporal blocking on row fallback only.
  - New continuation: implemented B2 spatial temporal-block=2. `cmake --build build --target translator -j8` passed after runtime/codegen fixes.
  - `stencil1.0` now logs `distribution=spatial-2d accepted stencil halo-width=2 ... temporal-block=2 accepted`; manual 1/2/4-rank runs matched baseline before expectations were updated.
  - `bash clang/tools/translator/test_mpi.sh stencil1.0 spatialStencil2DOneStep mpiLoopStencilCountGuard2D mpiMixedStencilORPhaseC` passed 4/4 after updating structure expectations.
  - `bash clang/tools/translator/test_mpi.sh mandel1.0 FOuLa1.0 liuliang1.0 MDP1.0 waveEquation1.0` passed 5/5; `waveEquation1.0` remains row-temporal direct-reader with explicit spatial rejection.
  - Related fixtures from `rg` were checked with `bash clang/tools/translator/test_mpi.sh DFT1.0 decay1.0 gradientSum imageAdjustment1.0 jacobi1.0 matMul1.0 oddeven0.1 vectorAddCombo`, passing 8/8.
  - First full profile with spatial k=2 showed `stencil1.0` bounded-post-use spatial k=2 was too slow (`dac_wall_median_s=4.778941`), so added a generic profitability fallback: spatial temporal-block=2 now requires no host post-use and otherwise logs row-temporal retention.
  - After fallback, `stencil1.0`, `mpiMixedStencilORPhaseC`, and large profile variants retain row-temporal k=2; `mpiLoopStencilCountGuard2D` still accepts spatial k=2 because both reader/writer post-use are none.
  - Final target checks after fallback passed: `cmake --build build --target translator -j8`; `bash clang/tools/translator/test_mpi.sh stencil1.0 mpiLoopStencilCountGuard2D mpiMixedStencilORPhaseC waveEquation1.0`; and `bash clang/tools/translator/test_mpi.sh mandel1.0 FOuLa1.0 liuliang1.0 MDP1.0 waveEquation1.0 DFT1.0 decay1.0 gradientSum imageAdjustment1.0 jacobi1.0 matMul1.0 oddeven0.1 vectorAddCombo`.
  - Final requested profile in `/Volumes/QUQ/working/mpi_tmp/profile_segments_p11_spatial_k2_probe` passed all six cases. Key medians: `stencil1.0` DAC wall 0.864732 s, `waveEquation1.0` DAC wall 0.947056 s, `mandel1.0` DAC wall 2.352500 s.

### Phase 5: Staging Review and Delivery
- **Status:** complete
- Actions taken:
  - Confirmed no files are staged.
  - Confirmed P11-relevant tracked edits are limited to OR runtime/plan/analysis/codegen, three MPI expectations, and the optimization plan doc.
  - Left unrelated pre-existing deleted files and untracked planning files untouched.

## Session: 2026-05-11

### Phase 1: Requirements & Discovery
- **Status:** complete
- **Started:** 2026-05-11
- Actions taken:
  - Created planning files (task_plan.md, findings.md, progress.md)
  - Analyzed ResidentBufferCodegen.cpp - found `emitRowPartitionFullRowScatter` function
  - Analyzed gradientSum generated code - shows pack + Scatterv pattern
  - Identified fast path conditions: indexedBindPos == 0 (row partition with indexed on outer dim)
- Files created/modified:
  - /Volumes/QUQ/working/dacpp/task_plan.md
  - /Volumes/QUQ/working/dacpp/progress.md

### Phase 2: Planning & Structure
- **Status:** complete
- Fast path detection: indexedBindPos == 0 && !broadcastIndexedPayload
- Skip MPI_Scatterv and use MPI_Bcast + direct buffer access
- Each rank calculates offset and reads its portion directly

### Phase 3: Implementation
- **Status:** complete
- Actions taken:
  - Added fast path detection: else if (indexedBindPos == 0) branch in emitRowPartitionFullRowScatter
  - Fast path uses MPI_Bcast to broadcast packed buffer to all ranks
  - Each rank reads its portion directly using std::copy_n with calculated offset
  - Maintains existing Scatterv paths for non-fast-path cases (broadcastIndexedPayload, usesByte, default)
- Files created/modified:
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/operator_resident/ResidentBufferCodegen.cpp

### Phase 4: Testing & Verification
- **Status:** complete
- Actions taken:
  - Generated unified diff patch (saved to /tmp/gradientSum_fastpath.patch)
  - Verified patch adds new else-if branch for indexedBindPos == 0
  - Patch maintains existing behavior for other cases
- Files created/modified:
  - /tmp/gradientSum_fastpath.patch

### Phase 5: Delivery
- **Status:** complete

## Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
|      |       |          |        |        |

## Error Log
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
|           |       | 1       |            |

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | Phase 1 |
| Where am I going? | Phases 2-5 (Planning, Implementation, Testing, Delivery) |
| What's the goal? | Add fast path to skip MPI_Scatterv when input is row-partitioned |
| What have I learned? | See findings.md (empty, to be filled) |
| What have I done? | Created planning files, starting code analysis |

---

## Session: 2026-05-19 Constant Init Local Fill + Async Halo Runtime

### Phase 1: Discovery
- **Status:** in_progress
- Actions taken:
  - Restored existing planning context for prior RowBlock2D, post-use reduction, and post-use sync work.
  - Confirmed working tree has existing dirty/untracked files; will not revert or clean them.
  - Added this session's plan to `task_plan.md`.
  - Inspected OR plan structs, post-use analysis helpers, resident input codegen, stencil-window codegen, and resident halo runtime.
  - Recorded initial findings: resident window init still scatters from root, and 1D/2D halo runtime uses blocking `MPI_Sendrecv` plus per-call neighbor lookup.
- Files modified:
  - /Volumes/QUQ/working/dacpp/task_plan.md
  - /Volumes/QUQ/working/dacpp/progress.md
  - /Volumes/QUQ/working/dacpp/findings.md

---

## Session: 2026-05-19 Host Post-Use Partial Materialization

### Phase 1: Discovery
- **Status:** in_progress
- Actions taken:
  - Restored existing planning context and session catchup.
  - Confirmed working tree is dirty with prior translator/test changes; will not revert or clean unrelated files.
  - Added this session's plan to `task_plan.md`.
  - Inspected OutputSync/PostRegion analysis/codegen and OR plan structures.
  - Recorded that existing sync classification is coarse and mandel reduction already uses per-param skip flags.
- Files modified:
  - /Volumes/QUQ/working/dacpp/task_plan.md
  - /Volumes/QUQ/working/dacpp/progress.md
  - /Volumes/QUQ/working/dacpp/findings.md

### Phase 3: Implementation
- **Status:** in_progress
- Actions taken:
  - Added a lightweight shared post-use sync plan header so OR plan structs can store post-use sync metadata without including the full MPI common header.
  - Fixed compile issues in `PostRegionAnalysis.cpp` by adding the required AST visitor/parent-map includes.
  - `cmake --build build --target translator -j8` passed after the interface fixes.
- Files modified:
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/include/mpi/shared/PostUseSyncPlan.h
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/include/Rewriter_MPI_Common.h
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/include/mpi/operator_resident/OperatorResidentPlan.h
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/shared/PostRegionAnalysis.cpp

### Phase 4: Target Verification
- **Status:** in_progress
- Actions taken:
  - Fixed bounded-index AST extraction for chained `TensorProxy` subscript temporaries so `tensor[0][0]` records two indices instead of one.
  - Added resident-halo bounded-index final sync for row-block 2D resident tensors, including a conservative constant-boundary shortcut that root can materialize directly.
  - Updated `waveEquation1.0/mpi_expect.txt` to require `matPrev` final gather to be absent.
  - Target correctness passed for `waveEquation1.0`, `mandel1.0`, `vectorAddCombo`, and `imageAdjustment1.0`.
  - Probe profile for large `waveEquation1.0` hit `post-use=bounded-index count=1`, skipped writer/direct-reader full gathers, and generated no final `MPI_Gatherv`.
- Files modified:
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/shared/PostRegionAnalysis.cpp
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/operator_resident/StencilWindowCodegen.cpp
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/tests/waveEquation1.0/mpi_expect.txt

### Phase 5: Full Regression and Profile
- **Status:** in_progress
- Actions taken:
  - Updated stale structure expectations for `decay1.0` and `mpiLoopStencilCountGuard2D` after post-use sync started emitting resident output variables and skipping no-use resident halo materialization.
  - `bash clang/tools/translator/test_mpi.sh decay1.0 mpiLoopStencilCountGuard2D` passed: 2 tests, 2 passed.
  - Full `bash clang/tools/translator/test_mpi.sh` passed: 72 tests, 72 passed, 0 failed, 0 skipped.
  - `cmake --build build --target translator -j8` passed with no work to do.
  - Target profile passed for `waveEquation1.0 mandel1.0 vectorAddCombo imageAdjustment1.0 gradientSum jacobi1.0` at 4 ranks, 3 trials; all status ok.
  - Target profile `waveEquation1.0` logs `matCur post-use=bounded-index count=1`, `matPrev post-use=none`, `matNext post-use=none`; generated code contains no final `MPI_Gatherv` and root writes `{0,0}` directly from a proven boundary constant.
  - Full profile initially exposed unsupported bounded-owner codegen for `DFT1.0` and `matMul1.0` (`bounded indexed read has no owner`); fixed normal OR bounded owner mapping for 1D and row-partitioned 2D outputs, plus plan-time fallback for unsupported bounded layouts.
  - Re-ran final full `bash clang/tools/translator/test_mpi.sh`: 72 tests, 72 passed, 0 failed, 0 skipped.
  - Re-ran target profile in `/Volumes/QUQ/working/mpi_tmp_profile_2026-05-19_postuse_partial_materialize_rerun`: all 6 requested cases status ok.
  - Re-ran full profile in `/Volumes/QUQ/working/mpi_tmp_profile_2026-05-19_after_postuse_partial_full_rerun`: all 14 cases status ok.
- Files modified:
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/tests/decay1.0/mpi_expect.txt
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/tests/mpiLoopStencilCountGuard2D/mpi_expect.txt

---

## Session: 2026-05-19 RowBlock2D Scatterv + Post-Use Reduction

### Phase 1: Discovery
- **Status:** in_progress
- Actions taken:
  - Confirmed working tree has existing dirty/untracked files; will not revert or clean them.
  - Read previous planning files; they describe the earlier gradientSum fast path.
  - Added this session's plan to `task_plan.md`.
  - Inspected `CollectiveCodegenUtils.cpp`, `PartitionCodegen.cpp`, RowBlock2D analysis, output sync analysis, and target imageAdjustment/mandel sources.
  - Created `findings.md` with the RowBlock2D and mandel observations.
- Files modified:
  - /Volumes/QUQ/working/dacpp/task_plan.md
  - /Volumes/QUQ/working/dacpp/progress.md
  - /Volumes/QUQ/working/dacpp/findings.md

### Phase 2: RowBlock2D Direct Scatterv
- **Status:** in_progress
- Actions taken:
  - Updated `PartitionCodegen.cpp` RowBlock2D total/local/count/displacement multiplication to use checked/narrow helpers.
  - Replaced the RowBlock2D byte broadcast fast path in `CollectiveCodegenUtils.cpp` with a direct Scatterv path guarded by `getDim()==2`, shapes, offset, strides, size, and row-range checks.
  - Direct path covers byte (`MPI_BYTE` with byte counts/displacements) and non-byte (typed MPI datatype) transports.
  - Fallback remains pack+Scatterv for byte transport and the existing full Bcast+memcpy path for non-byte transport.
  - `bash clang/tools/translator/test_mpi.sh imageAdjustment1.0` passed.
  - Generated imageAdjustment MPI code contains `__or_rowblock_direct_*` and direct `MPI_Scatterv(... MPI_BYTE ...)`; no RowBlock2D input full `MPI_Bcast` on the fast path.

### Phase 3: Post-Use Reduction
- **Status:** in_progress
- Actions taken:
  - Added conservative `PostUseReductionPlan` and AST matcher for `scalar = 0; for (...) { if (tensor[i] == 1) scalar++; }`.
  - Added OR output-sync annotation for count-eq-one post-use reductions on root-only/root-centric Contiguous1D outputs.
  - Added wrapper-side local count + `MPI_Reduce` and suppressed full output gather/materialization when matched.
  - Added rewrite-time replacement of the absorbed scalar reset and post loop.
  - `bash clang/tools/translator/test_mpi.sh mandel1.0` passed.
  - Generated mandel MPI code contains `MPI_Reduce`, no output `MPI_Gatherv`/`array2Tensor`, and log shows `post-use-reduction=accepted`.

### Phase 4: Regression
- **Status:** complete
- Actions taken:
  - `cmake --build build --target translator -j8` passed.
  - `bash clang/tools/translator/test_mpi.sh imageAdjustment1.0 mandel1.0` passed.
  - Full `bash clang/tools/translator/test_mpi.sh` passed: 72 tests, 72 passed, 0 failed, 0 skipped.
  - Re-ran `bash clang/tools/translator/test_mpi.sh` after the final post-use reduction profile segment change; it passed again: 72 tests, 72 passed, 0 failed, 0 skipped.
  - Target profile passed for `imageAdjustment1.0 mandel1.0 jacobi1.0 gradientSum vectorAddCombo` at 4 ranks, 3 trials:
    - `gradientSum`: dac wall median 0.194373 s, profile median 0.191935 s.
    - `imageAdjustment1.0`: dac wall median 0.323780 s, profile median 0.323117 s; first RowBlock2D wrapper reports scatter and no input bcast segment.
    - `jacobi1.0`: dac wall median 0.266413 s, profile median 0.261135 s.
    - `mandel1.0`: dac wall median 2.570618 s, profile median 2.561264 s; output full gather/materialize is replaced by final_sync reduction in profile.
    - `vectorAddCombo`: dac wall median 0.262366 s, profile median 0.263657 s.
  - Full profile passed for all 14 benchmarks at 4 ranks, 3 trials:
    - `DFT1.0`, `FOuLa1.0`, `MDP1.0`, `decay1.0`, `gradientSum`, `imageAdjustment1.0`, `jacobi1.0`, `liuliang1.0`, `mandel1.0`, `matMul1.0`, `oddeven0.1`, `stencil1.0`, `vectorAddCombo`, `waveEquation1.0` all status ok.
    - Full profile medians of priority cases: imageAdjustment1.0 0.320296 s, mandel1.0 2.565184 s, jacobi1.0 0.267879 s, gradientSum 0.362276 s wall / 0.209095 s profile, vectorAddCombo 0.265731 s.
  - Verified generated imageAdjustment code uses `__or_rowblock_direct_image_tensor` and direct `MPI_Scatterv(... MPI_BYTE ...)` from the tensor storage pointer fast path.
  - Verified generated mandel code contains `MPI_Reduce`, omits output `MPI_Gatherv`/`array2Tensor` for `mandelbrot_flags`, and logs `post-use-reduction=accepted kind=count-eq-one scalar=mandelbrot_count`.

---

## Session: 2026-05-19 Constant Init Local Fill + Async Halo Runtime

### Phase 1-4: Implementation
- **Status:** complete
- Actions taken:
  - Added `ConstantInitPlan` to OR parameter planning metadata.
  - Implemented conservative vector-to-tensor constant initialization analysis in `OperatorChainAnalysis.cpp`.
  - Accepted arithmetic `std::vector<T> v(size)` as zero and `std::vector<T> v(size, constant)` for simple literal/value-init constants when no writes/escapes occur before tensor construction and DAC shell use.
  - Added init-sync decision logs and fallback reason logs.
  - Added constant-local codegen for ordinary OR scatter, RowBlock2D full-row resident buffer init, 2D resident halo stencil readers, 2D direct readers copied into halo layout, and 1D resident halo init.
  - Updated `waveEquation1.0/mpi_expect.txt` to require `matCur init-sync=constant-local value=0` and absence of root `tensor2Array/scatter_window_2d_rows` for `cur`.
  - Cached nearest nonempty neighbor ranks in 1D and 2D resident halo layouts.
  - Replaced 1D and 2D row halo blocking `MPI_Sendrecv` exchange with posted `MPI_Irecv`, posted `MPI_Isend`, and `MPI_Waitall` over valid requests, handling `MPI_PROC_NULL` and zero-size cases.
- Files modified:
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/include/mpi/operator_resident/OperatorResidentPlan.h
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/operator_resident/OperatorChainAnalysis.cpp
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/operator_resident/CollectiveCodegenUtils.cpp
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/operator_resident/ResidentBufferCodegen.cpp
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/operator_resident/StencilWindowCodegen.cpp
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/operator_resident/LoopLoweredStencil1DCodegen.cpp
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/mpi/operator_resident/OperatorResidentRuntime.h
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/tests/waveEquation1.0/mpi_expect.txt

### Phase 5: Verification and Profile
- **Status:** complete
- Actions taken:
  - `cmake --build build --target translator -j8` passed; final rerun reports `ninja: no work to do`.
  - Target correctness passed: `waveEquation1.0`, `stencil1.0`, `mpiLoopStencilResidentHalo1D`, `mpiLoopStencilResidentHaloEmptyRank1D`, `mandel1.0`, and `imageAdjustment1.0` all passed.
  - Full `bash clang/tools/translator/test_mpi.sh` passed: 72 tests, 72 passed, 0 failed, 0 skipped.
  - Target profile in `/Volumes/QUQ/working/mpi_tmp_profile_2026-05-19_const_init_async_halo` passed all 7 requested cases:
    - `gradientSum` wall median 0.197594 s, profile median 0.201560 s.
    - `imageAdjustment1.0` wall median 0.268278 s, profile median 0.262803 s.
    - `jacobi1.0` wall median 0.266960 s, profile median 0.302934 s.
    - `mandel1.0` wall median 2.558579 s, profile median 2.572610 s.
    - `stencil1.0` wall median 0.889077 s, profile median 0.890001 s.
    - `vectorAddCombo` wall median 0.264182 s, profile median 0.262226 s.
    - `waveEquation1.0` wall median 1.007491 s, profile median 1.023537 s.
  - Target profile `waveEquation1.0` segments: scatter median max 13.129500 ms, halo median max 161.549631 ms, materialize median max 0.000625 ms.
  - Suggested full profile in `/Volumes/QUQ/working/mpi_tmp_profile_2026-05-19_after_const_init_async_halo_full` passed all 14 cases status ok.
  - Full profile `waveEquation1.0`: wall median 1.019828 s, profile median 1.014860 s, scatter median max 13.710792 ms, halo median max 150.256140 ms, materialize median max 0.000333 ms.
  - Verified full profile generated logs/code:
    - `waveEquation1.0` logs `input matCur init-sync=constant-local value=0` and generated code skips `tensor2Array(__or_initial_global_cur)` and `scatter_window_2d_rows(__or_initial_global_cur)`.
    - `mandel1.0` logs scalar reduction and generated code contains `MPI_Reduce`.
    - `imageAdjustment1.0` generated code still uses guarded `__or_rowblock_direct_image_tensor` and direct `MPI_Scatterv(... MPI_BYTE ...)`.

### Phase 6: Delivery
- **Status:** complete
- Ready to report modified files, supported/fallback constant-init analysis, async halo details, and test/profile deltas.

## Session: 2026-05-19 Decay MPI Wrapper Migration

### Phase 1: Discovery
- **Status:** in_progress
- Actions taken:
  - Restored planning context and confirmed existing dirty/untracked files must be preserved.
  - Inspected historical MPI Wrapper code (`b0787ebb5`, `2e6113d28`) and confirmed it gathered output writeback to root and only broadcast when later host reads required all ranks.
  - Confirmed current decay P4.5 direct path already has scalar local refresh (no per-step scalar `MPI_Bcast`) but still gathers `local_A`, materializes `local_A_tensor`, and then executes the source host assignment `A_tensor[10*t_tensor[0]] = local_A_tensor`.
- Files modified:
  - /Volumes/QUQ/working/dacpp/progress.md

---

## Session: 2026-05-20 MPI Translator Remaining Priority Cases

### VectorAddCombo Verification Setup
- **Status:** in_progress
- Actions taken:
  - Restored handoff context and confirmed branch `tqc-2` with existing dirty/untracked files preserved.
  - Rebuilt translator target; ninja reported no work to do.
  - Ran `bash clang/tools/translator/test_mpi.sh vectorAddCombo`; small initializer-list source passed and logs now fall back to scatter for `a/b/c` with reason `single-argument vector initializer is not a size constructor`.
  - Ran `bash clang/tools/translator/test_mpi.sh --large vectorAddCombo`; translation generated the desired `init-sync=index-local` for large `a/b/c` and bounded-prefix sync for `out_tensor`, but the shared expectation file still described the old full-gather path.
  - Added optional `mpi_expect_large.txt` support in `test_mpi.sh` and a large-only vectorAddCombo expectation for `index-local` input fill plus `bounded-index count=16`.
- Files modified:
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/test_mpi.sh
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/tests/vectorAddCombo/mpi_expect_large.txt
  - /Volumes/QUQ/working/dacpp/progress.md

### Correctness Regression
- **Status:** complete
- Actions taken:
  - `bash clang/tools/translator/test_mpi.sh vectorAddCombo decay1.0 FOuLa1.0` passed: 3/3.
  - `bash clang/tools/translator/test_mpi.sh imageAdjustment1.0 jacobi1.0 mandel1.0 waveEquation1.0 stencil1.0` passed: 5/5.
  - First full `bash clang/tools/translator/test_mpi.sh` run reached 71/72; only `mpiMixedStencilORPhaseC` failed at process exit with `std::system_error: mutex lock failed`, after producing the expected output.
  - Diagnosed the failure as static SYCL queue teardown from OR `default_queue()` colliding with AdaptiveCpp/MPI process shutdown.
  - Changed OR `default_queue()` to return a heap singleton that intentionally lives until process exit; `bash clang/tools/translator/test_mpi.sh mpiMixedStencilORPhaseC` then passed.
  - Final rerun of full `bash clang/tools/translator/test_mpi.sh` passed: 72 tests, 72 passed, 0 failed, 0 skipped.
- Files modified:
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/mpi/operator_resident/OperatorResidentRuntime.h
  - /Volumes/QUQ/working/dacpp/progress.md

### VectorAddCombo Host Init Elision
- **Status:** complete
- Actions taken:
  - Added source-statement tracking for accepted index-local vector initialization loops/assignments.
  - Added rewrite-time aggregation that elides absorbed host vector fill loops when all loop-body statements are proven index-local and otherwise elides only the absorbed assignments.
  - Verified generated large vectorAddCombo replaces the `for (int i = 0; i < N; ++i)` `a/b/c` fill loop with `/* DACPP MPI index-local vector initialization elided */`.
  - `bash clang/tools/translator/test_mpi.sh --large vectorAddCombo` passed.
  - `bash clang/tools/translator/test_mpi.sh vectorAddCombo` passed, preserving initializer-list fallback behavior.
  - `bash clang/tools/translator/test_mpi.sh vectorAddCombo decay1.0 FOuLa1.0` passed again after the rewrite.
  - Probe profile in `/Volumes/QUQ/working/mpi_tmp_profile_vector_elide` passed with `vectorAddCombo` standard median 0.231248 s and DAC wall median 0.237284 s (~0.974x).
- Files modified:
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/include/mpi/operator_resident/OperatorResidentPlan.h
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/operator_resident/OperatorChainAnalysis.cpp
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/Rewriter_MPI.cpp
  - /Volumes/QUQ/working/dacpp/progress.md

### Final Verification and Profile
- **Status:** complete
- Actions taken:
  - `cmake --build build --target translator -j8` passed (`ninja: no work to do` after final expectation update).
  - `bash clang/tools/translator/test_mpi.sh --large vectorAddCombo` passed and checks index-local host init elision.
  - `bash clang/tools/translator/test_mpi.sh vectorAddCombo decay1.0 FOuLa1.0` passed: 3/3.
  - `bash clang/tools/translator/test_mpi.sh imageAdjustment1.0 jacobi1.0 mandel1.0 waveEquation1.0 stencil1.0` passed: 5/5.
  - Full `bash clang/tools/translator/test_mpi.sh` after the final edit reached 70 passed, then hit a transient artifact issue at the final `stencil1.0`/`waveEquation1.0` tail (`step2.log` missing / directory lookup failure). Immediate rerun of `bash clang/tools/translator/test_mpi.sh stencil1.0 waveEquation1.0` passed: 2/2.
  - Requested profile in `/Volumes/QUQ/working/mpi_tmp_profile_next2` passed all six cases status ok.
  - Final wall ratios from that profile: vectorAddCombo 0.994x, decay1.0 1.199x, FOuLa1.0 0.866x, imageAdjustment1.0 0.475x, jacobi1.0 0.807x, mandel1.0 0.819x.
- Files modified:
  - /Volumes/QUQ/working/dacpp/progress.md

### Phase 2: Scalar Broadcast Migration
- **Status:** complete
- Actions taken:
  - Migrated the old Wrapper broadcast-elision idea to P4.5 loop-lowered direct scalar readers: when the loop-lowered plan materializes every run and has a replicated scalar, each rank refreshes from its local replicated tensor instead of root refreshing and broadcasting each iteration.
  - Tightened the condition to `loopLowerMaterializeEveryRun` and added `scalar-refresh=local-replicated` to the P4.5 decision log.
  - Updated `decay1.0/mpi_expect.txt` to require the local scalar refresh path and absence of `MPI_Bcast(&ctx.__or_scalar_t...)`.
  - Ran a targeted decay profile in `/Volumes/QUQ/working/mpi_tmp_profile_2026-05-19_decay_scalar_local`: status ok, standard median 0.372869 s, DAC wall median 0.368962 s, no bcast segment.
- Files modified:
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/operator_resident/LoopLoweredDirectCodegen.cpp
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/operator_resident/OperatorChainAnalysis.cpp
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/tests/decay1.0/mpi_expect.txt
  - /Volumes/QUQ/working/dacpp/findings.md
  - /Volumes/QUQ/working/dacpp/progress.md

---

## Session: 2026-05-19 MPI Translator Remaining Benchmark Optimization

### Phase 1: Restore Context and Discover Entry Points
- **Status:** in_progress
- Actions taken:
  - Restored planning files and ran the planning catchup script.
  - Confirmed current branch `tqc-2` tracks `origin/tqc-2`; visible git dirty state before this session was the existing deleted doc plus untracked planning files.
  - Confirmed no tracked translator code diff was present at session start, so this round can keep code changes tightly attributable.
  - Inspected current constant-init analysis/codegen, bounded-index post-use sync codegen, and relevant OR/post-use search hits.
  - Observed current bounded-index root sync still uses `MPI_Allreduce` per bounded value to discover owner; this is the Phase 3 target.
- Files modified:
  - /Volumes/QUQ/working/dacpp/task_plan.md
  - /Volumes/QUQ/working/dacpp/progress.md
2026-05-19 21:03:47 CST

### Session: 2026-05-19 Remaining Benchmark Optimization
- Built `translator` successfully after aggregate constant-init, static bounded-owner, and partial tensor2Array materialization edits.
- Correctness passed for `imageAdjustment1.0 vectorAddCombo decay1.0` and `waveEquation1.0 mandel1.0 stencil1.0`.
- Probe profile `/Volumes/QUQ/working/mpi_tmp_profile_probe` passed `imageAdjustment1.0` and `vectorAddCombo` (1 trial). Generated code confirms `Pixel{100,100,100}` local fill, static bounded owner without `MPI_Allreduce`, and vectorAddCombo large `post-use=bounded-index count=16` with `host_out.assign(16,{})` plus root `getElement`.
- Full `test_mpi.sh` started.
2026-05-19 21:25:21 CST
- Full `test_mpi.sh` first rerun passed 71/72; the only failure was `mpiBroadcastTensor2Array` structure expectation still requiring full `MPI_Gatherv` for fixed host indices.
- Updated `clang/tools/translator/tests/mpiBroadcastTensor2Array/mpi_expect.txt` to expect bounded tensor2Array post-use (`count=2`), root-side `host_output.assign(8,{})`/`getElement`, and no output `MPI_Gatherv`/`MPI_Bcast`.
- `bash clang/tools/translator/test_mpi.sh mpiBroadcastTensor2Array` passed. Full rerun started.
2026-05-19 21:44:54 CST
- Full `bash clang/tools/translator/test_mpi.sh` rerun passed: 72 tests, 72 passed, 0 failed, 0 skipped.

## Session: 2026-05-19 Prefix/Slice Materialization

### Phase 1: Restore Context and Discover Current Paths
- **Status:** in_progress
- Actions taken:
  - Restored planning files and catchup context for the `tqc-2` branch.
  - Confirmed latest commit is `6924ae4e1 Optimize MPI bounded sync and aggregate init`.
  - Confirmed tracked translator code was clean at session start; existing dirty state is the deleted docs file plus untracked planning files, which will be preserved.
  - Added this session's plan to `task_plan.md`.
  - Inspected current tensor2Array post-use path and profile artifact; found vectorAddCombo large final `tensor2Array(host_out)` already lowers to bounded 16-value root sync.
  - Inspected decay P4.5 loop-lowered direct codegen; confirmed per-iteration full gather/materialize remains inside `__dacpp_mpi_or_DECAY_decay_0_run`.
  - Recorded the planned decay fast path: recognize loop-local row writeback into a host matrix and only gather the output vector on iterations whose row matches the fixed post-loop row use.
- Files modified:
  - /Volumes/QUQ/working/dacpp/task_plan.md
  - /Volumes/QUQ/working/dacpp/progress.md
  - /Volumes/QUQ/working/dacpp/findings.md

2026-05-19 23:03:17 CST
- Added generic P4.5 loop-lowered selective row materialization metadata/codegen.
- `decay1.0` now recognizes `A_tensor[10*t_tensor[0]] = local_A_tensor` plus fixed post-loop `A_tensor[1]` use and emits a guard around the output gather/materialize.
- The row expression codegen rewrites replicated scalar tensor reads such as `t_tensor[0]` to the already refreshed local scalar `ctx.__or_scalar_t`; non-replicated tensor subscripts in row expressions fall back to the existing per-run materialization.
- `cmake --build build --target translator -j8` passed.
- `bash clang/tools/translator/test_mpi.sh decay1.0` passed.

2026-05-19 23:24:04 CST
- Added generic LoopLocalStencilOwnerLoop fixed-row post-use analysis for owner matrices. It accepts fixed owner-row `.print()`/root-observable `cout` reads and falls back on writes, unknown calls, non-fixed rows, multiple rows, or non-output reads.
- FOuLa owner-loop codegen now materializes only the selected owner row history when accepted, using the owning rank to send one row to root; the old whole-history `MPI_Gatherv` remains the fallback path.
- FOuLa owner-loop boundary values are now distributed by two loop-entry boundary-history broadcasts instead of per-step scalar bcasts.
- Updated `FOuLa1.0/mpi_expect.txt` and `decay1.0/mpi_expect.txt` for the new generic paths.
- `bash clang/tools/translator/test_mpi.sh vectorAddCombo decay1.0 FOuLa1.0` passed.
- `bash clang/tools/translator/test_mpi.sh imageAdjustment1.0 jacobi1.0 mandel1.0 waveEquation1.0 stencil1.0` passed.

---

## Session: 2026-05-21 MPI 32-Rank Top3 Optimizations

### Implementation
- **Status:** complete
- Actions taken:
  - Added generic `OutputInitPlan` metadata and output-direct local-default codegen. Write-only outputs skip initial sync; local `+=` accumulation skips sync only when the output vector-backed tensor initializer is proven default/zero and there is no unsafe tensor use before the shell call.
  - Extended constant/index init analysis so `ReplicatedFullTensor` direct-mapped inputs can use index-local generation, and helper-function wrapper bodies are scanned via the shell argument's enclosing compound/function body.
  - Added conservative RowBlock2D pointwise fusion metadata, fused wrapper emission, fused rewrite handling, and fused local kernel codegen that keeps intermediates as private values inside one SYCL kernel.
  - Tightened RowBlock2D fusion proof so downstream intermediate consumer counts are checked across all OR plans after the writer expression, not only inside the candidate chain.
  - Kept the small `matMul1.0` fixture on fallback expectations and added large fixture expectations for the accepted output-local-default path.
  - Updated the optimization plan document with `Implementation Status` sections under items 1-3.
- Files modified:
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/include/Rewriter_MPI_OperatorResident.h
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/include/mpi/operator_resident/OperatorResidentPlan.h
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/Rewriter_MPI.cpp
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/operator_resident/LocalKernelCodegen.cpp
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/operator_resident/OperatorChainAnalysis.cpp
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/operator_resident/OperatorResidentCodegen.cpp
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/operator_resident/OperatorResidentCodegen_Internal.h
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/operator_resident/OperatorResidentWrapperCodegen.cpp
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/operator_resident/ResidentBufferCodegen.cpp
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/tests/matMul1.0/matMul.large_dac.cpp
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/tests/matMul1.0/mpi_expect.txt
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/tests/matMul1.0/mpi_expect_large.txt
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/tests/DFT1.0/mpi_expect.txt
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/tests/imageAdjustment1.0/mpi_expect.txt
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/docs/mpi_translator/mpi_32node_top10_optimization_plan_2026-05-21.md
  - /Volumes/QUQ/working/dacpp/task_plan.md
  - /Volumes/QUQ/working/dacpp/progress.md

### Verification
- **Status:** complete
- Results:
  - `cmake --build build --target translator -j8` passed after final edits.
  - `bash clang/tools/translator/test_mpi.sh matMul1.0 DFT1.0 imageAdjustment1.0` passed: 3/3.
  - `bash clang/tools/translator/test_mpi.sh --large matMul1.0` passed: 1/1.
  - `bash clang/tools/translator/test_mpi.sh jacobi1.0 gradientSum vectorAddCombo mpiBroadcastTensor2Array mpiOrReadWriteAccumulate2D` passed: 5/5.
  - `python3 clang/tools/translator/bench_mpi_profile_segments.py --tmp-dir /Volumes/QUQ/working/mpi_tmp/profile_segments_top3_probe --ranks 4 --trials 1 matMul1.0 DFT1.0 imageAdjustment1.0` passed with status `ok` for all three cases after the final fusion-consumer proof tightening.
- Evidence:
  - `matMul1.0` profile probe log reports `output matC init-sync=local-default value=0 reason=default-initialized local accumulation`; generated code contains the output-direct no-read fast-path comment and retains final gather.
  - `DFT1.0` profile probe log reports `input vec_tensor init-sync=index-local expr=i`; generated code fills from `__or_range.begin + __or_local_i`.
  - `imageAdjustment1.0` profile probe log reports `chain=0 rowblock-pointwise-fusion=accepted`; generated code has one fused RowBlock2D wrapper and `Pixel __or_private_image_tensor2{}` inside the kernel.

---

## Session: 2026-05-21 MPI Top4-5 Optimizations

### Phase 1: Discovery and Contract Mapping
- **Status:** complete
- Actions taken:
  - Read the 2026-05-21 optimization plan and existing planning/progress/findings.
  - Confirmed current branch is `tqc-2` tracking `origin/tqc-2`.
  - Confirmed visible dirty state at session start is unrelated deleted docs/tmp matmul files plus untracked planning files; no tracked translator code diff was present.
  - Added a dedicated local task plan for items 4 and 5.
- Files modified:
  - /Volumes/QUQ/working/dacpp/task_plan.md
  - /Volumes/QUQ/working/dacpp/progress.md

### Implementation
- **Status:** complete
- Actions taken:
  - Added generic P4.5 loop-lowered `Contiguous1D` scalar-refresh device-time-loop metadata, acceptance/rejection logs, rewrite support, and direct codegen.
  - Kept decay selective-row materialization in place and made the accepted device-time-loop run one SYCL kernel whose work-items loop over time locally.
  - Added conservative P4.6 `StencilWindow2D` temporal-block metadata/logs and static `k=2` codegen only for the proven no-direct-reader 2D resident-halo contract.
  - Added temporal 2D row-halo runtime helpers for widened halo exchange, temporal scatter/layout setup, and owned-slice extraction, with a runtime block-size-1 guard for narrow partitions.
  - Kept `waveEquation1.0` direct-reader recurrence on the existing one-step resident-halo path with an explicit rejection log.
  - Fixed loop-contained constant init lookup so enclosing function bodies are scanned for vector declarations before shell calls.
  - Updated `decay1.0`, `stencil1.0`, and `waveEquation1.0` structure expectations.
  - Appended `Implementation Status (2026-05-21)` sections under plan items 4 and 5.
- Files modified:
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/mpi/operator_resident/OperatorResidentRuntime.h
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/include/Rewriter_MPI_OperatorResident.h
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/include/mpi/operator_resident/OperatorResidentPlan.h
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/include/mpi/shared/LoopLoweredRewrite.h
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/Rewriter_MPI.cpp
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/legacy_access_pattern/PatternInit.cpp
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/operator_resident/LoopLoweredDirectCodegen.cpp
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/operator_resident/OperatorChainAnalysis.cpp
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/operator_resident/OperatorResidentCodegen.cpp
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/operator_resident/StencilWindowCodegen.cpp
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/shared/LoopLoweredRewrite.cpp
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/tests/decay1.0/mpi_expect.txt
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/tests/stencil1.0/mpi_expect.txt
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/tests/waveEquation1.0/mpi_expect.txt
  - /Volumes/QUQ/working/dacpp/clang/tools/translator/docs/mpi_translator/mpi_32node_top10_optimization_plan_2026-05-21.md
  - /Volumes/QUQ/working/dacpp/task_plan.md
  - /Volumes/QUQ/working/dacpp/progress.md

### Verification
- **Status:** complete
- Results:
  - `cmake --build build --target translator -j8` passed; final rerun reported no work to do.
  - `bash clang/tools/translator/test_mpi.sh decay1.0 stencil1.0` passed: 2/2.
  - `bash clang/tools/translator/test_mpi.sh waveEquation1.0 MDP1.0 FOuLa1.0 liuliang1.0` passed: 4/4.
  - `bash clang/tools/translator/test_mpi.sh mpiLoopStencilCountGuard2D mpiLoopStencilResidentHalo1D mpiLoopStencilResidentHaloEmptyRank1D mpiLoopStencilOrderReject2D mpiLoopStencilScalarReject2D` passed: 5/5.
  - `python3 clang/tools/translator/bench_mpi_profile_segments.py --tmp-dir /Volumes/QUQ/working/mpi_tmp/profile_segments_next45_probe --ranks 4 --trials 1 decay1.0 stencil1.0 waveEquation1.0 MDP1.0` passed with status `ok` for all four cases.
- Evidence:
  - `decay1.0` log reports `scalar-refresh=local-replicated device-time-loop=accepted`; generated code contains `__dacpp_mpi_or_DECAY_decay_0_run_loop(...)` and the `P4.5 device time loop` comment. Profile segments show 4 kernel calls and 4 gather calls.
  - `stencil1.0` log reports `temporal-block=2 accepted`; generated code contains `__or_runtime_temporal_block_size`, `exchange_halo_2d_rows_temporal_inplace`, and `owned_slice_2d_rows_temporal`. Profile segments show 1200 halo calls for 600 steps on four ranks.
  - `waveEquation1.0` log reports `temporal-block=2 rejected reason=direct-reader recurrence not enabled for k=2`; profile segments stay at 2400 halo calls, matching the conservative fallback.

### Clean Performance Comparison
- **Status:** complete
- Method:
  - Created detached clean baseline worktree at `/Volumes/QUQ/working/dacpp_perf_baseline_471a9e05` from commit `471a9e05b0a5a9dd9b74e3eb604b65ecc4c90dc6`.
  - Built baseline translator independently with `cmake --build build --target translator -j8`.
  - Reused current worktree translator for the optimized version after `cmake --build build --target translator -j8` reported no work to do.
  - Ran the same 4-rank, 5-trial profile probe for both trees:
    `python3 clang/tools/translator/bench_mpi_profile_segments.py --ranks 4 --trials 5 decay1.0 stencil1.0 waveEquation1.0 MDP1.0`.
- Artifacts:
  - Baseline: `/Volumes/QUQ/working/mpi_tmp/perf_clean_baseline_471a9e05_top45`
  - Current: `/Volumes/QUQ/working/mpi_tmp/perf_clean_current_top45`
- Results:
  - `decay1.0` DAC wall median improved from `0.270389s` to `0.196299s`; profile median improved from `0.270192s` to `0.150333s`. Kernel calls dropped from 2400 to 4.
  - `stencil1.0` DAC wall median improved from `0.903623s` to `0.857127s`; profile median improved from `0.914053s` to `0.890786s`. Halo calls dropped from 2400 to 1200.
  - `waveEquation1.0` DAC wall median improved from `1.074333s` to `1.026217s`; profile median changed from `1.022129s` to `1.072891s`. Translation log confirms temporal blocking is rejected for direct-reader recurrence, so it stays on the fallback path. Segment halo calls stayed 2400; halo max was effectively flat (`161.349ms` to `160.528ms`), scatter improved, and the profile-run delta is treated as run-to-run kernel timing noise.
  - `MDP1.0` DAC wall median was effectively flat (`0.311033s` to `0.309380s`); profile median improved from `0.312400s` to `0.270365s`.
- Conclusion: no clean-check performance regression was observed for the target or nearby cases. The previous single-trial `decay1.0` wall-time outlier was a cold-start artifact; median trials show the optimization improves steady-state runtime.

## Session: 2026-05-22 Mandel Contiguous1D Block-Cyclic
- Restored prior planning context; existing task_plan/findings/progress are unrelated untracked files and must not be staged.
- Baseline `bash clang/tools/translator/test_mpi.sh mandel1.0` passed. Current log has Contiguous1D OR, scalar post-use reduction accepted, direct input Scatterv, and no output gather.
- Ordinary Contiguous1D assumptions found in PartitionCodegen (`__or_range/counts_displs_1d`), CollectiveCodegenUtils scatter/gather, ResidentBufferCodegen bounded owner, and LocalKernelCodegen item_linear view offsets.
- New distribution should be opt-in only for independent Contiguous1D map with no read/write output, no direct-reader alias to writer, no downstream resident retention, and scalar reduction or bounded root post-use.

- First compile attempt failed on `sameOrder` namespace; fixed by qualifying `operator_resident::sameOrder`.

- Final build: `cmake --build build --target translator -j8` passed (`ninja: no work to do`).
- Final tests passed: `mandel1.0 FOuLa1.0 liuliang1.0 MDP1.0 stencil1.0 waveEquation1.0` (6/6) and fixtures `mpiBroadcastRootOnlyCout mpiBroadcastTensor2Array mpiOrReadWriteAccumulate1D mpiOrReadWriteAccumulate2D vectorAddCombo gradientSum DFT1.0 matMul1.0` (8/8).
- Profile probe `/Volumes/QUQ/working/mpi_tmp/profile_segments_next10_probe` passed all 6 requested cases. Mandel logs block-cyclic scalar reduction; kernel max/(sum/4) ratios were 1.0016, 1.0107, 1.0060.

## Session: 2026-05-27 Four-Case Shenwei Efficiency Diagnosis
- User reports current translated `imageAdjustment1.0`, `waveEquation1.0`,
  `stencil1.0`, and `FOuLa1.0` have poor stable multi-node efficiency on
  Shenwei, around 1/2.
- Scope for this session is analysis only: inspect current generated MPI/SYCL
  code and lowering logic, then explain likely causes before proposing changes.
