# Findings

## 2026-05-09 Route B Handoff
- User goal is Phase 4.6 Route B resident halo, beginning with B1 1D stencil halo; do not modify matmul unless regression fails.
- Fresh `git status --short --branch` in this session showed `## tqc-2...origin/tqc-2` plus only untracked `findings.md`, `progress.md`, and `task_plan.md`; it did not show the previously noted staged Route A files. Continue preserving all uncommitted files and avoid reset/checkout.
- Existing Route A plan metadata already has explicit `OrLoopLowerKind` with `StencilResidentHalo` reserved, but current analysis always chooses `StencilFullSync` for P4.6 stencil candidates.
- Current 1D single-call stencil wrapper and loop-lowered full-sync family use `ContiguousView1D` over a full global reader vector. B1 should introduce explicit resident-halo storage/view instead of silently reusing the full-reader path.
- Existing P4.6 loop gate for 1D requires exactly one window reader, one WRITE-only direct writer, no direct readers, shell args declared before the loop, and no root-bridge/read-cache transition. This is a good starting boundary for B1, with Route A full-sync retained for rejected halo cases.

## 2026-05-09 Route B B1 Findings
- B1 resident halo is implemented only for loop-lowered `StencilWindow1D` with a single `writer -> reader` distributed followup mapping at offset `+1`, regular split stride `1`, window size `2` or `3`, no read-cache transition, no root bridge, native MPI element datatypes, and a loop body containing only the DAC expression plus post statements that OR can remove and internalize.
- B1 no longer requires reader and writer to be the same actual tensor; MDP/liuliang use separate `new_* -> state` followup tensors and are the intended accepted shape.
- For accepted B1, generated `run()` uses resident window storage, `ResidentHaloView1D`, local kernel, `apply_followup_1d`, and `exchange_halo_1d`. Per-run full reader `MPI_Bcast`, writer `MPI_Gatherv`, and followup full tensor `MPI_Bcast` are removed.
- Host-visible tensor semantics are restored in final `materialize()`: writer tensor is gathered, and reader tensor is rebuilt from root’s old full tensor plus gathered interior and supported boundary-local updates.
- Root-materialize/invariant/alias 1D tests remain Route A `StencilFullSync`; 2D `stencil1.0` and `waveEquation1.0` remain Route A full-sync fallback.
- Review follow-up fixed two B1 safety gaps: `exchange_halo_1d()` now communicates only with nearest non-empty owned-output ranks so tail empty ranks do not leave unmatched sends, and B1 accepts boundary-local updates only for literal left boundary target `0` that current codegen can apply correctly. Right-boundary or symbolic boundary expressions fall back to `StencilFullSync`.

## 2026-05-09 Mandel/Oddeven Efficiency Diagnosis
- Large benchmark numbers being explained: `mandel1.0` hand-written MPI 2.535479s vs DAC translated MPI 3.029412s; `oddeven0.1` hand-written MPI 1.895137s vs DAC translated MPI 5.617574s.
- `mandel1.0` translated code uses generic OR 1D direct semantics: rank 0 converts `complex_points` with `tensor2Array`, scatters the input, gathers flags, then broadcasts/materializes the full output tensor on all ranks. The hand-written MPI version computes the deterministic complex grid locally on every rank, slices it locally, and only gathers final flags to root.
- `mandel1.0` overhead is therefore mostly avoidable input distribution plus host-visible output broadcast/materialization. Kernel work is heavy enough that the overhead is only about 20%, not catastrophic.
- `oddeven0.1` translated code is still the pre-Phase-5 fixed-pair/fixed-block fallback shape. The first DAC site is loop-lowered but still does full input scatter and output gather/writeback every phase because partial exchange is disabled with reason `outer loop contains multiple DAC expressions`.
- `oddeven0.1` has two DAC expressions per phase. The second expression over `array_out_tensor[{1,N-1}]` still uses the legacy `AccessPattern`/`PackPlan` wrapper and rebuilds/gathers/scatters index lists and values every phase. The host loop then copies `array_out2_tensor` back into `array_tensor`.
- Hand-written `oddeven0.1` uses persistent rank-local chunks, one local SYCL kernel per phase, scalar `MPI_Sendrecv` boundary exchange with neighbors, and a single final `MPI_Gatherv`. The DAC translation instead repeatedly materializes root-visible tensor state, which is why it is about 3x slower.

## 2026-05-09 Matmul Efficiency Diagnosis
- Large benchmark numbers: `matMul1.0` hand-written MPI 5.921520s vs DAC translated MPI 0.902443s.
- The post-fix DAC `RowPartitionFullRow` code stores full payloads by unique indexed row/column, not per output item. For 2048x2048 on 4 ranks this changes per-rank A payload from the old conceptual `512*2048 output items * 2048` ints to `512 rows * 2048` ints, and B payload from duplicated per-output columns to one packed full column set.
- DAC generated code scatters only owned A rows and broadcasts a packed/transposed B-column payload. The local calc sees both `vecA[i]` and `vecB[i]` as contiguous vectors.
- The hand-written MPI reference computes the same row partition but keeps B in original row-major layout and reads `b[k * N + j]` inside the inner loop. For one output element, that is stride-2048 access through B. On the current backend this is much slower than DAC's contiguous packed-column payload.
- This is a benchmark/reference-shape advantage, not proof that the translator beats a tuned matmul. A hand MPI+SYCL implementation that pre-transposes/tiles B and uses USM/local memory would likely close or reverse the gap.

## 2026-05-09 Initial Context
- `git status --short` showed existing user changes in:
  - `clang/tools/translator/docs/mpi_translator/phase_4_6_optimization_plan_2026-05-09.md`
  - `clang/tools/translator/docs/mpi_translator/shell_derived_partition_implementation_plan_2026-05-07.md`
  - `clang/tools/translator/test_mpi.sh`
  - `clang/tools/translator/tests/mpiLoopChainReject1D/mpi_expect.txt`
  - `clang/tools/translator/tests/mpiLoopVariantInputReject1D/mpi_expect.txt`
- Canonical docs say Phase 4 and Phase 4.5 current acceptance slices are closed. P4.6 starts with A-1 helper extraction, then A0 measurement-only baseline, then A1 stencil full-sync skeleton.
- A-1 acceptance is pure refactor: generated Phase-C and P4.5 OR structure must remain equivalent, and no lowering gates or communication behavior should change.
- P4.6 route A/B constraints: no FixedBlock, no root-bridge stencil, no broad stencil route expansion, no host-visible semantic weakening.
- The duplicate A-1 rewrite slice is exactly:
  - insert `ctx` declaration and `init(ctx, args...)` before the outer loop,
  - replace the DAC expression with `run(ctx, args...)`,
  - insert `materialize(ctx, args...)` after the outer loop token.
- `rewriteStencilPhaseCSite()` has additional root/distributed post-region rewrites after this common slice; those must stay Phase-C local.
- `rewriteLoopLoweredOperatorResidentSite()` currently performs only the common slice after OR loop-lowered candidate gating.

## 2026-05-09 A0 Measurement Baseline
- Recorded A0 baseline in `clang/tools/translator/docs/mpi_translator/phase_4_6_a0_measurement_baseline_2026-05-09.md`.
- Current OR stencil communication counts per wrapper call:
  - `MDP1.0`: full reader bcast 1, writer gatherv 1, followup/read-cache bcast 1, direct-reader bcast 0, scalar bcast 0, separate final gather 0.
  - `liuliang1.0`: full reader bcast 1, writer gatherv 1, followup/read-cache bcast 1, direct-reader bcast 0, scalar bcast 0, separate final gather 0.
  - `stencil1.0`: full reader bcast 1, writer gatherv 1, followup/read-cache bcast 1, direct-reader bcast 0, scalar bcast 0, separate final gather 0.
  - `waveEquation1.0`: full reader bcast 1, direct-reader bcast 1, writer gatherv 1, followup/read-cache bcast 2, scalar bcast 0, separate final gather 0.
- Loop invocation counts are `MDP1.0` 1000, `liuliang1.0` 200, `stencil1.0` 10, `waveEquation1.0` 10.
- `mpirun -np 4` wall-clock samples:
  - `MDP1.0`: 0.87, 0.98, 0.90 s.
  - `liuliang1.0`: 0.21, 0.22, 0.21 s.
  - `stencil1.0`: 0.06, 0.06, 0.06 s.
  - `waveEquation1.0`: 0.06, 0.06, 0.07 s.

## 2026-05-09 A1 Initial Code Findings
- Worktree starts with existing A-1/A0 uncommitted changes plus planning files; preserve them.
- `OperatorResidentPlan.h` currently has P4.5 fields (`loopLowerCandidate`, `loopLowerOuterLoop`, `loopLowerMaterializeEveryRun`) but no P4.6-specific loop-lower metadata.
- `OperatorChainAnalysis.cpp` gates loop lowering only for `Contiguous1D` direct/resident; it logs `[P4.5]` and rejects all stencil layouts.
- `buildOperatorResidentWrapperCode()` currently routes all stencil layouts to single-call stencil wrappers before checking direct loop-lowered family codegen.
- `StencilWindow1DCodegen.cpp` single-call wrapper already contains the conservative full-sync semantics A1 must preserve: reader `tensor2Array` plus full `MPI_Bcast`, optional scalar bcast, local kernel, resident writer update, writer `MPI_Gatherv`/`array2Tensor`, then distributed followup/boundary full tensor `MPI_Bcast`.
- `rewriteLoopLoweredOperatorResidentSite()` uses the shared `LoopLoweredRewrite` helper and currently keys only off `loopLowerCandidate`; A1 should use explicit P4.6 metadata for stencil while keeping P4.5 direct behavior.
- A1 must reject loop-lowered stencil when shell call arguments are declared inside the target loop; otherwise `init()` inserted before the loop would reference out-of-scope variables. This conservatively keeps FOuLa-style inner-loop slice arguments on the single-call OR wrapper.
- Generated A1 full-sync family for `MDP1.0` hoists only metadata/allocation into `init()`; `run()` still does full reader `MPI_Bcast`, writer `MPI_Gatherv`, and followup full tensor `MPI_Bcast` every iteration.

## 2026-05-09 A2/A3 Initial Findings
- Existing `StencilWindow2D` single-call wrapper already has the A2 semantics to preserve: full window-reader `MPI_Bcast`, optional direct-reader full `MPI_Bcast`, local row-block kernel, writer `MPI_Gatherv`, read-cache transition full `MPI_Bcast`, and followup/boundary full `MPI_Bcast`.
- Existing 2D accepted apps:
  - `stencil1.0`: one window reader, one writer, no direct reader, distributed followup + boundary-local updates.
  - `waveEquation1.0`: one window reader, one direct reader, one writer, read-cache transition + distributed followup + boundary-local updates.
- A3 reader hoist must reject/avoid hoist for `stencil1.0` and `waveEquation1.0` because the window reader is updated by distributed followup and boundary-local updates each iteration.

## 2026-05-09 A2/A3 Final Findings
- `StencilWindow2D` loop-lowered full-sync can reuse the existing 2D helper functions in `StencilWindowCodegen.cpp`; `run()` aliases ctx state back to the old local names (`__or_output_rows`, `__or_output_cols`, `__or_global_*`) before invoking read-cache/followup/boundary materializers.
- For `stencil1.0` and `waveEquation1.0`, P4.6 accepts `StencilWindow2D` loop-lowering with `hoist-reader-sync=false`; reader and direct-reader full broadcasts remain in `run()`.
- A3 positive coverage uses a 1D scalar/root-materialize stencil with no followup/read-cache/boundary writes to the window reader; generated code puts `MPI_Bcast(ctx.__or_global_state.data(), ...)` in `init()` and omits the per-run `MPI_Bcast(__or_global_state.data(), ...)`.
- The old `tensorUsesDistributedFollowup()` answer was too broad for no-route supported stencil sites: it returned true for any distributed tensor. It now returns true only when a real followup mapping references the tensor, preserving existing distributed route behavior while letting no-route root-materialize OR paths classify output as root-only.
- `StencilWindow1DPartitionAnalysis` now accepts scalar root-materialize 1D stencil even when distributed site analysis is supported but has no route/read-cache/boundary work; this is still within the existing root-materialize slice and does not add root-bridge, halo, or broader route semantics.

## 2026-05-09 Review Follow-Up Findings
- A3 reader-sync hoist must treat actual tensor aliasing as loop-variant, even when there is no distributed followup/read-cache/boundary update and no explicit host assignment to the reader inside the loop.
- Conservative fix chosen: keep the P4.6 `StencilFullSync` loop-lowered family accepted, but force `hoistReaderSync=false` when `reader.actualTensorName == writer.actualTensorName`; `run()` then keeps the existing full reader `MPI_Bcast`.
- The alias refresh structure test demonstrates this behavior without adding root-bridge, FixedBlock, halo exchange, `PackPlan`, or Phase-C route semantics to the OR accepted path.
- Required new product files are staged so a submitted tracked diff includes the CMake-referenced helper/codegen sources and default-suite test directories; planning files remain local untracked notes.

## 2026-05-09 Reference Alias Follow-Up Findings
- Surface-name alias checking is insufficient for A3 hoist because `dacpp::Vector<double>& alias = state; aliasShell(state, alias, gain)` presents different shell argument names for the same object.
- `actualTensorName` now resolves simple C++ reference VarDecl initializers back to the base tensor key. The hoist gate also requires reader and writer keys to be precise and proven distinct; otherwise it keeps full reader refresh in `run()`.
- `mpiLoopStencilReferenceAliasRefresh1D` covers the reference alias shape and asserts P4.6 full-sync lowering is accepted with `hoist-reader-sync=false`, no init reader bcast, and no halo/FixedBlock/PackPlan path.

## 2026-05-09 Benchmark Issue Investigation
- New goal: diagnose/fix current generated `matmul2048` MPI `Scatterv` invalid count and determine whether `waveEquation1.0` large is slow-but-correct or a generated-code bug.
- Hard constraints remain: no Phase 5/FixedBlock, no root-bridge or broader stencil route, no Phase-C PackPlan/root-centric codegen in OR accepted path, no filename-based lowering, and no host-visible semantic weakening.
- `matmul2048` reproduces the MPI failure with the current generated binary: `MPI_Scatterv` reports `MPI_ERR_COUNT`.
- Generated `matmul2048` uses OR `RowPartitionFullRow` and allocates/scatters full-row/full-column payload per output item. For 2048x2048 output on 4 ranks, each rank owns `512*2048 = 1,048,576` output items and each full payload length is 2048, so the per-rank input payload count is `2,147,483,648`, exceeding MPI's `int` count range and wrapping negative.
- The bad count is independent of rank-specific shape: all non-empty ranks have the same overflowing payload count; `matC` OutputDirect itself scatters/gathers only `1,048,576` ints per rank and is not the overflow source.
- Fix direction: RowPartitionFullRow should keep one payload per unique indexed row/column, not one payload per output item. For row-partitioned 2D matmul, `matA[idx1][{}]` is scattered by owned output rows (`512*2048` ints/rank), while `matB[{}][idx2]` is the full set of output columns and is broadcast once to all ranks (`2048*2048` ints), then kernel view offsets reuse row/column payloads for each output item.
- `bash test_mpi.sh matMul1.0` passes with the new unique-payload layout.
- `waveEquation1.0` large generated code has no apparent deadlock/infinite loop: `TIME_STEPS=500` invokes one P4.6 `run()` per step. Each step performs full `matCur` Bcast, full `matPrev` direct-reader Bcast, local kernel, full writer Gatherv, read-cache transition full host loop+Bcast to `matPrev`, followup full host loop plus four O(N) boundary loops+Bcast to `matCur`, and `array2Tensor` refresh on all ranks. Final root `matCur.print()` exists, but the 300s timeout produced 0B output, so the timeout occurs before final print.
- `timeout 300 ... waveEquation1.0/mpi_bin >/tmp/wave_large.out` exits 124 and leaves no MPI process residue; `/tmp/wave_large.out` remains 0B.
- `matmul2048` post-fix efficiency samples (`/usr/bin/time -p`, `mpirun -np 4`, 5 runs) are 0.32, 0.32, 0.34, 0.31, 0.34 seconds; mean 0.326s, median 0.32s. All runs output `147385 147273 147961`.
- Per `waveEquation1.0` large step, generated full-sync collectives move two full `matCur` tensors (`2048*2048` doubles each), three interior tensors (`2046*2046` doubles each: direct-reader `matPrev`, writer gather, read-cache `matPrev` bcast), for `20,946,956` doubles or about 167.6 MB logical collective payload per step before MPI algorithm fanout. At 500 steps this is about 83.8 GB logical payload, plus host tensor copy/replay and SYCL buffer setup.
- Wave generated-code hot spots by line in `/Volumes/QUQ/working/p46_doc_bench_20260509_132128/waveEquation1.0/waveEquation1_0.mpi.dac_sycl_buffer.cpp`: reader `tensor2Array`/Bcast lines 123-127; direct reader lines 128-132; writer `Gatherv`/root `array2Tensor` lines 163-169; read-cache target copy/replay/Bcast/all-rank `array2Tensor` lines 171-189; followup target copy/replay/boundary/Bcast/all-rank `array2Tensor` lines 191-256.
- For this wave shape, the read-cache transition actually overwrites all of `matPrev` and followup+boundary overwrites all of `matCur`, but generated code uses generic partial-update materializers and therefore first copies the old target tensor with `tensor2Array`. That is conservative and correct, but slow; a future fix should use shape/coverage proof, not benchmark filename matching.

## 2026-05-09 B2 Scalar/Count Guard Fixes
- `StencilWindow2D` now rejects replicated scalar readers explicitly in `StencilWindowPartitionAnalysis.cpp`, and the P4.6 loop-lowered full-sync/B2 gates in `OperatorChainAnalysis.cpp` also reject them defensively. This closes the earlier accepted-analysis/broken-codegen gap even if layout analysis regresses again.
- `StencilWindowCodegen.cpp` now prevalidates 2D param sets before building wrappers/families and no longer leaves the mid-string `return` hazard inside the 2D kernel-view loops. Unsupported 2D param mixes now fail out before any wrapper text is emitted.
- Added runtime helpers in `OperatorResidentRuntime.h`: `checked_mul_int64_or_abort()` and `narrow_mpi_count_or_abort()`. `counts_displs_1d()`, `byte_counts_displs()`, `scatter_window_{1d,2d_rows}()`, and `exchange_halo_2d_rows()` now use them before any MPI `int` narrowing.
- New B2/full-sync 2D codegen in `StencilWindowCodegen.cpp` now guards shape-derived row-count products, displacements, gather sendcounts, and 2D broadcast counts with those helpers. Shared OR gather/scatter helper code in `CollectiveCodegenUtils.cpp` also now guards `__or_local_item_count` sendcounts.
- `mpiLoopStencilScalarReject2D` is a structure-only negative test that only asserts the 2D scalar site does not enter OR/P4.6/B2. The generated file is still allowed to contain legacy fallback structures because OR rejection correctly falls back to the older translator.
- `mpiLoopStencilCountGuard2D` is a structure-only large-shape test that keeps the accepted B2 `StencilResidentHalo` path and asserts the generated code contains `checked_mul_int64_or_abort(` / `narrow_mpi_count_or_abort(` and no raw `static_cast<int>` row-count/local-sendcount narrowing for the resident-halo 2D path.

## 2026-05-09 Count Guard Follow-Up Tightening
- `CollectiveCodegenUtils.cpp` previously still used unguarded `mpiPayloadCountExpr(...)` for the final `broadcastMaterializedOutput` `MPI_Bcast` on both root and non-root sides. It now uses guarded `checked_mul_int64_or_abort()` / `narrow_mpi_count_or_abort()` expressions there as well, and also guards the materialized output size before root `resize()`.
- Existing `jacobi1.0` structure coverage is now the shared broadcast regression anchor for this path: it no longer expects the old raw `MPI_Bcast(__or_materialized_x_new.data(), __or_arg3.getSize(), ...)` string and instead asserts the guarded materialized-output broadcast path is present.
- For 2D codegen, guard trigger points now happen before the extra allocation work called out in review:
  - `emitLoopLoweredInitFunction2D()` guards the full reader/direct-reader dense counts and local writer count before `tensor2Array()` / `resize()` / `assign()`.
  - `emitLoopLoweredRunFunction2D()` guards the per-run full reader/direct-reader dense counts before `tensor2Array()` / `resize()`.
  - `buildStencilWindow2DWrapperCode()` guards the full reader/direct-reader dense counts and local writer count before `tensor2Array()` / `resize()` / local output allocation.
  - `emitResidentHaloInitFunction2D()` guards resident local reader size, local writer size, and initial full reader dense count before local `assign()` and root `tensor2Array()`.
  - `emitResidentHaloMaterializeFunction2D()` guards the root gathered writer size before `resize()`.
  - 2D read-cache and followup broadcast helpers now guard non-root `resize()` before their `MPI_Bcast`.
- This does not make impossible oversized user tensors magically cheap to construct on the host side, but it does make the generated MPI layer abort on its own oversized communication buffers much earlier and more deterministically instead of allocating first and only guarding at the collective call.

## 2026-05-09 Route B B2 Completion / B3 Handoff
- Route B B2 is now complete for the intended narrow slice: `StencilWindow2D` row-block resident halo for `stencil1.0`, with exactly one 2D window reader, exactly one WRITE-only direct writer, current `+1,+1` distributed followup, and current boundary-local updates.
- B2 still intentionally rejects replicated scalar readers, direct readers, and read-cache transitions. `waveEquation1.0` remains on Route A `StencilFullSync` and is the primary B3 target.
- The last two high-severity overflow holes are closed:
  - the shared 2D materialize paths now compute `__or_materialized_output_items` with `checked_mul_int64_or_abort()` before passing the value into shared gather/materialize helpers, so the total-element guard no longer receives a raw `rows * cols` expression;
  - `resident_halo_2d_row_layout()` now computes `local_size` with `checked_mul_int64_or_abort()` before any later init/materialize allocation path observes it.
- Current B2 structural safety boundary is therefore:
  - no `AccessPattern`, `PackPlan`, `FixedBlock`, `root_bridge`, or Phase-C partial exchange in accepted OR/B2 generated code;
  - fallback to `StencilFullSync` whenever parameter shape, coverage proof, scalar/direct-reader presence, or resident-cache proof leaves the B2 slice.
- B3 should build on the new row-block resident-halo/runtime foundation rather than reopening any Route A/B1/B2 semantics. The remaining efficiency debt is mainly `waveEquation1.0` direct-reader resident state and `-1,-1` read-cache transition without regressing host-visible tensor semantics.
