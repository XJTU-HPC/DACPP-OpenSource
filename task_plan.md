# P4.6 Route B Work Plan

## Goal
Complete Phase 4.6 Route B resident halo optimization with conservative gates, preserving Route A full-sync fallback whenever B3 proof is incomplete.

## Constraints
- Preserve all existing uncommitted user changes.
- Do not advance Phase 5 or FixedBlock.
- Do not add root-bridge stencil support or broader stencil semantics.
- Do not bring Phase-C AccessPattern/PackPlan/root-centric gather-broadcast codegen into OR accepted paths.
- Do not delete legacy fallback.
- Do not identify lowering behavior by test filename.
- Do not trade host-visible tensor semantics for reduced communication.

## Phases
| Phase | Status | Notes |
|---|---|---|
| 1. Restore Route B context | complete | Ran `git status --short --branch`; re-read canonical docs and existing planning files. Current status output shows only planning files untracked. |
| 2. Inspect 1D resident-halo insertion points | complete | Read OR plan metadata, loop gates, StencilWindow1D/loop-lowered codegen, runtime helpers, kernel views, and structure tests. |
| 3. Add explicit Route B metadata/gate | complete | Added explicit `StencilResidentHaloMetadata` and conservative B1 gate for 1D READ_WRITE followup route shapes. |
| 4. Implement minimal B1 1D resident halo runtime/codegen | complete | Added resident halo view/runtime helpers and loop-lowered 1D resident halo family; Route A remains fallback when proof is incomplete. |
| 5. Add structure coverage | complete | Updated MDP/liuliang expectations and added `mpiLoopStencilResidentHalo1D`; alias/root-materialize and 2D/wave remain full-sync fallback. |
| 6. Verify requested suite | complete | Requested B1 build/tests passed; later Route B work preserved the coverage. |
| A1 historical code paths | complete | Read OR plan metadata, loop gates, StencilWindow1D codegen, rewrite dispatch, and test harness. |
| 3. Add P4.6 loop metadata/gate | complete | Added explicit `OrLoopLowerPlan` and conservative StencilWindow1D full-sync candidate gating. |
| 4. Generate StencilWindow1D family | complete | Added full-sync ctx/init/run/materialize family while preserving per-run full reader bcast, writer gatherv, followup bcast. |
| 5. Add structure tests | complete | Added P4.6 skeleton and no halo/FixedBlock/PackPlan expectations to MDP1.0 and liuliang1.0. |
| 6. Verify requested suite | complete | Build translator, requested MPI groups, and full 37-test suite passed. |
| 7. A2 inspect 2D stencil path | complete | Current 2D wrapper supports window reader, optional direct reader, writer, read-cache transition, followup, and boundary-local full-sync semantics. |
| 8. A2 generate StencilWindow2D family | complete | Added loop-lowered full-sync ctx/init/run/materialize in `StencilWindowCodegen.cpp`, preserving reader/direct-reader bcasts, writer gather, read-cache/followup/boundary full sync. |
| 9. A3 reader hoist | complete | `hoistReaderSync` moves only provably invariant window-reader full bcast to `init()`; loop-variant readers stay in `run()`. |
| 10. A2/A3 tests and verification | complete | Added/updated structure expectations and ran requested groups plus full 38-test MPI suite. |
| 11. Review blocker fixes | complete | Staged required new product files, blocked A3 reader hoist on reader/writer actual tensor alias, added alias refresh structure test, and reran requested groups plus full 39-test MPI suite. |
| 12. Reference alias hoist fix | complete | Resolved reference shell args to initializer base, required proven distinct reader/writer keys before hoist, added reference alias refresh test, and reran requested groups plus full 40-test MPI suite. |
| 13. Benchmark issue reproduction | complete | Reproduced matmul2048 MPI_ERR_COUNT; wave large timed out at 300s with 0B output and no MPI residue; inspected generated code. |
| 14. Matmul2048 count/shape fix | complete | Changed RowPartitionFullRow to store one payload per unique indexed row/column instead of per output item; large matmul now runs and small matMul passes. |
| 15. WaveEquation large diagnosis/fix | complete | No wave lowering change by user request; documented Route A full-sync large-performance debt for P4.6-B. |
| 16. Verification | complete | Build, matMul1.0, P4.6 stencil structure group, regenerated matmul2048 run and efficiency samples passed. |
| 17. Review fix: B1 halo/boundary safety | complete | Fixed empty-rank resident halo exchange, tightened B1 boundary gate to literal left boundary 0, added empty-rank runtime and right-boundary fallback tests. |
| 18. Filtered app benchmark and doc refresh | complete | Reran selected non-complex-stencil apps with the large-size `bench_mpi_only_requested.py` driver aligned with `docs/benchmarks`; updated P4.6 and shell-derived docs with B1 status, the hand-written MPI vs DAC translated MPI table, and B2 next-step boundary. |
| 19. B2 scalar/count guard fixes | complete | Rejected 2D scalar readers before OR/B2 loop lowering, added guarded MPI count/displ narrowing for B2/full-sync 2D paths and shared gather/scatter helpers, added `mpiLoopStencilScalarReject2D` and `mpiLoopStencilCountGuard2D`, and reran the requested validation groups plus `git diff --check`. |
| 20. Count guard follow-up tightening | complete | Guarded shared materialized-output broadcasts in `CollectiveCodegenUtils.cpp`, moved oversized 2D guard points ahead of full reader/direct/local buffer allocations in full-sync/plain/B2 codegen, refreshed `jacobi1.0` and `mpiLoopStencilCountGuard2D` structure assertions, and reran focused shared+B2 validation. |
| 21. B2 resident-halo completion and handoff | complete | Closed the remaining 2D checked-mul gaps at materialize/layout time, reran focused B2 validation, and marked Route B B2 complete with B3 handoff notes for `waveEquation1.0`. |
| 22. B3 resident direct-reader/read-cache completion | complete | Added a conservative `StencilWindow2D` B3 gate for one direct reader plus `(-1,-1)` read-cache, implemented resident direct-reader scatter/update/materialize in the 2D halo family, moved `waveEquation1.0` to Route B B3, and reran the requested serial validation commands plus `git diff --check`. |
| 23. B3 source-order guard | complete | Tightened the accepted `StencilWindow2D` B3 slice to require the current top-level `DAC -> read-cache -> followup -> boundary` source order, added `mpiLoopStencilOrderReject2D`, and reran the requested serial validation groups plus the new negative test. |
| 24. Full docs benchmark rerun and next-step prioritization | complete | Reran the full 14-case `docs/benchmarks` scale suite, inspected generated paths for `stencil1.0` / `waveEquation1.0` / `FOuLa1.0` / `oddeven0.1`, and updated P4.6 docs to prioritize 2D resident-state role rotation before broader gate work or Phase 5. |

## Errors Encountered
| Error | Attempt | Resolution |
|---|---|---|
| `LoopLoweredRewrite.cpp` could not call `BinaryOperator::getSourceRange()` because only `Stmt.h` was included. | Build attempt 1 | Added `clang/AST/Expr.h` to the helper implementation. |
| A3 positive test initially fell into Phase-C halo instead of OR loop-lowered path. | Test attempt 1 | Narrowed `tensorUsesDistributedFollowup()` to real followup mappings and allowed no-route scalar root-materialize 1D stencil to remain OR; updated test passed. |
| Alias-refresh test initially expected writer local buffer name from the actual tensor name. | Test attempt 1 | Matched existing codegen convention: writer buffers use shell/calc parameter names, so the assertion now checks `__or_local_next`. |
| `apply_patch` failed on docs because the expected text included a leading list marker that was not in the file. | Doc update attempt 1 | Re-read surrounding lines and applied the doc patch using the exact local text. |
| Ran `test_mpi.sh` from repo root instead of `clang/tools/translator/test_mpi.sh`. | B2 guard validation attempt 1 | Reran with the correct script path. |
| Parallel `test_mpi.sh` invocations clobbered the shared `/Volumes/QUQ/working/mpi_tmp` work dir. | B2 guard validation attempt 2 | Reran the requested groups serially so each script owned the temp dir. |
| Generated MPI code initially failed to compile because `[[noreturn]]` followed `inline` in `OperatorResidentRuntime.h`. | B2 guard build attempt 1 | Swapped it to `[[noreturn]] inline`, then rebuilt and reran all requested tests. |
| Two remaining B2 overflow reports showed checked guards still received raw 2D products and `resident_halo_2d_row_layout()` still multiplied before later guards. | B2 completion attempt 1 | Moved both 2D materialize call sites to a checked temporary and changed `resident_halo_2d_row_layout()` to compute `local_size` via `checked_mul_int64_or_abort()` before any allocation path. |
