# MPI Translator Current Status

Updated: 2026-05-10

This is the canonical current-state document for the new MPI translator model.
It replaces the phase handoff notes that were used while P4.6, P5, and P6 were
being closed.

The useful reading order is now:

1. This file for implementation status, accepted surfaces, and TODO.
2. `../benchmarks/mpi_efficiency_benchmark_summary.md` for benchmark history
   and current performance interpretation.
3. `mpi_performance_optimization_todolist.md` for the ordered optimization
   roadmap.
4. Source and `tests/*/mpi_expect.txt` for executable path expectations.

## Summary

The six-phase MPI model is complete for the currently proven surfaces.

The active model is shell-derived/operator-resident first. It uses the shell
split/bind structure to choose a distributed layout, emits local SYCL kernels
over resident or rank-local buffers, and falls back conservatively when a
required proof is missing.

The completed current surface includes:

- shell-derived OR layouts for direct 1D, row-block/row-payload 2D, replicated
  scalar/full-tensor readers, stencil windows, and the first FixedBlock slice
- loop-lowered `ctx/init/run/materialize` structure for stable loop sites
- P4.6 resident-halo stencil lowering for the proven 1D and 2D stencil slices
- P5 loop-resident `FixedBlockPhaseExchange` for the canonical `oddeven0.1`
  shape
- P6 lowering-contract metadata and consistency checks for accepted loop-lowered
  paths
- generic OR resident-buffer ownership optimization for ordinary OR chains
- a guarded `FOuLa1.0` owner-loop specialization for loop-local 1D stencil
  slices over a stable matrix owner
- default-off segmented MPI profiling and a Phase 1.4 benchmark/profile artifact
  driver
- conservative fallback to OR FullSync, Phase-C stencil, or legacy
  AccessPattern when proof is incomplete

This completion does not mean every possible DACPP shell shape is accepted by
the optimized path. The model is complete, and the accepted surface remains
intentionally narrow.

## Active Source Map

Main entry points:

- `translator.cpp`: CLI options, AST matcher, MPI enablement, final rewrite
  dispatch.
- `rewriter/lib/Rewriter_MPI.cpp`: top-level MPI rewrite orchestration.
- `rewriter/lib/mpi/shared/MpiPlanBuilder.cpp`: per-expression plan routing.
- `rewriter/include/Rewriter_MPI_Plan.h`: shared MPI lowering plan types.

Shared MPI analysis/codegen:

- `rewriter/lib/mpi/shared/OutputSyncAnalysis.cpp`: output visibility and final
  broadcast policy.
- `rewriter/lib/mpi/shared/ParamModeAnalysis.cpp`: read/write mode inference.
- `rewriter/lib/mpi/shared/LoopLoweredRewrite.cpp`: emits the loop-lowered
  `init`, `run`, and `materialize` call structure.
- `rewriter/include/mpi/shared/LoweringContract.h`: P6 lowering contract model.

Operator-resident path:

- `rewriter/include/mpi/operator_resident/OperatorResidentPlan.h`: OR IR,
  layouts, access plans, loop-lowered metadata, and resident facts.
- `rewriter/lib/mpi/operator_resident/ShellPartitionAnalysis.cpp`: shell/calc
  parameter analysis and layout selection.
- `rewriter/lib/mpi/operator_resident/OperatorChainAnalysis.cpp`: compatible
  chain construction, loop-lower annotation, P4.6 resident-halo selection, P5
  phase-exchange selection, and P6 contract checks.
- `rewriter/lib/mpi/operator_resident/*Codegen.cpp`: OR wrapper, local kernel,
  collective, resident-buffer, stencil, and FixedBlock code generation.

Runtime helpers:

- `dpcppLib/include/mpi/operator_resident/OperatorResidentRuntime.h`: rank
  ranges, counts/displacements, resident-halo layouts, halo exchange,
  owned-slice extraction, and FixedBlock phase exchange.
- `dpcppLib/include/mpi/common/Profile.h`: `DACPP_MPI_PROFILE=1` gated
  segmented profiling and legacy `collect_positions_for_item` diagnostics.
- `dpcppLib/include/mpi/legacy_access_pattern/*`: legacy AccessPattern/PackPlan
  runtime used by fallback wrappers.

Benchmark/profile helpers:

- `bench_mpi_only_requested.py`: enlarged wall-time benchmark driver.
- `bench_mpi_profile_segments.py`: Phase 1.4 companion driver that collects
  profile-off wall time, profile-on segmented logs, rank scaling, raw logs,
  generated source snapshots, and git/diff metadata.

Executable specifications:

- `tests/*/mpi_expect.txt`: expected logs and generated-code fragments for each
  accepted or rejected path.

## Lowering Decision Order

`buildMpiLoweringPlan()` currently routes expressions in this order:

1. Try shell-derived/operator-resident analysis.
2. If a stencil site cannot be handled by OR, use Phase-C stencil lowering.
3. Otherwise fall back to legacy AccessPattern/PackPlan wrapper lowering.

`rewriteMPI()` then emits wrappers and rewrites source expressions according to
the selected plan:

- loop-lowered OR: insert `ctx` and `init(...)` before the outer loop, replace
  the DAC expression with `run(...)`, and insert `materialize(...)` after the
  loop
- non-loop OR: replace the DAC expression with an OR wrapper call
- Phase-C stencil: use the Phase-C site rewrite
- legacy: replace with the generic AccessPattern wrapper call

## Accepted Surfaces

### P1-P3: Base Shell-Derived OR

The base OR path accepts currently supported shell-derived layouts and emits
rank-local kernels without legacy `AccessPattern`/`PackPlan` on accepted paths.

Representative accepted layouts:

- `Contiguous1D`
- `RowBlock2D`
- `RowPartitionFullRow`
- `ReplicatedScalar`
- `ReplicatedFullTensor`

Representative tests:

- `decay1.0`: `Contiguous1D`
- `jacobi1.0`: `RowPartitionFullRow` plus replicated full tensor
- `matMul1.0`: `RowPartitionFullRow`

### P4/P4.5/P4.6: Stencil OR And Resident Halo

The current optimized stencil surface is resident and loop-lowered.

Accepted 1D resident halo:

- stable loop site
- exactly one 1D stencil-window reader
- exactly one WRITE-only direct writer
- current `writer -> reader` followup route
- no direct reader
- no read-cache transition
- no root bridge
- provable supported boundary-local form

Representative tests:

- `MDP1.0`
- `liuliang1.0`
- `FOuLa1.0` uses the owner-loop specialization below, not the generic
  `ctx/init/run/materialize` stencil contract.

Accepted 2D resident halo:

- stable loop site
- row-block `StencilWindow2D`
- exactly one 2D stencil-window reader
- exactly one WRITE-only direct writer
- current `(+1,+1)` followup route
- current four boundary-local loops
- for the current direct-reader extension: at most one direct reader, exactly
  the current `(-1,-1)` read-cache transition, and the current top-level
  statement order

Representative tests:

- `stencil1.0`
- `waveEquation1.0`

Current implementation shape:

- initial root scatter of resident slices, including needed halo rows/items
- loop body computes directly into the next resident state
- in-place neighbor halo exchange
- role rotation with `swap()`
- final `Gatherv` of owned slices only, followed by required host-visible tensor
  materialization

### FOuLa Owner-Loop Specialization

`FOuLa1.0` is no longer handled by the ordinary per-step `StencilWindow1D`
wrapper. The generic P4.6 loop contract still rejects it because the shell
arguments are loop-local temporary slices, but `rewriteMPI()` now recognizes a
strict owner-loop shape and replaces the whole source loop with a generated
owner-loop function.

Accepted current shape:

- one shell-derived `StencilWindow1D` reader, one WRITE-only direct writer, and
  one replicated scalar reader
- all three calc element types are the same native MPI datatype
- the outer loop constructs `reader = owner[{}][k]`
- the writer slice is `writer = owner[{1,...}][k+1]`
- the scalar shell argument is built from a loop-local vector whose sole
  payload is the detected scalar expression
- the loop writes the computed writer slice back into `owner[*][k+1]`
- the current loop body has the proven seven top-level statements used by
  `FOuLa1.0`

Current implementation shape:

- scatter the initial owner column once into rank-local stencil windows
- compute each time step into resident local state
- broadcast only the two physical boundary scalar values per step
- exchange neighbor halos in place
- gather local owned history once at the end and update the owner matrix on
  root, preserving the existing host-visible `u_tensor` semantics

### P5: FixedBlock First Slice And Phase Exchange

The standalone FixedBlock first slice is a conservative fallback for a flat 1D
`RegularSplit(2,2)` shape.

Accepted standalone FixedBlock:

- exactly one 1D READ fixed-block input
- exactly one 1D WRITE fixed-block output
- one split per shell parameter
- block size equals stride
- current accepted block size is `2`
- no payload dimensions
- no multi-split parameters
- no READ_WRITE FixedBlock parameters
- native MPI element datatype

The optimized P5 path is the loop-resident phase-exchange lowering for the
canonical `oddeven0.1` shape.

Accepted phase exchange:

- stable outer loop
- exactly two FixedBlock DAC expressions in the loop body
- both expressions use block size 2 and stride 2
- phase B reads the phase A output through the recognized shifted helper slice
- phase shift offset is the integer literal `1`
- canonical interior copy and boundary preservation statements are present
- no post-loop use or alias of removed helper state
- statically proven even total, with runtime total checks before scatter

Current implementation shape:

- one initial `Scatterv` into resident rank-contiguous slices
- per iteration local phase A and phase B compare-swap work
- non-blocking exchange of cross-rank boundary elements
- final `Gatherv` plus broadcast to preserve host-visible source tensor
  semantics

Representative test:

- `oddeven0.1`

### P6: Lowering Contract Infrastructure

P6 is complete as infrastructure. It does not widen the accepted surface by
itself.

Each accepted loop-lowered path declares:

- which source statement is replaced
- which source statements are removed
- which tensors become resident state
- which host-visible tensors are materialized
- materialization timing
- compile-time fallback guards
- runtime abort guards
- accepted/rejected reason text for tests and review

The consistency checker validates current accepted facts and logs
`contract-check=pass reason=...` for the accepted paths. The checker is
diagnostic. It does not choose code generation.

## Important Fallbacks

Fallback is part of correctness, not an error.

Current fallback families:

- OR FullSync for stencil shapes that are shell-derived but not proven safe for
  resident halo
- Phase-C stencil for stencil sites not covered by OR
- legacy AccessPattern/PackPlan wrapper for general shapes outside OR support
- standalone P5 FixedBlock when the phase-exchange gate rejects

The legacy path remains intentionally available. It is slower on simple regular
patterns because it builds and communicates global-index pack metadata, but it
keeps unsupported shapes correct.

## Profiling Status

Segmented profiling is implemented for the current focused paths and is quiet by
default. Set `DACPP_MPI_PROFILE=1` on the generated MPI executable to emit
stderr rows in this format:

```text
DACPP_MPI_PROFILE	<label>	<segment>	calls=<n>	max_ms=<ms>	sum_ms=<ms>
```

Current segment names are `init`, `scatter`, `pack`, `kernel`, `halo`,
`gather`, `bcast`, `materialize`, and `final_sync`.

The focused Phase 1.4 artifact is:
`/Volumes/QUQ/working/mpi_tmp/phase1_4_profile_focus`.

The legacy fallback smoke artifact is:
`/Volumes/QUQ/working/mpi_tmp/phase1_4_legacy_profile_smoke`.

Profile `sum_ms` is summed across ranks and should be used for attribution, not
as external wall time. External wall time remains the `mpirun` timing in
`summary.tsv`/`results.tsv`.

## Current Case Map

| Case | Current path | Notes |
|---|---|---|
| `DFT1.0` | OR or legacy depending on current generated path | No focused `mpi_expect.txt` in tree. |
| `FOuLa1.0` | guarded owner-loop `StencilWindow1D` specialization | Replaces loop-local slice wrapper with resident local state, per-step halo exchange, and final history gather. |
| `MDP1.0` | P4.6 `StencilWindow1D` resident halo | Near hand-written MPI in closeout benchmark. |
| `decay1.0` | loop-lowered `Contiguous1D` | Materializes per run because scalar/host-visible semantics require it. |
| `gradientSum` | OR or legacy depending on current generated path | No focused `mpi_expect.txt` in tree. |
| `imageAdjustment1.0` | OR or legacy depending on current generated path | Semantics of hand-written reference were aligned after baseline. |
| `jacobi1.0` | `RowPartitionFullRow` + `ReplicatedFullTensor` | Full-vector broadcast is expected by the current accepted shape. |
| `liuliang1.0` | P4.6 `StencilWindow1D` resident halo | Current path uses in-place halo exchange and role rotation. |
| `mandel1.0` | OR or legacy depending on current generated path | Needs a current path-specific benchmark check. |
| `matMul1.0` | `RowPartitionFullRow` | Broadcasts one matrix/payload side by current row-partition design. |
| `oddeven0.1` | P5 loop-resident `FixedBlockPhaseExchange` | Current optimized path removes per-iteration materialization. |
| `stencil1.0` | P4.6 `StencilWindow2D` resident halo | Current closeout path is near hand-written MPI. |
| `vectorAddCombo` | `Contiguous1D` OR chain | Uses resident-buffer reference/move optimization for chain intermediates. |
| `waveEquation1.0` | P4.6 `StencilWindow2D` resident halo with direct-reader extension | Triple resident role rotation. |

## TODO

Keep future performance work in
`docs/mpi_translator/mpi_performance_optimization_todolist.md`. That roadmap is
ordered as:

1. segmented profiling
2. legacy AccessPattern simple contiguous fast paths
3. resident-chain and materialization elimination
4. generalized FOuLa loop-local argument contract
5. stencil and FixedBlock surface expansion

The core rule is that performance optimization must not change current
translation correctness. Every new fast path needs a proof, a conservative
fallback, structure tests, output comparison, and benchmark/profile evidence.
