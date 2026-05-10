# Shell-Derived Partition Implementation Status

Updated: 2026-05-10 (P6 lowering-contract infrastructure closeout)

## 1. Purpose

This document is the current-state handoff for the shell-derived MPI translation path. It
records what is already accepted, what remains intentionally out of scope, and where the
next phase boundary now sits after the Phase 5 closeout.

The document is written as a one-shot status report rather than a step-by-step build log.

## 2. Non-Negotiable Constraints

The accepted OR path continues to preserve these boundaries:

- no root-bridge stencil
- no Phase 5 `FixedBlock` beyond the currently proven first slice and the
  loop-resident phase-exchange slice for `oddeven0.1`
- no root-centric accepted `FixedBlock`
- no accepted-path reuse of legacy `AccessPattern` / `PackPlan`
- no accepted-path return to root-centric gather/broadcast stencil codegen
- no filename-based lowering decisions
- no weakening of host-visible tensor semantics
- no deletion of legacy fallback

If proof is incomplete, the implementation must fall back to the existing OR full-sync
path or the older fallback path.

## 3. Phase Status

| Phase | Status | Accepted result | Intentionally deferred |
|---|---|---|---|
| Phase 4 | closed | current OR stencil site acceptance, mixed-site expr dispatch, wrapper-internal followup/read-cache/boundary lowering for the current proven slice | broader stencil semantics, root-bridge |
| Phase 4.5 | closed | current loop-lowered direct/resident family for stable `Contiguous1D` sites | broader loop families, dynamic-shape and ownership expansion |
| Phase 4.6 | closed for the current proven slice | loop-lowered stencil resident-halo family, current 1D and 2D resident-state rotation, current `waveEquation1.0` direct-reader/read-cache extension, conservative soundness guards | `FOuLa1.0` rewrite-contract expansion, broader direct-reader/cache forms, root-bridge |
| Phase 5 | complete for the current first slice | `oddeven0.1` now enters the loop-resident `FixedBlockPhaseExchange` path; sites that fail the phase-exchange gate fall back to the standalone P5 wrapper without legacy `AccessPattern` / `PackPlan` | overlapping regular windows, non-shifted phase patterns, read-write fixed blocks, root-bridge |
| Phase 6 | closed for the current infrastructure slice | unified lowering-contract metadata for P4.6/P5 loop-lowered OR paths, contract-driven removal for proven-equivalent P4.6/P5 paths, and lightweight consistency-check logs | expanding P5 shapes, `FOuLa1.0` acceptance, payload/multi-split/read-write FixedBlock, codegen semantic changes |

## 4. Current Accepted P4.6 Surface

### 4.1 1D resident-halo slice

The accepted 1D loop-lowered slice is still intentionally narrow:

- stable loop site
- exactly one 1D window reader
- exactly one WRITE-only direct writer
- current `writer -> reader` followup shape
- no direct reader
- no read-cache transition
- no root-bridge
- current boundary-local form remains provable

The important implementation point is that the accepted path is now a rotated resident
state, not the earlier local-copy-heavy version:

- kernel writes to a full next-state buffer
- halo exchange updates that next-state buffer in place
- reader/writer roles rotate by swap
- final materialization gathers only the owned interior slice

This is the path used by `MDP1.0` and `liuliang1.0`.

### 4.2 2D resident-halo slice

The accepted 2D slice also remains narrow:

- stable loop site
- row-block `StencilWindow2D`
- exactly one 2D window reader
- exactly one WRITE-only direct writer
- current `(+1,+1)` followup shape
- current four boundary-local loops

For the base 2D resident path, the implementation now uses in-place halo exchange and
full-state rotation instead of per-step slab copying. This is the current `stencil1.0`
shape.

### 4.3 Current 2D direct-reader extension

The only accepted direct-reader/read-cache extension inside P4.6 is the current
`waveEquation1.0`-shaped slice:

- at most one direct reader
- current `(-1,-1)` read-cache transition
- current top-level `DAC -> read-cache -> followup -> boundary` statement order
- current proven actual-tensor distinctness and shape proof

The implementation keeps:

- resident `matCur`
- resident direct-reader slice for `matPrev`
- resident local writer slice for `matNext`
- triple-buffer role rotation across `prev/cur/next`

Anything outside this exact proof boundary falls back to Route A `StencilFullSync`.

## 5. Safety Guards That Define the Boundary

The closeout is considered sound because the following guards are part of the accepted
surface, not optional extras:

- 2D scalar readers are rejected before accepted-path lowering
- MPI count/displacement products are checked before `int` narrowing, including the
  current P5 FixedBlock materialized broadcast count
- 1D empty-rank halo exchange routes through the nearest non-empty owned rank
- unsupported 1D right-boundary forms fall back to full-sync
- current 2D B3 sites must preserve `DAC -> read-cache -> followup -> boundary` order
- P5 phase-exchange sites reject post-loop use of the phase-A output, including
  pre-loop initializer aliases
- P5 phase-exchange requires the phase-A output total to be statically even and
  the generated init checks reader runtime total, writer runtime total, and the
  proven even total before the first scatter

This is the line between the accepted OR path and the fallback path.

## 6. Benchmark Snapshot

### 6.1 Benchmark sources

- old wrapper baseline:
  `clang/tools/translator/docs/benchmarks/mpi_sycl_efficiency_benchmark_2026-05-06.md`
- current closeout benchmark:
  `/Volumes/QUQ/working/mpi_tmp/p46_final_close/results.tsv`

### 6.2 Comparison

| Case | Old wrapper / DAC-MPI (s) | Current P4.6 DAC-MPI (s) | Improvement vs old wrapper |
|---|---:|---:|---:|
| `FOuLa1.0` | 2.345472 | 1.663965 | 1.41x |
| `MDP1.0` | 0.801768 | 0.781158 | 1.03x |
| `liuliang1.0` | 0.949857 | 0.941800 | 1.01x |
| `stencil1.0` | 4.614142 | 1.462638 | 3.15x |
| `waveEquation1.0` | 4.790831 | 1.510119 | 3.17x |

The main readout is straightforward:

- 1D P4.6 is effectively at parity for `MDP1.0` and `liuliang1.0`
- 2D P4.6 moved `stencil1.0` and `waveEquation1.0` from clearly wrapper-bound to near
  parity with hand-written MPI
- `FOuLa1.0` improved only because the surrounding OR path is healthier; it is not yet a
  member of the accepted loop-lowered resident family

## 7. `FOuLa1.0` and Phase Ownership

`FOuLa1.0` should not be treated as a Phase 5 item.

Its blocker is the current loop-lowered rewrite contract: `init(ctx, original shell
args...)` is inserted before the outer loop, but `FOuLa1.0` builds its shell args
(`u_kin`, `u_kout`, `r`) inside the loop body. Accepting it without changing the
contract would generate an out-of-scope init call.

So the ownership is:

- not current accepted P4.6
- not Phase 5 `FixedBlock`
- a later P4.6-style rewrite/codegen contract expansion

Until that work exists, the correct behavior is to keep `FOuLa1.0` on the existing OR
wrapper path.

## 8. Phase 5 FixedBlock First Slice

The standalone P5 first slice is intentionally narrow and is now used only as
the conservative fallback when the phase-exchange gate rejects:

- exactly one 1D READ fixed-block input
- exactly one 1D WRITE fixed-block output
- `RegularSplit` size equals stride
- current accepted block size is `2`
- matching fixed-block split metadata across the accepted params
- exactly one split per accepted shell param
- 1D wrapper tensor rank and 1D calc view rank
- no payload/void dimensions
- no READ_WRITE fixed-block params
- native MPI element datatype

The standalone fallback codegen partitions the fixed-block item space by rank,
scatters contiguous block payloads, launches the existing local calc through
`ContiguousView1D`, gathers the block output, materializes the host-visible
output tensor, and broadcasts the materialized tensor so later host-side tensor
code sees the same state on all ranks.

Overlapping regular windows such as `split(3,2)`, explicit payload shapes such
as `tensor[{S1}][{}]`, and implicit non-flat Matrix payloads such as
`input[{S1}]` remain fallback.

## 9. Phase 5 FixedBlock Loop-Resident Phase-Exchange Slice

The `oddeven0.1`-shaped loop is now lifted to a loop-resident phase-exchange
wrapper. The accepted shape is intentionally narrow and is the only currently
proven entry point:

- stable outer for/while loop, exactly two DAC expressions in the loop body,
  both `FixedBlock` block-size 2 stride 2 with shared shell+calc names
- shell signature has exactly one READ block input and one WRITE block output,
  with both shell args of the first DAC expression declared before the loop
- the second DAC expression's reader is initialized inside the loop body from
  `<phaseAOutput>[{1, end}]`; only an integer-literal phase-shift offset of `1`
  is accepted
- the loop body matches the canonical sequence:
  1. first DAC expression (phase A on the source tensor)
  2. tensor decl initialized from `<phaseAOutput>[{1, end}]`
  3. transient vector decl
  4. transient tensor decl wrapping the vector
  5. second DAC expression (phase B on the shifted slice)
  6. inner for-loop writing `<source>[i] = <phaseBWriter>[i-1]` for
     `i = 1; i < N - 1; i++`
  7. boundary copy `<source>[0] = <phaseAOutput>[0]`
  8. boundary copy `<source>[N-1] = <phaseAOutput>[N-1]`

The accepted codegen emits a single ctx/init/run/materialize family at the
first DAC expression's wrapper name. `init` scatters the source tensor once
across ranks via a contiguous range partition (the rank partition does not
need to be block-aligned) and seeds the resident local buffer. `run` performs
phase A locally on every iteration, then phase B with non-blocking cross-rank
boundary exchanges between adjacent owning ranks so that the odd pair that
spans a rank boundary is handled correctly. The gate requires the phase-A
output tensor's underlying vector total to be statically proven even, and the
generated init checks that the source reader runtime total, phase-A writer
runtime total, and proven even total all match before scatter. Odd totals or
runtime mismatches therefore cannot enter `fixed_block_phase_exchange_step`.
`materialize` gathers the resident state to root and broadcasts it for
host-visible tensor semantics.
The follower DAC expression and helper statements are removed at rewrite
time, so no `MPI_Bcast`, `MPI_Gatherv`, or materialized broadcast happens
inside the loop body.

When the phase-exchange gate rejects, both sites continue to fall through to
the standalone P5 first-slice wrapper.

## 10. Phase 5 Completion Definition

Phase 5 is complete for the current first slice. The completed scope is:

- standalone FixedBlock first-slice fallback for flat 1D `RegularSplit(2,2)`
  with one READ input and one WRITE output
- loop-resident `FixedBlockPhaseExchange` for the canonical `oddeven0.1`
  two-phase pattern
- no accepted-path dependency on legacy `AccessPattern` / `PackPlan`
- no per-iteration gather/materialize/broadcast in the accepted phase-exchange
  loop body
- rank-contiguous resident partition with non-blocking cross-rank boundary
  exchange
- conservative rejection for payload/multi-split, Matrix payload, overlapping
  windows, wrong phase-B args, missing boundary copies, post-output use/alias,
  and odd or mismatched totals

Further FixedBlock shape widening is not part of this P5 closeout.

## 11. Phase 6 Closeout

Phase 6 now closes the first shared lowering-contract infrastructure slice for
OR loop-lowered paths. The target remains infrastructure, not a wider accepted
surface.

Each current accepted lowering now declares:

- which source statements are removed or replaced by the rewrite
- which tensors become resident state and which host-visible tensor is
  materialized at loop exit
- whether materialization is required before, during, or after the loop
- which post-loop uses, aliases, writes, and loop-local argument lifetimes must
  reject or fall back
- which shape checks are compile-time fallback gates and which are runtime abort
  guards
- the accepted/rejected reason text needed for review and regression tests

P6 also adds a lightweight consistency checker on the accepted contract facts.
The checker validates:

- contract enabled/name matches the accepted lowering kind
- the source DAC is a `Replace` statement
- the remove-list matches the current proven source of truth: legacy P4.6
  removal set for stencil paths, follower metadata for P5 phase exchange
- materialization timing matches the current path (`loop-exit` resident halo and
  phase exchange, `every-run` full-sync)
- materialized tensors are also declared as resident tensor facts
- statement, materialization, and guard reason metadata is present
- compile-time fallback guards and required runtime abort guards are represented,
  including 1D FullSync replicated-scalar and 2D FullSync/runtime-count cases

The checker is diagnostic. It logs `contract-check=pass reason=...` for current
accepted paths and does not choose codegen. P4.6 removal still uses contract
Remove statements only when the legacy-vs-contract removal set matches; mismatch
falls back to the legacy remover.

The current P4.6 contract construction and the legacy removal-set calculation
intentionally share the same source: both derive removable followup/read-cache
statements and boundary-local statements from `analyzeDistributedStencilSite()`.
Focused accepted fixtures therefore cover the `contract-removal-set=match` path.
The `contract-removal-set=mismatch` fallback is a defensive compatibility path
for a future contract-construction evolution that diverges from the legacy
source of truth. No current natural accepted source shape can produce that
mismatch without changing production behavior or artificially corrupting
contract construction, so this closeout does not add a mismatch fixture. P5
standalone fallback remains intact.

P6 non-goals remain:

- do not widen P5 FixedBlock beyond the closed first slice
- do not accept payload, multi-split, overlapping, or READ_WRITE FixedBlock
- do not absorb `FOuLa1.0` until the rewrite/init argument ownership contract is
  explicit
- do not weaken the P4.6 resident-halo accepted boundary
- do not change generated-code semantics

## 12. Handoff Direction

The current handoff is:

1. treat P4.6 as closed for the currently proven resident slice
2. treat P5 as complete for the current first slice and keep its gate intact
3. treat P6 lowering-contract infrastructure as the guardrail for future
   loop-lowered OR work
4. keep `FOuLa1.0` as a later targeted infrastructure or efficiency item
5. continue measuring and pinning focused contract logs before relaxing gates

That order keeps the semantic boundary stable and avoids mixing a rewrite-contract change
into the Phase 5 surface.
