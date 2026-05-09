# Shell-Derived Partition Implementation Status

Updated: 2026-05-10

## 1. Purpose

This document is the current-state handoff for the shell-derived MPI translation path. It
records what is already accepted, what remains intentionally out of scope, and where the
next phase boundary now sits after the Phase 4.6 closeout.

The document is written as a one-shot status report rather than a step-by-step build log.

## 2. Non-Negotiable Constraints

The accepted OR path continues to preserve these boundaries:

- no root-bridge stencil
- no Phase 5 `FixedBlock`
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
| Phase 4.6 | closed for the current proven slice | loop-lowered stencil resident-halo family, current 1D and 2D resident-state rotation, current `waveEquation1.0` direct-reader/read-cache extension, conservative soundness guards | `FOuLa1.0` rewrite-contract expansion, broader direct-reader/cache forms, root-bridge, Phase 5 `FixedBlock` |

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
- MPI count/displacement products are checked before `int` narrowing
- 1D empty-rank halo exchange routes through the nearest non-empty owned rank
- unsupported 1D right-boundary forms fall back to full-sync
- current 2D B3 sites must preserve `DAC -> read-cache -> followup -> boundary` order

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

## 8. Handoff Direction

The current handoff is:

1. treat P4.6 as closed for the currently proven resident slice
2. move on to Phase 5 and Phase 6 implementation work
3. keep `FOuLa1.0` as a later targeted infrastructure or efficiency item
4. continue measuring with focused benchmarks rather than relaxing gates

That order keeps the semantic boundary stable and avoids mixing a rewrite-contract change
into the Phase 5 surface.
