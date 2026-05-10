# P5 FixedBlock Bring-up Task Plan

## Objective

Start Phase 5 `FixedBlock` without widening the P4.6 stencil boundary. The first target is
`oddeven0.1`: move the currently provable non-overlapping fixed-block slice away from
legacy `AccessPattern` / `PackPlan` and into a conservative operator-resident family with
fallback preserved.

## Current Baseline

| Item | Status | Notes |
|---|---|---|
| Branch/worktree | complete | On `tqc-2`; review-fix session preserved the existing uncommitted P5 worktree. |
| P4.6 boundary | complete | P4.6 remains closed for the proven resident stencil slice; do not mix P5 with FOuLa or existing P4.6 accepted paths. |
| Benchmark split | complete | Old wrapper baseline has `oddeven0.1` at 4.529443s vs hand-written 1.598514s; current closeout benchmark does not include `oddeven0.1`. |
| P5 code search | complete | `FixedBlock` previously existed only as enum/string metadata; this session added the first accepted codegen path. |

## Planned Work

| Item | Status | Notes |
|---|---|
| Define minimal P5 slice | complete | 1D `RegularSplit` where split size equals stride, currently `split(2,2)`, with one READ fixed-block input and one WRITE fixed-block output. |
| Analysis/metadata/gate | complete | Conservative FixedBlock classification rejects overlapping windows, payload/multi-split params, non-1D wrapper tensors or calc views, mixed dimensions, extra params, read-write fixed blocks, block sizes other than 2, and byte-transport element types. |
| Codegen/runtime loop | complete | Added standalone OR wrapper path with fixed-block partition/scatter/kernel/gather/materialize/broadcast; materialized broadcasts use checked MPI count narrowing; no legacy `AccessPattern` / `PackPlan` in accepted path. |
| Positive and negative tests | complete | `oddeven0.1` checks accepted FixedBlock path and checked broadcast count; `mpiFixedBlockOverlapReject1D`, `mpiFixedBlockPayloadReject1D`, and `mpiFixedBlockMatrixSingleSplitReject1D` check fallback guards. |
| Serial verification | complete | Build, positive/negative P5 tests, P4.6 regression group, and `git diff --check` passed. |
| Focused benchmark | complete | `oddeven0.1` benchmark recorded in `/Volumes/QUQ/working/mpi_tmp/p5_fixedblock_oddeven/results.tsv`. |
| Docs/tracking sync | complete | Canonical docs and tracking updated with current P5 first-slice position. |

## Deferred On Purpose

| Item | Why it stays deferred |
|---|---|
| `FOuLa1.0` loop-site acceptance | Rewrite/init contract expansion for loop-local shell args; not P5. |
| P4.6 gate widening | P4.6 is closed for the current proven resident stencil slice. |
| Overlapping regular windows such as stride smaller than size | These remain stencil/legacy fallback until a separate proof exists. |
| Root-bridge or root-centric accepted FixedBlock | Not part of the new accepted path. |
