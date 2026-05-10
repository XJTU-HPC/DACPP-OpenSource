# Phase 4.6 Optimization Closeout

Updated: 2026-05-10 (P6 lowering-contract closeout)

## 1. Closeout Summary

Phase 4.6 is closed for the currently proven operator-resident stencil slice.
This closeout keeps the semantic boundary unchanged:

- no Phase 5 `FixedBlock` inside the P4.6 accepted stencil surface
- no root-bridge stencil
- no accepted-path reuse of legacy `AccessPattern` / `PackPlan` / root-centric codegen
- no weakening of host-visible tensor semantics
- no removal of existing fallback

The accepted P4.6 surface now includes:

1. `StencilWindow1D` resident halo for the current 1D loop-lowered slice
2. `StencilWindow2D` row-block resident halo for the current 2D slice
3. `waveEquation1.0`'s current direct-reader plus `(-1,-1)` read-cache extension
4. resident-state role rotation for both 1D and 2D accepted paths
5. conservative soundness guards for source order, scalar rejection, oversized MPI counts,
   empty-rank halo exchange, and unsupported boundary forms

The code paths that matter are:

- `MDP1.0`, `liuliang1.0`: accepted 1D `StencilResidentHalo`
- `stencil1.0`: accepted 2D `StencilResidentHalo`
- `waveEquation1.0`: accepted 2D `StencilResidentHalo` with one resident direct reader

`FOuLa1.0` is still outside the current loop-lowered family and remains on the plain OR
wrapper path.

## 2. What Landed

### 2.1 1D resident-state rotation

The accepted 1D path no longer pays the earlier per-step local followup-copy cost.
For `MDP1.0` and `liuliang1.0`, the generated shape is now:

- kernel writes directly into a full next-state buffer
- `exchange_halo_1d_inplace(...)` updates the halo in place on that next-state buffer
- boundary-local updates write into the same next-state buffer
- reader/writer roles rotate by `swap()`
- final `materialize()` gathers from `owned_slice_1d(...)`

This keeps the current P4.6 semantics intact while removing the extra local copy that was
still present in the earlier accepted B1 path.

### 2.2 2D resident-state rotation

The accepted 2D row-block resident path now uses full-state role rotation instead of
rebuilding per-step slabs before halo exchange.

- `stencil1.0`: in-place halo exchange plus double-buffer rotation
- `waveEquation1.0`: in-place halo exchange plus triple-buffer `prev/cur/next` rotation

This preserves the current B2/B3 resident-halo semantics and removes the main remaining
local-copy cost in the accepted 2D path.

### 2.3 B3 soundness guard

The current accepted 2D direct-reader/read-cache slice is intentionally narrow. In
addition to the existing shape proof, it now requires the current top-level statement
order:

`DAC -> read-cache -> followup -> boundary`

If the order is different, the site falls back to Route A `StencilFullSync`.

### 2.4 Safety guards kept in place

The closeout also keeps the earlier conservative guards that were added during Route B:

- 2D scalar readers are rejected
- oversized count/displacement products are guarded before MPI narrowing
- 1D empty-rank halo exchange routes to the nearest non-empty owned rank
- unsupported 1D right-boundary updates fall back to full-sync

## 3. Current Accepted Boundary

### 3.1 1D accepted slice

Accepted only when all of the following are true:

- stable loop site
- exactly one 1D window reader
- exactly one WRITE-only direct writer
- current `writer -> reader` followup shape
- no direct reader
- no read-cache transition
- no root-bridge
- current boundary form remains provable

Otherwise the site keeps the existing fallback.

### 3.2 2D accepted slice

Accepted only when all of the following are true:

- stable loop site
- row-block `StencilWindow2D`
- exactly one 2D window reader
- exactly one WRITE-only direct writer
- current `(+1,+1)` followup shape
- current four boundary-local loops
- for B3 only: at most one direct reader and the current `(-1,-1)` read-cache transition
- current top-level `DAC -> read-cache -> followup -> boundary` order

Otherwise the site falls back to Route A `StencilFullSync`.

## 4. Benchmark Position

### 4.1 Source of truth

This document uses two benchmark sources:

1. old wrapper baseline:
   `clang/tools/translator/docs/benchmarks/mpi_sycl_efficiency_benchmark_2026-05-06.md`
2. current closeout benchmark:
   `/Volumes/QUQ/working/mpi_tmp/p46_final_close/results.tsv`

Important: the hand-written MPI times come from two different benchmark batches. They
should be read as reference context, not subtracted from each other as if they were the
same run.

### 4.2 Full comparison

| Case | Old hand-written MPI+SYCL (s) | Old MPI wrapper / DAC-MPI (s) | Current hand-written MPI+SYCL (s) | Current P4.6 DAC-MPI (s) | Old wrapper / current speedup | Current DAC / current hand |
|---|---:|---:|---:|---:|---:|---:|
| `FOuLa1.0` | 0.876812 | 2.345472 | 0.929324 | 1.663965 | 1.41x | 1.79x |
| `MDP1.0` | 0.851989 | 0.801768 | 0.776213 | 0.781158 | 1.03x | 1.01x |
| `liuliang1.0` | 0.896834 | 0.949857 | 0.891906 | 0.941800 | 1.01x | 1.06x |
| `stencil1.0` | 1.494438 | 4.614142 | 1.589305 | 1.462638 | 3.15x | 0.92x |
| `waveEquation1.0` | 1.474793 | 4.790831 | 1.530221 | 1.510119 | 3.17x | 0.99x |

### 4.3 Reading the table

- `liuliang1.0` is the remaining 1D P4.6 closure item that has now been absorbed:
  the old accepted resident-halo path still had extra local-copy cost, while the current
  rotated path closes most of that gap without changing semantics.
- `stencil1.0` and `waveEquation1.0` are the biggest Phase 4.6 wins. They moved from the
  old wrapper baseline of roughly `3x` slower than hand-written MPI to near parity, and
  `stencil1.0` is now slightly faster than the current hand-written reference in this run.
- `FOuLa1.0` improved relative to the old wrapper baseline, but it still has not entered
  the loop-lowered resident family. Its number should not be read as a completed P4.6
  optimization result.

## 5. `FOuLa1.0` Status

`FOuLa1.0` still logs:

`[DACPP][MPI][OR][P4.6][Loop] ... rejected reason=not inside a stable loop site`

The important point is that this is not just a missing analysis toggle.

The current loop-lowered rewrite contract inserts:

- `ctx` declaration before the outer loop
- `init(ctx, original shell args...)` before the outer loop
- `run(ctx, original shell args...)` inside the loop
- `materialize(ctx, original shell args...)` after the loop

For `FOuLa1.0`, the shell args `u_kin`, `u_kout`, and `r` are loop-local temporaries
declared inside the time-step loop body. Accepting the site without changing that rewrite
contract would emit `init(...)` before those variables exist.

So the blocker is:

- not Phase 5
- not a missing resident-halo codegen case inside the current P4.6 family
- a later P4.6 infrastructure slice around rewrite/init argument ownership

Until that contract is extended safely, `FOuLa1.0` should remain on the existing OR
wrapper path.

## 6. Recommended Next-Step Boundary

The current recommendation is:

1. P4.6 can be treated as closed for the currently proven slice.
2. Phase 5 is complete for the current first slice. It has the conservative
   `FixedBlock` standalone fallback for `oddeven0.1`
   (1D non-overlapping `split(2,2)` READ input plus WRITE output via a
   standalone OR wrapper without legacy `AccessPattern` / `PackPlan`) and now
   the loop-resident `FixedBlockPhaseExchange` slice on top of it. The
   loop-resident path is gated on the canonical `oddeven0.1` shape (two
   sibling DAC expressions in one loop, slice offset `+1`, boundary `[0]` and
   `[N-1]` preserved); anything that fails the gate falls back to the
   standalone first slice.
3. `FOuLa1.0` should not be merged into Phase 5 planning; it belongs to a later targeted
   rewrite-contract expansion or a post-P6 efficiency pass.
4. The standalone P5 first slice is a correctness/structure bridge. The
   loop-resident phase-exchange slice removes the per-iteration
   gather/materialize/broadcast cost; further P5 efficiency widening should
   wait until the P6 lowering contract is explicit.
5. Phase 6 should build shared lowering-contract infrastructure for statement
   removal, resident-state ownership, host-visible materialization,
   use/alias/write guards, and compile-time fallback vs runtime-abort policy.
   It should not widen P5 or accept `FOuLa1.0` as its first step.
6. Benchmark checkpoints should stay in place so performance debt is measured, not guessed.

## 7. Verification

The current code and test expectations were verified with the requested serial commands:

```bash
cmake --build build --target translator -j8
bash clang/tools/translator/test_mpi.sh MDP1.0 liuliang1.0 stencil1.0 waveEquation1.0
bash clang/tools/translator/test_mpi.sh oddeven0.1 mpiFixedBlockOverlapReject1D mpiFixedBlockPayloadReject1D mpiFixedBlockMatrixSingleSplitReject1D mpiFixedBlockPhaseExchangeOffsetReject1D mpiFixedBlockPhaseExchangeMissingBoundaryReject1D mpiFixedBlockPhaseExchangeNonAdjacentReject1D mpiFixedBlockPhaseExchangeRank3Run1D mpiFixedBlockPhaseExchangeWrongArgsReject1D mpiFixedBlockPhaseExchangePostOutputUseReject1D mpiFixedBlockPhaseExchangePostOutputAliasReject1D mpiFixedBlockPhaseExchangeOddNReject1D
bash clang/tools/translator/test_mpi.sh mpiLoopStencilResidentHalo1D mpiLoopStencilResidentHaloEmptyRank1D mpiLoopStencilRightBoundaryFullSync1D
bash clang/tools/translator/test_mpi.sh mpiLoopStencilScalarReject2D mpiLoopStencilCountGuard2D mpiLoopStencilOrderReject2D
git diff --check
```

Focused benchmark command:

```bash
MPI_ONLY_BENCH_TMP_DIR=/Volumes/QUQ/working/mpi_tmp/p46_final_close \
MPI_ONLY_BENCH_RANKS=4 \
MPI_ONLY_BENCH_TIMEOUT_SECONDS=1800 \
python3 clang/tools/translator/bench_mpi_only_requested.py \
  MDP1.0 liuliang1.0 stencil1.0 waveEquation1.0 FOuLa1.0
```

## 8. Phase 5 Loop-Resident Phase-Exchange Benchmark

Source of truth:
`/Volumes/QUQ/working/mpi_tmp/p5_fixedblock_oddeven_loop_resident/results.tsv`

| Case | Standard hand-written MPI+SYCL (s) | Current P5 DAC-MPI loop-resident (s) | DAC vs hand-written |
|---|---:|---:|---:|
| `oddeven0.1` (N=4096) | 2.112300 | 1.129017 | 1.87x faster |

The same `oddeven0.1` workload on the standalone P5 first-slice path measured
at roughly `7.83s` against `1.81s` hand-written reference per the prior
checkpoint, so the loop-resident slice removes the per-iteration
gather/materialize/broadcast cost that dominated the standalone path.

The P5 accepted path also carries guards for phase-B argument binding,
post-loop phase-A output use and pre-loop initializer aliases, odd-total
fallback, runtime reader/writer/proven-total mismatch abort before scatter, and
rank-contiguous non-block-aligned partitions.

Focused benchmark command:

```bash
MPI_ONLY_BENCH_TMP_DIR=/Volumes/QUQ/working/mpi_tmp/p5_fixedblock_oddeven_loop_resident \
MPI_ONLY_BENCH_RANKS=4 \
MPI_ONLY_BENCH_TIMEOUT_SECONDS=1800 \
python3 clang/tools/translator/bench_mpi_only_requested.py oddeven0.1
```

## 9. Phase 6 Lowering-Contract Closeout

P6 closes the first reusable lowering-contract infrastructure slice for
loop-lowered OR paths rather than another shape-specific optimization.

The P6 contract now makes each accepted P4.6/P5 lowering describe:

- source statements removed by rewrite
- resident tensors and final host-visible tensor ownership
- materialize timing and required runtime guards
- post-loop use, alias, write, and loop-local argument lifetime rejection rules
- accepted/rejected reason strings that tests can pin

The current closeout also records a lightweight consistency check on accepted
contract facts. The checker logs `contract-check=pass reason=...` when:

- the contract is enabled and its name matches the accepted lowering kind
- the source DAC is marked `Replace`
- the remove-list matches the current proven source of truth: legacy P4.6
  removal set for stencil paths, P5 follower statements for phase exchange
- resident tensor facts, materialized tensor ownership, and materialization timing
  match the current path
- statement, materialization, and guard reason metadata is present
- compile-time fallback guards and required runtime abort guards are represented,
  including 1D FullSync replicated-scalar and 2D FullSync/runtime-count cases

This checker is diagnostic and does not choose codegen. P4.6 still removes via
contract Remove statements only when the legacy-vs-contract removal set matches;
mismatch falls back to the legacy remover. The current P4.6 contract
construction and legacy removal-set calculation intentionally share the same
source: both derive removable followup/read-cache statements and boundary-local
statements from `analyzeDistributedStencilSite()`. Current focused accepted
P4.6 tests assert the match path. The mismatch fallback is a defensive
compatibility path for future contract-construction evolution; no current
natural accepted source shape can produce it without changing production
behavior or artificially corrupting contract construction, so this closeout does
not add a mismatch fixture. The standalone P5 fallback remains available,
`FOuLa1.0` remains outside this accepted surface, and generated-code semantics
are unchanged.
