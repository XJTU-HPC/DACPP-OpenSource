# Findings

## 2026-05-10 P5 Start Position

Phase 4.6 is closed for the currently proven resident stencil slice. The accepted path now
includes 1D resident-state rotation, 2D resident-state rotation, and the current narrow
`waveEquation1.0` direct-reader/read-cache extension, all under conservative gates and
with legacy fallback preserved.

P5 starts from `oddeven0.1`, which is not a P4.6 stencil-halo issue. Its shell uses one
1D `RegularSplit` with size 2 and stride 2 on both the input and output. The current OR
analysis treats all regular splits as stencil-window readers and rejects writable regular
split params, so `oddeven0.1` stays on the legacy wrapper path with
`AccessPattern` / `PackPlan`.

## Benchmark Comparison

Old wrapper baseline comes from
`clang/tools/translator/docs/benchmarks/mpi_sycl_efficiency_benchmark_2026-05-06.md`.
Current closeout numbers come from
`/Volumes/QUQ/working/mpi_tmp/p46_final_close/results.tsv`.

Important: the hand-written MPI numbers come from two different benchmark batches, so they
should be read as contextual references rather than treated as the same run.

| Case | Old hand-written MPI+SYCL (s) | Old MPI wrapper / DAC-MPI (s) | Current hand-written MPI+SYCL (s) | Current P4.6 DAC-MPI (s) | Improvement vs old wrapper | Current DAC / current hand |
|---|---:|---:|---:|---:|---:|---:|
| `FOuLa1.0` | 0.876812 | 2.345472 | 0.929324 | 1.663965 | 1.41x | 1.79x |
| `MDP1.0` | 0.851989 | 0.801768 | 0.776213 | 0.781158 | 1.03x | 1.01x |
| `liuliang1.0` | 0.896834 | 0.949857 | 0.891906 | 0.941800 | 1.01x | 1.06x |
| `stencil1.0` | 1.494438 | 4.614142 | 1.589305 | 1.462638 | 3.15x | 0.92x |
| `waveEquation1.0` | 1.474793 | 4.790831 | 1.530221 | 1.510119 | 3.17x | 0.99x |

`oddeven0.1` is present only in the old wrapper benchmark so far:

| Case | Old hand-written MPI+SYCL (s) | Old MPI wrapper / DAC-MPI (s) |
|---|---:|---:|
| `oddeven0.1` | 1.598514 | 4.529443 |

## Main Technical Findings

- `liuliang1.0` and `MDP1.0` now use the same rotated 1D resident-state shape: kernel
  writes into a full next-state buffer, `exchange_halo_1d_inplace(...)` patches halo in
  place, and reader/writer roles swap each step.
- `stencil1.0` and `waveEquation1.0` keep the accepted 2D resident-halo family but no
  longer pay the earlier per-step slab-copy cost. The current path uses in-place halo
  exchange plus role rotation.
- The current 2D B3 slice remains intentionally narrow and now also requires the current
  `DAC -> read-cache -> followup -> boundary` statement order. If the proof is weaker,
  the site falls back to Route A `StencilFullSync`.

## `FOuLa1.0` Ownership

`FOuLa1.0` is still rejected with
`[DACPP][MPI][OR][P4.6][Loop] ... reason=not inside a stable loop site`, but the real
blocker is stricter than a missing analysis toggle:

- `u_kin`, `u_kout`, and `r` are loop-local temporaries
- the current loop-lowered rewrite inserts `init(ctx, original shell args...)` before the
  outer loop
- accepting `FOuLa1.0` as-is would emit an init call before those temporaries are in scope

So `FOuLa1.0` belongs to a later rewrite-contract expansion, not Phase 5.

## Recommended Direction

- P5 should continue from the current `FixedBlock` standalone correctness slice toward a
  loop-resident phase-exchange path.
- `FOuLa1.0` should be revisited later as a targeted infrastructure or efficiency slice.
- Benchmark checkpoints should remain part of that follow-up so any gate widening stays
  measurable.

## P5 Minimal Slice

- Accept only 1D non-overlapping `RegularSplit` fixed blocks where all fixed-block params
  share the same regular split metadata, currently `splitSize == splitStride == 2`.
- Require one READ fixed-block input and one WRITE fixed-block output, matching element
  types, same fixed-block logical item count, and no READ_WRITE fixed-block parameter.
- Require each accepted shell param to have exactly one split, a 1D wrapper tensor, a 1D
  calc view, and no payload/void dimensions. Shapes like `tensor[{S1}][{}]` and
  `Matrix` params with only `input[{S1}]` remain fallback because the current codegen only
  scatters, gathers, and materializes flat 1D block payloads.
- Preserve host-visible tensor semantics: any output tensor that subsequent host code
  reads must still be materialized/broadcast according to the existing sync classifier.
- Do not use legacy `AccessPattern` / `PackPlan` in the accepted P5 path; incomplete
  proof falls back to the existing path.

## P5 First-Slice Result

`oddeven0.1` now enters `LocalLayoutKind::FixedBlock` for both call sites. The generated
accepted path uses `__dacpp_mpi_or_ODDEVEN_oddeven_0/1`, `fixed_block_count_1d`, and
`ContiguousView1D` block views; it does not generate legacy `AccessPattern`, `PackPlan`,
or `__dacpp_mpi_stencil` code for the accepted sites.

The current codegen is standalone and conservative. It materializes and broadcasts the
FixedBlock output after each call so host-visible tensor slicing/copy code remains valid
on every rank. The materialized broadcast count now uses the same checked MPI count
narrowing as the scatter/gather path. This is sound for the first slice, but it is not
the final performance shape for odd-even.

Review hardening added after the first bridge:

- `mpiFixedBlockPayloadReject1D` pins the fallback for fixed-block plus payload
  dimensions / multi-split shell params.
- `mpiFixedBlockMatrixSingleSplitReject1D` pins the fallback for implicit non-flat Matrix
  payloads that use only one split bracket.
- `oddeven0.1` now asserts the generated materialized `MPI_Bcast` uses
  `narrow_mpi_count_or_abort(...)`.

Focused benchmark:

| Case | Scale | Standard MPI+SYCL (s) | Current P5 DAC-MPI (s) | Status |
|---|---:|---:|---:|---|
| `oddeven0.1` | N=4096 | 1.814108 | 7.835445 | ok |

Read this separately from the old wrapper baseline. In this run the P5 standalone bridge is
slower than both the hand-written reference and the old wrapper baseline because each
phase still pays two full materialize/broadcast cycles. The next P5 performance blocker is
runtime/codegen support for loop-resident phase exchange, not a wider analysis gate.
