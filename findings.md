# Findings

## 2026-05-10 Current Closeout Position

Phase 4.6 is closed for the currently proven resident stencil slice. The accepted path now
includes 1D resident-state rotation, 2D resident-state rotation, and the current narrow
`waveEquation1.0` direct-reader/read-cache extension, all under conservative gates and
with legacy fallback preserved.

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

- It is reasonable to move on to Phase 5 and Phase 6 now.
- `FOuLa1.0` should be revisited later as a targeted infrastructure or efficiency slice.
- Benchmark checkpoints should remain part of that follow-up so any gate widening stays
  measurable.
