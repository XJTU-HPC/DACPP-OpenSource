# MPI Efficiency Benchmark Summary

Updated: 2026-05-10

This is the canonical benchmark summary for the MPI translator. It merges the
earlier full-suite baseline and the temporary OR followup benchmark notes.

The old dated benchmark files are kept only as historical pointers:

- `mpi_sycl_efficiency_benchmark_2026-05-06.md`
- `or_followup_phase4_benchmark_2026-05-08.md`

## Measurement Policy

Unless a section says otherwise:

- compare hand-written MPI+SYCL reference with DACPP-generated MPI+SYCL
- use `mpirun -np 4`
- time only the external `mpirun` run phase
- exclude translation and compilation time
- use enlarged benchmark sizes from `bench_mpi_only_requested.py`

The main driver is:

```bash
MPI_ONLY_BENCH_TMP_DIR=/Volumes/QUQ/working/mpi_tmp/<run-name> \
MPI_ONLY_BENCH_RANKS=4 \
MPI_ONLY_BENCH_TIMEOUT_SECONDS=1800 \
python3 clang/tools/translator/bench_mpi_only_requested.py <cases...>
```

## Benchmark Timeline

### 2026-05-06 Full-Suite Baseline

This run covered 14 non-`mpi*` application tests. It is still useful as the
baseline for cases that have not been re-measured under the final P4.6/P5/P6
state.

| Case | Scale | Hand MPI+SYCL (s) | Old DAC-MPI (s) | Old DAC / hand |
|---|---:|---:|---:|---:|
| `DFT1.0` | N=4096 | 1.044311 | 0.783150 | 0.75x |
| `FOuLa1.0` | m=8192, n=600 | 0.876812 | 2.345472 | 2.67x |
| `MDP1.0` | N=8192, T=600 | 0.851989 | 0.801768 | 0.94x |
| `decay1.0` | numIsotopes=8192, steps=600 | 0.839301 | 0.794610 | 0.95x |
| `gradientSum` | 8192x4096 | 0.727669 | 1.631674 | 2.24x |
| `imageAdjustment1.0` | 4096x4096 | 3.125963 | 0.896448 | 0.29x |
| `jacobi1.0` | N=4096, iter=300 | 0.866027 | 0.775995 | 0.90x |
| `liuliang1.0` | WIDTH=8192, steps=1000 | 0.896834 | 0.949857 | 1.06x |
| `mandel1.0` | 4096x4096, max_iter=1000 | 2.530378 | 5.015036 | 1.98x |
| `matMul1.0` | 2048x2048 | 5.765554 | 7.576529 | 1.31x |
| `oddeven0.1` | N=4096 | 1.598514 | 4.529443 | 2.83x |
| `stencil1.0` | 2048x2048, steps=600 | 1.494438 | 4.614142 | 3.09x |
| `vectorAddCombo` | N=8388608 | 0.675558 | 4.901816 | 7.26x |
| `waveEquation1.0` | 2048x2048, steps=600 | 1.474793 | 4.790831 | 3.25x |

Main readout:

- `DFT1.0`, `MDP1.0`, `decay1.0`, `jacobi1.0`, and aligned
  `imageAdjustment1.0` were already competitive.
- `vectorAddCombo`, `stencil1.0`, `waveEquation1.0`, and `oddeven0.1` exposed
  the largest performance debt at that point.

### 2026-05-08 OR Followup Intermediate

This run is historical and superseded by the P4.6 resident-halo closeout.

It proved that the OR followup lowering was semantically usable, but it was a
correctness-first implementation that still synchronized full reader/writer
state every step.

| Case | Hand MPI+SYCL median (s) | Intermediate OR median (s) | Intermediate OR / hand |
|---|---:|---:|---:|
| `MDP1.0` | 0.795349 | 1.205218 | 1.52x |
| `liuliang1.0` | 0.714773 | 1.489254 | 2.08x |
| `stencil1.0` | 1.500077 | 18.675504 | 12.45x |
| `waveEquation1.0` | 1.532526 | 37.721034 | 24.61x |

Use this only to understand the motivation for resident halo and role rotation.
Do not use it as the current performance position.

### P4.6 Closeout

The closeout benchmark moved the proven stencil cases to resident halo with
in-place halo exchange and resident-state role rotation.

Source recorded in the closed phase document:
`/Volumes/QUQ/working/mpi_tmp/p46_final_close/results.tsv`

That TSV is not present in the current local workspace, so this summary treats
the markdown-recorded values as the available source of truth.

| Case | Current hand MPI+SYCL (s) | Current P4.6 DAC-MPI (s) | Current DAC / hand | Old DAC-MPI (s) | Old DAC / current DAC |
|---|---:|---:|---:|---:|---:|
| `FOuLa1.0` | 0.929324 | 1.663965 | 1.79x | 2.345472 | 1.41x |
| `MDP1.0` | 0.776213 | 0.781158 | 1.01x | 0.801768 | 1.03x |
| `liuliang1.0` | 0.891906 | 0.941800 | 1.06x | 0.949857 | 1.01x |
| `stencil1.0` | 1.589305 | 1.462638 | 0.92x | 4.614142 | 3.15x |
| `waveEquation1.0` | 1.530221 | 1.510119 | 0.99x | 4.790831 | 3.17x |

Readout:

- `stencil1.0` and `waveEquation1.0` are the largest P4.6 wins.
- `MDP1.0` and `liuliang1.0` are effectively at parity for the current scale.
- At this closeout point, `FOuLa1.0` still did not use loop-resident lowering.
  That gap is superseded by the focused owner-loop specialization recorded
  below.

### 2026-05-10 Focused Generic OR Resident-Buffer Pass

This run measured the ordinary OR chain optimization that avoids copying
resident chain intermediates through temporary host vectors.

| Case | Scale | Hand MPI+SYCL (s) | DAC-MPI before (s) | DAC-MPI after (s) | Readout |
|---|---:|---:|---:|---:|---|
| `vectorAddCombo` | N=8388608 | 1.074960 before / 0.869464 after | 0.772492 | 0.776589 | Structurally cleaner, wall-clock neutral/noisy. |

Readout:

- `vectorAddCombo` now routes as a `Contiguous1D` OR chain, not the old legacy
  AccessPattern path described by the 2026-05-06 baseline.
- The generated code reuses resident input references for downstream reads and
  moves write-only output buffers into resident state.
- The focused wall-clock result is effectively unchanged, so remaining cost at
  this scale is more likely scatter/gather, final visibility sync, SYCL launch,
  or benchmark noise than chain-intermediate host copies.

### 2026-05-10 FOuLa Owner-Loop Specialization

This run measured the guarded owner-loop specialization for the current
`FOuLa1.0` loop-local slice shape.

| Case | Scale | Hand MPI+SYCL (s) | DAC-MPI before owner-loop (s) | DAC-MPI after owner-loop (s) | After DAC / hand |
|---|---:|---:|---:|---:|---:|
| `FOuLa1.0` | m=8192, n=600 | 0.938183 | 1.595680 | 0.931847 | 0.99x |

Readout:

- The old generated path broadcast the full reader column and gathered the full
  writer slice every time step.
- The owner-loop path scatters the initial column once, broadcasts only two
  boundary scalars per step, exchanges neighbor halos, and gathers owned history
  once at the end.
- This removes the main full-column per-step communication cost and brings DAC
  generated code to parity with the hand-written reference for this focused
  scale.

### P5 Phase-Exchange Closeout

Source recorded in the closed phase document:
`/Volumes/QUQ/working/mpi_tmp/p5_fixedblock_oddeven_loop_resident/results.tsv`

That TSV is not present in the current local workspace, so this summary treats
the markdown-recorded values as the available source of truth.

| Case | Scale | Hand MPI+SYCL (s) | Current P5 DAC-MPI (s) | Current DAC / hand |
|---|---:|---:|---:|---:|
| `oddeven0.1` | N=4096 | 2.112300 | 1.129017 | 0.53x |

Readout:

- The loop-resident phase-exchange path removes the per-iteration
  gather/materialize/broadcast cost that dominated the standalone P5 wrapper.
- The measured current DAC code is 1.87x faster than the hand-written reference
  in that closeout run.

## Current Interpretation By Case

| Case | Current interpretation | Followup need |
|---|---|---|
| `DFT1.0` | Baseline already competitive. | Re-run under current code for confirmation. |
| `FOuLa1.0` | Guarded owner-loop specialization is at parity in the focused 2026-05-10 run. | Generalize owner-loop recognition beyond the current strict shape. |
| `MDP1.0` | Current resident-halo path is near hand-written MPI. | Keep as regression checkpoint. |
| `decay1.0` | Baseline competitive. | Re-run current full suite. |
| `gradientSum` | Baseline slower than hand reference. | Identify current lowering path and isolate overhead. |
| `imageAdjustment1.0` | DAC-MPI faster after semantic alignment. | Keep semantic alignment note with benchmark logs. |
| `jacobi1.0` | Baseline competitive despite replicated full tensor broadcast. | Re-run and inspect broadcast sensitivity by N/iter. |
| `liuliang1.0` | Current resident-halo path is close to parity. | Keep as 1D resident regression checkpoint. |
| `mandel1.0` | Baseline about 2x slower. | Identify current lowering path and kernel/communication split. |
| `matMul1.0` | RowPartition path has expected payload broadcast cost. | Compare against block/tile-aware future layout if added. |
| `oddeven0.1` | Current P5 phase exchange is strong. | Keep focused P5 regression and rank-scaling tests. |
| `stencil1.0` | Current 2D resident halo is near parity or slightly faster in closeout. | Keep as P4.6 regression checkpoint. |
| `vectorAddCombo` | Current OR chain is already near/faster than hand MPI; resident-buffer copy removal is structurally cleaner but wall-clock neutral. | Add phase timing before further tuning. |
| `waveEquation1.0` | Current 2D resident halo with direct-reader extension is near parity. | Keep as B3 direct-reader regression checkpoint. |

## Benchmark TODO

1. Re-run all 14 benchmark cases under the current P6 code with at least three
   runs per side and report medians.
2. Store raw `results.tsv`, logs, generated source snapshots, and the exact git
   commit or diff state with the benchmark summary.
3. Add per-phase timing for generated code:
   - init/scatter
   - per-run kernel
   - per-run halo or boundary exchange
   - pack/unpack or global-index collection
   - materialize/gather
   - final broadcast
4. Identify any remaining benchmark cases that still route through legacy
   AccessPattern and collect `collect_positions_for_item` profiling for those
   cases.
5. Add rank scaling for the main optimized cases and the remaining slow cases.
6. Separate `root-only` and `all-ranks` output-sync experiments where semantics
   allow both.
7. Keep the 2026-05-08 intermediate numbers only as historical motivation;
   avoid mixing them into current-performance tables.
