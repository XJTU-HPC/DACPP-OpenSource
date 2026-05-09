# Phase 4.6 A0 Measurement Baseline

Date: 2026-05-09
Branch: `tqc-2`

This is a measurement-only baseline for the current OR stencil implementation.
No lowering behavior was changed for this step.

## Commands

```bash
cmake --build build --target translator -j8
cd clang/tools/translator
bash test_mpi.sh
```

Timing samples were collected from the generated binaries under
`/Volumes/QUQ/working/mpi_tmp`:

```bash
mpirun -np 4 /Volumes/QUQ/working/mpi_tmp/<test>/mpi_bin
```

Wall-clock time was measured with `/usr/bin/time -p`. Each test used the
default small `.dac.cpp` input selected by `test_mpi.sh`.

## Static Communication Counts

Counts are per generated wrapper invocation. The "total per run" columns
multiply by the source time-step loop count.

| Test | Loop calls | Full reader `MPI_Bcast` | Direct-reader `MPI_Bcast` | Writer `MPI_Gatherv` | Followup/read-cache full tensor `MPI_Bcast` | Separate final gather | Scalar `MPI_Bcast` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `MDP1.0` | 1000 | 1 / call, 1000 total | 0 | 1 / call, 1000 total | 1 / call, 1000 total | 0 | 0 |
| `liuliang1.0` | 200 | 1 / call, 200 total | 0 | 1 / call, 200 total | 1 / call, 200 total | 0 | 0 |
| `stencil1.0` | 10 | 1 / call, 10 total | 0 | 1 / call, 10 total | 1 / call, 10 total | 0 | 0 |
| `waveEquation1.0` | 10 | 1 / call, 10 total | 1 / call, 10 total | 1 / call, 10 total | 2 / call, 20 total | 0 | 0 |

Notes:

- There is no distinct final-only gather in the current stencil OR path.
  Writer output is materialized with `MPI_Gatherv` on every wrapper invocation.
- `waveEquation1.0` has one full window-reader broadcast for `matCur`, one
  full direct-reader broadcast for `matPrev`, one read-cache transition
  broadcast back to `matPrev`, and one followup broadcast back to `matCur`.
- The current default target cases do not generate scalar `MPI_Bcast`.

## Generated Source Evidence

| Test | Generated source | Relevant collective sites |
|---|---|---|
| `MDP1.0` | `/Volumes/QUQ/working/mpi_tmp/MDP1.0/mdp.mpi.dac_sycl_buffer.cpp` | reader bcast line 97; writer gather line 123; followup bcast line 141 |
| `liuliang1.0` | `/Volumes/QUQ/working/mpi_tmp/liuliang1.0/liuliang.mpi.dac_sycl_buffer.cpp` | reader bcast line 93; writer gather line 119; followup bcast line 146 |
| `stencil1.0` | `/Volumes/QUQ/working/mpi_tmp/stencil1.0/stencil.mpi.dac_sycl_buffer.cpp` | reader bcast line 88; writer gather line 120; followup bcast line 211 |
| `waveEquation1.0` | `/Volumes/QUQ/working/mpi_tmp/waveEquation1.0/waveEquation.mpi.dac_sycl_buffer.cpp` | reader bcast line 88; direct-reader bcast line 100; writer gather line 136; read-cache bcast line 156; followup bcast line 223 |

## Runtime Samples

`mpirun -np 4`, seconds, 3 samples:

| Test | Samples | Min | Median | Mean | Max |
|---|---:|---:|---:|---:|---:|
| `MDP1.0` | 0.87, 0.98, 0.90 | 0.87 | 0.90 | 0.92 | 0.98 |
| `liuliang1.0` | 0.21, 0.22, 0.21 | 0.21 | 0.21 | 0.21 | 0.22 |
| `stencil1.0` | 0.06, 0.06, 0.06 | 0.06 | 0.06 | 0.06 | 0.06 |
| `waveEquation1.0` | 0.06, 0.06, 0.07 | 0.06 | 0.06 | 0.06 | 0.07 |

These are lightweight baseline samples, not a statistically robust benchmark.
They are sufficient for Phase 4.6 before/after communication-structure tracking.
