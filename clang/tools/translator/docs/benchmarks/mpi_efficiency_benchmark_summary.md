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

### 2026-05-10 Phase 1.4 Segmented Profile Reporting

This run validates the new default-off segmented profiling and records raw
artifacts for future before/after comparisons. It is a measurement run, not a
translator fast path change.

Driver:

```bash
python3 clang/tools/translator/bench_mpi_profile_segments.py \
  --tmp-dir /Volumes/QUQ/working/mpi_tmp/phase1_4_profile_focus \
  --ranks 1,2,4,8 \
  --trials 3 \
  vectorAddCombo FOuLa1.0 MDP1.0 stencil1.0 oddeven0.1 decay1.0
```

Artifacts:

- `/Volumes/QUQ/working/mpi_tmp/phase1_4_profile_focus/summary.tsv`
- `/Volumes/QUQ/working/mpi_tmp/phase1_4_profile_focus/results.tsv`
- `/Volumes/QUQ/working/mpi_tmp/phase1_4_profile_focus/profile_raw.tsv`
- `/Volumes/QUQ/working/mpi_tmp/phase1_4_profile_focus/profile_summary.tsv`
- `/Volumes/QUQ/working/mpi_tmp/phase1_4_profile_focus/collect_positions_profile.tsv`
- generated source snapshots, per-trial stdout/stderr, `metadata.json`,
  `git_status.txt`, `git_diff_stat.txt`, `git_diff.patch`, `git_untracked.txt`,
  and untracked-file snapshots

The profile driver runs hand MPI+SYCL, DAC-MPI with profiling disabled, and
DAC-MPI with `DACPP_MPI_PROFILE=1`. For every focused case/rank/trial in this
run, profile-enabled DAC stdout matched profile-disabled DAC stdout. Baseline
correctness against non-MPI DAC remains covered by `test_mpi.sh`; this driver is
for performance measurement and profile attribution.

4-rank medians:

| Case | Scale | Hand MPI+SYCL (s) | DAC-MPI profile off (s) | DAC / hand | DAC-MPI profile on (s) |
|---|---:|---:|---:|---:|---:|
| `FOuLa1.0` | m=8192, n=600 | 0.269842 | 0.373362 | 1.38x | 0.409075 |
| `MDP1.0` | N=8192, T=600 | 0.269820 | 0.269225 | 1.00x | 0.262492 |
| `decay1.0` | numIsotopes=8192, steps=600 | 0.330189 | 0.380968 | 1.15x | 0.384194 |
| `oddeven0.1` | N=4096 | 1.195697 | 0.187462 | 0.16x | 0.150047 |
| `stencil1.0` | 2048x2048, steps=600 | 1.009845 | 0.954323 | 0.95x | 0.954409 |
| `vectorAddCombo` | N=8388608 | 0.202725 | 0.267805 | 1.32x | 0.254276 |

DAC-MPI profile-off rank scaling medians:

| Case | np=1 (s) | np=2 (s) | np=4 (s) | np=8 (s) |
|---|---:|---:|---:|---:|
| `FOuLa1.0` | 0.270201 | 0.314536 | 0.373362 | 0.551603 |
| `MDP1.0` | 0.209464 | 0.194843 | 0.269225 | 0.434063 |
| `decay1.0` | 0.205890 | 0.257869 | 0.380968 | 0.494490 |
| `oddeven0.1` | 0.144489 | 0.150307 | 0.187462 | 0.196965 |
| `stencil1.0` | 0.908790 | 0.854567 | 0.954323 | 1.088306 |
| `vectorAddCombo` | 0.208281 | 0.253783 | 0.267805 | 0.417885 |

4-rank segmented profile readout:

| Case | Largest segment signals from `profile_summary.tsv` |
|---|---|
| `vectorAddCombo` | `scatter` 47.954ms, `materialize` 38.883ms, `bcast` 19.832ms, `kernel` 16.179ms, `gather` 10.269ms |
| `decay1.0` | `kernel` 591.297ms, `gather` 198.918ms, `bcast` 92.985ms, `materialize` 2.477ms |
| `FOuLa1.0` | `kernel` 482.170ms, `halo` 61.911ms, `bcast` 56.898ms, `materialize` 51.093ms |
| `MDP1.0` | `kernel` 417.994ms, `halo` 37.296ms; other segments below 1ms aggregate |
| `stencil1.0` | `kernel` 2318.965ms, `halo` 514.394ms, `scatter` 40.424ms, `materialize` 20.006ms |
| `oddeven0.1` | `kernel` dominates; communication and materialization are sub-millisecond to low single-digit aggregate |

Profile `sum_ms` is summed across ranks and is therefore an attribution signal,
not a value to add up into external wall time.

Readout:

- `vectorAddCombo` is still an OR `Contiguous1D` chain, not legacy
  AccessPattern. The current residual profile points at scatter/final
  materialize/bcast, not legacy pack.
- `decay1.0` shows per-step gather and bcast cost and is a strong Phase 3
  materialization/residency inventory candidate.
- `MDP1.0`, `stencil1.0`, and `FOuLa1.0` are mostly kernel+halo dominated in
  this run. Do not widen stencil or FOuLa accepted surface based on this
  profile alone.
- `oddeven0.1` remains a strong P5 phase-exchange regression checkpoint.

Legacy fallback smoke artifact:
`/Volumes/QUQ/working/mpi_tmp/phase1_4_legacy_profile_smoke`.

That smoke compiled and ran `mpiFixedBlockMatrixSingleSplitReject1D` through the
legacy AccessPattern path with `DACPP_MPI_PROFILE=1`, emitted
`init/scatter/pack/kernel/gather/materialize/final_sync`, preserved existing
`collect_positions_for_item` lines, and kept DAC profile-on stdout identical to
DAC profile-off stdout.

### 2026-05-10 Phase 2.1 Legacy Inventory

This was a translation/log/generated-code inventory run, not a benchmark or
translator fast-path change.

Artifact:
`/Volumes/QUQ/working/mpi_tmp/phase2_1_legacy_inventory`.

Contents:

- `translation_runs.tsv`
- `marker_scan.tsv`
- per-case `step2.log`
- per-case generated-code snapshots

Readout:

- The run covered the 14 benchmark cases plus focused `mpi*` fixtures, 63
  current translations total.
- All 14 benchmark generated-code snapshots are free of `AccessPattern` and
  `PackPlan`.
- `vectorAddCombo` remains OR `Contiguous1D` chain lowering, not legacy.
- Remaining true legacy wrapper evidence is in focused FixedBlock reject
  fixtures only. Remaining `PackPlan` evidence is Phase-C stencil fallback
  infrastructure.
- No simple contiguous 1D legacy fast-path candidate was selected from current
  evidence.

### 2026-05-10 Phase 2 Closeout

Phase 2.1 evidence was rechecked against the artifact above. The closeout did
not add benchmark numbers and did not change translator code.

Readout:

- Phase 2.2 through Phase 2.4 are closed as blocked/parked/no-op completion.
- No legacy simple contiguous 1D fast path was implemented because current
  simple contiguous 1D benchmark/fixture shapes already route through OR.
- The remaining legacy wrapper evidence is FixedBlock reject surface, and the
  remaining `AccessPattern`+`PackPlan` evidence is Phase-C stencil fallback
  surface. Neither is a Phase 2 fast-path target.

Prepared full-suite benchmark command:

```bash
cd /Volumes/QUQ/working/dacpp
MPI_ONLY_BENCH_TMP_DIR=/Volumes/QUQ/working/mpi_tmp/phase2_closeout_full_benchmark \
MPI_ONLY_BENCH_RANKS=4 \
MPI_ONLY_BENCH_TIMEOUT_SECONDS=1800 \
python3 clang/tools/translator/bench_mpi_only_requested.py \
  DFT1.0 FOuLa1.0 MDP1.0 decay1.0 gradientSum imageAdjustment1.0 \
  jacobi1.0 liuliang1.0 mandel1.0 matMul1.0 oddeven0.1 \
  stencil1.0 vectorAddCombo waveEquation1.0
```

Expected artifact:
`/Volumes/QUQ/working/mpi_tmp/phase2_closeout_full_benchmark`, with
`results.tsv` plus per-case patched sources, generated MPI source snapshots,
build logs, translation logs, run logs, and binaries.

### 2026-05-10 Phase 3 Materialization/Residency Closeout

Phase 3 built a generated-code materialization inventory and implemented one
narrow resident-registry optimization. It did not change final output
visibility semantics and did not defer all-ranks broadcasts.

Inventory artifact:
`/Volumes/QUQ/working/phase3_mpi_tmp/phase3_1_materialization_inventory`.

Benchmark artifacts:

- Before full benchmark:
  `/Volumes/QUQ/working/phase3_mpi_tmp/phase3_0_baseline_full_benchmark/results.tsv`
- After full benchmark:
  `/Volumes/QUQ/working/phase3_mpi_tmp/phase3_1_after_full_benchmark/results.tsv`
- Before focused profile:
  `/Volumes/QUQ/working/phase3_mpi_tmp/phase3_0_baseline_profile_focus`
- After focused profile:
  `/Volumes/QUQ/working/phase3_mpi_tmp/phase3_1_after_profile_focus`

Full benchmark 4-rank single-run DAC wall comparison:

| Case | Before DAC-MPI (s) | After DAC-MPI (s) | Delta |
|---|---:|---:|---:|
| `DFT1.0` | 0.662391 | 0.723748 | +9.3% |
| `FOuLa1.0` | 0.980144 | 0.903839 | -7.8% |
| `MDP1.0` | 0.829755 | 0.841678 | +1.4% |
| `decay1.0` | 0.993337 | 0.937886 | -5.6% |
| `gradientSum` | 1.055654 | 1.122434 | +6.3% |
| `imageAdjustment1.0` | 1.002949 | 0.946576 | -5.6% |
| `jacobi1.0` | 0.962429 | 0.934480 | -2.9% |
| `liuliang1.0` | 0.846241 | 0.827197 | -2.3% |
| `mandel1.0` | 3.017854 | 3.039062 | +0.7% |
| `matMul1.0` | 0.856220 | 0.830678 | -3.0% |
| `oddeven0.1` | 0.827537 | 0.634033 | -23.4% |
| `stencil1.0` | 1.448044 | 1.604763 | +10.8% |
| `vectorAddCombo` | 0.839850 | 0.829981 | -1.2% |
| `waveEquation1.0` | 1.528466 | 1.585345 | +3.7% |

Focused profile 4-rank DAC wall medians:

| Case | Before (s) | After (s) | Delta |
|---|---:|---:|---:|
| `DFT1.0` | 0.207593 | 0.191026 | -8.0% |
| `decay1.0` | 0.425867 | 0.380414 | -10.7% |
| `gradientSum` | 0.440146 | 0.436874 | -0.7% |
| `jacobi1.0` | 0.366836 | 0.328427 | -10.5% |
| `mandel1.0` | 2.511430 | 2.581752 | +2.8% |
| `vectorAddCombo` | 0.257293 | 0.265042 | +3.0% |

Readout:

- The optimization removes only unnecessary resident registry updates after a
  materialized write when no downstream resident reader exists.
- Required final `Gatherv`, `array2Tensor`, and `Bcast` remain in generated
  code. The inventory records their semantic reasons per case.
- The skipped registry update is not separately segmented, so profile
  attribution does not show a clean materialize/gather/bcast drop.
- `vectorAddCombo` remains structurally cleaner but wall-clock neutral at the
  focused scale.
- `decay1.0` still has semantic per-run gather+bcast because the source copies
  `local_A_tensor` into `A_tensor` immediately after each DAC expression.
- Single-run full benchmark deltas are noisy; use the 3-trial focused profile
  for close comparisons.

### 2026-05-11 FixedBlock Output-Sync Follow-up

The old legacy wrapper's Gather/Bcast policy was compared against current OR
paths. Ordinary OR already used the shared output-sync result, but standalone
FixedBlock still broadcast materialized output unconditionally.

Implemented follow-up:

- Standalone FixedBlock final materialization now emits final `MPI_Bcast` only
  when `broadcastMaterializedOutput` is true.
- Root `Gatherv` and root `array2Tensor` remain unchanged.
- Loop-resident P5 phase exchange was left unchanged because its final
  materializer writes the contracted source/reader tensor, not the standalone
  writer output.

Artifacts:

- Before:
  `/Volumes/QUQ/working/phase3_mpi_tmp/phase3_2_fixedblock_bcast_before`
- After:
  `/Volumes/QUQ/working/phase3_mpi_tmp/phase3_2_fixedblock_bcast_after`
- Micro benchmark:
  `/Volumes/QUQ/working/phase3_mpi_tmp/phase3_2_fixedblock_bcast_micro_benchmark`

Micro result for the tiny 8-element fixture:

| Variant | Median wall (s) | Mean wall (s) |
|---|---:|---:|
| Before | 0.076915 | 0.135527 |
| After | 0.076529 | 0.186126 |

Interpretation: runtime is too small/noisy for a speed claim. The useful
evidence is structural: root-only standalone FixedBlock removes two final
`MPI_Bcast` sites, while the all-ranks FixedBlock counterpart keeps Bcast.

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
| `decay1.0` | Phase 3 inventory confirms per-run gather+bcast is tied to source-level host copy into `A_tensor`; registry update after materialization is now skipped. | Needs a later source/contract proof to avoid per-run host visibility. |
| `gradientSum` | Current profile points mostly at row-partition input scatter, not materialization. | Investigate input distribution/layout reuse if this remains slow. |
| `imageAdjustment1.0` | Two-expression OR chain retains the first output for downstream resident use and materializes only the final host-visible output. | Keep semantic alignment note with benchmark logs. |
| `jacobi1.0` | Replicated full tensor broadcast is expected by current row-partition shape; final no-downstream resident update is skipped. | Re-run and inspect broadcast sensitivity by N/iter. |
| `liuliang1.0` | Current resident-halo path is close to parity. | Keep as 1D resident regression checkpoint. |
| `mandel1.0` | Current profile is kernel dominated with required final gather/bcast/materialize. | Investigate kernel-side and output-sync alternatives only with proof. |
| `matMul1.0` | RowPartition path has expected payload broadcast cost. | Compare against block/tile-aware future layout if added. |
| `oddeven0.1` | Current P5 phase exchange is strong. | Keep focused P5 regression and rank-scaling tests. |
| `stencil1.0` | Current 2D resident halo is near parity or slightly faster in closeout. | Keep as P4.6 regression checkpoint. |
| `vectorAddCombo` | Current OR chain keeps intermediates resident and only final output materializes/bcasts for all-ranks visibility. | Further broadcast/materialize removal requires output-sync proof. |
| `waveEquation1.0` | Current 2D resident halo with direct-reader extension is near parity. | Keep as B3 direct-reader regression checkpoint. |

## Benchmark TODO

1. Re-run all 14 benchmark cases under the current P6 code with at least three
   runs per side and report medians.
2. Store raw `results.tsv`, logs, generated source snapshots, and the exact git
   commit or diff state with the benchmark summary. Phase 1.4 now does this for
   the focused profile run; repeat it for full-suite runs.
3. Add per-phase timing for generated code. Phase 1.4 now covers focused OR,
   resident-halo, P5, FOuLa, and a legacy fallback smoke with:
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
