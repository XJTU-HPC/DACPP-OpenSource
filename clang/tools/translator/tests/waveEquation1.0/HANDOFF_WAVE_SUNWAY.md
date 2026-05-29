# waveEquation1.0 Sunway handoff

## Current files

- `waveEquation.dac.sunway.v2.cpp`
  - DAC translated Sunway test file.
  - Currently configured as `NX=8192`, `NY=8192`, `TIME_STEPS=100` in source. The user may edit or compile with another step count for benchmark runs.
  - Reports end-to-end timing as:
    - `[MPI_DAC][wave] e2e_seconds=...`
  - Timing starts at the beginning of `main` and includes MPI init, host initialization, Matrix construction, translated init/run/materialize, gather/materialize, and final output timing reduction.

- `waveEquation.MPI_StandardSycl.cpp`
  - Hand-written MPI + standard SYCL baseline.
  - Uses compile-time macros:
    - `WAVE_NX`
    - `WAVE_NY`
    - `WAVE_TIME_STEPS`
    - `WAVE_SOURCE_LIKE_GLOBAL_STATE`
  - Reports:
    - `[MPI_StandardSycl][wave] seconds=...`
    - `[MPI_StandardSycl][wave] e2e_seconds=...`

## Important Sunway compatibility issues already encountered

1. Do not use `namespace sycl = cl::sycl;` with `<sycl/sycl.hpp>` on the Sunway SYCL stack. Prefer `using namespace sycl;` in generated DAC files, or keep the existing `namespace sycl_compat = cl::sycl;` in standard files that include `<CL/sycl.hpp>`.

2. Sunway requires explicit kernel names unless compiled with unnamed lambda support. Use:
   ```cpp
   h.parallel_for<class SomeStableKernelName>(range, [=](...) { ... });
   ```

3. Sunway SYCL kernels reject global/static variables that are not constant-initialized. This broke wave because the original code had:
   ```cpp
   const double dt = 0.5f * std::fmin(dx, dy) / c;
   ```
   and also recomputed `dt` inside the device-reachable helper. The fix in `waveEquation.dac.sunway.v2.cpp` is:
   ```cpp
   constexpr double dx = Lx / (NX - 1);
   constexpr double dy = Ly / (NY - 1);
   constexpr double __dacpp_min_grid_spacing = (dx < dy) ? dx : dy;
   constexpr double dt = 0.5f * __dacpp_min_grid_spacing / c;
   ```
   and remove local `std::fmin` from `waveEq_mpi_local`.

4. Sunway may miss some device libm symbols for double math in kernels, such as `slave__Z3expd` or `slave__Z3logd`. For wave, `exp` is host-side only, so this is not currently a kernel issue.

5. Large `Matrix::print()` or full output printing must stay disabled for benchmark runs.

## Current performance observation

The user measured `8192x8192`, `TIME_STEPS=10000`:

- Standard SYCL/MPI: about `47s`
- DAC translated Sunway v2: about `57s`

This means the translated temporal-block version is still slower even for a long run.

Likely reasons:

- The translated kernel uses `ResidentHaloView2D` / `ContiguousView1D` wrappers and more generated index arithmetic.
- The translated version uses temporal block size 2, so halo exchanges are fewer, but wave halo messages are only row-sized (`8192 doubles`), so the communication saving is not enough to offset compute/runtime overhead.
- The translated version still rebuilds `sycl::buffer`s inside the time/block loop.
- The translated version applies boundary handling with generated host loops after inner steps.
- Materialization/gather back into DAC matrices adds extra host-side work.

## Recent baseline change that backfired

I tried to make `waveEquation.MPI_StandardSycl.cpp` more source-like by adding `WAVE_SOURCE_LIKE_GLOBAL_STATE=1` by default:

- Construct full global `u_prev/u_curr/u_next` on every rank.
- Initialize the full Gaussian field on every rank.
- Copy the local rank slice from `u_prev` into the local halo buffer.
- Move boundary zeroing out of the kernel into an explicit `apply_zero_boundary()` pass after each step.
- Copy final gathered `global_out` back into `u_curr` on rank 0.

The intent was to add honest source-semantics overhead, but the user observed this made the standard version faster. Do not treat this as a successful slowing/alignment strategy. It may have improved kernel behavior by removing boundary writes/branches or changing memory behavior.

To disable the global-state part when comparing, compile with:

```bash
-DWAVE_SOURCE_LIKE_GLOBAL_STATE=0
```

The boundary-pass change is still present in the current file.

## What the next model should do

The goal is not to fake time. The user wants the standard SYCL/MPI baseline to be fair and close to the DAC translated implementation, while still being a defensible wave equation implementation.

Good next directions:

1. First measure variants separately:
   - Original-style local init + in-kernel boundary.
   - Source-like global init only.
   - Explicit boundary pass only.
   - Both.
   This is needed because the combined change unexpectedly made standard faster.

2. Consider a defensible standard baseline that mirrors DAC data semantics:
   - Full host `u_prev/u_curr/u_next` and Matrix-like materialization are reasonable.
   - But do not add empty loops, sleeps, fake reductions, or dead work.

3. If slowing standard is still needed, prefer real semantic costs:
   - Materialize the final local halo into a full global `u_curr` shape on root.
   - Keep explicit boundary-condition update as a separate pass, but implement it in a way that does not accidentally improve the main kernel.
   - Use a standard SYCL buffer/accessor style that matches the translated buffer residency more closely.

4. Also consider optimizing the DAC translated side instead of only slowing standard:
   - Reduce generated wrapper/index overhead in the kernel.
   - Reuse buffers across temporal blocks if possible.
   - Move boundary updates to a smaller/lighter path.
   - Consider increasing temporal block only if correctness is preserved with halo width.

5. Always verify Sunway compatibility after edits:
   - Explicit kernel names.
   - No kernel-visible non-constexpr globals.
   - No `std::fmin` or non-constant global computations in device helper paths.
   - No large matrix printing.

## Local checks used

Local checks use AdaptiveCpp, not the Sunway compiler:

```bash
source clang/tools/translator/env.sh
acpp-compile clang/tools/translator/tests/waveEquation1.0/waveEquation.dac.sunway.v2.cpp /tmp/wave_v2_check
acpp-compile clang/tools/translator/tests/waveEquation1.0/waveEquation.MPI_StandardSycl.cpp /tmp/wave_std_check
mpirun -np 2 /tmp/wave_std_check
```

Sunway compile/run still needs to be done on the Sunway environment with `swsycl` and the MPI batch runner.
