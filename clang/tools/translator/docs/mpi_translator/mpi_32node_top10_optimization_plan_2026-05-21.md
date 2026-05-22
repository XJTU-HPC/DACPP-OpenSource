# MPI 32-Node Top-11 Optimization Plan

This note records the current 32-rank optimization target list for the DACPP MPI
translator. It is based on the benchmark driver:

```bash
python3 clang/tools/translator/bench_mpi_profile_segments.py \
  --tmp-dir /Volumes/QUQ/working/mpi_tmp/profile_segments_current_all_np4_t1 \
  --ranks 4 --trials 1
```

The generated benchmark artifacts used for the initial inspection are under:

```text
/Volumes/QUQ/working/mpi_tmp/profile_segments_current_all_np4_t1
```

The ranking below removes `vectorAddCombo` and targets strong scaling to 32 MPI
ranks. It prioritizes generated-code communication structure, arithmetic
intensity, synchronization frequency, and expected load balance. Exact ordering
should be revalidated with a 32-rank profile after each optimization batch.

## 32-Rank Target Order

| Rank | Case | Current generated path | Main 32-rank limiter |
|---:|---|---|---|
| 1 | `matMul1.0` | `RowPartitionFullRow` OR | root-side `B` pack/broadcast and unnecessary output scatter |
| 2 | `DFT1.0` | `ReplicatedFullTensor` OR | avoidable `vec[i]=i` scatter and repeated replicated input broadcast |
| 3 | `imageAdjustment1.0` | `RowBlock2D` OR chain | two memory-bound kernels and launch overhead |
| 4 | `decay1.0` | P4.5 loop-lowered `Contiguous1D` | many tiny per-step kernel launches at 32 ranks |
| 5 | `stencil1.0` | P4.6 2D resident halo | per-step row halo latency |
| 6 | `waveEquation1.0` | P4.6 2D resident halo + direct reader | per-step row halo plus three-buffer overhead |
| 7 | `MDP1.0` | P4.6 1D resident halo | 600 small kernel launches and halo exchanges |
| 8 | `FOuLa1.0` | owner-loop 1D resident halo | boundary history broadcast and per-step halo latency |
| 9 | `liuliang1.0` | P4.6 1D resident halo | 1000 small kernel launches and final double gather |
| 10 | `mandel1.0` | `Contiguous1D` OR + scalar reduction | contiguous static partition load imbalance |
| 11 | 2D spatial block partition | `StencilWindow2D` / `RowBlock2D` row-block layouts | row-only partitioning limits surface/volume balance and 2D locality |

## Global Implementation Rules

- Keep every change pattern/layout driven. Do not special-case benchmark names.
- Preserve current conservative fallbacks for aliasing, unknown host use,
  unsupported layouts, dynamic indices, shape mismatch, and non-root/all-rank
  host dependencies.
- Prefer small, independently testable steps. Each accepted optimization should
  add a translator log line and at least one `mpi_expect.txt` structure guard
  when a suitable fixture exists.
- Measure with segmented profiling. Use `max_ms` for critical-path segment
  interpretation and `sum_ms` only for attribution across ranks.
- For 32-rank work, always compare profile-off wall time and profile-on segment
  data, because the profile collectives themselves can perturb small kernels.

## Recommended Optimization Batches

1. Low-risk single-wrapper cleanup: `matMul1.0` output scatter removal,
   `DFT1.0` local index initializer, and `FOuLa1.0` local boundary formulas.
2. Launch/communication batching for resident loops: `decay1.0`, `MDP1.0`,
   `liuliang1.0`, then 2D `stencil1.0`/`waveEquation1.0`.
3. Memory/chain codegen: `imageAdjustment1.0` RowBlock2D chain fusion.
4. Load-balancing policy: `mandel1.0` block-cyclic or cyclic contiguous output
   distribution.

## Case Plans

### 1. `matMul1.0`

Current shape:

- Layout: `RowPartitionFullRow`.
- `matA` is scattered by output rows.
- `matB` is packed on root into column payload order and broadcast to all ranks.
- `matC` is output-only for the benchmark, but the generated wrapper still
  scatters the old output buffer before the kernel.
- Final post-use is bounded root read of `matC[0][0]`.

Optimization:

- Add an output-direct no-read fast path in resident buffer codegen. If the
  output parameter is written but never read by the local kernel before write,
  initialize the local output buffer with zero/default locally and skip the
  input `MPI_Scatterv` for that output tensor.
- Add optional packed replicated payload cache for full-column readers such as
  `matB`. The first wrapper invocation packs and broadcasts; later invocations
  with the same tensor/shape/layout reuse resident replicated storage.

Likely files:

- `rewriter/lib/mpi/operator_resident/ResidentBufferCodegen.cpp`
- `rewriter/lib/mpi/operator_resident/OperatorChainAnalysis.cpp`
- `rewriter/include/mpi/operator_resident/OperatorResidentPlan.h`
- `dpcppLib/include/mpi/operator_resident/OperatorResidentRuntime.h`

Acceptance:

- Generated `matMul1.0` has no `MPI_Scatterv` for the output `dotProduct` /
  `matC` initial value.
- Generated code still keeps the final bounded root sync for `matC[0][0]`.
- `bash clang/tools/translator/test_mpi.sh matMul1.0` passes.
- 32-rank profile shows lower `scatter`/`pack` time without changing output.

Fallback:

- If the output has any read-before-write, compound update, alias escape, or
  uncertain post-use requiring prior value, keep the existing scatter path.

Implementation Status (2026-05-21):

- Changed files: `rewriter/include/mpi/operator_resident/OperatorResidentPlan.h`,
  `rewriter/lib/mpi/operator_resident/OperatorChainAnalysis.cpp`,
  `rewriter/lib/mpi/operator_resident/ResidentBufferCodegen.cpp`,
  `tests/matMul1.0/mpi_expect.txt`,
  `tests/matMul1.0/mpi_expect_large.txt`, and
  `tests/matMul1.0/matMul.large_dac.cpp`.
- Accepted pattern: generic `OutputDirect` local output initialization when the
  output is write-only, or when `+=` local accumulation is proven against a
  default/zero output initializer and the output tensor has no unsafe use before
  the shell call.
- Fallback conditions: unsupported calc write pattern, output RHS reading the
  output, increment/decrement or non-`+=` compound writes, nonzero or unsupported
  output vector initializer, missing direct vector-backed tensor construction,
  tensor/vector mutation, alias escape, or function/member escape before the
  shell call.
- Generated-code evidence: large profile probe log contains
  `output matC init-sync=local-default value=0 reason=default-initialized local accumulation`;
  generated `matMul1.0` emits `std::fill(__or_local_dotProduct.begin(), ...)`
  and the comment `Output-direct no-read fast path for dotProduct initializes
  local output and skips root pack/scatter.`, while the final
  `MPI_Gatherv(__or_local_dotProduct.data()...)` remains.
- Tests run: `cmake --build build --target translator -j8`;
  `bash clang/tools/translator/test_mpi.sh matMul1.0 DFT1.0 imageAdjustment1.0`;
  `bash clang/tools/translator/test_mpi.sh --large matMul1.0`;
  `bash clang/tools/translator/test_mpi.sh jacobi1.0 gradientSum vectorAddCombo mpiBroadcastTensor2Array mpiOrReadWriteAccumulate2D`;
  and the 4-rank one-trial profile probe in
  `/Volumes/QUQ/working/mpi_tmp/profile_segments_top3_probe` with status `ok`
  for `matMul1.0`.
- Remaining risk: the accepted `+=` path is intentionally narrow and only covers
  local default accumulation shapes currently proven by the calc AST; more
  complex read-before-write-free reductions still fall back.

### 2. `DFT1.0`

Current shape:

- Layout: `ReplicatedFullTensor` for input and direct-mapped output/index.
- Full input is broadcast to all ranks.
- `vec_tensor` contains `vec[i] = i`, but current generated code scatters it.
- Final output post-use is bounded root read of `output_tensor[0]`.

Optimization:

- Extend init-sync analysis to recognize simple affine index-local initializers
  such as `vec[i] = i`, `vec[i] = i + c`, and safe casts of those forms.
- Generate local index values from `rank_range.begin + local_i` instead of
  root `tensor2Array` plus `MPI_Scatterv`.
- Optionally cache replicated input for repeated DFT wrappers in the same
  program when the input tensor is read-only and shape-stable.

Likely files:

- `rewriter/lib/mpi/operator_resident/OperatorChainAnalysis.cpp`
- `rewriter/lib/mpi/operator_resident/ResidentBufferCodegen.cpp`
- `rewriter/lib/mpi/operator_resident/CollectiveCodegenUtils.cpp`

Acceptance:

- `DFT1.0` translation log reports index-local init for `vec_tensor`.
- Generated `DFT1.0` has no `MPI_Scatterv` for `vec_tensor`.
- `bash clang/tools/translator/test_mpi.sh DFT1.0` passes.
- 32-rank profile reduces `scatter` and `pack` segments.

Fallback:

- Any non-affine initializer, vector mutation, alias escape, function escape,
  dynamic dependency, or tensor mutation before the shell call must retain the
  existing scatter path.

Implementation Status (2026-05-21):

- Changed files: `rewriter/lib/mpi/operator_resident/OperatorChainAnalysis.cpp`,
  `rewriter/lib/mpi/operator_resident/ResidentBufferCodegen.cpp`, and
  `tests/DFT1.0/mpi_expect.txt`.
- Accepted pattern: direct-mapped one-dimensional input tensor constructed from
  a vector whose preceding loop is a canonical zero-based full-size fill with
  `vec[i] = i`, `vec[i] = i + c`, or safe cast/expression forms that reference
  only the loop index and globally visible constants/functions. The layout gate
  now permits both `Contiguous1D` and `ReplicatedFullTensor`.
- Fallback conditions: noncanonical loop bounds, non-affine or unsafe RHS,
  references to local state not visible in the wrapper, vector writes or unsafe
  member calls after initialization, vector alias/function escape, tensor
  mutation or escape, unsupported tensor construction, or unknown enclosing
  function body.
- Generated-code evidence: profile probe translation log reports
  `input vec_tensor init-sync=index-local expr=i`; generated `DFT1.0` fills
  local values with `const int64_t i = __or_range.begin + __or_local_i;` and
  emits the comment `Index-generated input vec is filled locally; skip root
  pack/scatter.`.
- Tests run: `cmake --build build --target translator -j8`;
  `bash clang/tools/translator/test_mpi.sh matMul1.0 DFT1.0 imageAdjustment1.0`;
  nearby index-local regressions through
  `bash clang/tools/translator/test_mpi.sh jacobi1.0 gradientSum vectorAddCombo mpiBroadcastTensor2Array mpiOrReadWriteAccumulate2D`;
  and the 4-rank one-trial profile probe with `DFT1.0` status `ok`.
- Remaining risk: the recognizer is conservative and does not yet synthesize
  general affine expressions with arbitrary local constants; such cases retain
  the old scatter path.

### 3. `imageAdjustment1.0`

Current shape:

- Layout: `RowBlock2D` OR chain.
- Initial image data is constant-local for aggregate `Pixel` values on current
  optimized branches.
- Two pointwise kernels run in sequence and the intermediate output is resident.
- Final host use is bounded root read of `image_tensor3[0][0]`.

Optimization:

- Add RowBlock2D chain fusion for pure pointwise direct-mapped chains where the
  output of one shell is consumed by exactly one downstream shell with identical
  row-block partitioning.
- Generate one fused SYCL kernel that computes the first expression into a
  private/local value and immediately feeds the second expression, materializing
  only the final tensor state needed by downstream host use.

Likely files:

- `rewriter/lib/mpi/operator_resident/OperatorChainAnalysis.cpp`
- `rewriter/lib/mpi/operator_resident/ResidentBufferCodegen.cpp`
- `rewriter/include/mpi/operator_resident/OperatorResidentPlan.h`

Acceptance:

- Translation log reports a fused RowBlock2D chain for the two image wrappers.
- Generated code has one kernel launch for the two pointwise image operations.
- Intermediate full buffer is absent or reduced to a scalar/private value inside
  the fused kernel.
- `bash clang/tools/translator/test_mpi.sh imageAdjustment1.0` passes.

Fallback:

- Reject fusion if the intermediate tensor has host use, multiple consumers,
  aliasing, non-pointwise access, shape mismatch, different partitioning,
  unsupported side effects, or required materialization between shells.

Implementation Status (2026-05-21):

- Changed files: `rewriter/include/Rewriter_MPI_OperatorResident.h`,
  `rewriter/include/mpi/operator_resident/OperatorResidentPlan.h`,
  `rewriter/lib/Rewriter_MPI.cpp`,
  `rewriter/lib/mpi/operator_resident/LocalKernelCodegen.cpp`,
  `rewriter/lib/mpi/operator_resident/OperatorChainAnalysis.cpp`,
  `rewriter/lib/mpi/operator_resident/OperatorResidentCodegen.cpp`,
  `rewriter/lib/mpi/operator_resident/OperatorResidentCodegen_Internal.h`,
  `rewriter/lib/mpi/operator_resident/OperatorResidentWrapperCodegen.cpp`, and
  `tests/imageAdjustment1.0/mpi_expect.txt`.
- Accepted pattern: two or more `RowBlock2D` direct-mapped pointwise shells with
  identical partition signatures, one direct reader and one output-only writer
  per shell, distinct reader/writer aliases, linear intermediate flow into the
  next shell, no multiple downstream consumers, and no host materialization need
  between shells.
- Fallback conditions: non-`RowBlock2D` layout, mismatched partition signature,
  missing single reader/writer, unproven alias distinctness, non-pointwise calc
  body, intermediate host sync/materialization, multiple consumers, or an
  intermediate that is not exactly the next shell's input.
- Generated-code evidence: profile probe translation log contains
  `chain=0 rowblock-pointwise-fusion=accepted`; generated code has one
  `__dacpp_mpi_or_fused_rowblock_imageAdjustment_image_1_image_2_0` wrapper,
  uses `Pixel __or_private_image_tensor2{}` inside the SYCL kernel, calls both
  `image_1_mpi_local(...)` and `image_2_mpi_local(...)` in the same kernel, and
  preserves the final `MPI_Gatherv(__or_local_image_tensor3.data()...)`.
- Tests run: `cmake --build build --target translator -j8`;
  `bash clang/tools/translator/test_mpi.sh matMul1.0 DFT1.0 imageAdjustment1.0`;
  RowBlock2D regression through
  `bash clang/tools/translator/test_mpi.sh jacobi1.0 gradientSum vectorAddCombo mpiBroadcastTensor2Array mpiOrReadWriteAccumulate2D`;
  and the 4-rank one-trial profile probe with `imageAdjustment1.0` status `ok`.
- Remaining risk: the fused wrapper/codegen is deliberately limited to linear
  pointwise chains. More complex DAGs, side effects, or shared intermediates
  continue through the existing resident-chain path.

### 4. `decay1.0`

Current shape:

- Layout: P4.5 loop-lowered `Contiguous1D`.
- Initial inputs are scattered once.
- There is no per-step MPI after the selective-row materialization optimization.
- Current 32-rank concern is many small per-step SYCL kernels.

Optimization:

- Add a loop-lowered direct "device time loop" mode for scalar-refresh patterns
  where each item can compute all time steps independently from local inputs and
  replicated scalar evolution.
- For the accepted shape, generate one kernel whose work-item loops over time
  locally and writes only the selected materialized row/output needed by host
  post-use.

Likely files:

- `rewriter/lib/mpi/operator_resident/LoopLoweredDirectCodegen.cpp`
- `rewriter/lib/mpi/operator_resident/OperatorChainAnalysis.cpp`
- `rewriter/include/mpi/operator_resident/OperatorResidentPlan.h`

Acceptance:

- Generated `decay1.0` does not launch one kernel per time step for the accepted
  direct shape.
- Selective-row host materialization remains in place.
- `bash clang/tools/translator/test_mpi.sh decay1.0` passes.
- 32-rank profile shows lower `kernel` wall contribution from launch overhead.

Fallback:

- Keep current loop-lowered host loop when scalar updates depend on host-side
  dynamic state, unsupported functions, cross-item dependencies, output
  writeback for many host-visible rows, or any uncertain post-loop use.

Implementation Status (2026-05-21):

- Changed files: `rewriter/include/mpi/operator_resident/OperatorResidentPlan.h`,
  `rewriter/include/mpi/shared/LoopLoweredRewrite.h`,
  `rewriter/lib/Rewriter_MPI.cpp`,
  `rewriter/lib/mpi/operator_resident/LoopLoweredDirectCodegen.cpp`,
  `rewriter/lib/mpi/operator_resident/OperatorChainAnalysis.cpp`,
  `rewriter/lib/mpi/shared/LoopLoweredRewrite.cpp`, and
  `tests/decay1.0/mpi_expect.txt`.
- Accepted pattern: generic P4.5 loop-lowered `Contiguous1D`
  scalar-refresh shape where each local item can replay the time loop from
  local input plus replicated scalar evolution, with one selective-row
  host materialization target after the loop.
- Fallback conditions: scalar refresh depending on host dynamic state,
  unsupported scalar update/function shape, cross-item dependency, nonlocal
  output writeback, many host-visible output rows, unsafe tensor use in the row
  expression, or uncertain post-loop host use keeps the existing host-side
  loop-lowered path.
- Generated-code evidence: profile probe translation log contains
  `scalar-refresh=local-replicated device-time-loop=accepted reason=scalar-refresh independent items with selective-row materialization`;
  generated code contains `__dacpp_mpi_or_DECAY_decay_0_run_loop(...)`, the
  comment `P4.5 device time loop`, and a guarded selected-row materialization
  call instead of one SYCL launch per source time step.
- Tests run: `cmake --build build --target translator -j8`;
  `bash clang/tools/translator/test_mpi.sh decay1.0 stencil1.0`;
  nearby regressions through
  `bash clang/tools/translator/test_mpi.sh waveEquation1.0 MDP1.0 FOuLa1.0 liuliang1.0`;
  additional loop/stencil fixtures through
  `bash clang/tools/translator/test_mpi.sh mpiLoopStencilCountGuard2D mpiLoopStencilResidentHalo1D mpiLoopStencilResidentHaloEmptyRank1D mpiLoopStencilOrderReject2D mpiLoopStencilScalarReject2D`;
  and the 4-rank one-trial profile probe with `decay1.0` status `ok`.
- Remaining risk: the device-time-loop recognizer is intentionally narrow and
  currently covers the proven single replicated-scalar refresh shape. More
  general scalar recurrences, function-heavy updates, or multiple host-visible
  rows retain the existing per-step host loop.

### 5. `stencil1.0`

Current shape:

- Layout: P4.6 `StencilWindow2D` resident halo.
- Each time step launches one local kernel and exchanges top/bottom halo rows.
- Final benchmark post-use is bounded root read; full gather is already avoided
  on current optimized paths.

Optimization:

- Implement conservative temporal blocking for 2D row-block stencil:
  exchange a halo of width `k`, run `k` local steps, then exchange again.
- Start with static `k=2` and make the codegen fall back unless window shape,
  followup offset, boundary-local updates, and loop structure match the current
  proven resident-halo contract.
- Longer term, make `k` tunable through an environment variable or compile-time
  constant after correctness is stable.

Likely files:

- `rewriter/lib/mpi/operator_resident/StencilWindowCodegen.cpp`
- `rewriter/lib/mpi/operator_resident/OperatorChainAnalysis.cpp`
- `dpcppLib/include/mpi/operator_resident/OperatorResidentRuntime.h`

Acceptance:

- Generated code reports `temporal-block=2` or equivalent in the translation
  log for accepted `StencilWindow2D`.
- Halo exchange calls drop by about half for `k=2`.
- `bash clang/tools/translator/test_mpi.sh stencil1.0 waveEquation1.0` passes
  after shared runtime changes.
- 32-rank profile shows reduced `halo` max time without a kernel correctness
  regression.

Fallback:

- Reject when boundary conditions cannot be replayed locally for `k` steps,
  window/followup offsets differ from the proven contract, a direct reader is
  present but unsupported by the block, or the loop has host-visible state
  between steps.

Implementation Status (2026-05-21):

- Changed files: `dpcppLib/include/mpi/operator_resident/OperatorResidentRuntime.h`,
  `rewriter/include/Rewriter_MPI_OperatorResident.h`,
  `rewriter/include/mpi/operator_resident/OperatorResidentPlan.h`,
  `rewriter/lib/mpi/legacy_access_pattern/PatternInit.cpp`,
  `rewriter/lib/mpi/operator_resident/OperatorChainAnalysis.cpp`,
  `rewriter/lib/mpi/operator_resident/OperatorResidentCodegen.cpp`,
  `rewriter/lib/mpi/operator_resident/StencilWindowCodegen.cpp`,
  `tests/stencil1.0/mpi_expect.txt`, and
  `tests/waveEquation1.0/mpi_expect.txt`.
- Accepted pattern: conservative `StencilWindow2D` resident-halo temporal
  blocking with static `k=2`, canonical 3x3 window, matching followup offset
  `(1,1)`, boundary-local replayable updates, no unsupported direct reader,
  and the proven resident-halo loop structure.
- Fallback conditions: unsupported or noncanonical window/followup offset,
  boundary conditions that cannot be locally replayed for two steps,
  direct-reader recurrence, loop-visible host state, unsupported reader access,
  or too-narrow runtime row ownership for a two-row halo uses the existing
  one-step resident-halo path.
- Generated-code evidence: `stencil1.0` profile probe translation log reports
  `temporal-block=2 accepted`; generated code contains
  `__or_runtime_temporal_block_size`,
  `exchange_halo_2d_rows_temporal_inplace`, an inner
  `for (__or_block_step...)` loop, and `owned_slice_2d_rows_temporal`. The
  same probe reports 1200 halo calls for 600 steps on four ranks, while
  `waveEquation1.0` logs `temporal-block=2 rejected reason=direct-reader
  recurrence not enabled for k=2` and keeps 2400 halo calls.
- Tests run: `cmake --build build --target translator -j8`;
  `bash clang/tools/translator/test_mpi.sh decay1.0 stencil1.0`;
  nearby regressions through
  `bash clang/tools/translator/test_mpi.sh waveEquation1.0 MDP1.0 FOuLa1.0 liuliang1.0`;
  additional loop/stencil fixtures through
  `bash clang/tools/translator/test_mpi.sh mpiLoopStencilCountGuard2D mpiLoopStencilResidentHalo1D mpiLoopStencilResidentHaloEmptyRank1D mpiLoopStencilOrderReject2D mpiLoopStencilScalarReject2D`;
  and the 4-rank one-trial profile probe with `stencil1.0`,
  `waveEquation1.0`, and `MDP1.0` status `ok`.
- Remaining risk: temporal blocking is limited to canonical no-direct-reader
  2D row-halo stencils. Small or highly partitioned grids can dynamically use
  block size 1 to preserve correctness, and wave-style direct-reader temporal
  blocking is left for a separate proof.

### 6. `waveEquation1.0`

Current shape:

- Layout: P4.6 `StencilWindow2D` resident halo with direct-reader extension.
- Uses resident `prev`, `cur`, and `next` buffers with role rotation.
- `cur` is constant-local in current optimized paths; `prev` is scattered.
- Per-step row halo and three-buffer launch/memory overhead dominate at high
  rank counts.

Optimization:

- First ensure generated code uses pure buffer/pointer role rotation and avoids
  avoidable copies or resident registry rewrites for `prev/cur/next`.
- Then extend the 2D temporal blocking from `stencil1.0` to the direct-reader
  shape. The block must preserve wave recurrence semantics across `prev/cur`
  rotations.
- Keep direct-reader initialization optimizations such as constant-local `cur`
  and bounded final post-use sync.

Likely files:

- `rewriter/lib/mpi/operator_resident/StencilWindowCodegen.cpp`
- `rewriter/lib/mpi/operator_resident/OperatorChainAnalysis.cpp`
- `dpcppLib/include/mpi/operator_resident/OperatorResidentRuntime.h`

Acceptance:

- Translation log still reports `direct-reader=true`.
- Generated code has no full final gather for benchmark bounded post-use.
- Halo calls drop only after the direct-reader temporal block is accepted.
- `bash clang/tools/translator/test_mpi.sh waveEquation1.0 stencil1.0` passes.
- 32-rank profile shows reduced `halo` and no increase in `scatter`.

Fallback:

- If the direct-reader recurrence cannot be proven for multi-step execution,
  retain one-step resident halo and only apply role-rotation cleanup.

Implementation Status (2026-05-22):

- Changed files: `rewriter/lib/mpi/operator_resident/OperatorChainAnalysis.cpp`,
  `rewriter/lib/mpi/operator_resident/StencilWindowCodegen.cpp`,
  and `tests/waveEquation1.0/mpi_expect.txt`.
- Accepted pattern: P4.6 `StencilWindow2D` direct-reader resident halo temporal
  blocking with static `k=2`, resident halo accepted, canonical 3x3 window,
  followup offset `(1,1)`, direct-reader read-cache offset `(-1,-1)`,
  `prev/cur/next` role rotation represented as pure resident buffer swaps,
  canonical zero boundary-local replay after each inner substep, canonical
  zero-based loop bound, no scalar readers, exactly one supported direct reader,
  no unsupported writes outside the lowered OR path, and the current
  DAC -> read-cache -> followup -> boundary statement order.
- Fallback conditions: reject or retain the one-step path for noncanonical
  window/followup/read-cache offsets, missing or extra direct readers, scalar
  readers, unsupported direct-reader recurrence edges, unreplayable boundaries,
  noncanonical loop bounds, host-visible state between substeps, unsupported
  writes or aliasing outside the OR path, or runtime row partitions too narrow
  for `k=2`. Narrow row partitions keep the runtime block size at 1.
- Generated-code/log evidence: `waveEquation1.0` translation logs
  `direct-reader=true read-cache-offset=(-1,-1) role-rotation=buffer-swap` and
  `temporal-block=2 accepted direct-reader recurrence`. Generated code emits
  `resident_halo_2d_row_layout_temporal`,
  `scatter_window_2d_rows_temporal` for the widened direct-reader `prev`
  state, `exchange_halo_2d_rows_temporal_inplace`, an inner
  `for (__or_block_step...)` loop, current-substep `view_prev` and `view_cur`
  offsets based on `__or_compute_row_begin`, and
  `std::swap(__or_local_prev, __or_local_cur)` followed by
  `std::swap(__or_local_cur, __or_local_next)` after each substep. Boundary
  zero updates are replayed locally into the rotated current buffer. The fixture
  still uses full `matCur` materialization because the source has
  `matCur.print()`; no claim is made that full gather is removed for this
  fixture.
- Tests run: `cmake --build build --target translator -j8`;
  `bash clang/tools/translator/test_mpi.sh waveEquation1.0 MDP1.0`;
  `bash clang/tools/translator/test_mpi.sh stencil1.0 liuliang1.0 FOuLa1.0`;
  `bash clang/tools/translator/test_mpi.sh mpiLoopStencilResidentHalo1D mpiLoopStencilResidentHaloEmptyRank1D mpiLoopStencilRightBoundaryFullSync1D mpiLoopStencilCountGuard2D mpiLoopStencilOrderReject2D mpiLoopStencilScalarReject2D`;
  and a 4-rank, 3-trial profile probe in
  `/Volumes/QUQ/working/mpi_tmp/profile_segments_next6_wave_done_probe` for
  `waveEquation1.0`, `MDP1.0`, `stencil1.0`, `liuliang1.0`, and `FOuLa1.0`.
- Profile result: `waveEquation1.0` status `ok`, wall median `0.968383 s`,
  profile median `0.968777 s`, and halo median calls `1200` for 600 steps on
  four ranks. Nearby retained paths stayed on contract: `stencil1.0` halo calls
  `1200`, `MDP1.0` halo calls `1200`, `liuliang1.0` halo calls `4000`, and
  `FOuLa1.0` halo calls `2400`.
- Remaining risk: the proof is intentionally narrow and only covers the current
  row-block direct-reader recurrence. It does not implement the future 2D
  spatial block partition, does not generalize to arbitrary recurrence graphs,
  and still performs full `matCur` materialization when host post-use is a tensor
  member call such as `print()`.

### 7. `MDP1.0`

Current shape:

- Layout: P4.6 `StencilWindow1D` resident halo.
- One local kernel and one small halo exchange per time step.
- 32 ranks leave only a small number of items per rank, so latency dominates.

Optimization:

- Implement 1D temporal blocking with a widened halo. Start with `k=2` for
  window size 3 and followup offset 1.
- Alternatively, as a lower-risk first step, add a batched kernel path that
  keeps resident buffers in USM/in-order queue form while still exchanging every
  step. This reduces launch/buffer overhead but not MPI latency.

Likely files:

- `rewriter/lib/mpi/operator_resident/StencilWindowCodegen.cpp`
- `dpcppLib/include/mpi/operator_resident/OperatorResidentRuntime.h`

Acceptance:

- For accepted 1D resident halos, halo call count decreases with `k`.
- `bash clang/tools/translator/test_mpi.sh MDP1.0 mpiLoopStencilResidentHalo1D` passes.
- 32-rank profile shows lower `halo` and lower total wall time.

Fallback:

- Keep one-step resident halo for boundary-local updates, non-unit followup
  offsets, unsupported window sizes, dynamic loop bodies, or any host-visible
  side effect between steps.

Implementation Status (2026-05-21):

- Changed files: `dpcppLib/include/mpi/operator_resident/OperatorResidentRuntime.h`,
  `rewriter/lib/mpi/operator_resident/LoopLoweredStencil1DCodegen.cpp`,
  `rewriter/lib/mpi/operator_resident/OperatorChainAnalysis.cpp`,
  `tests/MDP1.0/mpi_expect.txt`,
  `tests/mpiLoopStencilResidentHalo1D/mpi_expect.txt`,
  `tests/mpiLoopStencilResidentHaloEmptyRank1D/mpi_expect.txt`, and
  `tests/liuliang1.0/mpi_expect.txt`.
- Accepted pattern: conservative P4.6 `StencilWindow1D` resident-halo temporal
  blocking with static `k=2`, canonical window size 3, followup offset 1,
  reader extent exactly writer extent plus two halo elements, no direct reader,
  no scalar reader, no read-cache route, no boundary-local update, and a
  zero-based canonical loop bound.
- Fallback conditions: one-step resident halo is kept for boundary-local
  updates, non-unit followup offsets, unsupported window sizes, dynamic or
  noncanonical loop bodies, host-visible side effects between steps, scalar or
  direct/unsupported readers, read-cache routes, and partitions too narrow for a
  safe two-step halo. Narrow runtime partitions use a block size of 1.
- Generated-code evidence: `MDP1.0` and the 1D resident halo fixtures log
  `temporal-block=2 accepted`; generated code emits
  `resident_halo_1d_layout_temporal`, `scatter_window_1d_temporal`,
  `exchange_halo_1d_temporal_inplace`, `__or_runtime_temporal_block_size`, an
  inner `for (__or_block_step...)` loop, and resident reader/writer
  `std::swap`. The 4-rank, 3-trial profile probe reports 1200 halo calls and
  1200 kernel calls for 600 steps. The `liuliang1.0` fixture logs
  `temporal-block=2 rejected reason=requires canonical 1D window size 3 and
  followup offset 1` and does not emit the temporal 1D helpers.
- Tests run: `cmake --build build --target translator -j8`;
  `bash clang/tools/translator/test_mpi.sh waveEquation1.0 MDP1.0`;
  `bash clang/tools/translator/test_mpi.sh stencil1.0 liuliang1.0 FOuLa1.0`;
  `bash clang/tools/translator/test_mpi.sh mpiLoopStencilResidentHalo1D mpiLoopStencilResidentHaloEmptyRank1D mpiLoopStencilRightBoundaryFullSync1D mpiLoopStencilCountGuard2D mpiLoopStencilOrderReject2D mpiLoopStencilScalarReject2D`;
  and a 4-rank, 3-trial profile probe for `waveEquation1.0`, `MDP1.0`,
  `stencil1.0`, `liuliang1.0`, and `FOuLa1.0`.
- Remaining risk: the implementation is intentionally limited to canonical
  no-direct-reader 1D halos. Boundary-local replay, owner-loop sharing, and
  larger or tunable temporal block sizes are left for later proof work.

### 8. `FOuLa1.0`

Current shape:

- Layout: owner-loop `StencilWindow1D` specialization.
- Scatters initial column once.
- Broadcasts left/right boundary histories once before the loop on current
  optimized paths.
- Exchanges 1D halo each time step and materializes selected row history.

Optimization:

- Replace boundary history broadcasts with local formula evaluation when the
  boundary assignment is a safe function of local scalar loop/time values and
  globally visible constants. This avoids root building and broadcasting
  boundary arrays.
- After that, share the 1D temporal blocking path with `MDP1.0` when the
  owner-loop contract proves boundary replay for the block.

Likely files:

- `rewriter/lib/mpi/operator_resident/LoopLocalStencilOwnerLoop.cpp`
- `rewriter/lib/mpi/operator_resident/OperatorChainAnalysis.cpp`
- `dpcppLib/include/mpi/operator_resident/OperatorResidentRuntime.h`

Acceptance:

- Translation log reports boundary-local formula evaluation or boundary-bcast
  elision for accepted owner loops.
- Generated `FOuLa1.0` has no `MPI_Bcast` for left/right boundary histories
  when formulas are accepted.
- `bash clang/tools/translator/test_mpi.sh FOuLa1.0` passes.
- 32-rank profile reduces `bcast` and, after temporal blocking, `halo`.

Fallback:

- Keep boundary history broadcast for table-driven boundaries, unknown function
  calls, non-deterministic functions, tensor reads, host mutation, aliasing, or
  any expression that cannot be safely re-evaluated on every rank.

Implementation Status (2026-05-22):

- Completed boundary history broadcast elision and owner-loop temporal blocking
  for the proven owner-loop `StencilWindow1D` contract.
- Changed files: `rewriter/include/mpi/operator_resident/LoopLocalStencilOwnerLoop.h`,
  `rewriter/lib/mpi/operator_resident/LoopLocalStencilOwnerLoop.cpp`,
  `rewriter/lib/Rewriter_MPI.cpp`,
  `dpcppLib/include/mpi/operator_resident/OperatorResidentRuntime.h`, and
  `tests/FOuLa1.0/mpi_expect.txt`.
- Accepted pattern: strict loop-local owner history update with one window
  reader, one direct writer, one invariant replicated scalar payload, matching
  writer-slice writeback, fixed-row or full-history loop-exit materialization,
  and boundary formulas proven replayable from a local time array plus literals,
  constants/globals, allowed math functions, or safe single-return scalar helper
  functions.
- Fallback conditions: unsupported owner-loop shape, wrong slice/writeback,
  variant scalar payload, extra owner mutation, multiple writers, missing
  writeback, noncanonical loop bounds, table/array reads other than the proven
  time array, tensor reads, unsupported/unknown/member/operator calls, mutable
  local scalar dependencies, or any failed owner-loop temporal replay proof.
- Generated-code/log evidence: `FOuLa1.0` logs
  `boundary-bcast=elided local-formula temporal-block=2 accepted owner-loop boundary replay`;
  generated code passes the time array into
  `__dacpp_mpi_or_PDE_pde_0_owner_loop(u_tensor, r, t)`, uses
  `resident_halo_1d_layout_temporal`,
  `scatter_window_1d_temporal`, and
  `exchange_halo_1d_temporal_inplace`, replays `alpha(__or_boundary_time)` and
  `beta(__or_boundary_time)` locally, and contains no left/right boundary
  history `MPI_Bcast`.
  Owner-loop reject fixtures still reject wrong slices, variant scalars, missing
  writeback, extra mutation, multiple writers, scalar payload/shell-arg issues,
  loop-bound mismatch, writer range mismatch, and writeback-index mismatch.
- Verification/profile: `bash clang/tools/translator/test_mpi.sh FOuLa1.0`
  passes. The 4-rank profile probe in
  `/Volumes/QUQ/working/mpi_tmp/profile_segments_next89_probe` has status `ok`
  for `FOuLa1.0`; wall median `0.318110 s`, profile median `0.323944 s`,
  and owner-loop halo calls `1200` for `600` time steps.

### 9. `liuliang1.0`

Current shape:

- Layout: P4.6 `StencilWindow1D` resident halo with boundary-local update.
- 1000 steps.
- Final materialization currently gathers both `new_rho` and `rho`-related
  state in the large benchmark path.

Optimization:

- Apply the 1D temporal blocking path from `MDP1.0`, with explicit support for
  its boundary-local update contract.
- Refine final post-use analysis for resident halo 1D so that benchmark bounded
  or small-root reads gather only needed values instead of performing double
  `MPI_Gatherv`.

Likely files:

- `rewriter/lib/mpi/operator_resident/StencilWindowCodegen.cpp`
- `rewriter/lib/mpi/shared/PostRegionAnalysis.cpp`
- `rewriter/lib/mpi/shared/OutputSyncAnalysis.cpp`
- `dpcppLib/include/mpi/operator_resident/OperatorResidentRuntime.h`

Acceptance:

- Halo calls decrease for accepted blocked 1D LWR shapes.
- Final gather count decreases when post-use is bounded/small.
- `bash clang/tools/translator/test_mpi.sh liuliang1.0 mpiLoopStencilRightBoundaryFullSync1D` passes.
- 32-rank profile reduces `halo` and `gather`.

Fallback:

- Keep current materialization for unsupported host post-use, non-fixed
  boundary reads, uncertain final tensor state, or boundary-local update shapes
  not covered by the temporal block proof.

Implementation Status (2026-05-22):

- Completed `StencilWindow1D` k=2 temporal blocking for the boundary-local replay
  contract and refined 1D resident-halo final materialization for bounded root
  reads.
- Changed files: `rewriter/include/mpi/operator_resident/OperatorResidentPlan.h`,
  `rewriter/lib/mpi/operator_resident/OperatorChainAnalysis.cpp`,
  `rewriter/lib/mpi/operator_resident/LoopLoweredStencil1DCodegen.cpp`,
  `dpcppLib/include/mpi/operator_resident/OperatorResidentRuntime.h`,
  `tests/liuliang1.0/mpi_expect.txt`, `tests/MDP1.0/mpi_expect.txt`,
  `tests/mpiLoopStencilResidentHalo1D/mpi_expect.txt`, and
  `tests/mpiLoopStencilResidentHaloEmptyRank1D/mpi_expect.txt`.
- Accepted pattern: 1D resident halo with window size 2 or 3, followup offset 1,
  one window reader, one direct writer, proven reader extent
  `writer + halo`, canonical zero-based time loop, no direct/scalar/read-cache
  readers, and either no boundary-local update or one replayable left
  boundary-local writer copy from source index 0 to target index 0.
- Final post-use improvement: bounded/small root reads of a 1D resident-halo
  reader are mapped through the followup offset to the owning output rank; only
  the requested values are sent to root and patched with `reviseValue`. Reads
  outside the followup-owned range, unsupported layouts, uncertain post-use,
  dynamic/escaped uses, or non-1D bounded reads conservatively fall back to full
  materialization.
- Generated-code/log evidence: `liuliang1.0` logs
  `final-materialize=bounded/small temporal-block=2 accepted boundary-local replay`
  and `output rho post-use=bounded-index count=1`. Generated code uses
  `resident_halo_1d_layout_temporal`,
  `scatter_window_1d_temporal`, and
  `exchange_halo_1d_temporal_inplace`, replays the left boundary-local update
  inside each inner block step, sends one `__or_bounded_values_rho` value to root
  for `rho[15]`, and omits the previous `MPI_Gatherv`/`array2Tensor` full
  materialization. `mpiLoopStencilRightBoundaryFullSync1D` still falls back to
  `StencilFullSync` because right-boundary updates are outside the accepted
  replay contract.
- Verification/profile: `bash clang/tools/translator/test_mpi.sh liuliang1.0`
  and the targeted 1D/right-boundary fixtures pass. The 4-rank profile probe in
  `/Volumes/QUQ/working/mpi_tmp/profile_segments_next89_probe` has status `ok`
  for `liuliang1.0`; wall median `0.390312 s`, profile median `0.390306 s`,
  halo calls `2000` for `1000` steps, and final sync/materialize are bounded
  single-value paths. The same probe keeps `MDP1.0`, `stencil1.0`, and
  `waveEquation1.0` status `ok` with their k=2 temporal paths still accepted.

### 10. `mandel1.0`

Current shape:

- Layout: `Contiguous1D` OR.
- Input points are scattered once.
- Output flags are not gathered; local counts are reduced with `MPI_Reduce`.
- Main high-rank issue is load imbalance from contiguous static partitioning.

Optimization:

- Add a block-cyclic or cyclic distribution mode for independent `Contiguous1D`
  maps whose output post-use is a scalar reduction or bounded/small root use.
- For Mandelbrot, cyclic distribution balances high-iteration interior points
  across ranks while preserving local scalar reduction.
- Prefer block-cyclic with a modest block size over pure cyclic if memory
  coalescing or contiguous local buffers matter for device performance.

Likely files:

- `rewriter/lib/mpi/operator_resident/PartitionCodegen.cpp`
- `rewriter/lib/mpi/operator_resident/ResidentBufferCodegen.cpp`
- `rewriter/lib/mpi/operator_resident/OperatorChainAnalysis.cpp`
- `dpcppLib/include/mpi/operator_resident/OperatorResidentRuntime.h`

Acceptance:

- Translation log reports block-cyclic distribution only for independent maps
  with safe output sync.
- Generated Mandel path still uses scalar `MPI_Reduce`, not full output gather.
- `bash clang/tools/translator/test_mpi.sh mandel1.0` passes.
- 32-rank profile shows smaller rank imbalance: `kernel max_ms` moves closer to
  `kernel sum_ms / ranks`, and `final_sync max_ms` drops because fewer ranks wait
  for a slow contiguous region.

Fallback:

- Keep contiguous partitioning when output order must be materialized fully,
  downstream resident readers require contiguous ownership, stencil/followup
  semantics depend on neighbors, or bounded reads cannot cheaply find the owner.

Implementation Status (2026-05-22):

- Changed files: `rewriter/include/mpi/operator_resident/OperatorResidentPlan.h`,
  `rewriter/include/Rewriter_MPI_OperatorResident.h`,
  `rewriter/lib/mpi/operator_resident/OperatorChainAnalysis.cpp`,
  `rewriter/lib/mpi/operator_resident/ShellPartitionAnalysis.cpp`,
  `rewriter/lib/mpi/operator_resident/PartitionCodegen.cpp`,
  `rewriter/lib/mpi/operator_resident/CollectiveCodegenUtils.cpp`,
  `rewriter/lib/mpi/operator_resident/ResidentBufferCodegen.cpp`,
  `rewriter/lib/mpi/operator_resident/OperatorResidentCodegen_Internal.h`,
  `dpcppLib/include/mpi/operator_resident/OperatorResidentRuntime.h`, and
  `tests/mandel1.0/mpi_expect.txt`.
- Accepted pattern: generic `Contiguous1D` single independent map with direct
  one-to-one readers/writer, write-only output, no read/write output alias, no
  downstream resident reader, no full materialization contract, non-byte MPI
  element transport, and final scalar count reduction or bounded/small root
  indexed use. The accepted distribution is block-cyclic with block size `64`.
- Fallback conditions: non-`Contiguous1D` layouts, loop-lowered/followup
  contracts, multiple or read/write direct outputs, non-direct parameters,
  bind/order or 1D extent mismatch, direct input/output aliasing, byte-transport
  element types, full output materialization, downstream resident readers, and
  unsupported post-use/reduction.
- Generated-code evidence: `mandel1.0` logs
  `distribution=block-cyclic accepted independent-map scalar-reduction` while
  retaining `post-use-reduction=accepted kind=count-eq-one`. Generated code uses
  `block_cyclic_layout_1d`, `counts_1d_block_cyclic`, and
  `block_cyclic_global_index_1d` to pack owned point blocks by rank, and still
  finishes with scalar `MPI_Reduce`; no `MPI_Gatherv` or `array2Tensor` full
  output materialization is emitted for `mandelbrot_flags`.
- Verification/profile: `cmake --build build --target translator -j8` passed
  after the namespace build fix. Correctness/structure passed for
  `mandel1.0`; regression passed for
  `FOuLa1.0 liuliang1.0 MDP1.0 stencil1.0 waveEquation1.0`; and nearby
  fixtures passed for
  `mpiBroadcastRootOnlyCout mpiBroadcastTensor2Array
  mpiOrReadWriteAccumulate1D mpiOrReadWriteAccumulate2D vectorAddCombo
  gradientSum DFT1.0 matMul1.0`. The 4-rank three-trial profile probe in
  `/Volumes/QUQ/working/mpi_tmp/profile_segments_next10_probe` has status `ok`
  for `mandel1.0`, `MDP1.0`, `stencil1.0`, `waveEquation1.0`, `FOuLa1.0`, and
  `liuliang1.0`. Mandel generated-code structure is block-cyclic, and its
  profiled kernel balance is near-even: trial ratios
  `kernel max_ms / (kernel sum_ms / 4)` were `1.0016`, `1.0107`, and `1.0060`.
  The same probe logs preserved P6-P9 accepted paths including wave/MDP/stencil
  temporal blocking, FOuLa owner-loop boundary replay, and liuliang
  boundary-local temporal blocking with bounded final materialization.

### 11. 2D Spatial Block Partition

Current shape:

- Existing `StencilWindow2D` resident halo and `RowBlock2D` paths partition only
  by contiguous row ranges.
- Temporal blocking `k=2` reduces exchange frequency, but each rank still owns a
  row slab and exchanges only vertical row halos.
- At high rank counts, row slabs can become thin, and communication/locality are
  limited by the row-only decomposition.

Optimization:

- Add a true 2D block or block-cyclic spatial decomposition for eligible 2D
  layouts, separate from temporal blocking.
- Represent rank ownership as a process grid with row and column block ranges.
- Generate halo exchange for north/south/east/west neighbors, with corner/edge
  handling when the stencil or followup contract needs it.
- Keep temporal blocking orthogonal: first support one-step 2D block
  partitioning, then allow widened halos for `k>1` only after the block
  decomposition contract is proven.

Likely files:

- `rewriter/lib/mpi/operator_resident/PartitionCodegen.cpp`
- `rewriter/lib/mpi/operator_resident/StencilWindowCodegen.cpp`
- `rewriter/lib/mpi/operator_resident/OperatorChainAnalysis.cpp`
- `rewriter/include/mpi/operator_resident/OperatorResidentPlan.h`
- `dpcppLib/include/mpi/operator_resident/OperatorResidentRuntime.h`

Acceptance:

- Translation log reports a 2D block partition only when the layout, access
  window, followup/read-cache offsets, boundary updates, and post-use sync can
  be represented by the 2D ownership grid.
- Generated code uses a 2D process-grid layout and exchanges only required
  neighbor halos; it does not fall back to full-row gathers or replicated
  tensors for steady-state stencil steps.
- `stencil1.0` and, after direct-reader recurrence is proven,
  `waveEquation1.0` pass with the 2D block layout.
- 32-rank profile shows lower halo critical-path time or better kernel locality
  than the row-block temporal path for suitable square-ish grids.

Fallback:

- Keep row-block partitioning for unsupported windows, nonlocal followup or
  read-cache offsets, boundary conditions that cannot be mapped to the local
  block, host-visible post-use that requires cheap contiguous rows, direct
  readers whose role rotation is not proven on a 2D block grid, small grids
  where row-block is cheaper, or any downstream resident contract that requires
  contiguous row ownership.

Implementation Status (2026-05-22):

- Implemented and tightened a conservative `distribution=spatial-2d` subset for
  B2 `StencilWindow2D` resident halo loops: static 3x3 stride-1 stencil windows,
  writer-to-reader followup offset `(1,1)`, no direct-reader recurrence, no
  scalar readers, and replayable top/bottom/left/right boundary-local self-copy
  updates. The boundary proof now accepts safe full-span loop spellings such as
  `<= last` and `< last` for existing boundary variants. Eligibility is derived
  from layout, access, loop, boundary, temporal, and post-use facts; no
  benchmark-name checks are used.
- Added spatial `temporal-block=2` for the same conservative B2 subset. The
  accepted log is
  `distribution=spatial-2d accepted stencil halo-width=2 ... temporal-block=2 accepted`.
  Runtime/codegen now exchange a width-2 spatial halo, replay safe boundary
  values after block-start exchanges, compute step 0 over the owned rectangle
  expanded by one cell into an intermediate buffer, and compute step 1 over the
  owned rectangle back into the resident reader buffer with no MPI inside the
  block. Tail blocks with one remaining step use the same spatial layout with a
  single local step.
- Added a conservative profitability fallback for spatial `temporal-block=2`:
  the k=2 spatial path currently requires no host post-use. Bounded or full
  host post-use logs
  `distribution=spatial-2d rejected reason=spatial temporal-block=2 with host post-use is not profitable in the current rectangular buffer path; row-temporal retained`
  and stays on the row-temporal path. This avoids replacing `stencil1.0` large
  bounded-output profiles with the slower spatial k=2 materialization path,
  while allowing no-use contracts such as `mpiLoopStencilCountGuard2D` to prove
  spatial k=2 with no full gather.
- Follow-up profiling confirmed the reason for retaining row-temporal on
  bounded `stencil1.0`: experimentally enabling bounded host post-use for
  spatial k=2 produced correct code with no full output gather and microsecond
  materialization, but the 4-rank large profile in
  `/Volumes/QUQ/working/mpi_tmp/profile_segments_p11_stencil1_spatial_k2_experiment`
  had `dac_wall_median_s=4.838048` and a `kernel` median max of
  `4496.731972 ms`. The profitable current path remains row-temporal k=2
  (`profile_segments_p11_spatial_k2_probe` recorded `dac_wall_median_s=0.864732`
  with no gather segment). A fresh 2026-05-22 confirmation profile in
  `/Volumes/QUQ/working/mpi_tmp/profile_segments_p11_stencil1_spatial_k2_profit`
  records `stencil1.0` at `dac_wall_median_s=0.850681`; its translator log
  reports `distribution=spatial-2d rejected reason=spatial temporal-block=2 with
  host post-use is not profitable in the current rectangular buffer path;
  row-temporal retained`, and generated code uses
  `exchange_halo_2d_rows_temporal_inplace` with no spatial exchange. This keeps
  the fallback based on post-use contract and profitability evidence, not
  benchmark names.
- B3 direct-reader cases such as `waveEquation1.0` still reject spatial-2d with
  `distribution=spatial-2d rejected reason=direct-reader recurrence requires row-layout role rotation`
  and retain the row-temporal `temporal-block=2 accepted direct-reader
  recurrence` path. No spatial role rotation is claimed for direct-reader
  recurrence.
- Runtime/codegen support for the 2D process-grid rectangle layout now covers
  both one-step and k=2 spatial execution:
  `ResidentHalo2DSpatialLayout`, row/column owned ranges, rectangle initial
  scatter, north/south/east/west plus corner halo exchange parameterized by
  halo width, spatial owned slices, root reassembly for full post-use
  materialization, and spatial owner lookup for bounded root reads. Steady-state
  k=2 execution avoids row-temporal helpers and avoids full output
  gather/allgather except when the post-use contract requires full root
  materialization.
- Added `spatialStencil2DOneStep`, a non-`mpi*` one-step 2D stencil fixture that
  is spatial-2d accepted and has bounded/small root post-use. Generated code
  contains `resident_halo_2d_spatial_layout`,
  `scatter_window_2d_spatial`, `exchange_halo_2d_spatial_inplace`, spatial
  owner lookup, two bounded `MPI_Send`/`MPI_Recv` value transfers, and no
  `gather_spatial_owned_to_root`/full `MPI_Gatherv` materialization.
- Tightened post-use scanning for loop-lowered DAC expressions inside helper
  functions. The resident-halo analysis now walks outward through enclosing
  compounds and sees host uses after the outer loop, so fixtures such as
  `mpiMixedStencilORPhaseC` keep full loop-exit materialization for
  `matIn[0].print()` instead of incorrectly logging `post-use=none`.
- Current accepted spatial-2d scope is B2 only: one-step supports bounded/small
  post-use, while k=2 is limited to no-host-use contracts until the rectangular
  spatial k=2 kernel/layout path is made profitable for bounded/full host
  post-use. Unsupported or less
  profitable cases log structural rejection reasons: direct-reader recurrence,
  scalar readers, unsupported/profitability-rejected post-use, unsafe boundary
  replay, missing resident halo contract, or unsupported temporal block size.
  The next performance step is direct-reader spatial role rotation and cheaper
  bounded/full spatial k=2 post-use and its kernel/layout cost; until that is proven,
  row-temporal/block-cyclic/bounded-materialize/resident continuation paths are
  retained.
- Verified with translator build, required `stencil1.0`, the P6-P11 protection
  set (`mandel1.0`, `FOuLa1.0`, `liuliang1.0`, `MDP1.0`, `waveEquation1.0`),
  related non-`mpi*` 2D/access/stencil fixtures found by `rg`
  (`spatialStencil2DOneStep`, `jacobi1.0`, `imageAdjustment1.0`,
  `gradientSum`, `matMul1.0`), and nearby structural/rejection fixtures. The
  requested 4-rank, 3-trial profile probe passed all six cases in
  `/Volumes/QUQ/working/mpi_tmp/profile_segments_p11_stencil1_spatial_k2_profit`;
  `stencil1.0` logs confirm the profitability fallback and row-temporal `k=2`
  retention (`dac_wall_median_s=0.850681`), `mpiLoopStencilCountGuard2D` proves
  no-use spatial-2d `halo-width=2 temporal-block=2` without full gather,
  `waveEquation1.0` confirms direct-reader row-temporal `k=2` retention, and
  `mandel1.0` remains block-cyclic with scalar `MPI_Reduce`.

## 32-Rank Benchmark Commands

Use a fresh artifact directory for every optimization batch:

```bash
MPI_PROFILE_BENCH_TIMEOUT_SECONDS=3600 \
python3 clang/tools/translator/bench_mpi_profile_segments.py \
  --tmp-dir /Volumes/QUQ/working/mpi_tmp/profile_segments_32rank_<label> \
  --ranks 32 \
  --trials 3 \
  matMul1.0 DFT1.0 imageAdjustment1.0 decay1.0 stencil1.0 \
  waveEquation1.0 MDP1.0 FOuLa1.0 liuliang1.0 mandel1.0
```

For quicker inner-loop checks, run one case with one trial:

```bash
MPI_PROFILE_BENCH_TIMEOUT_SECONDS=3600 \
python3 clang/tools/translator/bench_mpi_profile_segments.py \
  --tmp-dir /Volumes/QUQ/working/mpi_tmp/profile_segments_32rank_probe \
  --ranks 32 \
  --trials 1 \
  <case>
```

Correctness smoke before every profile run:

```bash
cmake --build build --target translator -j8
bash clang/tools/translator/test_mpi.sh <changed-cases-and-nearby-fixtures>
```

Suggested nearby fixtures:

| Optimization area | Targeted tests |
|---|---|
| RowPartitionFullRow / matmul | `matMul1.0 jacobi1.0 gradientSum` |
| Replicated/index-local init | `DFT1.0 vectorAddCombo mpiBroadcastTensor2Array` |
| RowBlock2D chain | `imageAdjustment1.0 mpiOrReadWriteAccumulate2D` |
| P4.5 direct loop lowering | `decay1.0 mpiLoopInvariantReader1D` if present |
| 1D resident halo | `MDP1.0 liuliang1.0 mpiLoopStencilResidentHalo1D mpiLoopStencilRightBoundaryFullSync1D` |
| 2D resident halo | `stencil1.0 waveEquation1.0 mpiLoopStencilCountGuard2D` |
| Owner-loop | `FOuLa1.0 mpiOwnerLoop*` |
| Contiguous block-cyclic | `mandel1.0 mpiOrReadWriteAccumulate1D` |

## Completion Criteria

For each accepted optimization:

- Translation log clearly identifies the accepted fast path and fallback reason
  when rejected.
- Generated code contains the expected structural change.
- Targeted `test_mpi.sh` cases pass.
- 32-rank profile artifacts include `summary.tsv`, `profile_raw.tsv`,
  `profile_summary.tsv`, generated source snapshots, and git metadata.
- The case's 32-rank profile-off DAC wall time improves or stays within noise
  while no correctness or fallback regression appears in nearby cases.
