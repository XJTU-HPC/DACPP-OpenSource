# MPI 32-Node Top-10 Optimization Plan

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
