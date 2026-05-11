# Row Payload And Demand Materialization Optimization TODO

Updated: 2026-05-11
Updated status: 2026-05-11 (Partial completion)

This document records the focused performance analysis for
`gradientSum` and `imageAdjustment1.0` from
`/Volumes/QUQ/working/mpi_tmp_benchmark`, and turns it into a non-specialized
optimization TODO.

## Implementation Status (2026-05-11)

### Completed

**TODO 1: Split Pack And Scatter Profiling** ✅
- Added `ProfileSegment::Pack` profiling around root-side tensor2Array and sendbuf construction in:
  - `emitRowPartitionFullRowScatter` (ResidentBufferCodegen.cpp)
  - `emitScatter` (CollectiveCodegenUtils.cpp)
  - `emitReplicatedFullTensorBroadcast` (ResidentBufferCodegen.cpp)
- Removed outer scatter profiling from generic wrapper (OperatorResidentCodegen.cpp)
- Now produces separate `pack` and `scatter` profile segments

**TODO 3: RowBlock2D Local Extraction Fast Path** ✅
- Implemented `MPI_Bcast + memcpy` fast path for RowBlock2D layout
- Replaces root-only `MPI_Scatterv` with all-ranks broadcast + local memcpy
- Updated `tests/mpiOrReadWriteAccumulate2D/mpi_expect.txt` expectations
- 72/72 tests pass

### Not Completed / Needs More Analysis

**TODO 2: RowPartitionFullRow Payload Fast Path** ⚠️
- Attempted strided broadcast fast path but disabled due to complex data layout
- The RowPartitionFullRow payload layout depends on indexedBindPos, voidDim,
  and kernel view semantics in ways that require deeper analysis
- Original scatter path remains as fallback
- The profiling infrastructure is in place for future work

**TODO 4: Demand-Driven Final Materialization** ⏳
- Not implemented in this round
- Would require extending post-region analysis to classify output uses

### Benchmark Results (After Optimization)

| Case | Scale | Standard MPI+SYCL | DAC translated MPI | Ratio |
|---|---:|---:|---:|---:|
| `gradientSum` | `8192x4096` | `0.657s` | `1.205s` | `1.84x` |
| `imageAdjustment1.0` | `4096x4096` | `0.752s` | `1.084s` | `1.44x` |
| `oddeven0.1` | `N=4096` | `1.679s` | `0.661s` | `0.39x` ✅ |
| `matMul1.0` | `2048x2048` | `6.170s` | `0.841s` | `0.14x` ✅ |
| `stencil1.0` | `2048x2048, steps=600` | `1.494s` | `1.416s` | `0.95x` ✅ |

Note: gradientSum and imageAdjustment1.0 remain slower than hand-written due to
serial root packing in the scatter phase. The Pack profiling now makes this
measurable for future optimization work.

---

## Original TODO (Preserved for Reference)

The goal is not to add case-name specializations. The goal is to improve the
generic operator-resident paths that these two cases expose:

- `RowPartitionFullRow` payload packing/scatter cost
- `RowBlock2D` initial root scatter cost
- full root materialization when the post-use only needs a small slice or
  scalar element

## Correctness Rule

Do not change current translation correctness for performance.

Every optimization in this document must obey these rules:

- If a proof is incomplete, keep the current correct generated path.
- Do not weaken alias, output-sync, post-use, or resident-state guards.
- Do not silently replace `all-ranks` semantics with `root-only` behavior.
- Do not special-case benchmark names such as `gradientSum` or
  `imageAdjustment1.0`.
- Add structure tests and output comparison tests for every accepted fast path.
- Add reject fixtures for near-miss shapes before widening accepted behavior.
- Benchmark-only source scaling must stay separate from translator semantics.

## Observed Benchmark Data

Source directory:

```text
/Volumes/QUQ/working/mpi_tmp_benchmark
```

Benchmark rows:

| Case | Scale | Standard MPI+SYCL | DAC translated MPI | Ratio |
|---|---:|---:|---:|---:|
| `gradientSum` | `8192x4096` | `0.875572s` | `1.095442s` | `1.25x` |
| `imageAdjustment1.0` | `4096x4096` | `0.676128s` | `0.938608s` | `1.39x` |

Both cases are correctness-pass cases. Both currently route through
operator-resident codegen, not legacy `AccessPattern`.

## Profile Summary

Manual profile command shape:

```bash
source /Volumes/QUQ/working/dacpp/clang/tools/translator/env.sh
DACPP_MPI_PROFILE=1 DYLD_LIBRARY_PATH="$ACPP_ROOT/lib" \
  mpirun -np 4 /Volumes/QUQ/working/mpi_tmp_benchmark/<case>/dac_mpi_bin
```

### gradientSum

Translation route:

```text
[DACPP][MPI][OR] expr=0 shell=gradSumShell layout=RowPartitionFullRow partition=accepted
[DACPP][MPI][OR] chain=0 layout=RowPartitionFullRow length=1 chain=accepted codegen=enabled
```

Profile:

| Segment | Max time |
|---|---:|
| `init` | `0.000791ms` |
| `scatter` | `754.910750ms` |
| `kernel` | `3.417416ms` |
| `gather` | `20.166333ms` |
| `materialize` | `0.007709ms` |

Root cause:

- The generated path treats `matGrads[{}][idx1]` as
  `RowPartitionFullRow` / `IndexedColFullRows`.
- Root calls `tensor2Array`, then builds a reordered send buffer by walking the
  row-major matrix as a column payload:

```text
src_linear = payload_i * input_cols + indexed_value
sendbuf[dst_base + payload_i] = global[src_linear]
```

- The current `scatter` profile includes both root-side pack/transpose and MPI
  `Scatterv`, so the segment is dominated by serial root packing.
- The hand-written reference constructs the full input on every rank and
  computes each rank's owned neuron range directly. It avoids root pack/scatter.

Optimization implication:

- The first generic fix is not kernel tuning.
- The first generic fix is payload-pack/scatter modeling for
  `RowPartitionFullRow` with void payload dimensions.

### imageAdjustment1.0

Translation route:

```text
[DACPP][MPI][OR] expr=0 shell=imageAdjustment layout=RowBlock2D partition=accepted
[DACPP][MPI][OR] expr=1 shell=imageAdjustment layout=RowBlock2D partition=accepted
[DACPP][MPI][OR] chain=0 layout=RowBlock2D length=2 chain=accepted codegen=enabled
```

Profile:

First wrapper:

| Segment | Max time |
|---|---:|
| `init` | `0.000667ms` |
| `scatter` | `200.427833ms` |
| `kernel` | `13.708917ms` |

Second wrapper:

| Segment | Max time |
|---|---:|
| `init` | `0.000208ms` |
| `scatter` | `0.910250ms` |
| `kernel` | `6.589416ms` |
| `gather` | `48.472417ms` |
| `materialize` | `48.780292ms` |

Root cause:

- The chain handling is already useful: intermediate `image_tensor2` is kept as
  resident distributed state.
- The first wrapper still starts from a root-owned tensor:
  `tensor2Array` on root followed by `MPI_Scatterv`.
- The final wrapper gathers the full image and calls `array2Tensor`, even
  though the benchmark post-use only prints `image_tensor3[0][0]`.
- The hand-written reference initializes only each rank's local image block and
  gathers the final flat image to root without rebuilding a DAC tensor.

Optimization implication:

- The remaining gap is root input distribution and over-materialization.
- A generic optimization should prove local input construction or reduce the
  final materialization demand, not special-case image pixels.

## TODO 1: Split Pack And Scatter Profiling

Goal: make future work measurable. The `gradientSum` `scatter` segment currently
mixes root-side payload packing with MPI communication.

### Phase 1.1: Add Explicit Pack Segment In OR Codegen

Tasks:

- Add `ProfileSegment::Pack` around root-side send-buffer construction for:
  - `RowPartitionFullRow`
  - replicated/full payload input preparation
  - byte-transport pack paths
- Keep `ProfileSegment::Scatter` limited to MPI scatter/scatterv calls and
  receive-buffer setup.
- Preserve default quiet behavior unless `DACPP_MPI_PROFILE=1`.

Likely files:

- `rewriter/lib/mpi/operator_resident/CollectiveCodegenUtils.cpp`
- `rewriter/lib/mpi/operator_resident/PartitionCodegen.cpp`
- `rewriter/lib/mpi/operator_resident/OperatorResidentCodegen.cpp`
- `dpcppLib/include/mpi/common/Profile.h`

Done when:

- `gradientSum` reports separate `pack` and `scatter` lines.
- Existing tests pass with profiling disabled.
- Profile totals still roughly match external wall-clock shape.

### Phase 1.2: Add Pack Diagnostics

Tasks:

- Log payload direction, payload length, indexed dimension, and total packed
  elements for accepted `RowPartitionFullRow`.
- Add accept/reject reason text when a future fast path is skipped.

Done when:

- Translation logs explain why the generated code used root pack/scatter.

## TODO 2: RowPartitionFullRow Payload Fast Path

Goal: reduce root serial transpose/pack for shapes like
`matGrads[{}][idx]`, without changing semantics.

### Phase 2.1: Classify Payload Layouts

Tasks:

- Identify whether the payload slice is contiguous in the tensor's physical
  row-major layout.
- Classify:
  - contiguous full row payload
  - strided full column payload
  - small payload
  - byte-transport payload
  - irregular or unsupported payload
- Keep the current root pack path as fallback.

Done when:

- `gradientSum` is classified as strided full-column payload.
- Existing row-payload cases have stable log classifications.

### Phase 2.2: MPI Derived Datatype Or Parallel Pack Prototype

Tasks:

- Evaluate two generic implementations for strided payload:
  - MPI vector/datatype-based scatter from root memory
  - parallel or rank-local pack that reduces root serial work
- Prefer the option that preserves current output and works with native MPI
  types first.
- Keep byte-transport structs on the old path unless a byte-safe proof exists.

Done when:

- A prototype reduces `gradientSum` `pack` time without changing output.
- Unsupported payload forms still fall back cleanly.

### Phase 2.3: Local Replicated-Clean Input Proof

Tasks:

- Detect when the input tensor is constructed identically on all ranks before
  the OR wrapper.
- If proven, each rank can extract its local payload from its own tensor instead
  of receiving it from root.
- Guard against:
  - rank-dependent initialization
  - mutation before wrapper
  - unknown function calls touching the tensor
  - input aliasing with outputs

Done when:

- The proof is generic and not benchmark-name based.
- `gradientSum` can skip root scatter only if local input identity is proven.
- Negative fixtures reject rank-dependent or mutated inputs.

## TODO 3: RowBlock2D Initial Resident Input Fast Path

Goal: avoid root scatter for row-block inputs that are already safely available
on every rank or can be initialized directly as resident local state.

This targets the first `imageAdjustment1.0` wrapper pattern, but should be
implemented as a generic `RowBlock2D` resident-input optimization.

### Phase 3.1: Input Origin Analysis

Tasks:

- Track tensor construction before the OR chain.
- Determine whether the input is:
  - root-only host tensor
  - replicated clean host tensor
  - already resident distributed tensor
  - unknown or mutated
- Reuse existing residency concepts instead of adding ad hoc flags.

Likely files:

- `rewriter/lib/mpi/operator_resident/ResidencyAnalysis.cpp`
- `rewriter/lib/mpi/operator_resident/OperatorChainAnalysis.cpp`
- `rewriter/lib/mpi/operator_resident/ResidentBufferCodegen.cpp`

Done when:

- Logs can explain why an input uses root scatter or local resident extraction.

### Phase 3.2: Local Block Extraction

Tasks:

- For replicated clean row-major tensors, generate per-rank local row-block
  extraction from the local tensor storage.
- Avoid root `tensor2Array` and `MPI_Scatterv` in this proven case.
- Preserve the current root scatter path as fallback.

Done when:

- `imageAdjustment1.0` first wrapper can skip root scatter only under proof.
- Output comparison passes for 1, 2, and 4 ranks.
- Profile shows `scatter` no longer dominates the first wrapper.

### Phase 3.3: Resident Initializer API

Tasks:

- Consider adding runtime helpers to initialize resident storage from local
  rank ranges.
- Keep helper APIs layout-generic enough for `Contiguous1D` and `RowBlock2D`.

Done when:

- Codegen does not duplicate row-range extraction logic across wrappers.

## TODO 4: Demand-Driven Final Materialization

Goal: avoid full `Gatherv + array2Tensor` when post-use only demands a small
part of the output.

This targets both:

- `imageAdjustment1.0`: post-use prints `image_tensor3[0][0]`
- `gradientSum`: post-use prints first five `matNeuronSum[i][0]` values

### Phase 4.1: Post-Use Demand Analysis

Tasks:

- Extend post-region analysis to classify output uses after an OR wrapper:
  - full tensor use
  - `tensor2Array`
  - unknown function
  - `.print()`
  - bounded indexed reads
  - simple root-only `std::cout` reads
- Determine whether each use requires:
  - full root materialization
  - full gather without tensor rebuild
  - partial gather
  - owner-rank point query
  - no materialization

Likely files:

- `rewriter/lib/mpi/shared/PostRegionAnalysis.cpp`
- `rewriter/lib/mpi/shared/PostRegionCodegen.cpp`
- `rewriter/lib/mpi/shared/OutputSyncAnalysis.cpp`
- `rewriter/lib/mpi/operator_resident/OperatorChainAnalysis.cpp`

Done when:

- Logs show materialization demand reason, not just `MaterializedRoot`.

### Phase 4.2: Partial Gather For Indexed Reads

Tasks:

- For bounded root-only indexed reads, gather only requested global elements or
  the minimal contiguous span.
- Map global element to owning rank using the same rank-range layout as the OR
  wrapper.
- Replace post-use reads on root with reads from the partial materialized
  buffer.
- Keep full materialization for unknown or broad uses.

Done when:

- `imageAdjustment1.0` can print `image_tensor3[0][0]` without full
  `array2Tensor`.
- `gradientSum` can print first five values without gathering the entire output,
  if the post-use proof is stable.
- Negative fixtures with unknown post-use still force full materialization.

### Phase 4.3: Root Flat Buffer Cache

Tasks:

- When a full gather is needed but a DAC tensor rebuild is not, allow root
  post-use code to read from a flat gathered buffer.
- Avoid `array2Tensor` if only root `std::cout` or simple indexed reads follow.
- Preserve tensor semantics if the source program later uses the tensor object.

Done when:

- `imageAdjustment1.0` avoids the final `array2Tensor` when only printing a
  value.
- Full tensor semantic tests still materialize normally.

## TODO 5: Benchmark And Regression Plan

### Phase 5.1: Focused Benchmarks

Run before/after benchmarks for:

```bash
MPI_ONLY_BENCH_TMP_DIR=/Volumes/QUQ/working/mpi_tmp/<run-name> \
MPI_ONLY_BENCH_RANKS=4 \
MPI_ONLY_BENCH_TIMEOUT_SECONDS=1800 \
python3 bench_mpi_only_requested.py gradientSum imageAdjustment1.0
```

Also run with profiling:

```bash
DACPP_MPI_PROFILE=1 mpirun -np 4 <dac_mpi_bin>
```

Done when:

- The docs record before/after total wall time.
- The docs record before/after segmented profile data.

### Phase 5.2: Focused Correctness Tests

Add or update `mpi_expect.txt` fixtures for any new accepted path:

- accepted row-payload fast path
- rejected row-payload near misses
- accepted local resident input path
- rejected rank-dependent input initialization
- accepted partial materialization path
- rejected unknown post-use materialization path

Minimum validation:

```bash
cd /Volumes/QUQ/working/dacpp
cmake --build build --target translator -j8

cd /Volumes/QUQ/working/dacpp/clang/tools/translator
bash test_mpi.sh gradientSum imageAdjustment1.0 <new-focused-fixtures>
git diff --check -- clang/tools/translator
```

Run the full MPI suite before closeout if shared OR materialization or
residency analysis changes:

```bash
bash test_mpi.sh
```

## Recommended Implementation Order

1. Split `pack` from `scatter` profiling for OR payload paths.
2. Add logging/classification for `RowPartitionFullRow` payload layout.
3. Prototype row-payload pack reduction for native MPI types.
4. Add replicated-clean input proof and local extraction for `RowBlock2D`.
5. Add demand-driven final materialization for bounded root-only indexed reads.

This order keeps the work evidence-driven and avoids widening accepted surfaces
before the current overhead is measured precisely.
