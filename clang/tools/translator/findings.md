# MPI Translator Findings

Updated: 2026-05-11

## P4 FOuLa Owner-Loop Inventory

Current implementation location before P4 edits:

- `rewriter/lib/Rewriter_MPI.cpp`
  - local `FoulaOwnerLoopSpecialization`
  - local `detectFoulaOwnerLoopSpecialization(...)`
  - local `buildFoulaOwnerLoopSpecializationCode(...)`
  - `rewriteMPI()` collects candidates, emits the generated owner-loop
    function, and replaces the whole outer source loop with
    `owner_loop(owner, scalarExpr);`

Current accepted proof dependencies:

- Plan-level guards:
  - expression must be operator-resident `StencilWindow1D`
  - calc must have exactly 3 params and plan must have exactly 3 params
  - exactly one read-only `StencilWindow` reader
  - exactly one WRITE-only `OutputDirect` writer
  - exactly one read-only `ReplicatedScalar`
  - reader/writer/scalar calc element types must match
  - element type must not require byte transport
- AST-derived guards:
  - enclosing outer statement is a `ForStmt`
  - loop init is a single `VarDecl`, used as the induction-variable name
  - loop body is a `CompoundStmt`
  - loop body has exactly seven top-level statements
- Source-text/regex guards:
  - reader construction matches `reader = owner[{}][k]`
  - writer construction matches `writer = owner[{1,...}][k+1]`
  - post-writeback matches `owner[...][k+1] = writer[...]`
  - scalar shell argument is constructed from a loop-local `std::vector`
    `push_back(payload)` followed by `dacpp::Vector<T> scalar(vectorName)`
  - current scalar payload regex accepts only one token-like expression with no
    parentheses/spaces/semicolon

Current generated owner-loop behavior:

- Function signature:
  `void __dacpp_mpi_or_<shell>_<calc>_<expr>_owner_loop(dacpp::Matrix<T>& owner, T scalar)`
- Scatters owner column 0 once with `scatter_window_1d`.
- Computes local owned rows into resident next state.
- Broadcasts two physical boundary scalar values per step from root.
- Exchanges halo in place with `exchange_halo_1d_inplace`.
- Stores local owned history for all columns.
- Gathers owned history once with `MPI_Gatherv`.
- Materializes owner rows `1..rows-2` on root at loop exit.
- Reports segmented profile for init/scatter/kernel/bcast/halo/gather/materialize.

Lowering-contract semantics for the current P4 shape:

- Source statement replacement: replace the whole owner `for` loop, not only the
  DAC expression.
- Source statements absorbed/removed by replacement:
  - writer slice construction
  - scalar temporary vector declaration
  - scalar payload `push_back`
  - scalar DAC vector construction
  - reader slice construction
  - DAC expression statement
  - post-DAC owner writeback loop
- Preserved semantics:
  - owner matrix remains the host-visible materialized object after loop exit
  - boundary row values for each next time step are read from the owner matrix
    on root and broadcast as scalars
  - no full-column per-step reader broadcast/gather is emitted
- Resident state:
  - owner-derived current stencil column
  - next stencil column
  - local owned history
- Materialization:
  - loop-exit gather of owned history and root writeback into owner matrix
- Guards:
  - compile-time fallback for unsupported shape/proof mismatch
  - runtime abort/count guards for MPI count narrowing

Current reject/fallback behavior:

- If owner-loop detection fails, the expression remains on the already selected
  operator-resident plan. For FOuLa-like `StencilWindow1D` loop-local slices
  this normally means ordinary `StencilWindow1D` wrapper code plus the existing
  source statements, not a new legacy fast path.
- Therefore P4 reject fixtures should assert owner-loop rejection and absence
  of owner-loop generated code, while keeping `AccessPattern`/`PackPlan` absent
  where the existing OR fallback remains selected.

P4 decision so far:

- First implementation slice should be behavior-preserving extraction and
  contract logging. Do not broaden accepted syntax or parameterize new shapes
  until AST proof replaces the current source-text dependency.
- Reject fixture note: a missing-writeback program that removes the
  writer-to-owner followup entirely falls out of `StencilWindow1D` OR and goes
  legacy. The P4 missing-writeback fixture therefore keeps a recognizable
  writer-to-owner followup but writes to `owner[*][k]`, so it remains OR-current
  while rejecting the owner-loop contract for missing `owner[*][k+1]`
  writeback.

## P4.x AST/Contract Follow-up

Implemented on 2026-05-11 as proof/refactor only. No accepted owner-loop shape
was added.

`LoopLocalStencilOwnerLoop.cpp` now builds a local AST-backed body proof before
accepting the strict current shape:

- enclosing loop must be a `for` with single declared induction variable,
  initial value `0`, `<`/`<=` condition on that variable, and unit increment
- the loop body must still have exactly the current seven top-level statements
- the DAC expression must be the sixth top-level statement
- reader slice statement is AST-checked as `reader = owner[{}][k]`
- writer slice statement is AST-checked as `writer = owner[{1,...}][k+1]`
- the scalar argument must use an empty loop-local `std::vector`, one
  `push_back(payload)`, and a single-argument `dacpp::Vector` construction from
  that storage
- scalar payload is still source-extracted and kept under the existing strict
  token-like rule, with loop-variable references rejected
- post-writeback must be a forward unit loop from `1` and contain exactly one
  assignment shaped like `owner[i][k+1] = writer[i-1]`
- before enforcing the seven-statement accepted shape, the detector scans the
  top-level body for multiple owner slices at `k+1` and multiple owner
  next-step assignments so existing reject reasons remain stable

Additional P4.x tightening on 2026-05-11 found and closed accidental
acceptance gaps without adding accepted syntax:

- reader/writer slice proof now checks the vector initializer itself rather
  than accepting any matching nested slice expression
- owner matrix shape must be proven from the owner declaration as
  `{interior+1,time+1}` and the outer loop bound must be tied to that time
  dimension
- writer range must end at the proven owner interior bound, not merely start at
  `1`
- scalar shell argument must source-text to the same loop-local vector name and
  AST-resolve through only C++ copy-construction wrappers to that variable
- scalar payload must still satisfy the old strict source-text rule and now
  also be a simple AST literal or decl-ref token
- writeback loop bound must match the proven owner interior bound, and the loop
  body must be the exact assignment `owner[i][k+1] = writer[i-1]`

Source-text/regex dependency is now limited to scalar payload extraction and
the existing conservative payload narrowness check. There is no fallback that
accepts a shape when the AST proof fails.

Added local contract consistency diagnostics:

- accepted contract must replace the outer loop
- absorbed roles must include writer slice, scalar vector storage, scalar
  payload, scalar shell arg, reader slice, DAC expression, and owner writeback
- resident tensor count must remain 3
- owner materialization must be loop-exit
- compile-time and runtime guards must both be present with reason text

Accepted `FOuLa1.0` now logs:
`contract-check=pass reason=owner-loop-contract-consistent`.

Focused validation after the AST proof change:

- `cmake --build build --target translator -j8`: passed.
- `bash test_mpi.sh FOuLa1.0 mpiOwnerLoopWrongSliceReject1D
  mpiOwnerLoopVariantScalarReject1D mpiOwnerLoopMissingWritebackReject1D
  mpiOwnerLoopExtraMutationReject1D mpiOwnerLoopMultipleWriterReject1D
  mpiOwnerLoopScalarPayloadExprReject1D
  mpiOwnerLoopScalarShellArgReject1D mpiOwnerLoopLoopBoundReject1D
  mpiOwnerLoopWriterRangeReject1D mpiOwnerLoopWritebackIndexReject1D
  MDP1.0 liuliang1.0 stencil1.0 waveEquation1.0
  mpiOrStencilRefreshPolicy1D`: passed `16 tests | 16 passed | 0 failed | 0
  skipped`.

Final validation:

- Full MPI suite passed:
  previous run passed `67 tests | 67 passed | 0 failed | 0 skipped`; after
  adding the stricter owner-loop reject coverage, the full suite passed
  `72 tests | 72 passed | 0 failed | 0 skipped`.
- Final whitespace check passed:
  `git diff --check -- clang/tools/translator`.

No benchmark was run for this P4.x slice because the work only changes proof
and contract diagnostics. It does not intentionally change the accepted surface
or generated communication pattern.

## Phase 2.1 Legacy Inventory

This inventory is documentation-only and does not change translator behavior.

Initial constraints from handoff:

- Phase 1.1-1.4 segmented profiling is complete and default-off behind
  `DACPP_MPI_PROFILE=1`.
- `vectorAddCombo` is current OR `Contiguous1D` chain behavior, not a legacy
  AccessPattern starting point unless fresh evidence proves otherwise.
- Phase 1.4 residual profile for `vectorAddCombo` points at
  scatter/materialize/bcast, not legacy pack.
- Phase 1.4 residual profile for `decay1.0` shows per-step gather+bcast and
  belongs to later materialization/residency work, not Phase 2.1 fast-path
  implementation.

Inventory artifact target:
`/Volumes/QUQ/working/mpi_tmp/phase2_1_legacy_inventory`.

Commands and artifact contents:

- Current translator build passed before inventory.
- Focused correctness/structure smoke passed for:
  `vectorAddCombo DFT1.0 gradientSum mandel1.0
  mpiFixedBlockMatrixSingleSplitReject1D mpiFixedBlockOverlapReject1D
  mpiFixedBlockPayloadReject1D mpiOrReadWriteAccumulate1D
  mpiOrReadWriteAccumulate2D`.
- The inventory artifact contains 63 current translation records:
  `translation_runs.tsv`, `marker_scan.tsv`, per-case `step2.log`, and
  generated-code snapshots.

Key readout:

- All 14 benchmark application cases translated successfully.
- None of the 14 benchmark generated-code snapshots contain `AccessPattern` or
  `PackPlan`.
- `vectorAddCombo` is confirmed current OR: three-expression `Contiguous1D`
  chain with a replicated scalar bias on `VSHIFT`; generated code has
  `__dacpp_mpi_or_` wrappers and no `AccessPattern`/`PackPlan`.
- Remaining true legacy wrapper evidence is limited to three focused FixedBlock
  reject fixtures.
- Remaining `AccessPattern`+`PackPlan` evidence outside those wrappers is
  Phase-C stencil fallback infrastructure for focused stencil fixtures, not the
  simple contiguous 1D legacy class targeted by Phase 2.

### Benchmark Cases

| Case | Source expression/shell/calc | Current path | Legacy evidence | Shape classification | Possible fast-path candidate | Reject/fallback reason |
|---|---|---|---|---|---|---|
| `DFT1.0` | `DFT(input_tensor, output_tensor, vec_tensor) <-> dft` | OR `ReplicatedFullTensor`, two accepted single-expression chains | none: no `AccessPattern`, no `PackPlan` | replicated full-tensor reader plus direct 1D output/input | no | Current OR accepted; not legacy. |
| `FOuLa1.0` | `PDE(u_kin, u_kout, r) <-> pde` inside owner loop | OR `StencilWindow1D` plus FOuLa owner-loop rewrite | none | resident 1D stencil plus replicated scalar | no | Current owner-loop specialization accepted; not legacy. |
| `MDP1.0` | `mdp_shell(p, new_p) <-> mdp` | OR P4.6 `StencilResidentHalo` | none | resident 1D stencil | no | Current resident halo accepted; Phase-C route logs are analysis only. |
| `decay1.0` | `DECAY(N0s, lambdas, local_A, t) <-> decay` in `while` | OR `Contiguous1D`, loop-lowered materialize per run | none | simple contiguous 1D plus replicated scalar | no for Phase 2 | Current OR accepted; residual cost belongs to Phase 3 materialization/residency. |
| `gradientSum` | `gradSumShell(matGrads, matNeuronSum) <-> gradSum` | OR `RowPartitionFullRow` | none | row-contiguous 2D/full-row payload | no | Current OR accepted; not legacy. |
| `imageAdjustment1.0` | two `imageAdjustment(...) <-> adjust` expressions | OR `RowBlock2D` chain length 2 | none | row-contiguous 2D direct map | no | Current OR accepted; not legacy. |
| `jacobi1.0` | `jacobiShell(A, b, x, x_new, nums) <-> jacobi` | OR `RowPartitionFullRow` with replicated full tensor reader | none | row-contiguous 2D plus replicated full tensor | no | Current OR accepted; not legacy. |
| `liuliang1.0` | `LWR_shell(rho, new_rho) <-> lwr` | OR P4.6 `StencilResidentHalo` | none | resident 1D stencil | no | Current resident halo accepted; Phase-C route logs are analysis only. |
| `mandel1.0` | `MANDEL(complex_points_tensor, mandelbrot_flags_tensor) <-> mandel` | OR `Contiguous1D` | none | simple contiguous 1D | no | Current OR accepted; not legacy. |
| `matMul1.0` | `matrixMultiply_shell(matA, matB, matC) <-> matrixMultiply_calc` | OR `RowPartitionFullRow` | none | row-contiguous 2D/full-row payload | no | Current OR accepted; not legacy. |
| `oddeven0.1` | two `ODDEVEN(...) <-> oddeven` expressions | OR `FixedBlock` plus P5 phase-exchange accepted | none | fixed-block 1D phase exchange | no | Current P5 accepted; not legacy. |
| `stencil1.0` | `stencilShell(matIn, matOut) <-> stencil` | OR P4.6 `StencilResidentHalo` | none | resident 2D stencil | no | Current resident halo accepted; Phase-C route logs are analysis only. |
| `vectorAddCombo` | `VADD; VSHIFT; VADD` chain | OR `Contiguous1D` chain length 3 | none: generated code has `__dacpp_mpi_or_`, no `AccessPattern`, no `PackPlan` | simple contiguous 1D plus replicated scalar bias | no for Phase 2 | Current OR accepted. Do not select as legacy candidate without new evidence. |
| `waveEquation1.0` | `waveEqShell(matCur, matPrev, matNext) <-> waveEq` | OR P4.6 `StencilResidentHalo` with direct reader/read-cache extension | none | resident 2D stencil plus direct reader | no | Current resident halo accepted; Phase-C route logs are analysis only. |

### Focused Legacy And Fallback Fixtures

| Case | Source expression/shell/calc | Current path | Legacy evidence | Shape classification | Possible fast-path candidate | Reject/fallback reason |
|---|---|---|---|---|---|---|
| `mpiFixedBlockMatrixSingleSplitReject1D` | `MATRIX_SINGLE_SPLIT(input, output) <-> copyMatrixRows` | OR rejects, legacy wrapper emits | `AccessPattern`; legacy wrapper `MATRIX_SINGLE_SPLIT_copyMatrixRows` | row-contiguous 2D / matrix FixedBlock first slice | maybe later, not first | Reject reason: fixed block first slice requires 1D wrapper tensors. Semantics are matrix row blocks, not simple contiguous 1D. |
| `mpiFixedBlockOverlapReject1D` | `OVERLAP(input, output) <-> copy3` with `split(3,2)` | OR rejects, legacy wrapper emits | `AccessPattern`; legacy wrapper `OVERLAP_copy3` | irregular/overlapping 1D fixed block | no | Reject reason: fixed block requires split size to equal split stride. Overlap makes direct contiguous fast path unsafe. |
| `mpiFixedBlockPayloadReject1D` | `PAYLOAD(input, output) <-> copyPayload` with matrix payload | OR rejects, legacy wrapper emits | `AccessPattern`; legacy wrapper `PAYLOAD_copyPayload` | row-contiguous 2D / payload FixedBlock | maybe later, not first | Reject reason: fixed block first slice requires one split per parameter. Payload dimensions are outside current simple 1D target. |
| `mpiLoopStencilScalarReject2D` | `scalarStencilShell(matIn, matOut, gain) <-> scalarStencil` | OR rejects, Phase-C/fallback code emits | `AccessPattern` + `PackPlan` in Phase-C context | 2D stencil plus scalar/replicated input | no for Phase 2 | Reject reason: unsupported bind rank for OR and Phase-C mixed Vector/Matrix limitation. This is stencil-surface work, not simple contiguous legacy. |
| `mpiDenseCoverSibling1.0` | `denseCoverShell(state, updates) <-> denseCoverStep` | Phase-C fallback | `AccessPattern` + `PackPlan` | irregular/dense row-band stencil-like payload | no | Uses dense row-band `split(3,2)` and full-row payload; belongs to stencil/Phase-C inventory. |
| `mpiDistributedStencil1D` | `smoothShell(state, next) <-> smooth_step` | Phase-C fallback | `AccessPattern` + `PackPlan` | stencil window 1D | no | OR rejects because 1D direct layout requires direct-only params; Phase-C owns this surface. |
| `mpiDistributedStencilAstRoute1D` | `smoothShell(state, next) <-> smooth_step` | Phase-C fallback | `AccessPattern` + `PackPlan` | stencil window 1D | no | Stencil route/followup fixture; not contiguous legacy. |
| `mpiDistributedStencilRouteFallback1D` | `smoothShell(state, next) <-> smooth_step` | Phase-C fallback with root bridge | `AccessPattern` + `PackPlan` | stencil window 1D, root-centric followup | no | Root-bridge/followup constraint; not simple contiguous legacy. |
| `mpiDistributedStencilAstRouteFallback1D` | `smoothShell(state, next) <-> smooth_step` | Phase-C fallback with root bridge | `AccessPattern` + `PackPlan` | stencil window 1D, root-centric followup | no | Root-bridge/followup constraint; not simple contiguous legacy. |
| `mpiDistributedStencilNoBridge1D` | `smoothShell(state, next) <-> smooth_step` | Phase-C fallback | `AccessPattern` + `PackPlan` | stencil window 1D | no | Stencil fallback with no root bridge; not simple contiguous legacy. |
| `mpiDistributedStencilSteady1D` | `smoothShell(state, next) <-> smooth_step` | Phase-C fallback | `AccessPattern` + `PackPlan` | stencil window 1D | no | Stencil steady-route fixture; not simple contiguous legacy. |
| `mpiDistributedStencilFanout1D` | `fanShell(left, right, next) <-> fan_step` | Phase-C fallback | `AccessPattern` + `PackPlan` | multi-input stencil/fanout 1D | no | Multiple routes/readers; not simple contiguous legacy. |
| `mpiDistributedStencilMultiRoute1D` | `twoShell(a, b, next_a, next_b) <-> two_step` | Phase-C fallback | `AccessPattern` + `PackPlan` | multi-route stencil 1D | no | Multiple routes/outputs; not simple contiguous legacy. |
| `mpiDistributedStencil2DRowBlock` | `heat2dShell(state, next) <-> heat2d_step` | Phase-C fallback | `AccessPattern` + `PackPlan` | 2D stencil row block | no | Stencil surface; not simple contiguous legacy. |
| `mpiOrStencilBoundaryStride2D` | `strideShell(state, next) <-> stride_step` | OR rejects, Phase-C fallback | `AccessPattern` + `PackPlan` | 2D stencil with boundary stride | no | Reject reason: 2D row-block layout requires direct-only params; boundary stride is stencil expansion work. |
| `mpiPhaseCWriteThenRead1D` | `clampShell(state, next) <-> clamp_step` | Phase-C fallback | `AccessPattern` + `PackPlan` | 1D stencil, read/write constrained | no | READ_WRITE/followup semantics; not simple contiguous legacy. |
| `mpiPhaseCHalo1D` | `haloShell(state, next) <-> halo_step` | Phase-C fallback | `AccessPattern` + `PackPlan` | 1D stencil halo | no | Phase-C halo fixture; not simple contiguous legacy. |
| `mpiPhaseCHaloWide1D` | `wideShell(state, next) <-> wide_step` | Phase-C fallback | `AccessPattern` + `PackPlan` | 1D wide/irregular stencil window | no | Wide split `split(5,2)`; stencil expansion work. |
| `mpiMixedStencilORPhaseC` | 2D stencil plus 1D clamp expressions | mixed OR + Phase-C fallback | `AccessPattern` + `PackPlan` for Phase-C portion | mixed stencil / read-write constrained | no | Mixed fixture intentionally exercises Phase-C fallback alongside OR. |

### Candidate Decision

No Phase 2.2 simple contiguous 1D legacy fast-path candidate was selected from
the current inventory.

The only current simple contiguous 1D benchmark shapes (`vectorAddCombo`,
`mandel1.0`, `decay1.0`, and focused broadcast/accumulate fixtures) are already
OR-current and have no legacy evidence. The true legacy wrapper fixtures are
FixedBlock rejects with matrix/payload/overlap constraints, while the
`PackPlan`-bearing focused fixtures are stencil Phase-C fallbacks. Starting a
Phase 2 fast path from any of those would expand surface or weaken guards, so
the safe next step is either to keep Phase 2.2 parked until a real simple
legacy case appears, or pivot to the documented Phase 3 materialization
inventory for current OR residual costs.

## Phase 2 Closeout Recheck

Rechecked on 2026-05-10 against the same Phase 2.1 artifact:
`/Volumes/QUQ/working/mpi_tmp/phase2_1_legacy_inventory`.

Artifact-level counts:

- `translation_runs.tsv`: 63 total translation records, 63 `ok`.
- Groups: 14 `app_cases`, 49 `focused_mpi`.
- `marker_scan.tsv`: 19 rows with `AccessPattern`, 16 rows with `PackPlan`,
  3 rows with a legacy wrapper, 16 rows with a Phase-C wrapper, and 45 rows
  with an OR wrapper.

Candidate recheck:

- All 14 benchmark cases remain free of `AccessPattern`, `PackPlan`, and
  legacy wrapper markers in generated snapshots.
- `vectorAddCombo` remains `__dacpp_mpi_or_` codegen with
  `VADD -> VSHIFT -> VADD`, `Contiguous1D` layout, and a replicated scalar
  bias. The log records `chain=accepted codegen=enabled`; no fresh legacy
  evidence appeared.
- The only generated legacy wrappers remain:
  `mpiFixedBlockMatrixSingleSplitReject1D`,
  `mpiFixedBlockOverlapReject1D`, and
  `mpiFixedBlockPayloadReject1D`.
- Their OR reject reasons are respectively:
  `fixed block first slice requires 1D wrapper tensors`,
  `fixed block requires split size to equal split stride`, and
  `fixed block first slice requires one split per parameter`.
- The other 16 `AccessPattern`+`PackPlan` rows are Phase-C stencil fallback
  fixtures such as `mpiDistributedStencil1D`, `mpiPhaseCHalo1D`,
  `mpiLoopStencilScalarReject2D`, and related multi-route/2D stencil cases.

Closeout decision:

- Phase 2.2 through Phase 2.4 are blocked/parked as no-op completion.
- No translator fast path was implemented, because there is no current legal
  simple contiguous 1D legacy candidate.
- The current FixedBlock reject fixtures and Phase-C stencil fallbacks are not
  Phase 2 targets. Optimizing them would be accepted-surface expansion work for
  later FixedBlock or stencil phases.

Full benchmark prep:

- Artifact path:
  `/Volumes/QUQ/working/mpi_tmp/phase2_closeout_full_benchmark`.
- Case list:
  `DFT1.0 FOuLa1.0 MDP1.0 decay1.0 gradientSum imageAdjustment1.0
  jacobi1.0 liuliang1.0 mandel1.0 matMul1.0 oddeven0.1 stencil1.0
  vectorAddCombo waveEquation1.0`.
- Expected primary output: `results.tsv`.
- Expected per-case outputs: patched DAC source, patched hand MPI+SYCL source,
  generated `case.mpi.dac_sycl_buffer.cpp`, `build_standard.log`,
  `translate_dac.log`, `build_dac.log`, `run_standard.log`, `run_dac.log`,
  and the two built binaries.

## Phase 3.1 Materialization Inventory

Artifact:
`/Volumes/QUQ/working/phase3_mpi_tmp/phase3_1_materialization_inventory`.

Contents:

- `materialization_inventory.tsv`
- `summary.md`
- per-case `generated.cpp`
- per-case `step2.log`

Scope: the 14 benchmark application cases:
`DFT1.0 FOuLa1.0 MDP1.0 decay1.0 gradientSum imageAdjustment1.0
jacobi1.0 liuliang1.0 mandel1.0 matMul1.0 oddeven0.1 stencil1.0
vectorAddCombo waveEquation1.0`.

Inventory readout:

| Case | Gatherv | Bcast | array2Tensor | replace_resident | materialize helpers | Reason |
|---|---:|---:|---:|---:|---:|---|
| `DFT1.0` | 2 | 2 | 3 | 0 | 0/0 | Root-visible output / all-ranks output sync; replicated full-tensor reader broadcasts are current OR semantics. |
| `FOuLa1.0` | 1 | 2 | 0 | 0 | 0/0 | Owner-loop final host-visible history materialization; per-step boundary scalar bcast and halo are semantic. |
| `MDP1.0` | 2 | 0 | 2 | 0 | 1/3 | Resident-halo final host-visible reader/writer materialization; no per-step gather/bcast. |
| `decay1.0` | 1 | 1 | 1 | 0 | 1/2 | Per-run materialization remains required because the source immediately copies `local_A_tensor` into `A_tensor`; registry update is skipped after host visibility. |
| `gradientSum` | 1 | 0 | 1 | 0 | 0/0 | Root-visible/all-ranks output materialization for row-partition output; no downstream resident reader. |
| `imageAdjustment1.0` | 1 | 0 | 1 | 1 | 0/0 | First expression output is retained for downstream resident read; final output materializes for visibility. |
| `jacobi1.0` | 1 | 3 | 2 | 0 | 0/0 | Replicated full `x` input bcast and final `x_new` host/all-ranks visibility; no downstream resident reader. |
| `liuliang1.0` | 2 | 0 | 2 | 0 | 1/3 | Resident-halo final host-visible reader/writer materialization; no per-step gather/bcast. |
| `mandel1.0` | 1 | 2 | 2 | 0 | 0/0 | Final host-visible/all-ranks output materialization for Contiguous1D OR; no downstream resident reader. |
| `matMul1.0` | 1 | 1 | 1 | 0 | 0/0 | Row-partition/full-row payload distribution plus final host/all-ranks output materialization. |
| `oddeven0.1` | 1 | 1 | 2 | 0 | 1/3 | P5 phase-exchange final host-visible source tensor materialization; per-iteration materialization was already removed. |
| `stencil1.0` | 1 | 0 | 2 | 0 | 1/3 | 2D resident-halo final host-visible materialization; no per-step full sync. |
| `vectorAddCombo` | 1 | 3 | 2 | 2 | 0/0 | OR chain intermediates stay resident; only final `out_tensor` gather/bcast/array2Tensor remains for all-ranks output sync. |
| `waveEquation1.0` | 2 | 0 | 3 | 0 | 1/3 | 2D resident-halo/direct-reader final host-visible materialization; no per-step full sync. |

All 14 inventory generated snapshots are still free of `AccessPattern` and
`PackPlan`.

## Phase 3 Optimization Decision

Implemented one narrow, proof-backed optimization:

- `ResidencyAnalysis` now records future non-scalar reads of each tensor inside
  the accepted OR chain.
- Each written output param gets `retainResidentAfterWrite=yes` only when a
  later expression in the same chain reads the same tensor from resident state.
- `ResidentBufferCodegen` and `LoopLoweredDirectCodegen` skip the resident
  registry update only when the output has already been materialized and
  `retainResidentAfterWrite=no`.
- Final `Gatherv`, `array2Tensor`, required `Bcast`, and materialize helpers
  are preserved exactly; the optimization removes resident-host-resident
  bookkeeping, not visibility sync.

Structure evidence:

- `vectorAddCombo`: `tmp_tensor` and `shifted_tensor` retain resident state for
  downstream readers; final `out_tensor` logs `retain=no` and skips the final
  registry rewrite after materialization.
- `imageAdjustment1.0`: first output is retained for the downstream chain read;
  final output skips the no-downstream registry update.
- `decay1.0` and `jacobi1.0`: no downstream resident readers, so generated
  code documents the skipped resident update while keeping materialization.

Parked as unsafe without more proof:

- General final broadcast deferral or root-only substitution. Existing
  all-ranks output-sync behavior remains in force unless `OutputSyncAnalysis`
  already proves otherwise.
- `decay1.0` per-run gather/bcast removal. The host-side
  `A_tensor[...] = local_A_tensor` statement immediately after the DAC
  expression requires materialized `local_A_tensor` under the current contract.

Additional guard fix:

- `OperatorChainAnalysis` now requires 1D resident-halo B1 to prove reader
  extent equals writer extent plus `windowSize - 1`.
- `mpiOrStencilRefreshPolicy1D` now rejects resident halo and falls back to
  `StencilFullSync` with:
  `resident halo B1 requires reader extent to cover writer slice plus halo`.

## Phase 3 Legacy Gather/Bcast Reuse Follow-up

The old legacy AccessPattern wrapper does not own a separate final-output
visibility analysis anymore. Its writeback path calls shared
`classifyOutputSyncRequirement(...)` and `requiresBroadcast(...)` before
emitting final `MPI_Bcast`.

Comparison against OR paths:

- Ordinary OR and loop-lowered direct materialization already reused the same
  proof through `ParamAccessPlan::broadcastMaterializedOutput`.
- Phase-C stencil fallback also calls the shared output-sync helper, with its
  explicit distributed-followup handling.
- Standalone FixedBlock final materialization had a real gap: it always emitted
  final `MPI_Bcast` even when `annotateOutputSync` had classified the writer
  output as `root-only`.

Implemented follow-up:

- `FixedBlockCodegen.cpp` now guards standalone FixedBlock final broadcast with
  `writer->broadcastMaterializedOutput`.
- Root rank still gathers and calls `array2Tensor`; only the non-root broadcast
  and non-root `array2Tensor` are skipped when the existing proof says
  `root-only`.
- `all-ranks-needed` FixedBlock outputs keep the prior Bcast path.

Structure/correctness evidence:

- Added `mpiFixedBlockRootOnlyCout1D`: standalone FixedBlock writer output is
  `sync=root-only`, generated code contains the final `Gatherv` and root
  `array2Tensor`, and no longer contains
  `MPI_Bcast(__or_materialized_output.data()...)`.
- Added `mpiFixedBlockAllRanksFunctionRead1D`: the same standalone FixedBlock
  shape is `sync=all-ranks-needed` after a host function read and still
  contains `MPI_Bcast(__or_materialized_output.data()...)`.

Artifacts:

- Before generated snapshot:
  `/Volumes/QUQ/working/phase3_mpi_tmp/phase3_2_fixedblock_bcast_before`.
- After generated snapshot:
  `/Volumes/QUQ/working/phase3_mpi_tmp/phase3_2_fixedblock_bcast_after`.
- Micro benchmark:
  `/Volumes/QUQ/working/phase3_mpi_tmp/phase3_2_fixedblock_bcast_micro_benchmark`.

Micro benchmark readout for the tiny 8-element fixture:

| Variant | Median wall (s) | Mean wall (s) | Trials |
|---|---:|---:|---:|
| Before | 0.076915 | 0.135527 | 5 |
| After | 0.076529 | 0.186126 | 5 |

Interpretation: the micro case is far below a useful timing scale and first-run
noise dominates. The useful evidence is structural: two final `MPI_Bcast` calls
were removed from the root-only FixedBlock generated code, while the all-ranks
counterpart preserves broadcast.

Still parked:

- Loop-resident P5 phase-exchange final broadcast. Its materializer writes the
  contracted source/reader tensor after absorbing phase-B and post-region
  semantics, so the standalone FixedBlock writer-output proof is not directly
  equivalent.

## Phase 3 Benchmark And Profile Evidence

Because no usable Phase 2 full-suite benchmark artifact existed, the Phase 3
before run was produced by temporarily reverse-applying the Phase 3 code patch,
benchmarking, and then reapplying the patch.

Patch snapshot:
`/Volumes/QUQ/working/phase3_code_changes.patch`.

Full benchmark artifacts:

- Before: `/Volumes/QUQ/working/phase3_mpi_tmp/phase3_0_baseline_full_benchmark`
- After: `/Volumes/QUQ/working/phase3_mpi_tmp/phase3_1_after_full_benchmark`

Selected 4-rank DAC wall results:

| Case | Before DAC (s) | After DAC (s) | Delta |
|---|---:|---:|---:|
| `decay1.0` | 0.993337 | 0.937886 | -5.6% |
| `imageAdjustment1.0` | 1.002949 | 0.946576 | -5.6% |
| `vectorAddCombo` | 0.839850 | 0.829981 | -1.2% |
| `stencil1.0` | 1.448044 | 1.604763 | +10.8% |
| `DFT1.0` | 0.662391 | 0.723748 | +9.3% |
| `oddeven0.1` | 0.827537 | 0.634033 | -23.4% |

Interpretation: the single-run full benchmark is noisy. The focused 3-trial
profile run is the better close-comparison artifact. It shows improvements for
`DFT1.0`, `decay1.0`, and `jacobi1.0`, a small regression for
`vectorAddCombo`, and no clean segment drop because the skipped registry update
is not a separately profiled segment.

Focused profile artifacts:

- Before: `/Volumes/QUQ/working/phase3_mpi_tmp/phase3_0_baseline_profile_focus`
- After: `/Volumes/QUQ/working/phase3_mpi_tmp/phase3_1_after_profile_focus`

4-rank DAC wall medians from focused profile:

| Case | Before (s) | After (s) | Delta |
|---|---:|---:|---:|
| `DFT1.0` | 0.207593 | 0.191026 | -8.0% |
| `decay1.0` | 0.425867 | 0.380414 | -10.7% |
| `gradientSum` | 0.440146 | 0.436874 | -0.7% |
| `jacobi1.0` | 0.366836 | 0.328427 | -10.5% |
| `mandel1.0` | 2.511430 | 2.581752 | +2.8% |
| `vectorAddCombo` | 0.257293 | 0.265042 | +3.0% |

Main 4-rank segment signals after the change:

- `vectorAddCombo`: scatter 46.638ms, materialize 38.676ms, bcast 19.371ms,
  kernel 16.276ms, gather 10.107ms.
- `decay1.0`: kernel 498.767ms, gather 96.181ms, bcast 53.385ms,
  materialize 2.444ms. The per-run materialization remains semantic.
- `gradientSum`: scatter remains dominant at 1031.820ms summed across ranks;
  gather/materialize are small.
- `mandel1.0`: kernel dominates, with significant final gather/bcast/materialize
  still required by current all-ranks output visibility.
