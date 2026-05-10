# MPI Performance Optimization TODO

Updated: 2026-05-10

This document is the working TODO for the next MPI translator performance
optimization pass. It is intentionally ordered from measurement and broad
overhead removal toward accepted-surface expansion.

## Correctness Rule

Performance work must not change current translation correctness.

Keep these rules as hard gates for every item below:

- Do not weaken existing semantic guards to make a benchmark faster.
- If an optimized path cannot prove its shape, fall back to the current correct
  path.
- Do not change generated-program output expectations unless the source program
  itself is intentionally changed and reviewed.
- Do not replace `all-ranks` output visibility with `root-only` behavior unless
  the selected output-sync policy or a proof explicitly allows it.
- Keep every new fast path covered by `mpi_expect.txt` structure checks and
  output comparison tests.
- Keep benchmark-only source scaling separate from translator correctness
  changes.
- Prefer additive instrumentation and guarded fast paths over edits that alter
  the behavior of existing accepted paths.

Minimum validation before considering any optimization complete:

```bash
cd /Volumes/QUQ/working/dacpp
cmake --build build --target translator -j8

cd /Volumes/QUQ/working/dacpp/clang/tools/translator
bash test_mpi.sh <focused-cases>
git diff --check -- clang/tools/translator
```

For high-blast-radius changes, also run the full MPI suite:

```bash
bash test_mpi.sh
```

## Priority Order

1. Add segmented profiling.
2. Optimize simple contiguous fast paths currently handled by legacy
   AccessPattern, with `vectorAddCombo` as the first checkpoint if it regresses
   into legacy or exposes residual pack overhead.
3. Remove resident-chain and materialization overhead that profiling proves is
   still relevant.
4. Generalize the `FOuLa1.0` loop-local argument contract.
5. Expand stencil and FixedBlock accepted surfaces.

## 1. Segmented Profiling

Goal: make performance work evidence-driven. The generated MPI code should be
able to report time spent in init/scatter, pack, kernel, halo, gather, bcast,
materialization, and final synchronization without changing normal execution.

Status on 2026-05-10: Phase 1.1 through Phase 1.4 are implemented for the
current focused surfaces. Profiling is disabled by default and enabled only by
`DACPP_MPI_PROFILE=1`.

Current driver:

```bash
python3 clang/tools/translator/bench_mpi_profile_segments.py \
  --tmp-dir /Volumes/QUQ/working/mpi_tmp/phase1_4_profile_focus \
  --ranks 1,2,4,8 \
  --trials 3 \
  vectorAddCombo FOuLa1.0 MDP1.0 stencil1.0 oddeven0.1 decay1.0
```

Primary artifact:
`/Volumes/QUQ/working/mpi_tmp/phase1_4_profile_focus`.

Legacy fallback smoke artifact:
`/Volumes/QUQ/working/mpi_tmp/phase1_4_legacy_profile_smoke`.

The profile driver stores `results.tsv`, `summary.tsv`, `profile_raw.tsv`,
`profile_summary.tsv`, per-trial stdout/stderr, generated source snapshots,
metadata, git status/diff, and snapshots of currently untracked translator
planning/script files. It checks that DAC profile-enabled stdout matches DAC
profile-disabled stdout. Baseline correctness remains covered by `test_mpi.sh`.

### Phase 1.1: Define Profiling Taxonomy

Tasks:

- Standardize profiling segment names:
  - `init`
  - `scatter`
  - `pack`
  - `kernel`
  - `halo`
  - `gather`
  - `bcast`
  - `materialize`
  - `final_sync`
- Decide whether each segment is measured per wrapper call, per loop step, or
  once per generated program.
- Define output format that can be aggregated by benchmark scripts, for example
  TSV lines gated by `DACPP_MPI_PROFILE=1`.
- Keep profiling disabled by default.

Likely files:

- `dpcppLib/include/mpi/common/Profile.h`
- `dpcppLib/include/mpi/operator_resident/OperatorResidentRuntime.h`
- `dpcppLib/include/mpi/legacy_access_pattern/*`
- `rewriter/lib/mpi/*Codegen.cpp`
- `bench_mpi_only_requested.py`

Done when:

- There is one documented set of segment names.
- A tiny generated MPI program can emit profile lines only when
  `DACPP_MPI_PROFILE=1`.
- Existing tests pass with profiling disabled.

Completion evidence:

- Shared runtime API is in `dpcppLib/include/mpi/common/Profile.h`.
- Output rows use:
  `DACPP_MPI_PROFILE\t<label>\t<segment>\tcalls=<n>\tmax_ms=<ms>\tsum_ms=<ms>`.
- Focused MPI tests passed with profiling disabled.

### Phase 1.2: Instrument Legacy AccessPattern Path

Tasks:

- Time global index collection and `AccessPattern` construction as `pack` or
  `init`.
- Time root-side pack plan construction separately from data movement if
  practical.
- Time input `Scatterv`, output `Gatherv`, and final output `Bcast`.
- Report per-wrapper names so repeated DAC expressions can be distinguished.

Done when:

- A legacy-path case reports pack/scatter/kernel/gather/bcast segments.
- Profiling does not change baseline output.
- The profiler can identify whether a legacy case is dominated by metadata,
  communication volume, kernel, or final synchronization.

Completion evidence:

- `mpiFixedBlockMatrixSingleSplitReject1D` legacy fallback emits segmented
  `init/scatter/pack/kernel/gather/materialize/final_sync` rows.
- Existing `collect_positions_for_item` profile output is preserved.
- DAC profile-on stdout matched DAC profile-off stdout in the legacy smoke
  artifact.

### Phase 1.3: Instrument OR, Stencil, FixedBlock, and FOuLa Paths

Tasks:

- OR ordinary wrappers: measure scatter, resident input reuse, kernel, gather,
  materialize, and bcast.
- OR resident loops: measure init scatter, per-step kernel, halo exchange, and
  final materialize.
- Stencil Phase-C: measure init, run kernel/exchange, deferred materialize.
- FixedBlock phase exchange: measure initial scatter, local phase kernels,
  boundary exchange, final gather/bcast.
- FOuLa owner loop: measure initial column scatter, per-step kernel, boundary
  bcast, halo exchange, final history gather, and owner writeback.

Done when:

- Focused optimized cases produce comparable profile tables.
- Segment totals are close enough to external `mpirun` wall time to be useful.
- Profile output remains quiet unless enabled.

Completion evidence:

- Focused profile run covers ordinary OR (`vectorAddCombo`), loop-lowered direct
  OR (`decay1.0`), 1D resident halo (`MDP1.0`), 2D resident halo
  (`stencil1.0`), P5 phase exchange (`oddeven0.1`), and FOuLa owner-loop
  (`FOuLa1.0`).
- `profile_summary.tsv` provides comparable segment medians for 1/2/4/8 ranks.

### Phase 1.4: Benchmark Reporting

Tasks:

- Extend the benchmark driver or add a companion script to collect segmented
  profile logs.
- Store raw logs and generated source snapshots under a documented temp or data
  path.
- Report medians across at least three runs for cases where noise is high.
- Add rank-scaling rows for 1/2/4/8 ranks when supported locally.

Done when:

- `docs/benchmarks/mpi_efficiency_benchmark_summary.md` can point to raw profile
  artifacts.
- Each future optimization has a before/after profile, not only total wall time.

Completion evidence:

- `docs/benchmarks/mpi_efficiency_benchmark_summary.md` now records the Phase
  1.4 command, artifact paths, 4-rank medians, rank scaling, and segmented
  readout.
- Future fast-path work should use this driver to capture a before profile and
  repeat it after the change.

Phase 1.4 readout for next work:

- `vectorAddCombo` remains OR `Contiguous1D`; residual cost points to
  scatter/final materialize/bcast, not legacy pack.
- `decay1.0` shows significant per-step gather+bcast and is a strong Phase 3
  materialization/residency inventory candidate.
- Resident halo cases are mostly kernel+halo dominated in the focused run; do
  not widen stencil or FOuLa surface before a separate proof-driven phase.

## 2. Legacy AccessPattern Simple Contiguous Fast Path

Goal: remove root-centric global-index metadata and pack overhead for simple
regular shapes that still route through legacy AccessPattern, while preserving
legacy fallback for all unsupported shapes.

`vectorAddCombo` should remain a checkpoint because it used to expose this class
of overhead. If current `vectorAddCombo` stays on OR, use profiling to find the
next remaining legacy contiguous case.

### Phase 2.1: Inventory Current Legacy Cases

Tasks:

- Run translation logs for the 14 benchmark cases and focused `mpi*` tests.
- Record which expressions still generate `AccessPattern`, `PackPlan`, or
  legacy wrapper calls.
- Classify each legacy case:
  - simple contiguous 1D
  - row-contiguous 2D
  - scalar/replicated input
  - irregular indexing
  - alias or post-use constrained
- Choose the first case only after confirming it is still legacy under current
  code.

Done when:

- A table lists remaining legacy expressions, source case, shape reason, and
  expected fast-path candidate.
- `vectorAddCombo` is either confirmed as OR-current or selected only if it has
  a real legacy segment left.

### Phase 2.2: Add Detection for Simple Contiguous Legacy Shapes

Tasks:

- Add a narrow analysis gate that recognizes simple contiguous access without
  needing global-index pack metadata.
- Require native MPI element type or add a clear byte-transport path.
- Require no unsupported aliasing, no irregular index expression, and no
  semantic dependence on global index order beyond contiguous ownership.
- Emit an explicit accept/reject reason in logs.

Likely files:

- `rewriter/lib/mpi/shared/MpiPlanBuilder.cpp`
- `rewriter/lib/mpi/operator_resident/ShellPartitionAnalysis.cpp`
- `rewriter/lib/mpi/legacy_access_pattern/WrapperCodegen.cpp`
- `rewriter/include/mpi/operator_resident/OperatorResidentPlan.h`

Done when:

- New fast path accepts only the intended contiguous shape.
- Near-miss tests still fall back to legacy and log why.

### Phase 2.3: Generate Fast Path Without AccessPattern Metadata

Tasks:

- Replace global-index collection with rank-range derived counts/displacements.
- Scatter contiguous inputs directly.
- Launch the existing local SYCL kernel view shape where possible.
- Gather contiguous outputs directly.
- Preserve final output-sync semantics exactly as the current wrapper would.

Done when:

- Generated code for the selected case has no `AccessPattern` or `PackPlan`.
- Output matches the baseline for 1, 2, and 4 ranks where feasible.
- Profile shows the expected reduction in pack/init overhead.

### Phase 2.4: Regression Surface

Tasks:

- Add or update `mpi_expect.txt` for accepted fast-path cases.
- Add reject fixtures for aliasing, irregular index, unsupported byte type, and
  post-output use if those are relevant to the gate.
- Benchmark before/after with segmented profiling.

Done when:

- The fast path is covered by structure and output tests.
- Legacy fallback still works for unsupported cases.

## 3. Resident Chain And Materialization Elimination

Goal: reduce unnecessary resident-to-host and host-to-resident movement inside
ordinary OR chains and loop-lowered OR paths after profiling proves the cost is
material.

The first resident-buffer reference/move optimization is already in place. This
item is for the next layer: eliminating materialization timing and visibility
sync that are not semantically required.

### Phase 3.1: Build Materialization Inventory

Tasks:

- For each benchmark case, record every generated `Gatherv`, `array2Tensor`,
  `replace_resident`, final `Bcast`, and materialize helper call.
- Mark why each materialization exists:
  - root-visible output needed immediately
  - all-ranks output sync required
  - downstream DAC expression needs resident data
  - host post-region reads output
  - conservative fallback because analysis lacks proof
- Use segmented profiling to identify high-cost materializations.

Done when:

- There is a case-by-case materialization table in benchmark notes or findings.
- Each high-cost materialization has a semantic reason or an optimization TODO.

### Phase 3.2: Strengthen Residency Facts

Tasks:

- Track root-valid, local-valid, dirty, materialized-root, and replicated state
  across chain expressions more explicitly.
- Distinguish host-visible source statements after the chain from generated
  wrapper-internal visibility needs.
- Avoid materializing outputs that are only consumed by downstream resident
  operators.
- Keep final materialization when root print, tensor2Array, unknown function,
  or all-ranks sync requires it.

Likely files:

- `rewriter/lib/mpi/operator_resident/ResidencyAnalysis.cpp`
- `rewriter/lib/mpi/operator_resident/OperatorChainAnalysis.cpp`
- `rewriter/lib/mpi/operator_resident/ResidentBufferCodegen.cpp`
- `rewriter/lib/mpi/shared/OutputSyncAnalysis.cpp`

Done when:

- Chain-local materialization decisions are explainable in logs.
- Existing output-sync tests still pass.

### Phase 3.3: Defer Or Remove Final Broadcast Where Semantically Safe

Tasks:

- Use `OutputSyncAnalysis` to prove whether non-root ranks need the final
  output copy.
- Keep default `all-ranks` semantics unless the selected policy or proof allows
  otherwise.
- Consider delaying bcast until a non-root read is actually needed.
- Keep `root-only` behavior controlled by existing option and never silently
  substitute it for default semantics.

Done when:

- Cases with root-only print avoid unnecessary all-rank sync only when semantics
  already allow it.
- Existing broadcast-root-only tests still pass.

### Phase 3.4: Benchmark And Guard

Tasks:

- Re-run `vectorAddCombo`, `decay1.0`, `jacobi1.0`, and any chain-heavy case
  found by profiling.
- Compare segment-level materialize/gather/bcast time before and after.
- Add `SYCL_NOT_CONTAINS` checks for removed unnecessary materialization only
  where the absence is a contract.

Done when:

- Each removed materialization has a documented proof.
- No benchmark improvement is accepted without output comparison passing.

## 4. FOuLa Loop-Local Argument Contract

Goal: turn the current strict `FOuLa1.0` owner-loop specialization into a
reusable loop-local slice ownership contract.

The current implementation is intentionally narrow and correct for the proven
shape. The next step should generalize the proof model, not loosen checks.

### Phase 4.1: Specify The Contract

Tasks:

- Define a contract for loop-local shell arguments:
  - stable owner tensor
  - loop induction variable and step direction
  - reader slice derived from owner at current step
  - writer slice derived from owner at next step
  - scalar or replicated arguments derived from invariant expressions
  - post-DAC writeback from writer slice into owner
- Record which statements are replaced, removed, or preserved.
- Define materialization timing for owner state.
- Define rejection reasons for shape mismatch.

Likely files:

- `rewriter/include/mpi/shared/LoweringContract.h`
- `rewriter/include/mpi/operator_resident/OperatorResidentPlan.h`
- `rewriter/lib/mpi/operator_resident/OperatorChainAnalysis.cpp`
- `rewriter/lib/Rewriter_MPI.cpp`

Done when:

- The contract can describe current `FOuLa1.0` without special-case wording.
- Rejection reasons are precise enough for tests.

### Phase 4.2: AST-Based Extraction

Tasks:

- Replace regex-heavy detection with AST extraction where practical:
  - loop induction variable
  - owner tensor slice expressions
  - scalar vector construction
  - DAC expression statement
  - post-writeback loop
- Keep source-text fallback only for syntax forms that the current AST helpers
  cannot yet represent.
- Reject ambiguous aliases, unknown mutation of owner, variant scalar payload,
  and multiple writer slices.

Done when:

- Current FOuLa still lowers to owner-loop.
- Negative fixtures reject with stable reason text.
- Detection does not depend on file names.

### Phase 4.3: Codegen Reuse

Tasks:

- Move generated owner-loop code out of `Rewriter_MPI.cpp` into a focused
  operator-resident codegen module.
- Reuse existing resident-halo runtime helpers where possible.
- Parameterize:
  - element type
  - window width
  - boundary source
  - scalar arguments
  - output history materialization
- Keep the current strict generated shape as the first supported instance.

Done when:

- `Rewriter_MPI.cpp` returns to orchestration.
- FOuLa owner-loop generated code remains identical or behaviorally equivalent.

### Phase 4.4: Broaden Carefully

Tasks:

- Add one new supported loop-local shape at a time.
- For each shape, add:
  - accepted fixture
  - at least one reject fixture
  - benchmark result
  - contract documentation
- Candidate expansions:
  - different scalar variable names
  - multiple replicated scalar arguments
  - constant boundary values
  - different 1D stencil window widths

Done when:

- The generic contract covers FOuLa plus at least one additional non-file-name
  shape.
- All old FOuLa parity numbers remain valid within benchmark noise.

## 5. Expand Stencil And FixedBlock Surface

Goal: grow accepted optimized surfaces after the broad overhead work is stable
and measurable.

Do this last because it increases semantic surface area. Every expansion needs
guards, reject tests, and benchmark evidence.

### Phase 5.1: Stencil Surface Inventory

Tasks:

- List current stencil rejects by reason:
  - scalar reader unsupported
  - direct-reader form unsupported
  - read-cache transition unsupported
  - boundary-local pattern unsupported
  - root bridge missing
  - unstable loop site
- Rank candidates by expected performance gain and proof simplicity.
- Prefer shapes that can reuse P4.6 resident-halo runtime without new
  communication semantics.

Done when:

- There is a prioritized list of stencil reject reasons and candidate cases.

### Phase 5.2: Stencil Expansion Slices

Tasks:

- Add one guard at a time:
  - broader boundary-local assignment forms
  - scalar readers where invariant and native MPI type are proven
  - additional direct-reader/read-cache offsets
  - wider 1D or 2D windows
- Keep resident ownership and final materialization rules explicit.
- Never add root-bridge behavior unless a sound resident model is specified.

Done when:

- Each new stencil slice has accepted and rejected fixtures.
- Existing `MDP1.0`, `liuliang1.0`, `stencil1.0`, and `waveEquation1.0`
  generated paths remain unchanged unless intentionally improved.

### Phase 5.3: FixedBlock Surface Inventory

Tasks:

- List current FixedBlock rejects:
  - payload dimensions
  - multi-split params
  - overlapping windows
  - READ_WRITE blocks
  - non-adjacent phase patterns
  - odd total length
  - post-output use or alias
- Decide which rejects are semantic hard limits and which are engineering
  limits.
- Use profiling on `oddeven0.1` and any new candidate to identify actual cost.

Done when:

- FixedBlock expansion candidates are ordered by proof simplicity and expected
  gain.

### Phase 5.4: FixedBlock Expansion Slices

Tasks:

- Extend phase-exchange only when boundary ownership and post-use safety are
  provable.
- Add payload support only with explicit layout and communication rules.
- Add READ_WRITE support only after defining freshness and alias semantics.
- Keep runtime guards for size, parity, and rank-local count assumptions.

Done when:

- `oddeven0.1` remains a regression checkpoint.
- New FixedBlock accepted cases have structure checks, reject fixtures, and
  benchmark/profile data.

## Cross-Cutting Deliverables

For every phase above, produce these artifacts:

- A short design note in the commit or task log describing the proof and
  fallback condition.
- Focused `mpi_expect.txt` updates for accepted/rejected generated structure.
- Before/after benchmark rows with profile segments when performance is the
  purpose of the change.
- A note in `docs/benchmarks/mpi_efficiency_benchmark_summary.md` only after
  numbers are measured.
- A note in `docs/mpi_translator/mpi_translator_current_status.md` only after
  the accepted surface actually changes.

## Suggested Immediate Next Step

Start with Phase 1.1 and Phase 1.2. They are low semantic risk and will make
the remaining optimization choices much less speculative.
