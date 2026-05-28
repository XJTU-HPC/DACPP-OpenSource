# Task Plan: Optimize gradientSum MPI Performance

---

# Task Plan: 2026-05-22 stencil1 spatial-k2 profitability follow-up

## Goal
Advance `stencil1.0` toward generic `distribution=spatial-2d` with `temporal-block=2` under bounded host post-use, without benchmark-name special cases, while preserving row-temporal k=2 fallback and all P6-P11 accepted paths.

## Current Phase
Phase 5 complete

## Phases

### Phase 1: Discovery and Reproduction
- [x] Re-read required plan/codegen/runtime files and target fixtures
- [x] Inspect current spatial-k2/profitability fallback and generated/profile evidence
- [x] Identify why bounded host post-use makes spatial k=2 slow
- **Status:** complete

### Phase 2: Minimal Generic Improvement
- [x] Improve post-use/final-sync/profitability only if evidence supports a safe generic change
- [x] Keep explicit reject/fallback translator logs for every downgrade
- [x] Preserve waveEquation direct-reader row-temporal path
- **Status:** complete

### Phase 3: Fixtures and Documentation
- [x] Update expectations for accepted/rejected spatial-k2 evidence
- [x] Update optimization-plan Implementation Status
- **Status:** complete

### Phase 4: Verification and Profile
- [x] Build translator
- [x] Run requested target/regression MPI tests
- [x] Run related fixture tests discovered by `rg`
- [x] Run requested profile set
- **Status:** complete

### Phase 5: Commit and Push
- [x] Confirm staged files only contain this task's relevant edits
- [x] Commit and push to `origin/tqc-2`
- **Status:** complete

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| No benchmark-name gates | User explicitly requires structural/profitability decisions only |
| Keep row-temporal k=2 unless spatial k=2 is proven profitable/correct | Avoid silent performance regression and preserve existing temporal path |
| Leave direct-reader B3 on row layout | Spatial role rotation is separate and not prioritized for this task |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|

---

# Task Plan: 2026-05-27 Four-Case Optimization Implementation

## Goal
Serially implement four Shenwei efficiency optimizations using one sub-agent per
optimization, with parent-side review, `test_mpi` verification after each step,
and final structural/performance-regression checks.

## Current Phase
Phase 5 in_progress

## Phases

### Phase 1: FOuLa Owner-Loop Cleanup and Larger Temporal Block
- [x] Delegate implementation to sub-agent
- [x] Review patch for no shortcuts and no unrelated edits
- [x] Build translator
- [x] Run `test_mpi.sh FOuLa1.0`
- **Status:** complete

### Phase 2: Spatial-2D Halo Runtime Optimization
- [x] Delegate implementation to sub-agent
- [x] Review patch for no shortcuts and no unrelated edits
- [x] Build translator
- [x] Run `test_mpi.sh stencil1.0 waveEquation1.0`
- **Status:** complete

### Phase 3: Larger Temporal Block for Stencil/Wave
- [x] Delegate implementation to sub-agent
- [x] Review patch for no shortcuts and no unrelated edits
- [x] Build translator
- [x] Run `test_mpi.sh stencil1.0 waveEquation1.0`
- **Status:** complete

### Phase 4: Partial Materialization for Row/Print Post-Use
- [x] Delegate implementation to sub-agent
- [x] Review patch for no shortcuts and no unrelated edits
- [x] Build translator
- [x] Run `test_mpi.sh stencil1.0 waveEquation1.0 imageAdjustment1.0`
- **Status:** complete

### Phase 5: Final Review and Regression
- [ ] Check generated large-case code for all four optimizations
- [ ] Run target regression batch
- [ ] Run available performance/profile smoke check or explain why unavailable
- [ ] Summarize final status
- **Status:** pending

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| One sub-agent per optimization | User explicitly requested delegated implementation |
| Serial integration | Avoid overlapping edits in shared OR runtime/codegen |
| Parent reviews every patch | User asked to ensure no shortcuts |
| No benchmark-name gates | Keep translator logic structural and reusable |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|

---

# Task Plan: 2026-05-22 P11 2D Spatial Block Partition

## Goal
Implement a conservative, benchmark-name-free 2D spatial/block partition for eligible `StencilWindow2D` resident-halo workloads on branch `tqc-2`, preserving P6-P10 paths and leaving unsupported/direct-reader/temporal cases on the existing row-block fallback with explicit translator reasons.

## Current Phase
Phase 5 complete

## Phases

### Phase 1: Contract Discovery
- [x] Restore planning context and confirm branch/dirty state
- [x] Read P11 plan section and core OR plan/codegen/runtime files
- [x] Identify safe subset: canonical B2 3x3 stride-1 stencil, no direct reader, no scalar readers, canonical followup/boundary updates
- **Status:** complete

### Phase 2: Design and Implementation
- [x] Add plan metadata and analysis logs for `distribution=spatial-2d`
- [x] Add runtime 2D process-grid layout, rectangle scatter, four-neighbor halo, owned-rectangle slice, and bounded owner lookup
- [x] Extend 2D resident-halo codegen for accepted spatial-2d while preserving row fallback
- **Status:** complete

### Phase 3: Fixtures and Docs
- [x] Update `stencil1.0/mpi_expect.txt` for accepted spatial-2d evidence
- [x] Update nearby 2D rejection/structure expectations if logs change
- [x] Update P11 Implementation Status in optimization plan
- **Status:** complete

### Phase 4: Verification
- [x] `cmake --build build --target translator -j8`
- [x] `bash clang/tools/translator/test_mpi.sh stencil1.0`
- [x] `bash clang/tools/translator/test_mpi.sh mandel1.0 FOuLa1.0 liuliang1.0 MDP1.0 waveEquation1.0`
- [x] Related 2D/AccessPattern/stencil/operator-resident fixtures
- [x] Requested 4-rank profile probe
- **Status:** complete

### Phase 5: Staging Review and Delivery
- [x] Confirm staged files only contain P11-relevant edits before any commit
- [x] Summarize code changes, tests, profile, and residual gaps
- **Status:** complete

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Start with canonical B2 only | Direct-reader recurrence and k=2 temporal row widening are distinct contracts and unsafe to silently combine with 2D blocks |
| Keep row-block fallback | Existing P6-P10 accepted paths must not regress |
| Avoid benchmark-name checks | User explicitly requires generic layout/access/loop/post-use gates |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|

## Goal
Add a fast path in ResidentBufferCodegen.cpp to detect when input tensor is row-partitioned and skip unnecessary MPI_Scatterv by letting each rank read directly from its local buffer portion.

## Current Phase
Phase 4

## Phases

### Phase 1: Requirements & Discovery
- [x] Understand user intent
- [x] Identify files to analyze
- [x] Read and analyze ResidentBufferCodegen.cpp
- [x] Read and analyze OperatorResidentCodegen_Internal.h
- [x] Read gradientSum.mpi.dac_sycl_buffer.cpp for reference
- **Status:** complete

### Phase 2: Planning & Structure
- [x] Define fast path detection criteria: indexedBindPos == 0 && !broadcastIndexedPayload
- [x] Design code changes: Add else-if branch for fast path using MPI_Bcast + direct copy
- **Status:** complete

### Phase 3: Implementation
- [x] Add fast path detection logic in scatter emission (else if indexedBindPos == 0 branch)
- [x] Emit MPI_Bcast + direct buffer copy with offset when fast path applies
- [x] Maintain correctness for non-fast-path cases (existing else branches)
- **Status:** complete

### Phase 4: Testing & Verification
- [x] Generate unified diff patch
- [ ] Verify patch correctness
- **Status:** in_progress

---

# Task Plan: 2026-05-27 Four-Case Shenwei Efficiency Diagnosis

## Goal
Analyze current translated MPI/SYCL code and translator lowering logic for
`imageAdjustment1.0`, `waveEquation1.0`, `stencil1.0`, and `FOuLa1.0` to explain
why multi-node efficiency on Shenwei is roughly stable around 1/2.

## Current Phase
Phase 3 complete

## Phases

### Phase 1: Gather Current Translation Evidence
- [x] Generate or locate current `--mode=buffer --mpi` outputs for the four cases
- [x] Identify selected lowering path and major MPI/SYCL segments per case
- **Status:** complete

### Phase 2: Case-by-Case Bottleneck Analysis
- [x] Inspect generated code for initialization, per-step halo/collective, kernel shape, and materialization
- [x] Compare generated behavior with hand-written/expected efficient multi-node structure where available
- **Status:** complete

### Phase 3: Root-Cause Summary
- [x] Group causes into translator-logic buckets
- [x] Record likely optimization directions without changing code
- **Status:** complete

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Diagnosis only | User asked to analyze reasons first, not implement fixes |
| Prefer generated code evidence | Shenwei results depend on translated output structure more than source-level intent |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|

### Phase 5: Delivery
- [ ] Present patch to user
- **Status:** pending

## Key Questions (Answered)
1. How is row-partition detected? indexedBindPos == 0 means indexed dimension is outer (row) dimension
2. Current scatter logic: Rank 0 packs via tensor2Array, then MPI_Scatterv distributes
3. Offset calculation: offset = __or_payload_index_begin_ * payloadLen

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Use MPI_Bcast + std::copy_n | Broadcast packed buffer to all ranks, then direct offset-based copy |
| Skip MPI_Scatterv | Each rank reads its portion directly from broadcast buffer |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|

---

# Task Plan: 2026-05-21 MPI Top4-5 Optimizations

## Goal
Continue from commit `471a9e05b Optimize MPI top3 resident paths` on branch `tqc-2` and implement optimization-plan items 4 and 5 without benchmark-name special cases:
1. `decay1.0`: generic P4.5 loop-lowered `Contiguous1D` device time loop for scalar-refresh patterns while preserving selective-row materialization.
2. `stencil1.0`: conservative P4.6 `StencilWindow2D` resident-halo temporal blocking with static `k=2`, protecting shared `waveEquation1.0` codegen/runtime.

## Current Phase
Phase 5 complete

## Phases

### Phase 1: Discovery and Contract Mapping
- [x] Read optimization plan and existing progress/findings
- [x] Confirm branch and dirty state; preserve unrelated deletions/untracked files
- [x] Inspect loop-lowered direct decay analysis/codegen and generated shape
- [x] Inspect resident-halo 2D stencil analysis/codegen/runtime and generated shape
- **Status:** complete

### Phase 2: Decay Device Time Loop
- [x] Add generic scalar-refresh/device-time-loop metadata and accepted/rejected logs
- [x] Generate single-kernel device time loop only for proven independent Contiguous1D scalar-refresh shapes
- [x] Preserve selective-row host materialization and fallback host loop
- [x] Update `decay1.0/mpi_expect.txt`
- **Status:** complete

### Phase 3: Stencil 2D Temporal Blocking
- [x] Add conservative k=2 acceptance metadata/logs for proven `StencilWindow2D` resident halo contract
- [x] Generate widened halo exchange and two local steps per exchange for accepted non-direct-reader shapes
- [x] Keep waveEquation/direct-reader and unsupported shapes on existing one-step path
- [x] Update `stencil1.0/mpi_expect.txt` and `waveEquation1.0` guard expectations if needed
- **Status:** complete

### Phase 4: Verification and Documentation
- [x] `cmake --build build --target translator -j8`
- [x] Target tests: `bash clang/tools/translator/test_mpi.sh decay1.0 stencil1.0`
- [x] Nearby regressions: `bash clang/tools/translator/test_mpi.sh waveEquation1.0 MDP1.0 FOuLa1.0 liuliang1.0`
- [x] Additional loop-lowered/stencil fixture tests if affected
- [x] Optional profile probe for requested cases
- [x] Append Implementation Status under plan items 4 and 5
- **Status:** complete

### Phase 5: Delivery
- [x] Summarize completion, changed files, tests, profile probe, and residual risk
- **Status:** complete

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Keep all new gates layout/access/loop-structure driven | User explicitly forbids benchmark-name special cases |
| Start stencil temporal blocking only for no-direct-reader 2D resident halo | WaveEquation shares codegen and has direct-reader recurrence risk |
| Keep fallbacks as generated one-step/host-loop paths | Small, conservative, reversible changes are required |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|

---

# Task Plan: 2026-05-21 MPI 32-Rank Top3 Optimizations

## Goal
Implement the first three generic optimizations from `clang/tools/translator/docs/mpi_translator/mpi_32node_top10_optimization_plan_2026-05-21.md` without benchmark-name special cases:
1. Output-direct no-read scatter skip for output-only resident tensors.
2. Index-local initializer generation for direct-mapped inputs such as `vec[i] = i`.
3. Conservative RowBlock2D pointwise OR-chain fusion for matching direct-mapped shells.

## Current Phase
Phase 5 complete

## Phases

### Phase 1: Discovery and Design
- [x] Read user task, optimization plan, existing planning context, and dirty tree
- [x] Inspect OR plan structures, init-sync analysis, scatter codegen, and RowBlock2D chain codegen
- [x] Inspect target tests and existing `mpi_expect.txt` structure guards
- **Status:** complete

### Phase 2: matMul Output-Direct No-Read
- [x] Add pattern/layout-driven output init-sync decision
- [x] Skip output initial `MPI_Scatterv` only when kernel proves no read-before-write/default local accumulation
- [x] Emit accepted/rejected translator evidence and update matMul expectation
- **Status:** complete

### Phase 3: DFT Index-Local Initializer
- [x] Recognize simple affine index-local vector initializers with conservative escape/mutation fallback
- [x] Generate local values from `rank_range.begin + local_i` instead of root pack/scatter
- [x] Emit accepted/rejected translator evidence and update DFT expectation
- **Status:** complete

### Phase 4: RowBlock2D Pointwise Chain Fusion
- [x] Detect matching pure pointwise direct-mapped RowBlock2D chains with single intermediate consumer
- [x] Generate one fused kernel using private intermediate values
- [x] Preserve resident-chain fallback and update imageAdjustment expectation
- **Status:** complete

### Phase 5: Verification and Documentation
- [x] `cmake --build build --target translator -j8`
- [x] Target tests: `matMul1.0 DFT1.0 imageAdjustment1.0`
- [x] Nearby regressions: `jacobi1.0 gradientSum vectorAddCombo mpiBroadcastTensor2Array mpiOrReadWriteAccumulate2D`
- [x] Optional profile probe for top3
- [x] Append Implementation Status under plan items 1-3
- **Status:** complete

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Keep matching based on access/layout/init facts, never benchmark names | User explicitly requires generic optimization |
| Prefer conservative rejection logs | Each optimization needs structural accepted/rejected evidence |
| Preserve unrelated dirty/untracked files | User explicitly requested no cleanup or revert |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|

---

# Task Plan: MPI Translator Prefix and Slice Materialization

## Goal
Continue MPI translator manycore communication optimization on branch `tqc-2` from commit `6924ae4e1` without reverting existing changes or cleaning unrelated dirty/untracked files. Add generic, benchmark-name-free sync/materialization optimizations for:
1. `tensor.tensor2Array(vec)` followed only by small constant-prefix/fixed-index host reads,
2. loop-lowered direct row/time-slice writes where final host use only needs a fixed row/slice,
3. 2D fixed-row post-use `print()`/slice reads and avoid redundant boundary broadcasts when values are locally computable.

## Current Phase
Phase 1 in progress

## Phases

### Phase 1: Restore Context and Discover Current Paths
- [x] Read existing plan/progress/findings and catchup context
- [x] Confirm current branch, latest commit, and dirty/untracked state
- [ ] Inspect tensor2Array post-use analysis/codegen and expectations
- [ ] Inspect loop-lowered direct decay materialization path
- [ ] Inspect FOuLa generated code/profile segments and resident history materialization path
- **Status:** in_progress

### Phase 2: Prefix/Slice tensor2Array Materialization
- [ ] Generalize post-use analysis for `tensor.tensor2Array(vec)` host vectors with fixed indices or small fixed prefix loops
- [ ] Rewrite accepted `tensor2Array(vec)` to initialize/fill only the needed vector prefix/slice
- [ ] Fallback full on unknown function call, nonfixed read, write, alias escape, or unsupported use
- [ ] Update `mpiBroadcastTensor2Array` and `vectorAddCombo` expectations
- **Status:** pending

### Phase 3: Decay Row/Time-Slice Materialization
- [ ] Detect loop-lowered direct output writeback into host tensor rows/slices
- [ ] If final post-use only reads a fixed row/slice, gather/materialize only the matching timestep row
- [ ] Preserve per-run gather fallback for dynamic rows, uncertain post-use, alias/function escape
- **Status:** pending

### Phase 4: FOuLa Fixed-Row History Materialization
- [ ] Support 2D fixed-row post-use slice/print materialization only for needed rows
- [ ] Avoid per-step root bcasts for boundary values proven locally computable
- [ ] Preserve correctness fallbacks for dynamic/escaped history uses
- **Status:** pending

### Phase 5: Verification and Profile
- [ ] `cmake --build build --target translator -j8`
- [ ] Target correctness: `vectorAddCombo decay1.0 FOuLa1.0`
- [ ] Regression correctness: `imageAdjustment1.0 jacobi1.0 mandel1.0 waveEquation1.0 stencil1.0`
- [ ] Full `test_mpi.sh`
- [ ] Requested target profile in `/Volumes/QUQ/working/mpi_tmp_profile_next2`
- **Status:** pending

### Phase 6: Delivery
- [ ] Summarize modified files
- [ ] Summarize new patterns and fallback conditions
- [ ] Report per-case 90% status and segment deltas
- [ ] Report correctness/profile results
- **Status:** pending

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Keep all new optimizations pattern/layout driven | User explicitly disallows benchmark-name special cases |
| Prefer conservative fallback | Host aliasing and post-use uncertainty must not change semantics |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|

---

# Task Plan: MPI Translator Remaining Benchmark Optimization

## Goal
Continue optimizing `/Volumes/QUQ/working/dacpp/clang/tools/translator` on branch `tqc-2` without benchmark-name special cases, without reverting existing changes, and without cleaning unrelated dirty/untracked files. Push remaining low-performing benchmarks toward at least 90% of standard MPI+SYCL by adding low-risk generic communication optimizations:
1. aggregate/trivially-copyable constant init local fill,
2. static owner computation for bounded-index post-use sync,
3. prefix/slice post-use materialization for small fixed host reads after `tensor2Array`.

## Current Phase
Phase 1 in progress

## Phases

### Phase 1: Restore Context and Discover Entry Points
- [x] Read existing planning files and catchup context
- [x] Confirm current dirty/untracked files and preserve them
- [x] Inspect current constant-init, bounded post-use, and tensor2Array sync code
- [ ] Inspect target benchmark sources/generated code/profile baselines
- **Status:** in_progress

### Phase 2: Aggregate Constant Init
- [ ] Extend constant-init analysis from arithmetic-only to source-expressible trivially-copyable aggregate values
- [ ] Support `std::vector<T>(size, { ... })` and equivalent aggregate fill constructor forms
- [ ] Keep conservative fallback for non-trivially-copyable, non-expressible, vector mutation/escape, or tensor mutation/escape
- [ ] Verify imageAdjustment initial image/image2 skip root pack/scatter
- **Status:** pending

### Phase 3: Bounded Owner Staticization
- [ ] Replace bounded-index owner discovery `MPI_Allreduce` with static owner computation for Contiguous1D and RowBlock2D/RowPartitionFullRow layouts
- [ ] Preserve full fallback/abort checks for unsupported layouts and out-of-domain indices
- [ ] Verify imageAdjustment final bounded read no longer pays owner Allreduce
- **Status:** pending

### Phase 4: Prefix/Slice Post-Use Materialization
- [ ] Recognize root-side `tensor.tensor2Array(host_vec)` followed by fixed-prefix or fixed-index host uses
- [ ] Generate only needed root values instead of full gather when all subsequent uses are bounded/prefix
- [ ] Preserve full materialization fallback for aliases, escapes, mutation, unknown indices, or non-root/all-rank needs
- [ ] Verify vectorAddCombo only syncs printed first 16 values
- **Status:** pending

### Phase 5: Verification and Profile
- [ ] `cmake --build build --target translator -j8`
- [ ] `bash clang/tools/translator/test_mpi.sh imageAdjustment1.0 vectorAddCombo decay1.0`
- [ ] `bash clang/tools/translator/test_mpi.sh waveEquation1.0 mandel1.0 stencil1.0`
- [ ] Full `bash clang/tools/translator/test_mpi.sh`
- [ ] Requested profile for `imageAdjustment1.0 vectorAddCombo decay1.0 jacobi1.0 FOuLa1.0 mandel1.0`
- **Status:** pending

### Phase 6: Delivery
- [ ] Summarize modified files
- [ ] Summarize supported generic patterns and fallback behavior
- [ ] Report whether each priority case reaches 90%
- [ ] Report segment before/after changes and correctness/profile results
- **Status:** pending

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| No benchmark-name special cases | User explicitly disallows them; all paths must be pattern/layout driven |
| Preserve dirty/untracked files | User explicitly requested no cleanup/revert of unrelated local state |
| Favor conservative fallback | These optimizations must preserve correctness when AST proof is incomplete |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|

---

# Task Plan: Host Post-Use Partial Materialization

## Goal
Generalize MPI translator final synchronization after DAC regions so host-visible uses drive whether distributed outputs/direct readers need no materialization, scalar reduction, bounded indexed root reads, or full gather fallback. Primary validation case is `waveEquation1.0`, whose post-region only prints `matCur[0][0]`; implementation must not special-case benchmark names and must preserve full fallback on uncertainty.

## Current Phase
Phase 6 complete

## Phases

### Phase 1: Discovery
- [x] Restore existing plan/progress/findings context
- [x] Confirm dirty tree and preserve unrelated changes
- [x] Inspect existing OutputSync/PostRegion analysis and codegen
- [x] Inspect OR resident final materialization and generated waveEquation/profile source flow
- **Status:** complete

### Phase 2: Analysis Design
- [x] Define conservative post-use categories: none, bounded-index, scalar reduction, full fallback
- [x] Define AST checks for bounded 1D/2D constant index root reads and alias/function escapes
- [x] Define owner-rank/local-buffer extraction for row-block resident tensors
- **Status:** complete

### Phase 3: Implementation
- [x] Extend shared post-use plan data structures and logs
- [x] Preserve mandel scalar reduction path
- [x] Suppress unused output/direct-reader materialization
- [x] Add bounded indexed root read codegen for row-block/resident tensors
- [x] Ensure uncertain uses fallback full materialization
- **Status:** complete

### Phase 4: Target Verification
- [x] Build translator
- [x] Run target correctness tests
- [x] Inspect generated waveEquation/mandel/imageAdjustment code and logs
- **Status:** complete

### Phase 5: Full Regression and Profile
- [x] Run full `test_mpi.sh`
- [x] Run target profile set
- [x] Run full profile if feasible
- **Status:** complete

### Phase 6: Delivery
- [x] Summarize modified files, sync classifications/fallbacks, bounded-index support, and benchmark deltas
- **Status:** complete

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| No benchmark-name special cases | User explicitly requires generic optimization |
| Fallback full materialization on alias/escape/unknown use | Correctness is higher priority than aggressive communication trimming |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
|       |         |            |
|       |         |            |

## Notes
- Update phase status as you progress: pending → in_progress → complete
- Re-read this plan before major decisions (attention manipulation)

---

# Task Plan: Constant Init Local Fill + Async Halo Runtime

## Goal
Generalize MPI translator communication optimizations without benchmark-name special cases: eliminate initial root scatter for tensors proven to be zero/constant initialized from safe `std::vector` construction, and replace resident halo blocking sendrecv exchanges with nonblocking receive/send/wait patterns while preserving correctness and existing post-use/direct-scatter optimizations.

## Current Phase
Phase 6 complete

## Phases

### Phase 1: Discovery
- [x] Restore current planning context
- [x] Confirm dirty tree and preserve unrelated changes
- [x] Inspect vector/tensor AST analysis and OR input codegen paths
- [x] Inspect resident halo runtime/codegen for 2D rows and 1D
- **Status:** complete

### Phase 2: Constant Initialization Analysis
- [x] Define conservative per-tensor constant-init metadata and fallback reasons
- [x] Match `std::vector<T>(size, constant)` and arithmetic default-zero vector construction
- [x] Prove no vector writes/escapes before tensor construction
- [x] Attach constant-init decision to OR/direct-reader input plans
- **Status:** complete

### Phase 3: Constant Local Fill Codegen
- [x] Emit local `assign/fill` for resident halo/window/direct-reader layouts
- [x] Emit local fill for ordinary OR Contiguous1D/RowBlock2D input scatter where layout is known
- [x] Skip root `tensor2Array`, `scatter_window_2d_rows`, and/or `MPI_Scatterv` only on proven constants
- [x] Add decision/fallback logs
- **Status:** complete

### Phase 4: Async Halo Runtime
- [x] Replace 2D row halo blocking sendrecv pair with posted Irecv/Isend + Waitall
- [x] Handle `MPI_PROC_NULL`, empty ranks, and zero owned rows
- [x] Review/apply same nonblocking pattern to 1D halo if low risk
- [x] Keep profile segment as Halo
- **Status:** complete

### Phase 5: Verification and Profile
- [x] Build translator
- [x] Run requested target correctness tests
- [x] Run full `test_mpi.sh`
- [x] Run target profile set
- [x] Run full profile if feasible
- **Status:** complete

### Phase 6: Delivery
- [x] Summarize files, supported/fallback analysis, async halo details, and benchmark/test/profile deltas
- **Status:** complete

---

# Task Plan: RowBlock2D Scatterv and Post-Use Reduction

## Goal
Continue MPI translator performance optimization without reverting existing dirty changes. Primary target is replacing RowBlock2D full-image broadcast with guarded direct Scatterv for imageAdjustment1.0. Secondary target is conservative post-use reduction for mandel1.0 when the output tensor is only used for root-side `if (tensor[...] == 1) count++` style statistics.

## Current Phase
Phase 4 complete; ready for delivery

## Phases

### Phase 1: Discovery
- [x] Check git status and existing dirty files
- [x] Read prior planning context
- [x] Inspect RowBlock2D scatter/broadcast codegen and generated imageAdjustment code
- [x] Inspect output sync/post-region analysis and mandel source/generated code
- **Status:** complete

### Phase 2: RowBlock2D Direct Scatterv
- [x] Add strict runtime guard for contiguous 2D row-major tensor storage
- [x] Emit byte and non-byte direct Scatterv using tensor storage pointer on root
- [x] Preserve complete fallback to existing pack/scatter or bcast path
- [x] Verify imageAdjustment1.0 correctness
- [x] Verify imageAdjustment1.0 segment profile
- **Status:** complete

### Phase 3: Post-Use Reduction
- [x] Recognize conservative root-side reduction post-use pattern
- [x] Generate distributed local count + MPI_Reduce/Allreduce path
- [x] Skip full output materialization only when no other full tensor use exists
- [x] Verify mandel1.0 correctness
- [x] Verify mandel1.0 segment profile
- **Status:** complete

### Phase 4: Regression
- [x] Build translator
- [x] Run imageAdjustment1.0 and mandel1.0 tests
- [x] Run full test_mpi.sh
- [x] Run target profile and full profile
- **Status:** complete

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Preserve existing RowBlock2D byte fast-path behavior as fallback | User explicitly requested no regression and full fallback on guard failure |
| Use existing checked_mul_int64_or_abort/narrow_mpi_count_or_abort in emitted code | Required for byte count overflow safety |
| Profile post-use reduction as FinalSync | Makes mandel full-output gather disappearance visible while still accounting for MPI_Reduce time |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
