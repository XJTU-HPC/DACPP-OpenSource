# MPI Translator P4 Plan

Updated: 2026-05-11

Goal: turn the strict `FOuLa1.0` owner-loop specialization into a named,
testable, reusable loop-local stencil owner-loop contract without widening the
accepted surface until the AST/contract proof supports it.

## Scope

- Extract the current FOuLa owner-loop proof and codegen boundary from
  `Rewriter_MPI.cpp`.
- Describe the current accepted shape as a `LoopLocalStencilOwnerLoop`
  lowering contract.
- Keep generated-code behavior equivalent for `FOuLa1.0`.
- Add reject fixtures for near-miss owner-loop shapes.
- Preserve all existing guard strictness and output visibility semantics.

## P4 Checklist

| Step | Status | Notes |
|---|---|---|
| Read current status, TODO, benchmark, plan, findings, and progress docs | done | Completed at session start in the requested order. |
| Locate current FOuLa owner-loop detection/codegen | done | Found in `rewriter/lib/Rewriter_MPI.cpp`: local struct, regex-heavy detection, and inline owner-loop codegen. |
| Inventory current proof dependencies | done | Documented AST/source-text checks, reject behavior, and contract semantics in `findings.md`. |
| Extract codegen/detection into focused operator-resident module | done | Behavior-preserving move to `LoopLocalStencilOwnerLoop`; no accepted shape expansion. |
| Add contract logging/checks for current FOuLa shape | done | Accepted logs record replace-loop source, removed roles, resident count, loop-exit materialization, compile/runtime guards. |
| Add reject fixtures | done | Added wrong owner slice, variant scalar payload, missing post-writeback, extra owner mutation, multiple writer slices. |
| Run focused validation | done | Build passed; owner-loop focused set passed; requested stencil/FOuLa set plus P4 rejects passed 11/11. |
| Run full MPI suite if blast radius requires it | done | Full MPI suite passed 67/67 after adding five P4 fixtures. |
| Update planning/docs with closeout | done | Current status/TODO/planning files updated; final build and whitespace checks passed. |

## Contract Boundary

The current contract is loop-local owner-loop replacement, not generic
`ctx/init/run/materialize` stencil lowering. It replaces the outer source loop
with one generated owner-loop function call. The generated function owns the
resident local stencil state, broadcasts boundary scalars per step, exchanges
neighbor halos, gathers owned history once, and materializes the owner matrix
on root at loop exit.

The current accepted shape must stay narrow:

- exactly one `StencilWindow1D` reader, one WRITE-only direct writer, and one
  replicated scalar reader
- all calc parameter element types must match and use native MPI transport
- the enclosing `for` loop must have a simple induction variable
- reader slice: `reader = owner[{}][k]`
- writer slice: `writer = owner[{1,...}][k+1]`
- scalar shell argument comes from a loop-local vector with one payload
  expression
- post-DAC writeback copies writer slice back to `owner[*][k+1]`
- current FOuLa seven top-level statements remain the only accepted body shape

## Correctness Rules

- Do not weaken guards to make reject fixtures pass through the optimized path.
- Do not add legacy fast paths or widen generic P4.6 stencil surface.
- Do not substitute root-only visibility for all-ranks behavior unless an
  explicit output-sync proof controls it.
- Rejected owner-loop candidates must continue through the current correct
  fallback path.

## Validation Commands

```bash
cd /Volumes/QUQ/working/dacpp
cmake --build build --target translator -j8

cd /Volumes/QUQ/working/dacpp/clang/tools/translator
bash test_mpi.sh FOuLa1.0 MDP1.0 liuliang1.0 stencil1.0 waveEquation1.0 mpiOrStencilRefreshPolicy1D

cd /Volumes/QUQ/working/dacpp
git diff --check -- clang/tools/translator
```

Because this task changes `Rewriter_MPI.cpp`/operator-resident structure, run
the full MPI suite before closeout:

```bash
cd /Volumes/QUQ/working/dacpp/clang/tools/translator
bash test_mpi.sh
```

## Parked For Later P4.x

- Broaden accepted scalar construction forms after AST proof exists.
- Parameterize owner-loop codegen beyond the current FOuLa window width and
  boundary policy.
- Add a non-FOuLa accepted owner-loop fixture only after the contract no longer
  relies on current seven-statement source text.

## P4.x AST/Contract Follow-up

Status on 2026-05-11: extended for stricter AST/contract proof without
accepted-surface expansion.

Goal: strengthen `LoopLocalStencilOwnerLoop` proof and contract diagnostics
without widening the accepted owner-loop shape.

| Step | Status | Notes |
|---|---|---|
| Replace whole-loop regex proof with local AST proof where practical | done | Added AST checks for outer loop shape, top-level statement roles, DAC expression position, reader/writer owner slices, scalar vector construction, post-writeback loop, multiple writer slices, and extra owner mutation. |
| Keep conservative source-text fallback only where still needed | done | Scalar payload source is still extracted as text and checked with the existing strict token-like rule; this avoids accepting broader expressions. |
| Add owner-loop contract consistency log | done | Accepted FOuLa logs `contract-check=pass reason=owner-loop-contract-consistent`. |
| Preserve accepted surface and reject reason stability | done | No new accepted shape added; five P4 reject fixtures keep their owner-loop rejection reasons and no owner-loop codegen. |
| Run required validation | done | Build passed; focused 11-case run passed; full MPI suite passed 67/67; whitespace check passed. |
| Close accidental acceptance gaps | done | Added AST checks for owner matrix shape, owner loop bound, exact writer range end, exact scalar shell vector argument, and exact writeback RHS/index. |
| Add stricter reject fixtures | done | Added scalar payload expression, scalar shell argument, loop-bound, writer-range, and writeback-index rejects; all remain OR fallback/no owner-loop. |

P4.x remains whole-loop replacement, not the generic P4.6/P6
`ctx/init/run/materialize` contract. The new consistency check is local to
`LoopLocalStencilOwnerLoop` and does not participate in resident-halo proof.

Validation completed:

- `cmake --build build --target translator -j8`: passed.
- Focused requested run passed:
  original 11-case run passed, and the tightened 16-case focused run passed
  `16 tests | 16 passed | 0 failed | 0 skipped`.
- Full MPI suite passed:
  previous run passed `67 tests | 67 passed | 0 failed | 0 skipped`; after
  adding five stricter owner-loop rejects, the full suite passed
  `72 tests | 72 passed | 0 failed | 0 skipped`.
- `git diff --check -- clang/tools/translator`: passed.

Benchmarks were not run for P4.x because this slice only changes AST proof and
contract diagnostics. It does not intentionally change accepted surface,
generated communication, or performance behavior.

## P4 Closeout

This P4 slice completed contract/refactor/test coverage only. It did not widen
the accepted owner-loop shape.

Validation completed:

- `cmake --build build --target translator -j8`: passed.
- Focused owner-loop validation:
  `FOuLa1.0` plus five P4 reject fixtures passed 6/6.
- User-requested focused stencil/FOuLa validation plus P4 rejects passed 11/11.
- Full MPI suite passed 67/67.
- `git diff --check -- clang/tools/translator`: passed.

Benchmarks were not run because this slice is structural/proof/code
organization work and does not intentionally change generated communication or
accepted performance surface. Existing FOuLa generated behavior remained
behaviorally equivalent under correctness tests.
