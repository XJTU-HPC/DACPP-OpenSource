# P4.6 Closeout Task Plan

## Objective

Close Phase 4.6 for the currently proven operator-resident stencil slice, keep all gates
conservative, and hand off the next-stage boundary cleanly.

## Completed

| Item | Status | Notes |
|---|---|---|
| 1D resident-state rotation | complete | `MDP1.0` and `liuliang1.0` now use full next-state write plus in-place halo exchange and reader/writer swap. |
| 2D resident-state rotation | complete | `stencil1.0` uses double-buffer rotation; `waveEquation1.0` uses triple-buffer rotation. |
| B3 soundness tightening | complete | Current 2D direct-reader/read-cache slice now requires `DAC -> read-cache -> followup -> boundary` source order. |
| Required regression coverage | complete | Requested serial `test_mpi.sh` groups passed, including resident-halo, empty-rank, right-boundary fallback, scalar reject, count guard, and order reject. |
| Focused benchmark closeout | complete | Results recorded in `/Volumes/QUQ/working/mpi_tmp/p46_final_close/results.tsv`. |
| Canonical docs and tracking sync | complete | P4.6 and shell-derived docs rewritten into current-state closeout form; tracking files reduced to the active findings and handoff. |

## Deferred On Purpose

| Item | Why it stays deferred |
|---|---|
| `FOuLa1.0` loop-site acceptance | Needs rewrite/init contract expansion for loop-local shell args; this is not a gate-only change and should not be hidden inside the P4.6 closeout patch. |
| `oddeven0.1` and broader fixed-pair exchange work | Phase 5 `FixedBlock`, outside current P4.6 scope. |
| Broader direct-reader/cache forms, root-bridge, dynamic-shape expansion | Beyond the currently proven resident slice. |

## Recommended Next Order

1. Continue with Phase 5.
2. Continue with Phase 6.
3. Revisit `FOuLa1.0` and any broader efficiency work as a separate targeted slice with
   benchmark checkpoints.
