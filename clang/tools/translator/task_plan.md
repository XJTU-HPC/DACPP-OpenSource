# Task Plan: MPI Translator Non-Specialized Performance Profiling

## Goal
Start the non-specialized MPI translator performance pass by adding segmented profiling that is disabled by default and gated by `DACPP_MPI_PROFILE=1`, without expanding accepted surfaces or weakening correctness guards.

## Phases
| Phase | Status | Notes |
|-------|--------|-------|
| 1. Restore context and constraints | complete | Read canonical status, benchmark summary, and optimization TODO; confirmed branch `tqc-2` at `96118823800ca66c45049e9bb879fa866bea9fd7`. |
| 2. Define reusable profiling API | complete | Added `SegmentedProfile`, fixed segment taxonomy, TSV output, and gated timer helpers in `Profile.h`. |
| 3. Instrument legacy AccessPattern path | complete | Legacy wrappers now time init/pack/scatter/kernel/gather/bcast/materialize/final_sync plus existing collect-positions profile. |
| 4. Instrument OR/stencil/FixedBlock/FOuLa paths | complete | Added profiling to ordinary OR wrappers, direct loop-lowered OR, 1D/2D stencil resident/full-sync helpers, P5 phase exchange, and FOuLa owner loop. |
| 5. Structure tests and runtime smoke | complete | Updated representative `mpi_expect.txt` checks and smoke-tested `DACPP_MPI_PROFILE=1` output for OR, loop-resident, P5, FOuLa, and legacy. |
| 6. Build and validation | complete | Build, focused MPI tests, profile smoke, and `git diff --check -- clang/tools/translator` passed. |
| 7. Phase 1.4 benchmark reporting | complete | Added `bench_mpi_profile_segments.py`; collected 3-trial rank-scaling profile artifacts for six representative application cases plus a legacy fallback smoke artifact. |

## Correctness Gates
- Do not change accepted surface.
- Do not weaken any guard or fallback.
- Profiling must be quiet unless `DACPP_MPI_PROFILE=1`.
- Structure tests should verify the generated profiling calls, while normal output comparison remains unchanged.

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| Large initial patch to 1D stencil did not apply cleanly | One broad patch | Split into small per-function patches. |
| Build log was warning-heavy and hid the actual error | First build failed | Filtered for `FAILED/error`; missing include for profile helper declarations was fixed. |
| `mpi_expect` checked old local variable name after profile aggregation moved into ctx | Focused tests failed for MDP/stencil/P5 structure | Updated expectations to check `__or_profile`/`profile` fields. |
| Profile benchmark snapshot copy failed with `SameFileError` | First `vectorAddCombo` script smoke | Skipped copying when generated source is already at the requested snapshot path. |
| Hand MPI+SYCL stdout and DAC stdout differ under the benchmark driver | Direct artifact comparison check | Treated `bench_mpi_profile_segments.py` like `bench_mpi_only_requested.py`: it is a performance driver, not a correctness oracle; correctness remains covered by `test_mpi.sh`, and the profile driver checks DAC profile-off vs profile-on stdout stability. |
