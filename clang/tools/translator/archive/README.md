# Translator Archive

This directory stores legacy or superseded files that are intentionally kept
out of the active `translator` code path.

## What Was Archived

- explicit `backup/` trees under `dacppLib/`, `dpcppLib/`, and `rewriter/`
- obvious old/copy headers that are not part of the active include chain
- the legacy buffer rewriter/template pair replaced by
  `Rewriter_Buffer_new.cpp` and `buffer_template_new.cpp`
- the legacy template-based MPI generator pair was deleted after the split-out
  MPI path became the only active code path
- the legacy monolithic MPI wrapper/region generators were deleted after the
  split-out MPI path became the only active code path:
  `rewriter/lib/mpi/Rewriter_MPI_Wrapper_Codegen.cpp`,
  `rewriter/lib/mpi/Rewriter_MPI_Region_Codegen.cpp`, and their helper files
- a stale generated backup test artifact

## What To Read First Instead

If you are trying to understand the current translator behavior, start from:

- `translator.cpp`
- `parser/lib/DacppStructure.cpp`
- `parser/lib/Shell.cpp`
- `rewriter/include/Rewriter.h`
- `rewriter/lib/Rewriter_Buffer_new.cpp`
- `rewriter/lib/buffer_template_new.cpp`
- `rewriter/lib/Rewriter_MPI.cpp`
- `dpcppLib/include/MPIPlanner.h`
- `dpcppLib/include/mpi/`

Only look in `archive/` if you need historical context or want to recover an
older implementation.
