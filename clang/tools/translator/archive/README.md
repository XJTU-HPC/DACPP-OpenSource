# Translator Archive

This directory stores legacy or superseded files that are intentionally kept
out of the active `translator` code path.

## What Was Archived

- explicit `backup/` trees under `dacppLib/`, `dpcppLib/`, and `rewriter/`
- obvious old/copy headers that are not part of the active include chain
- the legacy buffer rewriter/template pair replaced by
  `Rewriter_Buffer_new.cpp` and `buffer_template_new.cpp`
- the legacy template-based MPI generator pair
  `archive/rewriter/include/mpi_template.h` and
  `archive/rewriter/lib/mpi_template.cpp`
- the legacy monolithic MPI wrapper
  `archive/rewriter/lib/Rewriter_MPI_Wrapper.cpp`, replaced by the split-out
  `Rewriter_MPI_Wrapper_Codegen.cpp` (wrapper codegen),
  `Rewriter_MPI_Analysis.cpp` (AST analysis), and shared utilities kept in
  the active `Rewriter_MPI_Wrapper.cpp` under `rewriter/lib/mpi/`
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

Only look in `archive/` if you need historical context or want to recover an
older implementation.
