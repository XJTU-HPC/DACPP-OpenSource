# DACPP to SYCL Source-to-Source Translator

## Build

```shell
git clone --branch test git@github.com:XJTU-HPC/dacpp.git
cd dacpp
cmake -S llvm -B build -G Ninja \
  -DLLVM_ENABLE_PROJECTS="clang" \
  -DCMAKE_INSTALL_PREFIX=/data/user_name/clang \
  -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j8 --target translator
```

The translator binary is generated at `dacpp/build/bin/translator/translator`.

## Active Code Path

If you want to understand the current implementation, read these files first:

- `translator.cpp`: Clang tool entrypoint and mode dispatch
- `parser/lib/DacppStructure.cpp`: per-file state and `<->` extraction
- `parser/lib/Shell.cpp`: `split/index/binding()` parsing
- `rewriter/include/Rewriter.h`: `rewriteMain()` and shared rewriter interface
- `rewriter/lib/Rewriter_Buffer_new.cpp`: active buffer baseline rewriter
- `rewriter/lib/buffer_template_new.cpp`: active buffer template generation
- `rewriter/lib/Rewriter_MPI.cpp`: active MPI codegen path
- `dpcppLib/include/MPIPlanner.h`: MPI packing/binding runtime planner

## Archive

Legacy backups and superseded implementations were moved into `archive/`.
That directory is for historical reference only and is not part of the active
build or the recommended reading order.

## Directory Layout

```text
translator/
├── archive/        # legacy backups and superseded implementations
├── dacppLib/       # DACPP tensor/runtime-side support
├── dpcppLib/       # generated-code runtime support, including MPI planner
├── docs/           # handoff notes, bug analyses, setup notes
├── parser/         # AST parsing for shell/calc/binding metadata
├── rewriter/       # active code generation paths
├── tests/          # translator test inputs and generated reference outputs
├── env.sh          # local helper functions for translate/build/run
├── test_mpi.sh     # non-stencil MPI regression script
├── translator.cpp  # main entrypoint
└── CMakeLists.txt
```
