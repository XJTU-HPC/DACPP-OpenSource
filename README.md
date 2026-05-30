# DACPP

DACPP is a source-to-source translation project built on top of LLVM/Clang. It
translates DACPP C++ source files into C++/SYCL code, with separate paths for
single-node translation and MPI-enabled translation.

The main translator lives in:

```text
clang/tools/translator
```

The repository is based on the LLVM monorepo layout. Most upstream LLVM
components are kept in place because the translator is built as a Clang tool.

## Features

- Translates DACPP source files to SYCL buffer-mode C++.
- Uses a single-node translator path for normal non-MPI translation.
- Uses an MPI-capable translator path when `--mpi` is requested.
- Provides wrapper scripts for translation, compilation, and one-step
  translate-and-build workflows.
- Supports building generated SYCL code with AdaptiveCpp, Intel oneAPI `icpx`,
  or an MPI C++ wrapper such as `mpicxx`.

## Dependencies

### Build Dependencies

To build the translator from source, install:

- CMake 3.20 or newer.
- Ninja, or another CMake-supported build tool.
- A C++17-capable host compiler.
- Python 3, required by the LLVM/Clang build system.
- Git.

On macOS, install Xcode Command Line Tools:

```bash
xcode-select --install
```

### Runtime And Compilation Dependencies

To compile translated C++/SYCL output, install at least one SYCL compiler:

- AdaptiveCpp, configured with `ACPP_ROOT` or `ADAPTIVECPP_ROOT`.
- Intel oneAPI DPC++/C++ Compiler, available as `icpx`.
- An MPI C++ wrapper such as `mpicxx` when compiling MPI output.

For MPI translation and execution, install an MPI implementation such as
OpenMPI, MPICH, or Intel MPI. On macOS with Homebrew, `open-mpi` and `libomp`
are detected automatically when present.

## Build

Configure the LLVM/Clang build from the repository root:

```bash
cmake -S llvm -B build -G Ninja \
  -DLLVM_ENABLE_PROJECTS="clang" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DCMAKE_BUILD_TYPE=Release
```

Then build the DACPP translator:

```bash
cmake --build build --target translator -j8
```

This builds both executables:

```text
build/bin/translator/translator
build/bin/translator/translator_single
```

`translator_single` must stay in the same directory as `translator`, because the
main executable dispatches to it for non-MPI translation.

If your build directory is not `build`, set `DACPP_TRANSLATOR` before using the
wrapper scripts:

```bash
export DACPP_TRANSLATOR=/path/to/build/bin/translator/translator
```

## Environment

The public scripts are in `clang/tools/translator`:

```text
clang/tools/translator/setenv.sh
clang/tools/translator/dacpp.sh
```

For interactive use, source the environment:

```bash
source clang/tools/translator/setenv.sh
```

`dacpp.sh` sources `setenv.sh` automatically, so sourcing is optional when using
the wrapper directly.

Common environment variables:

| Variable | Purpose |
| --- | --- |
| `DACPP_TRANSLATOR` | Path to the built `translator` executable. |
| `DACPP_SYCL_COMPILER` | Build backend: `auto`, `adaptivecpp`, `icpx`, or `mpicxx`. Default is `auto`. |
| `ACPP_ROOT` / `ADAPTIVECPP_ROOT` | AdaptiveCpp installation prefix. |
| `ONEAPI_SETVARS` | Full path to oneAPI `setvars.sh`. |
| `ONEAPI_ROOT` / `INTEL_ONEAPI_ROOT` | oneAPI installation prefix. |
| `ICPX` | Explicit path to `icpx`. |
| `MPICXX` | Explicit path to `mpicxx`. |
| `DACPP_TMP_ROOT` | Default directory for compiled binaries. Defaults to `${TMPDIR:-/tmp}`. |

In `auto` mode, the compiler backend is selected in this order:

1. AdaptiveCpp.
2. oneAPI `icpx`.
3. `mpicxx`.

## Translate

Translate a non-MPI DACPP source file:

```bash
clang/tools/translator/dacpp.sh translate /path/to/file.dac.cpp --mode=buffer
```

Translate with MPI enabled:

```bash
clang/tools/translator/dacpp.sh translate /path/to/file.mpi.dac.cpp --mode=buffer --mpi
```

Pass compiler include paths and definitions after `--`:

```bash
clang/tools/translator/dacpp.sh translate /path/to/file.dac.cpp --mode=buffer -- \
  -I/path/to/include -DMY_DEFINE=1
```

Translator options, including `--mpi`, must appear before `--`.

Generated source files are written next to the input source file:

| Input | Generated output |
| --- | --- |
| `/path/to/file.dac.cpp` | `/path/to/file.dac_sycl_buffer.cpp` |
| `/path/to/file.mpi.dac.cpp` | `/path/to/file.mpi.dac_sycl_buffer.cpp` |

The translator does not currently provide a separate output-source path option.
To place generated source in another directory, put the input source there
before translation or move the generated file afterward.

## Compile Translated Code

Compile generated code with the automatically detected backend:

```bash
clang/tools/translator/dacpp.sh build /path/to/file.dac_sycl_buffer.cpp /tmp/file_app
```

Compile with AdaptiveCpp:

```bash
ADAPTIVECPP_ROOT=/path/to/adaptivecpp \
DACPP_SYCL_COMPILER=adaptivecpp \
  clang/tools/translator/dacpp.sh build /path/to/file.dac_sycl_buffer.cpp /tmp/file_app
```

Compile with oneAPI `icpx`:

```bash
ONEAPI_ROOT=/path/to/oneapi \
DACPP_SYCL_COMPILER=icpx \
  clang/tools/translator/dacpp.sh build /path/to/file.dac_sycl_buffer.cpp /tmp/file_app
```

Compile MPI output with `mpicxx`:

```bash
DACPP_SYCL_COMPILER=mpicxx \
  clang/tools/translator/dacpp.sh build /path/to/file.mpi.dac_sycl_buffer.cpp /tmp/file_mpi_app
```

If the output binary path is omitted, the wrapper writes it to:

```text
$DACPP_TMP_ROOT/<generated-file-basename-without-.cpp>
```

## Translate And Build

For a one-step workflow:

```bash
clang/tools/translator/dacpp.sh translate-build /path/to/file.dac.cpp /tmp/file_app
```

For MPI:

```bash
DACPP_SYCL_COMPILER=mpicxx \
  clang/tools/translator/dacpp.sh translate-build /path/to/file.mpi.dac.cpp /tmp/file_mpi_app --mpi
```

Run MPI binaries with your MPI launcher:

```bash
mpirun -np 4 /tmp/file_mpi_app
```

## More Documentation

See the translator-specific README for detailed script behavior and command
reference:

```text
clang/tools/translator/README.md
```

Upstream LLVM documentation is still present in component subdirectories because
DACPP is built on the LLVM/Clang source tree.

## License

This repository follows the LLVM project license. See `LICENSE.TXT`.
