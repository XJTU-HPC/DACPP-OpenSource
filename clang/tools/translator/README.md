# DACPP Translator

`clang/tools/translator` contains the DACPP source-to-source translator and the
release-facing shell wrappers:

- `setenv.sh`: configures paths and defines helper functions for translation and
  compilation.
- `dacpp.sh`: a command-line wrapper that sources `setenv.sh` and exposes
  `translate`, `build`, and `translate-build` commands.

The translator has two paths. Non-MPI translation dispatches to the
single-node translator overlay. Translation with `--mpi` uses the MPI-capable
translator path.

## Build The Translator

From the repository root, build the `translator` target:

```bash
cmake --build build --target translator -j8
```

This target also builds `translator_single`. By default, the wrapper expects the
main executable here:

```text
build/bin/translator/translator
```

If your build tree is somewhere else, set `DACPP_TRANSLATOR`:

```bash
export DACPP_TRANSLATOR=/path/to/build/bin/translator/translator
```

`translator_single` must be in the same directory as `translator`, because the
main executable dispatches to it for non-MPI translation.

## Configure The Environment

For interactive shell use, source the environment script:

```bash
source clang/tools/translator/setenv.sh
```

You do not need to source it before using `dacpp.sh`; the wrapper sources it
automatically. Sourcing it is useful when you want to call helper functions such
as `dacpp` or `dacpp-compile` directly.

Common environment variables:

| Variable | Purpose |
| --- | --- |
| `DACPP_TRANSLATOR` | Path to the `translator` executable. Defaults to `build/bin/translator/translator` relative to this directory. |
| `DACPP_SYCL_COMPILER` | Build backend: `auto`, `adaptivecpp`, `icpx`, or `mpicxx`. Default is `auto`. |
| `ACPP_ROOT` / `ADAPTIVECPP_ROOT` | AdaptiveCpp install prefix. The compiler is expected at `$ACPP_ROOT/bin/acpp`. |
| `ONEAPI_SETVARS` | Full path to a oneAPI `setvars.sh` file. Used when `icpx` is not already on `PATH`. |
| `ONEAPI_ROOT` / `INTEL_ONEAPI_ROOT` | oneAPI install prefix. The setup script is expected at `$ONEAPI_ROOT/setvars.sh`. |
| `ICPX` | Explicit path to `icpx`. |
| `MPICXX` | Explicit path to `mpicxx`. |
| `DACPP_TMP_ROOT` | Default directory for compiled binaries when no output path is passed. Defaults to `${TMPDIR:-/tmp}`. |

When `DACPP_SYCL_COMPILER=auto`, the build wrapper picks the first available
backend in this order:

1. AdaptiveCpp, if `ACPP_ROOT` or `ADAPTIVECPP_ROOT` points to a valid install.
2. oneAPI `icpx`, if `icpx` is on `PATH` or oneAPI is loaded from the variables above.
3. `mpicxx`, if an MPI C++ wrapper is on `PATH` or `MPICXX` is set.

For oneAPI MPI wrappers, `setenv.sh` sets `I_MPI_CXX` and `OMPI_CXX` to `icpx`
when `icpx` is found and those variables are not already set.

## Translate Source Files

Use `dacpp.sh translate`:

```bash
clang/tools/translator/dacpp.sh translate /path/to/file.dac.cpp --mode=buffer
```

The translator writes the generated C++/SYCL file next to the input source. It
does not currently provide a separate output-source path option.

Default generated source names:

| Input source | Generated source |
| --- | --- |
| `/path/to/file.dac.cpp` | `/path/to/file.dac_sycl_buffer.cpp` |
| `/path/to/file.mpi.dac.cpp` | `/path/to/file.mpi.dac_sycl_buffer.cpp` |

Re-running translation overwrites the same generated file. To choose another
directory for the generated source, place or copy the input source in that
directory before translation, or move the generated file after translation.

Pass Clang/compiler arguments after `--`:

```bash
clang/tools/translator/dacpp.sh translate /path/to/file.dac.cpp --mode=buffer -- \
  -I/path/to/includes -DMY_DEFINE=1
```

Translator options must appear before `--`. For example, `--mpi` after `--` is
treated as a compiler argument, not as a request for MPI translation.

## MPI Translation

Add `--mpi` before the `--` separator:

```bash
clang/tools/translator/dacpp.sh translate /path/to/file.mpi.dac.cpp --mode=buffer --mpi
```

Optional MPI output synchronization mode:

```bash
clang/tools/translator/dacpp.sh translate /path/to/file.mpi.dac.cpp \
  --mode=buffer --mpi --mpi-output-sync=root-only
```

Supported values are:

- `all-ranks`: gather output on root and broadcast updated outputs back to every
  rank. This is the default.
- `root-only`: gather output on root only and skip the final broadcast.

## Build Generated Code

Use `dacpp.sh build`:

```bash
clang/tools/translator/dacpp.sh build /path/to/file.dac_sycl_buffer.cpp /path/to/app
```

If the output binary path is omitted, the default is:

```text
$DACPP_TMP_ROOT/<generated-file-basename-without-.cpp>
```

For example:

```bash
export DACPP_TMP_ROOT=/tmp/dacpp-build
clang/tools/translator/dacpp.sh build /work/file.dac_sycl_buffer.cpp
```

creates:

```text
/tmp/dacpp-build/file.dac_sycl_buffer
```

### AdaptiveCpp

```bash
ADAPTIVECPP_ROOT=/path/to/adaptivecpp \
DACPP_SYCL_COMPILER=adaptivecpp \
  clang/tools/translator/dacpp.sh build /path/to/file.dac_sycl_buffer.cpp /tmp/file_app
```

`ACPP_ROOT` may be used instead of `ADAPTIVECPP_ROOT`.

### oneAPI icpx

```bash
ONEAPI_ROOT=/path/to/oneapi \
DACPP_SYCL_COMPILER=icpx \
  clang/tools/translator/dacpp.sh build /path/to/file.dac_sycl_buffer.cpp /tmp/file_app
```

If oneAPI is already loaded and `icpx` is on `PATH`, only
`DACPP_SYCL_COMPILER=icpx` is needed. You can also set `ICPX=/path/to/icpx`.

### MPI Wrapper

```bash
DACPP_SYCL_COMPILER=mpicxx \
  clang/tools/translator/dacpp.sh build /path/to/file.mpi.dac_sycl_buffer.cpp /tmp/file_mpi_app
```

If `mpicxx` is not on `PATH`, set `MPICXX=/path/to/mpicxx`.

Run the compiled MPI binary with your MPI launcher:

```bash
mpirun -np 4 /tmp/file_mpi_app
```

## Translate And Build In One Step

`translate-build` translates in buffer mode and then compiles the generated file:

```bash
clang/tools/translator/dacpp.sh translate-build /path/to/file.dac.cpp /tmp/file_app
```

Without an explicit output binary, it uses the same `DACPP_TMP_ROOT` rule as
`build`:

```bash
clang/tools/translator/dacpp.sh translate-build /path/to/file.dac.cpp
```

For MPI:

```bash
DACPP_SYCL_COMPILER=mpicxx \
  clang/tools/translator/dacpp.sh translate-build /path/to/file.mpi.dac.cpp /tmp/file_mpi_app --mpi
```

The generated source is still written next to the input source; the optional
second argument controls only the compiled binary path.

## Command Summary

```text
dacpp.sh translate <source.dac.cpp> [translator-options] [-- compiler-options]
dacpp.sh build <generated.cpp> [output-bin]
dacpp.sh translate-build <source.dac.cpp> [output-bin] [translator-options]
```

Common translator options:

```text
--mode=buffer
--mpi
--mpi-output-sync=all-ranks
--mpi-output-sync=root-only
```

For the current release, non-MPI translation supports buffer mode only.
