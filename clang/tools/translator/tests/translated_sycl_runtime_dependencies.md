# Translated SYCL Program Runtime Dependencies

## Scope

This note explains what extra files or runtime components a translated SYCL
program needs in order to compile and run after `dacpp` generates a
`*.dac_sycl_buffer.cpp`, `*.dac_sycl_usm.cpp`, or `*.mpi.dac_sycl_buffer.cpp`
file.

It is based on the current translator mainline in this repository and the
actual generated files under `tests/` and `/Volumes/QUQ/working/mpi_tmp`.

## Short Answer

There are two different dependency layers:

1. Files needed to compile the generated C++ source
2. Shared libraries needed to run the compiled executable

For this repository:

- The translator support code is mostly header-only.
- After the executable is built, you do not need to ship these repo headers
  next to the binary anymore.
- At runtime, the important external dependencies are usually:
  - AdaptiveCpp runtime libraries
  - OpenMP runtime library
  - MPI runtime library for MPI builds

## 1. Files Needed To Compile Generated SYCL Source

### 1.1 Common repo headers

Current generated SYCL files typically include these repo headers:

- `dacppLib/include/ReconTensor.h`
- `dpcppLib/include/ParameterGeneration.h`

Most current non-block generated outputs also include:

- `dpcppLib/include/DataReconstructor1.h`

If the translator takes the block/region-style path, the generated file may
instead include:

- `dpcppLib/include/DataReconstructor.new.h`
- `dpcppLib/include/utils.h`

MPI-generated outputs additionally include:

- `dpcppLib/include/MPIPlanner.h`
- the subheaders pulled in by `MPIPlanner.h` from `dpcppLib/include/mpi/`

Because `MPIPlanner.h` transitively includes `dacInfo.h`, MPI builds also rely
on:

- `rewriter/include/dacInfo.h`

### 1.2 Required include directories

If you compile the generated `.cpp` outside the helper script, you need these
include roots available:

- `dacppLib/include`
- `dpcppLib/include`
- `rewriter/include`

### 1.3 External toolchain headers

Generated code also includes external headers such as:

- `sycl/sycl.hpp`
- `mpi.h` for `--mpi` output

So the machine used for compilation must have:

- an AdaptiveCpp installation
- an MPI development installation for MPI outputs
- OpenMP headers if your compiler setup needs them

## 2. Shared Libraries Needed At Runtime

After the executable is compiled, the repo headers above are no longer runtime
dependencies. The runtime dependencies are the shared libraries linked into the
binary.

On the current macOS setup in this repo, `otool -L` shows these typical
dependencies:

- `@rpath/libacpp-rt.dylib`
- `@rpath/libacpp-common.dylib`
- `/opt/homebrew/opt/libomp/lib/libomp.dylib`
- `/opt/homebrew/opt/open-mpi/lib/libmpi.40.dylib`

System libraries like `libc++` and `libSystem` are provided by the OS and are
usually not something you need to package yourself.

### 2.1 AdaptiveCpp runtime

This is the main mandatory runtime dependency for translated SYCL executables.

In this repo, the default install location is:

- `/Volumes/QUQ/working/sycl-install`

The important runtime directory is:

- `$ACPP_ROOT/lib`

At minimum, the executable needs the AdaptiveCpp runtime libraries available at
launch time, especially:

- `libacpp-rt.dylib`
- `libacpp-common.dylib`

### 2.2 OpenMP runtime

Current builds also link against `libomp`.

So the runtime machine must also be able to locate:

- `libomp.dylib` on macOS
- `libomp.so` on Linux

### 2.3 MPI runtime

For MPI-generated programs, `libmpi` is required at runtime.

In this repo's current helper environment, even some non-MPI binaries may still
end up linked with OpenMPI, because `env.sh` adds OpenMPI link flags whenever it
detects an OpenMPI installation. So if you use the repo helper scripts as-is,
seeing `libmpi` in a non-MPI binary is not necessarily a bug.

## 3. Environment Variables Usually Needed To Run

### macOS

Usually you need:

```bash
export ACPP_ROOT=/Volumes/QUQ/working/sycl-install
export DYLD_LIBRARY_PATH="$ACPP_ROOT/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
```

If OpenMPI or `libomp` are not already in a loader-visible location, you may
also need to append their library directories to `DYLD_LIBRARY_PATH`.

### Linux

Usually you need:

```bash
export ACPP_ROOT=/path/to/sycl-install
export LD_LIBRARY_PATH="$ACPP_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

If MPI or `libomp` are installed in non-standard locations, add those library
directories as well.

## 4. What Is Needed For Normal SYCL vs MPI SYCL

### 4.1 Non-MPI translated SYCL program

If you only want to run a normal translated SYCL executable:

- compile-time:
  - `ReconTensor.h`
  - `ParameterGeneration.h`
  - usually `DataReconstructor1.h`
  - AdaptiveCpp headers
- runtime:
  - AdaptiveCpp shared libraries
  - OpenMP shared library
  - maybe `libmpi` too if you built it with this repo's current `env.sh`

### 4.2 MPI translated SYCL program

If you want to run a `--mpi` translated executable:

- compile-time:
  - everything above
  - `MPIPlanner.h`
  - `dpcppLib/include/mpi/`
  - MPI headers
  - `rewriter/include/dacInfo.h`
- runtime:
  - AdaptiveCpp shared libraries
  - OpenMP shared library
  - MPI shared library
  - `mpirun` or equivalent MPI launcher

## 5. Fast Checklist

If you want to move a generated source or binary to another machine, check:

- Is AdaptiveCpp installed there, and is `sycl/sycl.hpp` available?
- Are these include dirs available for compilation?
  - `dacppLib/include`
  - `dpcppLib/include`
  - `rewriter/include`
- Does the generated file include `MPIPlanner.h`?
  - If yes, also carry `dpcppLib/include/mpi/` and MPI headers/libs
- Does the runtime machine have:
  - `libacpp-rt`
  - `libacpp-common`
  - `libomp`
  - `libmpi` for MPI builds
- Is `DYLD_LIBRARY_PATH` or `LD_LIBRARY_PATH` configured so the loader can find
  those libraries?

## 6. Practical Commands In This Repo

Compile:

```bash
source /Volumes/QUQ/working/dacpp/clang/tools/translator/env.sh
acpp-compile /path/to/file.dac_sycl_buffer.cpp /tmp/my_bin
```

Run non-MPI:

```bash
DYLD_LIBRARY_PATH="$ACPP_ROOT/lib" /tmp/my_bin
```

Run MPI:

```bash
DYLD_LIBRARY_PATH="$ACPP_ROOT/lib" mpirun -np 4 /tmp/my_bin
```

## 7. Important Clarification

If the question is strictly "what must exist when I run the already-built
binary", the answer is:

- you do not need the repo headers anymore
- you do need the linked shared libraries, mainly AdaptiveCpp, `libomp`, and
  MPI for MPI builds

If the question is "what must I carry to another place so I can recompile the
generated `.cpp` there", then you need both:

- the repo runtime headers listed above
- the external SYCL/MPI toolchain and libraries
