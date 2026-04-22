# 翻译后 SYCL 程序的运行时依赖

## 范围

本文档说明了在 `dacpp` 生成 `*.dac_sycl_buffer.cpp`、`*.dac_sycl_usm.cpp` 或 `*.mpi.dac_sycl_buffer.cpp` 文件后，翻译后的 SYCL 程序在编译和运行时所需的额外文件或运行时组件。

内容基于本仓库中翻译器的主线代码，以及 `tests/` 和 `/Volumes/QUQ/working/mpi_tmp` 目录下的实际生成文件。

## 简要说明

存在两个不同的依赖层：

1. 编译生成的 C++ 源代码所需的文件
2. 运行编译后可执行文件所需的共享库

在本仓库中：

- 翻译器的支持代码大部分是仅头文件（header-only）形式。
- 可执行文件构建完成后，不再需要将这些仓库头文件与二进制文件一起分发。
- 运行时，主要的外部依赖通常是：
  - AdaptiveCpp 运行时库
  - OpenMP 运行时库
  - MPI 构建时需要的 MPI 运行时库

## 1. 编译生成的 SYCL 源代码所需的文件

### 1.1 仓库通用头文件

当前生成的 SYCL 文件通常包含以下仓库头文件：

- `dacppLib/include/ReconTensor.h`
- `dpcppLib/include/ParameterGeneration.h`

大多数当前的非 block 生成输出还包含：

- `dpcppLib/include/DataReconstructor1.h`

如果翻译器走 block/region 路径，生成的文件可能包含：

- `dpcppLib/include/DataReconstructor.new.h`
- `dpcppLib/include/utils.h`

MPI 生成的输出额外包含：

- `dpcppLib/include/MPIPlanner.h`
- `MPIPlanner.h` 从 `dpcppLib/include/mpi/` 引入的子头文件

由于 `MPIPlanner.h` 传递性地包含了 `dacInfo.h`，MPI 构建还依赖：

- `rewriter/include/dacInfo.h`

### 1.2 必需的包含目录

如果在辅助脚本之外编译生成的 `.cpp` 文件，需要以下包含根目录可用：

- `dacppLib/include`
- `dpcppLib/include`
- `rewriter/include`

### 1.3 外部工具链头文件

生成的代码还包含以下外部头文件：

- `sycl/sycl.hpp`
- `--mpi` 输出时的 `mpi.h`

因此，用于编译的机器必须安装：

- AdaptiveCpp
- MPI 输出所需的 MPI 开发环境
- 如果编译器配置需要，还需安装 OpenMP 头文件

## 2. 运行时所需的共享库

可执行文件编译完成后，上述仓库头文件不再是运行时依赖。运行时依赖是链接到二进制文件中的共享库。

在当前仓库的 macOS 环境中，`otool -L` 显示的典型依赖如下：

- `@rpath/libacpp-rt.dylib`
- `@rpath/libacpp-common.dylib`
- `/opt/homebrew/opt/libomp/lib/libomp.dylib`
- `/opt/homebrew/opt/open-mpi/lib/libmpi.40.dylib`

`libc++` 和 `libSystem` 等系统库由操作系统提供，通常不需要自行打包。

### 2.1 AdaptiveCpp 运行时

这是翻译后 SYCL 可执行文件的主要强制运行时依赖。

在本仓库中，默认安装路径为：

- `/Volumes/QUQ/working/sycl-install`

重要的运行时目录是：

- `$ACPP_ROOT/lib`

可执行文件在启动时至少需要 AdaptiveCpp 运行时库可用，特别是：

- `libacpp-rt.dylib`
- `libacpp-common.dylib`

### 2.2 OpenMP 运行时

当前构建也会链接 `libomp`。

因此运行时机器还需要能够找到：

- macOS 上的 `libomp.dylib`
- Linux 上的 `libomp.so`

### 2.3 MPI 运行时

对于 MPI 生成的程序，运行时需要 `libmpi`。

在本仓库当前的辅助脚本环境中，即使是非 MPI 的二进制文件，也可能被链接了 OpenMPI，因为 `env.sh` 在检测到 OpenMPI 安装时会添加 OpenMPI 链接标志。因此，如果按原样使用仓库辅助脚本，在非 MPI 二进制文件中看到 `libmpi` 不一定是 bug。

## 3. 运行时通常需要的环境变量

### macOS

通常需要：

```bash
export ACPP_ROOT=/Volumes/QUQ/working/sycl-install
export DYLD_LIBRARY_PATH="$ACPP_ROOT/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
```

如果 OpenMPI 或 `libomp` 不在加载器可发现的位置，可能还需要将它们的库目录追加到 `DYLD_LIBRARY_PATH`。

### Linux

通常需要：

```bash
export ACPP_ROOT=/path/to/sycl-install
export LD_LIBRARY_PATH="$ACPP_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

如果 MPI 或 `libomp` 安装在非标准路径，也需要添加对应的库目录。

## 4. 普通 SYCL 与 MPI SYCL 的需求对比

### 4.1 非 MPI 翻译的 SYCL 程序

如果只想运行普通翻译后的 SYCL 可执行文件：

- 编译时：
  - `ReconTensor.h`
  - `ParameterGeneration.h`
  - 通常需要 `DataReconstructor1.h`
  - AdaptiveCpp 头文件
- 运行时：
  - AdaptiveCpp 共享库
  - OpenMP 共享库
  - 如果使用了本仓库当前的 `env.sh` 构建，可能还需要 `libmpi`

### 4.2 MPI 翻译的 SYCL 程序

如果要运行 `--mpi` 翻译后的可执行文件：

- 编译时：
  - 以上所有内容
  - `MPIPlanner.h`
  - `dpcppLib/include/mpi/`
  - MPI 头文件
  - `rewriter/include/dacInfo.h`
- 运行时：
  - AdaptiveCpp 共享库
  - OpenMP 共享库
  - MPI 共享库
  - `mpirun` 或等效的 MPI 启动器

## 5. 快速检查清单

如果要将生成的源文件或二进制文件迁移到另一台机器，请检查：

- 该机器是否安装了 AdaptiveCpp，`sycl/sycl.hpp` 是否可用？
- 编译时以下包含目录是否可用？
  - `dacppLib/include`
  - `dpcppLib/include`
  - `rewriter/include`
- 生成的文件是否包含 `MPIPlanner.h`？
  - 如果是，还需携带 `dpcppLib/include/mpi/` 以及 MPI 头文件和库
- 运行时机器是否有以下库？
  - `libacpp-rt`
  - `libacpp-common`
  - `libomp`
  - MPI 构建时的 `libmpi`
- `DYLD_LIBRARY_PATH` 或 `LD_LIBRARY_PATH` 是否已配置，使加载器能找到这些库？

## 6. 本仓库中的实用命令

编译：

```bash
source /Volumes/QUQ/working/dacpp/clang/tools/translator/env.sh
acpp-compile /path/to/file.dac_sycl_buffer.cpp /tmp/my_bin
```

运行非 MPI 程序：

```bash
DYLD_LIBRARY_PATH="$ACPP_ROOT/lib" /tmp/my_bin
```

运行 MPI 程序：

```bash
DYLD_LIBRARY_PATH="$ACPP_ROOT/lib" mpirun -np 4 /tmp/my_bin
```

## 7. 重要说明

如果问题严格来说是"运行已构建好的二进制文件时需要什么"，答案是：

- 不再需要仓库头文件
- 需要已链接的共享库，主要是 AdaptiveCpp、`libomp`，以及 MPI 构建时的 MPI 库

如果问题是"我需要携带什么到其他地方才能重新编译生成的 `.cpp` 文件"，则需要：

- 上面列出的仓库运行时头文件
- 外部 SYCL/MPI 工具链和库
