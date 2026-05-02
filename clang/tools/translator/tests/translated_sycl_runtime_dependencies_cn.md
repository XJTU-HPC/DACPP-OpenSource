# 翻译后 SYCL / MPI 程序的编译与运行依赖

更新时间：2026-05-02

## 1. 范围

本文说明当前 `clang/tools/translator` 生成的 C++/SYCL 文件在编译和运行时需要什么。

覆盖的生成物包括：

- 普通 buffer 路径：`*.dac_sycl_buffer.cpp`
- MPI buffer 路径：`*.mpi.dac_sycl_buffer.cpp`
- 旧 USM 相关路径的少量遗留生成物：`*.dac_sycl_usm.cpp`

当前主线测试和脚本默认使用：

```bash
dacpp input.dac.cpp --mode=buffer
dacpp input.dac.cpp --mode=buffer --mpi
```

也就是说，当前本地和 MPI 回归都以 buffer 生成路径为主。

## 2. 两类依赖

需要区分两件事：

- 编译生成的 `.cpp` 文件时需要的头文件、编译器和链接库。
- 运行已经编译好的可执行文件时需要的共享库和启动器。

本仓库的 translator runtime 支持代码大多是 header-only。可执行文件构建完成后，通常不需要把仓库头文件放在运行目录；但如果要在另一台机器重新编译生成的 `.cpp` 文件，就必须提供这些头文件和外部工具链。

## 3. 编译时依赖

### 3.1 仓库头文件

普通 buffer 生成代码通常包含：

- `dacppLib/include/ReconTensor.h`
- `dpcppLib/include/DataReconstructor1.h`
- `dpcppLib/include/ParameterGeneration.h`

如果触发 buffer region 优化，生成代码可能包含：

- `dpcppLib/include/DataReconstructor.new.h`
- `dpcppLib/include/utils.h`

MPI 生成代码额外包含：

- `mpi.h`
- `cstdio`
- `dpcppLib/include/MPIPlanner.h`
- `dpcppLib/include/mpi/Common.h`
- `dpcppLib/include/mpi/Pack.h`
- `dpcppLib/include/mpi/Views.h`

`MPIPlanner.h` 会传递性依赖 translator 侧的数据结构定义，因此编译 MPI 生成代码还需要：

- `rewriter/include/dacInfo.h`

### 3.2 必需 include 目录

脱离仓库脚本手动编译生成代码时，至少需要加入：

```text
clang/tools/translator/dacppLib/include
clang/tools/translator/dpcppLib/include
clang/tools/translator/rewriter/include
```

如果源代码还包含 `std_lib/include` 中的 DACPP 前端声明，也需要在翻译阶段提供：

```text
clang/tools/translator/std_lib/include
```

### 3.3 外部工具链

当前脚本默认使用 AdaptiveCpp 编译生成的 SYCL 代码：

- 头文件：`sycl/sycl.hpp`
- 编译器：`$ACPP_ROOT/bin/acpp`
- 默认本机路径：`/Volumes/QUQ/working/sycl-install`

MPI 生成代码还需要：

- MPI 开发头文件：`mpi.h`
- MPI 链接库：`libmpi`
- MPI 启动器：`mpirun`

在当前 macOS 本机脚本中，`env.sh` 会尽量从 Homebrew 或 `PATH` 自动发现 OpenMPI 和 `libomp`。

## 4. 运行时依赖

已经编译好的二进制运行时不需要仓库头文件，但需要链接到的动态库可被加载器找到。

当前 macOS 环境中常见依赖包括：

- `libacpp-rt.dylib`
- `libacpp-common.dylib`
- `libomp.dylib`
- MPI 程序需要 `libmpi.dylib`

Linux 上对应通常是：

- `libacpp-rt.so`
- `libacpp-common.so`
- `libomp.so`
- MPI 程序需要 `libmpi.so`

系统 C++ 运行时和系统库一般由操作系统提供。

## 5. 环境变量

### macOS

常用设置：

```bash
export ACPP_ROOT=/Volumes/QUQ/working/sycl-install
export DYLD_LIBRARY_PATH="$ACPP_ROOT/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
```

如果 OpenMPI 或 `libomp` 不在系统默认搜索路径，也需要把对应 `lib` 目录加入 `DYLD_LIBRARY_PATH`。

### Linux

常用设置：

```bash
export ACPP_ROOT=/path/to/sycl-install
export LD_LIBRARY_PATH="$ACPP_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

如果 MPI 或 `libomp` 在非标准路径，也需要把对应目录加入 `LD_LIBRARY_PATH`。

## 6. 仓库脚本提供的入口

推荐先加载：

```bash
source /Volumes/QUQ/working/dacpp/clang/tools/translator/env.sh
```

加载后会提供几个 shell 函数。

翻译：

```bash
dacpp /path/to/input.dac.cpp --mode=buffer
dacpp /path/to/input.dac.cpp --mode=buffer --mpi
```

编译生成文件：

```bash
acpp-compile /path/to/input.dac_sycl_buffer.cpp /tmp/my_bin
```

运行普通程序：

```bash
DYLD_LIBRARY_PATH="$ACPP_ROOT/lib" /tmp/my_bin
```

运行 MPI 程序：

```bash
DYLD_LIBRARY_PATH="$ACPP_ROOT/lib" mpirun -np 4 /tmp/my_bin
```

## 7. 普通 buffer 与 MPI buffer 对比

普通 buffer 生成代码：

- 编译时需要 DACPP tensor/runtime 头文件、AdaptiveCpp。
- 运行时需要 AdaptiveCpp runtime 和 OpenMP runtime。
- 如果按当前 `env.sh` 编译，可能也会链接 MPI 库，因为脚本检测到 OpenMPI 后会统一加入 MPI include/link flags；这不一定表示生成代码使用了 MPI。

MPI buffer 生成代码：

- 编译时额外需要 MPI 头文件、`MPIPlanner.h`、`dpcppLib/include/mpi/`、`rewriter/include/dacInfo.h`。
- 运行时额外需要 MPI runtime 和 `mpirun`。
- 当前 MPI 输出会在普通 wrapper 路径和 MPI stencil loop 路径之间自动分流。

## 8. MPI 生成代码的当前形态

普通 MPI wrapper 路径大体执行：

```text
root pack input -> MPI_Scatterv -> local SYCL kernel
-> MPI_Gatherv writeback -> root apply writeback
-> optional MPI_Bcast output
```

位于 time-step loop 内、且 shell 实参可安全 hoist 的 `<->` 会进入 MPI stencil 路径，生成：

```cpp
__dacpp_mpi_stencil_ctx_xxx ctx;
__dacpp_mpi_stencil_init_xxx(ctx, ...);
for (...) {
    __dacpp_mpi_stencil_run_xxx(ctx, ...);
}
```

当前 stencil Phase 1 已把稳定 metadata 缓存在 `init()`，包括 gathered layout、counts/displs、root 侧 globals、writeback slots 和复用 buffer。`run()` 中保留每步真正需要的数据通信。

需要注意：

- `MPI_Gatherv` 用于 root 重建输出。
- `MPI_Bcast` 用于把 root 更新后的输出同步回非 root 副本。
- 二者不是替代关系。
- `--mpi-output-sync=root-only` 可以跳过最终 broadcast，但只有后续代码不需要 all-rank 副本一致时才安全。

## 9. Profiling

MPI 生成代码包含轻量 profiling 分支。只有开启运行时环境变量时才会执行相关 timing collective：

```bash
export DACPP_MPI_PROFILE=1
```

未开启时，生成代码中的 profiling `MPI_Reduce` 分支不会执行。

## 10. 迁移检查清单

如果迁移生成的 `.cpp` 文件到另一台机器重新编译，请检查：

- AdaptiveCpp 是否安装，`sycl/sycl.hpp` 是否可用。
- 是否提供以下 include 目录：
  - `dacppLib/include`
  - `dpcppLib/include`
  - `rewriter/include`
- 如果是 MPI 生成代码：
  - MPI 头文件和库是否可用。
  - `dpcppLib/include/mpi/` 是否完整。
  - `rewriter/include/dacInfo.h` 是否可用。
- 链接时是否能找到 AdaptiveCpp、OpenMP、MPI 库。

如果迁移已经编译好的二进制，请检查：

- AdaptiveCpp runtime 是否可加载。
- OpenMP runtime 是否可加载。
- MPI 程序的 MPI runtime 是否和编译时 ABI 匹配。
- macOS 用 `DYLD_LIBRARY_PATH`，Linux 用 `LD_LIBRARY_PATH` 配好非标准库路径。

## 11. 常见误区

- 当前主线 MPI 回归不是 `--mode=usm --mpi`，而是 `--mode=buffer --mpi`。
- 当前 MPI 主线生成物不是 `*.dac_sycl_buffer_mpi.cpp`，而是基于输入名生成 `*.mpi.dac_sycl_buffer.cpp`。
- 运行二进制不需要仓库头文件；重新编译生成 `.cpp` 才需要。
- 看到非 MPI 二进制链接了 `libmpi` 不一定是生成逻辑错误，可能是当前 `env.sh` 的统一链接参数导致。
