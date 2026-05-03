# DACPP Translator

更新时间：2026-05-03

`clang/tools/translator` 是 DACPP 到 C++/SYCL 的 source-to-source translator 主目录。当前主线以 buffer 生成路径为核心，并支持 `--mpi` 生成 MPI + SYCL 代码。

## 1. 快速开始

加载本机开发环境：

```bash
source /Volumes/QUQ/working/dacpp/clang/tools/translator/env.sh
```

修改 translator 代码后重新构建：

```bash
cd /Volumes/QUQ/working/dacpp
cmake --build build --target translator -j8
```

当前 translator 二进制：

```text
/Volumes/QUQ/working/dacpp/build/bin/translator/translator
```

`env.sh` 会提供常用 shell 函数：

- `dacpp`：调用 translator，并自动附加本仓库前端头文件路径。
- `acpp-compile`：用 AdaptiveCpp 编译生成的 SYCL C++ 文件。
- `dacpp-translate-and-build`：遗留便捷函数，当前默认行为不如显式命令清晰，建议日常使用 `dacpp` + `acpp-compile`。

## 2. 常用翻译命令

普通 buffer 翻译：

```bash
dacpp /path/to/file.dac.cpp --mode=buffer
```

MPI buffer 翻译：

```bash
dacpp /path/to/file.dac.cpp --mode=buffer --mpi
```

MPI 输出同步策略：

```bash
dacpp /path/to/file.dac.cpp --mode=buffer --mpi --mpi-output-sync=all-ranks
dacpp /path/to/file.dac.cpp --mode=buffer --mpi --mpi-output-sync=root-only
```

说明：

- `all-ranks` 是默认值：root gather 输出后，再把需要同步的输出 broadcast 给所有 rank。
- `root-only` 会跳过最终 broadcast，只适合后续代码不需要非 root 副本一致的场景。

当前主线生成文件名通常是：

```text
*.dac_sycl_buffer.cpp
*.mpi.dac_sycl_buffer.cpp
```

编译生成文件：

```bash
acpp-compile /path/to/file.dac_sycl_buffer.cpp /tmp/my_bin
```

运行普通程序：

```bash
DYLD_LIBRARY_PATH="$ACPP_ROOT/lib" /tmp/my_bin
```

运行 MPI 程序：

```bash
DYLD_LIBRARY_PATH="$ACPP_ROOT/lib" mpirun -np 4 /tmp/my_bin
```

## 3. 回归测试

本地 buffer smoke suite：

```bash
cd /Volumes/QUQ/working/dacpp/clang/tools/translator
bash test_local.sh
```

指定本地用例：

```bash
bash test_local.sh liuliang1.0 waveEquation1.0
```

使用 large 输入：

```bash
bash test_local.sh --large
```

MPI 主线回归：

```bash
cd /Volumes/QUQ/working/dacpp/clang/tools/translator
bash test_mpi.sh
```

指定 MPI 用例：

```bash
bash test_mpi.sh waveEquation1.0 stencil1.0 FOuLa1.0 mpiDenseCoverSibling1.0
```

默认临时目录：

- `/Volumes/QUQ/working/local_tmp`
- `/Volumes/QUQ/working/mpi_tmp`

`test_mpi.sh` 当前流程：

1. 复制 `.dac.cpp` 到临时目录。
2. 执行 `dacpp --mode=buffer` 生成 baseline 并编译运行。
3. 执行 `dacpp --mode=buffer --mpi` 生成 MPI 版本并编译。
4. 用 `mpirun -np 4` 运行 MPI 程序。
5. 对比 baseline 输出和 MPI 输出。

## 4. 当前主线架构

建议从这些文件理解当前真正生效的路径：

- `translator.cpp`
  - CLI option、AST matcher、主流程分发。
- `parser/include/DacppStructure.h`
- `parser/lib/DacppStructure.cpp`
- `parser/lib/Shell.cpp`
  - DACPP shell/calc、表达式、buffer region、MPI stencil site 等结构。
- `rewriter/include/Rewriter.h`
  - rewriter 总入口。
- `rewriter/lib/Rewriter_Buffer_new.cpp`
- `rewriter/lib/buffer_template_new.cpp`
  - 当前普通 buffer 主线和 buffer region 生成。
- `rewriter/lib/Rewriter_MPI.cpp`
  - MPI 总入口，负责普通 MPI wrapper 和 MPI stencil 路径分流。
- `rewriter/lib/mpi/Rewriter_MPI_Wrapper_Codegen.cpp`
  - 普通 MPI wrapper codegen。
- `rewriter/lib/mpi/Rewriter_MPI_Stencil_Analysis.cpp`
  - MPI stencil site 能力分析和 distributed-followup 判定。
- `rewriter/lib/mpi/Rewriter_MPI_Stencil_Codegen.cpp`
  - MPI stencil `ctx/init/run` 主 codegen。
- `rewriter/lib/mpi/Rewriter_MPI_OutputAnalysis.cpp`
  - MPI 输出同步分类、broadcast 需求分析。
- `rewriter/lib/mpi/Rewriter_MPI_PrintRewrite.cpp`
  - `.print()` / `std::cout` 的 root-only 改写。
- `rewriter/lib/mpi/Rewriter_MPI_ParamAnalysis.cpp`
  - 参数读写模式推断和语句访问摘要。
- `rewriter/lib/mpi/Rewriter_MPI_PostRegion_Analysis.cpp`
  - post-shell root-centric region 识别。
- `rewriter/lib/mpi/Rewriter_MPI_PostRegion_Codegen.cpp`
  - post-shell root-centric region helper 生成。
- `rewriter/lib/mpi/Rewriter_MPI_Pattern.cpp`
  - AccessPattern、PackPlan 相关生成。
- `dpcppLib/include/MPIPlanner.h`
- `dpcppLib/include/mpi/Common.h`
- `dpcppLib/include/mpi/Pack.h`
- `dpcppLib/include/mpi/Views.h`
  - MPI runtime helper、pack/writeback/layout/view。

`archive/` 和 `docs/**/archive/` 只保留历史材料，不代表当前主线。

## 5. 当前 MPI 逻辑摘要

普通 MPI wrapper 路径大体是：

```text
root tensor2Array / pack input
-> MPI_Scatterv
-> rank-local SYCL kernel
-> MPI_Gatherv writeback values
-> root apply_writeback_by_globals
-> optional MPI_Bcast updated outputs
```

当 `<->` 位于 loop 内，且 shell 实参能安全 hoist 到 loop 外时，会进入 MPI stencil loop 路径：

```cpp
__dacpp_mpi_stencil_ctx_xxx ctx;
__dacpp_mpi_stencil_init_xxx(ctx, ...);
for (...) {
    __dacpp_mpi_stencil_run_xxx(ctx, ...);
}
```

当前 MPI stencil Phase 1 已完成：

- pattern / binding / pack plan / rank range 在 `init()` 中初始化。
- input/output gathered layout 在 `init()` 中缓存。
- root 侧 globals、counts/displs、byte counts/displs、writeback slots 和临时 buffer 复用。
- `run()` 中删除重复 metadata gather，只保留每步数据通信。
- `WRITE` 参数每轮 kernel 前重置本地 buffer，保持 fresh vector 语义。
- profiling timing collective 只在 `DACPP_MPI_PROFILE=1` 时执行。

当前仍在设计中的问题：

- 输出后的非 root 副本一致性。
- `decay1.0` 这类 shell 输出被后续 host 代码读取时的 broadcast 分类。
- `liuliang1.0` 这类 shell 后可并行 host loop 的 MPI/SYCL region 化。
- 基于 pack/writeback plan 的 cache consistency / partial exchange，用 generalized halo 替代手写传统 halo。

## 6. Profiling

默认不执行 MPI wrapper timing collective。

需要 profiling 时：

```bash
export DACPP_MPI_PROFILE=1
```

生成代码中的 `dacpp::mpi::profilingEnabled()` 分支才会执行相关计时和 `MPI_Reduce`。

## 7. 运行时依赖

编译生成的 SYCL 文件通常需要：

- AdaptiveCpp
- `dacppLib/include`
- `dpcppLib/include`
- `rewriter/include`
- MPI 生成代码额外需要 MPI 开发环境和 `mpirun`

详细说明见：

- `tests/translated_sycl_runtime_dependencies_cn.md`
- `docs/macos_local_sycl_setup.md`

## 8. 推荐文档

- `tests/translated_sycl_runtime_dependencies_cn.md`
- `docs/macos_local_sycl_setup.md`
- `docs/local_translator/buffer_region_device_residency.md`
- `docs/mpi_translator/mpi_stencil_status.md`

## 9. 常见误区

- 当前 MPI 回归脚本走的是 `--mode=buffer --mpi`，不是 `--mode=usm --mpi`。
- 当前 MPI 主线生成并编译的是 `*.mpi.dac_sycl_buffer.cpp`，不是 `*.dac_sycl_buffer_mpi.cpp`。
- `MPI_Gatherv` 和 `MPI_Bcast` 不是替代关系：前者负责 root 结果重建，后者负责非 root 副本同步。
- `root-only` 输出同步不是通用优化，只有后续代码不需要 all-rank 副本一致时才安全。
- 普通 buffer 路径已经有 post-shell region 优化能力；MPI 路径当前已支持一维 root-centric post-shell region v1，尚未支持完整 distributed region pipeline。
