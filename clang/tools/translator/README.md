# DACPP Translator

更新时间：2026-05-07

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
  - MPI 总入口。通过 `buildMpiLoweringPlan()` 分发到 operator-resident、stencil 或 legacy wrapper 路径。
- `rewriter/include/Rewriter_MPI_Plan.h`
  - MPI lowering 公共 plan 总线（MpiPlanKind、MpiPlanResult、MpiLoweringPlan 等）。
- `rewriter/include/mpi/shared/MpiPlanBase.h`
  - MPI 分析基础节点（MpiAnalysisContext、DacExprNode）。
- `rewriter/include/mpi/operator_resident/OperatorResidentPlan.h`
  - Shell-derived / operator-resident 专属 IR（PartitionSignature、ShellPartitionPlan、OperatorResidentChainPlan 等）。

### 4.1 MPI Rewriter 分层

MPI rewriter 代码按职责分四层目录：

```
rewriter/lib/mpi/
  shared/                       # 模型无关的共享工具
    MpiPlanBuilder.cpp          # buildMpiLoweringPlan() 实现
    MpiTypes.cpp                # MPI 数据类型映射
    OutputSyncAnalysis.cpp      # MPI 输出同步分类、broadcast 需求分析
    ParamModeAnalysis.cpp       # 参数读写模式推断和语句访问摘要
    PrintRewrite.cpp            # .print() / std::cout 的 root-only 改写
    PostRegionAnalysis.cpp      # post-shell root-centric region 识别
    PostRegionCodegen.cpp       # post-shell root-centric region helper 生成

  legacy_access_pattern/        # 以 AccessPattern/PackPlan 为中心的普通 wrapper
    PatternInit.cpp             # AccessPattern、PackPlan 相关生成
    WrapperCodegen.cpp          # 普通 MPI wrapper codegen

  stencil_phase_c/              # loop stencil Phase-C 路径
    StencilAnalysis.cpp         # MPI stencil analysis 的 public entrypoint
    StencilAnalysisUtils.cpp    # tensor/source-text/split-domain helper
    StencilRouteParse.cpp       # route / boundary AST 解析和 assignment 提取
    StencilFollowupCollect.cpp  # distributed followup、read-transition、boundary-local update collector
    StencilCodegen.cpp          # MPI stencil ctx/init/run/materialize 主骨架
    StencilCodegenUtils.cpp     # stencil codegen 的 AST/fallback-input/pattern-init helper
    WaveSpecialization.cpp      # wave specialization 专属 code emission

  operator_resident/            # shell-derived / operator-resident Phase 1/2
    ShellPartitionAnalysis.cpp      # shell/calc 遍历、plan 组装、日志和 reject 编排
    SplitBindAnalysis.cpp           # split kind、bind order、shape helper
    ScalarAccessAnalysis.cpp        # `{}` scalar 判定和 calc 参数 `[0]` 使用检查
    Direct1DPartitionAnalysis.cpp   # Contiguous1D layout 判定
    RowBlock2DPartitionAnalysis.cpp # RowBlock2D layout 判定和 phase layout 编排
    FixedBlockPartitionAnalysis.cpp # P5 FixedBlock first-slice gate
    OperatorChainAnalysis.cpp       # 连续 compatible chain 识别
    ResidencyAnalysis.cpp           # RootOnly/DistributedDirty/ReplicatedScalar/MaterializedRoot 状态分析
    OperatorResidentCodegen.cpp     # OR wrapper 顶层拼装入口
    OperatorResidentWrapperCodegen.cpp # wrapper signature、参数命名、view 类型
    CollectiveCodegenUtils.cpp      # scatter/gather、byte counts/displs、resident-or-scatter
    PartitionCodegen.cpp            # 1D contiguous 和 2D row-block ownership codegen
    LocalKernelCodegen.cpp          # SYCL local kernel launch 和 ContiguousView1D 构造
    ResidentBufferCodegen.cpp       # scalar broadcast、resident buffer、final materialize
    FixedBlockCodegen.cpp           # P5 FixedBlock standalone OR wrapper codegen
```

共享 facade 头文件：
- `rewriter/include/Rewriter_MPI_Common.h` — 所有 MPI rewriter 共享的函数声明和数据结构
- `rewriter/include/Rewriter_MPI_Stencil_Common.h` — stencil rewriter 专用声明
- `rewriter/include/Rewriter_MPI_OperatorResident.h` — shell-derived / operator-resident public entrypoints

### 4.2 MPI 运行时头文件分层

```
dpcppLib/include/
  MPIPlanner.h                  # 生成代码的主 include 入口
  mpi/
    Common.h                    # facade → common/ + legacy_access_pattern/
    Wrapper.h                   # facade → legacy_access_pattern/Wrapper.h
    Stencil.h                   # facade → stencil/Stencil.h
    Views.h                     # facade → common/KernelViews.h + common/RegionViews.h
    Pack.h                      # facade → Wrapper.h + Stencil.h

    common/                     # 跨路径共享基础
      CoreTypes.h               # ItemRange, PackMap, AccessMode 等核心类型
      Profile.h                 # profiling 工具
      MpiTypes.h                # MPI 数据类型映射
      KernelViews.h             # View1D, View2DRow 等
      RegionViews.h             # Region packed element views

    legacy_access_pattern/      # 普通 wrapper 运行时
      Pattern.h                 # AccessPattern 定义
      PackMap.h                 # global-to-local 索引映射
      WrapperPack.h             # build_input_pack_plan 等包装函数
      Wrapper.h                 # 聚合头

    stencil/                    # stencil Phase-C 运行时
      StencilTypes.h            # stencil 数据结构
      StencilLayout.h           # MPI 通信 layout
      StencilExchangePlan.h     # exchange plan 构造
      StencilExchangeRuntime.h  # SYCL 加速 exchange 执行
      StencilExchange.h         # 聚合头
      WaveExchangeSpecialization.h  # wave span/row-copy 快速路径
      Stencil.h                 # 聚合头

    operator_resident/          # shell-derived / operator-resident runtime helper
      OperatorResidentRuntime.h # resident local storage、rank range、counts/displs helper
```

### 4.3 当前 Shell-Derived / Operator-Resident 路径

Phase 1/2 已接入 `buildMpiLoweringPlan()`，支持：

- `direct_1d` / `resident_chain_1d`：1D direct map、scalar broadcast、连续 resident chain。
- `row_block_2d`：2D row-major direct map，按 row block 分发和收集。
- `fixed_block_1d`：P5 first slice，非重叠 1D `RegularSplit(size == stride == 2)`，当前为 standalone OR wrapper。

当前已覆盖：

- `vectorAddCombo`：3 个 1D resident chain，`tmp_tensor` / `shifted_tensor` 不 materialize。
- `imageAdjustment1.0`：2 个 2D row-block image pass，`image_tensor2` 不 materialize。
- `mandel1.0`：single direct 1D。
- `decay1.0`：1D direct + replicated scalar。
- `oddeven0.1`：两个 `split(2,2)` fixed-block call site 进入 P5 `FixedBlock`，不生成 legacy `AccessPattern` / `PackPlan`。

accepted OR 路径不会生成 legacy `AccessPattern` / `PackPlan`，使用 `ContiguousView1D` 和 rank-local resident buffer；unsupported pattern 仍 fallback legacy 或 stencil Phase-C。完整设计见 `docs/mpi_translator/shell_derived_partition_implementation_plan_2026-05-07.md`。

### 4.4 当前 MPI stencil 代码分层

| 层 | 主要文件 | 职责 |
|---|---|---|
| 入口与分流 | `translator.cpp`, `Rewriter_MPI.cpp`, `Rewriter_MPI_Stencil.cpp` | 识别 `--mpi`，登记 stencil site，并在普通 wrapper / stencil path 之间分流 |
| 分析主骨架 | `rewriter/lib/mpi/stencil_phase_c/StencilAnalysis.cpp` | 只保留 Phase C 站点判定和 public entrypoint 编排 |
| 分析 helper | `StencilAnalysisUtils.cpp`, `StencilRouteParse.cpp`, `StencilFollowupCollect.cpp`, `StencilAnalysis_Internal.h` | 分别负责 tensor/split helper、route AST 解析、followup/transition/boundary collector 和内部共享声明 |
| Codegen 主骨架 | `StencilCodegen.cpp` | 生成 `ctx/init/run/materialize` 主流程，控制 fallback 和 distributed choreography |
| Codegen specialization/helper | `StencilCodegenUtils.cpp`, `WaveSpecialization.cpp` | 前者放通用 helper，后者只放 wave specialization emission |
| Runtime state / plan | `StencilTypes.h`, `StencilLayout.h`, `StencilExchangePlan.h` | 定义 distributed state，并构造 slots / exchange plan / halo plan |
| Runtime execution | `StencilExchangeRuntime.h`, `WaveExchangeSpecialization.h` | 执行 scatter/exchange/publish；wave 的 span-pair / row-copy fast path 已独立出去 |

## 5. 当前 MPI 逻辑摘要

MPI lowering 当前按表达式 owner 分三类：

- shell-derived / operator-resident：Phase 1/2 direct/resident chain，不生成 legacy `AccessPattern` / `PackPlan`。
- stencil Phase-C：loop stencil、halo、route、deferred materialize。
- legacy access-pattern wrapper：unsupported pattern 的保守 fallback。

legacy wrapper 路径大体是：

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

当连续 direct shell 被 OR 接管时，会生成一组 `__dacpp_mpi_or_*` wrapper：

```cpp
__dacpp_mpi_or_VADD_vadd_0(a_tensor, b_tensor, tmp_tensor);
__dacpp_mpi_or_VSHIFT_vshift_1(tmp_tensor, bias_tensor, shifted_tensor);
__dacpp_mpi_or_VADD_vadd_2(shifted_tensor, c_tensor, out_tensor);
```

中间 tensor 保持 rank-local resident，只有最终 root-visible 输出需要时才 `MPI_Gatherv` / `array2Tensor()`。

当前生成逻辑的结构边界是：

- generic orchestration 负责 `ctx/init/run/materialize` 主骨架；
- wave specialization 只通过 `ctx.wave` 挂接自己的 direct-kernel metadata 和 fast-path state；
- runtime plan 构造、generic exchange 执行、wave span/row-copy fast path 已分到不同头文件，不再混在一个大块 helper 里。

当前 MPI stencil 主线已完成：

- pattern / binding / pack plan / rank range 在 `init()` 中初始化。
- input/output gathered layout 在 `init()` 中缓存。
- root 侧 globals、counts/displs、byte counts/displs、writeback slots 和临时 buffer 复用。
- `run()` 中删除重复 metadata gather，只保留每步数据通信。
- `WRITE` 参数每轮 kernel 前重置本地 buffer，保持 fresh vector 语义。
- profiling timing collective 只在 `DACPP_MPI_PROFILE=1` 时执行。
- wave specialization state 统一收进 `ctx.wave`，generic ctx 不再直接持有 `use_wave_*` / `wave_direct_*` 字段。
- runtime helper 已按 `plan/layout`、`generic exchange execution`、`wave specialization execution` 三层拆开。

当前仍在设计中的问题：

- Phase 3 full payload / replicated full tensor：`gradientSum`、`DFT1.0`、`jacobi1.0`、初版 `matMul1.0`。
- Phase 4 stencil window：把 `RegularSplit` 纳入 shell-derived owner。
- Phase 5 fixed block / odd-even：`oddeven0.1` 已有保守 standalone OR wrapper 闭环；性能版 loop-resident phase exchange 仍待实现。
- 输出后的非 root 副本一致性。
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
- `docs/mpi_translator/shell_derived_partition_implementation_plan_2026-05-07.md`
- `docs/mpi_translator/archive/mpi_stencil_status.md`

## 9. 常见误区

- 当前 MPI 回归脚本走的是 `--mode=buffer --mpi`，不是 `--mode=usm --mpi`。
- 当前 MPI 主线生成并编译的是 `*.mpi.dac_sycl_buffer.cpp`，不是 `*.dac_sycl_buffer_mpi.cpp`。
- `MPI_Gatherv` 和 `MPI_Bcast` 不是替代关系：前者负责 root 结果重建，后者负责非 root 副本同步。
- `root-only` 输出同步不是通用优化，只有后续代码不需要 all-rank 副本一致时才安全。
- 普通 buffer 路径已经有 post-shell region 优化能力；MPI 路径当前已支持一维 root-centric post-shell region v1，尚未支持完整 distributed region pipeline。
