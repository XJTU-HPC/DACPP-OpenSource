# Operator-Resident Communication Model — 实施进度

日期：2026-05-06（最后更新）
分支：`tqc-2`

设计文档：[operator_resident_communication_model_2026-05-06.md](./operator_resident_communication_model_2026-05-06.md)

本文档跟踪算子驻留通信模型的实施进度，供接手者快速了解已完成和待完成的工作。

---

## 1. 实施总览

设计文档第 12 节定义了 8 个实施步骤。下表列出每步的状态和关键交付物。

| 步骤 | 状态 | 提交 | 说明 |
|------|------|------|------|
| 1. 建立目录骨架和 planner | **已完成** | `36f92415e` | rewriter/lib/mpi/ 分四层 + Rewriter_MPI_Plan.h |
| 2. 老实现收口为 legacy | **已完成** | 同上 | 文件重命名归入 legacy_access_pattern/ |
| 3. Stencil 与 wrapper 解耦 | **已完成** | `c6d13283d` + `f8d528bee` | dpcppLib 也分层；plan 类型已扩展；逐表达式 plan |
| 4. operator-resident 分析（不改生成） | **未开始** | | 下一步 |
| 5. 1D element-wise chain codegen | **未开始** | | vectorAddCombo |
| 6. 2D matrix row-block codegen | **未开始** | | imageAdjustment1.0 |
| 7. row-wise reduction codegen | **未开始** | | gradientSum |
| 8. 扩展 profile 和 benchmark | **未开始** | | |

---

## 2. 已完成工作详情

### 2.1 Rewriter 侧目录重构（步骤 1-2）

提交 `36f92415e refactor: restructure MPI rewriter into shared/legacy/stencil directories`

```
rewriter/lib/mpi/
  shared/                           # 模型无关的共享工具
    MpiPlanBuilder.cpp              # plan builder 实现
    MpiTypes.cpp
    OutputSyncAnalysis.cpp
    OutputSyncAnalysis_Internal.h
    ParamModeAnalysis.cpp
    PostRegionAnalysis.cpp
    PostRegionCodegen.cpp
    PostRegion_Internal.h
    PrintRewrite.cpp

  legacy_access_pattern/            # 以 AccessPattern/PackPlan 为中心的普通 wrapper
    PatternInit.cpp
    WrapperCodegen.cpp

  stencil_phase_c/                  # loop stencil Phase-C 路径
    StencilAnalysis.cpp
    StencilAnalysisUtils.cpp
    StencilRouteParse.cpp
    StencilFollowupCollect.cpp
    StencilAnalysis_Internal.h
    StencilCodegen.cpp
    StencilCodegenUtils.cpp
    WaveSpecialization.cpp
    StencilCodegen_Internal.h

  operator_resident/                # 轻量算子链快路径（空占位）
    .gitkeep
```

### 2.2 运行时头文件重构（步骤 3 的一部分）

提交 `c6d13283d` + `f8d528bee`

```
dpcppLib/include/mpi/
  Common.h                          # facade → common/ + legacy_access_pattern/
  Wrapper.h                         # facade → legacy_access_pattern/Wrapper.h
  Stencil.h                         # facade → stencil/Stencil.h
  Views.h                           # facade → common/KernelViews.h + common/RegionViews.h
  Pack.h                            # facade → Wrapper.h + Stencil.h

  common/                           # 跨路径共享基础
    CoreTypes.h                     # ItemRange, PackMap, AccessMode 等核心类型
    Profile.h                       # profiling 工具
    MpiTypes.h                      # MPI 数据类型映射
    KernelViews.h                   # View1D, View2DRow 等
    RegionViews.h                   # Region packed element views

  legacy_access_pattern/            # 普通 wrapper 运行时
    Pattern.h                       # AccessPattern 定义（依赖外部 DataReconstructor）
    PackMap.h                       # global-to-local 索引映射
    WrapperPack.h                   # build_input_pack_plan 等包装函数
    Wrapper.h                       # 聚合头

  stencil/                          # stencil Phase-C 运行时
    StencilTypes.h
    StencilLayout.h
    StencilExchangePlan.h
    StencilExchangeRuntime.h
    StencilExchange.h
    WaveExchangeSpecialization.h
    Stencil.h                       # 聚合头

  operator_resident/                # 未来算子链快路径运行时（空占位）
```

**facade 机制**：顶层 `Common.h` / `Wrapper.h` / `Stencil.h` / `Views.h` 不设自己的 include guard，只做 `#include` 转发。这样避免与移动文件的 guard 冲突。生成代码中的 `#include "mpi/Common.h"` 等路径不需要改变。

### 2.3 Plan 类型与逐表达式分发

**文件：** `rewriter/include/Rewriter_MPI_Plan.h`

```cpp
struct MpiAnalysisContext { DacppFile *dacppFile; };
struct DacExprNode { exprIndex, expr, shell, calc, dacExpr, parentStmt; };
enum class MpiPlanKind { LegacyAccessPattern, StencilPhaseC, OperatorResident, Unsupported };
struct MpiPlanResult { kind, exprIndex, reason };
struct LegacyWrapperPlan : MpiPlanResult { exprNode; };
struct StencilPhaseCPlan : MpiPlanResult { exprNode; };
struct MpiLoweringPlan { overallKind, exprNodes[], exprResults[] };
```

**文件：** `rewriter/lib/mpi/shared/MpiPlanBuilder.cpp`

逐表达式构建 plan：对每个 Expression 检查是否属于 stencil site（通过 `getMPIStencilSites()` 匹配 `exprIndex`），是则标记 `StencilPhaseC`，否则标记 `LegacyAccessPattern`。

**`Rewriter_MPI.cpp` 接入：**

```cpp
void Rewriter::rewriteMPI() {
    auto plan = mpi_rewriter::buildMpiLoweringPlan(dacppFile);
    if (plan.overallKind == mpi_rewriter::MpiPlanKind::StencilPhaseC) {
        rewriteMPIStencil();
        return;
    }
    // ... 原 wrapper 路径不变 ...
}
```

### 2.4 未动的文件

以下文件在重构中未修改：

- `dpcppLib/include/MPIPlanner.h`（生成代码的主 include 入口）
- `rewriter/include/Rewriter_MPI_Common.h`（所有 MPI rewriter 共享的 facade 头）
- `rewriter/include/Rewriter_MPI_Stencil_Common.h`（stencil rewriter 共享头）
- `rewriter/include/Rewriter.h`（Rewriter 类定义）
- `parser/` 下所有文件
- `translator.cpp`

---

## 3. 待完成工作

### 3.1 步骤 4：operator-resident 分析（不改生成）

在 `rewriter/lib/mpi/operator_resident/` 下实现链识别分析。**不改 codegen，只输出 debug log。**

需要实现的文件：

```
operator_resident/
  OperatorChainAnalysis.cpp    # 扫描 basic block 中连续 <->，尝试构造 OperatorChainPlan
  OperatorDomainAnalysis.cpp   # 分析输出域和 rank ownership
  ResidencyAnalysis.cpp        # 分析 tensor 驻留状态
```

核心逻辑（参考设计文档第 3、5 节）：

1. 扫描同一 basic block 中顺序出现的 `<->` 表达式
2. 对每段链检查快路径条件：
   - 每个 `<->` 的输出域可确定
   - 参数访问是简单 element-wise / scalar broadcast / row-wise reduction
   - 中间 WRITE tensor 无 host 侧读取穿插
3. 成功时输出：`[DACPP][MPI][OR] chain accepted length=N domain=... materialize=...`
4. 失败时输出：`[DACPP][MPI][OR] chain rejected reason=...; fallback=legacy_access_pattern`

调度集成（修改 `MpiPlanBuilder.cpp`）：

```
调度规则（设计文档 9.4 节）：
1. 先识别 loop stencil site → StencilPhaseC（优先级最高）
2. 对未被 stencil 接管的 <->，尝试 operator-resident chain
3. 剩余 <-> → LegacyAccessPattern
4. 新路径 unsupported 时不报错，只记录 reason 并回退
```

### 3.2 步骤 5：1D element-wise chain codegen

目标用例：`vectorAddCombo`（设计文档 7.1 节）

需要在 `rewriter/lib/mpi/operator_resident/` 下新增 codegen 文件，并在 `dpcppLib/include/mpi/operator_resident/` 下新增运行时头文件。

生成代码形态（设计文档第 6 节）：

```cpp
__dacpp_mpi_operator_chain_ctx_xxx ctx;
__dacpp_mpi_operator_chain_init_xxx(ctx, ...);
__dacpp_mpi_operator_chain_run_xxx(ctx, ...);
__dacpp_mpi_operator_chain_materialize_xxx(ctx, ...);
```

需要支持的模式（设计文档 4.1、4.2、4.5 节）：
- Element-Wise 一对一：`dataList{lhs[i], rhs[i], out[i]}`
- Scalar Broadcast：`dataList{in[i], bias[{}], out[i]}`
- 简单临时 Tensor 链：`tmp/shifted` 不 materialize

### 3.3 步骤 6：2D matrix row-block

目标用例：`imageAdjustment1.0`（设计文档 7.2 节）

支持模式（设计文档 4.3 节）：`dataList{image[idx1][idx2], image2[idx1][idx2]}`

### 3.4 步骤 7：row-wise reduction

目标用例：`gradientSum`（设计文档 7.3 节）

支持模式（设计文档 4.4 节）：`dataList{matGrads[{}][idx1], matNeuronSum[idx1][idx2]}`

### 3.5 步骤 8：benchmark 和 profile

基准数据（设计文档第 10 节）：

| 用例 | 手写 MPI+SYCL(s) | 当前 DAC-MPI(s) | 效率 |
|------|------------------:|----------------:|-----:|
| vectorAddCombo | 0.676 | 4.902 | 0.14 |
| imageAdjustment1.0 | 0.796 | 15.894 | 0.05 |
| gradientSum | 0.728 | 1.632 | 0.45 |

---

## 4. 关键代码入口

接手者需要了解的核心文件：

| 文件 | 作用 |
|------|------|
| `rewriter/lib/Rewriter_MPI.cpp` | MPI 总入口，分发 wrapper/stencil/未来 operator-resident |
| `rewriter/lib/mpi/shared/MpiPlanBuilder.cpp` | plan builder，当前区分 stencil/legacy，待扩展 operator-resident |
| `rewriter/include/Rewriter_MPI_Plan.h` | 抽象类型（DacExprNode、MpiPlanKind、MpiLoweringPlan、LegacyWrapperPlan、StencilPhaseCPlan） |
| `rewriter/include/Rewriter_MPI_Common.h` | 所有 MPI rewriter 共享的 facade 头（函数声明、数据结构） |
| `rewriter/include/Rewriter_MPI_Stencil_Common.h` | stencil rewriter 共享头 |
| `rewriter/lib/mpi/legacy_access_pattern/WrapperCodegen.cpp` | 当前普通 MPI wrapper 的 codegen |
| `rewriter/lib/mpi/stencil_phase_c/StencilCodegen.cpp` | stencil Phase-C codegen 主骨架 |
| `dpcppLib/include/MPIPlanner.h` | 生成代码的主 include 入口 |
| `dpcppLib/include/mpi/Common.h` | 运行时 facade |
| `CMakeLists.txt` | 构建配置，REWRITER_SOURCES 列表 |

---

## 5. 验证方法

```bash
# 构建 translator
cd /Volumes/QUQ/working/dacpp
cmake --build build --target translator -j8

# 全量 MPI 回归（28 个用例）
cd clang/tools/translator
bash test_mpi.sh

# 指定用例
bash test_mpi.sh vectorAddCombo imageAdjustment1.0 gradientSum
```

---

## 6. 设计约束提醒

- **不改源码语法**：DACPP 源码中 `<->` 语法不变
- **不改 CLI**：`--mpi`、`--mpi-output-sync` 等参数不变
- **不改运行时 include 路径**：`MPIPlanner.h`、`mpi/Common.h` 等公共 facade 入口不变（内部可转发到子目录）
- **不改 stencil 主路径**：Phase-C 逻辑独立，不受 operator-resident 影响
- **fallback 安全**：operator-resident 分析失败时回退 legacy，不改变语义
- **`Rewriter_MPI_Common.h` 暂不拆分**：当前作为 facade 保留，新代码不要再往里追加散函数
