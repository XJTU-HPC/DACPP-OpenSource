# Operator-Resident Communication Model — 实施进度

日期：2026-05-06
分支：`tqc-2`

设计文档：[operator_resident_communication_model_2026-05-06.md](./operator_resident_communication_model_2026-05-06.md)

本文档跟踪算子驻留通信模型的实施进度，供接手者快速了解已完成和待完成的工作。

---

## 1. 实施总览

设计文档第 12 节定义了 8 个实施步骤。下表列出每步的状态和关键交付物。

| 步骤 | 状态 | 说明 |
|------|------|------|
| 1. 建立目录骨架和 planner | **已完成** | `36f92415e` |
| 2. 老实现收口为 legacy | **已完成** | 同上 |
| 3. Stencil 与 wrapper 解耦 | **部分完成** | 文件已分层；stencil 分析仍直接消费 AST 散函数，尚未完全通过 plan 接口 |
| 4. operator-resident 分析（不改生成） | **未开始** | |
| 5. 1D element-wise chain codegen | **未开始** | vectorAddCombo |
| 6. 2D matrix row-block codegen | **未开始** | imageAdjustment1.0 |
| 7. row-wise reduction codegen | **未开始** | gradientSum |
| 8. 扩展 profile 和 benchmark | **未开始** | |

---

## 2. 已完成工作详情

### 2.1 目录重构（步骤 1-2）

提交 `36f92415e refactor: restructure MPI rewriter into shared/legacy/stencil directories`

将 `rewriter/lib/mpi/` 从平铺结构重组为四层子目录：

```
rewriter/lib/mpi/
  shared/                           # 模型无关的共享工具
    MpiTypes.cpp                    ← Rewriter_MPI_Types.cpp
    OutputSyncAnalysis.cpp          ← Rewriter_MPI_OutputAnalysis.cpp
    OutputSyncAnalysis_Internal.h   ← Rewriter_MPI_OutputAnalysis_Internal.h
    PrintRewrite.cpp                ← Rewriter_MPI_PrintRewrite.cpp
    ParamModeAnalysis.cpp           ← Rewriter_MPI_ParamAnalysis.cpp
    PostRegionAnalysis.cpp          ← Rewriter_MPI_PostRegion_Analysis.cpp
    PostRegionCodegen.cpp           ← Rewriter_MPI_PostRegion_Codegen.cpp
    PostRegion_Internal.h           ← Rewriter_MPI_PostRegion_Internal.h
    MpiPlanBuilder.cpp              # 新增：plan builder 实现

  legacy_access_pattern/            # 以 AccessPattern/PackPlan 为中心的普通 wrapper
    PatternInit.cpp                 ← Rewriter_MPI_Pattern.cpp
    WrapperCodegen.cpp              ← Rewriter_MPI_Wrapper_Codegen.cpp

  stencil_phase_c/                  # loop stencil Phase-C 路径
    StencilAnalysis.cpp             ← Rewriter_MPI_Stencil_Analysis.cpp
    StencilAnalysisUtils.cpp        ← Rewriter_MPI_Stencil_Analysis_Utils.cpp
    StencilRouteParse.cpp           ← Rewriter_MPI_Stencil_Analysis_RouteParse.cpp
    StencilFollowupCollect.cpp      ← Rewriter_MPI_Stencil_Analysis_Collect.cpp
    StencilAnalysis_Internal.h      ← Rewriter_MPI_Stencil_Analysis_Internal.h
    StencilCodegen.cpp              ← Rewriter_MPI_Stencil_Codegen.cpp
    StencilCodegenUtils.cpp         ← Rewriter_MPI_Stencil_Codegen_Utils.cpp
    WaveSpecialization.cpp          ← Rewriter_MPI_Stencil_Codegen_Wave.cpp
    StencilCodegen_Internal.h       ← Rewriter_MPI_Stencil_Codegen_Internal.h

  operator_resident/                # 轻量算子链快路径（当前为空占位）
    .gitkeep
```

### 2.2 新增抽象类型（步骤 1 的一部分）

**文件：** `rewriter/include/Rewriter_MPI_Plan.h`

```cpp
namespace dacppTranslator::mpi_rewriter {

struct MpiAnalysisContext {
    DacppFile *dacppFile = nullptr;
};

struct DacExprNode {
    int exprIndex = -1;
    Expression *expr = nullptr;
    Shell *shell = nullptr;
    Calc *calc = nullptr;
    const clang::BinaryOperator *dacExpr = nullptr;
    const clang::Stmt *parentStmt = nullptr;
};

enum class MpiPlanKind {
    LegacyAccessPattern,
    StencilPhaseC,
    OperatorResident,
    Unsupported
};

struct MpiPlanResult {
    MpiPlanKind kind = MpiPlanKind::Unsupported;
    int exprIndex = -1;
    std::string reason;
};

struct MpiLoweringPlan {
    MpiPlanKind overallKind = MpiPlanKind::Unsupported;
    std::vector<DacExprNode> exprNodes;
    std::vector<MpiPlanResult> exprResults;
};

MpiLoweringPlan buildMpiLoweringPlan(DacppFile *dacppFile);
}
```

**文件：** `rewriter/lib/mpi/shared/MpiPlanBuilder.cpp`

当前实现：遍历所有 Expression 构建 `DacExprNode`，用 `hasMPIStencilSites()` 判断走 StencilPhaseC 还是 LegacyAccessPattern。**尚未实现 operator-resident 分支。**

### 2.3 Rewriter_MPI.cpp 接入

`Rewriter_MPI.cpp` 的 `rewriteMPI()` 已改为通过 `buildMpiLoweringPlan()` 分发：

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

以下文件在重构中未修改，后续步骤可能需要改动：

- `dpcppLib/include/mpi/` 下所有 17 个运行时头文件（生成代码 include 这些，改了会影响输出）
- `dpcppLib/include/MPIPlanner.h`（生成代码的主入口 include）
- `rewriter/include/Rewriter_MPI_Common.h`（所有 MPI rewriter 文件的共享 facade）
- `rewriter/include/Rewriter_MPI_Stencil_Common.h`（stencil rewriter 的共享 facade）
- `rewriter/include/Rewriter.h`（Rewriter 类定义）

---

## 3. 待完成工作

### 3.1 步骤 3：完善 Stencil 与 wrapper 解耦

当前状态：文件已物理分离到 `stencil_phase_c/` 和 `legacy_access_pattern/`，但 stencil 分析和 codegen 仍直接通过 `Rewriter_MPI_Common.h` 中的散函数访问 AST。

待做：
- stencil 分析改为只消费 `DacExprNode` / `MpiAnalysisContext`
- stencil codegen 改为只消费 `StencilPhaseCPlan`（需在 `Rewriter_MPI_Plan.h` 中扩展）
- 验证 stencil test suite 全过

### 3.2 步骤 4：operator-resident 分析（不改生成）

在 `operator_resident/` 下实现链识别分析。**不改 codegen，只输出 debug log。**

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

### 3.3 步骤 5：1D element-wise chain codegen

目标用例：`vectorAddCombo`（设计文档 7.1 节）

需要在 `operator_resident/` 下新增 codegen 文件，并在 `dpcppLib/include/mpi/operator_resident/` 下新增运行时头文件。

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

### 3.4 步骤 6：2D matrix row-block

目标用例：`imageAdjustment1.0`（设计文档 7.2 节）

支持模式（设计文档 4.3 节）：`dataList{image[idx1][idx2], image2[idx1][idx2]}`

### 3.5 步骤 7：row-wise reduction

目标用例：`gradientSum`（设计文档 7.3 节）

支持模式（设计文档 4.4 节）：`dataList{matGrads[{}][idx1], matNeuronSum[idx1][idx2]}`

### 3.6 步骤 8：benchmark 和 profile

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
| `rewriter/lib/mpi/shared/MpiPlanBuilder.cpp` | plan builder，当前只区分 stencil/legacy，待扩展 operator-resident |
| `rewriter/include/Rewriter_MPI_Plan.h` | 抽象类型定义（DacExprNode、MpiPlanKind、MpiLoweringPlan 等） |
| `rewriter/include/Rewriter_MPI_Common.h` | 所有 MPI rewriter 共享的 facade 头（函数声明、数据结构） |
| `rewriter/include/Rewriter_MPI_Stencil_Common.h` | stencil rewriter 共享头 |
| `rewriter/lib/mpi/legacy_access_pattern/WrapperCodegen.cpp` | 当前普通 MPI wrapper 的 codegen |
| `rewriter/lib/mpi/stencil_phase_c/StencilCodegen.cpp` | stencil Phase-C codegen 主骨架 |
| `dpcppLib/include/MPIPlanner.h` | 生成代码的主 include 入口 |
| `dpcppLib/include/mpi/Common.h` | 运行时聚合头 |
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
- **不改运行时头文件 include 路径**：`MPIPlanner.h`、`mpi/Common.h` 等公共入口不变
- **不改 stencil 主路径**：Phase-C 逻辑独立，不受 operator-resident 影响
- **fallback 安全**：operator-resident 分析失败时回退 legacy，不改变语义
- **`Rewriter_MPI_Common.h` 暂不拆分**：当前作为 facade 保留，新代码不要再往里追加散函数
