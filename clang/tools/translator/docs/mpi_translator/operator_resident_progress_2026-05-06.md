# Operator-Resident Communication Model — 实施进度

日期：2026-05-07（最后更新）
分支：`tqc-2`

设计文档：[operator_resident_communication_model_2026-05-06.md](./operator_resident_communication_model_2026-05-06.md)

本文档跟踪算子驻留通信模型的实施进度。

---

## 1. 当前目标

在 DACPP 的编程模型下设计并实现一种**新的 MPI 访问模型**，降低现有 AccessPattern 方案的效率损失。新的访问模型尚未设计，当前阶段完成的是**代码架构重构**——为新模型的接入预留了独立的代码路径和目录空间。

---

## 2. 实施总览

| 步骤 | 状态 | 说明 |
|------|------|------|
| 1. 建立目录骨架和 plan 类型 | **已完成** | rewriter/lib/mpi/ 分四层 + Rewriter_MPI_Plan.h |
| 2. 老实现收口为 legacy | **已完成** | 文件重命名归入 legacy_access_pattern/ |
| 3. Stencil 与 wrapper 解耦 | **已完成** | dpcppLib 也分层；plan 类型已扩展；逐表达式 plan |
| 4. 分析性能瓶颈并设计新访问模型 | **进行中** | 瓶颈分析已完成（见 4.1），新模型设计未开始 |
| 5. 实现新访问模型的 translator 侧分析 | **未开始** | 依赖步骤 4 的设计 |
| 6. 实现新访问模型的 codegen | **未开始** | 依赖步骤 5 |
| 7. 实现新访问模型的 runtime 头文件 | **未开始** | 依赖步骤 6 |
| 8. Benchmark 和验证 | **未开始** | |

---

## 3. 已完成工作详情

### 3.1 Rewriter 侧目录重构（步骤 1-2）

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

  operator_resident/                # 新访问模型（空占位）
    .gitkeep
```

### 3.2 运行时头文件重构（步骤 3 的一部分）

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

  operator_resident/                # 新访问模型运行时（空占位）
```

**facade 机制**：顶层 `Common.h` / `Wrapper.h` / `Stencil.h` / `Views.h` 不设自己的 include guard，只做 `#include` 转发。这样避免与移动文件的 guard 冲突。生成代码中的 `#include "mpi/Common.h"` 等路径不需要改变。

### 3.3 Plan 类型与逐表达式分发

**文件：** `rewriter/include/Rewriter_MPI_Plan.h`

```cpp
struct MpiAnalysisContext { DacppFile *dacppFile; };
struct DacExprNode { exprIndex, expr, shell, calc, dacExpr, parentStmt };
enum class MpiPlanKind { LegacyAccessPattern, StencilPhaseC, OperatorResident, Unsupported };
struct MpiPlanResult { kind, exprIndex, reason };
struct LegacyWrapperPlan : MpiPlanResult { exprNode; };
struct StencilPhaseCPlan : MpiPlanResult { exprNode; };
struct MpiLoweringPlan { overallKind, exprNodes[], exprResults[] };
```

注意：`MpiPlanKind::OperatorResident` 已预留为枚举值，但暂无对应的 plan 结构体和分发逻辑。等新访问模型设计完成后再补充。

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

### 3.4 未动的文件

以下文件在重构中未修改：

- `dpcppLib/include/MPIPlanner.h`（生成代码的主 include 入口）
- `rewriter/include/Rewriter_MPI_Common.h`（所有 MPI rewriter 共享的 facade 头）
- `rewriter/include/Rewriter_MPI_Stencil_Common.h`（stencil rewriter 共享头）
- `rewriter/include/Rewriter.h`（Rewriter 类定义）
- `parser/` 下所有文件
- `translator.cpp`

---

## 4. 下一步工作

### 4.1 现有方案性能瓶颈分析（已完成）

> 详细 benchmark 数据见 `docs/benchmarks/mpi_sycl_efficiency_benchmark_2026-05-06.md`。

#### 4.1.1 效率差距总览

| 用例 | 效率比（手写 / 自动生成） | 核心特征 |
|------|--------------------------|----------|
| imageAdjustment1.0 | ~20x | 2 个连续 `<->`，2D element-wise，中间张量不落地 |
| vectorAddCombo | ~7x | 3 个连续 `<->`，1D element-wise + scalar broadcast |
| oddeven0.1 | ~2.8x | 偶奇排序，多轮迭代 |
| gradientSum | ~2.2x | 行归约，单 `<->` |

#### 4.1.2 根因：per-`<->` 的 root-centric 通信循环

当前 LegacyAccessPattern 方案对**每一个 `<->` 表达式**独立执行完整的通信周期：

```
对每个 <-> 表达式：
  1. 构建 AccessPattern + PackPlan（~50 行初始化代码）
  2. MPI_Gather 全局索引到 rank 0
  3. Rank 0 计算 root-centric pack plan（slots / offsets）
  4. MPI_Scatterv 分发数据到各 rank
  5. SYCL kernel 执行（通过 View1D 间接寻址）
  6. MPI_Gatherv 收集结果到 rank 0
  7. Rank 0 写回结果
  → 约 10 次 MPI 调用 / 表达式
```

对比手写参考代码（以 imageAdjustment1.0 为例，~120 行）：
```
1. 按 row range 划分一次，各 rank 获得本地行范围
2. 各 rank 独立执行 kernel_1（image_1）和 kernel_2（image_2），中间张量不离开本地
3. 最后一次 MPI_Gatherv 收集最终结果
→ 约 3 次 MPI 调用
```

#### 4.1.3 中间张量的不必要物化

以 imageAdjustment1.0 为例，源码有两个连续 `<->`：

```cpp
image_tensor2 = image_1(image_tensor, brightness, contrast) <-> image_calc_1(...);
final_tensor  = image_2(image_tensor2, gamma)              <-> image_calc_2(...);
```

生成代码中：
- `wrapper_1` 执行后，`image_tensor2` 被 Gather 到 rank 0 再 Scatter 出去
- `wrapper_2` 再次 Scatter `image_tensor2` 作为输入
- `image_tensor2` 从未在 host 端被读取，但经历了完整的 root-centric 往返

这是 20x 效率损失的主要来源——数据在 rank 间无意义地搬运。

#### 4.1.4 View1D 间接寻址开销

当前方案的 kernel 内部通过 `View1D` 访问数据：
```cpp
// 生成的 kernel 代码
int slot = pack_map.slots[idx];
auto val = view[slot];  // view 内部还有 key_offsets 间接层
```

而手写参考代码直接使用 `buffer[local_offset + i]`，无间接寻址。对于 element-wise 操作，shell 的 index 划分已经给出了完整的 rank-本地范围，不需要 slots/offsets 映射表。

#### 4.1.5 每 `<->` 独立初始化开销

每个 `<->` 表达式独立构建：
- `AccessPattern` 对象（遍历全局维度信息）
- `PackPlan`（计算 slots / offsets / key_counts）
- SYCL buffer 和临时缓冲区
- MPI 通信 buffer

对于共享相同划分方式的连续 `<->`，这些初始化完全重复。

#### 4.1.6 优化方向

基于以上分析，新的 MPI 访问模型应：

1. **利用 shell 的 index/void 信息建立 rank 数据归属**：shell 中 `index` 标记的维度由 MPI 划分，`{}` 标记的维度广播/本地持有——这足以确定每个 rank 持有哪些数据分片
2. **跨连续 `<->` 复用划分方案**：当多个 `<->` 共享相同的 index 划分时，rank 数据归属只需计算一次
3. **中间张量保持分布式**：一个 `<->` 的输出如果仅被后续 `<->` 读取，则不执行 Gather/Scatter，直接在本地传递
4. **仅在链边界物化**：只有当输出需要被 host 读取或划分方式改变时，才执行完整的 Gather
5. **消除 View1D 间接寻址**：对于 element-wise / broadcast 模式，直接使用本地范围索引

### 4.2 设计新的 MPI 访问模型

基于上述优化方向，设计具体的通信方案和实现架构。需要定义：
- 如何从 shell 的 split/index 信息推导 rank 数据归属
- 如何识别连续 `<->` 链及其中间张量
- 新的 codegen 路径结构
- 新的 runtime 头文件接口

### 4.3 实现新模型

设计完成后，在 `operator_resident/` 目录下实现 translator 侧分析和 codegen，在 `dpcppLib/include/mpi/operator_resident/` 下实现 runtime 头文件。

---

## 5. 关键代码入口

| 文件 | 作用 |
|------|------|
| `rewriter/lib/Rewriter_MPI.cpp` | MPI 总入口，分发 wrapper/stencil/未来新模型 |
| `rewriter/lib/mpi/shared/MpiPlanBuilder.cpp` | plan builder，当前区分 stencil/legacy，预留 OperatorResident |
| `rewriter/include/Rewriter_MPI_Plan.h` | 抽象类型（DacExprNode、MpiPlanKind、MpiLoweringPlan） |
| `rewriter/include/Rewriter_MPI_Common.h` | 所有 MPI rewriter 共享的 facade 头（函数声明、数据结构） |
| `rewriter/include/Rewriter_MPI_Stencil_Common.h` | stencil rewriter 共享头 |
| `rewriter/lib/mpi/legacy_access_pattern/WrapperCodegen.cpp` | 当前普通 MPI wrapper 的 codegen |
| `rewriter/lib/mpi/stencil_phase_c/StencilCodegen.cpp` | stencil Phase-C codegen 主骨架 |
| `dpcppLib/include/MPIPlanner.h` | 生成代码的主 include 入口 |
| `dpcppLib/include/mpi/Common.h` | 运行时 facade |
| `CMakeLists.txt` | 构建配置，REWRITER_SOURCES 列表 |

---

## 6. 验证方法

```bash
# 构建 translator
cd /Volumes/QUQ/working/dacpp
cmake --build build --target translator -j8

# 全量 MPI 回归（28 个用例）
cd clang/tools/translator
bash test_mpi.sh
```

---

## 7. 设计约束提醒

- **不改源码语法**：DACPP 源码中 `<->` 语法不变
- **不改 CLI**：`--mpi`、`--mpi-output-sync` 等参数不变
- **不改运行时 include 路径**：`MPIPlanner.h`、`mpi/Common.h` 等公共 facade 入口不变（内部可转发到子目录）
- **不改 stencil 主路径**：Phase-C 逻辑独立，不受新模型影响
- **fallback 安全**：新模型不支持的表达式回退 legacy，不改变语义
- **`Rewriter_MPI_Common.h` 暂不拆分**：当前作为 facade 保留，新代码不要再往里追加散函数
