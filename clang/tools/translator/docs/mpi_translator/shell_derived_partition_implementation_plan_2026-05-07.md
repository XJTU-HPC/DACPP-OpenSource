# Shell-Derived MPI Lowering 与 Operator-Resident 通信模型

日期：2026-05-07
分支：`tqc-2`

本文合并并取代：

- `archive/operator_resident_communication_model_2026-05-06.md`
- `archive/operator_resident_progress_2026-05-06.md`
- 本文件旧版 Phase 1/2 实现计划

本文档是后续 MPI lowering 工作的 canonical 设计、进度和文件结构计划。

---

## 1. 总体目标

Shell-Derived MPI Lowering 的长期目标是：`clang/tools/translator/tests` 下所有不以 `mpi` 开头的应用测试，最终都能由新的 shell-derived lowering family 覆盖，不再依赖 legacy `AccessPattern` / `PackPlan` 路径。legacy 路径仍保留为未知用户程序和开发期安全 fallback。

当前 legacy MPI wrapper 的核心问题是通信主轴偏低：

- legacy 主轴：每个 `<->`、每个参数、每个 item 访问哪些 global index。
- 新主轴：shell `dataList` 已经定义了迭代域、rank ownership、local layout、replicated/full payload 输入，以及中间 tensor 是否应继续驻留在 rank 本地。

新模型不只是一个小的 operator-resident optimization，而是新的默认 MPI lowering family：

```text
Shell-Derived MPI Lowering
  direct_1d             1D pure index direct map
  resident_chain_1d     1D direct map + scalar broadcast + chain residency
  row_block_2d          2D row-block direct map + chain residency
  full_payload          {} + index 的 row/full payload 或 replicated full tensor
  stencil_window        RegularSplit / halo / window
  fixed_block           odd-even 这类 fixed pair/block partition
```

`operator_resident` 仍可作为代码目录和日志前缀，但语义上它只是 shell-derived lowering 的一个子 backend。

---

## 2. 背景与性能问题

当前普通 MPI wrapper 和部分 stencil fallback 路径以 `AccessPattern` 为核心：

- 每个 shell 参数独立构造 `AccessPattern`。
- 每个参数独立 `build_input_pack_plan()` / `build_output_pack_plan()`。
- 每次 `<->` wrapper 独立准备输入、执行 kernel、收集输出、按需要写回 root tensor。

这种模型很通用，但在轻量算子链里固定通信和 tensor 重建成本偏高。典型低效形态：

```cpp
VADD(a, b, tmp) <-> vadd;
VSHIFT(tmp, bias, shifted) <-> vshift;
VADD(shifted, c, out) <-> vadd;
```

低效率用例的共同特征：

- element-wise 或 row-wise reduction，单 item 计算量小。
- 多个 `<->` 串联，中间 tensor 只被后续 `<->` 读取。
- 输出 tensor 在真正 host 读取前反复被分布式算子消费。
- 手写 MPI+SYCL 可以按 rank 直接持有本地切片，而 DAC-MPI 仍按参数重新打包/收集。

近期 benchmark 暴露的差距：

| 用例 | 效率比（手写 / 自动生成） | 核心特征 |
|---|---:|---|
| `imageAdjustment1.0` | ~20x | 2 个连续 `<->`，2D element-wise，中间张量不落地 |
| `vectorAddCombo` | ~7x | 3 个连续 `<->`，1D element-wise + scalar broadcast |
| `oddeven0.1` | ~2.8x | 偶奇排序，多轮迭代 |
| `gradientSum` | ~2.2x | 行归约，单 `<->` |

legacy per-`<->` root-centric 通信循环大致为：

```text
对每个 <-> 表达式：
  1. 构建 AccessPattern + PackPlan
  2. MPI_Gather 全局索引到 rank 0
  3. rank 0 计算 root-centric pack plan
  4. MPI_Scatterv 分发数据
  5. SYCL kernel 执行
  6. MPI_Gatherv 收集结果到 rank 0
  7. rank 0 写回结果
```

手写 reference 对 `imageAdjustment1.0` 这类程序通常只做：

```text
1. 按 row range 划分一次，各 rank 获得本地行范围
2. 各 rank 独立执行 kernel_1 和 kernel_2，中间张量不离开本地
3. 最后一次 MPI_Gatherv 收集最终结果
```

因此，新模型的目标是把通信中心从“参数如何被 item 访问”提升到“算子链的数据应该驻留在哪里、什么时候必须通信”。

---

## 3. Operator-Resident 通信模型

### 3.1 模型目标

Operator-Resident Communication Model 的目标：

- 以 `<->` 的输出迭代域作为通信主轴，而不是以每个参数的 `AccessPattern` 作为主轴。
- 对轻量算子链建立稳定 rank ownership，使输入、输出、中间 tensor 在多个 `<->` 之间复用同一分片。
- 中间 WRITE tensor 默认保持 distributed resident，不立即 gather 回 root。
- 只有遇到 host 侧读取、`print()`、`tensor2Array()`、跨 rank 依赖、最终输出等 materialize 边界时，才收集到 root-visible tensor。
- 标量和小常量使用 replicated/broadcast 状态，不进入 per-item pack。
- 任何不满足快路径条件的场景，保守回退现有 legacy 路径。

非目标：

- 不立即替代 stencil halo / Phase-C route 主路径。
- 不处理复杂非仿射索引、动态 shape、复杂别名或通用 `READ_WRITE` 冲突。
- 不改变 DACPP 源码语法、CLI 或现有普通 MPI wrapper 语义。

### 3.2 通信执行策略

链初始化：

1. 根据第一个输出 tensor 构造 rank ownership。
2. 检查链内所有 `<->` 的 domain 是否兼容。
3. 为每个输入 tensor 查询 residency。
4. 对 `RootOnly` 输入执行一次按 owner range 的 scatter。
5. 对 `ReplicatedScalar` 输入执行 broadcast。
6. 为中间 WRITE tensor 分配 rank-local buffer。

链内运行：

- 每个 `<->` 使用同一个 rank ownership。
- kernel view 直接绑定 rank-local input/output buffer。
- 对同分片 distributed tensor 不再 pack/unpack。
- scalar replicated 参数直接传本地 scalar view。

链边界 materialize：

- 若后续代码不读取输出 tensor，可延迟 materialize。
- 若遇到 `print()`、`tensor2Array()`、host loop 读取、未知函数调用，则 materialize 相关 tensor。
- materialize 使用 `MPI_Gatherv` 或 contiguous gather。
- root 副本更新后，状态变成 `MaterializedRoot`。

循环提升：

位于 `for` / `while` 循环内的 `<->` 使用统一的 loop-lowered shell-derived 结构。该结构把循环不变的通信和 metadata 放入 `init()`，把每步变化的数据和 kernel 放入 `run()`，并在可观察边界使用 `materialize()` 恢复 root-visible tensor：

```cpp
__dacpp_mpi_shell_ctx_xxx ctx;
__dacpp_mpi_shell_init_xxx(ctx, ...);
while (...) {
    __dacpp_mpi_shell_run_xxx(ctx, ...);
}
__dacpp_mpi_shell_materialize_xxx(ctx, ...);
```

- `init()` 负责 rank ownership、counts/displs、稳定 layout、loop-invariant READ 输入分发、replicated/full payload cache 和可复用 SYCL queue/buffer。
- `run()` 负责 loop-variant scalar broadcast、每步 kernel、必要的 distributed publish / route / refresh，以及本步必须发生的输出同步。
- `materialize()` 只在 loop 后或 host 可观察点执行，把 distributed resident/cache 状态恢复为 root-visible tensor。
- `stencil_window` backend 使用该结构承载 halo、boundary-local、writer-to-reader route 和 read-cache transition。
- `direct_1d` / `row_block_2d` / `full_payload` backend 同样使用该结构承载 loop-invariant direct inputs。`decay1.0` 这类循环中 direct shell 的目标形态是：`N0s` / `lambdas` 在 `init()` scatter 一次，`t` 在 `run()` broadcast，`local_A` 按 host assignment 或最终输出需求同步。
- 无法证明实参、shape、bind domain 或 loop-carried state 稳定时，保持普通 per-`<->` wrapper 或 legacy fallback。

Fallback 条件：

- 输出 domain 无法确定。
- 参数 index 非简单直接映射。
- 同一 tensor 有复杂 `READ_WRITE`。
- host 侧在链中间读取中间 tensor。
- 多个输出 tensor ownership 不一致且无法证明安全转换。
- 动态 shape 或 runtime-dependent split 本阶段无法证明。

---

## 4. 核心 IR

### 4.1 Shell dim 和 bind domain

parser 已经把 `dataList` 拆成：

- `IndexSplit`
- `RegularSplit`
- id 为 `"void"` 的普通 `Split`

新 lowering 先以 bind domain 建模，而不是直接以 tensor dim 建模：

```cpp
enum class ShellDimKind {
    Index,
    Void,
    Split
};

struct BindDomain {
    int bindId = -1;
    std::string representative;
    std::string offsetExpr = "0";
    int64_t runtimeSizeParam = -1;
    int dimId = -1;
};

struct TensorDimMapping {
    std::string tensorName;
    int shellParamIndex = -1;
    int tensorDim = -1;
    ShellDimKind kind = ShellDimKind::Void;
    int bindId = -1;
    std::string splitName;
};
```

`collectSplitBindMeta(shell)` / `Shell::GetBindInfo()` 的结果作为 `bindId` 来源。Phase 1/2 只接受 offset 为 `"0"`。

### 4.2 Layout 和访问类别

```cpp
enum class LocalLayoutKind {
    Contiguous1D,
    RowBlock2D,
    RowPartitionFullRow,
    ReplicatedScalar,
    ReplicatedFullTensor,
    StencilWindow1D,
    StencilWindow2D,
    FixedBlock,
    Unsupported
};

enum class ParamAccessKind {
    DirectMapped,
    OutputDirect,
    ReplicatedScalar,
    ReplicatedFullTensor,
    RowPartitionFullRow,
    StencilWindow,
    FixedBlock,
    Unsupported
};
```

Phase 1/2 只实现：

- `Contiguous1D`
- `RowBlock2D`
- `ReplicatedScalar`

其余枚举是长期覆盖目标的 IR 预留。

### 4.3 PartitionSignature

```cpp
struct PartitionSignature {
    std::vector<int64_t> bindSizes;
    std::vector<int> bindOrder;
    LocalLayoutKind layout = LocalLayoutKind::Unsupported;
    std::string linearization; // "1d-linear" / "2d-row-major"
};

inline bool isCompatibleForChain(const PartitionSignature& lhs,
                                 const PartitionSignature& rhs) {
    return lhs.bindSizes == rhs.bindSizes &&
           lhs.bindOrder == rhs.bindOrder;
}
```

规则：

- chain 兼容性只比较 `bindSizes + bindOrder`。
- `layout` 和 `linearization` 只用于 codegen 策略选择，不参与 chain 兼容。
- Phase 1 额外要求 layout 为 `Contiguous1D`。
- Phase 2 额外要求 layout 为 `RowBlock2D`。

### 4.4 Residency

```cpp
enum class ResidencyKind {
    RootOnly,
    DistributedClean,
    DistributedDirty,
    ReplicatedScalar,
    MaterializedRoot,
    Unknown
};

struct TensorResidencyState {
    std::string tensorName;
    ResidencyKind kind = ResidencyKind::Unknown;
    PartitionSignature partition;
    LocalLayoutKind layout = LocalLayoutKind::Unsupported;
    bool rootValid = true;
    bool localValid = false;
};
```

Phase 1/2 当前使用：

- `RootOnly`
- `DistributedDirty`
- `ReplicatedScalar`
- `MaterializedRoot`

`DistributedClean` 和 `Unknown` 保留为后续扩展点。

---

## 5. Planner 与调度

当前 planner 以 `MpiLoweringPlan` 为总线：

```cpp
struct MpiLoweringPlan {
    MpiPlanKind overallKind = MpiPlanKind::Unsupported;
    std::vector<DacExprNode> exprNodes;
    std::vector<MpiPlanResult> exprResults;
    std::vector<ShellPartitionPlan> shellPartitionPlans;
    std::vector<OperatorResidentChainPlan> residentChains;
    std::vector<int> operatorResidentChainByExpr;
};
```

调度要求：

1. 每个 `<->` 有唯一 owner。
2. accepted chain 的 exprIndex 不再走 legacy wrapper。
3. unsupported expr 保留 legacy fallback。
4. stencil Phase-C 仍按现有 stencil site 优先级接管。
5. debug log 能解释 accepted/rejected reason。

建议日志：

```text
[DACPP][MPI][OR] expr=0 shell=... layout=Contiguous1D accepted
[DACPP][MPI][OR] chain=0 layout=Contiguous1D length=3 accepted
[DACPP][MPI][OR] expr=... rejected reason=...
```

---

## 6. 当前实现进度总览

截至 2026-05-08，shell-derived MPI lowering 的当前状态如下。以下状态以代码中的已启用 layout、已接入 codegen 路径，以及完整 `bash test_mpi.sh` 结果为准。

| Phase | 状态 | 当前已完成能力 | 代表测试 | 当前未完成部分 |
|---|---|---|---|---|
| Phase 1 | 已完成 | `Contiguous1D` direct / resident chain，含 `ReplicatedScalar` | `vectorAddCombo` `mandel1.0` `decay1.0` | loop-lifted direct/resident 结构仍未做 |
| Phase 2 | 已完成 | `RowBlock2D` direct / resident chain | `imageAdjustment1.0` | 仅覆盖当前 row-major 2D row-block 形态 |
| Phase 3 | 已完成 | `ReplicatedFullTensor`、`RowPartitionFullRow`、受限 `READ_WRITE OutputDirect` | `DFT1.0` `gradientSum` `jacobi1.0` `matMul1.0` | 更广的 full-payload 形态仍未展开 |
| Phase 4 | 进行中 | `StencilWindow1D` / `StencilWindow2D` 当前切片已接入；mixed-site 按 expr 分派已接入 | `stencil1.0` `waveEquation1.0` `FOuLa1.0` `MDP1.0` `liuliang1.0` `mpiMixedStencilORPhaseC` | root-bridge 和更广 stencil 形态仍未展开 |
| Phase 4.5 | 未开始 | 仅有设计目标 | 无 | `init/run/materialize` 的 loop-lifted direct/resident family 尚未落地 |
| Phase 5 | 未开始 | 仅有 IR / 枚举占位（如 `FixedBlock`） | 无 | `oddeven0.1` 所需分析与 codegen 均未实现 |
| Phase 6 | 未完成 | 完整测试已通过，但尚未做到“全部非 `mpi*` tests 都走新路径” | `bash test_mpi.sh` 33/33 | 仍存在需要继续 fallback 的普通应用测试 |

### 6.1 当前已启用的 OR layout

当前 `supportedPhaseLayout()` 已启用并生成 OR codegen 的 layout：

- `Contiguous1D`
- `RowBlock2D`
- `ReplicatedFullTensor`
- `RowPartitionFullRow`
- `StencilWindow1D`
- `StencilWindow2D`

当前仍未启用 codegen 的 layout / 占位：

- `FixedBlock`

### 6.2 当前验证基线

已通过（2026-05-08）：

```bash
cmake --build build --target translator -j8

cd clang/tools/translator
bash test_mpi.sh
```

当前基线结果：

- `33 tests | 33 passed | 0 failed | 0 skipped`
- 已接入 OR 的路径不再生成 legacy `AccessPattern` / `PackPlan`
- mixed-site 已按 expr 独立分派，不再因同文件存在未接入 stencil site 而整文件退回 Phase-C

---

## 7. 非 `mpi*` tests 当前覆盖状态

下表描述的是“当前实际走到哪里”，不是长期目标。

| 测试 | 当前 lowering | 状态说明 |
|---|---|---|
| `vectorAddCombo` | OR `Contiguous1D` | 已完成 |
| `mandel1.0` | OR `Contiguous1D` | 已完成 |
| `decay1.0` | OR `Contiguous1D` + `ReplicatedScalar` | 已完成；但尚未进入 loop-lifted family |
| `imageAdjustment1.0` | OR `RowBlock2D` | 已完成 |
| `DFT1.0` | OR `ReplicatedFullTensor` | 已完成 |
| `gradientSum` | OR `RowPartitionFullRow` | 已完成 |
| `jacobi1.0` | OR `RowPartitionFullRow` + `ReplicatedFullTensor` | 已完成 |
| `matMul1.0` | OR full-payload path + `READ_WRITE OutputDirect` | 已完成 |
| `stencil1.0` | OR `StencilWindow2D` | 已完成 |
| `waveEquation1.0` | OR `StencilWindow2D` | 已完成；当前支持 “window reader + optional 2D direct reader + WRITE-only direct writer” 的最小增量形态 |
| `FOuLa1.0` | OR `StencilWindow1D` | 已完成；当前覆盖 scalar root-materialize 形态 |
| `MDP1.0` | OR `StencilWindow1D` | 已完成；当前覆盖 1D `+1` distributed-followup 形态 |
| `liuliang1.0` | OR `StencilWindow1D` | 已完成；当前覆盖 1D `+1` distributed-followup 形态 |
| `oddeven0.1` | fallback | `FixedBlock` 尚未开始 |

补充说明：

- `waveEquation1.0` 的执行 wrapper 目前是 OR `StencilWindow2D`；测试里仍出现的 `[DACPP][MPI][PhaseC]` 路由/读缓存日志来自共享的分布式 stencil 站点分析，不表示最终 wrapper 走旧 stencil Phase-C codegen。
- 当前 OR `StencilWindow1D` / `StencilWindow2D` 已 lower 当前 accepted stencil followup 切片：1D `+1` distributed-followup、2D `+1,+1` distributed-followup，以及 `waveEquation1.0` 所需的 2D `-1,-1` read-cache transition。OR rewrite 会移除这些 post-shell followup / boundary-local 语句，并在 wrapper 内把 materialized writer 值按 route 和边界更新写回 reader tensor，再同步 followup/read-cache 目标 tensor，避免非 root host-visible reader 状态陈旧或被旧 writer 边界语句覆盖。
- 当前 OR stencil 路线下的 `RootCentricFollowup` 仍未进入可达形态；`hasRootBridge` 仍保持 reject。
- `distributed-followup` 的 all-ranks materialized writer refresh 已从 accepted OR stencil path 去掉；非 OR-stencil 或尚未 lower followup 的非 `RootOnly` 输出仍保留保守 refresh。

补充回归：

- `mpiOrReadWriteAccumulate1D`
- `mpiOrReadWriteAccumulate2D`
- `mpiMixedStencilORPhaseC`
- `mpiOrStencilRefreshPolicy1D`

其中 `mpiMixedStencilORPhaseC` 当前用于验证同文件 mixed stencil expr 仍按 expr 独立分派；当前口径要求同一文件里同时存在 OR site 和 Phase-C site。
其中 `mpiOrStencilRefreshPolicy1D` 当前用于验证 OR `StencilWindow1D` 的输出同步策略：`root-only` 不 broadcast materialized output；accepted `distributed-followup` 不再 broadcast materialized writer output，而是生成 OR-side followup 写回并同步 reader tensor，覆盖 followup 后继续 host 读取 reader 的场景。

---

## 8. 未完成阶段与当前推进位置

### 8.1 Phase 4：Stencil Window

Phase 4 目前不是“未开始”，而是已经推进到 1D / 2D 当前切片完成、剩余范围待继续展开的状态。

当前已完成：

- `StencilWindow1D` 已进入 OR codegen
- `StencilWindow2D` 已进入 OR codegen
- `stencil1.0` 已接入
- `waveEquation1.0` 已接入
- `FOuLa1.0`、`MDP1.0`、`liuliang1.0` 已接入
- mixed-site 顶层分派已支持按 expr 独立选择 OR / Phase-C / legacy
- OR-side post-shell distributed followup lowering 已接入当前 accepted 1D/2D stencil 切片
- `mpiOrStencilRefreshPolicy1D` 已更新，用于钉住 OR stencil distributed-followup 不再依赖 materialized writer broadcast、且同步 followup reader 的口径

当前边界：

- direct writer 仍必须是 WRITE-only，不接受 `READ_WRITE` direct writer
- 1D 当前只覆盖 scalar root-materialize 形态和 `+1` distributed-followup 形态
- 2D 当前只覆盖当前 row-major ownership 切片
- 目前最多支持一个额外的 2D direct reader
- direct reader 当前只覆盖 `waveEquation1.0` 所需的 `read-cache transition (-1,-1)` 形态
- 仍不支持 root-bridge stencil site

Phase 4 的下一步应继续集中在：

- 更广的 1D stencil window 形态
- 更广的 2D direct reader / route 形态
- root-bridge stencil site

也就是继续沿 shell-derived `stencil_window` 路线做 1D / 更广 stencil 形态，而不是横向扩到 `FixedBlock`。

### 8.2 Phase 4.5：Loop-Lowered Direct / Resident

这一阶段在文档中有设计目标，但代码中尚未形成独立的 loop-lowered direct/resident family。

当前推进位置：

- `decay1.0` 结果正确，但仍走普通 direct OR wrapper
- 尚未出现单独的 `init/run/materialize` direct shell 结构
- 尚未把 loop-invariant scatter-once / per-step broadcast 这类优化从 stencil family 抽到 direct/resident family

因此，Phase 4.5 当前应视为“未开始实现，仅有路线定义”。

### 8.3 Phase 5：Fixed Block / Odd-even

这一阶段当前仍是占位状态。

代码现状：

- `LocalLayoutKind::FixedBlock` 已存在
- `ParamAccessKind::FixedBlock` 已存在
- 但没有对应的分析入口、layout 判定或 codegen

因此，`oddeven0.1` 当前仍应继续 fallback，Phase 5 尚未进入可验证实现阶段。

### 8.4 Phase 6：非 `mpi*` tests 全量新路径

这一阶段当前还不能标记为完成。

当前推进位置：

- 完整 `bash test_mpi.sh` 已通过
- 但“完整测试通过”不等于“全部普通应用都已纳入 shell-derived planner”
- 当前仍有 `oddeven0.1` 等测试保持 fallback

因此，Phase 6 当前应理解为“验证基线已经稳定，但迁移覆盖率尚未收口”。

---

## 9. 文件结构现状与拆分计划

### 9.1 当前结构

translator 侧：

```text
rewriter/include/
  Rewriter_MPI_Plan.h
  Rewriter_MPI_OperatorResident.h
  mpi/shared/MpiPlanBase.h
  mpi/operator_resident/OperatorResidentPlan.h

rewriter/lib/mpi/
  shared/
  legacy_access_pattern/
  stencil_phase_c/
  operator_resident/
```

runtime 侧：

```text
dpcppLib/include/mpi/operator_resident/
  OperatorResidentRuntime.h
```

### 9.2 已执行的阶段 1/2/3 拆分

阶段 1：拆 `Rewriter_MPI_Plan.h`

- 新增 `mpi/shared/MpiPlanBase.h`：保存 `MpiAnalysisContext` 和 `DacExprNode`。
- 新增 `mpi/operator_resident/OperatorResidentPlan.h`：保存 OR 专属 IR。
- `Rewriter_MPI_Plan.h` 收窄为公共 MPI lowering 总线。

阶段 2：拆 `ShellPartitionAnalysis.cpp`

- `ShellPartitionAnalysis.cpp`：只保留 shell/calc 遍历、plan 组装、日志和 reject 编排。
- `SplitBindAnalysis.cpp`：split kind、bind order、shape helper。
- `ScalarAccessAnalysis.cpp`：`{}` scalar 判定和 calc 参数 `[0]` 使用检查。
- `Direct1DPartitionAnalysis.cpp`：`Contiguous1D` layout 判定。
- `RowBlock2DPartitionAnalysis.cpp`：`RowBlock2D` layout 判定和 phase layout 编排。
- `ShellPartitionAnalysis_Internal.h`：operator_resident 内部 analysis helper 声明。

阶段 3：拆 `OperatorResidentCodegen.cpp`

- `OperatorResidentCodegen.cpp`：只保留 wrapper 顶层拼装入口。
- `OperatorResidentWrapperCodegen.cpp`：wrapper signature、参数命名、元素类型、view 类型。
- `CollectiveCodegenUtils.cpp`：scatter/gather、byte counts/displs、resident-or-scatter。
- `PartitionCodegen.cpp`：1D contiguous 和 2D row-block ownership codegen。
- `LocalKernelCodegen.cpp`：SYCL local kernel launch 和 `ContiguousView1D` 构造。
- `ResidentBufferCodegen.cpp`：scalar broadcast、output local buffer、resident state 写回、final materialize。
- `OperatorResidentCodegen_Internal.h`：operator_resident 内部 codegen helper 声明。

### 9.3 后续建议

Phase 3 前建议继续保持现有目录名 `operator_resident/`，不要急着整体改名为 `shell_derived/`。等 full payload、stencil window、fixed block 都进入 shell-derived planner 后，再考虑：

```text
rewriter/lib/mpi/shell_derived/
  analysis/
  codegen/
  runtime_adapters/
```

runtime 头文件目前还小，可以暂不拆。等支持更多 residency 后再拆为：

```text
dpcppLib/include/mpi/operator_resident/
  OperatorResidentTypes.h
  OperatorResidentStorage.h
  OperatorResidentCollectives.h
  OperatorResidentRuntime.h
```

---

## 10. 负向用例和 fallback

当前仍应 fallback 的情况：

- 当前未接入的 `RegularSplit` 形态
- bind offset 非 0
- `{}` 大 tensor，不是 scalar
- `READ_WRITE` 复杂 alias
- 中间 tensor 在 chain 中被 host 读取
- 2D 非 row-major 或非 `[idx1][idx2]`
- 不兼容 `bindSizes + bindOrder`
- 任何无法证明 local layout 的情况

上述情况当前都应继续保持 legacy 或 Phase-C 输出正确。

---

## 11. 实现陷阱

- 不要把所有 `{}` 当作 scalar。
- 不要把 shell-derived lowering 写成只能识别测试文件名；应识别 shell/dataList pattern。
- 不要内联重写 calc body；优先复用 `*_mpi_local` 和 contiguous view。
- 不要让 accepted OR chain 依赖 legacy `AccessPattern` / `PackPlan`。
- 不要删除 legacy；当前仍需要 fallback 覆盖未接入形态。

---

## 12. 验证计划

基础构建：

```bash
cd /Volumes/QUQ/working/dacpp
cmake --build build --target translator -j8
```

Phase 1/2 正向验证：

```bash
cd clang/tools/translator
bash test_mpi.sh vectorAddCombo imageAdjustment1.0 mandel1.0 decay1.0
```

剩余 fallback 验证：

```bash
bash test_mpi.sh matMul1.0 gradientSum DFT1.0 jacobi1.0 stencil1.0 \
  waveEquation1.0 oddeven0.1
```

结构验证：

- accepted OR generated code 不出现 `AccessPattern` / `PackPlan`。
- `vectorAddCombo` 的 `tmp_tensor` / `shifted_tensor` 不出现 `tensor2Array()` / `array2Tensor()`。
- `imageAdjustment1.0` 的 `image_tensor2` 不出现 `tensor2Array()` / `array2Tensor()`。
- accepted path 出现 `ContiguousView1D` 或 row-block direct buffer。
- final output 才 `MPI_Gatherv`。

---
