# Operator-Resident Communication Model 设计说明

日期：2026-05-06

本文提出一个面向轻量算子链的 MPI 通信模型：**Operator-Resident Communication Model**，中文称为“算子驻留通信模型”。它不是立即替换现有 `AccessPattern` / `PackPlan` 路径，而是作为增量快路径，用来改善 `vectorAddCombo`、`imageAdjustment1.0`、`gradientSum` 这类低算术强度 DAC 程序在 MPI+SYCL 翻译后的通信效率。

## 1. 背景问题

当前普通 MPI wrapper 和部分 stencil fallback 路径以 `AccessPattern` 为核心：

- 每个 shell 参数独立构造 `AccessPattern`。
- 每个参数独立 `build_input_pack_plan()` / `build_output_pack_plan()`。
- 每次 `<->` wrapper 独立准备输入、执行 kernel、收集输出、按需要写回 root tensor。

这种模型的优点是通用：它能从参数访问模式推导每个 item 需要哪些 global index。但在轻量算子链里，它的通信组织粒度偏低。

典型低效形态：

```cpp
VADD(a, b, tmp) <-> vadd;
VSHIFT(tmp, bias, shifted) <-> vshift;
VADD(shifted, c, out) <-> vadd;
```

每个 calc 只做极少计算，但每个 `<->` 都可能重新经历参数级 pack、root 分发、输出 gather、materialize 或后续 host 侧 tensor 同步。此时性能瓶颈不在 kernel，而在固定通信和 tensor 重建成本。

从近期 benchmark 看，低效率用例有共同特征：

- element-wise 或 row-wise reduction，单 item 计算量小。
- 多个 `<->` 串联，中间 tensor 只被后续 `<->` 读取。
- 输出 tensor 在真正 host 读取前反复被分布式算子消费。
- 手写 MPI+SYCL 可以按 rank 直接持有本地切片，而 DAC-MPI 仍按参数重新打包/收集。

因此，新模型应把通信中心从“参数如何被 item 访问”提升到“算子链的数据应该驻留在哪里、什么时候必须通信”。

## 2. 模型目标

算子驻留通信模型的目标：

- 以 `<->` 的输出迭代域作为通信主轴，而不是以每个参数的 `AccessPattern` 作为主轴。
- 对轻量算子链建立稳定的 rank ownership，使输入、输出、中间 tensor 在多个 `<->` 之间复用同一分片。
- 中间 WRITE tensor 默认保持 distributed resident，不立即 gather 回 root。
- 只有遇到 host 侧读取、`print()`、`tensor2Array()`、跨 rank 依赖、最终输出等 materialize 边界时，才收集到 root-visible tensor。
- 标量和小常量使用 replicated/broadcast 状态，不进入 per-item pack。
- 任何不满足快路径条件的场景，保守回退现有 `AccessPattern` 路径。

非目标：

- 不替代 stencil halo / Phase-C route 主路径。
- 不处理复杂非仿射索引、动态 shape、复杂别名或通用 `READ_WRITE` 冲突。
- 不改变 DACPP 源码语法、CLI 或现有普通 MPI wrapper 语义。
- 不要求第一版覆盖所有 `<->`，只覆盖轻量算子链。

## 3. 核心抽象

### 3.1 OperatorDomainPlan

`OperatorDomainPlan` 描述一个 `<->` 的全局输出域和 rank ownership。

建议字段：

```cpp
struct OperatorDomainPlan {
    int64_t total_items;
    std::vector<int64_t> domain_shape;
    std::vector<int64_t> tile_shape;
    ItemRange rank_item_range;
    bool contiguous_linear_domain;
};
```

含义：

- `domain_shape`：输出 tensor 或输出 view 的逻辑形状。
- `total_items`：输出域元素数量。
- `tile_shape`：后续可用于二维 row-block / tile-block 划分；第一版可只支持线性连续切分。
- `rank_item_range`：当前 rank 拥有的输出 item 区间。
- `contiguous_linear_domain`：快路径 guard，表示本 rank 输出在 global storage 上是连续或可简单 stride 表达。

与 `AccessPattern` 的差异：

- `AccessPattern` 说明“某参数对某 item 访问哪些位置”。
- `OperatorDomainPlan` 说明“这个算子由哪些 rank 负责哪些输出 item”。

### 3.2 TensorResidencyState

`TensorResidencyState` 描述一个 tensor 当前的数据驻留状态。

建议状态：

```cpp
enum class ResidencyKind {
    RootOnly,
    DistributedClean,
    DistributedDirty,
    ReplicatedScalar,
    MaterializedRoot,
    Unknown
};
```

建议记录：

```cpp
struct TensorResidencyState {
    std::string tensor_name;
    ResidencyKind kind;
    std::vector<int64_t> global_shape;
    ItemRange owner_range;
    bool owner_matches_current_chain;
    bool root_copy_valid;
    bool distributed_copy_valid;
};
```

语义：

- `RootOnly`：数据只在 root 完整有效，首次进入算子链时需要 scatter/broadcast。
- `DistributedClean`：rank-local 分片有效，root 副本也未失效或可认为同步。
- `DistributedDirty`：rank-local 分片是最新版本，root 完整 tensor 已过期。
- `ReplicatedScalar`：标量或 `{}` 小参数在所有 rank 上都有副本。
- `MaterializedRoot`：刚完成 gather，root 上完整 tensor 有效。
- `Unknown`：无法证明驻留关系，必须回退或 materialize。

### 3.3 OperatorChainPlan

`OperatorChainPlan` 描述一段连续轻量 `<->` 的通信阶段。

建议字段：

```cpp
struct OperatorChainPlan {
    std::vector<OperatorDomainPlan> operators;
    std::unordered_map<std::string, TensorResidencyState> residency;
    std::vector<int> chain_expr_indices;
    bool single_owner_partition;
    bool requires_final_materialize;
};
```

识别条件：

- 多个 `<->` 在同一 basic block 中顺序出现。
- 中间 WRITE tensor 被后续 `<->` 读取，且没有 host 侧读取穿插。
- 每个 `<->` 的输出域可确定，且 rank ownership 可复用或可安全转换。
- 参数访问是简单 element-wise、scalar broadcast、row-wise reduction 或简单矩阵 element-wise。

## 4. 支持模式

第一版建议只支持下面几类轻量模式。

### 4.1 Element-Wise 一对一

形态：

```cpp
dataList{lhs[i], rhs[i], out[i]}
```

通信：

- 按 `out` 的输出域切分 rank。
- root 首次把 `lhs/rhs` 对应切片 scatter 到 owner rank。
- `out` 写入 rank-local distributed buffer。
- 若 `out` 后续继续作为同分片 READ，直接复用本地 buffer。

适用：

- `vectorAddCombo`
- 简单一维 map 算子。

### 4.2 Scalar Broadcast

形态：

```cpp
dataList{in[i], bias[{}], out[i]}
```

通信：

- `bias` 被标记为 `ReplicatedScalar`。
- 使用一次 `MPI_Bcast` 或直接在所有 rank 初始化。
- 不为 `bias` 构造 per-item pack plan。

适用：

- `vectorAddCombo` 中 `VSHIFT` 的 `bias`。

### 4.3 Matrix Element-Wise

形态：

```cpp
dataList{image[idx1][idx2], image2[idx1][idx2]}
```

通信：

- 输出域是二维矩阵。
- 第一版可按 row-block 切分，rank 持有连续行。
- 输入和输出使用相同 row ownership。
- 链式 image pass 之间不 materialize。

适用：

- `imageAdjustment1.0` 两个连续 image kernel。

### 4.4 Row-Wise Reduction

形态：

```cpp
dataList{matGrads[{}][idx1], matNeuronSum[idx1][idx2]}
```

通信：

- 输出域是 neuron / row 维度。
- rank 按输出 row 切分。
- 每个 rank 只接收自己负责 row 的完整 input slice。
- 输出 local sum 保持 distributed，最终按需要 gather。

适用：

- `gradientSum`。

### 4.5 简单临时 Tensor 链

形态：

```cpp
tmp = f(a, b);
shifted = g(tmp, scalar);
out = h(shifted, c);
```

通信：

- `tmp`、`shifted` 不 materialize 到 root。
- 它们在 `TensorResidencyState` 中标记为 `DistributedDirty`。
- 后续 `<->` 如果读取同一 ownership 下的 `tmp/shifted`，直接传 local buffer。

适用：

- `vectorAddCombo`。

## 5. 通信执行策略

### 5.1 Chain Init

进入算子链前：

1. 根据第一个输出 tensor 构造 `OperatorDomainPlan`。
2. 检查链内所有 `<->` 的 domain 是否兼容。
3. 为每个输入 tensor 查询 `TensorResidencyState`。
4. 对 `RootOnly` 输入执行一次按 owner range 的 scatter。
5. 对 `ReplicatedScalar` 输入执行一次 broadcast 或本地复制。
6. 为所有中间 WRITE tensor 分配 rank-local buffer。

### 5.2 Chain Run

链内每个 `<->`：

- 使用同一个 `rank_item_range`。
- kernel view 直接绑定 rank-local input/output buffer。
- 对同分片 distributed tensor 不再 pack/unpack。
- 对 scalar replicated 参数直接传本地 scalar view。
- 对 row-wise reduction 传本 rank 的 rows。

### 5.3 Chain Materialize

离开算子链时：

- 若后续代码不读取输出 tensor，可延迟 materialize。
- 若遇到 `print()`、`tensor2Array()`、host loop 读取、未知函数调用，则 materialize 相关 tensor。
- materialize 使用一次 `MPI_Gatherv` 或 contiguous gather。
- root 副本更新后，状态变成 `MaterializedRoot`。

### 5.4 Fallback

下列情况直接回退现有 `AccessPattern`：

- 输出 domain 无法确定。
- 参数 index 非简单仿射或含未知表达式。
- 同一 tensor 同时有复杂 `READ_WRITE`。
- host 侧在链中间读取中间 tensor。
- 多个输出 tensor ownership 不一致且无法证明安全转换。
- 动态 shape 或 runtime-dependent split。

## 6. 与当前实现的衔接

建议作为 translator 分析阶段的新快路径：

1. 在普通 MPI wrapper 生成前扫描 basic block 中连续 `<->`。
2. 尝试构造 `OperatorChainPlan`。
3. 成功时生成 operator-resident wrapper。
4. 失败时不改变现有行为，继续走 `AccessPattern`。

和现有 Phase-C stencil 的关系：

- Phase-C 解决的是 loop stencil 的 distributed followup / halo / route。
- Operator-resident 解决的是轻量算子链的中间 tensor 驻留。
- 两者都使用“延迟 materialize”的思想，但适用场景不同。
- 第一版不复用 Phase-C route IR，避免扩大实现风险；后续可以共享 `DistributedTensorState` 和 profiling 设施。

建议最小 codegen 形态：

```cpp
__dacpp_mpi_operator_chain_ctx_xxx ctx;
__dacpp_mpi_operator_chain_init_xxx(ctx, ...);
__dacpp_mpi_operator_chain_run_xxx(ctx, ...);
__dacpp_mpi_operator_chain_materialize_xxx(ctx, ...);
```

对于没有 host materialize 需求的链，`materialize()` 可以为空或只在 root 输出需要时调用。

## 7. 首批落地用例

### 7.1 vectorAddCombo

目标：

- 三个 `<->` 合并为一个 operator chain。
- `a/b/c` 首次 scatter。
- `bias` replicated。
- `tmp/shifted/out` distributed resident。
- 只在最终 `tensor2Array(host_out)` 前 materialize `out`。

预期收益：

- 避免 `tmp`、`shifted` 每步 root gather/writeback。
- 避免 scalar bias per-item pack。
- 减少 wrapper 调用固定成本。

### 7.2 imageAdjustment1.0

目标：

- 两个 image element-wise `<->` 共用 row-block ownership。
- `image2` 保持 distributed resident。
- 只在最终 print/materialize 时 gather `image3`。

预期收益：

- 避免 4096x4096 级中间 image materialize。
- 将通信收敛为输入分发和最终输出收集。

### 7.3 gradientSum

目标：

- 按输出 neuron row 切分。
- 每 rank 只持有自己负责的 `matGrads` rows。
- `matNeuronSum` distributed resident，最终按需 gather。

预期收益：

- 避免按每个 item 反复构造大 slice pack。
- 把 row-wise reduction 转成 rank-local row block kernel。

## 8. 文件结构重构建议

当前 MPI translator 相关实现主要平铺在 `rewriter/lib/mpi/` 和两个公共头里：

- 普通 wrapper / `AccessPattern` 代码。
- stencil Phase-C 分析和 codegen。
- post-region root helper。
- output sync / print rewrite / param mode helper。

这种结构在功能还少时可以接受，但继续加入 operator-resident 快路径会让新旧模型共用同一批散函数和公共结构，后续很难判断某个分析结果服务于普通 wrapper、stencil，还是轻量算子链。因此建议先做目录和接口分层，再做新模型实现。

### 8.1 目标分层

建议把 MPI rewriter 分成四类模块：

```text
rewriter/lib/mpi/
  shared/
    MpiTypes.cpp
    OutputSyncAnalysis.cpp
    PrintRewrite.cpp
    ParamModeAnalysis.cpp
    AnalysisContext.cpp

  legacy_access_pattern/
    PatternInit.cpp
    WrapperPlan.cpp
    WrapperCodegen.cpp

  stencil_phase_c/
    StencilAnalysis.cpp
    StencilRouteParse.cpp
    StencilFollowupCollect.cpp
    StencilCodegen.cpp
    StencilCodegenUtils.cpp
    WaveSpecialization.cpp
    PostRegionAnalysis.cpp
    PostRegionCodegen.cpp

  operator_resident/
    OperatorChainAnalysis.cpp
    OperatorDomainAnalysis.cpp
    ResidencyAnalysis.cpp
    OperatorChainCodegen.cpp
```

命名不要求一次到位，但边界要明确：

- `shared/` 只放模型无关的基础工具，比如 MPI datatype、输出同步分类、root-only print rewrite、参数 mode 推断。
- `legacy_access_pattern/` 保留现在以 `AccessPattern` / `PackPlan` 为中心的普通 wrapper。
- `stencil_phase_c/` 保留 loop stencil、route、halo、deferred materialize 和 wave specialization。
- `operator_resident/` 只处理轻量算子链快路径，不反向依赖 legacy wrapper codegen。

### 8.2 Runtime 头文件分层

当前 runtime 头文件也混合了 pattern、wrapper、stencil、view 和 exchange。建议保持聚合头兼容，但内部语义分组更清楚：

```text
dpcppLib/include/mpi/
  common/
    CoreTypes.h
    MpiTypes.h
    Profile.h

  legacy_access_pattern/
    Pattern.h
    PackMap.h
    WrapperPack.h
    Wrapper.h

  stencil/
    StencilTypes.h
    StencilLayout.h
    StencilExchangePlan.h
    StencilExchangeRuntime.h
    WaveExchangeSpecialization.h

  operator_resident/
    OperatorResidentTypes.h
    TensorResidency.h
    OperatorChainRuntime.h
```

对外仍可保留 `MPIPlanner.h`、`mpi/Common.h`、`mpi/Wrapper.h`、`mpi/Stencil.h` 等聚合入口，避免一次性改动大量 include。新 runtime 头不要直接包含 legacy `Pattern.h`，除非走 fallback adapter。

### 8.3 老实现和新实现的关系

老实现不应被删除，而应明确命名为 legacy access-pattern path：

- 普通 wrapper 默认仍走 legacy。
- operator-resident 分析失败时直接返回 `Unsupported(reason)`，由调度层选择 legacy。
- stencil Phase-C 不经过 operator-resident，也不被 operator-resident fallback 影响。
- benchmark 和结构断言中要能区分生成代码走的是 `legacy_access_pattern` 还是 `operator_resident`。

建议生成 debug log：

```text
[DACPP][MPI][OR] chain accepted length=3 domain=1D contiguous materialize=out
[DACPP][MPI][OR] chain rejected reason=host read between ops; fallback=legacy_access_pattern
```

## 9. 分析接口重构建议

当前 `Rewriter_MPI_Common.h` 暴露了大量散函数，调用方需要自己组合 `Shell*`、`Calc*`、`BinaryOperator*`、param modes、output sync、post-region、stencil site plan 等信息。新模型需要先建立统一分析入口，否则 operator-resident 会继续复制一套零散判断。

### 9.1 统一上下文

建议引入只读分析上下文：

```cpp
struct MpiAnalysisContext {
    DacppFile* file = nullptr;
    clang::ASTContext* ast = nullptr;
    const clang::SourceManager* sourceManager = nullptr;
    const clang::LangOptions* langOptions = nullptr;
};
```

所有 MPI 分析函数都从该上下文读取公共状态，减少到处传 `DacppFile*` 和 AST helper 的散乱依赖。

### 9.2 统一表达式节点

建议把一个 `<->` 包装成稳定 IR：

```cpp
struct DacExprNode {
    int exprIndex = -1;
    Shell* shell = nullptr;
    Calc* calc = nullptr;
    const clang::BinaryOperator* expr = nullptr;
    const clang::Stmt* parentStmt = nullptr;
};
```

后续 legacy wrapper、stencil、operator-resident 都基于 `DacExprNode` 做分析，而不是各自重新从 `Shell/Calc/BinaryOperator` 拼上下文。

### 9.3 分析结果统一返回

建议所有候选路径返回统一 result：

```cpp
enum class MpiPlanKind {
    LegacyAccessPattern,
    StencilPhaseC,
    OperatorResident,
    Unsupported
};

struct MpiPlanResult {
    MpiPlanKind kind = MpiPlanKind::Unsupported;
    std::string reason;
};
```

operator-resident 可扩展为：

```cpp
struct OperatorResidentPlan : MpiPlanResult {
    OperatorChainPlan chain;
    std::vector<DacExprNode> exprs;
};
```

stencil 可扩展为：

```cpp
struct StencilPhaseCPlan : MpiPlanResult {
    DistributedStencilSitePlan site;
    DacExprNode expr;
};
```

legacy wrapper 可扩展为：

```cpp
struct LegacyWrapperPlan : MpiPlanResult {
    DacExprNode expr;
    std::vector<IOTYPE> effectiveModes;
};
```

### 9.4 调度接口

建议新增一个 MPI lowering planner：

```cpp
struct MpiLoweringPlan {
    std::vector<OperatorResidentPlan> operatorChains;
    std::vector<StencilPhaseCPlan> stencilSites;
    std::vector<LegacyWrapperPlan> legacyWrappers;
};

MpiLoweringPlan buildMpiLoweringPlan(const MpiAnalysisContext& ctx);
```

调度规则固定为：

1. 先识别 loop stencil site，保持现有 Phase-C 优先级。
2. 对未被 stencil 接管的 `<->`，尝试 operator-resident chain。
3. 对剩余 `<->`，生成 legacy access-pattern wrapper。
4. 任一新路径 unsupported，不抛错，不改变语义，只记录 reason 并回退。

这样可以避免未来在 `rewriteMPI()` / `rewriteMPIStencil()` 中继续叠加特殊分支。

### 9.5 分析接口的验收标准

重构后的接口应满足：

- 任意 `<->` 只被一个 plan owner 接管。
- fallback reason 可打印，可用于结构测试。
- codegen 不直接调用 AST 搜索函数；codegen 只消费 plan。
- legacy path 的输出代码与重构前保持一致。
- operator-resident 可以在不改 legacy wrapper 的情况下独立迭代。

## 10. 验证计划

baseline 使用当前 benchmark：

| 用例 | 当前标准 MPI+SYCL(s) | 当前 DAC-MPI(s) | 当前效率 |
|---|---:|---:|---:|
| vectorAddCombo | 0.675558 | 4.901816 | 0.1378 |
| imageAdjustment1.0 | 0.796038 | 15.894185 | 0.0501 |
| gradientSum | 0.727669 | 1.631674 | 0.4460 |

第一阶段正确性验证：

- 生成结果与现有 DAC-MPI 输出一致。
- 与手写标准 MPI+SYCL 输出一致。
- 生成代码不得残留 `<->`。
- fallback case 仍走现有 `AccessPattern`。

第一阶段结构验证：

- `vectorAddCombo` 中 `tmp/shifted` 不出现 root gather materialize。
- `imageAdjustment` 中 `image2` 不在两个 kernel 之间 materialize。
- `gradientSum` 中输入按 row-block 分发，不按每个 output item 重复 pack 全 row。
- scalar `{}` 参数生成 broadcast/replicated 路径。

第一阶段性能目标：

- `vectorAddCombo`、`imageAdjustment1.0`、`gradientSum` 的 DAC-MPI 时间明显接近手写标准 MPI+SYCL。
- 至少消除 5x 以上的固定通信开销差距。
- 若无法达到标准实现性能，也应在 profile 中看到 `input/writeback/materialize` 次数减少。

负向验证：

- 复杂 index 表达式回退。
- `READ_WRITE` 复杂别名回退。
- 链中间存在 host 侧读取时强制 materialize。
- 不影响 stencil Phase-C suite。

结构重构验证：

- 重构前后完整 MPI suite 结果一致。
- legacy wrapper 生成文本的关键结构保持一致。
- stencil Phase-C 结构断言不变化。
- 新增 planner debug log 能显示每个 `<->` 的 plan owner。

## 11. 风险和开放问题

主要风险：

- DAC tensor/view 可能存在隐式 host 读取，分析必须保守。
- 中间 tensor 的 root 副本失效后，后续普通 C++ 代码若读取，需要准确插入 materialize。
- 一维 linear domain 与二维 row-block domain 的统一抽象需要谨慎，否则容易破坏矩阵 storage 顺序。
- 与现有 wrapper/stencil 两套 codegen 共存时，需要避免重复 MPI init/finalize 或重复 output gating。

开放问题：

- 是否将 `TensorResidencyState` 设计成 translator 侧 IR，还是 runtime ctx 字段。
- 是否复用 Phase-C 的 `DistributedTensorState<T>`，或为轻量链单独定义更薄的 runtime state。
- 是否允许跨 chain 复用 residency，还是第一版只在单个 basic block 内有效。
- 是否为 operator-resident 路径增加独立 profile 输出，例如 `chain_input/chain_kernel/chain_materialize`。
- 文件搬迁是否一次性完成，还是先建立新目录并逐步迁移。
- `Rewriter_MPI_Common.h` 是否拆成多个 public header，还是先保留 facade 头做兼容。

## 12. 建议实施顺序

第一步：建立 planner 和目录骨架，不改变生成行为。

- 新增 `shared/`、`legacy_access_pattern/`、`stencil_phase_c/`、`operator_resident/` 目录。
- 新增 `MpiAnalysisContext`、`DacExprNode`、`MpiLoweringPlan`。
- 先让所有 `<->` 都进入 legacy plan，生成代码保持不变。

第二步：把老实现显式收口为 legacy access-pattern path。

- `PatternInit`、`WrapperCodegen`、普通 wrapper plan 归入 legacy 命名空间或目录。
- `Rewriter_MPI_Common.h` 保留 facade，但新代码不再继续往里面追加散函数。

第三步：把 stencil Phase-C 与普通 wrapper 解耦。

- stencil 分析只消费 `DacExprNode` / `MpiAnalysisContext`。
- stencil codegen 只消费 `StencilPhaseCPlan`。
- 保持现有 Phase-C suite 全过。

第四步：只做 operator-resident 分析和结构断言。

- 识别 `vectorAddCombo` 的三段链。
- 输出 debug log：chain length、domain shape、resident tensors、materialize boundary。
- 不改变生成代码。

第五步：实现 1D element-wise chain。

- 支持 `vectorAddCombo`。
- 实现 replicated scalar。
- 实现最终 materialize。

第六步：实现 2D matrix element-wise row-block。

- 支持 `imageAdjustment1.0`。
- 中间 image tensor distributed resident。

第七步：实现 row-wise reduction。

- 支持 `gradientSum`。
- 输出按 row ownership 分布。

第八步：扩展 profile 和 benchmark。

- 与当前 benchmark 表做对比。
- 保留 `AccessPattern` fallback 性能作为安全基线。
