# Shell-Derived MPI Lowering 与 Operator-Resident 通信模型

日期：2026-05-07
分支：`tqc-2`

本文合并并取代：

- `archive/operator_resident_communication_model_2026-05-06.md`
- `archive/operator_resident_progress_2026-05-06.md`
- 本文件旧版 Phase 1/2 实现计划

本文档是后续 MPI lowering 工作的 canonical 设计、进度和文件结构计划。

阅读本文档时应区分两类边界：

- **语义覆盖边界**：某个 layout / followup / route 形态是否已经能正确 lowering。
- **效率优化边界**：在语义已经正确后，是否继续减少 full sync、重复 scatter/gather 或 materialize。

Phase 1-4 主要收语义覆盖，Phase 4.5 建立 loop-lowered direct/resident 骨架，Phase 4.6 才进入效率优化；Phase 5 是新的 `FixedBlock` layout，不是 Phase 4.x 的性能补丁。

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

- 不处理复杂非仿射索引、动态 shape、复杂别名或通用 `READ_WRITE` 冲突。
- 不改变 DACPP 源码语法、CLI 或现有普通 MPI wrapper 语义。
- 不用 shell-derived 路径强行接管尚未定义语义的 stencil / fixed-block 形态；这些形态必须按阶段逐个进入。

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

循环提升目标：

长期目标是让位于 `for` / `while` 循环内且可证明稳定的 `<->` 使用统一的 loop-lowered shell-derived 结构。该结构把循环不变的通信和 metadata 放入 `init()`，把每步变化的数据和 kernel 放入 `run()`，并在可观察边界使用 `materialize()` 恢复 root-visible tensor：

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
- 当前 OR `stencil_window` wrapper 已承载当前切片所需的 boundary-local、writer-to-reader route 和 read-cache transition，但仍保留较多 full sync；减少这些通信属于 Phase 4.6。
- `direct_1d` backend 已有 Phase 4.5 首个窄切片：稳定 loop 内的 OR `Contiguous1D` direct/resident 可生成 `init/run/materialize` family。`decay1.0` 的当前形态是：`N0s` / `lambdas` 在 `init()` scatter/cache 一次，`t` 在 `run()` broadcast，output 在每轮保守 materialize；减少这类同步属于 Phase 4.6。`row_block_2d` / `full_payload` 的 loop-lowered family 尚未落地。
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

`collectSplitBindMeta(shell)` / `Shell::GetBindInfo()` 的结果作为 `bindId` 来源。当前已实现 layout 大多只接受 offset 为 `"0"`；更复杂 offset 应按对应 layout 单独定义和测试。

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

当前已进入 shell-derived / OR codegen 的 layout：

- `Contiguous1D`
- `RowBlock2D`
- `RowPartitionFullRow`
- `ReplicatedFullTensor`
- `StencilWindow1D`
- `StencilWindow2D`

`ReplicatedScalar` 是参数 residency / access kind，不是独立输出 layout。`FixedBlock` 仍是长期覆盖目标的 IR 预留。

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

当前已使用的 residency kind：

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
[DACPP][MPI][OR][P4.5] expr=... shell=... layout=Contiguous1D loop-lower=candidate structure=init/run/materialize
[DACPP][MPI][OR][P4.5] expr=... shell=... layout=... loop-lower=rejected reason=...
```

---

## 6. 当前实现进度总览

截至 2026-05-09，shell-derived MPI lowering 的当前状态如下。以下状态以代码中的已启用 layout、已接入 codegen 路径，以及完整 `bash test_mpi.sh` 结果为准。

| Phase | 状态 | 本阶段交付能力 | 代表测试 | 剩余/下一阶段边界 |
|---|---|---|---|---|
| Phase 1 | 已完成 | `Contiguous1D` direct / resident chain，含 `ReplicatedScalar` | `vectorAddCombo` `mandel1.0` `decay1.0` | loop-lowered 结构已由 Phase 4.5 首切片接入，通用化仍在 P4.5 |
| Phase 2 | 已完成 | `RowBlock2D` direct / resident chain | `imageAdjustment1.0` | 仅覆盖当前 row-major 2D row-block 形态 |
| Phase 3 | 已完成 | `ReplicatedFullTensor`、`RowPartitionFullRow`、受限 `READ_WRITE OutputDirect` | `DFT1.0` `gradientSum` `jacobi1.0` `matMul1.0` | 更广的 full-payload 形态仍未展开 |
| Phase 4 | 当前验收切片已收口 | `StencilWindow1D` / `StencilWindow2D` 当前切片已接入；OR-side distributed-followup / read-cache / boundary-local lowering 已接入；mixed-site 按 expr 分派已接入 | `stencil1.0` `waveEquation1.0` `FOuLa1.0` `MDP1.0` `liuliang1.0` `mpiMixedStencilORPhaseC` `mpiOrStencilRefreshPolicy1D` `mpiOrStencilBoundaryStride2D` | root-bridge 和更广 stencil 形态属于后续 stencil 语义扩展，不阻塞当前 P4 验收 |
| Phase 4.5 | 首切片已落地 | 稳定 loop 内 OR `Contiguous1D` direct/resident 可生成 `init/run/materialize`；当前只覆盖 `decay1.0` 形态 | `decay1.0` `mpiLoopReadWriteReject1D` `mpiLoopAliasReject1D` `mpiLoopVariantInputReject1D` `mpiLoopChainReject1D` | `RowBlock2D` / full-payload loop-lowered family 未覆盖；普通 buffer translator 对部分 loop body 形态的 baseline 编译仍待后续修复 |
| Phase 4.6 | 未开始 | 仅有设计目标：效率优化层 | 无 | stencil/direct resident 通信优化、长期 resident halo、减少 full sync 尚未落地 |
| Phase 5 | 未开始 | 仅有 IR / 枚举占位（如 `FixedBlock`） | 无 | `oddeven0.1` 所需分析与 codegen 均未实现 |
| Phase 6 | 未完成 | 完整测试已通过，但尚未做到“全部非 `mpi*` tests 都走新路径” | `bash test_mpi.sh` 37/37 | 仍存在需要继续 fallback 的普通应用测试 |

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

已通过（2026-05-09）：

```bash
cmake --build build --target translator -j8

cd clang/tools/translator
bash test_mpi.sh
```

当前基线结果：

- `37 tests | 37 passed | 0 failed | 0 skipped`
- 默认 suite 已包含 `mpiLoopReadWriteReject1D`、`mpiLoopAliasReject1D`，用于钉住 P4.5 首切片不能误接管的 `READ_WRITE OutputDirect` 和 direct-read/output-write alias 边界
- 默认 suite 已包含 `mpiLoopVariantInputReject1D`、`mpiLoopChainReject1D`，通过 `mpi_expect.txt` 中的 `MPI_STRUCTURE_ONLY:1` 走 translate-only / structure-only 路径，只执行 `dacpp --mode=buffer --mpi` 并检查 log / generated SYCL 结构，不依赖普通 buffer baseline 编译运行
- 已接入 OR 的路径不再生成 legacy `AccessPattern` / `PackPlan`
- mixed-site 已按 expr 独立分派，不再因同文件存在未接入 stencil site 而整文件退回 Phase-C

### 6.3 Phase 路线图和验收契约

后续实现和 review 应优先按本节判断“当前改动属于哪个阶段”。阶段编号不是性能等级，而是风险边界：先证明 layout 和 host-visible 语义正确，再把稳定形态抽成 loop-lowered 骨架，再减少通信成本，最后进入新的 `FixedBlock` layout。

| Phase | 阶段定位 | 准入条件 | 必须交付 | 明确非目标 | 退出标准 |
|---|---|---|---|---|---|
| Phase 1 | 1D direct / resident chain | 单 bind 1D direct map；输出 ownership 可由 shell domain 唯一确定 | `Contiguous1D` ownership、scalar broadcast、简单 chain residency、中间 tensor 不立即 materialize | 2D row-block；loop-lowered `init/run/materialize`；stencil | `vectorAddCombo`、`mandel1.0`、`decay1.0` 等 1D/direct 路径正确，accepted OR path 不依赖 legacy pack |
| Phase 2 | 2D row-major direct / resident chain | 2D row-major direct map；row-block ownership 可由第 0 维稳定切分 | `RowBlock2D` input/output 分发、本地 row-block kernel、最终 gather、chain residency | stencil window；full-payload；FixedBlock | `imageAdjustment1.0` 走 OR row-block，结果与 baseline 一致 |
| Phase 3 | full-payload / row-partition payload | full tensor 或 full row payload 可安全 replicated / row partition；alias 范围受限 | `ReplicatedFullTensor`、`RowPartitionFullRow`、受限 `READ_WRITE OutputDirect` | stencil window；loop-lowered direct family；复杂 alias 扩展 | `DFT1.0`、`gradientSum`、`jacobi1.0`、`matMul1.0` 走 OR full-payload 路径 |
| Phase 4 | stencil window 当前验收切片 | site 可判定为当前支持的 OR `StencilWindow1D/2D`；followup / boundary-local 可在 wrapper 内等价重放 | `StencilWindow1D/2D` codegen、mixed-site expr 级分派、当前 accepted distributed-followup / read-cache / boundary-local lowering | `RootCentricFollowup` 伪支持；root-bridge；FixedBlock；把性能优化作为收口条件 | `FOuLa1.0`、`MDP1.0`、`liuliang1.0`、`stencil1.0`、`waveEquation1.0` 走 OR stencil 当前切片；P4 stencil 守护用例保持通过；当前完整 `test_mpi.sh` 为 37/37 |
| Phase 4.x stencil 扩展 | 更广 stencil 语义形态 | 已有 P4 切片保持稳定；新增形态能定义清楚 post-shell 语义 | 更广 1D/2D route、direct reader、boundary-local、可能的 root-bridge 语义支持 | 性能优化替代语义证明；影响 P4.5/P4.6 的 direct/resident 路线 | 每个新增 stencil 形态有正向和负向结构测试，unsupported 形态继续 fallback |
| Phase 4.5 | loop-lowered direct/resident 骨架 | P1-P3 direct/full-payload 语义稳定；loop 中 shell 参数、shape、domain 可证明稳定 | 对稳定 OR direct/full-payload layout 生成 `init/run/materialize` family，hoist loop-invariant metadata / collective setup | 新增 stencil 形态；root-bridge stencil；FixedBlock；通信优化和新语义混在一个 patch | `decay1.0` 已出现 direct/resident `init/run/materialize` 结构并保持正确；READ_WRITE output / alias / loop-chain 等未证明形态有负向边界 |
| Phase 4.6 | 已支持路径效率优化 | 相关语义 lowering 已正确；测试能覆盖 host-visible tensor 状态 | 减少 full broadcast/gather/materialize；推进 stencil resident halo / 分片同步；复用 P4.5 骨架降低 direct/resident loop 成本 | 牺牲语义换性能；新增 FixedBlock；把 root-bridge stencil support 伪装成优化 | benchmark 对比手写 MPI+SYCL 的差距收敛；结构上减少不必要 full sync |
| Phase 5 | FixedBlock / odd-even 新 layout | `FixedBlock` ownership、local block/pair 依赖和边界交换语义已定义 | 新增 `FixedBlock` 分析、ownership、codegen，覆盖 `oddeven0.1` 这类 fixed pair/block partition | 背负 P1-P4 已有路径的性能债；重写 stencil；替代 P4.6 | `oddeven0.1` 不再 fallback，FixedBlock 正向/负向结构测试通过 |
| Phase 6 | 非 `mpi*` 应用全量新路径覆盖 | P1-P5 各 layout 均有稳定 fallback 边界和结构测试 | 将普通应用测试逐步迁移到 shell-derived planner，减少 legacy fallback 覆盖面 | 要求删除 legacy fallback；把“测试通过”等同于“覆盖完成” | 非 `mpi*` tests 中除明确 unsupported 形态外均由 shell-derived path 接管 |

阶段依赖说明：

- Phase 4 当前验收切片可以收口；更广 1D/2D stencil 或 root-bridge 应作为后续 Phase 4.x stencil 扩展单独评审。
- Phase 4.5 只建立 direct/resident 的 loop-lowered 骨架，避免和 stencil 语义扩展互相污染。
- Phase 4.6 可以在 Phase 5 之前做，因为它优化的是 P1-P4.5 已经语义正确的路径。
- Phase 5 完成后，FixedBlock 可以复用 Phase 4.6 沉淀出的 runtime/helper 思路；FixedBlock 自己的性能优化应作为 P5 后续，不阻塞 P4.6。
- root-bridge stencil support 属于“更广 stencil 形态”，不是 Phase 4.6 的性能优化项；除非先定义并验证语义，否则不应通过移除通信来假装优化。

---

## 7. 非 `mpi*` tests 当前覆盖状态

下表描述的是“当前实际走到哪里”，不是长期目标。

| 测试 | 当前 lowering | 状态说明 |
|---|---|---|
| `vectorAddCombo` | OR `Contiguous1D` | 已完成 |
| `mandel1.0` | OR `Contiguous1D` | 已完成 |
| `decay1.0` | OR `Contiguous1D` + `ReplicatedScalar` + P4.5 loop-lowered family | 已完成；`N0s` / `lambdas` scatter/cache 进入 `init()`，`t` 留在 `run()` broadcast；当前首切片仍每轮保守 materialize output |
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

## 8. 后续阶段执行说明

### 8.1 Phase 4：Stencil Window

Phase 4 的当前验收切片已经收口。这里的“收口”指当前承诺的 OR stencil 语义形态已经能正确 lowering，并且不再依赖 legacy `AccessPattern` / `PackPlan` 或保守 materialized writer broadcast 才能维持 accepted path 语义。

当前已完成：

- `StencilWindow1D` 已进入 OR codegen
- `StencilWindow2D` 已进入 OR codegen
- `stencil1.0` 已接入
- `waveEquation1.0` 已接入
- `FOuLa1.0`、`MDP1.0`、`liuliang1.0` 已接入
- mixed-site 顶层分派已支持按 expr 独立选择 OR / Phase-C / legacy
- OR-side post-shell distributed followup lowering 已接入当前 accepted 1D/2D stencil 切片
- `mpiOrStencilRefreshPolicy1D` 已更新，用于钉住 OR stencil distributed-followup 不再依赖 materialized writer broadcast、且同步 followup reader 的口径
- `mpiOrStencilBoundaryStride2D` 已加入，用于钉住非 unit-step 2D boundary-local loop 不被误 lower

当前边界：

- direct writer 仍必须是 WRITE-only，不接受 `READ_WRITE` direct writer
- 1D 当前只覆盖 scalar root-materialize 形态和 `+1` distributed-followup 形态
- 2D 当前只覆盖当前 row-major ownership 切片
- 目前最多支持一个额外的 2D direct reader
- direct reader 当前只覆盖 `waveEquation1.0` 所需的 `read-cache transition (-1,-1)` 形态
- 仍不支持 root-bridge stencil site

Phase 4 后续如果继续扩语义，应作为 “Phase 4.x stencil 扩展” 处理，重点是：

- 更广的 1D stencil window 形态
- 更广的 2D direct reader / route 形态
- root-bridge stencil site

这些扩展不应改变“当前 Phase 4 验收切片已收口”的结论，也不应横向扩到 `FixedBlock`。

### 8.2 Phase 4.5：Loop-Lowered Direct / Resident

这一阶段应保持窄化定义：只做已经有 OR layout 的 direct / resident family 的 loop lowering，不继续扩 stencil route，不做 root-bridge stencil，也不进入 `FixedBlock`。

Phase 4.5 的长期目标是把“每次遇到 `<->` 就完整执行 wrapper”的形态，提升为可跨 loop 生命周期复用的 direct/resident family：

- loop 外：生成 `init`，完成可 hoist 的 buffer / ownership / collective metadata 初始化。
- loop 内：生成 `run`，只执行每轮必要的 kernel、必要输入同步和输出 resident 更新。
- loop 后或 host-visible 边界：生成 / 调用 `materialize`，把 distributed resident 输出恢复为 root-visible tensor。当前首切片为了语义保守，允许在 `run()` 内 per-run materialize；减少这类同步属于 Phase 4.6。

P4.5 后续扩展对象应优先来自已经稳定的 OR direct/full-payload 路径，但必须逐个证明 loop 稳定性和 host-visible 语义：

- `Contiguous1D`
- `RowBlock2D`
- `ReplicatedFullTensor`
- `RowPartitionFullRow`
- 受限 `READ_WRITE OutputDirect` 只属于未来可能扩展；当前 P4.5 gate 明确 reject。

该阶段不应做：

- 新增 stencil followup / route / boundary-local 形态。
- 新增 root-bridge stencil 支持。
- `FixedBlock` / `oddeven0.1`。
- 把通信性能优化和新语义形态混在一个 patch 里。

当前推进位置：

- `decay1.0` 已进入 P4.5 loop-lowered OR `Contiguous1D` family。
- 生成结构包含 `ctx/init/run/materialize`：`init()` 建立 ownership、counts/displs、stable local buffers，并 scatter/cache loop-invariant `N0s` / `lambdas`；`run()` 逐轮 broadcast scalar `t`、执行 kernel，并在当前首切片里 per-run materialize output；after-loop `materialize()` 入口仍生成，用作非 per-run 形态的 root-visible 恢复边界。
- 当前首切片对 `decay1.0` 保留 per-run output materialize；减少这类通信属于 Phase 4.6，不作为 P4.5 首切片目标。
- P4.5 candidate 当前代码只接受稳定 `for` / `while` loop 内的 OR `Contiguous1D` direct/resident，且必须满足：outer loop 内恰好一个 DAC expr、存在 `OutputDirect`、输出不是 `READ_WRITE OutputDirect`、存在可 hoist 的 `DirectMapped` read input、direct read input 不在 loop AST 中被显式写入、direct read input 的 actual tensor 不 alias 任一 output direct write。
- `mpiLoopReadWriteReject1D` 和 `mpiLoopAliasReject1D` 已纳入默认 `test_mpi.sh`，用于钉住不应 P4.5 接管的负向边界。
- `mpiLoopVariantInputReject1D` 和 `mpiLoopChainReject1D` 已通过 `MPI_STRUCTURE_ONLY:1` 纳入默认 `test_mpi.sh` 的 translate-only / structure-only 路径，分别钉住 loop-variant direct read input 和 loop 内多 DAC expr chain 不被 P4.5 首切片误接管。
- `RowBlock2D`、`ReplicatedFullTensor`、`RowPartitionFullRow`、受限 `READ_WRITE OutputDirect` 的 loop-lowered family 尚未覆盖；这些形态继续走普通 OR wrapper 或 fallback。

因此，Phase 4.5 当前应视为“首个 Contiguous1D direct/resident loop-lowered 骨架已落地，通用化仍未完成”。

Phase 4.5 TODO 状态：

- 已完成：增加 `translate-only` / `structure-only` MPI 测试 harness。测试可在 `mpi_expect.txt` 中声明 `MPI_STRUCTURE_ONLY:1`，使 `test_mpi.sh` 跳过普通 buffer baseline 翻译/编译/运行和 MPI 编译/运行，只执行 `dacpp --mode=buffer --mpi` 并检查 `mpi_expect.txt`。
- 已完成：`mpiLoopVariantInputReject1D`、`mpiLoopChainReject1D` 已纳入自动验证路径，覆盖 loop-variant input、loop 内 resident chain / 多 DAC expr 负向结构。
- 修复或扩展普通 buffer translator 对 loop body 的覆盖：支持 `<->` 后 sibling statement 正确捕获外部 tensor 变量，支持同一 loop 多个 `<->` 的 helper signature / call args 一致，或保守 fallback 到逐 `<->` 普通替换。该任务是 P4.5 测试基础设施/前置质量 TODO，不应与扩大 P4.5 lowering 语义混在一个 patch。
- 仍待处理：`RowBlock2D`、`ReplicatedFullTensor`、`RowPartitionFullRow`、受限 `READ_WRITE OutputDirect` 的 loop-lowered family 尚未覆盖；扩大接受范围前必须先补明确结构测试和语义证明。

### 8.3 Phase 4.6：Efficiency Optimization

这一阶段是 Phase 4 / Phase 4.5 之后的效率优化层。它的前提是相关语义 lowering 已经正确，并且测试可以证明不会改变 host-visible tensor 状态。它可以在 Phase 5 之前推进，因为它不依赖 `FixedBlock` 语义。

Phase 4.6 的目标不是新增更多 stencil 语义形态，而是降低已支持路径的通信和 materialize 成本：

- 对 OR stencil 2D：减少每步 full reader broadcast、full writer gather、followup/read-cache full tensor sync，逐步转向长期 resident halo / 分片同步。
- 对 OR stencil 1D：在保证 distributed-followup 后 host-visible reader 语义的前提下，减少不必要的全量 reader sync。
- 对 direct/resident family：复用 Phase 4.5 的 `init/run/materialize` 骨架，减少循环内重复 scatter/broadcast/gather。
- 对 benchmark：把 `MDP1.0`、`liuliang1.0`、`stencil1.0`、`waveEquation1.0` 和手写 MPI+SYCL 的差距作为主要观测对象。

推荐拆分顺序：

1. 先做 measurement-only benchmark，把旧实现、新实现和手写 MPI+SYCL 的通信次数与运行时间口径固定。
2. 再做 direct/resident loop 成本优化，因为它主要复用 Phase 4.5 的 skeleton，语义风险较低。
3. 再做 1D stencil reader/followup sync 收窄，必须保留 distributed-followup 后 host-visible reader 正确性测试。
4. 最后做 2D stencil resident halo / 分片同步，因为它会触碰 writer gather、reader broadcast、boundary-local replay 和 read-cache transition 的组合语义。

Phase 4.6 的明确边界：

- 不以牺牲语义为代价去移除 broadcast / gather。
- 不新增 `FixedBlock`。
- 不把 root-bridge stencil support 伪装成性能优化；root-bridge 是否支持仍属于更广 stencil 形态工作。

因此，Phase 4.6 应理解为“已完成语义 lowering 之后的性能收敛阶段”，不是 Phase 4 收口条件。

### 8.4 Phase 5：Fixed Block / Odd-even

这一阶段当前仍是占位状态。Phase 5 的目标是新增一种 layout family，而不是优化已有 stencil/direct 路径。

代码现状：

- `LocalLayoutKind::FixedBlock` 已存在
- `ParamAccessKind::FixedBlock` 已存在
- 但没有对应的分析入口、layout 判定或 codegen

Phase 5 应交付：

- `FixedBlock` ownership / block range / neighbor relation 的分析结果。
- fixed block 输入输出的 local buffer 和 boundary exchange codegen。
- `oddeven0.1` 对应的正向结构测试。
- 对不满足 fixed block 条件的 odd-even 变体保留 fallback 的负向测试。

Phase 5 不应要求 Phase 4.6 已经完成。若 P4.6 已经沉淀出通用 resident storage、halo exchange 或 materialize helper，P5 可以复用；若没有，P5 也可以先用自己的最小 helper 证明 `FixedBlock` 语义正确，再把性能优化留给后续。

因此，`oddeven0.1` 当前仍应继续 fallback，Phase 5 尚未进入可验证实现阶段。

### 8.5 Phase 6：非 `mpi*` tests 全量新路径

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
- `LoopLoweredDirectCodegen.cpp`：P4.5 首切片的 `Contiguous1D` loop-lowered direct/resident `ctx/init/run/materialize` family codegen。
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

阶段回归验证：

```bash
cd clang/tools/translator
bash test_mpi.sh vectorAddCombo imageAdjustment1.0 mandel1.0 decay1.0 \
  matMul1.0 gradientSum DFT1.0 jacobi1.0 \
  FOuLa1.0 MDP1.0 liuliang1.0 stencil1.0 waveEquation1.0 \
  mpiMixedStencilORPhaseC mpiOrStencilRefreshPolicy1D mpiOrStencilBoundaryStride2D
```

完整基线：

```bash
bash test_mpi.sh
```

结构验证：

- accepted OR generated code 不出现 `AccessPattern` / `PackPlan`。
- `vectorAddCombo` 的 `tmp_tensor` / `shifted_tensor` 不出现 `tensor2Array()` / `array2Tensor()`。
- `imageAdjustment1.0` 的 `image_tensor2` 不出现 `tensor2Array()` / `array2Tensor()`。
- accepted path 出现 `ContiguousView1D` 或 row-block direct buffer。
- final output 才 `MPI_Gatherv`。
- accepted OR stencil distributed-followup 不再 broadcast `__or_materialized_*` writer，而是在 wrapper 内 lower followup 并按语义同步 reader/followup tensor。
- mixed-site stencil 文件必须按 expr 独立分派，不能因为同文件存在 Phase-C site 而整文件退回旧 Phase-C。
- `oddeven0.1` 在 Phase 5 前仍可 fallback；fallback 正确不表示 Phase 5 已完成。

---
