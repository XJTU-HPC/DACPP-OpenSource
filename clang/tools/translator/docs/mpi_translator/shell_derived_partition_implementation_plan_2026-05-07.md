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

## 6. Phase 1/2 已实现范围

提交 `f4bce7dde implement shell-derived MPI lowering phase 1/2` 已完成 Phase 1/2 初版。

### 6.1 Phase 1：1D Direct / Resident Chain

覆盖测试：

- `vectorAddCombo`
- `mandel1.0`
- `decay1.0`

支持 pattern：

```cpp
dataList{lhs[i], rhs[i], out[i]}
dataList{in[i], bias[{}], out[i]}          // bias.getSize() == 1
dataList{input[i], output[i]}
dataList{N0s[i], lambdas[i], local_A[i], t[{}]}
```

通信策略：

- `DirectMapped` READ input：root `tensor2Array()`，按 rank contiguous slice `MPI_Scatterv`。
- `ReplicatedScalar`：root 取 scalar，`MPI_Bcast`。
- intermediate WRITE：rank-local buffer，`DistributedDirty`，不 materialize。
- final WRITE：`MPI_Gatherv` 到 root，root `array2Tensor()`。

Codegen：

- 复用 `buildLocalCalcCode()` 生成 `*_mpi_local`。
- kernel view 使用 `dacpp::mpi::ContiguousView1D<T>`。
- 不内联重写 calc body。
- accepted path 不生成 legacy `AccessPattern` / `PackPlan`。

### 6.2 Phase 2：2D Row-Block Direct / Resident Chain

覆盖测试：

- `imageAdjustment1.0`

支持 pattern：

```cpp
dataList{image[idx1][idx2], out[idx1][idx2]}
```

约束：

- 两个 bind domain，row-major。
- rank ownership 按 row block，不按任意 linear item range 切碎。
- local buffer size = `local_rows * cols`。
- 两个连续 image pass 共用 row ownership。
- 中间 `image_tensor2` 不 materialize。

通信策略：

- READ image：root `tensor2Array()`，按 row block `MPI_Scatterv`。
- intermediate image：rank-local buffer，`DistributedDirty`，不 materialize。
- final image：在 root 需要 `print()` 前 `MPI_Gatherv` 到 root。

Codegen：

- 当前按 local linear item 构造 `ContiguousView1D<Pixel>{local.data(), item_linear}`，保持 `out[0]` / `in[0]` 语义。
- accepted image chain 不生成 `AccessPattern` / `PackPlan`。

### 6.3 当前验证结果

已通过：

```bash
cmake --build build --target translator -j8

cd clang/tools/translator
bash test_mpi.sh vectorAddCombo imageAdjustment1.0 mandel1.0 decay1.0
bash test_mpi.sh matMul1.0 gradientSum DFT1.0 jacobi1.0 stencil1.0 \
  waveEquation1.0 FOuLa1.0 MDP1.0 liuliang1.0 oddeven0.1
```

结构检查：

- `vectorAddCombo`：chain length 3，layout `Contiguous1D`，`tmp_tensor` / `shifted_tensor` 不 materialize。
- `imageAdjustment1.0`：chain length 2，layout `RowBlock2D`，`image_tensor2` 不 materialize。
- accepted OR 输出不包含 `AccessPattern` / `PackPlan`。
- final output 才出现 `MPI_Gatherv`。
- DFT/Jacobi 等 full payload 模式正确 fallback。

---

## 7. 非 `mpi*` 测试覆盖分类

| 测试 | shell 形态 | 目标 lowering |
|---|---|---|
| `vectorAddCombo` | `lhs[i], rhs[i], out[i]` / `bias[{}]` | `resident_chain_1d` |
| `mandel1.0` | `complex_points[i], mandelbrot_flags[i]` | `direct_1d` |
| `decay1.0` | `N0s[i], lambdas[i], local_A[i], t[{}]` | `direct_1d` + `ReplicatedScalar` |
| `imageAdjustment1.0` | `image[idx1][idx2], image2[idx1][idx2]` | `row_block_2d` + residency |
| `gradientSum` | `matGrads[{}][idx1], matNeuronSum[idx1][idx2]` | `full_payload` / `RowPartitionFullRow` |
| `matMul1.0` | `matA[idx1][{}], matB[{}][idx2], matC[idx1][idx2]` | row partition matmul：A row block + B replicated/full |
| `DFT1.0` | `input[{}], output[i], vec[i]` | `ReplicatedFullTensor` + 1D output |
| `jacobi1.0` | `A[{idx1}][{}], b[{idx1}], x[{}], x_new[{idx1}], nums[{idx1}]` | row full payload + replicated vector |
| `stencil1.0` | `matIn[sp1][sp2], matOut[idx1][idx2]` | `stencil_window` 2D halo |
| `waveEquation1.0` | `matCur[sp1][sp2], matPrev[idx1][idx2], matNext[idx1][idx2]` | `stencil_window` + resident state |
| `FOuLa1.0` | `u_kin[s], u_kout[i], r[{}]` | `stencil_window` 1D + scalar |
| `MDP1.0` | `p[sp], new_p[idx]` | `stencil_window` 1D |
| `liuliang1.0` | `rho[S1], new_rho[idx1]` | `stencil_window` 1D |
| `oddeven0.1` | `array[{S1}], array_out[{S1}]` | `fixed_block` / pair partition |

长期验收标准：上述测试在 `--mpi` 下均由 shell-derived planner 接管，生成代码不再出现 legacy `AccessPattern` / `PackPlan`。复杂未知用户程序仍可 fallback legacy。

---

## 8. 后续阶段路线

### Phase 3：Full Payload / Replicated Input

目标测试：

- `gradientSum`
- `DFT1.0`
- `jacobi1.0`
- 初版 `matMul1.0`

新增 layout：

- `RowPartitionFullRow`
- `ReplicatedFullTensor`

重点：

- `matGrads[{}][idx1]` 应作为 row partition + full row payload 专门模式。
- `input[{}]`、`x[{}]` 这类大输入不能误当 scalar，应显式走 `ReplicatedFullTensor` 或更优分布策略。
- `matMul1.0` 初版可按 C rows 划分，A scatter row block，B broadcast full matrix，C gather row block。

### Phase 4：Stencil Window

目标测试：

- `stencil1.0`
- `waveEquation1.0`
- `FOuLa1.0`
- `MDP1.0`
- `liuliang1.0`

思路：

- `RegularSplit` 不再 fallback legacy，而由 `stencil_window` backend 接管。
- `stencil_window` 采用 loop-lowered `ctx/init/run/materialize` 结构，`init()` 固化稳定 layout 和 cache，`run()` 执行 halo/exchange/boundary-local/route，`materialize()` 在 loop 后恢复 root-visible tensor。
- 现有 `stencil_phase_c` 的 halo、exchange、boundary-local、read-cache transition 和 materialize 能力作为 backend 实现来源，planner ownership 归入 shell-derived lowering。
- Phase 4 的结构产物同时定义通用 loop-lowered shell-derived contract，供后续 direct/resident/full-payload 循环形态复用。

### Phase 4.5：Loop-Lowered Direct / Resident

目标测试：

- `decay1.0`
- 循环内稳定 direct/resident shell 的后续用例

新增执行形态：

- `LoopLiftedDirect1D`
- `LoopLiftedRowBlock2D`
- `LoopLiftedFullPayload`

思路：

- 对循环不变 READ 输入执行一次性 `init()` scatter/cache，避免每轮重复分发大输入。
- 对循环变化 scalar、小 replicated 输入、收敛状态或迭代参数在 `run()` 中执行轻量 broadcast/refresh。
- 对 loop-carried distributed tensor 维持 resident/cache 状态，只在 host assignment、print、unknown call 或最终输出处 materialize。
- `decay1.0` 的目标结构是 `N0s` / `lambdas` loop-invariant scatter-once，`t` per-step broadcast，`local_A` per-step kernel 后按 `A_tensor[...] = local_A_tensor` 的可观察需求同步。

### Phase 5：Fixed Block / Odd-even

目标测试：

- `oddeven0.1`

新增 layout：

- `FixedBlock`

思路：

- 支持 `array[{S1}]`、`S1(2,2)` 这种 pair/block window。
- 按 fixed block 分发，处理 odd-even 的 phase/iteration 语义。

### Phase 6：非 `mpi*` tests 全量新路径

验收：

- 所有非 `mpi*` tests 不再触发 legacy `AccessPattern` / `PackPlan`。
- 完整 `bash test_mpi.sh` 通过。
- benchmark 中高通信开销用例应看到 collective 次数、metadata、materialize 次数下降。

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

Phase 1/2 必须 fallback 的情况：

- `RegularSplit`
- bind offset 非 0
- `{}` 大 tensor，不是 scalar
- `READ_WRITE` 复杂 alias
- 中间 tensor 在 chain 中被 host 读取
- 2D 非 row-major 或非 `[idx1][idx2]`
- 不兼容 `bindSizes + bindOrder`
- 任何无法证明 local layout 的情况

fallback 必须保持 legacy 输出正确。

---

## 11. 实现陷阱

- 不要把所有 `{}` 当作 scalar。
- 不要把 Phase 1/2 写成只能识别测试文件名；应识别 shell/dataList pattern。
- 不要内联重写 calc body；优先复用 `*_mpi_local` 和 contiguous view。
- 不要让 accepted Phase 1/2 chain 依赖 legacy `AccessPattern` / `PackPlan`。
- 不要在 Phase 1/2 处理 matmul/DFT/Jacobi/gradientSum/stencil/oddeven。
- 不要删除 legacy；Phase 1/2 仍需要 fallback。

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

fallback 验证：

```bash
bash test_mpi.sh matMul1.0 gradientSum DFT1.0 jacobi1.0 stencil1.0 \
  waveEquation1.0 FOuLa1.0 MDP1.0 liuliang1.0 oddeven0.1
```

结构验证：

- accepted OR generated code 不出现 `AccessPattern` / `PackPlan`。
- `vectorAddCombo` 的 `tmp_tensor` / `shifted_tensor` 不出现 `tensor2Array()` / `array2Tensor()`。
- `imageAdjustment1.0` 的 `image_tensor2` 不出现 `tensor2Array()` / `array2Tensor()`。
- accepted path 出现 `ContiguousView1D` 或 row-block direct buffer。
- final output 才 `MPI_Gatherv`。

---

## 13. Phase 3 Analysis 状态 (2026-05-07)

### 已完成

Analysis 层实现已完成，支持识别以下 layout：

- `RowPartitionFullRow`：混合 void + index split 的参数（如 `matGrads[{}][idx1]`）
- `ReplicatedFullTensor`：纯 void split 的非 scalar 参数（如 `input[{}]`）

新增文件：
- `FullPayloadPartitionAnalysis.cpp` - 实现 `assignRowPartitionFullRowLayout` 和 `assignReplicatedFullTensorLayout`

修改文件：
- `ShellPartitionAnalysis.cpp` - 更新参数访问类型识别逻辑
- `RowBlock2DPartitionAnalysis.cpp` - `assignPhaseLayout` 添加 Phase 3 分发
- `ShellPartitionAnalysis_Internal.h` - 添加新函数声明
- `CMakeLists.txt` - 添加新源文件

验证结果：
```
[DACPP][MPI][OR] expr=0 shell=gradSumShell layout=RowPartitionFullRow accepted
[DACPP][MPI][OR]   param=matGrads access=RowPartitionFullRow reads=1 writes=0
[DACPP][MPI][OR]   param=matNeuronSum access=OutputDirect reads=0 writes=1
```

### Phase 3 Codegen 当前状态

Phase 3 已启用两个经过验证的 codegen path：
- `ReplicatedFullTensor`：`DFT1.0` 走 OR codegen，root `tensor2Array()` 后 `MPI_Bcast` 全量 tensor，direct 1D 输入继续 `MPI_Scatterv`，最终输出 `MPI_Gatherv` + `array2Tensor()`。
- `RowPartitionFullRow`：`gradientSum` 走 OR codegen，root 按 output ownership pack 每个 local output item 对应的 full payload，local calc 通过 full payload view 访问，最终输出 `MPI_Gatherv` + `array2Tensor()`。

当前 `supportedPhaseLayout()` 已启用：
- `Contiguous1D`
- `RowBlock2D`
- `ReplicatedFullTensor`
- `RowPartitionFullRow`

已接受的 DFT / gradientSum OR path 不再生成 legacy `AccessPattern` / `PackPlan`。

### 仍未完成

- `jacobi1.0` 仍 fallback：需要同时处理 `RowPartitionFullRow` + `ReplicatedFullTensor` 的混合 full-payload 输入。
- `matMul1.0` 仍 fallback：当前 full-payload IR/codegen 尚未处理该形态里的 read-write / ownership 细节。
- `RowPartitionFullRow` 目前只覆盖 gradientSum 需要的 2D full payload 场景，更广泛的 Phase 3 泛化后续再推进。
