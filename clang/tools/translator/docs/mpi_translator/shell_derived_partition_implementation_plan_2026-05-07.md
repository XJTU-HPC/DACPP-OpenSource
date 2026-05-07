# Shell-Derived Resident Partition Model — 实现路径方案

日期：2026-05-07

相关文档：

- [operator_resident_communication_model_2026-05-06.md](./operator_resident_communication_model_2026-05-06.md)
- [operator_resident_progress_2026-05-06.md](./operator_resident_progress_2026-05-06.md)

本文把 operator-resident 的设计进一步收敛成可实现路径。核心改进是：不再把 shell `dataList` 降级为每个参数的 `AccessPattern`，而是直接把 `dataList` 当成 MPI partition IR，从 `index` / `split` / `{}` 标注推导 rank ownership，并用 tensor residency 跨连续 `<->` 复用该划分。

---

## 1. 核心判断

当前 legacy MPI wrapper 的根本问题是通信主轴选错了：

- 当前主轴：每个 `<->`、每个参数、每个 item 访问哪些 global index。
- 应改为：每个 shell 的 `dataList` 如何定义迭代域、哪些维度被 rank 划分、哪些数据可以 replicated、哪些中间 tensor 应驻留在 rank 本地。

对不含 `dacpp::split` 的 pure `index` / `{}` shell，`dataList` 已经给出完整的 MPI ownership 信息：

| shell 标注 | parser 表示 | MPI 语义 |
|---|---|---|
| `idx` / `dacpp::index` | `IndexSplit` | 参与 bind domain，按 item/rank 切分 |
| `sp` / `dacpp::split` | `RegularSplit` | stencil/window，需要 halo 或 fallback |
| `{}` | `Split` 且 id 为 `void` | 不随 item 划分，可能是 scalar、dense slice 或 replicated tensor |

因此，第一版 operator-resident 应只接管 pure `index` / `{}` 模式。含 `RegularSplit` 的表达式继续交给 `stencil_phase_c` 或 `legacy_access_pattern`，避免把 halo 问题混入轻量算子链快路径。

---

## 2. 相比原方案的改进点

### 2.1 Partition 以 bind domain 为中心

原先容易把 partition 直接建在 tensor dim 上，例如“tensor 第 0 维被切分”。这不够稳，因为 DACPP shell 中真正的迭代域是 `binding()` 关系形成的 bind set。现有 parser 中 `Shell::GetBindInfo()` 已经把相关 `index/split` 归入连通分量，legacy 路径的 `collectSplitBindMeta()` 也把它转成 `bindId`。

新模型应先建立 bind domain：

```cpp
struct BindDomain {
    int bindId = -1;
    std::string representative;      // index 名，例如 "i" / "idx1"
    std::string offsetExpr = "0";    // binding offset，第一版只接受 "0"
    int64_t sizeExprParam = -1;       // 可选：来源 shell 参数编号
    int dimId = -1;                   // 来源 tensor 维度
};
```

然后再记录 tensor dim 到 bind domain 的映射，而不是反过来：

```cpp
enum class ShellDimKind {
    Index,
    Void,
    Split
};

struct TensorDimMapping {
    std::string tensorName;
    int shellParamIndex = -1;
    int tensorDim = -1;
    ShellDimKind kind = ShellDimKind::Void;
    int bindId = -1;                  // kind == Index 时有效
    std::string splitName;            // kind == Split 时有效
};
```

这样 `dataList{A[i][j], B[i][j]}`、`dataList{lhs[i], out[i]}`、`dataList{matGrads[{}][idx1], matNeuronSum[idx1][idx2]}` 都能用同一套 IR 表达。

### 2.2 `{}` 必须细分

`{}` 只表示该维度不随当前 item 变化，不等价于 scalar broadcast。

需要区分：

```cpp
enum class VoidAccessKind {
    ScalarElement,    // bias[{}]
    FullRowPayload,   // matGrads[{}][idx1] 这类 row partition + full row payload
    ReplicatedTensor, // 小 tensor 整体复制，第一版可不主动启用
    Unsupported
};
```

判断原则：

- 如果 tensor rank 为 1，且唯一维度是 `{}`，且 tensor size 为 1 或可证明小，按 `ScalarElement`。
- 如果 tensor 同时含 `Void` 和 `Index`，按 `FullRowPayload` 或后续更具体 layout 候选，例如 `matGrads[{}][idx1]`。该类模式必须结合 calc view 语义确认，不允许泛化成普通 scalar broadcast。
- 如果所有维度都是 `{}`，但 size 不可证明为 1，第一版回退，避免误把大 tensor broadcast。

### 2.3 local layout 不用 bool

`isContiguous` 太粗。codegen 需要知道 local buffer 如何映射到 calc view。

```cpp
enum class LocalLayoutKind {
    Contiguous1D,
    RowBlock2D,
    RowPartitionFullRow,
    ReplicatedScalar,
    Unsupported
};
```

第一版支持范围：

- `Contiguous1D`：`a[i]`、`out[i]`
- `RowBlock2D`：`image[idx1][idx2]`
- `RowPartitionFullRow`：按 row/neuron 维划分 ownership，每个 owned row 携带完整 row payload，例如 `gradientSum` 的 `matGrads[{}][idx1]`
- `ReplicatedScalar`：`bias[{}]`

`matGrads[{}][idx1]` 这种形态要先确认现有 DACPP 语义下 `{}` 对应的是 row 内完整列 payload，还是 column/full-height payload。如果无法证明本地 slice 连续，第一版应该保守回退。Phase 3 应把它作为 `RowPartitionFullRow` 专门模式实现，不要塞进泛化的 dense-slice 概念。

### 2.4 继续复用 calc，不内联改写 calc body

实现上不要先把 calc body 重写成 `acc[idx] = ...`。现有 legacy 路径已经生成：

```cpp
template <typename V0, typename V1, typename V2>
inline void vadd_mpi_local(V0 lhs, V1 rhs, V2 out) {
    out[0] = lhs[0] + rhs[0];
}
```

operator-resident 应继续复用这个 local calc wrapper，只是把 view 换成轻量 contiguous view：

```cpp
dacpp::mpi::ContiguousView1D<float> view_lhs{local_lhs.data(), item_linear};
dacpp::mpi::ContiguousView1D<float> view_out{local_out.data(), item_linear};
vadd_mpi_local(view_lhs, view_rhs, view_out);
```

这能避免解析和重写 calc body，风险小很多。

### 2.5 Residency 是和 partition 绑定的状态

只知道 partition 不足以减少通信。必须记录 tensor 当前数据是否已经以同一 partition 驻留在 rank 本地：

```cpp
enum class ResidencyKind {
    RootOnly,
    DistributedClean,
    DistributedDirty,
    ReplicatedScalar,
    MaterializedRoot,
    Unknown
};

struct PartitionSignature {
    std::vector<int64_t> bindSizes;
    std::vector<int> bindOrder;
    LocalLayoutKind layout = LocalLayoutKind::Unsupported;
    std::string linearization;        // "1d-linear" / "2d-row-major"
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

两个 `<->` 能连成 chain 的条件不是“都用了 index”，而是它们读写中间 tensor 时 `PartitionSignature` 兼容。

Phase 1 不需要完整状态机，只实现并使用以下状态：

- `RootOnly`：外部输入 tensor 初始状态，进入 chain 时需要 scatter。
- `DistributedDirty`：rank-local buffer 是最新数据，root tensor 已过期或尚未同步。
- `ReplicatedScalar`：`bias[{}]` 这类 scalar 已广播到所有 rank。
- `MaterializedRoot`：final output gather 后 root tensor 有效。

`DistributedClean` 和 `Unknown` 先保留为 IR 扩展点，Phase 1 不依赖它们。

Partition 兼容性也必须明确。`PartitionSignature` 可以保留 `layout` 和 `linearization`，但 chain 兼容只比较 bind domain：

```cpp
inline bool isCompatibleForChain(const PartitionSignature& lhs,
                                 const PartitionSignature& rhs) {
    return lhs.bindSizes == rhs.bindSizes &&
           lhs.bindOrder == rhs.bindOrder;
}
```

`layout` 和 `linearization` 是 codegen 策略选择字段，不参与 chain 兼容性判断。Phase 1 作为实现限制，可以额外要求每个 accepted expression 的 layout 都是 `Contiguous1D`。

---

## 3. 最终模型

模型名称建议定为：

```text
Shell-Derived Resident Partition Model
```

简称可在代码中使用 `operator_resident`，日志中使用 `[DACPP][MPI][OR]`。

输入：

- `DacExprNode`
- `Shell*`
- `ShellParam -> Split[]`
- `Calc*` 和 effective param modes
- AST 中 `<->` 的语句顺序和 host 使用信息

输出：

```cpp
struct ShellPartitionPlan {
    bool supported = false;
    std::string rejectReason;
    std::vector<BindDomain> bindDomains;
    std::vector<TensorDimMapping> mappings;
    PartitionSignature signature;
    std::vector<ParamAccessPlan> params;
};

struct OperatorResidentChainPlan {
    bool supported = false;
    std::string rejectReason;
    int chainId = -1;
    std::vector<DacExprNode> exprs;
    PartitionSignature signature;
    std::unordered_map<std::string, TensorResidencyState> residency;
    std::vector<std::string> materializeTensors;
};
```

参数访问计划：

```cpp
enum class ParamAccessKind {
    DirectMapped,
    ReplicatedScalar,
    RowPartitionFullRow,
    OutputDirect,
    Unsupported
};

struct ParamAccessPlan {
    int paramIndex = -1;
    std::string shellTensorName;
    std::string calcParamName;
    IOTYPE mode = IOTYPE::READ;
    ParamAccessKind access = ParamAccessKind::Unsupported;
    LocalLayoutKind layout = LocalLayoutKind::Unsupported;
};
```

---

## 4. 分析算法

### 4.1 单个 shell 的 partition 分析

入口建议：

```cpp
ShellPartitionPlan analyzeShellPartition(const MpiAnalysisContext& ctx,
                                         const DacExprNode& node);
```

步骤：

1. 调用或复用 `collectSplitBindMeta(shell)`，建立 split id 到 `bindId/offset` 的映射。
2. 遍历 `shell->getShellParam(paramIdx)->getSplit(dimIdx)`。
3. 遇到 `RegularSplit`：标记 `hasSplit`，返回 unsupported，reason 为 `contains regular split`。
4. 遇到 `IndexSplit`：记录 `TensorDimMapping{Index, bindId}`，并从对应 tensor 的 `getShape(dim)` 生成 runtime size 表达式。
5. 遇到 id 为 `"void"` 的普通 `Split`：记录 `TensorDimMapping{Void}`。
6. 检查所有非 void 的 bind domain 是否有一致 size。第一版只接受 offset 为 `"0"`。
7. 根据每个参数的 dim mapping 和 mode 推断 `ParamAccessKind` 和 `LocalLayoutKind`。
8. 如果至少一个 WRITE/READ_WRITE 输出可以定义 domain，则生成 `PartitionSignature`。

拒绝条件：

- 包含 `RegularSplit`
- bind offset 非 0
- index 维度大小不一致且无法证明
- 同一参数出现复杂 index/void 组合且不在支持表内
- `READ_WRITE` 参数不是 direct mapped
- 输出参数没有 index 维度

### 4.2 chain 识别

入口建议：

```cpp
std::vector<OperatorResidentChainPlan>
buildOperatorResidentChains(const MpiAnalysisContext& ctx,
                            const std::vector<DacExprNode>& exprNodes,
                            const std::vector<ShellPartitionPlan>& partitions);
```

第一版只识别同一 compound statement 内的连续 `<->`。如果当前 `DacExprNode::parentStmt` 暂未填充，应先补齐 parent statement；短期可用 source order 做保守连续判断，但最终应基于 AST parent。

chain 接受条件：

1. 表达式顺序连续。
2. 每个表达式的 `ShellPartitionPlan` supported。
3. `PartitionSignature` 兼容。兼容定义为 `bindSizes + bindOrder` 完全相等；`layout/linearization` 不参与 chain 兼容，只参与 codegen 策略选择。
4. 前一个 WRITE tensor 若被后一个 READ 使用，标记为 distributed resident。
5. 中间没有 host 侧读取、未知函数调用、普通 C++ 下标访问、`tensor2Array()` 或 `.print()`。
6. 每个 tensor 在 chain 内没有复杂 alias 或多个不兼容 writer。

chain 边界：

```cpp
enum class ChainBoundaryKind {
    None,
    HostRead,
    UnknownCall,
    PartitionChange,
    AliasConflict,
    ControlFlowBoundary,
    UnsupportedPattern
};
```

### 4.3 materialize 分析

第一版采用保守策略：

- chain 最终 WRITE 输出如果后续有 `.tensor2Array()`、`.print()`、`std::cout`、未知函数传参、host 下标访问，则插入 materialize。
- 如果无法判断后续使用，默认 materialize 最终输出。
- 中间 tensor 如果仅被 chain 内后续 `<->` 读取，不 materialize。
- `--mpi-output-sync=all-ranks` 仍要尊重：如果后续非 root rank 可能读 root materialized tensor，要在 gather 后 broadcast 或回退 legacy。

---

## 5. Codegen 形态

### 5.1 生成函数

对每条 accepted chain 生成：

```cpp
struct __dacpp_mpi_or_chain_ctx_0 { ... };

void __dacpp_mpi_or_chain_init_0(__dacpp_mpi_or_chain_ctx_0& ctx,
                                 /* shell call args used by chain */);

void __dacpp_mpi_or_chain_run_0(__dacpp_mpi_or_chain_ctx_0& ctx,
                                /* shell call args used by chain */);

void __dacpp_mpi_or_chain_materialize_0(__dacpp_mpi_or_chain_ctx_0& ctx,
                                        /* output tensors */);
```

source rewrite：

```cpp
__dacpp_mpi_or_chain_ctx_0 __dacpp_mpi_or_ctx_0;
__dacpp_mpi_or_chain_init_0(__dacpp_mpi_or_ctx_0, a_tensor, b_tensor, c_tensor, bias_tensor, out_tensor);
__dacpp_mpi_or_chain_run_0(__dacpp_mpi_or_ctx_0, a_tensor, b_tensor, c_tensor, bias_tensor, out_tensor);
__dacpp_mpi_or_chain_materialize_0(__dacpp_mpi_or_ctx_0, out_tensor);
```

被 chain 接管的原始 `<->` 语句移除或替换为空语句。未接管表达式继续走 legacy wrapper。

### 5.2 runtime helper

第一版尽量少加 runtime，优先生成直接代码。需要沉到 runtime 的公共函数：

```cpp
namespace dacpp::mpi::oruntime {

inline std::vector<int> build_counts_for_range(int64_t total, int mpi_size);
inline std::vector<int> build_displs_for_counts(const std::vector<int>& counts);

template <typename T>
void scatter_contiguous_1d(const std::vector<T>& rootData,
                           std::vector<T>& localData,
                           int64_t total,
                           MPI_Datatype mpiType,
                           MPI_Comm comm);

template <typename T>
void gather_contiguous_1d(const std::vector<T>& localData,
                          std::vector<T>& rootData,
                          int64_t total,
                          MPI_Datatype mpiType,
                          MPI_Comm comm);

}
```

文件位置：

```text
dpcppLib/include/mpi/operator_resident/
  OperatorResidentTypes.h
  OperatorResidentRuntime.h
  OperatorResident.h
```

`MPIPlanner.h` 暂不必直接包含新头；生成 operator-resident 代码时可以通过新的 facade `mpi/operator_resident/OperatorResident.h` 引入。等路径稳定后再决定是否纳入 `MPIPlanner.h`。

### 5.3 local view

现有 `mpi/common/KernelViews.h` 已有 `ContiguousView1D` / `ContiguousView2D`，第一版直接复用，不新增 view 类型。需要 dense slice 时再补：

```cpp
template <typename T>
struct DenseSliceView1D {
    T* data = nullptr;
    int offset = 0;
    decltype(auto) operator[](int idx) const { return data[offset + idx]; }
};
```

Phase 3 如果实现 `RowPartitionFullRow`，应优先新增名字明确的 row payload view，而不是使用泛化的 `DenseSliceView1D` 名称。例如：

```cpp
template <typename T>
struct RowPayloadView1D {
    T* data = nullptr;
    int rowOffset = 0;
    decltype(auto) operator[](int idx) const { return data[rowOffset + idx]; }
};
```

---

## 6. 文件落点

translator 侧：

```text
rewriter/include/Rewriter_MPI_OperatorResident.h

rewriter/lib/mpi/operator_resident/
  ShellPartitionAnalysis.cpp
  OperatorChainAnalysis.cpp
  ResidencyAnalysis.cpp
  OperatorResidentCodegen.cpp
```

plan 类型扩展：

```text
rewriter/include/Rewriter_MPI_Plan.h
```

新增内容：

- `ShellPartitionPlan`
- `OperatorResidentChainPlan`
- `OperatorResidentPlan`
- `MpiLoweringPlan::operatorChains`
- `MpiLoweringPlan::legacyWrappers`
- `MpiLoweringPlan::stencilSites`

调度入口：

```text
rewriter/lib/mpi/shared/MpiPlanBuilder.cpp
rewriter/lib/Rewriter_MPI.cpp
```

runtime 侧：

```text
dpcppLib/include/mpi/operator_resident/
  OperatorResidentTypes.h
  OperatorResidentRuntime.h
  OperatorResident.h
```

CMake：

```text
CMakeLists.txt
```

加入 operator-resident `.cpp` 到 `REWRITER_SOURCES`。

---

## 7. 分阶段实施路径

### Phase 0：结构探针，不改变生成代码

目标：

- 新增 operator-resident 分析文件和头文件。
- 对每个 `<->` 打印 partition 分析结果。
- 所有表达式仍走 legacy/stencil。

验收：

```text
[DACPP][MPI][OR] expr=0 shell=VADD partition=1d-linear accepted-analysis-only
[DACPP][MPI][OR] expr=1 shell=VSHIFT partition=1d-linear scalar=bias
[DACPP][MPI][OR] expr=... rejected reason=contains regular split
```

验证：

```bash
cmake --build build --target translator -j8
cd clang/tools/translator
bash test_mpi.sh vectorAddCombo imageAdjustment1.0 gradientSum
```

### Phase 1：1D direct mapped chain，接管 vectorAddCombo

支持：

- `dataList{a[i], b[i], out[i]}`
- `dataList{in[i], bias[{}], out[i]}` 且 `bias.getSize() == 1`
- 同一 basic block 内连续 chain
- 最终 output materialize
- chain 兼容只比较 `bindSizes + bindOrder`
- codegen 额外要求 layout 为 `Contiguous1D`
- residency 只实现 `RootOnly`、`DistributedDirty`、`ReplicatedScalar`、`MaterializedRoot`

生成：

- scatter root-only inputs：`a/b/c`
- broadcast scalar：`bias`
- local buffers：`tmp/shifted/out`
- kernel 使用 `ContiguousView1D`
- gather final `out`

结构验收：

- 生成代码中 `tmp_tensor`、`shifted_tensor` 不出现 `tensor2Array()` / `array2Tensor()` writeback。
- operator-resident 接管后不生成 `VADD_vadd` / `VSHIFT_vshift` 的 legacy wrapper 调用，或 wrapper 未被调用。
- `MPI_Gatherv` 只用于 final output，不用于中间 tensor。

### Phase 2：2D row-block direct mapped，接管 imageAdjustment1.0

支持：

- `dataList{image[idx1][idx2], out[idx1][idx2]}`
- row-major layout，按 row block 划分。
- 两个连续 image pass 共享 row ownership。

生成：

- scatter input image row blocks。
- `image_tensor2` local buffer 驻留。
- final `image_tensor3` materialize 后供 `.print()`。

结构验收：

- 两个 kernel 之间没有 `image_tensor2.tensor2Array()` 或 `image_tensor2.array2Tensor()`。
- local buffer 大小为 `local_rows * width`。

### Phase 3：row partition full-row payload，接管 gradientSum

支持：

- 输出按 row/neuron 切分。
- 输入按 owner row block 分发，每个 owned row 携带完整 row payload。
- calc 读取本 rank 的完整 row slice。

注意：

- 先明确 `matGrads[{}][idx1]` 在现有 parser/calc view 下对应的实际 layout。
- 从 `gradientSum.dac.cpp` 的 calc 看，`grads[j]` 语义应是“当前 row 的所有列”，但实现前必须用生成代码或 parser dump 确认。
- 如果该形态不是连续 row payload，优先回退，不要把 `{}` 泛化错。

验收：

- 输入按 row block 一次分发，payload count 为 `local_rows * INPUT_SIZE`。
- 不为每个 output item 构造 `PackPlan`。
- final `matNeuronSum` 按需要 gather。

### Phase 4：混合调度

目标：

- 同一文件中 stencil、operator-resident、legacy 可共存。
- `rewriteMPI()` 不再因为 `hasMPIStencilSites()` 就整文件转入 stencil dispatcher。

调度：

1. stencil site 优先。
2. pure index/void chain 尝试 operator-resident。
3. 其余 legacy。

验收：

- 每个 `<->` 只有一个 owner。
- fallback reason 可打印。
- 完整 `test_mpi.sh` 通过。

### Phase 5：profiling 和 benchmark

新增 profile tag：

```text
[DACPP][PROFILE][OR] chain_init_ms(max)
[DACPP][PROFILE][OR] chain_run_ms(max)
[DACPP][PROFILE][OR] chain_materialize_ms(max)
```

benchmark：

- `vectorAddCombo`
- `imageAdjustment1.0`
- `gradientSum`

性能目标：

- 明显减少 collective 次数。
- `vectorAddCombo`、`imageAdjustment1.0` 接近手写 MPI+SYCL 的通信结构。
- 如果总时间仍不接近，profile 应能说明瓶颈已经从 root pack/gather 转移到 kernel 或 tensor conversion。

---

## 8. 负向用例

需要新增或保留结构测试：

- 含 `dacpp::split` 的 shell：不走 operator-resident。
- `{}` 大 tensor 且无法证明 scalar：不 broadcast，回退。
- bind offset 非 0：第一版回退。
- 中间 tensor 在 chain 中被 host 读取：强制 materialize 或回退。
- `READ_WRITE` 参数复杂 alias：回退。
- 不同 partition signature 的连续 `<->`：拆 chain 或回退。
- stencil Phase-C 用例行为不变。

---

## 9. 实现顺序建议

最小可提交序列：

1. `docs: add shell-derived resident partition implementation plan`
2. `mpi-or: add partition analysis IR and debug logs`
3. `mpi-or: detect straight-line operator chains`
4. `mpi-or: add runtime helpers for contiguous 1d scatter/gather`
5. `mpi-or: generate 1d operator chain for vectorAddCombo`
6. `mpi-or: add 2d row-block direct mapped chain`
7. `mpi-or: add row partition full-row chain`
8. `mpi: allow mixed stencil/operator-resident/legacy lowering`
9. `bench: add operator-resident profile and benchmark notes`

每一步都应保持 legacy fallback 可用，避免一次性替换现有 wrapper。

---

## 10. 当前应避免的实现陷阱

- 不要把 `{}` 全部当作 scalar broadcast。
- 不要用 `bool isContiguous` 决定所有 codegen。
- 不要内联重写 calc body；先复用 `*_mpi_local` 和 contiguous view。
- 不要让 operator-resident 依赖 legacy `AccessPattern` / `PackPlan`，否则会把旧模型的问题带回来。
- 不要第一版处理 bind offset、复杂 split、非连续 layout；这些应明确 fallback。
- 不要在 `Rewriter_MPI_Common.h` 继续堆散函数；新声明放独立 operator-resident 头或 plan 头。
