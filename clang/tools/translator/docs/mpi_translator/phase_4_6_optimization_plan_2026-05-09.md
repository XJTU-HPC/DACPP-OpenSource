# Phase 4.6 Shell-Derived / Operator-Resident MPI Optimization Plan

日期：2026-05-09
分支：`tqc-2`

本文是 Phase 4.6 的执行计划。它不替代
`shell_derived_partition_implementation_plan_2026-05-07.md`，而是把其中
“效率优化层”的工作拆成两个可独立 review 的路线：

- **路线 A：简单循环拆分 / loop-lowered sync narrowing**  
  先复用 Phase 4.5 的 `ctx/init/run/materialize` 结构，把循环不变的
  metadata、scatter、buffer 分配和 queue 建立 hoist 到 loop 外，并在不改变
  host-visible 语义的前提下减少循环内 full sync。
- **路线 B：resident halo 方案**  
  在 OR stencil 当前语义切片上引入 owned + halo resident storage，用邻居交换
  替代每轮 full reader broadcast / full writer gather / full followup broadcast。

Phase 4.6 的首要原则是：**只优化已经支持并已证明正确的语义形态，不新增
root-bridge、FixedBlock 或更广 stencil route。**

执行口径：

- Step 0 measurement-only baseline 是必做项，不作为可选项。P4.6 的每个后续
  patch 都应能说明它减少了哪类通信或只是在铺设结构。
- 路线 A 主要是 loop-lowered 结构铺垫和 sync narrowing，不应把
  “hoist full reader broadcast” 当成主线收益点；stencil reader 通常会被
  distributed followup / read-cache 更新，能证明 loop-invariant 的场景较窄。
- 真正减少 stencil 通信量的第一刀应是路线 B 的 1D resident halo，而不是先押注
  full-reader hoist。

---

## 1. 当前基线

当前 OR stencil wrapper 的通信形态偏保守：

```text
每次 <-> wrapper:
  root tensor2Array(reader)
  MPI_Bcast(full reader)
  rank-local kernel writes owned writer slice
  ensure_resident(writer local slice)
  MPI_Gatherv(full writer to root)
  root array2Tensor(writer)
  root replay distributed followup / read-cache / boundary-local
  MPI_Bcast(full followup/read-cache tensor)
  array2Tensor(reader/followup on all ranks)
```

优点是语义清晰：kernel 中的 stencil window 读完整 reader，因此无需真正 halo。
缺点是循环内每轮都做 full tensor sync。

Phase 4.5 已经为 `Contiguous1D` direct/resident 的窄切片建立
`init/run/materialize` family。P4.6 应复用这个方向，但不能把新的 stencil 语义
混进优化 patch。

---

## 2. 非目标

- 不新增 `FixedBlock` / `oddeven0.1`。
- 不支持 root-bridge stencil。
- 不扩展当前未定义的复杂 route / non-unit stride / dynamic shape。
- 不删除 legacy fallback。
- 不把 “少一次通信” 建立在 host-visible tensor 状态变旧的前提上。
- 不让 accepted OR path 重新依赖 legacy `AccessPattern` / `PackPlan`。

---

## 3. 路线 A：简单循环拆分 / Sync Narrowing

### 3.1 目标

路线 A 的目标是用最小语义风险铺设 loop-lowered 结构，并只在能证明语义等价时做
sync narrowing：

- 把可 hoist 的 ownership、counts/displs、resident buffer 分配和 SYCL queue
  初始化移到 `init()`。
- 只有当 reader / direct reader 可证明 loop-invariant 时，才把 full reader
  scatter/broadcast 移到 `init()`；否则继续在 `run()` 中 refresh。
- `run()` 只做每轮必须变化的输入同步、kernel 和必要 output sync。
- `materialize()` 只在 loop 后或 host-visible 边界执行。
- 对 loop 内立即 host-visible read 的形态继续保守 materialize，不提前进入 halo。

路线 A 不改变 rank ownership，也不改变当前 OR stencil 的 full-reader 语义；
它首先把重复 setup 从 loop 内移出去。full reader sync hoist 只是 opportunistic
子优化，不是路线 A 的主线假设。

### 3.2 适用候选

首批候选应保守限制为：

- 已有 P4.5 `Contiguous1D` direct/resident candidate。
- 后续可扩展到 `RowBlock2D` direct/resident。
- stencil 只接受当前 P4 已 supported 的 `StencilWindow1D/2D`，且必须满足：
  - 首批只接受 loop 内恰好一个 DAC expr；多个 expr 同 ownership 的 loop chain
    需要单独结构测试和语义证明后再进入。
  - shell args、shape、bind domain 稳定。
  - reader 在 loop 内不会被非 OR host assignment 改写。
  - post-shell followup / read-cache / boundary-local 已被 OR wrapper 内等价 lowering。
  - 如果 reader 会被本轮 distributed followup / read-cache 更新，则 reader
    视为 loop-variant，full reader sync 继续留在 `run()`，不得 hoist 到 `init()`。

### 3.3 需要新增的 plan metadata

建议在 `OperatorResidentPlan.h` 中继续扩展 `ShellPartitionPlan`，避免只用
`loopLowerCandidate` 表达所有优化状态：

```cpp
enum class OrLoopLowerKind {
    None,
    Direct1D,
    RowBlock2D,
    StencilFullSync,
    StencilResidentHalo
};

struct OrLoopLowerPlan {
    OrLoopLowerKind kind = OrLoopLowerKind::None;
    const clang::Stmt* outerLoop = nullptr;
    bool hoistReaderSync = false;
    bool runMaterializeEveryStep = false;
    bool finalMaterializeRequired = false;
    std::string rejectReason;
};
```

短期也可以继续使用现有字段，但 P4.6 开始建议单独建结构，否则
`loopLowerCandidate` 会同时承载 P4.5 direct、P4.6 stencil-full-sync 和
P4.6 halo，review 很难判断边界。

### 3.4 分析侧修改计划

文件：

- `rewriter/lib/mpi/operator_resident/OperatorChainAnalysis.cpp`
- `rewriter/lib/mpi/operator_resident/StencilWindow1DPartitionAnalysis.cpp`
- `rewriter/lib/mpi/operator_resident/StencilWindowPartitionAnalysis.cpp`
- `rewriter/include/mpi/operator_resident/OperatorResidentPlan.h`

任务：

1. 抽出公共 loop 稳定性判定 helper：
   - `findStableOuterLoopForExpr()`
   - `loopContainsSingleOrCompatibleDacExpr()`
   - `tensorWrittenInLoopOutsideCurrentOrPath()`
   - `shellArgsDeclaredBeforeLoop()`
   - `shapeStableAcrossLoop()`

2. 为路线 A 增加 reject log：

```text
[DACPP][MPI][OR][P4.6][Loop] expr=... layout=StencilWindow1D candidate=...
[DACPP][MPI][OR][P4.6][Loop] expr=... rejected reason=reader modified in loop
```

3. 禁止隐式扩大 stencil 语义：
   - `hasRootBridge` 继续 reject。
   - unsupported followup/read-cache/boundary-local 继续 fallback。
   - `READ_WRITE` direct writer 继续 reject。

### 3.5 Codegen 修改计划

新增或拆分文件：

- `rewriter/lib/mpi/operator_resident/LoopLoweredStencilCodegen.cpp`
- `rewriter/lib/mpi/operator_resident/LoopLoweredStencil1DCodegen.cpp`
- `rewriter/lib/mpi/operator_resident/LoopLoweredStencil2DCodegen.cpp`
- 更新 `OperatorResidentCodegen.cpp`
- 更新 `OperatorResidentCodegen_Internal.h`
- 更新 `Rewriter_MPI.cpp`
- 更新 `CMakeLists.txt`

目标生成结构：

```cpp
struct __dacpp_mpi_or_stencil_x_ctx {
    int mpi_rank;
    int mpi_size;
    sycl::queue q;
    // ownership
    int64_t total_items / rows / cols;
    RankRange1D range / row_range;
    std::vector<int> counts;
    std::vector<int> displs;
    // cached full reader for route A
    std::vector<T> reader_global;
    std::vector<T> direct_reader_global;
    std::vector<T> writer_local;
};

void init(ctx, args...) {
    compute ownership;
    allocate buffers;
    if reader is loop-invariant:
        root tensor2Array(reader);
        MPI_Bcast(full reader);
}

void run(ctx, args...) {
    if reader is not hoisted:
        refresh full reader;
    launch kernel on owned writer slice;
    ensure_resident(writer, writer_local);
    if per-step host-visible state is required:
        materialize/update followup now;
}

void materialize(ctx, args...) {
    gather writer only if root-visible output required;
    replay followup/read-cache/boundary-local if not already done per-step;
}
```

关键 review 点：

- `init()` 插入位置必须在 loop 前，且所有传入 args 在该位置已经声明。
- `run()` 替换原 `<->`，不能移除 loop 内非 OR 语句。
- 如果 loop 内后续语句读取 output/followup tensor，`run()` 仍必须同步到正确可见状态。
- `materialize()` 对 `root-only`、`all-ranks-needed`、`distributed-followup`
  的行为必须与 `classifyOutputSyncRequirement()` 一致。

### 3.6 路线 A 验收

结构检查：

- accepted P4.6 loop path 出现 `ctx/init/run/materialize`。
- loop-invariant reader sync 只出现在 `init()`；loop-variant reader sync 必须仍在
  `run()`。
- loop 内不重复生成 ownership/counts/displs。
- 不出现 legacy `AccessPattern` / `PackPlan`。

测试建议：

- `decay1.0`：保持 P4.5 direct 行为。
- 新增 `mpiLoopDirectFinalMaterialize1D`：loop 内 output 不被 host-visible read，
  用于证明 direct/resident 可以把 gather 推迟到 final `materialize()`；不要用
  `decay1.0` 证明该优化，因为 `decay1.0` loop 内会读取 output 写入 `A_tensor`。
- 新增 `mpiLoopStencilFullSync1D`：验证 stencil loop-lowered full-sync skeleton
  的结构，重点是 hoist setup，不要求 full-reader hoist。
- 新增 `mpiLoopStencilInvariantReader1D`：仅当 reader 真正 loop-invariant 时验证
  full-reader hoist 的结构。
- 新增 `mpiLoopStencilReaderVariantReject1D`：reader 在 loop 内被 host 改写时 reject。
- 新增 `mpiLoopStencilFollowupVisible1D`：loop 内 followup 后继续 host read 时仍正确。
- `MDP1.0`、`liuliang1.0`、`stencil1.0`、`waveEquation1.0` 全量回归。

---

## 4. 路线 B：Resident Halo 方案

### 4.1 目标

路线 B 是真正减少 stencil 通信量的方案：

```text
init:
  scatter reader owned slice
  allocate owned + halo buffers
  exchange initial halo

run:
  local kernel reads owned + halo reader, writes owned writer
  distributed route writer -> reader owned
  apply boundary-local updates
  exchange reader halo for next iteration

materialize:
  gather only at host-visible boundary
```

目标是用邻居通信替代循环内 full tensor broadcast/gather。

### 4.2 首批语义范围

路线 B 只覆盖当前 P4 已 accepted 的 stencil 切片，但必须分批进入：

- B1：`StencilWindow1D`
  - exactly one window reader
  - exactly one WRITE-only direct writer
  - no direct reader
  - supported `+1` distributed followup
  - supported constant-index boundary-local updates

- B2：`StencilWindow2D` row-block halo，不含 direct-reader / read-cache transition
  - row-block ownership
  - exactly one window reader
  - exactly one WRITE-only direct writer
  - supported `+1,+1` distributed followup
  - supported current boundary-local updates
  - 目标用例优先是 `stencil1.0`

- B3：`StencilWindow2D` direct-reader / read-cache transition
  - 在 B2 通过后再支持 at most one direct reader
  - supported `-1,-1` read-cache transition for `waveEquation1.0`
  - direct reader/cache tensor 也必须有 resident halo state

不在首批做：

- root-bridge
- 复杂 multi-route
- non-unit boundary stride
- dynamic shape
- arbitrary halo width
- column-block / tiled 2D ownership

### 4.3 Runtime 数据结构

建议新增 runtime 头：

```text
clang/tools/translator/dpcppLib/include/mpi/operator_resident/
  OperatorResidentHalo.h
```

建议结构：

```cpp
struct Halo1DLayout {
    int64_t global_size = 0;
    int64_t owned_begin = 0;
    int64_t owned_count = 0;
    int halo_left = 1;
    int halo_right = 1;
    int64_t local_size() const;
    int64_t core_offset() const; // halo_left
    int64_t local_slot_for_global(int64_t global) const;
    bool owns(int64_t global) const;
};

struct Halo2DRowLayout {
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t owned_row_begin = 0;
    int64_t owned_rows = 0;
    int halo_top = 1;
    int halo_bottom = 1;
    int64_t local_rows() const;
    int64_t core_row_offset() const; // halo_top
    int64_t local_slot_for_global(int64_t row, int64_t col) const;
    bool owns_row(int64_t row) const;
};
```

建议通信 helper：

```cpp
template <typename T>
void exchange_halo_1d(std::vector<T>& buffer,
                      const Halo1DLayout& layout,
                      int rank,
                      int size,
                      MPI_Datatype type);

template <typename T>
void exchange_halo_2d_rows(std::vector<T>& buffer,
                           const Halo2DRowLayout& layout,
                           int rank,
                           int size,
                           MPI_Datatype type);
```

实现可以先用 `MPI_Sendrecv`，比 `MPI_Isend/Irecv` 更容易 review。后续如果需要
overlap，再单独优化。

### 4.4 Halo view

当前 `ContiguousView1D/2D` 只支持简单 base offset。resident halo 需要新的 view，
让 calc 中的相对下标能映射到 owned+halo buffer：

文件：

- `clang/tools/translator/dpcppLib/include/mpi/common/KernelViews.h`

新增：

```cpp
template <typename T>
struct HaloView1D {
    T* data = nullptr;
    int global_base = 0;   // current output global index
    int owned_begin = 0;
    int core_offset = 0;

    decltype(auto) operator[](int idx) const {
        const int global = global_base + idx;
        const int local = core_offset + (global - owned_begin);
        return data[local];
    }
};

template <typename T>
struct HaloView2D {
    T* data = nullptr;
    int global_row = 0;
    int global_col = 0;
    int owned_row_begin = 0;
    int core_row_offset = 0;
    int cols = 0;
    // operator[] returns a row proxy whose col offset remains global_col + idx
};
```

首版假设 halo width = 1，并通过 analyzer 保证 window access 不超过该范围。更广
window width 后续再做。

### 4.5 1D Resident Halo Flow

初始化：

```text
compute owned range from output size
allocate reader_local_with_halo = owned_count + left_halo + right_halo
scatter reader owned slice into core
apply initial boundary defaults if needed
exchange_halo_1d(reader)
allocate writer_owned = owned_count
```

每轮 run：

```text
kernel:
  for local item j:
    global i = owned_begin + j
    HaloView1D reader(data, global_base=i, owned_begin, core_offset)
    ContiguousView1D writer(writer_owned, offset=j)

route:
  for each owned writer index i:
    target = i + 1
    if owner(target) == self:
      reader_core[target] = writer[i]
    else:
      send writer[i] to neighbor owner(target)

boundary:
  apply supported constant-index boundary-local update

halo:
  exchange_halo_1d(reader)
```

对于当前 `+1` followup，跨 rank route 只会去右邻居；但实现上仍建议使用
`owner_of_1d(target)`，避免把规则写死到 rank+1。

### 4.6 2D Resident Halo Flow

2D resident halo 必须拆成两个 review 单元：

- B2 只处理 row-block window reader + direct writer + `+1,+1` followup，不处理
  direct reader / read-cache transition。
- B3 在 B2 稳定后，再加入 `waveEquation1.0` 所需 direct reader/cache resident
  state 和 `-1,-1` read-cache transition。

初始化：

```text
compute owned row range from output rows
allocate reader rows = owned_rows + top_halo + bottom_halo, cols = full cols
scatter reader owned rows into core
exchange top/bottom halo rows
allocate writer_owned = owned_rows * cols
```

每轮 run：

```text
kernel:
  for local item:
    local_row, col
    global_row = owned_row_begin + local_row
    HaloView2D reader(global_row, col)
    writer_owned[local_row, col]

route:
  target = (global_row + 1, col + 1)
  if target row owner == self:
    reader_core[target] = writer[source]
  else:
    pack to previous/next rank based on target row owner

read-cache transition:
  target = (source_row - 1, source_col - 1)
  apply same owner-based route to direct-reader/cache tensor

boundary:
  apply current supported boundary-local updates

halo:
  exchange top/bottom reader halo rows
  exchange direct-reader/cache halo if that tensor is read by next iteration
```

因为当前 2D ownership 是 row-block，每个 rank 持有完整 columns，所以 halo 只需要
top/bottom rows，不需要 left/right column exchange。

B2 的生成代码不得包含 direct-reader/cache halo；B3 的生成代码必须能结构断言
direct-reader/cache halo state、read-cache route 和对应 halo exchange 均存在。

### 4.7 Route message helper

建议新增 runtime helper：

```cpp
template <typename T>
struct RoutedValue1D {
    int64_t target = 0;
    T value{};
};

template <typename T>
struct RoutedValue2D {
    int64_t target_row = 0;
    int64_t target_col = 0;
    T value{};
};
```

首版可以不做通用 all-to-all，直接针对当前 unit route：

- 1D `+1`：最多与左右邻居交换少量值。
- 2D `+1,+1`：因为 target row 只跨相邻 row block，最多与上下邻居交换一行中的若干值。
- 2D `-1,-1` read-cache：同理最多与上下邻居交换。

后续再把 route generalize 到 sparse send lists。

### 4.8 Codegen 修改计划

新增文件：

- `rewriter/lib/mpi/operator_resident/ResidentHalo1DCodegen.cpp`
- `rewriter/lib/mpi/operator_resident/ResidentHalo2DCodegen.cpp`
- `rewriter/lib/mpi/operator_resident/ResidentHaloCodegenUtils.cpp`

更新文件：

- `OperatorResidentCodegen.cpp`
- `OperatorResidentCodegen_Internal.h`
- `OperatorResidentPlan.h`
- `Rewriter_MPI.cpp`
- `CMakeLists.txt`
- `dpcppLib/include/mpi/operator_resident/OperatorResidentRuntime.h`
- 新增 `dpcppLib/include/mpi/operator_resident/OperatorResidentHalo.h`
- `dpcppLib/include/mpi/common/KernelViews.h`

生成结构：

```cpp
struct __dacpp_mpi_or_halo_x_ctx {
    int mpi_rank;
    int mpi_size;
    sycl::queue q;
    Halo1DLayout / Halo2DRowLayout reader_layout;
    std::vector<T> reader_with_halo;
    std::vector<T> writer_owned;
    // optional direct reader/cache with halo
};

void init(ctx, args...) {
    compute layout;
    scatter owned reader;
    exchange_halo(reader);
}

void run(ctx, args...) {
    launch kernel with HaloView;
    publish writer resident;
    apply distributed route writer -> reader/core;
    apply read-cache transition if any;
    apply boundary-local update;
    exchange_halo(reader/cache);
}

void materialize(ctx, args...) {
    gather owned writer or reader state only when host-visible.
}
```

### 4.9 输出同步语义

P4.6 必须继续尊重 `OutputSyncRequirement`：

- `RootOnly`：只在 materialize 边界 gather 到 root，不 broadcast。
- `AllRanksNeeded`：必须让所有 rank 的 root-visible tensor 一致，可以先沿用
  gather + broadcast。
- `DistributedFollowup`：不能 broadcast materialized writer 来伪装正确；必须把
  writer route 到 reader/followup resident state，再 exchange halo。
- `RootCentricFollowup`：当前仍不进入 resident halo，继续 fallback 或保持普通 OR。

### 4.10 路线 B 验收

结构检查：

- halo accepted path 的 loop body 不出现 full reader `MPI_Bcast`。
- halo accepted path 的 loop body 不出现 full writer `MPI_Gatherv`。
- 只在 `materialize()` 出现 final gather。
- 出现 `exchange_halo_1d` 或 `exchange_halo_2d_rows`。
- 出现 `HaloView1D` / `HaloView2D`。

语义测试：

- `MDP1.0` / `liuliang1.0`：1D `+1` route 后下一轮继续读 reader。
- `stencil1.0`：2D row-block halo 边界正确。
- `waveEquation1.0`：2D direct reader / `-1,-1` read-cache transition 正确。
- rank boundary stress：tensor rows/items 不能整除 rank 数。
- one-rank fallback equivalence：`mpirun -np 1` 与 normal baseline 一致。

负向测试：

- root-bridge stencil 继续 reject。
- non-unit stride boundary-local 继续 reject。
- direct writer `READ_WRITE` 继续 reject。
- unsupported route count > 1 继续 reject。
- 上述 reject 用例必须结构断言“不出现 P4.6 loop/halo accepted path”，例如不出现
  halo ctx、`HaloView*`、`exchange_halo_*`，也不出现对应的 P4.6 accepted log。

---

## 5. Phase-C 复用边界

旧 MPI Phase-C 已经有两类可复用资产：一类是 AST/rewrite/同步分类等结构性逻辑，
另一类是基于 `AccessPattern` / `PackPlan` 的 root-centric 通信生成。P4.6 应复用
前者，避免把后者带回 OR/resident 路径。

### 5.1 可以复用或抽公共 helper 的部分

- loop-lowered rewrite 骨架：旧 Phase-C 的
  `rewriteStencilPhaseCSite()` 已经实现 loop 前插入 `ctx/init`、loop 内把 `<->`
  替换为 `run`、loop 后插入 `materialize`；P4.5 OR loop lowering 中也有同构逻辑。
  P4.6 开始前建议抽 shared rewrite helper，参数化 `ctx` 类型名、`init/run/materialize`
  函数名、ctx 变量前缀、outer loop、DAC expr 和 shell args，避免继续复制字符串拼接。
- `OutputSyncRequirement`：继续使用 shared
  `classifyOutputSyncRequirement()` / `requiresBroadcast()` / `outputSyncRequirementName()`。
  P4.6 不应新增一套 root-only / all-ranks-needed / distributed-followup 分类。
- stencil site / followup / post-region 分析：继续复用
  `analyzeDistributedStencilSite()`、`collectDistributedFollowupRegions()`、
  `collectRootCentricPostRegions()` 和已有 route / boundary-local 判定作为 guard。
  P4.6 的优化只能接管这些分析已证明 supported 的形态。
- Phase-C profiling 思路：可以借鉴现有 wrapper breakdown 和
  `reportCollectPositionsProfile()` 的输出位置，用于 Step 0 计数和 before/after 对照。

### 5.2 不应直接复用的部分

- 不直接搬 Phase-C 的 `AccessPattern` / `PackPlan` / root-centric gather-broadcast
  codegen。那套逻辑的通信中心是 root materialized tensor；P4.6 的目标是
  OR resident state、loop-lowered skeleton 和 halo exchange。
- 不让 P4.6 accepted path 依赖 legacy `AccessPattern` / `PackPlan`。这些可以作为
  fallback 或 measurement baseline 的参照，不能成为 resident halo 的执行主轴。
- Phase-C `materialize()` 中的 root gather / apply route 逻辑只能作为语义参照；
  halo accepted path 应生成 resident route + halo exchange，并只在 host-visible
  边界 final gather。
- 不把 Phase-C 的 `use_partial_exchange`、fallback cache 状态或 wave special-case
  直接变成 P4.6 IR。P4.6 应在 OR plan 中明确记录 `OrLoopLowerPlan` /
  resident-halo plan，review 时能看出该 patch 是路线 A 还是路线 B。

### 5.3 建议的准备 patch

在 Step 0 之前可以先做一个纯重构准备 patch：

```text
Phase-C / OR loop rewrite helper extraction
  - 不改变生成代码语义
  - Phase-C 和 P4.5 OR 输出结构保持等价
  - 默认 test_mpi.sh 仍为 37/37
```

该 patch 的目标只是让 P4.6 后续 skeleton / halo path 复用成熟的 loop rewrite 接口，
不是通信优化本身。

---

## 6. 推荐执行顺序

### Step 0：Measurement-only baseline

改动：

- 必须新增或记录 profile log / 结构计数脚本。
- 不改 lowering 行为。

验收：

- 记录当前 `MDP1.0`、`liuliang1.0`、`stencil1.0`、`waveEquation1.0`
  的 full broadcast/gather 次数和运行时间。
- 至少统计：full reader `MPI_Bcast`、direct-reader `MPI_Bcast`、writer
  `MPI_Gatherv`、followup/read-cache full tensor `MPI_Bcast`、final gather、
  scalar `MPI_Bcast`。
- 建立后续 patch 的 before/after 对照表；没有 Step 0 数据，不进入 P4.6
  行为改动。

### Step 1：路线 A direct/resident sync narrowing

改动：

- 巩固 P4.5 direct loop candidate。
- 对 loop 内不需要 per-run materialize 的 direct 形态，把 gather 移到
  `materialize()`。
- 必须新增正向用例证明 loop 内 output 没有 host-visible read；`decay1.0` 只能作为
  “仍需 per-run materialize” 的守护用例。

验收：

- direct tests 正确。
- 新增 final-materialize direct 用例结构上不在 `run()` 内生成 output gather。
- `decay1.0` 仍保持当前 P4.5 per-run materialize 语义。
- P4 stencil tests 不变。

### Step 2：路线 A stencil loop-lowered full-sync skeleton

改动：

- 给 `StencilWindow1D/2D` 生成 `ctx/init/run/materialize`，但仍保留 full reader
  sync 作为语义 baseline。
- hoist ownership/counts/displs/queue/buffer allocation。

验收：

- 结构出现 stencil `init/run/materialize`。
- 通信次数只减少 setup，不改变 full-sync 语义。

### Step 3：路线 A hoist loop-invariant full reader

改动：

- 当 reader 可证明 loop-invariant，full reader broadcast 移到 `init()`。
- loop-variant reader 继续在 `run()` refresh。
- 这是 opportunistic substep，不是进入 resident halo 前的必经阶段。

验收：

- 新增 positive test 钉住 invariant reader hoist。
- 新增 negative tests 钉住 reader variant reject / refresh-in-run。

### Step 4：路线 B 1D resident halo

改动：

- 新增 `OperatorResidentHalo.h`。
- 新增 `HaloView1D`。
- 新增 `ResidentHalo1DCodegen.cpp`。
- 支持 `StencilWindow1D` `+1` distributed followup。

验收：

- `MDP1.0`、`liuliang1.0` 通过。
- loop body 不再 full reader broadcast / writer gather。

### Step 5：路线 B 2D row-block resident halo（无 direct reader/cache）

改动：

- 新增 `HaloView2D`。
- 新增 `ResidentHalo2DCodegen.cpp`。
- 支持 row-block top/bottom halo。
- 支持 current `+1,+1` followup。
- 不支持 direct-reader / read-cache transition；相关用例继续走普通 OR 或后续 B3。

验收：

- `stencil1.0` 通过。
- rows 不整除 rank 数时正确。
- 结构断言不出现 direct-reader/cache halo state。

### Step 6：路线 B 2D read-cache transition

改动：

- 对 `waveEquation1.0` 所需 direct reader/cache 加 resident halo state。
- 支持 `-1,-1` read-cache transition route。

验收：

- `waveEquation1.0` 通过。
- mixed-site fallback 仍正确。
- 结构断言 direct-reader/cache halo state、read-cache route 和 halo exchange 存在。

---

## 7. Review Checklist

每个 P4.6 patch review 时必须回答：

- 这个 patch 是路线 A 还是路线 B？
- 是否复用了 shared Phase-C/OR rewrite 或 output-sync helper；如果没有，为什么需要
  新写一套？
- 是否只优化已 supported layout？
- 是否新增了 stencil 语义？如果新增，应移出 P4.6。
- loop 内 host-visible tensor 状态是否仍正确？
- root-only / all-ranks-needed / distributed-followup 是否分别处理？
- accepted path 是否不依赖 legacy `AccessPattern` / `PackPlan`？
- fallback 是否仍覆盖 unsupported route/root-bridge/FixedBlock？
- 是否有正向结构测试和负向 reject 测试？
- 如果 patch 声称减少通信，Step 0 的 before/after 计数是否同步更新？
- 如果 patch 只是 skeleton，是否明确说明“不以性能收益作为该 patch 验收条件”？

---

## 8. 最小首切片建议

如果只做一个最小 P4.6 patch，建议选两段式首切片：

```text
A-1:
  Phase-C / OR loop rewrite helper extraction
  不改变通信行为

A0:
  Measurement-only baseline
  建立通信静态计数和运行时间表

A1:
  StencilWindow1D loop-lowered full-sync skeleton
  不移除 full broadcast/gather
  只 hoist ownership/counts/displs/queue/buffer allocation
```

原因：

- A-1 先复用旧 Phase-C 和 P4.5 已证明稳定的 loop rewrite 形态，降低后续
  `ctx/init/run/materialize` skeleton 的接入风险。
- A0 先固定性能讨论口径，避免后续 patch 只凭直觉判断收益。
- A1 不改变通信语义，回归风险最低。
- 能为后续 halo ctx 结构打地基。
- 可以先验证 `Rewriter_MPI.cpp` 的 loop rewrite 对 stencil path 是否稳。

真正减少通信的首个 patch 建议选：

```text
路线 B Step 4:
  StencilWindow1D +1 resident halo
```

原因：

- 1D route 和 halo 都最简单。
- `MDP1.0` / `liuliang1.0` 能直接观察收益。
- 不涉及 2D boundary loop、read-cache transition 和 full row halo 的组合复杂度。
