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

### 1.1 Route A 后 large benchmark 观察

2026-05-09 的 large benchmark 暴露了两个性能/正确性边界：

- `matmul2048` 走 OR `RowPartitionFullRow`，旧生成代码曾按每个 output item
  复制 full row / full column payload，导致 `MPI_Scatterv` count 为
  `512*2048*2048 = 2,147,483,648`，超过 MPI `int` 上限。修复口径是按
  unique indexed row/column 存 payload：`matA[idx1][{}]` 按 owned rows
  scatter，`matB[{}][idx2]` 对所有 row-partition rank broadcast。修复后
  4 rank `matmul2048` 5 次 wall time 为 0.32 / 0.32 / 0.34 / 0.31 / 0.34s，
  输出均为 `147385 147273 147961`。
- `waveEquation1.0` large (`2048x2048`, 500 steps) 在 Route A full-sync skeleton
  下 300s 内未到最终输出。生成代码没有明显 deadlock；每步仍执行 full
  window-reader bcast、full direct-reader bcast、writer gatherv、read-cache full
  tensor replay+bcast、followup/boundary full tensor replay+bcast，并在 root/all-ranks
  反复 `tensor2Array` / `array2Tensor`。这说明 Route A 只完成结构铺垫，large
  2D stencil 的主要收益必须来自 Route B resident halo / resident cache。

### 1.2 当前 P4.6 进度与 large benchmark 快照

截至 2026-05-09 当前工作树，Route A loop-lowered full-sync skeleton、A3
安全 reader hoist、RowPartitionFullRow count 修复，以及 Route B B1/B2/B3 resident
halo 已落地。B1 当前接受保守的 1D `+1` distributed-followup 形态；boundary-local
只接受当前 codegen 能正确表达的字面左边界 target `0`，右边界或符号边界表达式继续走
Route A `StencilFullSync` fallback。B2 当前接受保守的 2D row-block resident halo：
exactly one 2D window reader、exactly one WRITE-only direct writer、当前
`+1,+1` distributed-followup 与当前 boundary-local updates；B3 在此基础上再保守加入
`waveEquation1.0` 当前所需的 at most one 2D direct reader、`-1,-1` read-cache
transition、direct-reader resident state 和当前四条零值 boundary-local updates。
无法证明 distinct tensor key、coverage、boundary 或 host-visible tensor 语义时，
仍继续 fallback 到 Route A `StencilFullSync`。空 owned-output rank 的 halo exchange
review 问题已修复：halo 交换按最近非空 rank 匹配，不再要求所有相邻 rank 都有
owned slice；B2/B3 还补齐了 2D scalar reject、shape-derived
count/displ/materialize/layout checked-mul guard 和过晚 guard 的前移。

2026-05-09 按 `docs/benchmarks` 里的大规模口径重跑了筛选后的 app benchmark：

```bash
MPI_ONLY_BENCH_TMP_DIR=/Volumes/QUQ/working/mpi_tmp/p46_b1_selected_large_20260509 \
MPI_ONLY_BENCH_RANKS=4 \
MPI_ONLY_BENCH_TIMEOUT_SECONDS=1800 \
python3 clang/tools/translator/bench_mpi_only_requested.py \
  DFT1.0 MDP1.0 decay1.0 gradientSum imageAdjustment1.0 jacobi1.0 \
  liuliang1.0 mandel1.0 matMul1.0 oddeven0.1 vectorAddCombo
```

本轮只统计 `mpirun` 运行阶段的外部 wall-clock，不包含翻译/编译，不列串行
baseline。本轮故意不包含尚未进入 large benchmark 对齐阶段的复杂 stencil：
`FOuLa1.0` 与尚未补 large-size before/after 数据的 `waveEquation1.0`；`stencil1.0`
和 `waveEquation1.0` 的 B2/B3 正确性由专项回归覆盖。

结果文件：
`/Volumes/QUQ/working/mpi_tmp/p46_b1_selected_large_20260509/results.tsv`。

| Test | Data size | Hand-written MPI+SYCL s | DAC translated MPI s | Status |
|---|---:|---:|---:|---|
| `DFT1.0` | `N=4096` | 0.877217 | 0.888717 | ok |
| `MDP1.0` | `N=8192, T=600` | 0.827930 | 0.828125 | ok |
| `decay1.0` | `numIsotopes=8192, steps=600` | 1.108226 | 0.891583 | ok |
| `gradientSum` | `8192x4096` | 0.662446 | 1.074782 | ok |
| `imageAdjustment1.0` | `4096x4096` | 0.717041 | 0.892109 | ok |
| `jacobi1.0` | `N=4096, iter=300` | 0.737592 | 0.877352 | ok |
| `liuliang1.0` | `WIDTH=8192, steps=1000` | 0.937474 | 0.840554 | ok |
| `mandel1.0` | `4096x4096, max_iter=1000` | 2.535479 | 3.029412 | ok |
| `matMul1.0` | `2048x2048` | 5.921520 | 0.902443 | ok |
| `oddeven0.1` | `N=4096` | 1.895137 | 5.617574 | ok |
| `vectorAddCombo` | `N=8388608` | 0.824343 | 0.749346 | ok |

解释口径：

- 大规模口径下，本表只比较手写 `MPI_StandardSycl` reference 和 DAC 翻译 MPI。
- `MDP1.0` / `liuliang1.0` 是当前 B1 resident halo 的重点观测项。
  本轮二者都与手写 MPI reference 基本齐平。
- `matMul1.0` 的 DAC 翻译 MPI 为 0.902443s，明显好于旧 benchmark 文档中的
  7.576529s，符合 RowPartitionFullRow unique-payload/count 修复后的预期。
- `stencil1.0` 已进入 Route B B2；它的正确性以专项结构/运行回归为准，不在这张
  非复杂 stencil benchmark 表里评估。
- `waveEquation1.0` 已进入 Route B B3 resident path，但 large-size benchmark 仍需
  另做 before/after 口径对齐，不在这张表里混算。

2026-05-09 晚些时候，又按 `docs/benchmarks/mpi_sycl_efficiency_benchmark_2026-05-06.md`
里的完整 14 个样例和同一组大规模口径，直接用
`clang/tools/translator/bench_mpi_only_requested.py` 重跑了一遍：

```bash
MPI_ONLY_BENCH_TMP_DIR=/Volumes/QUQ/working/mpi_tmp/docs_bench_20260509_210303 \
MPI_ONLY_BENCH_RANKS=4 \
MPI_ONLY_BENCH_TIMEOUT_SECONDS=1800 \
python3 clang/tools/translator/bench_mpi_only_requested.py
```

结果文件：
`/Volumes/QUQ/working/mpi_tmp/docs_bench_20260509_210303/results.tsv`。

| Test | Data size | Hand-written MPI+SYCL s | DAC translated MPI s | DAC/hand |
|---|---:|---:|---:|---:|
| `DFT1.0` | `N=4096` | 0.878103 | 0.698609 | 0.80x |
| `FOuLa1.0` | `m=8192, n=600` | 0.772162 | 1.608849 | 2.08x |
| `MDP1.0` | `N=8192, T=600` | 0.769904 | 0.818233 | 1.06x |
| `decay1.0` | `numIsotopes=8192, steps=600` | 0.843943 | 0.845569 | 1.00x |
| `gradientSum` | `8192x4096` | 0.660082 | 0.952764 | 1.44x |
| `imageAdjustment1.0` | `4096x4096` | 0.699212 | 0.947734 | 1.36x |
| `jacobi1.0` | `N=4096, iter=300` | 0.710596 | 0.827108 | 1.16x |
| `liuliang1.0` | `WIDTH=8192, steps=1000` | 0.826948 | 0.840660 | 1.02x |
| `mandel1.0` | `4096x4096, max_iter=1000` | 2.594314 | 3.077729 | 1.19x |
| `matMul1.0` | `2048x2048` | 5.883256 | 0.848780 | 0.14x |
| `oddeven0.1` | `N=4096` | 1.684896 | 4.965771 | 2.95x |
| `stencil1.0` | `2048x2048, steps=600` | 1.460170 | 2.217828 | 1.52x |
| `vectorAddCombo` | `N=8388608` | 0.661225 | 0.767165 | 1.16x |
| `waveEquation1.0` | `2048x2048, steps=600` | 1.527755 | 3.212622 | 2.10x |

从当前代码生成路径看，下一步优化优先级已经很明确：

1. `stencil1.0` / `waveEquation1.0` 虽然已经进入 Route B B2/B3 resident halo，
   但每步仍做本地 slab 级的 `apply_followup_2d(...)` / `apply_read_cache_transition_2d(...)`
   复制，再做 halo exchange；下一步应先把这两条路径改成 resident state 角色轮转
   或 buffer swap，而不是继续扩大新的 stencil 语义。
2. `FOuLa1.0` 当前并没有进入 P4.6 loop-lowered path，而是
   `[DACPP][MPI][OR][P4.6][Loop] ... rejected reason=not inside a stable loop site`；
   它的性能问题更像“还没吃到 P4.6 loop path”，下一步应在不放宽语义边界的前提下，
   增强稳定 loop site 识别，把这类 1D stencil 吃进现有 skeleton / resident family。
3. `oddeven0.1` 仍落在 legacy `AccessPattern` / `PackPlan` fallback，且
   `ctx.use_partial_exchange = false`；它需要的是 Phase 5 `FixedBlock`，不应把这部分
   工作混入 P4.6。

因此，P4.6 的下一轮实现顺序建议改为：

1. 先做 `StencilWindow2D` resident state 角色轮转：
   - `stencil1.0`：writer/reader 双缓冲 swap，替掉每步 `apply_followup_2d(...)`
   - `waveEquation1.0`：`prev/cur/next` 三缓冲轮转，替掉每步 read-cache/followup 全量本地复制
2. 只重跑 `MDP1.0`、`liuliang1.0`、`stencil1.0`、`waveEquation1.0` 的 focused benchmark
   和结构回归，先验证 resident 路径的真正收益
3. 再处理 `FOuLa1.0` 的 stable loop site gate
4. `oddeven0.1` 留到 Phase 5 `FixedBlock`

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

Route A 后的 `waveEquation1.0` large 是 B3 的核心动机：在不牺牲
host-visible tensor 语义的前提下，`matCur` / `matPrev` / `matNext` 需要长期
resident state，`matCur -> matPrev` 的 read-cache transition 和
`matNext -> matCur` 的 followup/boundary update 应更新 owned slice + halo，而不是
每步在 root 上重建完整 tensor 后 broadcast。

### 4.2 首批语义范围

路线 B 只覆盖当前 P4 已 accepted 的 stencil 切片，但必须分批进入：

- B1：`StencilWindow1D`
  - exactly one window reader
  - exactly one WRITE-only direct writer
  - no direct reader
  - supported `+1` distributed followup
  - supported literal left-boundary `0` boundary-local update only; right
    boundary or symbolic boundary expressions continue to full-sync fallback

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
  - 必须保留 Route A 的 full-sync fallback；若 resident cache coverage 或 boundary
    update 无法证明完整，继续使用 full-sync skeleton。

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

状态：已完成当前 B1 首切片。

实际改动：

- `KernelViews.h` 新增 `ResidentHaloView1D`。
- `OperatorResidentRuntime.h` 新增 1D resident halo layout、scatter、followup apply、
  nearest-nonempty-rank halo exchange、owned-slice helper。
- `OperatorChainAnalysis.cpp` 新增显式 `StencilResidentHalo` gate / metadata。
- `LoopLoweredStencil1DCodegen.cpp` 新增 1D resident halo `ctx/init/run/materialize`
  family。
- 支持 `StencilWindow1D` `+1` distributed followup。

验收：

- `MDP1.0`、`liuliang1.0` 通过并走 `StencilResidentHalo`。
- `mpiLoopStencilResidentHalo1D` 结构测试通过。
- `mpiLoopStencilResidentHaloEmptyRank1D` 覆盖 `outputTotal < mpi_size` 空 rank
  exchange 边界。
- `mpiLoopStencilRightBoundaryFullSync1D` 覆盖右边界 fallback。
- loop body 不再 full reader broadcast / writer gather。

### Step 5：路线 B 2D row-block resident halo（无 direct reader/cache）

状态：已完成。B2 保持独立于 Phase-C `AccessPattern` / `PackPlan` / root-centric
codegen，也没有提前支持 `waveEquation1.0` 的 direct reader / read-cache transition。

改动：

- 新增 `HaloView2D`。
- 新增 `ResidentHalo2DCodegen.cpp`。
- 支持 row-block top/bottom halo。
- 支持 current `+1,+1` followup。
- 显式 reject scalar reader，并在 2D resident/full-sync materialize、layout、count、
  displ、payload/buffer size 路径补齐 checked-mul / MPI `int` 上界 guard。
- 不支持 direct-reader / read-cache transition；相关用例继续走普通 OR 或后续 B3。

验收：

- `stencil1.0` 通过。
- rows 不整除 rank 数时正确。
- 结构断言不出现 direct-reader/cache halo state，也不出现 `AccessPattern`、
  `PackPlan`、`FixedBlock`、`root_bridge` 或 Phase-C partial exchange。
- `mpiLoopStencilScalarReject2D`、`mpiLoopStencilCountGuard2D` 通过。

### Step 6：路线 B 2D read-cache transition

状态：已完成。B3 保持在 B2 的 row-block resident-halo/runtime 基础上，只补
`waveEquation1.0` 当前所需的 direct-reader / read-cache 增量，不引入 root-bridge、
更广 route、dynamic shape 或 Phase 5 `FixedBlock`。

改动：

- 对 `waveEquation1.0` 所需 direct reader/cache 加 resident state。
- 支持 at most one 2D direct reader，并保守要求 window/direct-reader/writer 的
  actual tensor key 可证明 distinct。
- 支持 `matCur -> matPrev` 的 `-1,-1` read-cache transition route，并在 `run()`
  内先从 resident window reader 刷新 direct-reader local slice，再做 writer->reader
  `+1,+1` followup、halo exchange 与 boundary refresh。
- direct-reader resident tensor 在 `init()` 一次性 row-block scatter，在
  `materialize()` 单独 `MPI_Gatherv` 恢复 host-visible `matPrev`。
- 当前只接受 `waveEquation1.0` 形态的四条零值 top/bottom/left/right
  boundary-local updates，以及当前 `DAC -> read-cache -> followup -> boundary`
  顶层语句顺序；证明不完整继续 fallback 到 Route A `StencilFullSync`。

验收：

- `waveEquation1.0` 通过。
- mixed-site fallback 仍正确。
- 结构断言 direct-reader/cache resident state、read-cache route 和 halo exchange 存在，
  且 accepted path 不出现 `AccessPattern`、`PackPlan`、`FixedBlock`、`root_bridge`
  或 Phase-C partial exchange。
- `mpiLoopStencilOrderReject2D` 通过，钉住“同 shape 但 post-loop 语句顺序不同”
  的 2D stencil 必须拒绝 B3 并回退到 `StencilFullSync`。

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
