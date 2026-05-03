# MPI Translator Stencil, Wrapper Consistency And Post-Shell Regions

更新时间：2026-05-03

## 1. 总目标和边界

本文是 MPI translator 当前状态的单一入口，合并记录：

- MPI stencil loop 路径状态。
- Phase 1 保守优化。
- wrapper / stencil 输出副本一致性分类。
- post-shell root-centric region v1。
- 后续“缓存一致性 + 部分交换”方向。

当前总目标：

- 保持 root-centric MPI wrapper / MPI stencil 语义正确。
- 对位于 time-step loop 内的 `<->` 生成 `ctx + init + run` 形态。
- 把稳定的通信 metadata 缓存在 `init()`，让 `run()` 只做每步数据相关通信。
- 建立输出副本一致性模型，区分 root 结果正确和非 root 副本正确。
- 识别 shell 后明显可并行的一维 host loop，并在 MPI stencil v1 中生成 root-centric region helper。
- 在现有 pack / writeback plan 基础上，探索用缓存一致性和部分交换替代手写传统 halo 的 stencil 优化路线。

当前明确不做或尚未完成：

- 不修改 DACPP 源码语法。
- 不新增 CLI flag。
- 不改变普通 MPI wrapper 的基本语义。
- Phase 1 不做 rank 间 halo exchange。
- Phase 1 不把 sibling update loops 合并进 rank-local distributed stencil pipeline；v1 只生成 root-centric helper。
- 尚未完成真正分布式 stencil / halo / generalized cache exchange。

## 2. 当前 MPI stencil 路径

这条路径服务于 MPI 下的 stencil / time-step 类循环：

- 输入使用 `--mode=buffer --mpi`。
- 源码中存在位于 `for` / `while` 内部的 `<->`。
- shell 实参在外层循环之前已经声明，循环内每次迭代复用这些对象。

目标生成形态：

```cpp
__dacpp_mpi_stencil_ctx_xxx ctx;
__dacpp_mpi_stencil_init_xxx(ctx, ...);
for (...) {
    __dacpp_mpi_stencil_run_xxx(ctx, ...);
}
```

这样可以把稳定不变的 pattern / binding / pack plan / rank range 初始化从每次迭代中提出来，循环内只执行每步真正需要重复的 scatter、kernel、gather、writeback、broadcast。

当前采用 `ctx + init + run + compatibility wrapper` 结构：

- `ctx`
  - 保存 MPI rank / size、SYCL queue、AccessPattern、PackPlan、ItemRange、cached layout 和复用 buffer。
- `init`
  - 在循环外执行一次。
  - 初始化 pattern、binding split sizes、pack plan、本 rank item range、gather / scatter layout。
- `run`
  - 在每次循环迭代执行。
  - 完成数据分发、本地 kernel、写回收集和必要 broadcast。
- compatibility wrapper
  - 对非 loop stencil site 保留一次性调用形式。
  - 内部执行 `ctx; init(ctx, ...); run(ctx, ...);`。

不能安全 hoist 的场景会回退普通 MPI wrapper 路径：

- shell 实参是在循环体内部临时声明的 view / tensor。
- 每次迭代 shell 实参的形状或绑定语义会变化。
- 不能安全把 `init(ctx, args...)` 提到外层循环之前的 `<->`。

## 3. 已完成的 stencil 接线和修复

### 3.1 AST 侧记录 MPI stencil site

涉及文件：

- `clang/tools/translator/parser/include/DacppStructure.h`
- `clang/tools/translator/translator.cpp`

已加入 `MpiStencilSite`，记录：

- `<->` 的表达式编号。
- 当前 `BinaryOperator* dacExpr`。
- 包裹 `<->` 的外层 loop。

当前登记条件已经收紧：

- `<->` 必须位于 `for` / `while` 中。
- shell 调用实参引用的非全局变量必须声明在外层 loop 之前。

这样可以避免把循环体内部临时变量错误 hoist 到循环外。

### 3.2 MPI 总入口分流

涉及文件：

- `clang/tools/translator/rewriter/lib/Rewriter_MPI.cpp`

逻辑：

```cpp
if (dacppFile && dacppFile->hasMPIStencilSites()) {
    rewriteMPIStencil();
    return;
}
```

普通 MPI wrapper 路径仍然保留；只有安全登记为 stencil site 的 loop `<->` 才进入新路径。

### 3.3 Stencil rewriter 文件

新增或接入文件：

- `clang/tools/translator/rewriter/include/Rewriter_MPI_Stencil_Common.h`
- `clang/tools/translator/rewriter/lib/Rewriter_MPI_Stencil.cpp`
- `clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Stencil_Analysis.cpp`
- `clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Stencil_Codegen.cpp`
- `clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_OutputAnalysis.cpp`
- `clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_PrintRewrite.cpp`
- `clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_ParamAnalysis.cpp`
- `clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_PostRegion_Analysis.cpp`
- `clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_PostRegion_Codegen.cpp`
- `clang/tools/translator/rewriter/include/Rewriter.h`
- `clang/tools/translator/CMakeLists.txt`

`rewriteMPIStencil()` 当前负责：

1. 复用普通 MPI prelude。
2. 生成 local calc。
3. 生成 stencil ctx / init / run / compatibility wrapper。
4. 删除原 shell / calc 声明。
5. 在 `main` 中插入 MPI init / finalize。
6. 在 loop 前插入 ctx/init。
7. 将 loop 内 `<->` 替换为 `run(ctx, ...)`。
8. 设置 `mainAlreadyRewritten`，避免通用 `rewriteMain()` 二次替换。

### 3.4 修复错误 hoist

问题：

`FOuLa1.0` 中 `<->` 位于循环内，但 shell 实参 `u_kin`、`u_kout`、`r` 都是在循环体内部临时声明的。旧逻辑只要看到 `<->` 在 loop 中就走 stencil 路径，导致 `init` 被插到变量声明之前，生成代码编译失败。

修复：

- 在登记 `MpiStencilSite` 前检查 shell 实参引用的变量声明位置。
- 如果变量不是全局变量，且声明位置不在外层 loop 之前，则不登记 stencil site。
- 该 `<->` 回退普通 MPI wrapper。

### 3.5 修复 MPI 写回 gather 值错位风险

问题：

旧代码用：

```cpp
MPI_Gatherv(local_xxx.data(), send_count_xxx, ...);
```

但 `send_count_xxx` 对应的是 `writeback_globals` 数量，不一定等于 `local_xxx` 前缀中需要写回的元素。对于 compact pack / writeback globals 不等于 pack globals 前缀的情况，会把错误 slot 的值 gather 回 root。

修复：

普通 MPI wrapper 和 stencil run 都改为：

```cpp
std::vector<T> writeback_values_xxx =
    dacpp::mpi::build_writeback_values_parallel(local_xxx, pack_xxx);
MPI_Gatherv(writeback_values_xxx.data(), send_count_xxx, ...);
```

这样 gather 的值和 `writeback_globals` 一一对应。

## 4. Phase 1 保守优化

Phase 1 只优化当前 root-centric MPI stencil 路径，不改变通信语义，不引入 halo 分布式 stencil。

目标：

- 把 `run()` 中每个 time step 重复计算的通信元数据搬到 `init()`。
- 复用本地和 root 侧临时 buffer。
- 保持当前 scatter / kernel / gather / broadcast 语义不变。
- 不修改 DACPP 源码语法。
- 不新增命令行开关。
- 不改变普通 MPI wrapper 路径。

### 4.1 旧生成代码的问题

以 `waveEquation1.0` 为例，Phase 1 前的 MPI stencil `run()` 每次迭代都会重复执行：

- `MPI_Gather(&local_global_count_*)`
- `MPI_Gatherv(pack.globals)`
- `MPI_Gather(&send_count_*)`
- `MPI_Gatherv(writeback_globals)`
- counts / displs 计算
- root 侧 send / recv 临时 vector 重新分配
- 默认情况下每轮执行 wrapper timing `MPI_Reduce`

这些信息只依赖 shape、split、binding 和 rank item range，不依赖每步 tensor 数据，适合在 `init()` 中缓存。

### 4.2 Runtime helper

涉及文件：

- `clang/tools/translator/dpcppLib/include/MPIPlanner.h`
  - 生成代码统一 include 的兼容入口。
- `clang/tools/translator/dpcppLib/include/mpi/Common.h`
  - 共享基础聚合头，转入 `CoreTypes.h`、`Profile.h`、`MpiTypes.h`、`Pattern.h`。
- `clang/tools/translator/dpcppLib/include/mpi/Wrapper.h`
- `clang/tools/translator/dpcppLib/include/mpi/WrapperPack.h`
  - 普通 wrapper 仍使用的 `PackPlan`、global pack、writeback helper。
- `clang/tools/translator/dpcppLib/include/mpi/Stencil.h`
- `clang/tools/translator/dpcppLib/include/mpi/StencilTypes.h`
- `clang/tools/translator/dpcppLib/include/mpi/StencilLayout.h`
- `clang/tools/translator/dpcppLib/include/mpi/StencilExchange.h`
  - stencil ctx/init/run、Phase C partial exchange、root bridge 使用的 layout / distributed state / exchange helper。
- `clang/tools/translator/dpcppLib/include/mpi/Views.h`
  - view 聚合头，转入 `KernelViews.h` 和 `RegionViews.h`。

新增：

- `dacpp::mpi::GatheredIndexLayout`
  - `local_count`
  - `counts`
  - `displs`
  - `byte_counts`
  - `byte_displs`
  - `globals`
- `init_gathered_index_layout(...)`
  - 在 `init()` 中一次性收集 local count、displs 和 root 侧 globals。
- `init_layout_byte_counts(...)`
  - 为 byte transport 类型预计算 byte counts / displs。
- `pack_values_by_globals_parallel_range_into(...)`
  - root 侧复用 send buffer，按 cached globals 打包输入值。
- `build_local_slots_for_globals(...)`
  - 为 writeback globals 预计算本地 slot。
- `pack_values_by_slots_parallel_into(...)`
  - 每步按 cached slots 抽取 writeback values 到复用 buffer。

### 4.3 Stencil codegen

涉及文件：

- `clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Stencil_Codegen.cpp`

已调整：

- `ctx` 为每个参数保存 `local_<name>`。
- `READ/READ_WRITE` 参数保存：
  - `input_layout_<name>`
  - `global_<name>`
  - `sendbuf_<name>`
- `WRITE/READ_WRITE` 参数保存：
  - `output_layout_<name>`
  - `writeback_slots_<name>`
  - `writeback_values_<name>`
  - `global_recv_values_<name>`
  - `global_out_<name>`
- `init()` 构建 pack plan 后初始化 input / output layout。
- `run()` 删除重复 metadata gather，只保留数据相关通信。
- `WRITE` 参数每轮 kernel 前重置 local buffer，保持原 fresh vector 语义。
- wrapper timing `MPI_Reduce` 只在 `DACPP_MPI_PROFILE` 开启时执行。

### 4.4 当前生成代码形态

`waveEquation.mpi.dac_sycl_buffer.cpp` 现在的 stencil context 中包含：

```cpp
dacpp::mpi::GatheredIndexLayout input_layout_cur;
std::vector<double> global_cur;
std::vector<double> sendbuf_cur;
dacpp::mpi::GatheredIndexLayout output_layout_next;
std::vector<int32_t> writeback_slots_next;
std::vector<double> writeback_values_next;
```

`init()` 中执行一次：

```cpp
dacpp::mpi::init_gathered_index_layout(ctx.input_layout_cur, ...);
dacpp::mpi::init_gathered_index_layout(ctx.output_layout_next, ...);
dacpp::mpi::build_local_slots_for_globals(...);
```

`run()` 中只保留：

```cpp
MPI_Scatterv(... input_layout_*.counts/displs ...);
MPI_Gatherv(writeback_values_*.data(), ... output_layout_*.counts/displs ...);
```

结构检查结果：

- `run()` 中不再出现 `MPI_Gather(&local_global_count_*)`。
- `run()` 中不再出现 `MPI_Gatherv(...pack.globals...)`。
- `run()` 中不再出现 `MPI_Gatherv(...writeback_globals...)`。
- `MPI_Reduce` 仍存在于生成代码中，但已包在 `dacpp::mpi::profilingEnabled()` 分支内。

## 5. 副本一致性问题

涉及文件：

- `clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_OutputAnalysis.cpp`
- `clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_PrintRewrite.cpp`
- `clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_OutputAnalysis_Internal.h`

MPI 翻译后，一个 tensor 的一致性分成两层：

1. root 结果一致性。
2. 非 root 本地副本一致性。

root 结果一致性要求把每个 rank 计算出的局部写回收集到 root，并在 root 上重建完整 tensor。这一步需要 `MPI_Gatherv`，因为每个 rank 写回的 global index / value 数量和位置都可能不同。

非 root 本地副本一致性是另一件事。root 重建 tensor 后，其他 rank 上同名 tensor 仍然可能是旧值。如果后续 host 代码会在所有 rank 上读取该 tensor，则需要把 root 的完整结果再 `MPI_Bcast` 给所有 rank。

因此 `Gatherv` 和 `Bcast` 不是二选一关系：

- `MPI_Gatherv`：收集分布式输出，保证 root 拥有正确全局结果。
- `MPI_Bcast`：把 root 的全局结果同步回其他 rank，保证后续 all-rank host 读看到新值。

当前普通 MPI wrapper / stencil 输出路径大体是：

```cpp
MPI_Gather(send_count);
MPI_Gatherv(writeback_globals);
MPI_Gatherv(writeback_values);
if (rank == 0) {
    apply_writeback_by_globals(...);
}
if (requiresBroadcast(syncRequirement)) {
    MPI_Bcast(full_tensor);
}
```

`tensorNeedsBroadcast(...)` 仍保留为兼容 wrapper，但内部已经升级为 `classifyOutputSyncRequirement(...)`：

- `RootOnly` / `root-only`：只要求 root 结果正确，不生成 broadcast。
- `AllRanksNeeded` / `all-ranks-needed`：后续普通 host 代码需要 all-rank 副本，生成 `Gatherv + Bcast`。
- `RootCentricFollowup` / `root-centric-followup`：后续读落在已识别并会被替换的 root-centric post-shell region 中，不先 broadcast shell 输出。
- `DistributedFollowup` / `distributed-followup`：当前只用于满足 Phase C 条件的 loop-lowered MPI stencil site。普通 wrapper 路径和不在 `rewriteMPIStencil()` outer-loop lowering 内的 site 不会生成该分类。

### 5.1 `decay1.0` 的 root-only 基准

`decay1.0` 源码形态：

```cpp
while (t_tensor[0] <= T) {
    DECAY(N0s_tensor, lambdas_tensor, local_A_tensor, t_tensor) <-> decay;
    A_tensor[10*t_tensor[0]] = local_A_tensor;
    t_tensor[0] += dt;
}
```

当前它实际进入 MPI stencil 路径，而不是普通 wrapper 路径。生成代码中，`local_A` 的写回值会通过 `MPI_Gatherv(writeback_values_local_A...)` 收到 root，并在 root 上 `apply_writeback_by_globals(...)` 后写回 `local_A`。

`decay_chain.MPI_StandardSycl.cpp` 证明了一个关键基准：`Gatherv` 和 `Bcast` 不是绑定关系。标准 MPI/SYCL 版本中，每个 rank 只计算本地 `local_A`，root 通过 `MPI_Gatherv` 收集完整结果，后续全局 `A` 的写入和最终输出只要求 rank 0 正确，因此不需要把 `local_A` 再 broadcast 回所有 rank。

当前生成代码对 `decay1.0` 的日志为：

```text
[DACPP][MPI] output local_A sync=root-only
```

对应生成结构是：

```cpp
MPI_Gatherv(writeback_values_local_A.data(), ...);
if (mpi_rank == 0) {
    apply_writeback_by_globals(...);
    local_A.array2Tensor(global_out_local_A);
}
```

没有为 `local_A` 生成 `MPI_Bcast`。

仍需注意的风险点在后续 host 语句：

```cpp
A_tensor[10*t_tensor[0]] = local_A_tensor;
```

这行在原始 AST 中仍是普通 host 语句。如果它未来被证明是 all-rank observable 计算，而不是 root-only observable 路径，则需要分类为 `AllRanksNeeded` 或把后续语句也 region 化。v1 对 `decay1.0` 的目标是对齐标准 MPI/SYCL 的 root-only 基准，不把 `Gatherv` 机械绑定到 `Bcast`。

### 5.2 一致性分类 v1

当前已经把 bool 级别的 `needsBcast` 内部升级成后续使用分类。普通 MPI wrapper 和 MPI stencil codegen 都先拿分类，再用 `requiresBroadcast(...)` 决定是否生成 `MPI_Bcast`。

v1 分类规则保守：

- 分类入口先把 shell 形参名映射成当前 `<->` 调用的实际实参名；例如 `DECAY(..., local_A_tensor, ...)` 会用 `local_A_tensor` 扫描后续 host 使用，而不是只看 shell 形参 `local_A`。
- 没有后续读，或当前策略认定只要求 root 结果正确：`RootOnly`。
- 后续读全部落在已支持、且会被替换的 root-centric post-shell region 中：`RootCentricFollowup`。
- 后续存在普通 host read / read-write，或分析不能证明安全：`AllRanksNeeded`。
- v1 支持简单 root-only observable 传播：如果输出实参只被赋给另一个 tensor，且传播目标最终只进入 root 可观察输出，例如 `A_tensor[...]=local_A_tensor` 后 `A_tensor[1].print()`，仍可分类为 `RootOnly`。
- root-only observable 当前和输出语句 rewrite 闭环：`.print()` 和 `std::cout/cout << ...` 都会被分析层识别为 root-only observable，且会被 codegen 改写成 `if (__dacpp_mpi_is_root_rank()) { ... }`。
- `DistributedFollowup` 不再只是占位；当前只有在 `rewriteMPIStencil()` loop lowering 路径、且 site 通过 1D `dacpp::Vector` + 有效 `READ/WRITE` + post-shell region 可识别等 guard 时，才会进入这条分类。

当前 Broadcast analyze 的实际决策树：

1. `classifyOutputSyncRequirement(dacppFile, tensorName, dacExpr)` 先解析当前 `<->` 的实际 tensor 名。
2. `TensorUseVisitor` 从当前 `<->` 之后扫描 main body 中的后续读写。
3. 读使用按位置分类：
   - 落在 root-centric post-shell region：计入 `HasReadInsideRootRegion`。
   - 落在 `.print()` / `std::cout` 输出语句：视为 root-only observable，不计入 all-rank read。
   - 落在赋值 RHS：记录传播目标，递归证明传播目标最终是否只进入 root-only observable。
   - 其他普通 host read / read-write：计入 `HasReadOutsideRootRegion`。
4. 分类结果再交给 `requiresBroadcast(...)`：
   - `RootOnly`：不 `MPI_Bcast`。
   - `RootCentricFollowup`：不 `MPI_Bcast`。
   - `AllRanksNeeded`：生成 `Gatherv + Bcast`。
   - `DistributedFollowup`：只在 loop-lowered partial-exchange site 上出现。若运行时检查关闭 partial path，生成代码会整站退回旧的 root-centric path，并保守走 broadcast-safe 行为。

当前验证到的分类：

- `decay1.0`：`local_A sync=root-only`，无 `MPI_Bcast`。
- `liuliang1.0`：`new_rho sync=root-centric-followup`，无 `MPI_Bcast`。
- `mpiDenseCoverSibling1.0`：`updates sync=all-ranks-needed`，保留 `MPI_Bcast`。

### 5.3 MPI 输出语句 root-only rewrite

MPI 代码生成不再通过全局 stdout 重定向来压掉非 root 输出。也就是说，生成模板中已经删除了：

```cpp
if (mpi_rank != 0) {
    std::freopen("/dev/null", "w", stdout);
}
```

当前做法是更局部地改写可见输出语句：在 MPI 生成代码中插入 root-rank helper，并把输出语句包成 root-only。

```cpp
static inline bool __dacpp_mpi_is_root_rank();

static inline bool __dacpp_mpi_is_root_rank() {
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        return true;
    }
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank == 0;
}
```

典型改写：

```cpp
if (__dacpp_mpi_is_root_rank()) {
    A_tensor[1].print();
}
```

以及：

```cpp
if (__dacpp_mpi_is_root_rank()) {
    std::cout << rho[15] << std::endl;
}
```

覆盖范围：

- `dacpp` tensor 的 `.print()` 成员调用，包括 `A_tensor[1].print()` 这类下标 / member chain。
- `std::cout << ...` / `cout << ...` 输出语句。
- `main()` 之外的用户函数输出，例如 `mandel1.0` 的 `PrintStats()`、`imageAdjustment1.0` 的 `printImage()`。
- 无花括号 loop body 中的输出语句，例如 `gradientSum` 中 `for (...) std::cout << ...;`。

这个 rewrite 是输出可观察行为层面的 guard，不等同于 tensor 副本一致性。`RootOnly` 分类仍然表示 root 结果正确即可；`AllRanksNeeded` 仍然会为后续 all-rank host read 保留 `MPI_Bcast`。

闭环关系：

- Broadcast analyze 把 `.print()` / `std::cout` 输出读识别为 root-only observable。
- Output rewrite 保证这些 observable 真的只在 rank 0 执行。
- 因此当 shell 输出只流向这些 observable，或者只通过简单赋值传播到这些 observable 时，可以安全分类为 `RootOnly` 并跳过 `MPI_Bcast`。
- 如果同一个 tensor 后续还有普通 host read，仍然分类为 `AllRanksNeeded`，不会因为存在一个 root-only 输出就跳过必要 broadcast。

## 6. Post-shell 可并行循环问题

涉及文件：

- `clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_PostRegion_Analysis.cpp`
- `clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_PostRegion_Codegen.cpp`
- `clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_PostRegion_Internal.h`

`liuliang1.0` 源码中，shell 之后紧跟两个 host loop：

```cpp
LWR_shell(rho, new_rho) <-> lwr;
for (int i = 1; i <= WIDTH-2; i++) {
    rho[i] = new_rho[i-1];
}
for (int i = 0; i < 1; i++) {
    rho[0] = new_rho[0];
}
```

第一个 loop 是明显的 element-wise copy / shift：

- 每个迭代写不同的 `rho[i]`。
- 每个迭代读 `new_rho[i-1]`。
- 没有 loop-carried dependency。
- 适合翻译成 `sycl::parallel_for`。

第二个 loop 实际只有一次迭代，语义上是边界赋值。它也可以被 region 化，但没必要为了性能单独并行；更合理的是作为小 region 或直接 root/all-rank host assignment 处理。

非 MPI buffer 路径已经能做类似优化。当前生成的普通 SYCL buffer 代码中，`liuliang1.0` 会生成：

```cpp
__dacpp_submit_region_LWR_shell_lwr_stmt_0(ctx);
__dacpp_submit_region_LWR_shell_lwr_stmt_1(ctx);
```

其中 `stmt_0` 内部就是：

```cpp
h.parallel_for(sycl::range<1>(__N), ... {
    d_rho[i] = d_new_rho[i-1];
});
```

MPI stencil v1 现在会把已支持的 sibling loop 替换成 root-centric helper：

```cpp
__dacpp_mpi_stencil_run_LWR_shell_lwr(ctx, rho, new_rho);
__dacpp_mpi_region_LWR_shell_lwr_stmt_0(ctx, rho, new_rho);
__dacpp_mpi_region_LWR_shell_lwr_stmt_1(ctx, rho, new_rho);
```

helper 入口现在先显式计算 root guard：

```cpp
bool __dacpp_root_rank = (ctx.mpi_rank == 0);
if (__dacpp_root_rank) {
    ...
}
```

`stmt_0` 在 root 上用 SYCL buffer + `parallel_for` 执行：

```cpp
acc_rho[i] = acc_new_rho[i-1];
```

`stmt_1` 也作为小的一维 region 处理：

```cpp
acc_rho[0] = acc_new_rho[0];
```

因此 `new_rho` 的输出分类为 `RootCentricFollowup`，shell 输出 `Gatherv` 到 root 后不再为了原始 all-rank host loop 生成全量 `MPI_Bcast`。

在当前 Phase C 初版里，helper 还会为被识别为 root-bridge 写入的 tensor 预留一次 root -> distributed cache 刷新：

```cpp
if (ctx.use_partial_exchange && ctx.dist_state.root_bridge_plan.supported) {
    dacpp::mpi::exchange_values_by_slots(...);
}
```

这一步只在站点真正启用了 partial-exchange 时生效；`liuliang1.0` 目前由于 effective `READ_WRITE` kernel param 仍会整站回退，所以它继续走 root-centric helper + 旧通信路径，而不会启用 Phase C 分支。

### 6.1 MPI 下把 post-shell loop 变成 `parallel_for`

当前已经落地 root-centric v1，但仍需要分层推进。

保守实现：

1. shell 输出仍然 `Gatherv` 到 root。
2. 如果 post-shell loop 需要 all-rank 输入，则继续 `Bcast` 必要 tensor。
3. 将后续可并行 host loop 在每个 rank 上用 SYCL `parallel_for` 执行。

优点是实现风险低，语义接近当前代码；缺点是仍有全量 broadcast，每个 rank 仍可能重复执行同样 region。

已实现的 root-centric region v1：

1. shell 输出 `Gatherv` 到 root。
2. root 在 SYCL 上执行 post-shell region，更新 root tensor。
3. 根据下一次使用决定是否 broadcast `rho`。

对 `liuliang1.0`，下一轮 `LWR_shell(rho, new_rho)` 需要读取 `rho`。在当前 root-centric scatter 模型下，每轮 shell 输入本来只要求 root 拥有最新 `rho`，所以 post-shell loop 可以只在 root 上执行，然后下一轮由 root scatter `rho` 的局部窗口。

未来的 distributed follow-up region：

1. rank 本地执行 shell kernel。
2. rank 本地直接执行后续 copy / shift region。
3. 只在必要边界交换或最终 observable 点做通信。

这会更接近真正分布式 stencil / time-step 执行，但需要更完整的 region 化、依赖分析、边界处理和一致性建模。

### 6.2 建议的识别条件

v1 支持的 post-shell loop 条件：

- `BufferRegionPlan` 已经识别为 `<->` 后 sibling statement。
- 语句是简单 `for` loop。
- loop bound 可直接抽取为一维 `__L` / `__R` / `__N`。
- loop body 中写入的 tensor 下标是当前 loop induction variable 的 affine 表达。
- body 是简单赋值，不包含嵌套控制流、I/O、MPI 调用、`std::` 调用等。
- 涉及的 tensor 是 shell 参数中的一维 `dacpp::Vector`。
- 不捕获非 shell 变量。

对 `liuliang1.0` 的第一个 loop：

```cpp
rho[i] = new_rho[i-1];
```

满足这些条件，已生成 `__dacpp_mpi_region_LWR_shell_lwr_stmt_0`。

对边界赋值：

```cpp
rho[0] = new_rho[0];
```

当前作为小 region 生成 `__dacpp_mpi_region_LWR_shell_lwr_stmt_1`。关键点是它被纳入一致性分析：`new_rho` 的后续读全部落在 root-centric helper 中，所以不需要先 broadcast `new_rho`。

v1 不支持的 region：

- `dacpp::Matrix` / 二维 sibling loop。
- 多语句复杂 loop body。
- 函数调用、别名、slice、间接访问。
- rank-local distributed follow-up。

## 7. 基于缓存一致性和部分交换的 stencil 方向

用户提出的思路：

> 当前 MPI wrapper 已经把需要的内存重复 copy 到本地了。只要解决好缓存一致性问题，就可以不用传统手写 halo 的方式解决 stencil，而是在现在的逻辑上加部分交换。

这个方向总体成立，但需要精确定义：它不是完全不需要 halo，而是不必局限于传统结构化 halo。更准确地说，是基于 global index / pack plan / writeback plan 的 generalized halo 或 cache exchange。

当前代码已经具备的基础：

- 每个 rank 有自己的 `local_*` buffer。
- input pack plan 已经知道本 rank 需要读哪些 global index。
- output writeback plan 已经知道本 rank 会写哪些 global index。
- Phase 1 后，counts / displs / globals / slots 这类 metadata 可以缓存。
- 对 stencil 来说，读窗口通常比写窗口大，例如读 `rho[i]`、`rho[i+1]`，写 `new_rho[i]`。多出来的读元素其实就是 ghost/cache 副本。

因此后续可以把 root-centric 每步通信：

```text
root pack -> Scatterv input window -> kernel -> Gatherv output -> root apply -> optional Bcast
```

逐步优化成：

```text
本地保留 read cache
根据 writeback_globals 标记 dirty
只交换下一步会被其他 rank 读取的 dirty overlap
本地 kernel
必要时在 observable 点 gather 或 broadcast
```

对规则 stencil，这会退化成自动生成的左右边界、上下边界或 tile halo exchange。对不规则访问，它仍然可以作为基于 global index 集合交集的 general peer exchange。

需要建模的核心概念：

- owner：某个 global element 的主拥有者。
- reader set：哪些 rank 的 read cache 持有该 global element。
- writer set：哪些 rank 可能写该 global element。
- dirty set：本步被写过、其他 cache 可能变旧的 global elements。
- exchange plan：dirty set 和其他 rank read set 的交集，决定每步 peer-to-peer 发送什么。
- post-shell region 影响：host loop region 也会读写 tensor，必须参与 dirty / reader / writer 分析。

风险和限制：

- 双 buffer stencil 比 in-place stencil 容易。`rho -> new_rho` 这种读旧写新，只要在 step 边界处理一致性；原地更新需要处理同一步内读写顺序。
- 规则访问模式可以生成简洁的邻居交换；不规则访问可能退化成更通用但更重的 index exchange。
- 如果后续 host 代码仍在所有 rank 上执行，则需要 all-rank 副本一致；如果能降成 root-only 或 distributed region，通信策略完全不同。
- 仅说“解决缓存一致性”会低估难度，必须显式建模 owner、reader、writer、dirty 和 exchange plan。

结论：

现有 MPI wrapper / stencil 的 pack plan 已经很接近“按 global index 构建本地读写缓存”的中间表示。下一阶段可以不从传统手写 halo 入手，而是先做缓存一致性分类和部分交换计划。这样既能覆盖规则 stencil，也给不规则 stencil 留出统一路径。

### 7.1 当前已经落地的 Phase C 初版

这次实际已经落地的，不再只是方向：

- 运行时新增了 `StencilTypes.h` 中的 `AllRankIndexLayout`、`PeerSlotExchange`、`ExchangePlan`、`DistributedTensorState<T>`，并在 `StencilLayout.h` / `StencilExchange.h` 中补上 `init_all_rank_index_layout()`、`build_exchange_plan_from_layouts()`、`exchange_values_by_slots()` 等 helper。
- Phase C 只允许进入 `rewriteMPIStencil()` 的 loop-lowered site；普通 wrapper 和不在 outer loop 内的 `<->` 不会误进这条路径。
- 当前 site-level guard 是整站回退规则，不做站内 tensor 混合模式：只要有一个条件不满足，整站回到原来的 `Scatterv -> kernel -> Gatherv -> apply_writeback -> optional Bcast`。
- 当前首个 shipping 范围是 1D `dacpp::Vector`、effective `READ/WRITE`、post-shell sibling 可识别的 loop stencil site。
- `init()` 阶段已经会做一次 root scatter seed，把读 cache 固化到 `ctx.dist_*` 本地缓存中，并在 all-rank 元数据上做 unique-writer 校验。
- 代码生成已经有真实的 `DistributedFollowup` run path；正向用例 `mpiDistributedStencil1D` 会生成 `ctx.use_partial_exchange = true`、root-bridge plan 和 helper 内的 `exchange_values_by_slots(...)`。
- 当前已经打通的是“root-centric helper 写回后刷新 distributed read cache”这条 bridge 路径。它验证了 persistent cache + partial refresh 这套模型是通的。

这次还没有完成的部分也需要明确：

- 运行时虽然已经有通用 `exchange_plan` 结构，但 steady-state 的 shell writer -> shell reader peer exchange 还没有完全替代 root bridge。
- root bridge 目前仍是保守实现，writer 侧按 root-authoritative dense cover 建计划，还不是精确 helper-written subset。
- `liuliang1.0` 目前仍因为 effective `READ_WRITE` kernel param 被 Phase C guard 拦住，保持 root-centric fallback；它还不是 partial-exchange 的正向 benchmark。
- 2D / `dacpp::Matrix` / in-place `READ_WRITE` / one-shot MPI wrapper integration 都还在后续范围内。

## 8. 验证结果

构建：

```bash
cd /Volumes/QUQ/working/dacpp
cmake --build build --target translator -j8
```

结果：通过。

重点回归：

```bash
cd /Volumes/QUQ/working/dacpp/clang/tools/translator
bash test_mpi.sh decay1.0 mpiBroadcastRootOnlyCout mpiBroadcastTensor2Array mpiBroadcastUnknownFunction mpiBroadcastAliasRead mpiDistributedStencil1D
```

结果：

```text
6 tests | 6 passed | 0 failed | 0 skipped
```

生成代码结构检查：

- `decay1.0`：`step2.log` 中有 `output local_A sync=root-only`；生成代码有 `MPI_Gatherv(writeback_values_local_A...)` 和 root `apply_writeback_by_globals(...)`，没有为 `local_A` 生成 `MPI_Bcast`。
- `liuliang1.0`：`step2.log` 中有 `partial-exchange disabled: phase-c does not support READ_WRITE kernel params`；它保留 `output new_rho sync=root-centric-followup`，生成代码仍有 `__dacpp_mpi_region_LWR_shell_lwr_stmt_0/1`，但不会启用 Phase C partial path。
- `mpiDistributedStencil1D`：`step2.log` 中有 `partial-exchange enabled (root-bridge)` 和 `output next sync=distributed-followup`；生成代码里有 `ctx.use_partial_exchange = true`、`root_bridge_plan = dacpp::mpi::build_exchange_plan_from_layouts(...)`，以及 helper 内的 `dacpp::mpi::exchange_values_by_slots(...)`。
- `mpiDenseCoverSibling1.0`：`step2.log` 中有 `output updates sync=all-ranks-needed`；二维 sibling loop 目前不 region 化，生成代码保留 `MPI_Bcast`。
- `gradientSum` / `mandel1.0` / `imageAdjustment1.0`：覆盖 `std::cout`、main 外输出函数、无花括号输出 loop body 的 root-only rewrite。

Broadcast analyze 结构断言：

`test_mpi.sh` 支持测试目录下的 `mpi_expect.txt`，在 MPI 翻译 / 编译后、运行前检查 `step2.log` 和生成的 MPI SYCL 文件。当前支持：

- `LOG_CONTAINS:<literal>`
- `LOG_NOT_CONTAINS:<literal>`
- `SYCL_CONTAINS:<literal>`
- `SYCL_NOT_CONTAINS:<literal>`

新增 Broadcast analyze 测试：

- `mpiBroadcastRootOnlyCout`：`std::cout << output_tensor[i]` 被分类为 `root-only`，生成输出 root guard，且不生成 `MPI_Bcast`。
- `mpiBroadcastTensor2Array`：`output_tensor.tensor2Array(...)` 是普通 host read，必须分类为 `all-ranks-needed` 并生成 `MPI_Bcast`。
- `mpiBroadcastUnknownFunction`：shell 输出传入未知函数读，必须分类为 `all-ranks-needed` 并生成 `MPI_Bcast`。
- `mpiBroadcastAliasRead`：shell 输出经引用别名读取，必须保守分类为 `all-ranks-needed` 并生成 `MPI_Bcast`。

完整 MPI 回归：

```bash
cd /Volumes/QUQ/working/dacpp/clang/tools/translator
bash test_mpi.sh
```

结果：

```text
18 tests | 18 passed | 0 failed | 0 skipped
```

覆盖到的关键用例：

- `mpiDenseCoverSibling1.0`
- `decay1.0`
- `liuliang1.0`
- `mandel1.0`
- `imageAdjustment1.0`
- `gradientSum`
- `mpiBroadcastRootOnlyCout`
- `mpiBroadcastTensor2Array`
- `mpiBroadcastUnknownFunction`
- `mpiBroadcastAliasRead`
- `mpiDistributedStencil1D`
- `FOuLa1.0`
- `stencil1.0`
- `waveEquation1.0`
- 以及普通 MPI 主线 case

## 9. 当前状态判断

可以认为当前已经完成：

- MPI stencil 路径接线。
- ctx / init / run codegen 第一版。
- loop stencil site 安全分流。
- 普通 MPI 和 stencil MPI 写回值 gather 修复。
- Phase 1 保守优化：通信 metadata hoist 到 `init()`，`run()` 复用 layout 和 buffer。
- 输出同步分类 v1：`RootOnly`、`AllRanksNeeded`、`RootCentricFollowup`、`DistributedFollowup`。
- 输出同步分类已基于当前 `<->` 的实际实参名分析后续使用，并支持 `.print()` / `std::cout` root-only observable 传播。
- Broadcast analyze 结构断言测试：已覆盖 root-only `std::cout`、`tensor2Array` 普通 host read、未知函数读、引用别名读。
- MPI 输出语句 root-only rewrite：删除非 root stdout 全局重定向，改为 guard `.print()` 和 `std::cout` 输出语句。
- `liuliang1.0` 一维 post-shell loop 的 MPI root-centric region helper 生成和替换。
- Phase C C0 guard 和 loop-only eligibility analysis 已落地。
- Phase C 运行时基础设施已落地：all-rank layout、unique writer validate、exchange plan、persistent distributed tensor state。
- Phase C 初版 distributed run path 已落地，当前支持 1D `dacpp::Vector` loop stencil site 的 persistent cache + root-bridge partial refresh。
- `DistributedFollowup` 已经是实 codegen path，不再只是分类占位。
- 新增 `mpiDistributedStencil1D` 正向回归，验证 partial-exchange enabled (root-bridge) 的生成和运行结果。
- `translator` 构建验证。
- `test_mpi.sh` 完整回归验证，当前是 `18 tests | 18 passed | 0 failed | 0 skipped`。
- 已记录副本一致性、post-shell region v1、缓存一致性 + 部分交换的后续方向。

当前不能宣称的是：

- 已支持所有 loop 内 `<->` 的 hoist 优化。
- 已支持循环体内部临时 view 的跨迭代 ctx 复用。
- 已完成完整别名 / 函数调用 / 复杂表达式数据流 / 完整 observable root-only 证明。
- 已修复所有非 root stale 副本风险。
- 已把所有可能产生 stdout/stderr 的 C/C++ I/O API 都纳入输出语句 root-only rewrite；当前只覆盖 `.print()` 和 `std::cout`。
- `liuliang1.0` 已经切到 partial-exchange steady-state；它当前仍因 effective `READ_WRITE` kernel param 走 fallback。
- 已支持 Matrix / 二维 / 复杂语句的 post-shell region。
- 已把 shell writer -> shell reader 的 steady-state peer exchange 全部打通；当前正向路径主要还是 root-helper bridge。
- 已实现精确 helper-written subset 的 bridge payload；当前 root bridge 还是保守 dense-root-authoritative 方案。
- 已把 Phase C 接到 one-shot MPI wrapper 路径。
- 已实现 generalized halo / cache exchange。

## 10. 后续路线

Phase A：一致性分类 v1 已完成，后续增强

- 继续增强复杂表达式数据流和 root-only observable 证明。
- 继续增加 slice / subview / 复杂表达式的保守降级测试；引用别名和未知函数读已有结构断言覆盖。
- 为 `decay1.0` 增加能暴露非 root stale 副本的测试。

Phase B：MPI root-centric post-shell region v1 已完成，后续增强

- 扩展到 `dacpp::Matrix` / 二维 loop。
- 支持更复杂的一维赋值语句和多个 tensor 的读写组合。
- 将 root-centric helper 的 copy in/out 进一步优化，减少 host vector 临时量。

Phase C：cache consistency / partial exchange，当前已做一半

已完成：

- 基于 pack globals / writeback globals 引入 all-rank layout、owner / reader / writer 所需的运行时承载结构。
- 在 `dpcppLib/include/mpi/StencilTypes.h` 增加 `DistributedTensorState<T>`、`ExchangePlan`；在 `StencilLayout.h` 增加 unique-writer validate。
- 把 `DistributedFollowup` 接成真实的 loop-lowered MPI stencil codegen path。
- 支持 1D `dacpp::Vector` site 的 persistent local cache seed。
- 支持 root-centric helper 写回后的 root -> distributed cache bridge，正向回归是 `mpiDistributedStencil1D`。
- 保持 whole-site fallback 规则，所有不满足条件的 site 都回退到旧 root-centric 通信路径。

下一步：

- 真正把 steady-state 的 shell writer -> shell reader peer exchange 接到 `exchange_plan`，减少对 root bridge 的依赖。
- 把 root bridge 从保守 dense cover 收紧到 helper-written subset。
- 继续验证双 buffer 1D stencil，并决定是否为 `liuliang1.0` 这类 effective `READ_WRITE` case 放宽分析或改写形态。
- 对规则 stencil 自动退化为传统 halo 形态。
- 对不规则访问保留 generalized index exchange。

Phase D：distributed follow-up region / generalized halo

- 将 post-shell region 和 shell pack/writeback 信息关联。
- 尽量在 rank-local buffer 上执行后续 region。
- 引入必要边界交换。
- 在最终 observable 点才 gather / broadcast。

## 11. 一句话总结

当前 MPI stencil 路径已经完成 Phase 1、输出一致性 / root-centric region v1，以及 Phase C 初版：metadata 缓存、buffer 复用、输出同步分类、`liuliang1.0` root-centric helper、`mpiDistributedStencil1D` 的 partial-exchange root-bridge 正向路径都已落地，完整 MPI 回归是 `18 / 18` 通过。下一步的核心不再是“要不要做缓存一致性”，而是把已经铺好的 distributed cache / exchange runtime 真正推进到通用 shell writer -> reader steady-state 交换，并继续扩到 2D、`READ_WRITE` 和 wrapper 路径。
