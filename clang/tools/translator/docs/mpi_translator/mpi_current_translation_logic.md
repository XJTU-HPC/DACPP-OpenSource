# DACPP Translator MPI 交接说明

这份文档说明 `clang/tools/translator` 中 MPI 翻译路径的入口、分支、代码生成结构、运行时状态、测试范围与后续工作接口。阅读它可以直接建立对当前主线的整体认知，不需要再沿着历史演进记录倒推。

## 1. 总览

MPI 翻译路径由两条实现线组成：

- 通用 wrapper 路径
- buffer region 路径

两条路径都建立在 item-space / pack-map 的 planner 语义上，差别在于执行生命周期：

- wrapper 路径按一次 `<->` 调用完成 scatter、局部执行和 gather
- region 路径把时间推进类程序展开为 `init -> submit -> halo -> sibling helpers -> sync`

## 2. 入口与路由

入口文件是 [translator.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/translator.cpp)。

MPI 翻译阶段会：

1. 识别 `--mpi`
2. 注入 `<mpi.h>`、`<cstdio>`、`"MPIPlanner.h"`
3. 调用 `analyzeBufferRegionPlan()`
4. 根据 `hasBufferRegionPlan()` 选择路径
5. 最后执行 `rewriteMain()`

路由规则是：

- `hasBufferRegionPlan()` 为真时调用 `rewriteMPI_Region()`
- 否则调用 `rewriteMPI()`

[Rewriter_MPI.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/Rewriter_MPI.cpp) 中：

- `rewriteMPI()` 对应通用 wrapper 路径
- `rewriteMPI_Stencil()` 承担 region 主体实现
- `rewriteMPI_Region()` 直接转发到 `rewriteMPI_Stencil()`

## 3. 通用 Wrapper 路径

wrapper 路径的目标是按 rank 的真实访问集合分发与回收数据，而不是按 tensor 物理空间做规则切块。

核心流程是：

1. 从 shell / split / binding 构造 `AccessPattern`
2. 合并出统一 `binding_split_sizes`
3. 计算全局 `total_items`
4. 用 `get_rank_item_range(...)` 给每个 rank 分配 item range
5. 根据参数读写模式构造 pack-map
6. root 只打包该 rank 真正访问的元素
7. rank 在局部 packed 数组上通过 `View1D` / `View2D` 执行 SYCL kernel
8. 写参数只按真实写回位置 gather 回 root
9. 对需要所有 rank 同步的 tensor 执行 broadcast

该路径的主要代码位于：

- [Rewriter_MPI_Wrapper_Codegen.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Wrapper_Codegen.cpp)

它适用于：

- 无 buffer region 的 MPI 样例
- region 条件不满足的程序

## 4. Region 路径

region 路径面向时间推进类程序和带 sibling `for` 的 buffer region。

生成代码的骨架为：

```cpp
__dacpp_mpi_ctx_<shell>_<calc> ctx;
__dacpp_mpi_init_<shell>_<calc>(ctx, tensors...);
for (...) {
    __dacpp_mpi_submit_<shell>_<calc>(ctx);
    __dacpp_mpi_halo_<shell>_<calc>(ctx);
    __dacpp_mpi_submit_region_<shell>_<calc>_stmt_*(ctx);
}
__dacpp_mpi_sync_<shell>_<calc>(ctx, tensors...);
```

这条路径把程序拆成五个阶段：

- `ctx`
- `init`
- `submit`
- `halo`
- `sync`

sibling `for` 以 `__dacpp_mpi_submit_region_*` helper 形式插入循环体中。

region 主代码由 [Rewriter_MPI_Region_Codegen.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Region_Codegen.cpp) 生成。

### 4.1 ctx

`ctx` 是 region 生命周期的权威状态容器，包含：

- MPI 基本信息：`mpi_rank`、`mpi_size`
- 设备队列：`sycl::queue q`
- item-space 信息：`total_items`、`local_item_count`、`binding_split_sizes`
- halo 标志：`has_halo`
- 每个参数的：
  - `pattern_*`
  - `pack_*`
  - `slots_*`
  - `local_*`
  - `buf_*`
  - `slots_buf_*`
  - 视图辅助字段
  - `halo_*`

### 4.2 init

`init` 阶段负责：

- 初始化 `AccessPattern`
- 合并绑定维度的 split size
- 计算本 rank 的 `item_range`
- 构造 `pack_*` 与 `slots_*`
- 完成一次性 scatter
- 构造 `local_*` 和 `buf_*`
- 为每个参数执行 `computeParamHalo(...)`

init / sync 的传输策略由 [Rewriter_MPI_Region_Policy.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Region_Policy.cpp) 决定：

- `analyzeMPIRegionTransferPolicy(...)`
- `inferMPIRegionStorageModes(...)`

### 4.3 submit

`submit` 只负责提交一次本地 SYCL kernel，不做主机侧同步。它在局部 packed 数组上为每个参数构造 `View1D` / `View2D`，再调用 `calc_mpi_local(...)`。

### 4.4 halo

`halo` 阶段负责 kernel 后的邻居同步：

1. `ctx.q.wait()`
2. 对写参数执行 D2H，把 `buf_*` 同步回 `local_*`
3. 调用 `exchangeHalo(...)`
4. 把更新后的 `local_*` 写回 `buf_*`

该阶段依赖 [MPIPlanner.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/MPIPlanner.h) 中的 halo runtime：

- `HaloRegion`
- `ParamHalo`
- `computeParamHalo(...)`
- `exchangeHalo(...)`

halo 的语义是：

- `recv halo = my_inputs ∩ neighbor_outputs`
- `send halo = my_outputs ∩ neighbor_inputs`

邻居集合按一维 item-space 连续分区计算，只覆盖 `rank - 1` 与 `rank + 1`。

### 4.5 sync

`sync` 在外层循环结束后执行：

- 等待设备队列完成
- 对需要写回的参数执行 D2H
- 收集 `writeback_globals` 与写回值
- `MPI_Gather` / `MPI_Gatherv`
- root 执行 `apply_writeback_by_globals(...)`
- 按 tensor 需求执行 `MPI_Bcast`

## 5. Sibling Loops

region 内的 sibling `for` 通过 [Rewriter_MPI_Region_Sibling.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Region_Sibling.cpp) 生成 helper。

每个 helper 的流程是：

1. `ctx.q.wait()`
2. 从 `ctx.buf_*` 回收 `ctx.local_*`
3. 基于 `ctx.pack_*` 和全局维度重建 dense temporary
4. 通过 `MPI_Allreduce` 合并 dense 数值数组与 present 标记数组
5. 创建 `DenseVectorView` / `DenseMatrixView`
6. 按原 sibling `for` 的源码执行 host-side 更新
7. 把 dense temporary 同步回 `ctx.local_*`
8. 再同步回 `ctx.buf_*`

这条路径解决的是 region 状态闭环问题。因为 region 路径中的真实状态位于：

- `ctx.local_*`
- `ctx.buf_*`
- `ctx.pack_*`

如果 sibling loop 直接操作原始 host tensor，就无法自然并入 `submit -> halo -> sync` 生命周期。

### 5.1 Dense Bridge 的语义边界

helper 中的 dense bridge 使用：

- 数值数组上的 `MPI_Allreduce(..., MPI_SUM)`
- present 计数数组上的 `MPI_Allreduce(..., MPI_SUM)`
- `present > 1` 的位置按出现次数平均

这种规则适用于重复位置上副本值一致的场景，目标是优先保证翻译语义闭环和回归通过。更进一步的设备化 sibling lowering、只同步真实写入位置的稀疏 bridge、按邻居增量同步的数据路径属于后续优化方向。

## 6. Runtime / Planner

[MPIPlanner.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/MPIPlanner.h) 提供 MPI 路径公用的 planner 与运行时支撑。

### 6.1 Item-space 与 pack-map

核心结构与函数包括：

- `ItemRange`
- `AccessPattern`
- `PackMap`
- `get_rank_item_range(...)`
- `collect_positions_for_item(...)`
- `build_input_pack_map(...)`
- `build_output_pack_map(...)`
- `build_rw_pack_map(...)`
- `build_item_slots(...)`
- `build_writeback_values(...)`
- `apply_writeback_by_globals(...)`

### 6.2 Packed Local Views

提交 kernel 时使用：

- `View1D`
- `View2D`

它们把 packed local array 映射成 calc 需要的局部访问视图。

### 6.3 Dense Host Views

sibling helper 使用：

- `DenseElementRef`
- `DenseVectorView`
- `DenseMatrixView`

这组视图负责在 dense temporary 上保留原 sibling loop 的 `tensor[idx]` / `tensor[i][j]` 访问语义。

## 7. 代码结构

### 7.1 公开入口

- [translator.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/translator.cpp)
- [Rewriter_MPI.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/Rewriter_MPI.cpp)
- [Rewriter_MPI_Common.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/include/Rewriter_MPI_Common.h)

### 7.2 已接入构建的 MPI 模块

- [Rewriter_MPI_Analysis.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Analysis.cpp)
- [Rewriter_MPI_Types.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Types.cpp)
- [Rewriter_MPI_Pattern.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Pattern.cpp)
- [Rewriter_MPI_Wrapper_Codegen.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Wrapper_Codegen.cpp)
- [Rewriter_MPI_Region_Policy.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Region_Policy.cpp)
- [Rewriter_MPI_Region_Codegen.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Region_Codegen.cpp)
- [Rewriter_MPI_Region_Sibling.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Region_Sibling.cpp)

### 7.3 参考源码

以下文件保留在工作区中供对照阅读，但不属于当前主线构建输入：

- [Rewriter_MPI_Wrapper.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Wrapper.cpp)
- [Rewriter_MPI_Region.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Region.cpp)

## 8. 生成代码核对结果

对以下样例执行 `dacpp ... --mode=buffer --mpi` 后，生成文件中都可以看到完整 region 生命周期：

- `stencil1.0`
- `waveEquation1.0`
- `MDP1.0`

核对项包括：

- `__dacpp_mpi_ctx_*`
- `__dacpp_mpi_init_*`
- `__dacpp_mpi_submit_*`
- `__dacpp_mpi_halo_*`
- `__dacpp_mpi_sync_*`
- `__dacpp_mpi_submit_region_*`

主函数中的调用顺序也与 region 设计一致：

- 循环前 `ctx` 与 `init`
- 循环内 `submit -> halo -> sibling helpers`
- 循环后 `sync`

## 9. 回归范围

[test_mpi.sh](/Volumes/QUQ/working/dacpp/clang/tools/translator/test_mpi.sh) 运行 12 个 MPI 样例：

- `matMul1.0`
- `FOuLa1.0`
- `decay1.0`
- `DFT1.0`
- `liuliang1.0`
- `MDP1.0`
- `mandel1.0`
- `imageAdjustment1.0`
- `vectorAddCombo`
- `gradientSum`
- `stencil1.0`
- `waveEquation1.0`

脚本行为是：

1. 为每个样例复制源文件到 `/Volumes/QUQ/working/mpi_tmp/<test>`
2. 生成 baseline：`dacpp ... --mode=buffer`
3. 编译并执行 baseline
4. 生成 MPI 版本：`dacpp ... --mode=buffer --mpi`
5. 编译并执行 `mpirun -np 2`
6. 清洗 AdaptiveCpp 运行时警告并对比输出

脚本会过滤：

- `*.mpi.dac.cpp`
- `*.retranslated.dac.cpp`
- `*.large_dac.cpp`

以避免误选源输入。

## 10. 推荐的验证顺序

```bash
cd /Volumes/QUQ/working/dacpp
cmake --build build --target translator -j8

cd /Volumes/QUQ/working/dacpp/clang/tools/translator
bash test_local.sh
INCLUDE_EXTENDED_LOCAL_TESTS=1 bash test_local.sh
bash test_mpi.sh
```

## 11. 后续工作

主线后续工作集中在以下方向：

- sibling loop 设备化，直接在 `ctx.buf_*` 上执行
- dense bridge 稀疏化，只同步真实读写位置
- 按 halo / neighbor 信息收紧 sibling 需要的数据同步
- 扩展 region 计划接受的 sibling 语法与控制流边界
- 进一步拆分 region codegen 模块，细化 `ctx / init / submit / halo / sync / sibling`

## 12. 一句话理解

这套 MPI 实现把通用程序维持在稳定的 wrapper 主线，把时间推进类 region 程序推进到一次初始化、循环内 halo 同步、结束统一回写的执行模型，并用 host-side dense bridge 把 sibling loop 并入同一条 region 生命周期。
