# MPI Region 实施说明

这份文档说明 `clang/tools/translator` 中 `--mode=buffer --mpi` 的 region 实现、运行时数据结构、代码生成顺序与回归范围。它对应的落地点是时间推进类程序、stencil 类程序，以及带 sibling `for` 的 buffer region。

## 目标

MPI region 路径采用一次初始化、循环内局部推进、结束时统一同步的执行模型：

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

这个模型承担三件事：

- `init` 完成一次性 scatter，把 rank 需要访问的局部数据和 halo 数据放入 `ctx.local_*`
- `submit + halo + sibling helpers` 在循环内维持 rank 本地状态
- `sync` 在循环结束时执行 gather / writeback / broadcast

## 路由规则

MPI 翻译入口位于 [translator.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/translator.cpp)。

路由条件是：

- `dacppFile->hasBufferRegionPlan()` 为真时，调用 `rewriteMPI_Region()`
- 否则调用 `rewriteMPI()`

`rewriteMPI_Region()` 由 [Rewriter_MPI.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/Rewriter_MPI.cpp) 中的 `rewriteMPI_Stencil()` 承担主体实现。该函数负责：

- 生成 `ctx / init / submit / halo / sync`
- 在外层循环前插入 `ctx` 声明和 `init`
- 把 `<->` 替换为 `submit + halo`
- 把每个 sibling `for` 替换为 `__dacpp_mpi_submit_region_*`
- 在外层循环后插入 `sync`

## 运行时数据模型

region 代码生成的上下文结构体包含：

- `mpi_rank` / `mpi_size`
- `sycl::queue q`
- `total_items` / `local_item_count`
- `binding_split_sizes`
- `has_halo`
- 每个参数对应的：
  - `pattern_<name>`
  - `pack_<name>`
  - `slots_<name>`
  - `local_<name>`
  - `buf_<name>`
  - `slots_buf_<name>`
  - 视图形状辅助字段
  - `halo_<name>`

这些字段把 MPI planner、局部 packed array、SYCL buffer 与 halo runtime 串成一个完整状态容器。

## 数据分发与同步

### Item-space 与 pack-map

MPI 分发不按 tensor 物理区间切块，而是按 item-space 分配 rank 的逻辑工作集：

1. 从 shell / split / binding 构造 `AccessPattern`
2. 合并出统一的 `binding_split_sizes`
3. 计算 `total_items`
4. 用 `get_rank_item_range(total_items, mpi_rank, mpi_size)` 给每个 rank 分配连续 item range
5. 按访问模式构造 pack-map

pack-map 选择规则：

- `READ` 使用 `build_input_pack_map`
- `WRITE` 使用 `build_output_pack_map`
- `READ_WRITE` 使用 `build_rw_pack_map`

### Init

`init` 负责：

- 建立每个参数的 `AccessPattern`
- 推导统一的 `binding_split_sizes`
- 计算本 rank 的 `item_range`
- 构造 `pack_*` 与 `slots_*`
- 在需要时执行 root 侧打包和 `MPI_Scatterv`
- 构造 `local_*` 与 `buf_*`
- 调用 `computeParamHalo(...)` 计算每个参数的 halo 信息

`init` 的 scatter / sync 决策来自 [Rewriter_MPI_Region_Policy.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Region_Policy.cpp)：

- `analyzeMPIRegionTransferPolicy(...)` 决定 init 是否需要 scatter、sync 是否需要 gather
- `inferMPIRegionStorageModes(...)` 把 sibling loop 的读写行为并入参数存储模式

### Submit

`submit` 只负责 SYCL kernel 提交。它在 `ctx.local_item_count > 0` 时：

- 为每个参数获取 `buf_*` accessor 和 `slots_buf_*` accessor
- 用 `View1D` / `View2D` 在 packed 局部数组上映射视图
- 调用 `calc_mpi_local(...)`

### Halo

`halo` 的执行顺序是：

1. `ctx.q.wait()`
2. 对写参数执行 D2H，把 `buf_*` 内容同步回 `local_*`
3. 调用 `exchangeHalo(...)`
4. 把更新后的 `local_*` 回写到 `buf_*`

halo runtime 位于 [MPIPlanner.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/MPIPlanner.h)，核心结构与函数包括：

- `HaloRegion`
- `ParamHalo`
- `computeParamHalo(...)`
- `exchangeHalo(...)`

`computeParamHalo(...)` 的语义是：

- `recv halo = my_inputs ∩ neighbor_outputs`
- `send halo = my_outputs ∩ neighbor_inputs`

邻居集合按一维连续 item-space 分区计算，只包含 `rank - 1` 和 `rank + 1`。

### Sync

`sync` 负责：

- 等待设备队列完成
- 对需要写回的参数执行 D2H
- 构造 `writeback_globals` 和写回值
- `MPI_Gather` / `MPI_Gatherv` 回 root
- root 执行 `apply_writeback_by_globals(...)`
- 对需要全 rank 可见结果的 tensor 执行 `MPI_Bcast`

## Sibling Loops

region 内的 sibling `for` 由 [Rewriter_MPI_Region_Sibling.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Region_Sibling.cpp) 生成 helper。

helper 的工作方式是：

1. `ctx.q.wait()`
2. 把 `buf_*` 同步回 `local_*`
3. 为参与 sibling 写入的参数构造“全局索引可覆盖”的 `pack_*` 与 `global_to_local_*` 查表
4. 为 sibling 中读参数构造只读 dense fallback，为写参数构造 `dirty` 位图与 `dense_shadow`
5. 直接在设备侧提交 sibling kernel，访问视图使用 `PackedVectorView` / `PackedMatrixView`
6. 从 `dirty` 位图提取真实写入的全局索引集合，跨 rank 合并去重
7. 仅对真实写入索引执行稀疏 `MPI_Allreduce`，把结果写回 `ctx.local_*`
8. 将同步后的 `ctx.local_*` 回写 `ctx.buf_*`

这条路径依赖 [MPIPlanner.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/MPIPlanner.h) 中的：

- `PackedElementRef`
- `PackedVectorView`
- `PackedMatrixView`

其中 `PackedElementRef` 统一了三类语义：

- 命中本地 slot 时直接读写 `ctx.buf_*` 对应数据
- 未命中本地 slot 时读 fallback dense 值
- 写路径同时更新 `dense_shadow` 并打 `dirty` 标记

稀疏同步阶段使用：

- 索引集合上的 `MPI_Allgatherv`（收集各 rank 真实写入位置）
- 值数组上的 `MPI_Allreduce(..., MPI_SUM)`
- present 计数上的 `MPI_Allreduce(..., MPI_SUM)`
- `present > 1` 的位置按出现次数求平均

## 代码模块

### 入口与公共接口

- [translator.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/translator.cpp)
- [Rewriter_MPI.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/Rewriter_MPI.cpp)
- [Rewriter_MPI_Common.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/include/Rewriter_MPI_Common.h)

### Wrapper 路径

- [Rewriter_MPI_Wrapper_Codegen.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Wrapper_Codegen.cpp)

### Region 路径

- [Rewriter_MPI_Region_Policy.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Region_Policy.cpp)
- [Rewriter_MPI_Region_Codegen.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Region_Codegen.cpp)
- [Rewriter_MPI_Region_Sibling.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Region_Sibling.cpp)

### Planner / Runtime

- [MPIPlanner.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/MPIPlanner.h)

## 验证

### 生成代码核对

对下列样例执行 `dacpp ... --mode=buffer --mpi` 后，生成文件中都可以看到 `ctx / init / submit / halo / sibling helper / sync` 的完整结构：

- `stencil1.0`
- `waveEquation1.0`
- `MDP1.0`

核对点包括：

- 函数定义存在
- 主函数调用顺序与 region 生命周期一致
- sibling `for` 被替换为 `__dacpp_mpi_submit_region_*`
- `sync` 放在外层循环之后

### MPI 回归脚本

[test_mpi.sh](/Volumes/QUQ/working/dacpp/clang/tools/translator/test_mpi.sh) 覆盖 12 个样例：

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

脚本流程是：

1. 复制源样例到 `/Volumes/QUQ/working/mpi_tmp/<test>`
2. 生成单机 baseline：`dacpp ... --mode=buffer`
3. 编译并运行 baseline
4. 生成 MPI 版本：`dacpp ... --mode=buffer --mpi`
5. 编译并运行 `mpirun -np 2`
6. 清洗 AdaptiveCpp 警告并比对输出

## 当前实现要点

- region codegen 使用 `ctx / init / submit / halo / sync` 五段式生成与拼接
- sibling helper 支持设备侧执行与 dirty 稀疏同步
- sibling 写参数使用全局可覆盖 pack，保证全局索引写入语义稳定
- `sync` 阶段沿用 `writeback_globals` 进行最终 gather/writeback/broadcast
