# MPI Stencil 实现详解

本文档详细描述 DACPP translator 的 MPI region 路径如何把用户级 stencil 程序翻译为可分布式执行的 SYCL+MPI 代码。以 `tests/stencil1.0` 为贯穿示例。

## 1. 用户级代码结构

一个典型的 stencil 程序由三部分组成：**shell**（声明数据关联模式）、**calc**（定义计算逻辑）、**主循环**（迭代执行）。

```cpp
// shell: 声明访问模式
shell dacpp::list stencilShell(dacpp::Matrix<double>& matIn READ_WRITE,
                               dacpp::Matrix<double>& matOut READ_WRITE) {
    dacpp::split sp1(3, 1), sp2(3, 1);   // 3×1 滑动窗口，步长 1
    dacpp::index idx1, idx2;              // 单点索引
    binding(sp1, idx1);                   // sp1 和 idx1 绑定到同一维度
    binding(sp2, idx2);                   // sp2 和 idx2 绑定到同一维度
    dacpp::list dataList{matIn[sp1][sp2], matOut[idx1][idx2]};
    return dataList;
}

// calc: 定义单个分区的计算
calc void stencil(dacpp::Matrix<double>& mat, double* out) {
    out[0] = mat[1][1] + alpha * delta_t * (
        ((mat[2][1] - 2*mat[1][1] + mat[0][1]) / (dx*dx)) +
        ((mat[1][2] - 2*mat[1][1] + mat[1][0]) / (dy*dy)));
}

// 主循环
for (int t = 0; t < TIME_STEPS; t++) {
    stencilShell(matIn, matOut) <-> stencil;
    // sibling loops: 回写 + 边界处理
    for (...) matIn[i][j] = matOut[i-1][j-1];
    // ... 边界条件 ...
}
```

翻译器把 `<->` 识别为 buffer region 规划点，把循环体中的 sibling `for` 识别为辅助逻辑，然后生成 MPI 分布式代码。

## 2. 生成代码的整体结构

MPI region 路径生成的代码骨架为：

```cpp
struct __dacpp_mpi_ctx_stencilShell_stencil ctx;
__dacpp_mpi_init_stencilShell_stencil(ctx, matIn, matOut);
for (int t = 0; t < TIME_STEPS; t++) {
    __dacpp_mpi_submit_stencilShell_stencil(ctx);
    __dacpp_mpi_halo_stencilShell_stencil(ctx);
    __dacpp_mpi_submit_region_stencilShell_stencil_stmt_0(ctx);
}
__dacpp_mpi_sync_stencilShell_stencil(ctx, matIn, matOut);
```

五个阶段：**ctx** → **init** → **submit** → **halo** → **sync**，加上 sibling helper。

## 3. Item-Space 与工作项划分

### 3.1 什么是 item

translator 不按 tensor 物理空间切分，而是按 **item-space** 分配工作。每个 item 代表一个独立的计算任务（对应 calc 的一次调用）。

item 的数量由 shell 的 split/bind 决定：

- 每个 binding 组对应一个 split 维度
- `bind_split_sizes[B]` = binding 组 B 的 item 数

对于上面的 stencil 例子：

```
binding(sp1, idx1)  →  bind_id=0
binding(sp2, idx2)  →  bind_id=1

matIn[sp1(3,1)][sp2(3,1)]:
  bind_id=0, dim=0: op.size=3, op.stride=1, split_size=(NX-3)/1+1=6
  bind_id=1, dim=1: op.size=3, op.stride=1, split_size=(NY-3)/1+1=6

matOut[idx1][idx2]:
  bind_id=0, dim=0: is_index=true, split_size=NX-2=6
  bind_id=1, dim=1: is_index=true, split_size=NY-2=6

统一后 bind_split_sizes = [6, 6]
total_items = 6 × 6 = 36
```

### 3.2 item 到 rank 的分配

`get_rank_item_range()` 把 total_items 按连续区间切给各 rank：

```
P 个 rank，N 个 items:
  base = N / P
  rem  = N % P
  rank r 分配到 items [r*base + min(r,rem), r*base + min(r,rem) + base + (r<rem ? 1 : 0))
```

对于 36 items、4 ranks：每 rank 分到 9 个连续 item。

```
rank 0: items [0, 9)
rank 1: items [9, 18)
rank 2: items [18, 27)
rank 3: items [27, 36)
```

### 3.3 item 到全局位置的映射

每个 item 通过 `collect_positions_for_item()` 映射到 tensor 中的具体位置。

首先 `decode_item_id()` 把线性 item_id 解码为各 bind 维度的索引：

```
bind_split_sizes = [6, 6]
item_id = bind[0] * 6 + bind[1]

item 0  → bind[0]=0, bind[1]=0
item 1  → bind[0]=0, bind[1]=1
...
item 5  → bind[0]=0, bind[1]=5
item 6  → bind[0]=1, bind[1]=0
...
```

然后根据每个 op 的类型计算 base_pos：

- **IndexSplit**（`is_index_op=true`）：`base_pos[dim] = bind_index + offset`
- **RegularSlice**（`is_index_op=false`）：`base_pos[dim] = bind_index * stride + offset`

最后按 partition_shape 展开所有位置并线性化。

对于 `matIn[sp1(3,1)][sp2(3,1)]`，item 0 的映射：

```
bind[0]=0, bind[1]=0
dim 0 (sp1): base_pos[0] = 0 * 1 + 0 = 0, partition_shape[0] = 3
dim 1 (sp2): base_pos[1] = 0 * 1 + 0 = 0, partition_shape[1] = 3

展开 partition:
  (0,0) (0,1) (0,2)
  (1,0) (1,1) (1,2)
  (2,0) (2,1) (2,2)

线性化（row-major, dimLength=[8,8]）:
  globals = [0, 1, 2, 8, 9, 10, 16, 17, 18]
```

这就是 item 0 对应的 9 个全局位置（一个 3×3 块）。

## 4. 数据分配：Pack-Map 机制

### 4.1 为什么需要 pack

每个 rank 只需要 tensor 的一部分数据（自己负责的 items 访问到的位置）。如果给每个 rank 都分配完整的 tensor，会浪费大量内存。Pack-map 机制只打包 rank 真正需要的数据。

### 4.2 Pack-Map 的构建

以 rank 0（items [0,9)）为例：

```
build_input_pack_map(item_range, pattern):
  遍历 items 0..8:
    对每个 item 调用 collect_positions_for_item()
    收集所有全局位置
  去重并排序 → globals
  构建 g2l 映射：globals[i] → i
  识别连续段 → segments
```

对于 `matIn`，rank 0 的 9 个 items 涉及的全局位置：

```
items 0-5 (bind[0]=0): 行 0-2, 列 0-7 → 3×8=24 个位置
items 6-8 (bind[0]=1): 行 1-3, 列 0-2 → 3×3=9 个位置
去重后共 27 个全局位置
```

这些位置被打包到一个连续的 `local_matIn` 数组中，`g2l` 记录了全局索引到本地数组下标的映射。

### 4.3 Slot 数组

`build_item_slots()` 为每个 item 的每个 partition 元素记录其在 local 数组中的位置：

```
item 0 → partition 位置 [0,1,2,8,9,10,16,17,18] → slots [0,1,2,3,4,5,6,7,8]
item 1 → partition 位置 [1,2,3,9,10,11,17,18,19] → slots [1,2,?,4,5,?,7,8,?]
                                                  （? 表示需在 pack-map 中查找）
```

submit 阶段用 slots 构造 View，使 calc 中的 `mat[i][j]` 访问到正确的本地数据。

### 4.4 数据 Scatter

init 阶段由 root（rank 0）完成数据分发：

```
rank 0:
  将 tensor 转为全局数组 global_data
  对每个 rank r:
    计算 r 的 item_range
    构建 r 的 pack_map
    从 global_data 中提取 r 需要的值: pack_values_by_globals(global_data, pack_map.globals)
    打包到 sendbuf

MPI_Scatterv(sendbuf, sendcounts, displs, ...)
  → 每个 rank 接收自己的 local_data
```

关键点：root 不是把整个 tensor 发给每个 rank，而是只发送该 rank 的 pack_map 覆盖的数据。

## 5. 计算提交：Submit 阶段

生成的 submit 代码将每个 item 映射为一个 SYCL work-item：

```cpp
void __dacpp_mpi_submit_stencilShell_stencil(ctx_type& ctx) {
    if (ctx.local_item_count <= 0) return;
    ctx.q.submit([&](sycl::handler& h) {
        auto acc_matIn = ctx.buf_matIn->get_access<read_write>(h);
        auto slots_acc_matIn = ctx.slots_buf_matIn->get_access<read>(h);

        h.parallel_for(range<1>(ctx.local_item_count), [=](id<1> idx) {
            const int item_linear = idx[0];
            auto* data = acc_matIn.get_multi_ptr<...>().get();
            auto* slots = slots_acc_matIn.get_multi_ptr<...>().get();

            // 构造 View：用 slots 把逻辑索引映射到 packed 数组的物理位置
            View2D<double> view_matIn{data, slots,
                                      item_linear * partition_size, cols};

            // 调用用户定义的 calc 函数
            stencil_mpi_local(view_matIn, view_matOut);
        });
    });
}
```

**View 的作用**：calc 中的 `mat[i][j]` 通过 View 转换为：

```
data[slots[offset + i * cols + j]]
```

其中 `offset = item_linear * partition_size` 是当前 item 在 slots 数组中的起始偏移。这样 calc 代码不需要知道数据在本地数组中的实际位置。

## 6. Halo 交换

### 6.1 为什么需要 halo

每个 rank 只持有自己负责的 items 的数据。但 stencil 计算中，边界 item 的 partition 会延伸到相邻 rank 的 item 区域。halo 交换确保每个 rank 持有计算所需的全部数据。

### 6.2 Halo 规划：computeParamHalo()

初始化阶段调用 `computeParamHalo()` 为每个参数计算 halo 信息。

**第一步：计算本 rank 的全局位置集合**

```
对 my_range 内的每个 item:
  调用 collect_positions_for_item()
  收集到 my_input_globals 和 my_output_globals
排序去重
```

**第二步：用 item-reach 过滤候选 rank**

```
reach = computeItemReach(pattern)     // 分析 stencil 的 item-space 触及距离
win   = computeRankWindow(reach, ...) // 转换为 rank 窗口
```

典型 3×3 stencil：reach=2，窗口=[rank-1, rank+1]。

**第三步：对窗口内的每个候选 rank 计算交集**

```
对候选 rank nb_rank:
  计算 nb_range 和 nb 的全局位置集合
  recv_globals = my_input_globals ∩ nb_output_globals  // 我需要的、邻居产生的
  send_globals = my_output_globals ∩ nb_input_globals  // 我产生的、邻居需要的
  如果交集非空 → 加入 halo.regions
```

**第四步：slot 映射**

```
对每个 recv_global:
  local_slot = ctx_pack.g2l[recv_global]  // 映射到本地数组下标
对每个 send_global:
  local_slot = ctx_pack.g2l[send_global]
```

### 6.3 Halo 执行：exchangeHalo()

每次迭代的 halo 阶段执行实际的数据交换：

```cpp
void __dacpp_mpi_halo_stencilShell_stencil(ctx_type& ctx) {
    ctx.q.wait();
    if (!ctx.has_halo) return;

    // 对每个非 READ 参数:
    // 1. D2H: 设备缓冲区 → 本地数组
    sycl::host_accessor ha_matIn(*ctx.buf_matIn, read_only);
    for (i = 0; i < ctx.local_matIn.size(); i++)
        ctx.local_matIn[i] = ha_matIn[i];

    // 2. MPI halo 交换
    MPI_Datatype mpi_dt = MPI_DOUBLE;
    dacpp::mpi::exchangeHalo(ctx.local_matIn, ctx.halo_matIn, &mpi_dt);

    // 3. H2D: 本地数组 → 设备缓冲区
    sycl::host_accessor ha_w_matIn(*ctx.buf_matIn, write_only);
    for (i = 0; i < ctx.local_matIn.size(); i++)
        ha_w_matIn[i] = ctx.local_matIn[i];
}
```

`exchangeHalo()` 的内部逻辑（对每个 halo region）：

```
1. 打包 send buffer: 从 local_data 中按 send_local_slots 提取
2. MPI_Isend → 非阻塞发送给 neighbor_rank
3. MPI_Irecv → 非阻塞从 neighbor_rank 接收
4. MPI_Waitall → 等待通信完成
5. 解包 recv buffer: 按 recv_local_slots 写入 local_data
```

### 6.4 具体例子

8×8 网格，3×3 stencil，2 ranks：

```
rank 0: items [0,18) → 对应行 0-3, 列 0-5 的 3×3 块
rank 1: items [18,36) → 对应行 3-7, 列 0-5 的 3×3 块

rank 0 的 matIn input_globals 覆盖: 行 0-5, 列 0-7
rank 1 的 matIn input_globals 覆盖: 行 3-7, 列 0-7

重叠区域: 行 3-5

halo 规划结果:
  rank 0 需要 recv 的 = rank 0 input ∩ rank 1 output = 行 3-5 的部分数据
  rank 0 需要 send 的 = rank 0 output ∩ rank 1 input = 行 3-5 的部分数据
```

## 7. Sibling Loop 处理

`<->` 后面的 `for` 循环（sibling loops）负责将计算结果写回、处理边界条件等。translator 把它们生成为独立的 helper 函数。

### 7.1 设备侧执行路径

对于可以用 SYCL 并行化的 sibling loop，生成设备侧 kernel：

```
1. 从 ctx.buf_* 回收数据到 ctx.local_*
2. 构建 dense 读取视图（全局 Allreduce 后的完整数据）
3. 提交 SYCL parallel_for，在设备上执行 sibling loop 体
4. 从 dirty 位图提取实际写入的全局索引
5. 稀疏同步（见 7.2）
6. 写回 ctx.local_* → ctx.buf_*
```

### 7.2 稀疏同步

当多个 rank 对同一全局位置有写入时，需要合并。当前采用"求和后按出现次数取平均"的策略：

```
1. 从 dirty 位图提取本 rank 实际写入的全局索引 (written_idx)
2. MPI_Allgatherv 合并所有 rank 的写入索引
3. 对合并后的索引集合，收集各 rank 的值
4. MPI_Allreduce(..., MPI_SUM) 对值求和
5. MPI_Allreduce 对 present 计数求和
6. 最终值 = sum / present_count（平均）
7. 写回 ctx.local_*
```

### 7.3 Dense-Cover Pack

如果参数被 sibling 写入，其 `ctx.pack_*` 会在 init 阶段扩成覆盖整个全局线性空间。这是因为 sibling loop 使用原始的 `tensor[i][j]` 索引语法，需要所有全局位置都有对应的本地 slot。

## 8. 结果回收：Sync 阶段

迭代循环结束后，sync 函数将各 rank 的局部结果汇总回 root：

```cpp
void __dacpp_mpi_sync_stencilShell_stencil(ctx_type& ctx, ...) {
    ctx.q.wait();

    // 1. D2H: 设备数据写回本地数组
    sycl::host_accessor ha_sync_matIn(*ctx.buf_matIn, read_only);
    for (i = 0; i < ctx.local_matIn.size(); i++)
        ctx.local_matIn[i] = ha_sync_matIn[i];

    // 2. 构建写回数据（按全局位置排序）
    auto wb_matIn = build_writeback_values(ctx.local_matIn, ctx.pack_matIn);

    // 3. MPI_Gatherv: 收集全局位置和对应值
    MPI_Gather(&send_count, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, ...);
    MPI_Gatherv(wb_globals, send_count, MPI_LONG_LONG, ..., 0, ...);
    MPI_Gatherv(wb_values, send_count, ..., 0, ...);

    // 4. root 合并: 按全局位置写回原始 tensor
    if (mpi_rank == 0) {
        tensor.tensor2Array(global_out);
        apply_writeback_by_globals(global_recv_values, global_recv_globals, global_out);
        tensor.array2Tensor(global_out);
    }

    // 5. 如果需要，MPI_Bcast 同步到所有 rank
}
```

关键特性：
- 只收集每个 rank 真正修改的数据（writeback_globals），不是整个 tensor
- 按全局位置精确写回，避免覆盖其他 rank 的结果
- root 完成写回后可选择广播给所有 rank

## 9. 完整数据流总结

```
Init:
  root: tensor → global_array → pack_values_per_rank → MPI_Scatterv
  worker: recv → local_data → sycl::buffer (buf_*)

Per-iteration:
  Submit:  SYCL parallel_for (每个 item 一个 work-item)
           View 映射: calc 的 tensor[i][j] → local_data[slots[offset + ...]]
  Halo:    D2H → exchangeHalo (MPI_Isend/Irecv) → H2D
  Sibling: D2H → SYCL kernel (dense view + dirty tracking)
           → MPI_Allgatherv (written indices)
           → MPI_Allreduce (sparse value sync)
           → writeback → H2D

Sync:
  D2H → MPI_Gatherv (globals + values) → root apply_writeback → tensor
  可选: MPI_Bcast → 所有 rank 同步
```

## 10. 相关代码文件索引

| 文件 | 职责 |
|------|------|
| `dpcppLib/include/MPIPlanner.h` | 运行时支撑：item-space、pack-map、halo、views |
| `rewriter/lib/mpi/Rewriter_MPI_Region_Ctx.cpp` | 生成 ctx 结构体 |
| `rewriter/lib/mpi/Rewriter_MPI_Region_Init.cpp` | 生成 init 函数（scatter + halo 规划） |
| `rewriter/lib/mpi/Rewriter_MPI_Region_Submit.cpp` | 生成 submit 函数（SYCL kernel） |
| `rewriter/lib/mpi/Rewriter_MPI_Region_Halo.cpp` | 生成 halo 函数（邻居数据交换） |
| `rewriter/lib/mpi/Rewriter_MPI_Region_Sync.cpp` | 生成 sync 函数（gather + writeback） |
| `rewriter/lib/mpi/Rewriter_MPI_Region_Sibling.cpp` | 生成 sibling helper（稀疏同步） |
| `rewriter/lib/mpi/Rewriter_MPI_Region_Policy.cpp` | 传输策略分析 |
