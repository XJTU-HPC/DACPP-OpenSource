# MPI Halo And Sibling Risk Analysis

本文档整理当前 MPI region 路径里，围绕 `halo` 邻居发现、`sibling` 设备化、`dense-cover pack` 与 `dirty sparse sync` 这几块已确认的问题，以及后续很可能继续暴露的潜在 bug 点。

文档描述当前实现的事实、风险边界和建议验证项，方便后续继续改动时直接作为 checklist 使用。

## 1. 范围

当前分析覆盖以下代码路径：

- `dpcppLib/include/MPIPlanner.h`
- `rewriter/lib/mpi/Rewriter_MPI_Pattern.cpp`
- `rewriter/lib/mpi/Rewriter_MPI_Region_Init.cpp`
- `rewriter/lib/mpi/Rewriter_MPI_Region_Halo.cpp`
- `rewriter/lib/mpi/Rewriter_MPI_Region.cpp`
- `rewriter/lib/mpi/Rewriter_MPI_Region_Sibling.cpp`

重点关注三个事实：

- 当前 rank 划分按一维 `item-space` 连续分区
- halo 邻居集合通过数据驱动的全局位置交集发现（已修复，见 3.3）
- 存在 sibling 写入的参数会被扩成覆盖全局线性索引的 dense pack，然后再走 dirty 稀疏同步

## 2. 当前实现摘要

### 2.1 item-space 与 pack

当前 item 到 rank 的分配由 `get_rank_item_range()` 完成，做法是把总 item 数按连续区间切到各个 rank。

每个 item 访问哪些全局位置，由 `collect_positions_for_item()` 结合：

- `bind_set_id`
- `bind_offset_expr`
- `is_index_op`
- `partition_shape`

共同推导出来。

普通 region 参数的 pack 由 `buildPackBuilderExpr()` 选择：

- `READ` -> `build_input_pack_map(item_range, pattern)`
- `WRITE` -> `build_output_pack_map(item_range, pattern)`
- `READ_WRITE` -> `build_rw_pack_map(item_range, pattern)`

但只要参数被判定为"存在 sibling 写入"，`Rewriter_MPI_Region_Init.cpp` 就会直接把 `ctx.pack_*` 扩成覆盖整个全局线性空间的 dense pack，而不是继续沿用按 item 推导出的原始 pack。

### 2.2 halo

`computeParamHalo()` 当前对**所有**其他 rank 做数据驱动的邻居发现：

- 遍历所有 rank（除自身外）
- 对每个 rank 计算其 input/output 全局位置集合
- recv = `my_input_globals ∩ neighbor_output_globals`
- send = `my_output_globals ∩ neighbor_input_globals`
- 只有存在非空交集的 rank 才成为 halo 邻居

slot 映射直接使用 `ctx_pack.g2l`，确保与 `exchangeHalo()` 读写 `ctx.local_*` 的布局完全一致。

`exchangeHalo()` 再根据 `send_local_slots` / `recv_local_slots` 去直接访问 `local_data`。

### 2.3 sibling 写入后的同步

当前 sibling 路径已经有两种同步策略：

- 一部分旧路径仍然会把 `ctx.pack_*` 展成全量 dense 数组后做全量 `Allreduce`
- 新路径会把真实写入的全局索引收集出来，构造稀疏索引集合，再对 value/present 做 `Allreduce`

## 3. 已确认并修复的问题

### 3.1 ~~`computeParamHalo()` 的本地 slot 映射没有和 `ctx.pack_*` 对齐~~ [已修复]

**状态：已修复。**

`computeParamHalo()` 之前会自己重建一份本地 `PackMap`：

```cpp
PackMap my_pack = make_pack_map_from_globals(my_input_globals);
```

然后用这份临时 `my_pack.g2l` 去生成 `recv_local_slots` / `send_local_slots`。

但是 `exchangeHalo()` 真正读写的是 `ctx.local_*`，而 `ctx.local_*` 的真实 layout 由 `ctx.pack_*` 决定，不是由 `computeParamHalo()` 里这份临时 pack 决定。

**修复方式：**

- `computeParamHalo()` 新增 `const PackMap& ctx_pack` 参数
- 直接使用 `ctx_pack.g2l` 做 slot 映射，不再私自重建
- 两个调用点（`Rewriter_MPI_Region_Init.cpp` 和 `Rewriter_MPI_Region.cpp`）均已传入 `ctx.pack_<name>`

### 代码位置

- `MPIPlanner.h` 中 `computeParamHalo()` — 函数签名和 slot 查找逻辑
- `Rewriter_MPI_Region_Init.cpp` 中 `computeParamHalo()` 调用点
- `Rewriter_MPI_Region.cpp` 中 `computeParamHalo()` 调用点

### 3.2 ~~halo send 路径复用了 input-only 的 `g2l`，会静默丢掉合法 send~~ [已修复]

**状态：已修复（与 3.1 一并修复）。**

`computeParamHalo()` 之前把 recv 和 send 都映射到同一个 `my_pack` 上，而 `my_pack` 只来自 `my_input_globals`。这导致 WRITE 参数的 send global 如果不在 input 集合中，查找失败后直接被跳过（静默丢掉）。

**修复方式：**

- 使用 `ctx_pack.g2l` 替代 input-only 的 `my_pack`
- 对于 sibling dense-cover pack，`ctx_pack` 覆盖全局 dense 空间，所有合法 send global 均可正确映射

### 代码位置

- `MPIPlanner.h` 中 `computeParamHalo()` 的 `ctx_pack.g2l` 统一使用
- send_globals 到 send_local_slots 的映射循环

### 3.3 ~~halo 邻居集合只看左右 rank~~ [已修复]

**状态：已修复。**

之前实现把 halo 邻居固定成 `rank - 1` 和 `rank + 1`，这只有在下面条件同时成立时才安全：

- 真实依赖在扁平化后的 item-space 里是带状局部依赖
- 任意 rank 需要的远端数据，只会落在相邻 rank 的 item 区间边界上

一旦出现下面任一情况，就有遗漏风险：

- halo 半径超过一个 rank 边界
- 多维 bind flatten 后，空间邻近关系不再等价于 item id 邻近关系
- offset / stride 导致访问图在扁平 item-space 中变成跳跃依赖

**修复方式：**

将邻居发现改为数据驱动：

```cpp
// 旧代码：
for (int delta : {-1, 1}) {
    const int neighbor = mpi_rank + delta;
    ...
}

// 新代码：
for (int nb_rank = 0; nb_rank < mpi_size; ++nb_rank) {
    if (nb_rank == mpi_rank) continue;
    // 计算该 rank 的 input/output globals
    // 通过交集判断是否有 halo 依赖
    ...
}
```

对每个 rank 计算其 input/output 全局位置集合，通过 `my_input ∩ nb_output` 和 `my_output ∩ nb_input` 判断是否存在 halo 依赖。只有存在非空交集的 rank 才成为 halo 邻居。

该方案正确处理所有场景：多维 bind、非单位 stride、`[{}]` 全维度广播、大偏移 stencil。

### 代码位置

- `MPIPlanner.h` 中 `computeParamHalo()` 的邻居迭代循环

### 3.4 halo 输出集合按整块 partition 估计，默认假设分区内都可能被写

`computeParamHalo()` 当前构造 `my_output_globals` 的方式，是直接把 item 对应 partition 覆盖到的全体位置都视为"可能输出"。

这套做法在"每个 item 会完整写满自己的输出 partition"时是安全的，但它额外引入了一个隐藏假设：

- calc 的真实写集合至少覆盖这块 partition 中所有对外可见位置

如果实际写入是条件性的、稀疏的，或者只写 partition 的一部分，那么 halo send 可能把尚未写过的旧值也一并发出去。

**当前状态：未修复，属于保守估计。可能造成过度通信或传播 stale data。**

### 影响

- 可能造成过度通信
- 更严重时可能传播 stale data

## 4. 已确认并修复的额外问题

### 4.5 ~~当前缺少"交集存在但 slot 映射失败"的断言~~ [已修复]

**状态：已修复。**

在 halo 规划里，一旦出现下面这种情况：

- `recv_globals` 或 `send_globals` 的交集非空
- 但 `g2l` 查找失败

之前代码会直接把这项跳过，不会报错。这会让很多真正的布局问题变成 silent correctness bug。

**修复方式：**

recv/send 路径的 `g2l.find()` 失败时改为 `throw std::runtime_error`，抛出包含 global index 和 "pack layout mismatch" 描述的异常信息，实现 fail-fast。

### 代码位置

- `MPIPlanner.h` 中 `computeParamHalo()` 的 recv/send slot 映射循环

### 4.3 ~~sibling dense 初始化路径在尺寸不匹配时会静默跳过 copy-in~~ [已修复]

**状态：已修复。**

存在 sibling 写入时，某些参数会走 full dense broadcast 初始化。之前逻辑只有在 global 数组大小与 `ctx.local_*` 大小完全相等时，rank 0 才会把全局数据拷进 `ctx.local_*`。

如果尺寸不相等，之前逻辑不会报错，只是直接跳过这次 copy，然后继续 `MPI_Bcast`。这会把"copy-in 失败"降级成"广播未初始化或旧数据"。

**修复方式：**

在 `else` 分支添加 `throw std::runtime_error`，在 rank 0 上抛出包含参数名、global_size 和 local_size 的异常信息，实现 fail-fast。

### 代码位置

- `Rewriter_MPI_Region_Init.cpp` 中 sibling dense-cover pack 初始化的 copy-in 路径

## 5. 可能继续出现的其他 bug 点

下面这些点不是每一项都已经在现有样例中复现，但从当前实现看，都属于值得优先防守的高风险区域。

### 5.1 sibling dense-cover pack 与 halo planner 的"双重 pack 语义漂移"

当前 sibling 写入参数会把 `ctx.pack_*` 扩成全局 dense pack，但 `computeParamHalo()` 现在已经直接消费 `ctx_pack`（即 `ctx.pack_*`），不再私自重建 slot 映射。此风险已通过 3.1 的修复缓解。

**残留风险：** 如果后续有新代码路径不经 `computeParamHalo()` 而直接访问 slot，仍可能出现语义漂移。建议在新路径中也始终使用 `ctx.pack_*` 的 `g2l`。

### 5.2 sibling 稀疏同步目前默认用 `SUM + present-count + 平均` 合并冲突

当前稀疏同步路径会：

- 对稀疏 values 做 `MPI_Allreduce(..., MPI_SUM, ...)`
- 对 present mask / count 再做一次 `Allreduce`
- 如果某个索引出现多份写入，就按出现次数做平均

这隐含了一个语义假设：

- 多个 rank 对同一全局索引的写入要么本来就相同，要么允许用平均值合并

如果真实语义不是"可交换的聚合写"，而是：

- 单写者语义
- 最后写者语义
- 必须报冲突语义

那么"平均"会把错误悄悄抹平，结果看起来像数值偏差，而不是显式同步错误。

### 5.3 ~~full dense pack 会把一些"原本依赖 pack 边界"的逻辑变成全局语义~~ [部分缓解]

很多辅助逻辑原本默认 `pack.globals.size()` 近似表示"当前 rank 实际拥有或实际关心的局部集合"。

但 sibling dense-cover 后，`ctx.pack_*` 的语义已经变成"全局 dense 线性空间"。

**缓解状态：** `computeParamHalo()` 已通过直接使用 `ctx.pack_*` 的 `g2l` 解决了 halo slot lookup 的语义对齐问题。其他依赖 pack 边界的逻辑（writeback 集合、present / dirty 掩码大小等）仍需注意全局 dense 语义的影响。

### 5.4 多维 item-space 线性切分后，空间邻居关系可能和 rank 邻居关系脱钩

`get_rank_item_range()` 现在切的是线性 item id 区间，不是几何块。

如果 bind 维度大于一维，线性 item id 一般是按字典序展开的。这样一来：

- 连续的 rank 区间不一定对应连续的几何块
- 真正的空间邻居可能分散到多个非相邻 rank

**当前状态：此风险已被 3.3 的修复兜住。** `computeParamHalo()` 现在对所有 rank 做数据驱动的交集检测，不再依赖 rank 编号的相邻性。即使空间邻居分散到非相邻 rank，也能被正确发现。

## 6. 建议优先验证项

为了尽快锁定 correctness，建议下一轮至少补下面几类验证：

1. ~~WRITE 参数存在 send halo、但 send global 不属于 input pack 的样例~~ [3.1/3.2 已修复，slot 映射现在使用 ctx_pack]
2. ~~sibling 写入参数开启 dense-cover pack 后，halo 前后数据仍能对齐到正确 slot 的样例~~ [3.1 已修复]
3. ~~多维 bind item-space 下，真实依赖跨越非相邻 rank 的样例~~ [3.3 已修复]
4. 实际写集合小于 partition 覆盖范围的样例（验证 stale data 不被传播）
5. 多 rank 同写同一全局索引时，验证当前"平均合并"是否符合语义预期
6. ~~dense init 尺寸不匹配时，验证系统是否能 fail-fast~~ [4.3 已修复]

## 7. 修复收敛总结

以下三件事已收口：

1. **halo planner 直接消费 `ctx.pack_*`** — `computeParamHalo()` 接受 `const PackMap& ctx_pack` 参数，不再私自重建 slot 映射。[3.1/3.2]
2. **slot 映射失败改为显式断言** — recv/send 路径 `g2l.find()` 失败时抛出 `std::runtime_error`。[4.5]
3. **邻居发现从"左右 rank"升级为数据驱动** — 对所有 rank 计算实际全局位置交集，正确处理多维 bind、非单位 stride、全维度广播等场景。[3.3]

额外收口：

4. **dense init 尺寸不匹配改为 fail-fast** — copy-in 失败时抛出异常而非静默跳过。[4.3]
