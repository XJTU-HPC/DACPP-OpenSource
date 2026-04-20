# MPI Halo And Sibling Risk Analysis

本文档整理当前 MPI region 路径里，围绕 `halo` 邻居发现、`sibling` 设备化、`dense-cover pack` 与 `dirty sparse sync` 这几块已确认的问题，以及后续很可能继续暴露的潜在 bug 点。

文档描述当前实现的事实、风险边界和建议验证项，方便后续继续改动时直接作为 checklist 使用。

## 1. 范围

当前分析覆盖以下代码路径：

- `dpcppLib/include/MPIPlanner.h`
- `rewriter/lib/mpi/Rewriter_MPI_Pattern.cpp`
- `rewriter/lib/mpi/Rewriter_MPI_Region_Init.cpp`
- `rewriter/lib/mpi/Rewriter_MPI_Region_Halo.cpp`
- `rewriter/lib/mpi/Rewriter_MPI_Region_Codegen.cpp`
- `rewriter/lib/mpi/Rewriter_MPI_Region_Sibling.cpp`

重点关注三个事实：

- 当前 rank 划分按一维 `item-space` 连续分区
- halo 邻居集合通过数据驱动的全局位置交集发现，并利用 item-reach 分析将候选 rank 范围收缩到局部窗口
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

`computeParamHalo()` 对候选 rank 做数据驱动的邻居发现：

- 通过 `computeItemReach()` 分析 AccessPattern 的 split/bind 结构，计算 item-space 中的最大触及距离
- 通过 `computeRankWindow()` 将 item-reach 转换为 rank 窗口 `[rank - W, rank + W]`
- 只枚举窗口内的 rank，而非全部 P 个 rank
- 对每个候选 rank 计算其 input/output 全局位置集合
- recv = `my_input_globals ∩ neighbor_output_globals`
- send = `my_output_globals ∩ neighbor_input_globals`
- 只有存在非空交集的 rank 才成为 halo 邻居

对于广播/全维度等无界访问模式（`computeItemReach()` 返回 -1），自动退化为全量枚举，保证正确性。

slot 映射直接使用 `ctx_pack.g2l`，确保与 `exchangeHalo()` 读写 `ctx.local_*` 的布局完全一致。

`exchangeHalo()` 再根据 `send_local_slots` / `recv_local_slots` 去直接访问 `local_data`。

#### 2.2.1 item-reach 分析

`computeItemReach()` 的计算逻辑：

- **RegularSlice**（`is_index_op=false`）：`bind-index space` 触及半径 = `ceil(op.size / op.stride) - 1`
- **IndexSplit**（`is_index_op=true`）：每个 item 精确访问一个位置，触及半径 = 0
- 将 `bind-index` 触及半径乘以对应 bind 维度在 item-space 中的步幅（`product(bind_split_sizes[B+1:])`），得到 item-space 触及距离
- 如果某个 IndexSplit 的 `split_size == dimLength[dim]`（广播模式），返回 -1 表示无界

`computeRankWindow()` 将 item-reach 转换为 rank 窗口：

- `rank_reach = ceil(item_reach / min_items_per_rank) + 1`
- +1 补偿 `get_rank_item_range()` 的不均匀分配

典型 stencil 场景下（3~5 点），rank 窗口仅覆盖 rank±1，即只枚举 2 个候选 rank。

### 2.3 sibling 写入后的同步

当前 sibling 路径有两种同步策略：

- 一部分旧路径仍然会把 `ctx.pack_*` 展成全量 dense 数组后做全量 `Allreduce`
- 新路径会把真实写入的全局索引收集出来，构造稀疏索引集合，再对 value/present 做 `Allreduce`

## 3. 已确认的设计事实与正确性边界

### 3.1 halo slot 映射直接使用 `ctx.pack_*`

`computeParamHalo()` 接受 `const PackMap& ctx_pack` 参数，直接使用 `ctx_pack.g2l` 做 slot 映射。

`exchangeHalo()` 读写的是 `ctx.local_*`，而 `ctx.local_*` 的 layout 由 `ctx.pack_*` 决定。由于 `computeParamHalo()` 直接消费 `ctx_pack`（即 `ctx.pack_*`），slot 映射与实际数据布局保持一致。

对于 sibling dense-cover pack（`ctx.pack_*` 覆盖全局 dense 空间），所有合法 send/recv global 均可正确映射。

recv/send 路径的 `g2l.find()` 失败时直接抛出 `std::runtime_error`（包含 global index 和 "pack layout mismatch" 描述），实现 fail-fast。

### 代码位置

- `MPIPlanner.h` 中 `computeParamHalo()` — 函数签名和 slot 查找逻辑
- `Rewriter_MPI_Region_Init.cpp` 中 `computeParamHalo()` 调用点

### 3.2 halo 邻居发现为数据驱动 + item-reach 窗口过滤

邻居发现不依赖 rank 编号的相邻性。`computeParamHalo()` 对候选 rank 计算实际全局位置交集来判断 halo 依赖：

- recv 方向：`my_input_globals ∩ nb_output_globals`
- send 方向：`my_output_globals ∩ nb_input_globals`

候选 rank 的范围由 `computeItemReach()` + `computeRankWindow()` 确定：

- 普通 stencil 模式：只枚举 rank 窗口内的少量 rank（通常 rank±1）
- 广播/全维度等无界模式：退化为全量枚举

该方案正确处理所有场景：多维 bind、非单位 stride、`[{}]` 全维度广播、大偏移 stencil。

### 代码位置

- `MPIPlanner.h` 中 `computeItemReach()`、`computeRankWindow()`、`computeParamHalo()` 的邻居迭代循环

### 3.3 halo 输出集合按整块 partition 估计，默认假设分区内都可能被写

`computeParamHalo()` 构造 `my_output_globals` 的方式，是直接把 item 对应 partition 覆盖到的全体位置都视为"可能输出"。

这套做法在"每个 item 会完整写满自己的输出 partition"时是安全的，但它引入了一个隐藏假设：

- calc 的真实写集合至少覆盖这块 partition 中所有对外可见位置

如果实际写入是条件性的、稀疏的，或者只写 partition 的一部分，那么 halo send 可能把尚未写过的旧值也一并发出去。

**当前状态：属于保守估计。可能造成过度通信或传播 stale data。**

### 影响

- 可能造成过度通信
- 更严重时可能传播 stale data

### 3.4 sibling dense 初始化路径的尺寸校验

存在 sibling 写入时，某些参数会走 full dense broadcast 初始化。当前逻辑在 global 数组大小与 `ctx.local_*` 大小不匹配时，rank 0 会直接抛出 `std::runtime_error`（包含参数名、global_size 和 local_size），实现 fail-fast。

### 代码位置

- `Rewriter_MPI_Region_Init.cpp` 中 sibling dense-cover pack 初始化的 copy-in 路径

## 4. 可能继续出现的其他 bug 点

下面这些点不是每一项都已经在现有样例中复现，但从当前实现看，都属于值得优先防守的高风险区域。

### 4.1 sibling dense-cover pack 与 halo planner 的"双重 pack 语义漂移"

当前 sibling 写入参数会把 `ctx.pack_*` 扩成全局 dense pack，但 `computeParamHalo()` 已经直接消费 `ctx_pack`（即 `ctx.pack_*`），不再私自重建 slot 映射。

**残留风险：** 如果后续有新代码路径不经 `computeParamHalo()` 而直接访问 slot，仍可能出现语义漂移。建议在新路径中也始终使用 `ctx.pack_*` 的 `g2l`。

### 4.2 sibling 稀疏同步目前默认用 `SUM + present-count + 平均` 合并冲突

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

### 4.3 full dense pack 会把一些"原本依赖 pack 边界"的逻辑变成全局语义

很多辅助逻辑原本默认 `pack.globals.size()` 近似表示"当前 rank 实际拥有或实际关心的局部集合"。

但 sibling dense-cover 后，`ctx.pack_*` 的语义变成"全局 dense 线性空间"。

`computeParamHalo()` 已通过直接使用 `ctx.pack_*` 的 `g2l` 解决了 halo slot lookup 的语义对齐问题。其他依赖 pack 边界的逻辑（writeback 集合、present / dirty 掩码大小等）仍需注意全局 dense 语义的影响。

### 4.4 多维 item-space 线性切分后，空间邻居关系可能和 rank 邻居关系脱钩

`get_rank_item_range()` 切的是线性 item id 区间，不是几何块。

如果 bind 维度大于一维，线性 item id 一般是按字典序展开的。这样一来：

- 连续的 rank 区间不一定对应连续的几何块
- 真正的空间邻居可能分散到多个非相邻 rank

**当前状态：此风险已被数据驱动的邻居发现兜住。** `computeParamHalo()` 对候选 rank 做实际全局位置交集检测，不依赖 rank 编号的相邻性。即使空间邻居分散到非相邻 rank，只要落在 item-reach 窗口内就能被正确发现。对于 item-reach 窗口无法覆盖的极端场景，`computeItemReach()` 返回 -1 触发全量枚举。

## 5. 建议优先验证项

为了尽快锁定 correctness，建议下一轮至少补下面几类验证：

1. WRITE 参数存在 send halo、但 send global 不属于 input pack 的样例 — slot 映射使用 ctx_pack，已正确处理
2. sibling 写入参数开启 dense-cover pack 后，halo 前后数据仍能对齐到正确 slot 的样例 — 已正确处理
3. 多维 bind item-space 下，真实依赖跨越非相邻 rank 的样例 — 数据驱动邻居发现已正确处理
4. 实际写集合小于 partition 覆盖范围的样例（验证 stale data 不被传播）— 尚未验证
5. 多 rank 同写同一全局索引时，验证当前"平均合并"是否符合语义预期 — 尚未验证
6. dense init 尺寸不匹配时，验证系统是否能 fail-fast — 已正确处理

## 6. 实现要点总结

1. **halo planner 直接消费 `ctx.pack_*`** — `computeParamHalo()` 接受 `const PackMap& ctx_pack` 参数，slot 映射与实际数据布局保持一致。`g2l` 查找失败时抛出异常。
2. **邻居发现为数据驱动 + item-reach 窗口过滤** — `computeItemReach()` 从 AccessPattern 的 split/bind 结构计算 item-space 触及距离，`computeRankWindow()` 将其转换为 rank 窗口，`computeParamHalo()` 只枚举窗口内的候选 rank。无界模式自动退化为全量枚举。正确处理多维 bind、非单位 stride、全维度广播等场景。
3. **dense init 尺寸不匹配 fail-fast** — copy-in 失败时抛出异常而非静默跳过。
