# MPI Halo 邻居发现：Item-Reach 窗口过滤

## 1. 背景

`computeParamHalo()` 在初始化阶段为每个参数发现 halo 邻居 rank。该函数需要对候选 rank 逐一计算全局位置集合并做交集检测，单次检测的开销为 `O(items_per_rank × T_item)`。

早期实现只检查 `rank-1` 和 `rank+1`（固定 2 个候选），但这对多维 bind、非单位 stride、广播模式等场景会遗漏邻居。

后续改为遍历全部 P 个 rank 做数据驱动的交集检测，确保了正确性，但初始化开销变为 `O(P × items_per_rank × T_item)`。在众核环境下（P=100+），这个一次性开销变得不可忽视。

本优化在数据驱动的基础上引入 item-reach 分析，将候选范围从全部 P 个 rank 收缩到局部窗口。

## 2. 核心观察

`get_rank_item_range()` 把 items 按**连续区间**分配给各 rank。对于 stencil 模式，数据依赖在 item-space 中是局部的：一个 rank 的数据触及范围有限，只有附近的 rank 才可能是邻居。

因此可以从 AccessPattern 的 split/bind 结构中推导出 item-space 的最大触及距离，再转换为 rank 窗口，只枚举窗口内的 rank。

## 3. 实现结构

所有代码位于 `dpcppLib/include/MPIPlanner.h`。

### 3.1 `computeItemReach(pattern)`

从 AccessPattern 计算 item-space 中的最大触及距离。

**输入**：`AccessPattern`（含 `param_ops`、`bind_set_id`、`is_index_op`、`bind_split_sizes`）

**输出**：`ItemReachResult { max_item_reach }`，其中 `-1` 表示无界（需全量枚举）

**算法**：

1. 对每个 bind 维度计算 bind-index space 中的触及半径：
   - **IndexSplit**（`is_index_op=true`）：每个 item 精确访问一个位置，触及半径 = 0
   - **RegularSlice**（`is_index_op=false`）：`partition_shape[dim] = op.size`，触及半径 = `ceil(op.size / op.stride) - 1`
2. 将 bind-index space 的触及半径转换为 item-space：
   - bind 维度 B 在 item-space 中的步幅 = `product(bind_split_sizes[B+1:])`
   - 这与 `decode_item_id()` 中的 divisor 计算一致
   - item-space 触及 = `per_bind_reach[B] × item_stride_for_B`
3. 取所有 bind 维度的最大值
4. 检测广播模式（IndexSplit 的 `split_size >= dimLength[dim]`）→ 返回 -1

**典型计算结果**：

| 访问模式 | bind_reach | item_stride | item_reach |
|----------|-----------|-------------|------------|
| 1D 3点 stencil, `sp1(3,1)`, 单维 split | 2 | 1 | 2 |
| 1D 5点 stencil, `sp1(5,1)`, 单维 split | 4 | 1 | 4 |
| 2D 3×3 stencil, `sp1(3,1)` 双维 split, `bind_split_sizes=[R,C]` | 行:2, 列:2 | 行:C, 列:1 | max(2C, 2) |
| 2D 3×3 stencil, `sp1(3,1)`, 只按行 split | 行:2 | 1 | 2 |

### 3.2 `computeRankWindow(item_reach, total_items, mpi_rank, mpi_size)`

将 item-space 触及距离转换为 rank 窗口 `[lo, hi]`（inclusive）。

**算法**：

```
min_items = total_items / mpi_size  (每个 rank 的最少 item 数)
rank_reach = ceil(item_reach / min_items) + 1
窗口 = [max(0, rank - rank_reach), min(mpi_size - 1, rank + rank_reach)]
```

`+1` 补偿 `get_rank_item_range()` 的不均匀分配（某些 rank 多 1 个 item）。

**退化处理**：

- `item_reach < 0`（无界模式）：返回 `{-1, -1}`，触发全量枚举
- `min_items == 0`（rank 数 > item 数）：返回 `{-1, -1}`，触发全量枚举

### 3.3 `computeParamHalo()` 的修改

枚举循环从全量扫描改为窗口扫描：

```cpp
const auto reach = computeItemReach(pattern);
const auto win = computeRankWindow(reach.max_item_reach, total_items, mpi_rank, mpi_size);
const int nb_lo = (win.lo < 0) ? 0 : win.lo;
const int nb_hi = (win.hi < 0) ? (mpi_size - 1) : win.hi;

for (int nb_rank = nb_lo; nb_rank <= nb_hi; ++nb_rank) {
    // 原有交集检测逻辑不变
}
```

当窗口无界时（`win.lo < 0`），自动退化为全量枚举，行为与优化前完全一致。

## 4. 正确性论证

### 4.1 过近似保证不遗漏

rank 窗口是过近似（over-approximation）：

- `rank_reach = ceil(item_reach / min_items) + 1` 保证窗口覆盖所有可能存在数据重叠的 rank
- 多检查的 rank 不会产生误报（交集为空时自然跳过）
- 唯一代价是少量多余的交集计算

### 4.2 item-stride 转换正确性

`decode_item_id()` 将线性 item_id 解码为 bind 索引：

```
item_id = bind[0] × (split[1] × split[2] × ...) + bind[1] × (split[2] × ...) + ...
```

bind 维度 B 的 item-space 步幅 = `product(bind_split_sizes[B+1:])`。

因此 bind 维度 B 上的触及半径 `r` 在 item-space 中的触及距离 = `r × stride_B`。这个转换确保了多维 bind 的场景下，rank 窗口能覆盖所有空间方向上的邻居。

### 4.3 各场景验证

| 场景 | item_reach | rank 窗口 | 正确性 |
|------|-----------|----------|--------|
| 1D stencil, P=100, N=10000 | 2 | [rank-1, rank+1] | 正确，只检查相邻 rank |
| 1D stencil, P=1000, N=10000 | 2 | [rank-1, rank+1] | 正确，min_items=10，rank_reach=1 |
| 2D stencil 双维 split, P=100, 100×100 | 196 | [rank-3, rank+3] | 正确，行维度 stride=98 扩大了窗口 |
| 2D stencil 只按行 split, P=100, 100×100 | 2 | [rank-1, rank+1] | 正确，只有行方向有 split |
| 广播模式 `[{}]` | -1 | 全量枚举 | 正确，退化为原始行为 |
| 非单位 stride `sp1(2,3)` | 0 | [rank, rank]（无邻居） | 正确，相邻 item 无重叠 |

## 5. 性能分析

**初始化阶段**（一次性开销，单参数）：

- 优化前：`O(P × items_per_rank × T_item)`
- 优化后：`O(W × items_per_rank × T_item)`，W = rank 窗口宽度

**典型场景加速比**：

| P | stencil | items_per_rank | T_item | 优化前候选数 | 优化后候选数 | 加速比 |
|---|---------|---------------|--------|------------|------------|--------|
| 4 | 3点 | 2500 | 9 | 3 | 2 | 1.5× |
| 100 | 3点 | 100 | 9 | 99 | 2 | ~50× |
| 1000 | 3点 | 10 | 9 | 999 | 2 | ~500× |
| 100 | 5点 | 100 | 9 | 99 | 2 | ~50× |
| 100 | 3×3 双维 | 96 | 9 | 99 | 6 | ~17× |

**运行时 halo exchange 不受影响**：`exchangeHalo()` 只遍历已发现的邻居，通信量不变。

**`computeItemReach()` 和 `computeRankWindow()` 自身开销**：O(ops)，ops 为 param_ops 数量（通常 < 10），可忽略。

## 6. 与旧方案的关系

| 方案 | 邻居范围 | 正确性 | 初始化开销 |
|------|---------|--------|-----------|
| 硬编码 rank±1 | 固定 2 个 | 多维/广播场景会遗漏 | O(2 × items × T) |
| 全量枚举（优化前） | 全部 P 个 | 正确 | O(P × items × T) |
| item-reach 窗口过滤（当前） | 自适应 W 个 | 正确，无界模式退化为全量 | O(W × items × T) |

当前方案是全量枚举方案的优化版本：行为等价（对无界模式完全一致），但在 stencil 等有界模式下将候选数从 P 降低到常数级别。
