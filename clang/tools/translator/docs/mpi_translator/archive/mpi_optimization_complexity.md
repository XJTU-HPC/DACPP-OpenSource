# MPI Region 优化复杂度评估

本文对 `plan.md` 中已落地的关键优化给出统一的时间复杂度模型，描述优化前后在计算量、通信量与典型场景下的数量级变化。

## 1. 记号约定

- `P`：MPI rank 数
- `N`：参数对应全局 tensor 元素数（dense 元素总量）
- `L_r`：第 `r` 个 rank 的 pack 元素数，`L = max_r L_r`
- `W_r`：第 `r` 个 rank 在一个 sibling helper 内真实写入的全局索引数
- `W = max_r W_r`，`W_union = |∪_r written_idx_r|`
- `I`：外层时间推进迭代次数
- `S`：sibling helper 数量（每次时间推进中）
- `T_item`：单个 item 对应的 partition 元素数（由 `partition_element_count` 计算）
- `G_r`：第 `r` 个 rank 的全局位置集合大小（input 或 output）

当未特别说明时，复杂度关注单参数、单次 helper；整体复杂度可按参数数目线性叠加。

## 2. 优化项 A：按 item-space + pack-map 分发（替代 dense 全量分发）

### 2.1 优化目标

将参数初始化与回收从"按 tensor dense 全量处理"收紧为"按 rank 实际访问集合处理"。

### 2.2 复杂度对比

- 优化前（dense 全量思路）：
  - root 侧打包/解包：`O(P * N)`
  - 通信数据量：`O(P * N)`
- 优化后（pack-map）：
  - root 侧打包/解包：`O(Σ_r L_r)`
  - 通信数据量：`O(Σ_r L_r)`

### 2.3 收益表达

- 复杂度缩减比例约为：`(P * N) / (Σ_r L_r)`
- 当访问稀疏且 `L_r << N` 时，收益接近 `N / L_avg` 量级。

## 3. 优化项 B：sibling 设备化执行（直接作用 `ctx.buf_*`）

### 3.1 优化目标

将 sibling 执行路径从 host dense bridge 的"重建 dense + host 执行 + 回写"改为设备侧 packed 视图执行，仅保留必要同步。

### 3.2 复杂度对比（单次 helper）

- 优化前（host dense bridge）：
  - dense temporary 构建与回写：`O(N)`
  - dense 级同步：`O(N)`
  - sibling 计算：`O(T_helper)`（与循环体规模相关）
  - 合计主项：`O(N + T_helper)`
- 优化后（device packed + fallback/shadow）：
  - 预同步（buf->local）：`O(L)`
  - sibling 计算：`O(T_helper)`
  - 后同步（local->buf）：`O(L)`
  - 合计主项：`O(L + T_helper)`

### 3.3 收益表达

- 与数据搬运相关的主项由 `O(N)` 收紧为 `O(L)`。
- 当 `L << N` 时，helper 级别的非计算开销下降到原来的 `L / N`。

## 4. 优化项 C：全局可覆盖 pack（针对存在 sibling 写入的参数）

### 4.1 优化目标

确保 sibling 内任何全局索引写入都有稳定落点，避免"写入位置超出局部 pack 覆盖"导致的语义不完整。

### 4.2 复杂度变化

- 初始化阶段：
  - 额外构建全局映射与查表：`O(N)`
  - 全局初值广播：`O(N)`
- 运行阶段（每次 helper）：
  - 保持写入语义正确，后续仍可使用 dirty 稀疏同步收紧通信。

### 4.3 结论

- 该优化的核心是**语义稳定性提升**，不是单独降低复杂度。
- 它为优化项 D（dirty 稀疏同步）提供正确性前提。

## 5. 优化项 D：sibling 后从 dense 全量 Allreduce 收紧为 dirty 稀疏同步

### 5.1 优化目标

只同步真实写入位置，避免每次 helper 对全局 dense 空间做全量归并。

### 5.2 复杂度对比（单次 helper）

- 优化前（dense 全量 Allreduce）：
  - 值归并：`O(N)`
  - present 归并：`O(N)`
  - 合计：`O(N)`
- 优化后（dirty 稀疏）：
  - 本地脏索引提取：`O(L)`
  - 脏索引集合合并（Allgatherv + 去重）：`O(Σ_r W_r + W_union log W_union)`
  - 值/present 稀疏归并：`O(W_union)`
  - 写回局部：`O(W_union)`
  - 合计主项：`O(L + Σ_r W_r + W_union log W_union)`

在常见 `W_union << N` 场景下，可近似为 `O(L + Σ_r W_r)`。

### 5.3 收益表达

- 归并复杂度从 `O(N)` 收紧为 `O(W_union)` 主导。
- 单次 helper 通信量缩减比例约为：`N / W_union`。

## 6. 优化项 E：`PackedElementRef` 读写路径统一（含 ref-to-ref 赋值）

### 6.1 优化目标

统一 packed 视图的读写行为：命中 slot 直接写、miss 走 fallback、写入更新 shadow 并打 dirty；确保 `ref = ref` 触发真实写入语义。

### 6.2 复杂度变化

- 单次元素读写仍为 `O(1)`。
- 该优化不改变 Big-O，主要收益是：
  - 避免写丢失导致的重复迭代调试成本
  - 保障 dirty 稀疏同步输入集合正确

## 7. 优化项 F：数据驱动的 halo 邻居发现

### 7.1 优化目标

将 halo 邻居发现从硬编码 `rank ± 1` 改为数据驱动：对所有 rank 计算实际全局位置交集，只与存在依赖的 rank 通信。

### 7.2 复杂度对比（初始化阶段，单参数）

- 优化前（rank ± 1）：
  - 邻居数固定为 2
  - 每个邻居的全局位置计算：`O(items_per_rank * T_item)`
  - 交集计算：`O(G_self + G_nb)`
  - 合计：`O(2 * items_per_rank * T_item)`

- 优化后（数据驱动全 rank）：
  - 遍历 `P - 1` 个 rank
  - 每个 rank 的全局位置计算：`O(items_per_rank * T_item)`
  - 交集计算：`O(G_self + G_nb)`
  - 合计：`O(P * items_per_rank * T_item)`

### 7.3 复杂度分析

- 初始化阶段成本从 `O(2 * items_per_rank * T_item)` 增长到 `O(P * items_per_rank * T_item)`
- 对于典型使用场景（`P = 2~8`），额外开销可忽略不计
- 对于大规模 `P`（100+），可考虑添加基于边界 item 的快速过滤优化

### 7.4 正确性收益

- 消除了 rank ± 1 假设下的邻居遗漏风险
- 正确处理多维 bind（item-space 线性化导致空间邻居分散到非相邻 rank）
- 正确处理非单位 stride（如 interleaved 访问模式）
- 正确处理 `[{}]` 全维度广播访问（自然发现所有 rank 都是邻居）
- 正确处理大偏移 stencil（依赖跨越多个 rank 边界）

### 7.5 运行时通信复杂度

- halo exchange 的通信量取决于实际发现的邻居数 `K`（而非固定 2）
- 对于局部 stencil：`K` 通常仍为 2（与 rank ± 1 相同），通信量不变
- 对于全维度广播：`K = P - 1`，通信量增大但结果是正确的
- exchangeHalo 已支持任意数量的 region，无需额外修改

## 8. 端到端复杂度视角（region 主循环）

对一个参数，`I` 次时间推进、每次 `S` 个 sibling helper 的开销主项可写为：

- 优化前（dense sibling 同步 + rank ± 1 halo）：
  - `O(I * S * N)`
- 优化后（dirty 稀疏同步 + 数据驱动 halo）：
  - `O(I * S * (L + Σ_r W_r + W_union log W_union))`
  - 初始化阶段 halo 规划：`O(P * items_per_rank * T_item)`（一次性开销）

在 `W_union << N` 且 `L << N` 的常见 stencil/局部更新场景下，复杂度主项由 `N` 量级降为"脏写规模"量级。

## 9. 场景化数量级示例

设：

- `N = 10^8`
- `L = 10^6`
- `W_union = 10^5`
- `P = 4`
- `items_per_rank = 10^4`
- `T_item = 9`（3×3 stencil）

则：

- dense 全量归并：`~10^8` 规模
- dirty 稀疏归并：`~10^5` 规模
- 单次 helper 归并量级约下降 `10^3` 倍
- halo 规划（初始化一次性）：`4 × 10^4 × 9 = ~3.6 × 10^5`，与单次 dense 同步相比可忽略

若 `W_union` 进一步降到 `10^4`，则可达到 `10^4` 倍量级下降。

## 10. 结论

- 降低复杂度最显著的优化是：**dense 全量 Allreduce -> dirty 稀疏同步**（`O(N)` 到 `O(W_union)` 主导）。
- **item-space + pack-map** 将分发/回收从 dense 全量压缩到真实访问集合规模（`O(P*N)` 到 `O(Σ_r L_r)`）。
- **sibling 设备化** 将 helper 级数据搬运从 `N` 量级收紧到 `L` 量级，并与 dirty 稀疏同步形成闭环。
- **全局可覆盖 pack** 与 **PackedElementRef 写路径统一**主要提升语义完备性与结果稳定性，是复杂度优化可持续生效的前提条件。
- **数据驱动 halo 邻居发现**以 `O(P)` 的额外初始化开销，换取了邻居发现的通用正确性，消除了 rank ± 1 假设带来的隐患。
