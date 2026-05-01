# Compact Slots 优化组会汇报

## 1. 背景：为什么要做这个优化

这次优化针对的是 DACPP MPI 翻译路径里的一个数据准备瓶颈。

在 MPI 版本中，一个 DACPP 算子执行前，translator 会先根据 shell 里的划分规则，为每个 rank 准备它需要访问的数据。这个过程不只是把数据 scatter 下去，还要构造一套索引映射，让 kernel 里每个 work-item 能通过 `View1D` 或 `View2D` 找到自己对应的数据片段。

问题出现在矩阵乘法这类访问模式上。以矩阵乘法为例，每个输出元素 `C[i][j]` 都需要读取：

```cpp
A[i][:]
B[:][j]
C[i][j]
```

从计算角度看，每个 rank 只负责一部分输出元素，局部计算量是可控的。但在数据准备阶段，旧实现会为每一个 item 都展开一份完整的 slots 映射。对于 `A[i][:]` 来说，同一行会被很多不同的 `j` 重复使用；对于 `B[:][j]` 来说，同一列会被很多不同的 `i` 重复使用。旧实现没有充分利用这种重复性，而是把相同的索引范围反复复制。

结果就是：计算还没开始，光是构造索引映射就已经非常重。

神威上的 profiling 数据能很直观地说明这个问题。矩阵规模为 `256 × 256`、32 个 rank 时，即使注释掉真正的计算，数据准备阶段也会花 20 秒以上：

```text
总执行时间(所有 rank 最大值): 23804.6 ms
collect_positions_for_item 总调用次数(所有 rank 求和): 524288
collect_positions_for_item 总耗时(所有 rank 求和): 46467.2 ms
collect_positions_for_item 最大 rank 耗时: 16279.7 ms
```

这说明瓶颈并不在 SYCL kernel 的计算本身，而是在 kernel 之前的 MPI 数据规划和索引表构建。

## 2. 原来的 slots 是怎么工作的

MPI wrapper 会先为每个参数构造一个 pack plan。这个 plan 里有两类信息：

- 当前 rank 需要哪些全局数据位置
- kernel 里每个 item 应该访问这些局部数据中的哪些 slot

旧实现里，slots 的组织方式非常直接：每个 item 都有一段连续的 slots，长度等于这个 item 访问的数据片段大小，也就是 `partition_size`。

kernel 中构造 view 时，大致是这样：

```cpp
View1D<T>{data, slots, item_linear * partition_size}
```

这意味着第 `item_linear` 个 work-item 的 slots 起点就是：

```cpp
item_linear * partition_size
```

这种设计简单，但它隐含了一个问题：每个 item 都必须在 `slots` 里有一份自己的完整映射。

对于普通 vector add 这种 `v[i]` 访问，`partition_size = 1`，所以问题不大。但对于矩阵乘法里的 `A[i][:]`，每个 item 访问一整行，`partition_size = N`。总 item 数是 `N²`，于是 slots 大小就会变成：

```text
N² × N = O(N³)
```

矩阵规模一大，这个数组会非常夸张。

比如 `N = 2048` 时，单是 `A[i][:]` 这一个参数的 slots 就接近：

```text
2048 × 2048 × 2048 × sizeof(int32_t)
≈ 32 GB
```

这还只是一个参数。如果再加上 `B[:][j]` 和输出参数，数据准备阶段的内存和时间都会被放大。

## 3. 根因：重复 item key 没有被压缩到底

旧代码其实已经做了一部分去重：它会根据 bind key 判断哪些 item 的访问模式是一样的。

例如矩阵乘法中，`A[i][:]` 对所有相同的 `i` 来说访问的是同一行。也就是说：

```text
C[i][0], C[i][1], C[i][2], ...
```

这些 item 对 `A` 的访问位置完全相同。理论上这一行的 slots 只需要存一份。

但是旧实现的问题是：虽然前面识别出了 unique key，最后仍然把每个 item 对应的 slots 重新展开了一遍。也就是说：

```text
第 0 行 slots 复制 N 次
第 1 行 slots 复制 N 次
...
第 N-1 行 slots 复制 N 次
```

所以去重只做了一半。positions 的计算减少了，但 slots 的最终存储仍然膨胀成 O(N³)。

这就是 Compact Slots 优化要解决的核心问题。

## 4. 核心思路：slots 只存 unique key，item 只存 offset

Compact Slots 的想法很直接：既然很多 item 的访问模式一样，就不要为每个 item 都复制一份 slots。

我们把原来的一个大 slots 数组拆成两个数组：

```text
compact_slots
item_key_offsets
```

其中：

- `compact_slots` 只保存每个 unique key 对应的 slot range
- `item_key_offsets` 保存每个 item 应该从 `compact_slots` 的哪个位置开始读

这样 kernel 构造 view 时，就从原来的：

```cpp
View1D<T>{data, slots, item_linear * partition_size}
```

变成：

```cpp
View1D<T>{data, compact_slots, item_key_offsets[item_linear]}
```

`View1D` 和 `View2D` 本身不用改。因为它们本来就只需要一个 slots 指针和一个起始 offset。我们只是把 offset 的来源从“按 item 线性计算”改成“查表”。

这个改动的好处是，重复的 slots 不再复制。

## 5. 用矩阵乘法直观看收益

还是看 `N = 2048` 的矩阵乘法。

对于参数 `A[i][:]`：

- item 数量是 `N²`
- 每个 item 访问一整行，`partition_size = N`
- 但 unique key 只有 `N` 个，因为只有 `N` 行

旧实现：

```text
slots 大小 = N² × N = O(N³)
```

新实现：

```text
compact_slots 大小 = N × N = O(N²)
item_key_offsets 大小 = N² = O(N²)
```

用内存估算就是：

| 项目 | 改前 | 改后 |
| --- | --- | --- |
| slots | 约 32 GB | 无 |
| compact_slots | 无 | 约 16 MB |
| item_key_offsets | 无 | 约 16 MB |
| 合计 | 约 32 GB | 约 32 MB |

这就是数量级上的变化：从 GB 级降到 MB 级。

更重要的是，这不仅减少内存，也减少了大量重复构造 slots 的时间。

## 6. 这个优化不只影响 2D

虽然最明显的问题出现在矩阵乘法这种二维访问里，但 Compact Slots 不是只针对 2D。

一维场景里，如果出现 broadcast 式访问，也会有类似问题。例如：

```cpp
bias[{}]
x[{}]
```

如果 `{}` 表示整个向量，那么每个 item 都可能访问同一份完整向量。旧 slots 会变成：

```text
item 数 × 向量长度
```

而实际上所有 item 访问的都是同一个完整向量，unique key 只有一个。Compact Slots 后，只需要保存一份完整向量的 slots，再让每个 item 的 offset 指向同一个位置。

所以这个方案是统一的：不管是一维还是二维，只要存在重复访问模式，就能减少 slots 展开。

## 7. 具体实现改了什么

实现上主要改了两层。

第一层是在运行时 pack plan 结构里，把旧的 `slots` 改成：

```cpp
std::vector<int32_t> compact_slots;
std::vector<int32_t> item_key_offsets;
```

构建 pack plan 时，不再为每个 item 追加一整段 slots，而是：

1. 先按 bind key 收集 unique positions。
2. 每个 unique key 只生成一次 slot range，放进 `compact_slots`。
3. 对每个 item 记录它对应的 key 在 `compact_slots` 里的起始位置，放进 `item_key_offsets`。

第二层是在 MPI wrapper 的代码生成里，把 kernel 侧的 view 构造改成查 offset 表。

旧生成逻辑相当于：

```cpp
view_x{data_x, slots_x, item_linear * x_partition_size}
```

新生成逻辑变成：

```cpp
view_x{data_x, compact_slots_x, key_offsets_x[item_linear]}
```

同时，生成代码里会多创建一个 `key_offsets_buffer`，在 kernel 中读取 `key_offsets[item_linear]`。

这个改动非常小，但它把 slots 的规模从“按 item 完整展开”变成了“按 unique key 存一份”。

## 8. 调用次数为什么也下降

原来的 profiling 里，`collect_positions_for_item` 调用了 524,288 次。这个函数负责根据 item id 和访问模式计算该 item 需要访问哪些全局位置。

旧流程中，很多重复的 item 会反复调用它。例如矩阵乘法中，同一行的 `A[i][:]` 会被不同的 `j` 重复计算很多次。

Compact Slots 后，只有新的 bind key 第一次出现时才需要真正收集 positions。后续 item 如果 key 相同，就直接复用已有结果。

以 `256 × 256`、32 rank 为例，理论调用次数可以从：

```text
524K
```

降到大约：

```text
82K
```

这解释了为什么优化不仅省内存，也能减少数据准备时间。

## 9. 正确性为什么不变

这个优化不改变每个 item 最终访问的数据，只改变 slots 的存储方式。

旧方式是：

```text
每个 item 一份 slots
```

新方式是：

```text
相同访问模式的 item 共用一份 slots
每个 item 用 offset 找到自己对应的位置
```

如果所有 item 的访问模式都不同，那么：

```text
compact_slots 等价于旧 slots
item_key_offsets[i] = i * partition_size
```

这时行为和旧实现完全一致。

如果多个 item 的访问模式相同，它们本来就应该访问同一组数据，所以共享同一段 slots 是安全的。

因此，这个优化是存储布局优化，不是语义变化。

## 10. 验证方式

验证可以分三步。

第一步，看生成代码里是否已经不再使用旧的 `plan_x.slots`，而是使用：

```cpp
plan_x.compact_slots
plan_x.item_key_offsets
```

第二步，跑本地和 MPI 回归，确认结果不变。

```bash
cd /Volumes/QUQ/working/dacpp/clang/tools/translator
bash test_local.sh
bash test_mpi.sh
```

第三步，在神威上重复矩阵乘法 profiling，重点看：

- `collect_positions_for_item` 调用次数是否下降
- 数据准备阶段耗时是否下降
- slots 相关内存是否不再爆炸

预期对于 `256 × 256, 32 rank` 的矩阵乘法，`collect_positions_for_item` 调用次数会从约 `524K` 降到约 `82K`；对于更大的矩阵，slots 内存会从 O(N³) 降到 O(N²)。

## 11. 总结

Compact Slots 优化解决的是 MPI 路径里一个典型的“索引表重复展开”问题。

旧实现虽然已经知道很多 item 的访问模式重复，但最后仍然把每个 item 的 slots 都完整复制一份，导致矩阵乘法这类场景出现 O(N³) 的 slots 膨胀。

新实现把 slots 拆成 `compact_slots` 和 `item_key_offsets`：重复访问模式只存一份 slots，每个 item 只保存一个 offset。这样既不改变 kernel 里的访问语义，又能显著降低内存占用和数据准备时间。

从效果上看，这个优化让 MPI wrapper 的数据准备更接近真实计算需求：计算是局部的，索引结构也应该是压缩和局部友好的，而不是在计算前先构造一个远大于实际必要规模的展开表。
