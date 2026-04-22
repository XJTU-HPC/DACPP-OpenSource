# 组会报告：MPI Wrapper 核心索引冗余消除与性能优化

**背景:**
在早期的 MPI Wrapper 性能测试中，非 stencil 类负载（如 256x256 的矩阵乘法）在 Host 端的调度开销巨大（经常超过 20,000 ms）。Profiler 显示瓶颈死死卡在 `collect_positions_for_item()`，其被疯狂调用（千万级），主要是由于架构设计上对同一计算逻辑进行了多重冗余触发。

本次报告重点拆解两大核心架构冗余，以及我们是如何在代码层面上将其根除的。

## 冗余一：同一个 item 的位置集合被 `pack` 和 `slots` 算两遍

### 1. 之前的问题分析
在旧有逻辑中，当为本地的数据参数建立缓存和地址映射表时，有两个独立且严格解耦的过程：
- 第一步调用 `build_input_pack_map` 计算全局覆盖：这里对所有管辖的 `item` 进行一次遍历，调用 `collect_positions_for_item()` 生成出所有会被访问到的全局坐标，再做 sort+unique 去重，得到 `pack_map`。
- 第二步调用 `build_item_slots` 构建偏移索引用表：再次遍历每个 `item`，**又一次调用了重开销的 `collect_positions_for_item()`** 生成物理坐标，然后拿这些物理坐标去刚刚生成的 `pack.g2l`（Global to Local）哈希表里查询拿到一维内存地址上的数组下标(slot)。

这导致：算法虽然“模块化”得很好，但对于 $N^2$ 个元素，直接让 `collect_positions_for_item()` 被平白无故调用了 $2 \times N^2$ 次。

### 2. 解决方案：`PackPlan` 与动态缓存联合打样
我们打破了这两个步骤的严格串行解耦，引入了全新的联合计算实体 `PackPlan`，同时合并计算与映射：
- **引入按 bind-key 缓存机制**：并不是简单把代码拼在同一个 for 循环中解决复算，而是进一步引入了对“逻辑等同 item”的识别。程序通过 `build_item_bind_key()` 先判断出这个元素实际上索要访问的行列规律序列（bind-key）。
- **建立唯一计算集**：如果这是一个第一次遇到的 bind-key（比如一个新的矩阵行），我们才完整调用 `collect_positions_for_item()` 获取准确绝对位置。第二次碰到需要同样行列模式的 item 时，直接跨过生成。
- **打包与插槽同源分配 (Pack and Slots merged)**：用缓存出来的唯一全局位置快速填充 `pack_map.globals`，随后，利用这些原本已经生成存在的数组快速反填出 `slots`（也就是 GPU 在计算 kernel 时最终查表需要的索引缓冲）。

**结果：** 这个 item 集合不仅不需算两遍，在存在行/列遍历等价的情况下，甚至可以把原本千万次的计算折叠为一小撮低次生成，极大地省下了生成索引的时间。

---

## 冗余二：Root 进程给所有 Rank 重新计算远端 Pack

### 1. 之前的问题分析
Scatter（分发数据给工作 Rank）与 Gather（从工作 Rank 回收数据）时，Root 需要明确：**我该发给 Rank A 哪些坐标的数据？**
在之前的实现里，哪怕远端的所有 Rank 在其本地已经详尽地生成好了它们自己需要的 `local_items -> local_globals` 缓存包(pack)，由于相互的网络不知道细节，Root 为了得知该怎么切分数据，它会在自己主进程写一个可怕的循环：
```cpp
for (int r = 0; r < mpi_size; ++r) {
    auto r_pack = dacpp::mpi::build_input_pack_map(r_range, pattern_vecB); // <--- 问题出在这里
    ... 
```
Root 为**每一个远端节点重新模拟建档**，重算一遍它们应得的包裹... 也就意味着它一个人额外承担了 $MPI\_SIZE \times (单点位置集)$ 庞大次数的超级冗余重算。计算规模越大，Rank 越多，集群扩展性反而变成了一个计算自缚的惩罚效应。

### 2. 解决方案：数据驱动的通信拓扑转换 (Gather before Scatter)
把 Root 的“自算式集权推导”改为“汇报式集市回收”：
- 各自为战：在生成好上述带有加速魔法的 `PackPlan` 之后，所有的工作节点 Rank（包括主节点自己）手上都已经握有了自身确切需求的所有全局地址阵列 `pack.globals`。
- 翻转通信向导：既然从节点已经知道了需要什么，干脆不用 Root 去盲猜重推了。我们让各个 Rank 在通讯开始时，通过 `MPI_Gather / MPI_Gatherv` 将自己的 `pack.globals` 数组（这只是一条干瘪的索引用整数）先一步全部发送给 Root。
- 零计算截取：Root 收到这些带着目的签章的索引后，不再使用任何 Planner (即摒弃 `build_input_pack_map`)进行重算，转而直接调用 `pack_values_by_globals_parallel` 去全尺寸矩阵中进行数值切取与打包。

**结果：** Root 节点的 planner 重算大山被完全移除，伴随而来的是所有 $O(N^2 \times MPI\_SIZE)$ 级别的暴风重算全部被等效分解成为了一个轻量级的 `MPI_Gatherv` 纯索引搜集包通讯，彻底排雷。

## 整体结语
通过梳理上述两次对 Host 冗长工作流程的大规模重剪裁（`PackPlan` + Root Gather/Scatter 翻转架构），再加上最底层的针对 `collect_positions_for_item` 的 Odometer 算法强效降维，MPI Wrapper 规划层面的阻塞已经被根本肃清。系统的准备前导时间耗费断崖式落回到了健康的毫秒级开销谱图内，完全能够真正承载大型科学算例矩阵以及多结点的有效扩容！
