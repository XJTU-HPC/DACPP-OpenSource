# 面向 AccessPattern 读者的 Operator-Resident 访问模型说明

这篇文档面向已经理解旧 `AccessPattern` 模型的读者，目标是解释当前
operator-resident，也就是 OR，MPI 翻译模型到底如何理解访问、划分算子、
划分 MPI 数据，以及为什么它看起来没有 `AccessPattern` 那么直观，但更适合
生成高性能 MPI 代码。

一句话概括：

- `AccessPattern` 从计算项出发，问：这个 item 会访问哪些全局位置？
- OR 从输出所有权出发，问：这个输出应该由哪个 rank 负责计算？负责这个输出
  的 rank 需要哪些输入数据？

两者都可以把计算分配给 MPI rank。真正的区别在于，OR 把**输出所有权**和
**resident 状态**保留下来作为一等语义，而 `AccessPattern` 更多是在操作已经
展开后的 global position 列表。

## 1. AccessPattern 的心智模型

旧模型是 item-centric，也就是以计算 item 为中心。

对每个 DAC 表达式，旧 wrapper 会给每个参数构造一个 `AccessPattern`。然后
大致执行下面这些步骤：

1. 根据 bind split size 计算全局 `total_items`。
2. 每个 rank 拿一段连续的 item id。
3. 对每个本地 item id，反推出它的 bind 坐标。
4. 根据每个参数的访问模式，枚举这个 item 需要访问的全局 tensor 位置。
5. 对这些全局位置去重，构造 `PackPlan`。
6. root 端按 global positions 打包数据。
7. `Scatterv` 把 packed buffer 分发给各 rank。
8. rank 上运行本地 kernel。
9. 写回数据通过 `Gatherv` 回到 root，必要时再广播。

所以 `AccessPattern` 的直觉很清楚：

```text
item id -> bind 坐标 -> global positions -> packed local slots
```

例如一个一维向量加法：

```cpp
lhs[i], rhs[i], out[i]
```

对于 item `i = 5`，访问就是：

```text
lhs[5], rhs[5], out[5]
```

再例如一个 stencil：

```cpp
state[sp], next[idx]
binding(idx, sp)
split sp(3, 1)
```

对于 item `idx = 5`，访问可能是：

```text
state[5], state[6], state[7], next[5]
```

这就是旧模型直观的原因：它直接告诉你每个 item 访问了哪些全局位置。

但这个模型的弱点也在这里。它展开之后主要看到的是：

```text
global index 列表
slot 映射
pack buffer
writeback 列表
```

它不太保留高层语义，比如：

```text
这个 rank 天然拥有 out[0..1024)
这个中间结果后面还会被同一个 rank 继续读
这个 stencil 只需要和邻居交换 halo
```

这些信息在高性能 MPI 里很关键。

## 2. OR 的心智模型

OR 是 owner-centric，也就是以输出所有者为中心。

它不会先枚举每个 item 访问哪些 global positions，而是先识别写入域：

```text
哪个 tensor 被写？
写入维度由哪些 index/bind 定义？
这个输出域的每一段应该由哪个 rank 拥有？
```

然后再从输出所有权反推输入需求：

```text
为了计算自己拥有的输出区间，这个 rank 需要哪些输入切片？
```

所以 OR 的基本流程是：

```text
Shell/Calc 访问关系
        |
        v
输出 ownership domain
        |
        v
每个 rank 的本地输出区间
        |
        v
本地输入切片 / payload / scalar / halo
        |
        v
在 resident 或 rank-local buffer 上运行本地 kernel
        |
        v
只有语义需要时才 materialize
```

可以这样记：

```text
AccessPattern：枚举每个 item 需要的数据。
OR：先分配每段输出归谁算，再把这段输出需要的数据交给对应 rank。
```

## 3. OR 中的 rank ownership

OR 最常用的分配原语是均衡连续区间：

```text
base  = total / mpi_size
rem   = total % mpi_size
count = base + (rank < rem ? 1 : 0)
begin = rank * base + min(rank, rem)
```

如果 `N = 8`，`mpi_size = 2`，那么：

```text
rank 0 owns [0, 4)
rank 1 owns [4, 8)
```

对一维输出，这个区间就是输出元素区间。

对二维 row-block layout，OR 通常对行做同样的划分：

```text
rank 0 owns rows [0, r0)
rank 1 owns rows [r0, r1)
...
```

每个 rank 本地保存自己拥有的完整行块。

关键点是：**ownership 来自输出参数，而不是每个输入参数各自独立决定。**

输入参数只是被映射到这个输出 ownership 上。

## 4. 例子一：一维链式算子

看一个链式例子：

```cpp
shell dacpp::list VADD(const Vector<float>& lhs,
                       const Vector<float>& rhs,
                       Vector<float>& out) {
    dacpp::index i;
    return {lhs[i], rhs[i], out[i]};
}

shell dacpp::list VSHIFT(const Vector<float>& in,
                         const Vector<float>& bias,
                         Vector<float>& out) {
    dacpp::index i;
    return {in[i], bias[{}], out[i]};
}

VADD(a, b, tmp)       <-> vadd;
VSHIFT(tmp, bias, y)  <-> vshift;
VADD(y, c, out)       <-> vadd;
```

假设 `N = 8`，`mpi_size = 2`。

### 第一个算子

第一个算子写：

```text
tmp[i]
```

所以 OR 用 `i` 定义输出所有权：

```text
rank 0 owns tmp[0..3]
rank 1 owns tmp[4..7]
```

输入也是同一个 `i`：

```text
a[i], b[i]
```

因此 OR 分发匹配的输入切片：

```text
rank 0 gets a[0..3], b[0..3]
rank 1 gets a[4..7], b[4..7]
```

每个 rank 在本地运行相同的 `vadd` kernel：

```text
rank 0 computes tmp[0..3]
rank 1 computes tmp[4..7]
```

这里最重要的一点是：如果后续 DAC 表达式还会以同样的 resident layout 消费
`tmp`，OR 不会立刻把 `tmp` gather 回 root。

OR 会记录：

```text
tmp is resident
rank 0 owns tmp[0..3]
rank 1 owns tmp[4..7]
```

### 第二个算子

第二个算子写：

```text
y[i]
```

ownership 还是同一个 `i`：

```text
rank 0 owns y[0..3]
rank 1 owns y[4..7]
```

它的输入 `tmp[i]` 已经 resident 在相同 rank 上：

```text
rank 0 reads resident tmp[0..3]
rank 1 reads resident tmp[4..7]
```

`bias[{}]` 是标量，OR 把它复制或广播给所有 rank。

然后每个 rank 计算：

```text
rank 0 computes y[0..3]
rank 1 computes y[4..7]
```

同样，`y` 也可以继续 resident。

### 第三个算子

第三个算子写：

```text
out[i]
```

OR 继续使用同样的 ownership：

```text
rank 0 owns out[0..3]
rank 1 owns out[4..7]
```

它读取 resident 的 `y[i]`，并且按相同区间 scatter `c[i]`：

```text
rank 0 reads y[0..3], c[0..3]
rank 1 reads y[4..7], c[4..7]
```

只有当后续 host 代码需要看到完整输出时，OR 才 materialize：

```text
rank 0 sends out[0..3]
rank 1 sends out[4..7]
root assembles out[0..7]
```

所以完整通信链路是：

```text
scatter a,b
kernel VADD -> tmp resident

broadcast bias
kernel VSHIFT -> y resident

scatter c
kernel VADD -> out resident

final gather out if host-visible
```

而 `AccessPattern` 更像是把每个 wrapper 都独立处理：

```text
build positions -> pack -> scatter -> kernel -> gather/writeback
```

OR 的核心改进就是：中间结果的 ownership 事实可以跨算子保留下来。

## 5. 例子二：带 binding 的 stencil window

再看一个一维 stencil：

```cpp
shell dacpp::list HALO(Vector<double>& state,
                       Vector<double>& next) {
    dacpp::index idx;
    dacpp::split sp(3, 1);
    binding(idx, sp);
    return {state[sp], next[idx]};
}

calc void step(Vector<double>& state, double* next) {
    next[0] = 0.25 * state[0]
            + 0.50 * state[1]
            + 0.25 * state[2];
}
```

`binding(idx, sp)` 表示 `idx` 和 stencil split `sp` 属于同一个逻辑坐标系。

OR 先看输出：

```text
next[idx]
```

所以 rank ownership 由 `idx` 决定：

```text
rank 0 owns next[0..3]
rank 1 owns next[4..7]
```

输入是：

```text
state[sp]
```

因为 `sp` 绑定到了 `idx`，每个 rank 需要的是自己拥有的输出区间对应的 state
window。

如果 `sp(3, 1)` 表示三点窗口，那么：

```text
next[0] needs state[0], state[1], state[2]
next[1] needs state[1], state[2], state[3]
next[2] needs state[2], state[3], state[4]
next[3] needs state[3], state[4], state[5]
```

所以 rank 0 拥有的输出是：

```text
next[0..3]
```

但它可能需要的输入是：

```text
state[0..5]
```

这里多出来的数据就是 halo/window 数据。

注意，输出 ownership 仍然只是：

```text
next[0..3]
```

额外的 `state` 值只是为了计算这些 owned output 而携带的只读支持数据。

如果这个 stencil 在循环中，OR 的 loop-lowered resident halo path 会把它变成：

```text
init:
  scatter owned state plus halo

each loop step:
  compute owned next locally
  exchange boundary halo with neighbor ranks
  keep state/next resident

exit:
  gather final owned state only if required
```

这就是 OR 比 `AccessPattern` 更接近手写 MPI 的地方：rank 拥有一个空间子域，
循环中只和邻居交换边界 halo，而不是每一轮都回到 root 做全局 pack/gather。

## 6. binding 在 OR 中到底意味着什么

`binding` 并不是直接说 MPI rank 怎么切。

`binding` 说的是：

```text
这些 index/split 维度表示同一个逻辑坐标。
```

OR 用写入输出决定 rank ownership，然后用 binding 把输入维度映射到这个
ownership 上。

例如：

```cpp
binding(idx, sp);
state[sp], next[idx]
```

含义是：

```text
next[idx] 定义 owner domain
state[sp] 按同一个逻辑坐标读取
```

二维例子：

```cpp
binding(sp1, idx1);
binding(sp2, idx2);
matIn[sp1][sp2], matOut[idx1][idx2]
```

OR 会形成两个 bind domain：

```text
bind0: sp1 <-> idx1
bind1: sp2 <-> idx2
```

输出 `matOut[idx1][idx2]` 定义二维 owner domain。当前 row-block layout 通常按
第一个 bind domain，也就是行方向，切给不同 rank：

```text
rank 0 owns a range of output rows
rank 1 owns the next range of output rows
...
```

输入 `matIn[sp1][sp2]` 会被映射成对应的行块，再加上 stencil 所需的 halo 行。

## 7. OR 如何分类参数

OR 识别输出 ownership 后，会把每个参数相对于这个 ownership 分类。

常见分类如下：

| 访问形状 | OR 中的解释 |
| --- | --- |
| `out[i]` | direct output，定义或匹配 ownership |
| `in[i]` | direct input，和 ownership 匹配 |
| `bias[{}]` 标量 | replicated scalar |
| `x[{}]` 完整向量或矩阵 | replicated full tensor，满足 guard 时接受 |
| `A[idx][{}]` | 每个 owned `idx` 需要一整行 payload |
| `A[{}][idx]` | 每个 owned `idx` 需要一整列 payload |
| `state[sp]` 绑定到 `next[idx]` | owned output 范围上的 stencil window |
| fixed block split | fixed-block ownership，并带额外对齐检查 |

正是因为有这些分类，OR 才能生成 layout-specific 通信，而不是统一退回
`PackPlan`。

## 8. 为什么 OR 看起来不如 AccessPattern 直观

`AccessPattern` 直观，因为它会给出具体列表：

```text
item 5 touches global positions [5, 6, 7]
```

OR 通常不把这个列表显式写出来。它保留的是更高层的事实：

```text
rank 1 owns outputs [4, 8)
rank 1 needs a 3-wide state window for those outputs
```

这个事实没有 global-position 列表那么具体，但它更适合优化。它让编译器可以：

- 避免为规则 layout 构造大量 global-index metadata
- 在 kernel 中使用直接的本地 offset
- 让中间 tensor 跨算子保持 resident
- 把循环中的重复 gather/scatter 换成邻居 halo exchange
- 推迟 materialization，直到 host-visible 语义真的需要

所以 OR 的“不直观”来自它没有完全展开访问列表；但它的性能优势也来自同一个
地方：它保留了更高层的 ownership 和 residency。

## 9. Side-by-side 对比

| 问题 | AccessPattern 的回答 | OR 的回答 |
| --- | --- | --- |
| MPI rank 切分什么？ | 抽象 item id | 输出 ownership domain |
| 输入数据怎么找？ | 对每个 item 枚举 global positions | 从 owned output 推导输入 slice/payload/halo |
| 通信什么？ | packed global-position buffers | layout-specific slices、payload、scalar、halo |
| kernel 看到什么？ | compact slots 和 item offsets | resident 或 rank-local buffer 上的直接 local view |
| 算子结束后发生什么？ | 通常为该 wrapper gather/writeback | 输出可以留在 resident state 给后续算子 |
| 正确性依据是什么？ | 显式 global-position expansion | layout proof 加 ownership/materialization contract |
| 为什么规则形状更快？ | 通常仍有 metadata/pack 开销 | 保留高层 layout，避免不必要 materialization |

## 10. 一句话讲给 AccessPattern 读者

可以这样说：

```text
AccessPattern 展开“每个 item 访问什么”；
OR 保留“每段输出归谁拥有”，并让这个 ownership 跨算子和跨循环存活。
```

后面的 resident chain、halo exchange、materialize 消除，都是从这个区别自然推出来
的。

