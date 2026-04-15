# DACPP Translator 当前 MPI 主线与优化方式

本文档描述当前仓库里 `--mpi` 这条主线真正做了哪些优化、它的抽象边界是什么，以及哪些更激进的 MPI 优化还没有进入默认路径。

先记住三件事：

- 当前 MPI 回归脚本实际走的是 `--mode=buffer --mpi`
- 当前生成并编译的文件名仍然是 `*.dac_sycl_buffer.cpp`
- 即使识别到了 region，当前主入口 `rewriteMPI_Region()` 也会先回退到稳定的 `rewriteMPI()`，优先保证正确性

## 1. 入口和主线事实

`translator.cpp` 在打开 `--mpi` 后会：

1. 注入 `<mpi.h>`、`MPIPlanner.h` 等头文件
2. 进行一次 buffer region 分析
3. 如果命中 region 入口，则进入 `rewriteMPI_Region()`
4. 否则进入 `rewriteMPI()`
5. 最后统一执行 `rewriteMain()`

需要特别说明的是：

- 当前 `rewriteMPI_Region()` 不是一个独立成熟的 device-resident MPI backend
- 它现在只是一个 region-safe 入口，内部直接委托给稳定的 `rewriteMPI()`

所以今天默认真正生效的 MPI 优化，仍然要以 `Rewriter_MPI.cpp` 里的通用单步 wrapper 逻辑为准。

## 2. 当前主线优化的核心思路

当前 MPI 路径优化的对象，不是几何意义上的空间子域，而是 shell/binding 推导出来的 item space。

也就是说，主线并不是先做：

- 规则 block ownership
- halo exchange
- 邻居通信

而是先做：

1. 根据 shell/binding 建 item space
2. 计算每个 rank 负责的连续 item range
3. 只把这些 item 真正会访问到的元素打包并分发过去
4. 在 rank 本地用压缩后的局部数组执行 SYCL kernel
5. 对写回参数再按全局索引 gather 回 root

这条路径的价值在于：

- 不需要完整复制整块 tensor 到每个 rank
- 不需要为每个样例手写 halo 规则
- 对 non-stencil 和当前已有 stencil 样例都能用同一条 wrapper 兜底

## 3. 当前已经生效的优化层

### 3.1 基于 AST 的真实读写模式推导

当前主线不会再机械地信 shell 参数声明上的 `READ/WRITE/READ_WRITE`。

`inferEffectiveParamModes(...)` 会直接扫描 `calc` 的 AST body，并通过 `ParamAccessVisitor` 统计每个参数的：

- `Reads`
- `UpdateReads`
- `Writes`

它已经覆盖：

- 普通赋值
- 复合赋值
- `++` / `--`
- `CXXOperatorCallExpr` 形式的重载赋值和重载自增自减

这层优化直接减少了两类冗余：

- 把“表面上是 `READ_WRITE`、实际上只读”的参数收紧成 `READ`
- 把“表面上是 `WRITE`、实际上先读后写”的参数纠正成真实模式

对 MPI 来说，这意味着 scatter/gather 只会围绕真实需要的数据流发生，而不是围绕壳子标注做保守搬运。

### 3.2 基于 binding 的统一 item-space 规划

当前 wrapper 会先从 shell 上抽取 split/binding 元信息，构造 `dacpp::mpi::AccessPattern`：

- `collectSplitBindMeta(...)`
- `init_partition_shape(...)`
- `init_bind_split_sizes(...)`

然后把多个参数的 binding split size 合并成统一的 `binding_split_sizes`，进一步得到：

- 全局 `total_items`
- 当前 rank 的 `item_range`

这层优化的关键点是：

- rank 切分的是“逻辑 item 数”，不是“原张量完整线性区间”
- 不同参数可以共享同一套 item-space 规划
- 后续 pack、slot、writeback 都围绕同一个 binding 视图展开

### 3.3 只打包当前 rank 真正需要的数据

拿到 `item_range` 之后，MPI 主线不会直接把整块 tensor scatter 给每个 rank，而是按参数模式选择 pack 策略：

- `build_input_pack_map(...)`
- `build_output_pack_map(...)`
- `build_rw_pack_map(...)`

随后 root rank 会基于 pack map：

- 计算每个远端 rank 真正需要的全局元素索引
- 用 `pack_values_by_globals(...)` 构造紧凑 payload
- 用 `MPI_Scatter` / `MPI_Scatterv` 只发送这部分元素

结果是：

- `WRITE` 纯输出参数不会做无意义的输入 scatter
- `READ` / `READ_WRITE` 参数只会发送用得到的元素，而不是整个 tensor

### 3.4 本地紧凑执行，而不是恢复整块张量

每个 rank 收到 payload 后，不会先把它还原成完整全局张量副本。

当前路径会生成：

- 本地压缩数组 `local_*`
- 槽位映射 `slots_*`
- `dacpp::mpi::View1D<T>` / `View2D<T>`

随后在本地 SYCL `parallel_for` 里，通过 `slots` 把逻辑窗口重新映射到压缩数组：

- `build_item_slots(...)`
- `View1D`
- `View2D`

这层优化的收益是：

- 本地 kernel 仍能按原来的窗口/矩阵访问方式写 `calc`
- 但 rank 内部只为真实访问到的元素分配存储

### 3.5 只对真正写回的参数做 gather/writeback

在 kernel 结束之后，当前 wrapper 只会对 `WRITE` / `READ_WRITE` 参数做回收：

- `build_writeback_values(...)`
- `MPI_Gather`
- `MPI_Gatherv`
- `apply_writeback_by_globals(...)`

这一步不是“每个 rank 直接回传整块局部数组”，而是：

1. 先回传本次需要写回的全局索引
2. 再回传这些索引对应的值
3. root 在 host 侧按索引把结果贴回全局 tensor

因此它避免了：

- 对只读参数做无意义 gather
- 对局部未写区域做整块回传

### 3.6 选择性输出同步

当前 root gather 完成后，不一定会把结果再广播回所有 rank。

主线会用 `TensorUseVisitor` 扫描 `main()` 里该 tensor 是否在 `<->` 之外继续被 host 侧使用，再决定是否执行最终的 `MPI_Bcast`。

这层优化的意义是：

- 如果后续只需要 root 持有 gather 后结果，就不必再做一轮全量广播
- 如果后续 host 逻辑还要在所有 rank 上读取该 tensor，就保留同步

换句话说，当前实现已经不是“写回后永远广播”，而是带 host-use 分析的选择性广播。

### 3.7 非标准 MPI 类型的字节传输兜底

`mpiDatatypeFor(...)` 会先尝试把基础类型映射到标准 MPI datatype。

如果映射不到，主线会回退到 `MPI_BYTE`，并在 `Scatterv/Gatherv` 前把：

- `sendcounts`
- `displs`
- `recvcounts`
- `recvdispls`

统一换算成真实字节数。

这一步解决的是：

- 某些类型没有直接 MPI datatype 时，之前按“元素个数”传输会错位
- 现在可以至少用正确的 byte count 保持数据布局正确

## 4. 这些优化在代码里的生成结构

对每个 `<->` 表达式，当前默认主线仍然生成两层代码。

### 4.1 本地计算函数

形如：

- `xxx_mpi_local(...)`

它直接复用原 `calc` 函数体，只是把参数类型换成 `View1D/View2D` 风格的局部视图。

### 4.2 单步 MPI wrapper

形如：

- `ShellName_CalcName(...)`

它负责：

1. 初始化 `AccessPattern`
2. 规划 `binding_split_sizes` 和 `item_range`
3. root pack
4. `MPI_Scatterv`
5. rank 本地 SYCL 执行
6. `MPI_Gatherv`
7. root writeback
8. 必要时再做广播同步

`rewriteMain()` 另外还会注入：

- `MPI_Init`
- `MPI_Comm_rank / MPI_Comm_size`
- 非 root rank 的 `stdout` 重定向
- `MPI_Finalize`

## 5. 当前已经写出来、但还没有作为默认主线启用的优化雏形

`Rewriter_MPI.cpp` 里其实已经有一套更像 region backend 的生成雏形：

- `MPIRegionTransferPolicy`
- `buildMPIRegionCode(...)`
- `__dacpp_mpi_init_*`
- `__dacpp_mpi_submit_*`
- `__dacpp_mpi_sync_*`

这套代码的方向是：

- 把 init / submit / sync 拆开
- 让局部 buffer 在 region 生命周期内常驻
- `READ/READ_WRITE` 参数只在 init 阶段做 scatter
- `WRITE/READ_WRITE` 参数只在 sync 阶段做 gather

但今天默认入口还没有真的切过去，因为：

- 这条路径还没有取代稳定 wrapper 成为主回归通路
- 当前 `rewriteMPI_Region()` 为了保证 region 样例先能正确编译和运行，仍然直接委托给 `rewriteMPI()`

所以要准确理解当前状态：

- region 优化方向已经有代码雏形
- 但真正生效的默认行为仍然是稳定的单步 wrapper

## 6. 当前还没有做成的 MPI 优化

下面这些能力目前还不属于默认主线：

- 真正的 stencil-aware block ownership
- halo / ghost exchange
- 邻居直接通信替代 root pack/scatter/gather
- 时间推进循环里的分布式状态长期常驻
- 跨时间步的 frontier-only 增量同步

所以当前 stencil 样例之所以能跑，不是因为已经有 halo runtime，而是因为：

- 通用 item-space wrapper 已经足够正确
- AST 真实读写模式推导已经足够准确
- root 仍然可以在每一步把需要的窗口数据打包给各 rank

## 7. 应该怎样理解“现在 MPI 的优化方式”

如果只用一句话概括，当前主线的 MPI 优化方式就是：

- 用 AST 和 binding 把“真正需要访问的数据”收紧成 item-space 下的紧凑 payload
- 用紧凑 payload 做 scatter、本地视图执行、按索引 gather/writeback
- 在不确定或更激进的 region/stencil 优化尚未完全稳定前，始终保留一条 correctness-first 的通用 wrapper 作为主回退路径

这也是当前代码为什么既能覆盖 non-stencil MPI 回归，也能正确跑通现有 stencil 样例，但又还没有进入真正 halo/subdomain 优化阶段的原因。

## 8. 继续看代码时，最值得先读的文件

- `translator.cpp`
- `rewriter/lib/Rewriter_MPI.cpp`
- `rewriter/include/Rewriter.h`
- `dpcppLib/include/MPIPlanner.h`
- `parser/lib/Shell.cpp`

如果只是想跑当前主线回归，直接看：

- `test_mpi.sh`
- `docs/macos_local_sycl_setup.md`
