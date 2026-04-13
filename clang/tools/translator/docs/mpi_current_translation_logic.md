# DACPP Translator 当前 MPI 主链路说明

本文档描述当前仓库里 `--mpi` 实际生成代码时走到的主链路，以及这条链路今天已经覆盖到的能力边界。

当前结论可以先概括成一句话：

- `--mpi` 会统一走 `rewriteMPI()` 这条链路
- 这条链路对 non-stencil 样例已经完成回归验证
- stencil 样例也会走这条链路，但它们需要的 halo / 子域语义还没有单独建模

如果要看当前实现，应优先看：

- [translator.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/translator.cpp)
- [Rewriter_MPI.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/Rewriter_MPI.cpp)
- [MPIPlanner.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/MPIPlanner.h)
- [Shell.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/parser/lib/Shell.cpp)
- [Rewriter.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/include/Rewriter.h)

归档里的 MPI 模板实现仍然保留在：

- [mpi_template.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/archive/rewriter/include/mpi_template.h)
- [mpi_template.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/archive/rewriter/lib/mpi_template.cpp)

但当前主入口使用的是本文描述的这套实现。

---

## 1. 入口与生成流程

当用户传入 `--mpi` 时，`translator.cpp` 会：

1. 给待生成文件补充 MPI 相关头文件
2. 调用 `rewriter->rewriteMPI()`
3. 再统一调用 `rewriteMain()`

测试脚本通常使用：

- `--mode=usm --mpi`

因此输出文件名通常是：

- `_sycl_usm.cpp`

当前 MPI 生成代码里的本地执行层使用的是：

- `sycl::buffer`
- 压缩后的线性本地数组
- slot 映射
- `View1D / View2D`

---

## 2. 生成结果的结构

对每个 `<->` 表达式，当前重写器会生成两层代码。

### 2.1 本地计算函数

形如：

- `xxx_mpi_local(...)`

特点：

- 直接复用原 `calc` 的函数体
- 通过替换形参类型来保留原有索引写法
- 参数会变成 `dacpp::mpi::View1D<T>` 或 `dacpp::mpi::View2D<T>`

### 2.2 MPI wrapper

形如：

- `ShellName_CalcName(...)`

wrapper 负责：

- 构造每个参数的 `AccessPattern`
- 计算 item 空间和每个 rank 的 item 区间
- 在 Root 上打包输入数据
- 用 `MPI_Scatter` / `MPI_Scatterv` 分发 payload
- 在各 rank 上提交本地 SYCL kernel
- 对 `WRITE / READ_WRITE` 参数收集写回值
- 在 Root 上写回 tensor
- 按当前同步策略决定是否广播更新后的输出 tensor

### 2.3 `main()` 中的 MPI 生命周期

`rewriteMPI()` 还会向 `main()` 注入：

- `MPI_Initialized`
- 条件性的 `MPI_Init`
- `MPI_Comm_rank / MPI_Comm_size`
- 非 0 号 rank 的 `stdout` 重定向
- return 前或函数末尾的 `MPI_Finalize`

---

## 3. shell 绑定信息怎样进入 planner

### 3.1 `Shell.cpp` 中的 binding 图

parser 会把 shell 里的 split / index 和 `binding()` 关系建成图。

这张图里的关键语义是：

- 顶点表示 split / index 这类符号
- 边上可以带 offset 表达式
- `Shell::GetBindInfo(...)` 会按连通分量遍历，并把路径上的 offset 串接起来

### 3.2 `collectSplitBindMeta(...)` 输出 planner 元数据

MPI 重写阶段会把这份 parser 结果整理成：

- `bind_set_id`
- `bind_offset_expr`

对于没有出现在 binding 图里的 split，当前实现会：

- 分配 fallback bind id
- 使用 `"0"` 作为默认 offset

---

## 4. item 空间与 `AccessPattern`

### 4.1 item 空间的含义

当前 MPI 主链路先切的是：

- shell / binding 定义出来的 item 空间

item 由 shell 中的 split / index 组合定义，是这条主链路里的基本工作单元。

例如：

```cpp
dacpp::list dataList{matA[idx1][{}], matB[{}][idx2], matC[idx1][idx2]};
```

这里的工作单元由：

- `idx1`
- `idx2`

共同决定。

### 4.2 `AccessPattern` 的关键字段

每个 shell 参数都会被翻译成一个 `AccessPattern`。当前最关键的字段有：

- `data_info`
- `param_ops`
- `partition_shape`
- `bind_set_id`
- `bind_offset_expr`
- `is_index_op`
- `bind_split_sizes`
- `mode`

### 4.3 统一 item 空间的大小

每个 pattern 初始化后会得到自己的：

- `bind_split_sizes`

wrapper 会把所有参数的 `bind_split_sizes` 做逐维 `max` 合并，得到统一 item 空间。

然后：

- `total_items = 所有 binding 维度 split size 的乘积`

每个 rank 用：

- `get_rank_item_range(total_items, rank, mpi_size)`

拿到一段连续 item id 区间。

---

## 5. 一个 item 怎样变成位置集合

`MPIPlanner.h` 里的展开过程可以直接理解成四步：

1. 用 `decode_item_id(...)` 把 `item_id` 解码成各 binding 分量的索引
2. 结合 `param_ops + bind_set_id + bind_offset_expr` 计算 base position
3. 用 `partition_shape` 展开这个 item 覆盖到的全部元素
4. 把这些元素统一线性化成全局 `global_idx`

因此当前 planner、打包和写回都围绕下面这类对象工作：

- 全局线性下标集合

---

## 6. offset 求值能力

`bind_offset_expr` 以字符串形式存储在 `AccessPattern` 里，但 `eval_bind_offset_expr(...)` 当前只支持：

- 纯整数
- 只含 `+` / `-` 的简单常数表达式

例如：

- `"0"`
- `"1"`
- `"-1"`
- `"1-2+3"`

带有下列成分的表达式当前不会被真正求值：

- 变量名
- 更复杂的符号表达式
- 函数调用

这意味着当前 planner 的 offset 能力适合常数偏移场景。

---

## 7. Root 打包与 MPI 分发

### 7.1 pack builder 的选择

当前会按参数模式选择：

- `READ` -> `build_input_pack_map(...)`
- `WRITE` -> `build_output_pack_map(...)`
- `READ_WRITE` -> `build_rw_pack_map(...)`

当前 `build_output_pack_map(...)` 和 `build_rw_pack_map(...)` 的核心仍然建立在输入位置集合之上，再标记对应的 writeback 信息。

### 7.2 输入参数的分发流程

对需要输入 copy-in 的参数，Root 会：

1. `tensor2Array(global_xxx)`
2. 为每个 rank 构造该 rank 的 `r_pack`
3. 用 `pack_values_by_globals(global_xxx, r_pack.globals)` 生成 payload
4. 先用 `MPI_Scatter` 分发 count
5. 再用 `MPI_Scatterv` 分发真正数据

### 7.3 `WRITE` 参数的本地布局

对纯 `WRITE` 参数，当前 wrapper 会：

- 构造 output 对应的 `PackMap`
- 分配本地 `local_xxx`
- 分配本地 `slots_xxx`

这类参数不做输入 `Scatterv`，本地数组只承载该参数负责写入的位置集合。

### 7.4 MPI datatype 的选择

当前会优先映射到标准 MPI datatype。

无法直接映射的类型会退回：

- `MPI_BYTE`

同时 count 会按 `sizeof(T)` 放大。

---

## 8. rank 本地看到的数据形态

每个 rank 本地拿到的是两类对象。

### 8.1 `local_xxx`

压缩后的线性数组，特点是：

- 只包含本 rank 实际访问到的全局元素
- 重复元素会先去重

### 8.2 `slots_xxx`

局部映射表，作用是：

- 把当前 item 的逻辑元素位置映射到 `local_xxx` 的槽位

对应逻辑是：

- `build_item_slots(item_range, pattern, pack)`

---

## 9. calc 形参与本地执行模型

### 9.1 保留原函数体

当前策略是：

- 保留原 `calc` 函数体
- 用 view 类型承接本地数据访问

### 9.2 `View1D / View2D`

`inferViewRank(...)` 当前按下面规则决定形参 view：

- 裸指针 / `Vector<...>` -> 1D
- `Matrix<...>` -> 2D

所以本地 kernel 会临时构造：

- `dacpp::mpi::View1D<T>`
- `dacpp::mpi::View2D<T>`

并把它们传给 `xxx_mpi_local(...)`

### 9.3 当前 view 层覆盖范围

当前本地执行层正式提供的是：

- `View1D`
- `View2D`

因此这条主链路暴露给 `calc` 的访问层目前覆盖到 1D / 2D。

### 9.4 SYCL kernel 结构

wrapper 会把：

- `local_xxx`
- `slots_xxx`

放进：

- `sycl::buffer<T, 1>`
- `sycl::buffer<int32_t, 1>`

然后按 `local_item_count` 提交一个线性 `parallel_for`。

每个 work-item 会：

1. 计算自己对应的 item 线性编号
2. 构造 `view_xxx`
3. 调用 `xxx_mpi_local(view_a, view_b, ...)`

---

## 10. 参数访问模式

当前 `inferEffectiveParamModes(...)` 的判定规则是：

1. 先以 shell 上的 `READ / WRITE / READ_WRITE` 标注为基线
2. 再用 `ParamAccessVisitor` 检查 calc body
3. 收窄范围只作用于原本标成 `READ_WRITE` 的参数

因此当前行为是：

- shell 标成 `READ_WRITE` 的参数，会进一步收敛到真实的 `READ` / `WRITE` / `READ_WRITE`
- shell 标成 `READ` 的参数，保持 `READ`
- shell 标成 `WRITE` 的参数，保持 `WRITE`

这条规则决定了当前 MPI 主链路仍然依赖 shell 标注本身的正确性。

---

## 11. 写回与输出同步

### 11.1 写回流程

对 `WRITE / READ_WRITE` 参数，当前流程是：

1. 本地用 `build_writeback_values(local_xxx, pack_xxx)` 提取写回值
2. `MPI_Gather` 回收各 rank 的 send_count
3. `MPI_Gatherv` 回收全局下标
4. `MPI_Gatherv` 回收对应值
5. Root 上用 `apply_writeback_by_globals(...)` 写回完整 host array
6. 再 `array2Tensor(...)` 回写 tensor

### 11.2 输出同步判定

输出同步会使用 `needsBcast`。

当前实现里：

1. 先读取 `dacppFile->getMPIBroadcastOutputs()`
2. 如果 `mainBody` 存在，就运行 `TensorUseVisitor`
3. 最终 `needsBcast` 使用 `visitor.NeedsBcast`

在通常存在 `main()` 的程序里，broadcast 判定由 host 侧后续使用分析主导。

---

## 12. `rewriteMain()` 的控制流语义

`rewriteMain()` 的作用是：

- 把 `<->` 表达式替换成 wrapper 调用
- 保留宿主侧控制流结构

当检测到包裹 `<->` 的最外层 `for` 时，正常链路下会保留该循环，只替换其中的 `<->`。

如果文件通过 `DACPP_TRANSLATE_MODE` 把 `mode` 设成 `1`，则会走简单替换分支。

---

## 13. 当前能力边界

这条 MPI 主链路当前覆盖的是：

- item-space planner
- Root pack / scatter
- rank 本地压缩数组与 slot 映射
- `View1D / View2D`
- Root gather / writeback / 输出同步

当前还没有单独建模的能力包括：

- 基于子域所有权的 stencil domain decomposition
- rank 邻接 halo exchange
- 时间步之间持久化 ghost region
- 高维 view 层
- 完整的符号 offset 求值

---

## 14. 当前验证状态

我在 2026-04-13 重新跑了：

```bash
cd /Volumes/QUQ/working/dacpp/clang/tools/translator
bash test_mpi.sh
```

结果是：

- `10 tests | 10 passed | 0 failed | 0 skipped`

这说明当前主链路对 10 个 non-stencil MPI 样例已经实测通过。

---

## 15. 调试优先级

如果继续排 MPI 问题，建议优先看这些点：

1. [Shell.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/parser/lib/Shell.cpp)
   `GetBindInfo(...)`

2. [Rewriter_MPI.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/Rewriter_MPI.cpp)
   `collectSplitBindMeta(...)`

3. [Rewriter_MPI.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/Rewriter_MPI.cpp)
   `buildPatternInitCode(...)`

4. [MPIPlanner.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/MPIPlanner.h)
   `eval_bind_offset_expr(...)`

5. [MPIPlanner.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/MPIPlanner.h)
   `collect_positions_for_item(...)`

6. [MPIPlanner.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/MPIPlanner.h)
   `build_item_slots(...)`

7. [Rewriter_MPI.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/Rewriter_MPI.cpp)
   `inferEffectiveParamModes(...)`

8. [Rewriter_MPI.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/Rewriter_MPI.cpp)
   输出同步的 `needsBcast` 判定逻辑
