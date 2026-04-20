# MPI Wrapper 性能优化实现说明

本文档说明这次 MPI wrapper 路径性能优化是怎么实现的，重点覆盖：

- 优化前的瓶颈在哪里
- 每一项优化对应改了哪些代码
- 新旧路径在语义上有什么变化
- 这轮优化目前已经验证到了什么程度

该文档聚焦已经落地的 wrapper 主线实现，不讨论尚未迁移到 region 的部分。

## 1. 优化目标

这次优化针对的是 MPI wrapper 路径在非 stencil 工作负载下的 host 侧开销，尤其是矩阵乘法这类“访问规则但 item 数量很大”的场景。

优化前的主要问题有四类：

- `build_input_pack_map(...)` 和 `build_item_slots(...)` 对同一批 item 分别遍历，重复调用 `collect_positions_for_item()`
- root 为每个 rank 重新构造远端 pack，再次重复调用 `collect_positions_for_item()`
- `collect_positions_for_item()` 对简单一维连续或跨步访问仍走通用多维展开逻辑
- writeback 与 root gather/scatter 中的规则 gather 仍完全走 host 串行路径

这次实现的目标不是改 kernel，而是先把 planner / codegen / runtime 中明显重复的 host 计算削掉。

## 2. 总体实现思路

这轮实现分成四层：

1. planner 层去重
2. wrapper codegen 改写
3. AST broadcast 判断收窄
4. profiling 与验证支持

主线思想是：

- 把 pack 和 slots 的构建合并为同一套 item-position 规划
- 把 root 侧“重算远端访问集合”改成“收集远端访问集合”
- 对矩阵乘法这类规则模式提供保守的 fast path
- 保持 MPI 仍基于 host memory，不引入 GPU-aware MPI 假设

## 3. planner 层改动

主要代码在 [MPIPlanner.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/MPIPlanner.h)。

### 3.1 新增 `PackPlan`

新增了一个统一结构：

- [MPIPlanner.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/MPIPlanner.h#L76)

```cpp
struct PackPlan {
    PackMap pack;
    std::vector<int32_t> slots;
};
```

它的作用是把原先分开的：

- `build_*_pack_map(...)`
- `build_item_slots(...)`

合并为一次规划结果。

### 3.2 按 bind-key 缓存 item positions

这次优化的关键不是简单把两次循环拼到一起，而是进一步识别“逻辑上等价的一批 item”。

实现上新增了：

- `build_item_bind_key(...)`
- `VectorIntHash`
- `build_pack_plan(...)`

关键代码在：

- [MPIPlanner.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/MPIPlanner.h#L610)
- [MPIPlanner.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/MPIPlanner.h#L640)

具体做法是：

1. 对每个 item 生成 bind-key
2. 如果 key 首次出现，则调用一次 `collect_positions_for_item(...)`
3. 后续同 key 的 item 直接复用同一份 positions
4. 所有唯一 positions 合并生成 `pack`
5. 再把 positions 映射成 `slots`

这样处理后，像矩阵乘法这种：

- A 的“每个 item 实际是一整行”
- B 的“每个 item 实际是一整列”
- C 的“每个 item 实际是单元素”

就不需要为每个输出元素都重新展开相同行或列的全局位置。

### 3.3 新增 `build_input_pack_plan(...)` / `build_output_pack_plan(...)` / `build_rw_pack_plan(...)`

这三个 helper 是 wrapper codegen 的直接入口：

- [MPIPlanner.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/MPIPlanner.h#L698)

它们只是对 `build_pack_plan(...)` 的 mode 包装，语义分别对应：

- 只读输入
- 只写输出
- 读写参数

### 3.4 `collect_positions_for_item()` fast path

为了进一步降低单次调用成本，在 `collect_positions_for_item(...)` 前面增加了简单跨步检测：

- [MPIPlanner.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/MPIPlanner.h#L325)
- [MPIPlanner.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/MPIPlanner.h#L378)

新增了：

- `SimpleStrideRange`
- `try_build_simple_stride_range(...)`

检测条件很保守：

- `part_shape` 最多只有一个维度大于 1

满足时直接生成：

- `start + i * stride`

不满足时完全回退到原来的通用逻辑。

这个 fast path 覆盖的典型模式包括：

- 矩阵一整行
- 矩阵一整列
- 单元素输出
- 一维整段向量访问

但不会误伤：

- 2D stencil 这类多维同时扩展的分区

### 3.5 新增可选 SYCL gather/writeback helper

这次还补了两个可选的规则 gather helper：

- `pack_values_by_globals_parallel(...)`
- `build_writeback_values_parallel(...)`

代码位于：

- [MPIPlanner.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/MPIPlanner.h#L725)
- [MPIPlanner.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/MPIPlanner.h#L789)

实现策略是：

- 默认仍走 host 串行实现
- 当元素数量超过阈值时，用一个简单的 SYCL `parallel_for` 做 gather
- gather 结果仍然回到 host vector，再进入 `MPI_Scatterv` / `MPI_Gatherv`

也就是说，这里优化的是“规则索引采样”本身，不改变 MPI 的 host memory 前提。

### 3.6 增加 profiling 入口

为了验证优化是否真的打到了瓶颈，这次还在 runtime 里增加了 profiling：

- `profilingEnabled()`
- `resetCollectPositionsProfile()`
- `recordCollectPositionsSample(...)`
- `reportCollectPositionsProfile(...)`

代码位于：

- [MPIPlanner.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/MPIPlanner.h#L91)

行为是：

- 仅在设置 `DACPP_MPI_PROFILE=1` 时启用
- 统计 `collect_positions_for_item()` 的调用次数和总耗时
- root 汇总所有 rank，并输出：
  - 总调用数
  - 总耗时
  - 最慢 rank 耗时

这些日志打印到 `stderr`，避免破坏测试输出比对。

## 4. wrapper codegen 改动

主要代码在 [Rewriter_MPI_Wrapper_Codegen.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Wrapper_Codegen.cpp)。

### 4.1 wrapper 从 `PackMap + slots` 切到 `PackPlan`

原先 wrapper 会生成类似：

```cpp
auto pack_X = build_input_pack_map(...);
auto slots_X = build_item_slots(...);
```

现在改成：

- [Rewriter_MPI_Wrapper_Codegen.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Wrapper_Codegen.cpp#L65)

```cpp
auto plan_X = build_*_pack_plan(...);
auto& pack_X = plan_X.pack;
auto& slots_X = plan_X.slots;
```

这一步让 codegen 自动吃到 planner 层的去重收益。

### 4.2 root 不再重算远端 pack

这项是 wrapper 的第二个核心优化。

原先 root 会对每个 rank：

1. 重新算 `r_range`
2. 重新跑 `build_input_pack_map(...)`
3. 再用 `r_pack.globals` 去抽数据

现在改成：

- 每个 rank 先汇报自己的 `pack.globals.size()`
- 再用 `MPI_Gatherv` 把自己的 `pack.globals` 发给 root
- root 直接根据收到的 globals 去 `pack_values_by_globals_parallel(...)`

实现代码在：

- [Rewriter_MPI_Wrapper_Codegen.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Wrapper_Codegen.cpp#L72)

这一步直接消掉了 root 侧对远端访问集合的重建。

### 4.3 writeback 路径切到并行 helper

在输出回收时，wrapper 现在直接调用：

- [Rewriter_MPI_Wrapper_Codegen.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Wrapper_Codegen.cpp#L209)

```cpp
auto writeback_X = dacpp::mpi::build_writeback_values_parallel(local_X, pack_X);
```

这样 writeback 从 local packed array 抽出“要写回的逻辑输出值”时，也能在大规模数据上用 SYCL 并行 gather。

### 4.4 wrapper 性能统计输出

wrapper 生成代码时还补上了整体 wall-clock 计时，并汇总最大 rank 时间：

- [Rewriter_MPI_Wrapper_Codegen.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Wrapper_Codegen.cpp#L277)

配合 planner 的 profile，可以同时看到：

- wrapper 总耗时
- `collect_positions_for_item()` 在其中占多少

## 5. pattern/codegen 入口改动

为了让 wrapper codegen 更干净，这次把“生成 plan 构建表达式”的逻辑单独抽了一层：

- 声明在 [Rewriter_MPI_Common.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/include/Rewriter_MPI_Common.h#L76)
- 实现在 [Rewriter_MPI_Pattern.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Pattern.cpp#L181)

新增 helper：

```cpp
buildPackPlanBuilderExpr(IOTYPE mode, rangeName, patternName)
```

它的作用是：

- 根据参数读写模式，返回对应的 `build_input_pack_plan(...)` / `build_output_pack_plan(...)` / `build_rw_pack_plan(...)`

这样 wrapper codegen 不需要自己判断 mode 分支，只需要统一拼接表达式。

## 6. AST broadcast 判断修正

这次优化里还有一个语义修正，虽然不是主要性能点，但对 wrapper 正确性很重要。

问题是原先 broadcast 判断过于保守，只要输出 tensor 在主函数后面任意地方被引用，就会触发广播，哪怕那个引用发生在当前 `<->` 之前。

这次在 analysis 层改成：

- 只看“当前 `<->` 之后是否还有对该 tensor 的外部使用”

实现位于：

- [Rewriter_MPI_Analysis.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Analysis.cpp#L14)
- [Rewriter_MPI_Analysis.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Analysis.cpp#L229)

具体做法是：

- `TensorUseVisitor` 持有 `CurrentDacExpr`
- 遍历 AST 时，只有在“已经见过当前 dacExpr”之后出现的使用，才会把 `NeedsBcast` 置为 true

同时，wrapper codegen 也把当前 `dacExpr` 显式传入：

- [Rewriter_MPI_Wrapper_Codegen.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Wrapper_Codegen.cpp#L207)

这个改动的收益是：

- 避免不必要的整 tensor 广播
- 保持语义仍然保守，但不再被前序引用误触发

## 7. rewrite 入口上的配合改动

为了让 analysis 能拿到当前 `<->`，`rewriteMPI()` 在生成 wrapper 时改成把 `expr->getDacExpr()` 传给 `buildWrapperCode(...)`：

- [Rewriter_MPI.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/Rewriter_MPI.cpp#L25)

对应公共声明也改成了：

- [Rewriter_MPI_Common.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/include/Rewriter_MPI_Common.h#L82)

```cpp
std::string buildWrapperCode(..., const clang::BinaryOperator* dacExpr = nullptr);
```

这一步主要是为 broadcast 精化服务。

## 8. 语义保持与设计边界

这轮实现刻意控制在下面这些边界内。

### 8.1 没有引入 matmul 专用 codegen

所有优化都仍然是通用 planner/runtime/codegen 层能力，不依赖矩阵乘法特判。

### 8.2 没有把不规则 planner 搬到 device

下面这些仍然保留在 host：

- bind-key 去重
- `sort + unique`
- `unordered_map g2l`
- 变长 positions 生成

device 只参与规则 gather/writeback。

### 8.3 MPI 仍然使用 host memory

即使 `pack_values_by_globals_parallel(...)` 和 `build_writeback_values_parallel(...)` 用了 SYCL gather，最终结果仍然回到 host vector 后再进入 MPI。

这保证：

- 不依赖 GPU-aware MPI
- 现有执行环境不需要额外能力

## 9. 实测验证

这次实现做了两类验证。

### 9.1 功能回归

通过的 MPI 用例包括：

- `matMul1.0`
- `gradientSum`
- `FOuLa1.0`
- `decay1.0`
- `DFT1.0`
- `liuliang1.0`

此外，在 `opt3` 之后又补跑了：

- `bash test_mpi.sh matMul1.0`
- `bash test_mpi.sh gradientSum`

两者都通过。

### 9.2 256×256 matmul 性能验证

为了避免仓库里现成的大样例并不是 256×256，这次使用了临时 256×256 matmul case 进行 profiling。

验证条件是：

- `DACPP_MPI_PROFILE=1`
- `OMP_NUM_THREADS=8`
- `mpirun -np 2`

在引入 `PackPlan + root globals gather` 之后，`collect_positions_for_item()` 指标已经显著下降到：

- `total_calls(sum): 66304`
- `total_ms(sum): 11.394`
- `max_rank_ms: 5.703`

再加上 stride fast path 之后，变成：

- `wrapper_total_ms(max): 59.126`
- `collect_positions_for_item total_calls(sum): 66304`
- `collect_positions_for_item total_ms(sum): 6.864`
- `collect_positions_for_item max_rank_ms: 3.544`

可以看到：

- 调用次数不变，说明优化点确实是“每次调用更快”
- `collect_positions_for_item total_ms` 继续下降约 40%
- `max_rank collect ms` 继续下降约 38%

这说明：

- `opt1` 解决的是“重复调用太多”
- `opt2` 解决的是“root 还在重算”
- `opt3` 解决的是“单次调用本身太慢”

## 10. 当前结论

这次实现把 wrapper 路径的主要 host-side 冗余切掉了三层：

1. item positions 不再为 pack 和 slots 各算一遍
2. root 不再为别的 rank 重算 pack
3. 简单跨步访问不再走通用多维展开

与此同时，还补上了两类工程化能力：

- 可选 SYCL gather/writeback helper
- 可观测的 profiling 输出

这让 wrapper 路径从“明显被 planner 开销支配”，变成了“planner 热点显著下降，后续可以继续观察 MPI / buffer / writeback / slots 访问的次级瓶颈”。
