# MPI Region 对齐 Wrapper 优化说明

本文档用于把已经在 MPI wrapper 路径中落地的性能优化，对齐到 MPI region 路径，方便后续实现和评审时统一目标、边界与优先级。

重点不是重新解释 wrapper 做了什么，而是回答这三个问题：

- wrapper 已经优化到了哪里
- region 当前还停在哪一步
- region 应该怎样最小风险地跟进

## 1. 背景

在非 stencil 的矩阵乘法场景中，wrapper 路径的主要瓶颈并不在 kernel，而在 host 侧的数据组织：

- `collect_positions_for_item()` 被重复调用
- `build_input_pack_map(...)` 和 `build_item_slots(...)` 对同一批 item 分别遍历
- root 为每个 rank 重新构造远端 pack
- writeback / gather / scatter 前后还存在额外 host 组织成本

这条链路已经在 wrapper 路径上做过一轮优化，核心收益来自：

- 用 `PackPlan` 合并 pack 和 slots 构建
- 按 bind-key 复用同一类 item 的 positions
- root 直接收集各 rank 的 `pack.globals`，不再重算远端 pack
- 对规则 gather/writeback 增加可选 SYCL 并行 helper
- 在 `collect_positions_for_item()` 内增加 stride fast path

这些优化中的大部分并不依赖 wrapper 特有语义，理论上都可以复用到 region。

## 2. 当前代码状态

### 2.1 wrapper 路径

wrapper 主线已经切到 `PackPlan`：

- `PackPlan` 与 `build_input_pack_plan(...)` / `build_output_pack_plan(...)` / `build_rw_pack_plan(...)` 位于 [MPIPlanner.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/MPIPlanner.h#L640)
- wrapper codegen 使用 `plan.pack + plan.slots`，位于 [Rewriter_MPI_Wrapper_Codegen.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Wrapper_Codegen.cpp#L65)
- root 侧已改为 `Gather/Gatherv pack.globals`，不再调用远端 `build_input_pack_map(...)`，位于 [Rewriter_MPI_Wrapper_Codegen.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Wrapper_Codegen.cpp#L72)
- `collect_positions_for_item()` 的 stride fast path 位于 [MPIPlanner.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/MPIPlanner.h#L325)

### 2.2 region 路径

region 当前主线不是 [Rewriter_MPI_Region.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Region.cpp) 里的旧式拼接骨架，而是：

- [Rewriter_MPI_Region_Codegen.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Region_Codegen.cpp)
- [Rewriter_MPI_Region_Init.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Region_Init.cpp)
- [Rewriter_MPI_Region_Submit.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Region_Submit.cpp)
- [Rewriter_MPI_Region_Halo.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Region_Halo.cpp)
- [Rewriter_MPI_Region_Sync.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Region_Sync.cpp)

当前 region runtime 入口在：

- [MPIRegionRuntime.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/MPIRegionRuntime.h)

其中初始化与回收的关键路径是：

- `init_region_param_storage(...)` 位于 [MPIRegionRuntime.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/MPIRegionRuntime.h#L76)
- `writeback_region_output(...)` 位于 [MPIRegionRuntime.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/MPIRegionRuntime.h#L229)

## 3. wrapper 与 region 的关键差异

当前 wrapper 和 region 的 planner/runtime 差异主要在下面几项。

### 3.1 pack/slots 构建方式不同

wrapper 已经是：

- `build_*_pack_plan(...)`
- 单次构建得到 `pack + slots`
- 同 bind-key 复用 `collect_positions_for_item()`

region 现在仍然是：

- 先构建 `item_pack`
- 再用 `runtime_pack` 调 `build_item_slots(...)`

对应代码在 [MPIRegionRuntime.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/MPIRegionRuntime.h#L82)：

```cpp
state.item_pack =
    detail::build_pack_for_mode(item_range, state.pattern, state.pattern.mode);
...
state.slots =
    build_item_slots(item_range, state.pattern, state.runtime_pack);
```

这意味着 region 还保留着 wrapper 已经消掉的“双遍历 item + 重复算 positions”。

### 3.2 root 侧远端 pack 重算仍然存在

在 region 的 `init_region_param_storage(...)` 中，root 仍然按 rank 循环重算远端 pack：

- [MPIRegionRuntime.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/MPIRegionRuntime.h#L121)

```cpp
for (int rank = 0; rank < mpi_size; ++rank) {
    const auto rank_range =
        get_rank_item_range(total_items, rank, mpi_size);
    const auto rank_pack = detail::build_pack_for_mode(
        rank_range, state.pattern, state.pattern.mode);
    auto rank_values =
        pack_values_by_globals(global_values, rank_pack.globals);
    ...
}
```

这正是 wrapper 已经通过 `Gather/Gatherv pack.globals` 干掉的那部分。

### 3.3 region 存在 item_pack/runtime_pack 双 pack 语义

region 不像 wrapper 只有一个 pack。它有：

- `item_pack`：由 item-space 真正访问集合导出
- `runtime_pack`：实际 local storage 采用的 pack

对应结构在 [MPIRegionRuntime.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/MPIRegionRuntime.h#L12)。

这点很重要，因为 region 的 sibling 场景可能把 `runtime_pack` 扩成 dense cover：

- [Rewriter_MPI_Region_Init.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Region_Init.cpp#L223)

也就是说，wrapper 的 `PackPlan = 访问计划 = 存储布局`，但 region 不一定成立。

因此 region 对齐 wrapper 优化时，不能简单把 `item_pack` 删除掉，而是要区分：

- 哪些优化作用在 item positions 生成阶段
- 哪些优化作用在实际 runtime storage 打包阶段

## 4. 可以直接复用到 region 的优化

下面这些优化原则上可以直接迁移，收益明确，语义风险也较低。

### 4.1 `collect_positions_for_item()` stride fast path

这项已经在 runtime 公共层实现了：

- [MPIPlanner.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/MPIPlanner.h#L378)

因此 region 已经自动吃到这部分收益，不需要额外 codegen 改造。

这意味着：

- matmul 这种一行 / 一列 / 单点的模式
- 以及一维整段访问的模式

在 region/runtime 共用 planner 时都会变快。

### 4.2 `pack_values_by_globals_parallel(...)`

该 helper 已经在 runtime 层可用：

- [MPIPlanner.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/MPIPlanner.h#L725)

region 目前 `init_region_param_storage(...)` 仍调用串行版 `pack_values_by_globals(...)`：

- [MPIRegionRuntime.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/MPIRegionRuntime.h#L132)

这部分可以低风险切换为：

- 小规模保持 host 路径
- 超阈值时走 `pack_values_by_globals_parallel(...)`

前提仍然和 wrapper 一样：

- kernel 结果必须回到 host 后再喂给 MPI
- 不假设 GPU-aware MPI

### 4.3 `build_writeback_values_parallel(...)`

该 helper 也已经在 runtime 层可用：

- [MPIPlanner.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/MPIPlanner.h#L789)

region 当前 `writeback_region_output(...)` 仍然调用串行版：

- [MPIRegionRuntime.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/MPIRegionRuntime.h#L237)

```cpp
auto writeback_values = build_writeback_values(state.local, state.runtime_pack);
```

这部分同样可以直接切到并行 helper，并保持阈值控制。

## 5. 需要改造后才能迁移到 region 的优化

下面这些优化不能直接照搬，需要针对 region 的 `item_pack/runtime_pack` 双语义做设计。

### 5.1 `PackPlan` 合并 pack + slots

wrapper 的核心收益之一来自 `PackPlan`：

- [MPIPlanner.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/MPIPlanner.h#L640)

但 region 不能直接“把 `build_pack_plan()` 的结果当成 runtime pack”，因为 region 可能有这两种情况：

- `runtime_pack == item_pack`
- `runtime_pack` 被 sibling 扩成 dense cover

因此 region 要复用 `PackPlan`，推荐拆成两层：

1. item access plan
2. runtime storage plan

更具体地说：

- 当 `runtime_pack == item_pack` 时，region 应直接使用 `PackPlan`
- 当 `runtime_pack` 是 dense cover 或其他扩展布局时，region 仍需要：
  - 用 item-side positions 生成逻辑访问顺序
  - 但 slots 要基于 `runtime_pack` 做 lookup

推荐新增一个更贴近 region 的 runtime helper：

```cpp
build_region_item_plan(item_range, pattern)
```

它至少返回：

- `item_pack`
- `item_positions_by_key` 或等价缓存
- `slots_for_runtime_pack(runtime_pack)`

如果不想新开类型，也可以在 `PackPlan` 基础上新增一个 helper：

```cpp
build_slots_from_cached_positions(cached_positions, runtime_pack)
```

这样可以做到：

- item positions 只算一次
- `item_pack` 和 `runtime_pack` 分别按需要生成
- dense cover sibling 场景仍成立

### 5.2 root 收集各 rank `pack.globals` 替代远端重算

wrapper 已经实现这条链路，但 region 目前仍然把远端 pack 重算放在 runtime 里。

region 迁移这项优化时要注意：

- root 应该收集的是实际用于 init scatter 的 `runtime_pack.globals`
- 不是简单默认 `item_pack.globals`

原因是 sibling dense-cover 场景下：

- `runtime_pack.globals` 可能是 dense cover
- root 如果只按 `item_pack.globals` 发初值，会导致运行期 local storage 不完整

因此 region 的 root gather 逻辑应该围绕 `state.runtime_pack.globals` 来做。

建议直接在 `init_region_param_storage(...)` 中改造，而不是在 codegen 层重写一套。这样可以复用 runtime helper，也更容易统一 wrapper/region planner 语义。

## 6. 不建议直接搬到 region 的点

有几项 wrapper 优化思路，region 不能简单照搬，或者当前不值得优先做。

### 6.1 把不规则 pack-map 构建搬到 SYCL device

不建议。

原因和 wrapper 一样：

- `sort + unique`
- `unordered_map g2l`
- 变长 `positions`
- bind-key 去重缓存

这些都更适合先留在 host 侧。

region 比 wrapper 还复杂，因为还叠加了：

- dense cover runtime pack
- sibling lookup
- halo pack/layout 依赖

### 6.2 直接把 region 全部改成 wrapper 模式

不建议。

region 的生命周期是：

- `init`
- `submit`
- `halo`
- `sibling`
- `sync`

它本身就需要跨多轮迭代保持 local state，因此不能退化成“每次 `<->` 一次性 scatter + gather”的 wrapper 模式。

正确方向应该是：

- 共用 planner/runtime 优化
- 保留 region 自己的状态机和 sibling/halo 语义

## 7. 推荐的 region 对齐顺序

为了降低风险，建议按下面顺序推进。

### Phase 1: runtime helper 低风险替换

目标：

- 先拿到收益最稳的改造
- 不改 region codegen 结构

建议改动：

1. 在 `writeback_region_output(...)` 中切到 `build_writeback_values_parallel(...)`
2. 在 `init_region_param_storage(...)` 中切到 `pack_values_by_globals_parallel(...)`
3. 保持阈值控制，默认 host 路径不变

涉及文件：

- [MPIRegionRuntime.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/MPIRegionRuntime.h)

### Phase 2: 去掉 root 远端 pack 重算

目标：

- 对齐 wrapper 的 `opt2`
- 消除 region init 中对远端 `build_pack_for_mode(...)` 的重复计算

建议改动：

1. 每个 rank 上报 `state.runtime_pack.globals.size()`
2. `MPI_Gatherv` 收集各 rank 的 `runtime_pack.globals`
3. root 根据收集到的 globals 做 `pack_values_by_globals[_parallel](...)`
4. 删除 root 侧 `for rank -> build_pack_for_mode(...)`

涉及文件：

- [MPIRegionRuntime.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/MPIRegionRuntime.h#L76)

### Phase 3: 引入 region 版 plan，复用 positions 缓存

目标：

- 对齐 wrapper 的 `opt1`
- 消除 region 中 `build_pack_for_mode(...) + build_item_slots(...)` 的双遍历

推荐实现方式：

1. 保留 `PackPlan` 作为公共基础结构
2. 新增 region 专用 helper，显式区分：
   - item positions
   - item-derived pack
   - runtime-pack-based slots
3. 当 `runtime_pack == item_pack` 时走最简路径
4. 当 `runtime_pack` 为 dense cover 时，直接基于缓存 positions 重新映射 slots

优先修改 runtime，而不是先动 codegen。

原因是当前 region 主线已经把大部分初始化收敛到 `init_region_param_storage(...)`，在 runtime 层做更容易保持 wrapper / region 一致。

## 8. 对 halo 与 sibling 的影响

### 8.1 halo

当前 halo 计算使用的是 `ctx_pack`：

- [MPIPlanner.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/MPIPlanner.h#L1501)

而 region 在调用时传的是 `runtime_pack`：

- [Rewriter_MPI_Region_Init.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Region_Init.cpp#L263)

因此 region 若引入 item-plan 缓存，必须保持：

- halo 的 local slot lookup 仍然对齐 `runtime_pack`

这一点不能退回到 `item_pack`，否则 dense cover sibling 场景会错位。

### 8.2 sibling

sibling 依赖：

- `runtime_pack`
- `global_to_local`
- `dense_fallback`
- `dense_shadow`
- dirty 稀疏同步

因此 region 的优化原则是：

- 可以减少 item positions 的重复生成
- 不能破坏 `runtime_pack` 作为 sibling 实际布局的权威性

## 9. 推荐的实现落点

如果要正式把 wrapper 优化继续推到 region，建议优先从这些文件入手：

- runtime 公共层：
  - [MPIRegionRuntime.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/MPIRegionRuntime.h)
  - [MPIPlanner.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/MPIPlanner.h)
- region codegen 主线：
  - [Rewriter_MPI_Region_Init.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Region_Init.cpp)
  - [Rewriter_MPI_Region_Sync.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Region_Sync.cpp)
- 可忽略的旧路径：
  - [Rewriter_MPI_Region.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Region.cpp)

## 10. 一句话结论

region 已经自动吃到 `collect_positions_for_item()` 的 fast path，但还没有吃到 wrapper 的两项核心 host 优化：

- `PackPlan` 合并 pack + slots
- root 侧远端 pack 重算消除

下一步最合理的方向不是“把 region 改成 wrapper”，而是把 wrapper 已验证有效的 planner/runtime 去重能力，按 `item_pack/runtime_pack` 双语义重新落到 region runtime 里。
