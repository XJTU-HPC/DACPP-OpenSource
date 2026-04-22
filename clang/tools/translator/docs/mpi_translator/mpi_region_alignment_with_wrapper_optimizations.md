# MPI Region 对齐 Wrapper 优化说明

更新日期：2026-04-20

本文档记录当前 MPI wrapper 优化已经落到什么程度、MPI region 还差哪些点，以及后续如果要把优化继续对齐到 region，推荐从哪里下手。

这份文档只描述当前仓库的真实状态，不再沿用旧版 monolithic MPI 文件或拆分前的 `MPIPlanner.h` 行号引用。

## 1. 当前结论

当前状态可以先概括成三句话：

- wrapper 主线的 host 侧冗余优化已经落地，并且已经通过当前非 stencil 样例回归验证。
- planner/runtime 头文件已经拆分到 `dpcppLib/include/mpi/` 下，`MPIPlanner.h` 和 `MPIRegionRuntime.h` 现在只是兼容入口。
- region 还没有完整对齐 wrapper 的 host 去重链路，尤其是 `PackPlan`、root 侧远端 pack 重算消除、以及输出广播路径收紧这几项。

## 2. 代码入口已经发生的变化

### 2.1 planner/runtime 头文件已拆分

当前实际实现不再集中塞在一个超大 `MPIPlanner.h` 里，而是拆成了下面几个子头：

- [MPIPlanner.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/MPIPlanner.h)
- [MPIRegionRuntime.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/MPIRegionRuntime.h)
- [Common.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/mpi/Common.h)
- [Pack.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/mpi/Pack.h)
- [Views.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/mpi/Views.h)
- [Halo.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/mpi/Halo.h)
- [RegionRuntime.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/mpi/RegionRuntime.h)

其中：

- `MPIPlanner.h` 现在只是 umbrella include
- `MPIRegionRuntime.h` 现在也只是 umbrella include
- 实际 planner 逻辑主要在 `Common.h` 和 `Pack.h`
- 实际 region runtime 逻辑在 `RegionRuntime.h`

因此后续 region 对齐 wrapper 优化时，建议直接看 `dpcppLib/include/mpi/` 下的子头，而不是继续围着旧的单文件心智模型转。

### 2.2 当前活跃 codegen 路径

wrapper 当前主线：

- [Rewriter_MPI_Wrapper_Codegen.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Wrapper_Codegen.cpp)

region 当前主线：

- [Rewriter_MPI_Region_Codegen.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Region_Codegen.cpp)
- [Rewriter_MPI_Region_Init.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Region_Init.cpp)
- [Rewriter_MPI_Region_Submit.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Region_Submit.cpp)
- [Rewriter_MPI_Region_Halo.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Region_Halo.cpp)
- [Rewriter_MPI_Region_Sync.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Region_Sync.cpp)
- [Rewriter_MPI_Region_Sibling.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Region_Sibling.cpp)
- [Rewriter_MPI_Region_Policy.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Region_Policy.cpp)

旧版 `Rewriter_MPI_Region.cpp` monolithic 实现已经删除，不再参与任何活跃构建或代码生成。

## 3. wrapper 目前已经落地的优化

### 3.1 `PackPlan` 已经替代 `PackMap + build_item_slots`

当前 wrapper 已经改成：

- 在 [Pack.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/mpi/Pack.h) 中定义 `PackPlan`
- 通过 `build_input_pack_plan(...)` / `build_output_pack_plan(...)` / `build_rw_pack_plan(...)` 一次性拿到 `pack + slots`
- 在 [Rewriter_MPI_Wrapper_Codegen.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Wrapper_Codegen.cpp) 中直接生成 `plan.pack` 和 `plan.slots`

`PackPlan` 背后的关键点不是“把两段代码写进一个函数”，而是：

- 对 item 按 bind-key 去重
- 相同 bind-key 的 item 复用同一份 `collect_positions_for_item(...)` 结果
- 从而减少矩阵乘法这类规则访问中的重复位置展开

### 3.2 root 不再重算每个 rank 的远端 pack

当前 wrapper 已经改成：

- 各 rank 直接上报本地 `pack.globals`
- root 用 `MPI_Gather` + `MPI_Gatherv` 收齐这些 globals
- root 侧再用 `pack_values_by_globals_parallel_range(...)` 从全局 tensor 抽值

这意味着 wrapper 已经消除了过去那种：

- root 按 rank 循环重跑 `build_input_pack_map(...)`
- 每个远端 rank 的访问集合再算一遍

的路径。

### 3.3 `collect_positions_for_item()` 全面提速 (Fast Path 与 Odometer)

当前这部分在 [Common.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/mpi/Common.h) 里，并在最新版本中得到了决定性的增强：

对于规则访问的 Fast Path 已经能涵盖：
- 一整行、一整列、单个元素、一维连续段
- **多维连续分块（新增降维合并检测）**

对于不满足 Fast Path 从而完全回退到通用逻辑的复杂多维展开，已经升级为：
- **无除模运算的 Odometer (步长增量里程计) 算法**，在 Host 端达到了极致的生成效率。

这部分是 planner/runtime 公共层优化，所以 region 也已经自动吃到了这些红利。

### 3.4 wrapper 已增加规则 gather/writeback 的可选 SYCL helper

当前公共 helper 已存在于 [Pack.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/mpi/Pack.h)：

- `pack_values_by_globals_parallel(...)`
- `pack_values_by_globals_parallel_range(...)`
- `build_writeback_values_parallel(...)`

wrapper 当前已经用上的是：

- root 输入打包：`pack_values_by_globals_parallel_range(...)`
- writeback：`build_writeback_values_parallel(...)`

这些 helper 仍然保持：

- 小规模走 host 串行路径
- 超阈值时才走 SYCL `parallel_for`
- 结果回到 host 后再进入 MPI

也就是说，这些优化并不依赖 GPU-aware MPI。

### 3.5 wrapper 已收紧完整输出广播路径

wrapper 现在已经做了下面这些收紧：

- 不再生成 `synced_count_*`
- 非 root 不再先收一个 count 再分配
- 直接按 `tensor.getSize()` 分配完整输出缓冲
- 再直接 `MPI_Bcast(...)`
- 之后执行 `array2Tensor(...)`

同时 wrapper 里：

- `READ` 参数分发不再生成 `MPI_Scatter(... recv_count_...)`
- root 打包输入时不再构造 `std::vector<int64_t> r_globals(...)` 临时切片
- 非 `MPI_BYTE` 类型不再生成 `sendcounts_bytes_*` / `recvcounts_bytes_*`
- `MPI_BYTE` 类型仍保留 bytes 计数数组，这是预期行为

### 3.6 wrapper 已带 profiling 输出

当前 wrapper 会打印：

- `wrapper_total_ms(max)`
- `collect_positions_for_item total_calls(sum)`
- `collect_positions_for_item total_ms(sum)`
- `collect_positions_for_item max_rank_ms`

因此 wrapper 现在不仅“代码结构上更干净”，还具备持续做性能 AB 的基础。

## 4. region 当前还停在哪一步

当前 region runtime 入口是：

- [RegionRuntime.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/mpi/RegionRuntime.h)

它和 wrapper 的关键差异主要有四类。

### 4.1 region 仍保留 `item_pack + build_item_slots(...)` 的双遍历

当前 `init_region_param_storage(...)` 仍然是：

1. 先按 `state.pattern.mode` 构建 `item_pack`
2. 如果没有预配置 `runtime_pack`，让 `runtime_pack = item_pack`
3. 再基于 `runtime_pack` 调 `build_item_slots(...)`

也就是说，region 还保留着 wrapper 已经消掉的这段成本：

- `build_*_pack_map(...)` 一轮 item 遍历
- `build_item_slots(...)` 再来一轮 item 遍历
- `collect_positions_for_item(...)` 重复计算

### 4.2 region root 侧仍然重算远端 pack

当前 `init_region_param_storage(...)` 在非 dense-cover 的 scatter 路径里，root 仍然会：

1. 枚举每个 rank
2. 重新根据 `rank_range` 调 `build_pack_for_mode(...)`
3. 再用 `rank_pack.globals` 打包输入

这正是 wrapper 已经通过 `Gather/Gatherv pack.globals` 干掉的那部分。

### 4.3 region 还没切到并行 gather/writeback helper

当前 region 仍然使用：

- `pack_values_by_globals(...)`
- `build_writeback_values(...)`

还没有切到：

- `pack_values_by_globals_parallel(...)`
- `pack_values_by_globals_parallel_range(...)`
- `build_writeback_values_parallel(...)`

因此即便公共 helper 已经存在，region 目前还没有实际吃到这一块收益。

### 4.4 region 完整输出广播仍然保留 `synced_count`

当前 `writeback_region_output(...)` 仍然是旧路径：

- root 先构造 `synced_values`
- 广播 `synced_count`
- 非 root 按 `synced_count` 分配
- 再广播整块数据

这和 wrapper 现在“直接按 `tensor.getSize()` 广播完整输出”的路径还不一致。

## 5. region 和 wrapper 的一个核心语义差别

region 不能直接照搬 wrapper 的实现，最核心的原因是它有双 pack 语义：

- `item_pack`
- `runtime_pack`

当前 `RegionParamState` 里这两个字段都还在：

- `item_pack` 表示 item-space 真正访问到的集合
- `runtime_pack` 表示实际 local storage 的布局

这在 sibling dense-cover 场景里尤其关键，因为 region 可能会把：

- `runtime_pack` 扩成 dense cover

而这时：

- `item_pack != runtime_pack`

所以 region 的对齐方向不能是“把 wrapper 的 `PackPlan` 直接替换进去完事”，而是要拆开看：

- 哪些优化作用在 item positions 生成阶段
- 哪些优化作用在 runtime storage 布局阶段
- 哪些 lookup 一定要继续以 `runtime_pack` 为准

## 6. region 已经自动吃到的优化

### 6.1 `collect_positions_for_item()` fast path

这项已经在公共 planner 层里了，所以 region 已经自动获得收益。

### 6.2 头文件拆分带来的维护收益

虽然这不是性能优化，但现在 planner/runtime 被拆到：

- `Common.h`
- `Pack.h`
- `Views.h`
- `Halo.h`
- `RegionRuntime.h`

之后，后续 region 对齐 wrapper 时可以更明确地只改：

- `Pack.h`
- `RegionRuntime.h`

而不是继续在一个超大单头里滚动修改。

## 7. region 下一步最值得对齐的点

### 7.1 先做 runtime 层低风险替换

优先建议先在 [RegionRuntime.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/mpi/RegionRuntime.h) 做三项：

1. `pack_values_by_globals(...)` 切到并行 helper
2. `build_writeback_values(...)` 切到并行 helper
3. 完整输出广播改成 wrapper 现在的直接 `tensor.getSize()` 路径

这三项的共同特点是：

- 不需要改 region codegen 主体
- 不碰 halo/sibling 的主体语义
- 风险明显低于直接改 `item_pack/runtime_pack` 规划模型

### 7.2 再消掉 root 远端 pack 重算

第二阶段建议把 region 的 init scatter 改成 wrapper 同款思路：

1. 每个 rank 上报实际用于 scatter 的 `runtime_pack.globals`
2. root 收集所有 rank 的 globals
3. root 基于这些 globals 做 gather
4. 删除 root 侧按 rank 重建 `build_pack_for_mode(...)`

这里要特别注意：

- region 要上报的是 `runtime_pack.globals`
- 不是简单默认 `item_pack.globals`

因为 dense-cover sibling 场景下：

- root 如果只按 `item_pack.globals` 发值
- local runtime storage 可能不完整

### 7.3 最后再引入 region 版 plan

最后才建议做最关键但也最容易出语义问题的一步：

- 把 region 从 `build_pack_for_mode(...) + build_item_slots(...)`
- 升级为“共享 positions 缓存 + 按 runtime_pack 映射 slots”

更合适的方向不是直接复用 wrapper 的 `PackPlan` 成品，而是引入一个更贴近 region 的模型，至少能表达：

- item positions 的缓存
- item-side pack
- runtime-pack-side slots

一个实用的中间方案是：

- 先保留 `PackPlan` 作为公共基础结构
- 再补一个基于缓存 positions 构建 runtime slots 的 helper

这样可以做到：

- item positions 只算一次
- `item_pack` 仍然保留
- `runtime_pack` 仍然是 sibling / halo / sync 的权威布局

## 8. 迁移时必须守住的约束

### 8.1 halo 的 slot lookup 仍然必须对齐 `runtime_pack`

当前 halo 计算最终依赖 `ctx_pack.g2l` 去查 local slot。

因此 region 如果引入 item-plan 缓存，必须继续保证：

- halo 的 local slot lookup 仍然对齐 `runtime_pack`

不能偷懒退回 `item_pack`，否则 dense-cover sibling 场景会错位。

### 8.2 sibling 仍然以 `runtime_pack` 为权威布局

sibling 逻辑依赖：

- `runtime_pack`
- `global_to_local`
- dense fallback / dense shadow
- dirty sparse sync

所以 region 的优化原则只能是：

- 减少 item positions 的重复生成
- 减少 root 远端 pack 重算
- 减少规则 gather/writeback 的 host 成本

不能破坏 `runtime_pack` 作为实际存储布局的权威性。

### 8.3 `MPI_BYTE` 分支不能误删

wrapper 当前只在 `MPI_BYTE` 才生成 bytes 计数数组。

region 后续对齐时也应该保持同样原则：

- 标准 MPI datatype 不额外生成 bytes 计数数组
- `MPI_BYTE` 继续保留 bytes 路径

### 8.4 仍然不假设 GPU-aware MPI

无论 wrapper 还是 region，目前这些 helper 的设计前提都还是：

- gather/writeback 可以用 SYCL 提速
- 但 MPI 收发仍然基于 host memory

后续 region 对齐时也不建议把不规则 pack-map 生成、变长 positions 构建、`unordered_map g2l` 等逻辑强行搬到 device。

## 9. 当前验证状态

### 9.1 wrapper 生成代码已静态确认

已经重新翻译并静态检查过代表性 wrapper 样例，确认当前生成代码里：

- 已使用 `build_*_pack_plan(...)`
- 已使用 `pack_values_by_globals_parallel_range(...)`
- 已去掉 `recv_count_*`
- 已去掉 `synced_count_*`
- 已去掉 root 侧 `std::vector<int64_t> r_globals(...)`
- 非 `MPI_BYTE` 类型不再生成 bytes 计数数组

### 9.2 当前非 stencil 回归已通过

已通过的非 stencil MPI 回归包括：

- `matMul1.0`
- `gradientSum`
- `DFT1.0`
- `decay1.0`
- `FOuLa1.0`
- `MDP1.0`
- `mandel1.0`
- `imageAdjustment1.0`
- `vectorAddCombo`
- `liuliang1.0`
- `oddeven0.1`
- `jacobi1.0`

这里面：

- wrapper 路径样例验证了 wrapper 主线优化没有回退
- `liuliang1.0`、`jacobi1.0` 这类样例也顺带验证了公共 planner/runtime 变更没有破坏 region 路径

但要注意，这不等于 region 已经吃到了 wrapper 的全部 host 去重优化，只说明当前公共层拆分和清理没有引入功能回归。

## 10. 推荐的实现落点

如果下一步要正式把 wrapper 优化继续推到 region，建议优先从这些文件入手：

- 公共 planner/runtime：
  - [Common.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/mpi/Common.h)
  - [Pack.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/mpi/Pack.h)
  - [RegionRuntime.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/mpi/RegionRuntime.h)
- region codegen 主线：
  - [Rewriter_MPI_Region_Init.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Region_Init.cpp)
  - [Rewriter_MPI_Region_Sync.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Region_Sync.cpp)
  - [Rewriter_MPI_Region_Sibling.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Region_Sibling.cpp)

## 11. 一句话结论

当前 region 已经自动吃到了公共 planner 的 fast path，但还没有吃到 wrapper 最关键的三类 host 侧优化：

- `PackPlan` 带来的 item positions 去重
- root 侧远端 pack 重算消除
- 完整输出广播与规则 gather/writeback 路径的进一步收紧

下一步最稳妥的方向不是“把 region 改成 wrapper”，而是把 wrapper 已验证有效的 planner/runtime 去重能力，按 `item_pack/runtime_pack` 双语义重新落到 region runtime 里。
