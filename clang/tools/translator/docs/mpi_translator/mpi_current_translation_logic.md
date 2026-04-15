# DACPP Translator MPI Handoff

这份文档是 `clang/tools/translator` 里 MPI 翻译路径的正式交接说明。

它回答四个核心问题：

1. 这个 MPI 路径现在在做什么
2. 代码真正从哪里进入、如何分支
3. 数据是如何分配、分发、同步的
4. 下一个模型应该从哪里继续接手

## 1. 项目状态概览

MPI 翻译路径现在有两条实现线：

- 稳定主线：基于 item-space / pack-map 的通用 MPI wrapper
- region 路线：面向时间推进类程序的 `init -> submit -> halo -> sync` MPI region 骨架

真正默认生效的行为是：

- 命中 buffer region，且没有 sibling loops
  - 走 `rewriteMPI_Region()`
- 其他情况
  - 走 `rewriteMPI()`

因此可以直接把当前实现理解成：

- 无 sibling loops 的 region 已经走 region 翻译路径
- 有 sibling loops 的 region 仍然走稳定 wrapper 路径

## 2. 入口与分支决策

### 2.1 入口文件

MPI 路径由 [translator.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/translator.cpp) 驱动。

主流程是：

1. 识别 `--mpi`
2. 注入：
   - `<mpi.h>`
   - `<cstdio>`
   - `"MPIPlanner.h"`
3. 执行 `analyzeBufferRegionPlan()`
4. 根据 region plan 选择：
   - `rewriteMPI_Region()`
   - 或 `rewriteMPI()`
5. 再交给通用 `rewriteMain()` 做后续收尾

### 2.2 实际分支条件

MPI 分支判断的核心条件是：

- `hasBufferRegionPlan() && siblingForStmts.empty()`
  - 调用 `rewriteMPI_Region()`
- 否则
  - 调用 `rewriteMPI()`

当存在 sibling loops 时，`translator.cpp` 会打印：

`MPI region optimization deferred: sibling loops are not deviceized yet; falling back to stable MPI wrapper`

这条信息可以直接视为当前 sibling 策略的外部行为说明。

## 3. 两条 MPI 翻译路径

### 3.1 稳定主线：通用 MPI wrapper

稳定主线由 `rewriteMPI()` 生成。

这条路径的核心思想不是按物理 tensor 简单切块，而是：

1. 基于 shell / binding / split 建立 item-space
2. 给每个 rank 分配连续 `item_range`
3. 对每个参数根据读写模式构造 pack-map
4. root 只打包并发送该 rank 真正会访问的数据
5. rank 在压缩后的局部数组上通过 `slots + View1D/View2D` 执行 SYCL kernel
6. 最终只把需要写回的元素按全局索引 gather 回 root

这条路径优化的是：

- 真实访问的数据集合

不是：

- 几何意义上的整块规则子域

### 3.2 稳定主线已经完成的关键能力

稳定主线已经具备：

- 基于 AST 推导参数真实读写模式
- 基于 binding / split 统一规划 item-space
- 基于 pack-map 的精确 scatter / gather
- 压缩局部数组上的 `View1D / View2D` 本地执行
- 仅对需要写回的参数执行 gather / writeback
- 必要时对结果做广播
- 对无标准 MPI datatype 的类型回退到 `MPI_BYTE`，并按字节数传输

### 3.3 Region 路线

region 路线由 `rewriteMPI_Region()` 生成，实际入口在 [Rewriter_MPI.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/Rewriter_MPI.cpp) 里。

它面向的执行模型是：

`循环前 init 一次 -> 每步 submit -> halo -> 最后 sync`

设计目标是：

1. 在 region 生命周期开始时完成一次性分发
2. 让本地 `local_* / buf_* / pack_*` 常驻
3. 每步只做 kernel 提交和 halo 交换
4. region 结束时统一 gather / writeback

这条路径已经具备完整骨架：

- `ctx`
- `init`
- `submit`
- `halo`
- `sync`

## 4. 数据如何分配与分发

### 4.1 稳定主线的分配方式

稳定主线的分配过程是：

1. 从 shell / split / binding 构建 `AccessPattern`
2. 合并出统一的 `binding_split_sizes`
3. 计算全局 `total_items`
4. 使用 `get_rank_item_range(total_items, mpi_rank, mpi_size)` 给每个 rank 分连续 item range

所以每个 rank 拿到的是：

- 一段逻辑 item 区间

不是：

- 一段简单线性的 tensor 物理区间

### 4.2 稳定主线的分发方式

对每个参数，按模式构造 pack-map：

- `READ` 使用 `build_input_pack_map`
- `WRITE` 使用 `build_output_pack_map`
- `READ_WRITE` 使用 `build_rw_pack_map`

root 对每个 rank 的处理流程是：

1. 根据该 rank 的 pack-map 算出真正需要的全局索引
2. 使用 `pack_values_by_globals(...)` 打包
3. 使用 `MPI_Scatter` / `MPI_Scatterv` 仅发送这部分 payload

因此当前的数据分发本质是：

- 按访问需求精确分发

### 4.3 Region 路线的目标分发方式

region 路线的目标模型是：

- `init` 阶段完成一次性 scatter
- 本地 `local_*` 从一开始就持有 interior + halo
- 后续每步不再做 root scatter / gather
- 每步只执行局部 kernel 和 halo exchange

## 5. Halo 运行时支持

Halo 运行时位于：

- [MPIPlanner.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/MPIPlanner.h)

核心数据结构：

- `HaloRegion`
- `ParamHalo`

核心函数：

- `computeParamHalo(...)`
- `exchangeHalo(...)`

### 5.1 Halo 组织方式

Halo 是按以下维度组织的：

- 参数
- 邻居方向

也就是：

- 外层按参数划分
- 内层按左右邻居划分

### 5.2 Halo 计算语义

对 rank `r`：

- `my_inputs`
  - 本 rank 读取到的全局位置
- `my_outputs`
  - 本 rank 写出的全局位置

对邻居 rank：

- `recv halo`
  - 我需要读，但由邻居生产的数据
- `send halo`
  - 我生产，且邻居需要读取的数据

### 5.3 Region 路线里已经接上的 halo 部分

在 region 代码生成中，halo 已经真正接入：

- `ctx` 中保存 `halo_<param>`
- `init` 阶段生成 `computeParamHalo(...)`
- `halo` 阶段执行：
  - `q.wait()`
  - D2H：把 `buf_*` 同步回 `local_*`
  - `exchangeHalo(...)`
  - H2D：把更新后的 `local_*` 写回 `buf_*`

## 6. Sibling loops

### 6.1 当前处理策略

sibling loops 是 region 路线尚未接通的部分。

当前策略是：

- 如果 region 内存在 sibling loops
  - 直接回退到稳定 wrapper 路径

### 6.2 难点

region 路线里，权威状态不再是原始 host tensor，而是：

- `ctx.local_*`
- `ctx.pack_*`
- `ctx.buf_*`

因此 sibling loop 如果继续直接操作原始 tensor，会与 region 生命周期脱节：

- kernel 更新发生在 `ctx.buf_*`
- halo 更新发生在 `ctx.local_*`
- host 侧直接操作原 tensor 无法自然并入 region 状态

### 6.3 已有 sibling helper

下面这些 helper 已经拆到独立模块中并参与编译：

- `rewriteMPISiblingBody(...)`
- `buildMPISiblingDenseSyncCode(...)`
- `rewriteMPIIndexExpr(...)`
- `parseForLoopBoundsMPI(...)`

它们的目标方向是：

- 从 AST / 源码层重写 sibling loop
- 建立 host-side dense / sparse bridge
- 再同步回 region 局部状态

这些 helper 目前属于：

- 已编译校验
- 尚未接入主 rewrite 流程

## 7. 代码结构

### 7.1 公开入口

- [Rewriter_MPI.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/Rewriter_MPI.cpp)

公开入口只保留：

- `Rewriter::rewriteMPI()`
- `Rewriter::rewriteMPI_Stencil()`
- `Rewriter::rewriteMPI_Region()`

### 7.2 已接入 CMake 的 MPI 模块

当前参与构建的 MPI 模块是：

- [Rewriter_MPI_Common.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/include/Rewriter_MPI_Common.h)
- [Rewriter_MPI_Analysis.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Analysis.cpp)
- [Rewriter_MPI_Types.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Types.cpp)
- [Rewriter_MPI_Pattern.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Pattern.cpp)
- [Rewriter_MPI_Wrapper_Codegen.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Wrapper_Codegen.cpp)
- [Rewriter_MPI_Region_Policy.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Region_Policy.cpp)
- [Rewriter_MPI_Region_Codegen.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Region_Codegen.cpp)
- [Rewriter_MPI_Region_Sibling.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Region_Sibling.cpp)

### 7.3 模块职责

#### `Analysis`

负责：

- AST 读写分析
- return 收集
- broadcast 判定

#### `Types`

负责：

- MPI datatype 映射
- payload 计数
- 视图类型推导

#### `Pattern`

负责：

- split / binding 元数据整理
- pattern 初始化代码
- local calc wrapper
- pack builder 表达式

#### `Wrapper_Codegen`

负责：

- 稳定主线 `rewriteMPI()` 的 wrapper 生成

#### `Region_Policy`

负责：

- region init / sync 的传输策略分析
- shell call 参数提取

#### `Region_Codegen`

负责：

- region `ctx / init / submit / halo / sync` 主代码生成

#### `Region_Sibling`

负责：

- sibling loop WIP helper

### 7.4 参考源文件

以下两个文件保留在工作区里，方便对照阅读，但不参与当前构建：

- [Rewriter_MPI_Wrapper.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Wrapper.cpp)
- [Rewriter_MPI_Region.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Region.cpp)

## 8. 当前行为总结

### 8.1 `rewriteMPI()`

这是稳定 wrapper 路径，适用于：

- 无 region
- region 条件不满足
- region 内存在 sibling loops

### 8.2 `rewriteMPI_Region()`

这是 region 路径入口。

它由 `rewriteMPI_Stencil()` 驱动，生成真正的 region 代码，并在以下条件下被调用：

- 命中 buffer region
- `siblingForStmts.empty()`

## 9. 已完成能力

已经完成并影响当前行为的能力包括：

- AST 真实读写模式推导
- item-space / binding 统一规划
- pack-map 驱动的精确 scatter / gather
- 本地 `View1D / View2D` 紧凑执行
- root 写回后的选择性广播
- 非标准 MPI 类型的字节传输兜底
- `mainAlreadyRewritten` 状态显式化
- region `ctx / init / submit / halo / sync` 骨架
- `MPIPlanner.h` 中的 halo runtime 支持
- sibling loops 的保守 fallback
- MPI 代码从大文件拆分为可独立构建的子模块

## 10. 推荐继续工作

### 10.1 第一优先级：完成 sibling loops 的 region 策略

需要在以下方向中选定一条并完成落地：

- 路线 A：把 sibling loops 直接翻译成 region 专用 helper
- 路线 B：建立 host bridge，再把结果同步回 `ctx.local_* / ctx.buf_*`

无论选哪条，都必须围绕 region 的真实状态容器实现。

### 10.2 第二优先级：移除 sibling fallback

当 sibling loops 稳定接入 region 生命周期后，可以移除：

- “有 sibling loops 就走稳定 wrapper”

### 10.3 第三优先级：继续细拆 region 子模块

最适合继续下拆的是：

- [Rewriter_MPI_Region_Codegen.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Region_Codegen.cpp)
- [Rewriter_MPI_Region_Sibling.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Region_Sibling.cpp)

建议的继续拆分方向：

- `region-ctx`
- `region-init`
- `region-submit`
- `region-halo-sync`
- `region-sibling-rewrite`
- `region-sibling-sync`

## 11. 推荐接手顺序

如果下一个模型刚进入这个目录，推荐按下面顺序建立上下文：

1. 先读：
   - [translator.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/translator.cpp)
   - [Rewriter_MPI.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/Rewriter_MPI.cpp)
2. 再读：
   - [Rewriter_MPI_Region_Codegen.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Region_Codegen.cpp)
   - [Rewriter_MPI_Region_Policy.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Region_Policy.cpp)
   - [Rewriter_MPI_Region_Sibling.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Region_Sibling.cpp)
   - [MPIPlanner.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/MPIPlanner.h)
3. 再看：
   - [plan.md](/Volumes/QUQ/working/dacpp/clang/tools/translator/docs/plan.md)
4. 然后决定继续方向：
   - sibling loops
   - region 细拆
   - MPI 回归修复

## 12. 构建与回归

### 12.1 构建

已确认通过：

```bash
cmake --build /Volumes/QUQ/working/dacpp/build --target translator -j8
```

### 12.2 MPI 回归

建议继续执行：

```bash
cd /Volumes/QUQ/working/dacpp/clang/tools/translator
bash test_mpi.sh
```

重点关注：

- `tests/liuliang1.0/liuliang.dac.cpp`
- `tests/MDP1.0/mdp.dac.cpp`

## 13. 一句话理解这套 MPI 实现

这套实现可以直接理解成：

- 用稳定的 item-space / pack-map wrapper 保证通用正确性
- 用 region 路线把时间推进类程序推进到“一次 init、每步 halo、最终 sync”的执行模型
- 目前决定 region 覆盖范围的关键因素，是 sibling loops 何时稳定并入 region 生命周期
