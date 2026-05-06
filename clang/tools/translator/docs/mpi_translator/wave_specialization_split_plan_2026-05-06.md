# Wave 特化逻辑与通用 Phase C 逻辑完全分离方案

更新时间：2026-05-06

这份文档不是性能记录，而是一份代码整理设计文档。目标很明确：

- 把 `waveEquation1.0` 的特化逻辑和 generic Phase C 逻辑完全分开
- 让后续 halo 实现试验发生在清楚的边界上
- 给出可执行的拆分方案、迁移顺序和验收标准

本文里的“完全分离”不是指：

- 把所有代码都复制一份
- 或者把 generic runtime 推翻重写

本文里的“完全分离”是指：

- generic Phase C 只保留通用语义、通用 plan、通用 exchange/runtime
- wave specialization 只存在于明确的 wave 专用入口、wave 专用状态、wave 专用 helper 中
- generic 主路径里不再散落 wave 特化 if-branch

## 1. 背景和动机

当前 wave 的优化已经不只是一个窄 patch，而是横跨了三层：

1. codegen 层
   - `Rewriter_MPI_Stencil_Codegen.cpp`
2. runtime state 层
   - `StencilTypes.h`
3. helper / exchange 层
   - `StencilExchange.h`

这些优化包括：

- wave direct kernel
- `wave_direct_slots_buffer`
- `local_next` sparse clear
- halo runtime preallocation
- span-pair fast path
- row-copy block fast path

从性能角度看，这些改动是值得保留的。

但从代码结构看，当前已经出现一个明显问题：

- wave 特化不是独立层
- 而是嵌入 generic Phase C 主路径内部

只要接下来还想继续试新的 halo 方案，这种结构会越来越难维护，也越来越难解释 benchmark 结果。

## 2. 当前代码的耦合现状

### 2.1 codegen 层耦合

主文件：

- `clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Stencil_Codegen.cpp`

当前它大约 2200 行，已经同时承担：

- generic ctx / init / run / materialize 生成
- wave ctx 字段生成
- wave init metadata 构造
- wave direct kernel 生成
- wave read-transition fast path 生成
- wave publish fast path 生成

典型耦合点：

- ctx 里直接塞 wave 专用状态
  - `use_wave_span_pairs`
  - `use_wave_direct_kernel`
  - `wave_direct_slots_buffer`
  - `wave_direct_next_stale_slots`
  - 位置：`Rewriter_MPI_Stencil_Codegen.cpp:788-801`

- generic route 初始化里直接插 wave fast-path metadata
  - 位置：`Rewriter_MPI_Stencil_Codegen.cpp:1080-1150`

- generic run 的 read-transition / publish 段里直接分 wave 分支
  - 位置：`Rewriter_MPI_Stencil_Codegen.cpp:1830-1905`

这意味着：

- 现在没有“generic codegen”和“wave codegen”两个明确阶段
- 只有一条大的字符串拼接流程，中间插入 wave 逻辑

### 2.2 runtime state 层耦合

主文件：

- `clang/tools/translator/dpcppLib/include/mpi/StencilTypes.h`

当前 `DistributedTensorState<T>` 同时保存：

- generic distributed tensor state
- generic route-level exchange/halo state
- wave route local fast-path state
- wave read-transition fast-path state

典型字段：

- `local_write_spans_by_route`
- `local_target_spans_by_route`
- `local_row_copy_blocks_by_route`
- `use_span_pairs_by_route`
- `use_row_copy_blocks_by_route`
- `read_cache_transition_*`
- `halo_runtimes_by_route`

位置：`StencilTypes.h:77-108`

这意味着：

- generic distributed tensor state 和 wave fast path state 没有分层
- 读类型定义时无法区分“语义必要状态”和“特化执行加速状态”

### 2.3 helper 层耦合

主文件：

- `clang/tools/translator/dpcppLib/include/mpi/StencilExchange.h`

当前这里既有：

- generic `ExchangePlan` / `HaloExchangePlan`
- generic `exchange_values_by_slots`
- generic `exchange_values_by_halo_spans`
- prepared halo runtime
- local-only span pair filter
- row-copy block build
- publish span-pair / row-block fast path

典型函数：

- `build_local_span_pairs_from_slots(...)`
  - `StencilExchange.h:617`
- `build_contiguous_row_copy_blocks_from_span_pairs(...)`
  - `StencilExchange.h:658`
- `publish_local_writes_with_span_pairs_or_exchange_cache_only(...)`
  - `StencilExchange.h:1398`

这意味着：

- generic exchange runtime 和 wave local fast path 没有边界
- 后续只要继续试新的 wave halo 方案，helper 层会继续膨胀

## 3. 为什么必须把“特化逻辑”和“一般逻辑”彻底分开

这里说的“彻底分开”不是为了代码风格，而是为了三件工程上的事。

### 3.1 让 generic Phase C 重新成为稳定基线

generic Phase C 应该只回答这些问题：

- distributed cache 怎么初始化
- route / exchange / halo plan 怎么建立
- fallback 什么时候触发
- materialize 怎么做
- 非 wave site 怎么运行

只要 wave 特化继续混在 generic 主路径里，generic Phase C 就很难再作为稳定基线存在。

### 3.2 让 wave 变成可替换的 specialization backend

后续 wave 继续试 halo 方案时，理想结构应该是：

- generic route / halo plan 保持不变
- wave 只替换自己的 local fast path / publish backend / halo backend

如果做不到这一点，那么每试一个新 backend，都要改 generic 主流程，比较结果就不干净。

### 3.3 让 profile 归因变干净

当前 benchmark 如果变化，很容易同时受这些因素影响：

- generic fallback 是否绕开
- wave direct kernel 是否命中
- span-path 是否命中
- row-copy path 是否命中
- halo runtime 是否复用

如果 wave 特化单独成层，profile 解释就会干净很多：

- generic 变了，说明基础结构变了
- wave backend 变了，说明特化执行方式变了

## 4. 本文定义的“完全分离”是什么

为了避免口头上说分离、实现上又回到老路，这里给出明确边界。

### 4.1 generic 层允许保留的内容

generic 层允许保留：

- site 是否 eligible 的通用判定
- distributed tensor 的通用 plan/state
- `ExchangePlan` / `HaloExchangePlan`
- prepared halo runtime 这种通用复用机制
- generic route publish / transition / materialize helper
- generic `ctx/init/run/materialize` 结构

### 4.2 generic 层不应再出现的内容

generic 主路径不应再直接出现：

- `use_wave_*` 开关
- `wave_direct_*` metadata
- wave read-transition row-copy metadata
- wave publish row-copy metadata
- “如果是 wave 就走这条分支”的主流程内联判断

换句话说，generic 主路径不应该知道 wave 的内部优化细节。

### 4.3 wave 特化层应该承担的内容

wave 特化层负责：

- wave direct-kernel metadata 和 direct-kernel run path
- wave `local_next` sparse clear 策略
- wave read-transition local fast path
- wave publish local fast path
- wave halo backend 选择和试验

generic 层只需要知道：

- 这个 site 使用了某个 specialization backend

而不需要知道：

- 这个 backend 内部是 span-pair 还是 row-copy，还是别的 halo 实现

### 4.4 这轮整理明确不做的事

为了保证“完全分开”是一次结构整理，而不是又一次边整理边扩需求，这轮设计明确不做下面几类事：

- 不把 generic `ExchangePlan` / `HaloExchangePlan` 复制成一套 wave 私有 plan
- 不把 prepared halo runtime 这类通用复用机制下沉成 wave 私有实现
- 不引入新的 CLI 选项、translator 开关或用户可见语义差异
- 不在拆边界的同时继续叠加新的 kernel 算子特化
- 不预先设计覆盖所有 future specialization 的大而全框架

这轮的目标只有一个：

- 先把 generic 骨架、specialization dispatch、wave backend 三层边界做实

## 5. 推荐的目标架构

推荐拆成三层。

### 5.1 Layer A: Generic Phase C 骨架

职责：

- 通用 analysis 结果消费
- 通用 ctx/init/run/materialize 骨架
- 通用 distributed plan/state
- 通用 publish / transition / materialize fallback

主要位置：

- `Rewriter_MPI_Stencil_Codegen.cpp` 的 generic 部分
- `StencilTypes.h` 的 generic state
- `StencilExchange.h` 的 generic exchange/runtime

### 5.2 Layer B: Specialization dispatch

职责：

- 判断当前 site 是否启用某个 specialization
- 把 wave specialization 接到 generic 骨架上
- 保证 generic 和 specialization 的边界是显式的

这层不做具体优化动作，只做调度。

### 5.3 Layer C: Wave specialization backend

职责：

- wave ctx 扩展字段
- wave init-time metadata 构造
- wave run-time fast path
- wave publish / halo backend 试验

这一层是以后试不同 halo 方案的主要实验区。

## 6. 推荐的文件 / 类型拆分边界

这里给出的是推荐边界，不要求一次到位。

### 6.1 Codegen 文件拆分建议

当前：

- `Rewriter_MPI_Stencil_Codegen.cpp`

建议目标：

- `Rewriter_MPI_Stencil_Codegen.cpp`
  - 保留 generic 主骨架
- `Rewriter_MPI_Stencil_Codegen_Wave.cpp`
  - 只放 wave specialization code emission
- `Rewriter_MPI_Stencil_Codegen_Internal.h`
  - 共享的内部 helper 声明

至少应把这些逻辑从 generic 主流程抽出来：

- wave ctx field emission
- wave init specialization emission
- wave direct kernel emission
- wave read-transition emission
- wave publish emission

### 6.2 Runtime state 类型拆分建议

当前最大问题是 `DistributedTensorState<T>` 过载。

建议引入：

```cpp
template <typename T>
struct GenericRouteRuntimeState {
    ExchangePlan exchange_plan;
    HaloExchangePlan halo_plan;
    HaloExchangeRuntime<T> halo_runtime;
};

template <typename T>
struct WaveRouteFastPathState {
    std::vector<SlotSpan> local_write_spans;
    std::vector<SlotSpan> local_target_spans;
    std::vector<ContiguousRowCopyBlock> local_row_copy_blocks;
    bool use_span_pairs = false;
    bool use_row_copy_blocks = false;
};

template <typename T>
struct WaveTransitionFastPathState {
    std::vector<int32_t> source_slots;
    std::vector<int32_t> target_slots;
    std::vector<SlotSpan> source_spans;
    std::vector<SlotSpan> target_spans;
    std::vector<ContiguousRowCopyBlock> row_copy_blocks;
    bool use_span_pairs = false;
    bool use_row_copy_blocks = false;
};
```

然后：

- generic distributed tensor state 只保留通用语义和通用 route runtime
- wave 特化自己的 state 放到 wave 专用 ctx 扩展里

关键点：

- 不要把 wave fast path metadata 继续塞进 generic distributed tensor state

### 6.3 Helper 文件拆分建议

建议最终至少分成两组：

1. generic exchange runtime
   - `StencilExchange.h`
2. specialization local fast path / backend helper
   - `StencilFastPath.h`
   - 或更明确的 `WaveExchangeSpecialization.h`

generic 文件里只保留：

- plan build
- halo runtime preallocation
- MPI exchange
- generic scatter / pack / unpack

wave specialization 文件里再保留：

- local-only span filter
- row-copy block build
- wave publish fast path
- wave transition fast path
- future wave halo backend

### 6.4 依赖方向和 ownership 规则

“完全分开”最终要落实成稳定的依赖方向。建议把规则写死：

1. generic codegen 不能直接读写 wave 专用 state 字段
   - generic 只面向 specialization dispatch 暴露的接口

2. generic runtime/helper 不能包含 wave fast-path 细节
   - generic helper 只处理通用 pack/unpack/exchange/materialize

3. wave backend 可以依赖 generic plan/runtime
   - 但只能把 generic 层当作下层能力使用，不能反向把自己的 metadata 塞回 generic 类型

4. wave state 必须挂在 wave specialization 自己的 ctx 扩展下
   - 不能继续把 `use_wave_*`、row-copy metadata、transition metadata 塞回 `DistributedTensorState<T>`

5. specialization dispatch 只负责“选谁”，不负责“怎么快”
   - dispatch 不内联 span-pair、row-copy、halo backend 的具体逻辑

如果以上五条里有任何一条不成立，就不能算“完全分开”。

## 7. 推荐的迁移顺序

这里很关键：要先拆边界，再拆文件，再换 backend。

### Phase 1: 逻辑分段，不改行为

目标：

- 先把 `Rewriter_MPI_Stencil_Codegen.cpp` 里的 wave 逻辑提成独立 helper
- 仍然可以先放在同一个文件里

验收标准：

- 生成代码行为不变
- correctness 不变
- benchmark 不要求提升

### Phase 2: generic state / wave state 拆开

目标：

- 把 `DistributedTensorState<T>` 里的 wave fast-path metadata 移出去
- 建立 wave specialization 自己的 state

验收标准：

- generic 类型定义不再直接出现 wave fast-path 细节
- wave 特化仍可正常运行

### Phase 3: helper 分层

目标：

- generic exchange/runtime 和 wave fast-path helper 分文件、分章节
- generic helper 不再直接暴露 wave 内部结构

验收标准：

- `StencilExchange.h` 回到通用语义
- wave backend helper 独立存在

### Phase 4: specialization backend 可替换化

目标：

- wave publish / halo backend 变成可切换实现
- 开始真正比较 halo backend

验收标准：

- 可以在不改 generic 主流程的前提下替换 wave halo backend

### Phase 5: 清理遗留兼容分支

目标：

- 删除 generic 主路径里残留的 wave 条件分支
- 删除只为迁移期保留、但最终不再需要的过渡 helper
- 把波形特化相关 benchmark / profiling 开关整理到 wave backend 自己的调试边界内

验收标准：

- generic 主文件和 generic helper 中不再存在“迁移期临时兼容 wave”的残余逻辑
- 代码阅读时可以仅通过目录和类型边界定位 generic 或 wave 责任范围

## 8. 拆分后的 halo 试验空间

如果按上面的方式拆开，后续最值得试的 halo 方向有：

### 8.1 generic prepared halo runtime 继续优化

这条仍然最稳：

- 改动小
- correctness 风险低
- 对 non-wave 也可能复用

### 8.2 wave row-block contiguous peer slice backend

这是 wave 最自然的下一步：

- 本地 overlap 已经有 row-copy block
- 可以继续探索跨 rank halo 是否也能收束成更直接的 row-block backend

### 8.3 persistent request / datatype / device-side halo

这些在结构没拆干净之前不建议碰。

拆完之后，它们才真正变成：

- backend 试验

而不是：

- 改整条主流程

## 9. 风险和不建议路线

### 9.1 不建议一次性大重构 generic Phase C

当前 generic Phase C 已经承担：

- correctness fallback
- non-wave site 运行
- benchmark 基线

如果为了分离 wave 先大改 generic，风险过大。

### 9.2 不建议边整理边继续塞新 wave 优化

如果结构边界还没稳定，就继续叠新特化逻辑，整理工作会不断返工。

更好的顺序是：

- 先定边界
- 再恢复后续优化

### 9.3 不建议先为了“抽象优雅”做大框架

当前最需要的是：

- 清楚边界
- 可替换 backend
- 可解释 benchmark

不是先做一个覆盖未来所有 specialization 的大抽象框架。

## 10. 结构整理完成后的验收标准

如果要说“特化逻辑和一般逻辑已经完全分开”，至少应满足下面这些条件。

### 10.1 codegen 验收

- generic codegen 主文件中不再直接出现 `use_wave_*` / `wave_direct_*`
- wave 特化生成逻辑有独立入口
- generic 主流程只通过 specialization dispatch 接入 wave

### 10.2 类型验收

- generic distributed tensor state 不再直接持有 wave fast-path metadata
- wave route / transition fast-path state 有自己的专用类型

### 10.3 helper 验收

- generic exchange/runtime helper 中不再直接掺 wave publish fast path
- wave local fast-path / halo backend helper 独立成组

### 10.4 工程验收

- 可以只改 wave backend，而不改 generic 主流程
- 可以单独 benchmark 两个 wave backend
- `stencil1.0` / `jacobi1.0` 等 generic correctness path 不受 wave backend 改动影响

### 10.5 代码审查验收

除了 correctness 和 benchmark，还需要一个结构层面的 review checklist：

- 看 generic codegen 主骨架时，不需要理解 wave metadata 才能读懂主流程
- 看 `DistributedTensorState<T>` 时，可以一眼区分通用状态和特化状态
- 看 generic helper 时，不会遇到 wave-only 的 publish/transition 快路径实现
- 新增一种 wave halo backend 时，改动范围主要局限在 wave specialization 文件组内

## 11. 建议结论

如果目标是继续把 `waveEquation1.0` 当成 halo / publish 试验场，那么仅仅继续沿当前代码慢慢优化，不是最可行的长期路线。更可行的路线是：先把 wave specialization 和 generic Phase C 做成真正的分层结构，让 generic 重新成为稳定骨架，让 wave 变成清楚的 specialization backend，然后再继续比较不同的 halo 实现方式。
