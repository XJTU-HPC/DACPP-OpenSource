# MPI Stencil 当前处理逻辑

更新时间：2026-05-06

这份文档专门解释当前 `clang/tools/translator` 里 MPI stencil 的处理逻辑：什么情况下走普通 MPI wrapper，什么情况下走 stencil path，识别逻辑怎么做，数据流怎么走，以及现在的 stencil 是怎样建立在 MPI wrapper 基础上的。

## 1. 入口分流：什么时候走普通 MPI wrapper，什么时候走 stencil

MPI 模式入口在：

- `clang/tools/translator/translator.cpp`
- `clang/tools/translator/rewriter/lib/Rewriter_MPI.cpp`
- `clang/tools/translator/rewriter/lib/Rewriter_MPI_Stencil.cpp`

整体分成两层判断。

第一层是文件级入口：

- 如果 `dacppFile->hasMPIStencilSites()` 为假，`Rewriter::rewriteMPI()` 直接走普通 MPI wrapper。
- 如果存在已登记的 stencil site，`Rewriter::rewriteMPI()` 会转到 `rewriteMPIStencil()`。

第二层是站点级回退：

- 即使文件进入了 `rewriteMPIStencil()`，也不是每个 `<->` 都一定能走 loop-lowered stencil。
- 已登记的 loop site 会被改写成 `ctx/init/run/materialize` 结构。
- 没登记进去、或者只是同一 shell/calc 的其他 `<->` occurrence，仍然会在 `rewriteMPIStencil()` 的尾部回退成普通 wrapper 调用。

所以当前实现不是“要么全文件 stencil，要么全文件 wrapper”，而是：

- 能安全 hoist 的 loop site 走 stencil
- 其余 site 继续走 wrapper

## 2. Stencil site 是怎么识别出来的

识别第一步发生在 `translator.cpp` 的 AST 匹配阶段。

对每个 `<->`：

1. 先找到它对应的 shell call 和 calc function。
2. 向上找最外层包裹它的 `for` / `while`。
3. 检查 shell 实参能不能安全 hoist 到这个外层 loop 之前。

当前 hoist guard 很保守。只要 shell 实参引用了 loop 体里临时声明的变量或临时 view，就不把这个 `<->` 登记成 `MPIStencilSite`。

这一步的目的很直接：

- stencil path 需要把 `init(ctx, ...)` 提到 loop 外
- 如果实参绑定在每轮迭代里都可能变，就不能安全复用 ctx

所以现在只有“外层 loop 稳定、实参稳定、可以 loop lowering”的 `<->` 会进入 stencil path。

## 3. `rewriteMPIStencil()` 实际做了什么

一旦某个 site 被识别成 MPI stencil，`rewriteMPIStencil()` 会把原来的 loop 内 `<->` 改写成四段式：

```cpp
__dacpp_mpi_stencil_ctx_xxx ctx;
__dacpp_mpi_stencil_init_xxx(ctx, ...);
for (...) {
    __dacpp_mpi_stencil_run_xxx(ctx, ...);
}
__dacpp_mpi_stencil_materialize_xxx(ctx, ...);
```

含义分别是：

- `ctx`：保存 MPI rank/size、queue、AccessPattern、PackPlan、layout、distributed cache、route plan、halo plan，以及可复用的 profiling / writeback metadata。
- `init()`：只做一次的稳定工作，包含 pattern/pack/layout 初始化、local item range 计算、初始 READ cache seed、route/halo plan 构建。
- `run()`：每步真正执行的路径，包含 local kernel、必要的 distributed publish、read-cache transition、boundary-local update。
- `materialize()`：只在 loop 末尾把分布式 reader cache 收敛回 root-visible tensor。

这一步已经说明一个关键点：

当前 stencil 不是一套脱离 wrapper 的新 backend，而是把原来“每轮都调用一次 wrapper”的形式，拆成了可 hoist 的 `init()` 和每步轻量的 `run()`。

## 4. Phase C / no-root distributed path 是怎么判定的

真正决定能不能进入 Phase C distributed path 的，不是 `translator.cpp`，而是：

- `clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Stencil_Analysis.cpp`
- 入口函数：`analyzeDistributedStencilSite(...)`

它要求当前 site 已经经过 loop lowering，并且 `BufferRegionPlan` 已经建立起来。没有这一层，就不会进入 Phase C。

当前核心 guard 有这些。

### 4.1 参数类型 guard

- 所有参与的 stencil tensor 必须都是 `dacpp::Vector` 或都是 `dacpp::Matrix`
- 不支持 mixed Vector/Matrix site
- 不支持 transport 仍然需要 `READ_WRITE` 的 kernel param

这就是为什么 `jacobi1.0` 现在必须继续停在 fallback path：

- 它是 mixed Matrix/Vector site
- `analyzeDistributedStencilSite()` 会直接给出 `phase-c does not support mixed Vector/Matrix sites`

### 4.2 post-shell sibling stmt 识别

在 loop lowering 之后，stencil shell 后面的 sibling stmt 会被逐条分析。

当前 Matrix 站点能识别的主要类别是：

- 2D distributed followup route
- 2D read-cache state transition
- 2D boundary-local update
- 可支持的 root-centric helper region

如果某条 stmt 能被识别成 distributed route / transition / boundary，就会进入 no-root distributed path。

如果某条 stmt 虽然不能 distributed，但还能用 root helper 表达，就会把 site 标成：

- `supported = true`
- `hasRootBridge = true`

如果某条 stmt 既不能 distributed，也不能进 helper fallback，这个 site 就直接不支持 Phase C。

## 5. 当前“识别出来的结构”具体长什么样

当前 `DistributedStencilSitePlan` 里最关键的几类 IR 是：

- `followupMappings`
- `readCacheTransitions`
- `boundaryLocalUpdates`
- `hasRootBridge`

它们分别表达：

- writer 到后续 reader 的 distributed route
- writer cache 到 reader cache 的 state transition
- 不需要跨 rank 的本地边界修补
- 是否还残留必须 root-centric 处理的后续逻辑

以当前 wave 为例，分析结果应当稳定保持：

- `matCur -> matPrev` 是 read-cache transition
- `matNext -> matCur` 是 distributed route
- 边界清零是 boundary-local update
- 没有 root helper / root bridge

以 `stencil1.0` 为例：

- `matOut -> matIn` 是 distributed route
- 边界 copy 是 boundary-local update

以 `jacobi1.0` 为例：

- mixed Matrix/Vector guard 失败
- 不会进入 Matrix Phase C

## 6. 普通 MPI wrapper 的数据流

普通 wrapper 仍然是整个实现的基础形态。

它的典型数据流是：

```text
root pack globals
-> MPI_Scatterv / local input
-> local kernel
-> MPI_Gatherv writer results
-> root apply_writeback_by_globals
-> optional MPI_Bcast
```

其中几个关键公共能力，不管 stencil path 还是 wrapper path 都在复用：

- `buildPrelude()`
- `buildLocalCalcCode()`
- `AccessPattern` / `PackPlan`
- `GatheredIndexLayout`
- 输出一致性分类
- root-only print rewrite
- root gather 后的 `writeback` 逻辑

也就是说，stencil path 不是把 wrapper 推翻重写，而是复用了 wrapper 已经稳定的 pack/layout/writeback/output 体系。

## 7. Stencil path 的数据流

stencil path 的数据流和普通 wrapper 最大的区别，是把“每步 root 收敛”换成“持久 distributed cache”。

当前目标形态是：

```text
init:
  build pattern / pack / layout / exchange plan
  seed initial reader cache

run per step:
  local kernel on local cache
  read-cache transition
  publish writer local writes to remote reader cache
  boundary-local update

materialize once:
  gather distributed reader cache back to root-visible tensor
```

和普通 wrapper 相比，最大的变化有两个：

1. root gather / broadcast 不再每步发生
2. loop 内后续数据流不再靠 host 语句顺次执行，而是被压进 `ctx` 里的 distributed state machine

所以 `rewriteMPIStencil()` 在 no-root distributed path 下，会把已经被 Phase C 吞掉的 followup stmt / boundary stmt 直接删掉，避免它们在 host 侧再跑一遍。

## 8. 当前 stencil 是怎么建立在 MPI wrapper 基础上的

可以把现在的实现理解成四层叠加。

第一层：普通 MPI wrapper

- 解决基本的 MPI init/finalize
- 解决 local kernel 调用
- 解决 root gather / broadcast / writeback

第二层：loop lowering

- 把 loop 内 `<->` 拆成 `ctx/init/run/materialize`
- 让稳定 metadata 能离开 time-step loop

第三层：Phase C distributed cache

- 把每步 root gather/helper/broadcast 改成 distributed cache publish / transition / halo exchange
- 让真正 observable 的 materialize 延迟到 loop 末尾

第四层：wave-specific fast path

- span-pair transition/publish
- guarded direct kernel

这四层是“叠加”关系，不是替代关系。

## 9. 当前 wave 的专用逻辑

现在 wave 是单独优先推进的。

### 9.1 span-pair transition / publish

在 `StencilExchange.h` / `StencilTypes.h` 里，wave 现在有一套更轻的 span-pair helper：

- `build_span_pairs_from_slots(...)`
- `scatter_values_by_span_pairs_into(...)`
- `publish_local_writes_with_span_pairs_or_exchange_cache_only(...)`

它不是新的 generalized Phase C 语义，只是对当前 wave row-block 规则访问做更轻的 slot 表达。

### 9.2 direct kernel

在 `Rewriter_MPI_Stencil_Codegen.cpp` 里，当前还加入了 guarded wave direct-kernel path。

只有在这些条件都满足时才会启用：

- `waveEqShell` / `waveEq`
- distributed site supported
- 没有 root bridge
- 只有一条 distributed route
- 只有一条 read-cache transition
- 只有 3 个 shell 参数

启用后会：

- 在 `init()` 里预计算每个 item 对应的 `center/up/down/left/right/prev/next` slot
- 在 `run()` 里直接生成 wave 专用 kernel
- 绕过通用 `View1D/2D` 和 `waveEq_mpi_local(...)`

这就是最近这轮 wave kernel 内优化的主体。

## 10. 为什么现在还没有把所有 stencil 都做成同一条快路径

原因很简单：当前实现是故意保守的。

- `stencil1.0` 虽然已经是 no-root distributed path，但当前不主动扩成 wave 那样的 direct kernel
- `FOuLa1.0` 继续走 compatibility wrapper
- `jacobi1.0` 必须继续停在 mixed Matrix/Vector fallback

当前策略是先把 wave 这条最值得的热点路径压实，再决定哪些东西值得泛化。

## 11. 现在怎么看一份生成代码是不是走对了

当前最重要的结构断言有这些。

`waveEquation1.0`：

- 必须能看到 `matCur->matPrev` read-cache transition
- 必须能看到 `matNext->matCur` distributed route
- 必须保留 boundary-local update
- 不得出现 `partial-exchange disabled`
- 不得出现 root helper / root bridge 痕迹

`jacobi1.0`：

- 必须继续停在 mixed Matrix/Vector fallback

所有生成的 `*.dac_sycl_buffer.cpp`：

- 不得残留 `<->`

## 12. 现阶段最值得继续改的地方

在 direct-kernel 落地之后，当前 wave 的剩余大头已经更清楚了：

- `kernel` 仍大，但已经不是绝对碾压
- `read_transition + publish` 开始和 kernel 站到同一量级

所以再往下推进时，更值得看的方向是：

- `matCur -> matPrev` 的 multi-state alias/swap
- kernel / transition / publish 生命周期融合
- row-span halo / publish 压缩

而不是再继续围绕小规模 slot helper 做零碎微调。
