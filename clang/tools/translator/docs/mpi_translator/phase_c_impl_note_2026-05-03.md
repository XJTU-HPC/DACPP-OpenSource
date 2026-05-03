# Phase C 本次实现说明

日期：2026-05-03

这份文档只讲这一次具体做了什么，不重复展开整个 MPI stencil 设计背景。目标是把“这一步到底落了哪些代码、修了哪些坑、验证到了什么程度”说清楚。

## 1. 这一步的目标

这一步的目标不是一次性做完整个 Phase C，而是先把 Phase C 的第一条可运行路径落下来：

- 只覆盖 MPI stencil loop lowering 路径。
- 只覆盖 1D `dacpp::Vector`。
- 只覆盖 effective `READ/WRITE` 的 site。
- 不碰 one-shot MPI wrapper。
- 整站 fallback 仍然是默认安全规则。

更具体地说，这一步要把“distributed cache / partial exchange”从设计稿变成真实代码路径，哪怕初版只先打通一条保守的 root-bridge 正向路径。

## 2. 这一步实际改了哪些文件

运行时：

- `clang/tools/translator/dpcppLib/include/mpi/Common.h`
- `clang/tools/translator/dpcppLib/include/mpi/Pack.h`

分析 / 分类：

- `clang/tools/translator/rewriter/include/Rewriter_MPI_Common.h`
- `clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Analysis.cpp`
- `clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Stencil_Analysis.cpp`

codegen：

- `clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Stencil_Codegen.cpp`
- `clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_PostRegion.cpp`

测试：

- `clang/tools/translator/test_mpi.sh`
- `clang/tools/translator/tests/auto_test.sh`
- `clang/tools/translator/tests/mpiDistributedStencil1D/mpiDistributedStencil1D.dac.cpp`
- `clang/tools/translator/tests/mpiDistributedStencil1D/mpi_expect.txt`

文档：

- `clang/tools/translator/docs/mpi_translator/mpi_stencil_status.md`
- `clang/tools/translator/docs/mpi_translator/phase_c_impl_note_2026-05-03.md`

## 3. 运行时层具体做了什么

这一步先把 Phase C 需要的 runtime 结构补齐。

新增的数据结构：

- `AllRankIndexLayout`
  - 作用：在所有 rank 上持有某个 tensor 的 all-rank global index 布局。
  - 和原来 root-only gather layout 的区别是：它用 `MPI_Allgather/MPI_Allgatherv` 初始化，所有 rank 都能看到完整 metadata。

- `PeerSlotExchange`
  - 作用：描述一次和某个 peer 之间的 slot 级收发。

- `ExchangePlan`
  - 作用：保存 `send_transfers` / `recv_transfers`，并带 `supported` / `unsupported_reason`。

- `DistributedTensorState<T>`
  - 作用：保存一个 distributed-managed tensor 的本地持久 cache，以及 read/write layout、bridge plan、状态位。

新增的 helper：

- `init_all_rank_index_layout(...)`
- `validate_unique_writers(...)`
- `build_exchange_plan_from_layouts(...)`
- `unpack_values_by_slots_into(...)`
- `exchange_values_by_slots(...)`

这里的关键点有两个：

- steady-state 交换形式已经定成非阻塞点对点：`MPI_Irecv` / `MPI_Isend` / `MPI_Waitall`
- Phase C 仍保留第一次 `init()` 的 root scatter seed，不试图把初始装载也改写成 distributed exchange

## 4. 分析层具体做了什么

这一步把 `DistributedFollowup` 从“占位枚举值”变成了真正会影响 codegen 的分析结果。

新增了站点级分析入口：

- `analyzeDistributedStencilSite(...)`
- `tensorUsesDistributedFollowup(...)`

当前 static eligibility 规则：

- 必须在 `rewriteMPIStencil()` 的 loop-lowered site 内
- 必须是 1D `dacpp::Vector`
- effective mode 不能出现 `READ_WRITE`
- post-shell sibling statement 如果存在，必须都能被当前 root-centric helper 识别

这里最重要的一条修正是：

- `DistributedFollowup` 现在只会出现在 loop-lowered MPI stencil site 上
- 普通 wrapper 路径和不在 outer loop lowering 内的 `<->` 不会误被标成 `DistributedFollowup`

这条 guard 后来证明非常关键。没有它的话，`decay1.0` 和几组 broadcast-analyze 测试会被错误卷进 Phase C 分类。

## 5. stencil codegen 具体做了什么

### 5.1 新增 partial-exchange 分支

`buildStencilWrapperCode(...)` 现在会生成两条 run path：

- 旧路径：root-centric `Scatterv -> kernel -> Gatherv -> apply_writeback -> optional Bcast`
- 新路径：partial-exchange path

同时在 `init()` 里新增：

- `ctx.use_partial_exchange`
- `ctx.partial_exchange_disable_reason`
- 每个 eligible tensor 的 `ctx.dist_*`

行为上：

- static 不支持：直接 `ctx.use_partial_exchange = false`
- static 支持但 runtime 校验失败：例如 unique writer 校验失败，也会关掉 partial path
- 一旦 partial path 被关闭，整站回退，不做站内 tensor 混合模式

### 5.2 init 阶段新增 distributed metadata 构建

对 READ tensor：

- 构建 `read_layout`
- 用原有 root scatter 逻辑做一次 cache seed
- 把 local read cache 固化到 `ctx.dist_<name>.local_cache`

对 WRITE tensor：

- 构建 `write_layout`
- 做 `validate_unique_writers(...)`

### 5.3 新增 root-bridge plan

对于被 post-shell helper 写回的 tensor，这一步加了 root-bridge 计划：

- root 作为 authoritative writer
- helper 结束后，把 root 上的新值按 slot exchange 刷回其他 rank 的 persistent read cache

当前 bridge 还是保守实现：

- writer 侧用 dense root-authoritative layout 建计划
- 不是精确 helper-written subset

但这已经足够把“persistent cache + partial refresh”的基本链路打通。

## 6. post-shell helper 具体做了什么

这一步没有推翻已有的 root-centric helper，而是在它上面接了一层 distributed cache refresh。

当前 helper 的形态变成：

- 先在 root 上执行原有 helper
- 如果当前 site 真正启用了 partial-exchange，并且这个 tensor 有 `root_bridge_plan`
- 则在 helper 末尾调用 `dacpp::mpi::exchange_values_by_slots(...)`

此外还补了一个很关键的 root-side refresh：

- root 不能只给别人发 bridge payload
- root 自己的 `ctx.dist_* .local_cache` 也必须同步刷新

否则下一轮 partial path 下，root 自己也会继续读旧 cache。

## 7. 这一步过程中实际修掉的坑

这一步不是只堆功能，也顺手修了几处实现里暴露出来的真实问题。

### 7.1 codegen const AST 断点

`Rewriter_MPI_Stencil_Codegen.cpp` 里有一个 `DeclRefExpr` 提取的 constness 编译错误，导致 translator 一开始都编不过。这个已经修掉。

### 7.2 static unsupported site 仍生成 partial branch

一开始即便 site static 不支持，generated run function 里仍然会吐出 partial branch。结果 2D fallback case 虽然逻辑上不走 partial path，生成代码本身却会因为 `mat_cols` 之类变量不完整而编译失败。

修正后：

- static unsupported site 只生成旧路径
- 真正符合 whole-site fallback 的语义

### 7.3 rootBridge tensor 识别被 `for (...) i = ...` 误伤

最早的 `rootBridgeTensors` 识别方式是拿 statement 文本，找到第一个 `=` 前的 lhs。这个在 `for (int i = 1; ...)` 这种 loop header 上会直接截错，导致 helper 实际写了 tensor 却没识别出来。

修正后：

- 不再只截第一个 `=`
- 改成在整个 statement 文本里做保守词级匹配

### 7.4 root-bridge plan payload 不对称导致卡死

最初 root bridge 的 writer layout 错用了 reader globals 的拼接版，导致 root 可能对同一个 global 发多份 payload，peer 侧 `Irecv` 长度却更短，运行时会卡在 exchange 上。

修正后：

- root bridge 的 writer layout 改成真正的 dense root-authoritative layout
- send/recv 规模重新对称

### 7.5 wrapper / non-loop site 被误判成 Phase C

没有 loop-only guard 时：

- `decay1.0`
- `mpiBroadcastRootOnlyCout`
- `mpiBroadcastTensor2Array`
- `mpiBroadcastUnknownFunction`
- `mpiBroadcastAliasRead`

都会被错误标成 `DistributedFollowup`，从而破坏原有 broadcast 分类和结果。

修正后：

- `DistributedFollowup` 只出现在 loop-lowered MPI stencil site
- 这些 case 全部恢复到原语义

### 7.6 测试脚本的 warning 清洗误差

`mpirun` 下 AdaptiveCpp warning 混流后会在 `.clean` 文件里留下空行，导致逐字 diff 失败。

修正后：

- `test_mpi.sh` 的 `clean_output()` 增加了空白行过滤

## 8. 新增了什么验证

这一步新增了一个正向 Phase C 用例：

- `mpiDistributedStencil1D`

它的作用不是替代 `liuliang1.0`，而是提供一个当前 guard 下可以稳定命中的正向 case：

- 1D `dacpp::Vector`
- effective `READ/WRITE`
- 在 loop lowering 路径内
- 有可识别的 post-shell helper

它验证了这些点：

- `step2.log` 里有 `partial-exchange enabled (root-bridge)`
- 输出同步分类是 `distributed-followup`
- 生成代码里真的有 `ctx.use_partial_exchange = true`
- 生成代码里真的有 `root_bridge_plan = dacpp::mpi::build_exchange_plan_from_layouts(...)`
- helper 里真的有 `dacpp::mpi::exchange_values_by_slots(...)`
- 运行结果和 baseline 一致

## 9. 这一步跑了哪些测试

我这一步实际跑过的测试包括：

定向回归：

```bash
bash clang/tools/translator/test_mpi.sh liuliang1.0 stencil1.0 waveEquation1.0
bash clang/tools/translator/test_mpi.sh mpiDistributedStencil1D
bash clang/tools/translator/test_mpi.sh decay1.0 mpiBroadcastRootOnlyCout mpiBroadcastTensor2Array mpiBroadcastUnknownFunction mpiBroadcastAliasRead mpiDistributedStencil1D
```

全量 MPI 回归：

```bash
bash clang/tools/translator/test_mpi.sh
```

最终结果：

```text
18 tests | 18 passed | 0 failed | 0 skipped
```

## 10. 这一步之后，当前真实状态是什么

可以明确说已经做成的：

- Phase C 的 runtime 基础设施已经在代码里了
- `DistributedFollowup` 已经不是空壳
- loop-lowered MPI stencil 已经有真实 partial-exchange run path
- 当前已经有一个稳定正向 case `mpiDistributedStencil1D`
- 全部现有 MPI 回归通过

还不能说已经做成的：

- 通用 shell writer -> shell reader steady-state peer exchange
- 2D / `dacpp::Matrix`
- effective `READ_WRITE`
- one-shot wrapper 接入
- 精确 helper-written subset bridge plan
- generalized halo / irregular generalized exchange 的完整形态

## 11. 一句话总结

这一步真正做成的事情是：把 Phase C 从“文档计划”推进成“有 runtime、有 analysis、有 codegen、有正向测试、有完整回归”的第一版实现，但它目前仍然是保守的初版，核心正向路径还是 1D loop-lowered stencil + root-bridge partial refresh，而不是完整 steady-state distributed exchange。
