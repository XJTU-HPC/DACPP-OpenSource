# Phase C Route IR And Deferred Materialize 实现说明

日期：2026-05-04

这份文档记录 Phase C 1/2/3 的增量实现：把 distributed follow-up 从单个 writer-reader copy/shift 特例升级成多 route 机制，并把 no-helper steady-state 的 root materialization 从每轮 `run()` 延后到 loop 后一次执行。

## 1. 本次目标

本次只扩展已经存在的 loop-lowered MPI stencil Phase C 路径：

- 支持一个 sibling `for` body 中多条简单 route。
- 支持多个 WRITE tensor 分别发布到 READ cache。
- 支持一个 WRITE tensor fanout 到多个 READ cache。
- 对不满足 route 语法的 sibling loop 保持整站保守 fallback。
- 对 no-root-bridge partial-exchange site，把最终 root materialization 延后到 lowered outer loop 之后。

仍然不做：

- 不支持 `READ_WRITE` kernel param。
- 不支持 2D / `dacpp::Matrix`。
- 不支持复杂表达式、函数调用或通用数据流。
- 不改变 DACPP 源码语法、CLI 或普通 MPI wrapper 语义。

## 2. Route IR

`DistributedFollowupMapping` 从单个 tensor mapping 扩展为 route 记录：

- `writerTensor`
- `readerTensor`
- `writerParamIndex`
- `readerParamIndex`
- `targetOffset`

`targetOffset` 按 `readerOffset - writerOffset` 计算。比如：

```cpp
state[i] = next[i - 1];
```

对应 writer global index 到 reader global index 的 offset 是 `+1`。

## 3. 分析层

实现位置：

- `rewriter/include/Rewriter_MPI_Common.h`
- `rewriter/lib/mpi/Rewriter_MPI_Stencil_Analysis.cpp`

新增 route 专用抽取逻辑：

- 仍要求 `regionPlan.siblingStmts.size() == 1` 才尝试 distributed follow-up。
- sibling stmt 必须是简单 `for`。
- loop body 按分号拆成多条 assignment。
- 每条 assignment 必须匹配 `reader[i +/- C] = writer[i +/- C]`。
- reader 必须是 effective `READ`，writer 必须是 effective `WRITE`。
- 任一 assignment 不满足条件，则整个 sibling loop 不走 distributed route，继续交给 root-centric helper / fallback。

这样保留了 Phase C 的 whole-site fallback 语义，不做站内混合猜测。

## 4. Runtime State

实现位置：

- `dpcppLib/include/mpi/StencilTypes.h`

`DistributedTensorState<T>` 新增 route 级状态：

```cpp
std::vector<std::vector<int32_t>> local_target_slots_by_route;
std::vector<ExchangePlan> exchange_plans_by_route;
```

旧字段 `local_target_slots` 和 `exchange_plan` 仍保留，用于兼容现有生成代码和调试断言。

## 5. Codegen

实现位置：

- `rewriter/include/Rewriter_MPI_Stencil_Common.h`
- `rewriter/lib/mpi/Rewriter_MPI_Stencil_Codegen.cpp`
- `rewriter/lib/Rewriter_MPI_Stencil.cpp`

`init()` 中：

- 对每个 WRITE tensor 找到所有 route。
- 为每条 route 构建 target slots 和 exchange plan。
- route 有 offset 时使用 offset-aware helper。
- 任一路由 runtime validation 失败，就设置 `ctx.use_partial_exchange = false`，整站回到旧 root-centric path。
- 对没有显式 follow-up route 的旧 no-followup case，保留历史默认 route 行为。

`run()` 中：

- kernel 后按 route 循环调用 `publish_local_writes_with_exchange(...)`。
- 有显式 route 时，目标 cache 是对应 reader 的 `ctx.dist_<reader>.local_cache`。
- 无显式 route 的兼容路径继续发布到原默认目标。

## 6. Deferred Materialize

新增函数名：

```cpp
__dacpp_mpi_stencil_materialize_xxx(ctx, args...)
```

loop-lowered 生成形态变为：

```cpp
__dacpp_mpi_stencil_ctx_xxx ctx;
__dacpp_mpi_stencil_init_xxx(ctx, ...);
for (...) {
    __dacpp_mpi_stencil_run_xxx(ctx, ...);
}
__dacpp_mpi_stencil_materialize_xxx(ctx, ...);
```

materialize 行为：

- `ctx.use_partial_exchange == false` 时直接返回。
- static unsupported 或 root-bridge site 直接返回。
- no-root-bridge partial site 才把 READ cache `MPI_Gatherv` 回 root，并写回用户 tensor。

compatibility wrapper 保持一次性调用语义：

```cpp
ctx;
init(ctx, ...);
run(ctx, ...);
materialize(ctx, ...);
```

## 7. 新增回归

新增测试：

- `mpiDistributedStencilMultiRoute1D`
  - 两个独立 writer -> reader route。
  - 检查 partial-exchange enabled、route vector 初始化、两个 publish call、无 root bridge。

- `mpiDistributedStencilFanout1D`
  - 一个 WRITE tensor 同时发布到两个 READ cache。
  - 检查 `local_target_slots_by_route.resize(2)`、两个 reader cache 目标、materialize 函数。

- `mpiDistributedStencilRouteFallback1D`
  - sibling assignment RHS 是复杂表达式。
  - 确认不误走 distributed route，保守生成 root-bridge/root-centric helper，结果仍正确。

更新测试：

- `mpiDistributedStencilNoBridge1D`
- `mpiDistributedStencilSteady1D`

二者结构断言现在检查生成 materialize 函数。

## 8. 验证结果

构建：

```bash
cmake --build build --target translator -j8
```

通过。

既有 Phase C focused 回归：

```bash
bash test_mpi.sh mpiDistributedStencilNoBridge1D mpiDistributedStencilSteady1D mpiDistributedStencil1D
```

通过。

新增 route 回归：

```bash
bash test_mpi.sh mpiDistributedStencilMultiRoute1D mpiDistributedStencilFanout1D mpiDistributedStencilRouteFallback1D
```

通过。

Phase C 1/2/3 组合回归：

```bash
bash test_mpi.sh mpiDistributedStencilNoBridge1D mpiDistributedStencilSteady1D mpiDistributedStencil1D mpiDistributedStencilMultiRoute1D mpiDistributedStencilFanout1D mpiDistributedStencilRouteFallback1D
```

结果：

```text
6 tests | 6 passed | 0 failed | 0 skipped
```

完整默认 MPI 回归：

```bash
bash test_mpi.sh
```

结果：

```text
20 tests | 20 passed | 0 failed | 0 skipped
```

## 9. 当前真实状态

已经完成：

- route IR 泛化。
- 多 writer route。
- writer fanout route。
- route 级 target slots / exchange plan 初始化。
- route 级 publish 到 reader distributed cache。
- no-helper no-root-bridge loop 后 materialize。
- unsupported complex assignment 的保守 fallback 回归。

还不能宣称：

- 通用数据流 route。
- 复杂 RHS route。
- `READ_WRITE` steady-state。
- 2D / `dacpp::Matrix` route。
- 精确 helper-written subset bridge。
- one-shot wrapper Phase C。
- generalized halo / cache exchange。

## 10. 一句话总结

这一步把 Phase C 从“单个 writer-reader copy/shift 特例”推进到“一个 sibling loop 内多条简单 route”的稳态交换，并把 no-helper partial path 的 root materialization 从每轮通信收紧到 loop 后一次执行；现有默认 MPI 回归保持 `20 / 20` 通过，新增 route 回归也通过。
