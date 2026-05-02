# MPI Wrapper Replica Consistency And Post-Shell Region Notes

更新时间：2026-05-02

## 1. 目标和边界

本文记录两个相关问题：

- MPI wrapper / MPI stencil 输出后的副本一致性问题。
- shell 执行完成后，紧跟的 host-side 可并行 `for` 循环是否可以继续翻译成 SYCL `parallel_for`。

本文只做问题归纳和后续方案记录，不表示已经实现 region 化的 MPI 代码生成。

当前讨论范围：

- 普通 MPI wrapper 路径。
- 已进入 MPI stencil loop 路径的用例，例如 `decay1.0`、`liuliang1.0`。
- root-centric 通信模型下的保守改进。

暂不讨论：

- halo 分布式 stencil。
- 改写 DACPP 源码语法。
- 新增用户可见 CLI flag。

## 2. 副本一致性问题

MPI 翻译后，一个 tensor 的一致性其实分成两层：

1. root 结果一致性。
2. 非 root 本地副本一致性。

root 结果一致性要求把每个 rank 计算出的局部写回收集到 root，并在 root 上重建完整 tensor。这一步需要 `MPI_Gatherv`，因为每个 rank 写回的 global index / value 数量和位置都可能不同。

非 root 本地副本一致性是另一件事。root 重建 tensor 后，其他 rank 上同名 tensor 仍然可能是旧值。如果后续 host 代码会在所有 rank 上读取该 tensor，则需要把 root 的完整结果再 `MPI_Bcast` 给所有 rank。

因此 `Gatherv` 和 `Bcast` 不是二选一关系：

- `MPI_Gatherv`：收集分布式输出，保证 root 拥有正确全局结果。
- `MPI_Bcast`：把 root 的全局结果同步回其他 rank，保证后续 all-rank host 读看到新值。

当前普通 MPI wrapper 输出路径大体是：

```cpp
MPI_Gather(send_count);
MPI_Gatherv(writeback_globals);
MPI_Gatherv(writeback_values);
if (rank == 0) {
    apply_writeback_by_globals(...);
}
if (needsBcast) {
    MPI_Bcast(full_tensor);
}
```

`needsBcast` 当前由 `tensorNeedsBroadcast(...)` 给出，是一个 bool 判断：当前 DAC 表达式之后，如果目标 tensor 在 DAC 表达式外还有读或读改写，就认为需要 broadcast；纯写覆盖则不需要。

## 3. `decay1.0` 暴露的问题

`decay1.0` 源码形态：

```cpp
while (t_tensor[0] <= T) {
    DECAY(N0s_tensor, lambdas_tensor, local_A_tensor, t_tensor) <-> decay;
    A_tensor[10*t_tensor[0]] = local_A_tensor;
    t_tensor[0] += dt;
}
```

当前它实际进入 MPI stencil 路径，而不是普通 wrapper 路径。生成代码中，`local_A` 的写回值会通过 `MPI_Gatherv(writeback_values_local_A...)` 收到 root，并在 root 上 `apply_writeback_by_globals(...)` 后写回 `local_A`。

风险点在后续 host 语句：

```cpp
A_tensor[10*t_tensor[0]] = local_A_tensor;
```

这行会在所有 rank 上执行。如果 `local_A_tensor` 没有 broadcast，则：

- root 的 `local_A_tensor` 是新值。
- 非 root 的 `local_A_tensor` 可能仍是旧值。
- 非 root 对 `A_tensor` 的更新可能基于旧副本。

当前测试可能没有暴露这个问题，因为非 root stdout 被静默，最终可观察输出主要来自 root。但语义上，这属于潜在副本不一致。

这个例子说明：仅仅保证 root 写回正确还不够，必须知道后续 host 代码是 root-only observable，还是 all-rank computation。

## 4. 副本一致性的改进方向

当前 bool 级别的 `needsBcast` 不够表达后续语义，建议提升成更细的分类：

- `root-only`：只需要 root 拿到正确结果，不需要 broadcast。
- `all-ranks-needed`：后续 host 代码会在所有 rank 读取该 tensor，需要 `Gatherv + Bcast`。
- `distributed-followup`：后续 host 代码本身可以被翻译成 MPI/SYCL region，不一定需要先 broadcast 完整 tensor。

静态分析也需要区分不同后续使用：

- 后续只是 root 输出、打印、写文件。
- 后续作为下一次 DAC shell 的输入。
- 后续参与普通 host 计算。
- 后续被纯写覆盖。
- 后续通过别名、slice、函数调用、容器访问间接使用。

保守策略：

- 分析不清楚时，优先 `Gatherv + Bcast` 保证语义。
- 能证明 root-only 时，跳过 broadcast。
- 能证明后续 host loop 可 region 化时，考虑把后续计算下沉到 SYCL/MPI region，减少不必要的全量副本同步。

## 5. Post-Shell 可并行循环问题

`liuliang1.0` 源码中，shell 之后紧跟两个 host loop：

```cpp
LWR_shell(rho, new_rho) <-> lwr;
for (int i = 1; i <= WIDTH-2; i++) {
    rho[i] = new_rho[i-1];
}
for (int i = 0; i < 1; i++) {
    rho[0] = new_rho[0];
}
```

第一个 loop 是明显的 element-wise copy / shift：

- 每个迭代写不同的 `rho[i]`。
- 每个迭代读 `new_rho[i-1]`。
- 没有 loop-carried dependency。
- 适合翻译成 `sycl::parallel_for`。

第二个 loop 实际只有一次迭代，语义上是边界赋值。它也可以被 region 化，但没必要为了性能单独并行；更合理的是作为小 region 或直接 root/all-rank host assignment 处理。

非 MPI buffer 路径已经能做类似优化。当前生成的普通 SYCL buffer 代码中，`liuliang1.0` 会生成：

```cpp
__dacpp_submit_region_LWR_shell_lwr_stmt_0(ctx);
__dacpp_submit_region_LWR_shell_lwr_stmt_1(ctx);
```

其中 `stmt_0` 内部就是：

```cpp
h.parallel_for(sycl::range<1>(__N), ... {
    d_rho[i] = d_new_rho[i-1];
});
```

但 MPI stencil 生成代码目前仍保留原始 C++ for：

```cpp
__dacpp_mpi_stencil_run_LWR_shell_lwr(ctx, rho, new_rho);
for (int i = 1; i <= WIDTH-2; i++) {
    rho[i] = new_rho[i-1];
}
```

这带来两个问题：

- 每个 time step 先对 `new_rho` 做完整 broadcast，保证所有 rank 都能执行后续 host loop。
- 每个 rank 重复执行同样的串行 host loop，无法复用前面 shell 已经建立的 SYCL queue、buffer、rank range 和数据分布信息。

所以这类问题本质上也是副本一致性问题的延伸：如果后续 host loop 可以被 region 化，就不一定要把 shell 输出先同步成所有 rank 的完整副本，再让所有 rank 重复串行执行。

## 6. MPI 下能否把 post-shell loop 变成 `parallel_for`

可以，但需要分层实现。

### 6.1 保守实现

保持现有 root-centric 语义：

1. shell 输出仍然 `Gatherv` 到 root。
2. 如果 post-shell loop 需要 all-rank 输入，则继续 `Bcast` 必要 tensor。
3. 将后续可并行 host loop 在每个 rank 上用 SYCL `parallel_for` 执行。

优点：

- 语义接近当前代码。
- 实现风险低。
- 可以复用非 MPI buffer region 的 AST 识别和代码生成思路。

缺点：

- 仍有全量 broadcast。
- 每个 rank 仍可能重复执行同样 region，只是从串行变成设备并行。
- 对 MPI 通信量没有本质优化。

### 6.2 更好的 root-centric region

把 post-shell loop 识别为 root-side region：

1. shell 输出 `Gatherv` 到 root。
2. root 在 SYCL 上执行 post-shell region，更新 root tensor。
3. 根据下一次使用决定是否 broadcast `rho`。

对 `liuliang1.0`，下一轮 `LWR_shell(rho, new_rho)` 需要读取 `rho`。在当前 root-centric scatter 模型下，每轮 shell 输入本来只要求 root 拥有最新 `rho`，所以 post-shell loop 可以只在 root 上执行，然后下一轮由 root scatter `rho` 的局部窗口。

优点：

- 可以避免让非 root 重复执行 host loop。
- 可能避免 `new_rho` 的全量 broadcast。
- 更贴合当前 scatter-from-root 的 MPI wrapper/stencil 模型。

缺点：

- 需要确认后续是否有 all-rank host read。
- 需要生成 root-only SYCL region 或 root-only host fallback。
- 如果下一步不是 DAC input，而是普通 all-rank host 计算，仍可能需要 broadcast。

### 6.3 分布式 follow-up region

更进一步，将 shell 输出和 post-shell loop 作为同一个分布式 region pipeline：

1. rank 本地执行 shell kernel。
2. rank 本地直接执行后续 copy / shift region。
3. 只在必要边界交换或最终 observable 点做通信。

优点：

- 通信量最低。
- 接近真正分布式 stencil / time-step 执行。

缺点：

- 需要 region 化、依赖分析、halo/边界处理。
- 已超出当前 Phase 1 root-centric 优化范围。

## 7. 建议的识别条件

post-shell loop 可以自动 region 化的基本条件：

- loop bound 可静态或运行时稳定表达。
- loop body 中写入的 tensor 下标是当前 loop induction variable 的 affine 表达。
- 每个迭代写集合不重叠，或能证明写冲突无害。
- 读取 tensor 不被同一 loop 的其他迭代写后读影响。
- body 内没有不可设备化的函数调用、I/O、MPI 调用、异常控制流。
- 涉及的 tensor 和 shell 参数能映射到同一个 ctx 或可构建独立 region ctx。

对 `liuliang1.0` 的第一个 loop：

```cpp
rho[i] = new_rho[i-1];
```

满足这些条件，适合作为首批测试。

对边界赋值：

```cpp
rho[0] = new_rho[0];
```

可以作为小 region，也可以先保持 host assignment。关键是它必须被纳入一致性分析：如果只在 root 执行，那么下一轮输入 scatter 是否只依赖 root；如果所有 rank 后续读 `rho[0]`，则需要 broadcast 或 all-rank 更新。

## 8. 后续工作建议

Phase A：一致性分类

- 将 `tensorNeedsBroadcast(...)` 从 bool 扩展为后续使用分类。
- 标注每个输出 tensor 的后续需求：root-only、all-ranks-needed、distributed-followup。
- 为 `decay1.0` 增加能暴露非 root stale 副本的测试。

Phase B：MPI root-centric post-shell region

- 复用非 MPI buffer region 的识别结果。
- 在 MPI stencil / wrapper ctx 中生成 post-shell region helper。
- 首先支持 root-only SYCL region，避免不必要的 `new_rho` broadcast。
- 用 `liuliang1.0` 验证 `rho[i] = new_rho[i-1]`。

Phase C：分布式 region pipeline

- 将 post-shell region 和 shell pack/writeback 信息关联。
- 尽量在 rank-local buffer 上执行后续 region。
- 引入必要边界交换，逐步走向 halo stencil。

## 9. 当前结论

`decay1.0` 和 `liuliang1.0` 反映的是同一类问题：DAC shell 之后的 host 代码不是无关尾巴，而是 MPI 数据一致性和通信策略的一部分。

当前 root-centric MPI 路径能保证 root 结果，但对非 root 副本和后续 host loop 的处理还比较粗：

- 该 broadcast 时可能漏掉，导致非 root stale 副本。
- 可 region 化的 host loop 仍作为普通 C++ for 保留，导致额外 broadcast 和重复串行执行。

下一步应该先做语义分类，再做保守 root-centric region 化；等这条路稳定后，再考虑真正分布式 follow-up region 和 halo 优化。
