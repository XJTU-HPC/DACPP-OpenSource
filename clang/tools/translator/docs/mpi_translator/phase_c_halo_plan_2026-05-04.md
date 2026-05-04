# Phase C Halo 交换计划

日期：2026-05-04

这份文档记录 Phase C 下一步 halo 交换的核心方案。它不是要替换已经落地的 generalized exchange，而是在现有 `ExchangePlan` 的正确性基础上加一层规则场景的 compact path：能证明 peer 间传输是连续边界区间时走 halo span，不能证明时继续走 slot-list exchange。

## 1. 背景和目标

当前 Phase C 已经具备这些基础：

- `AllRankIndexLayout` 能让每个 rank 看到所有 rank 的 read / write global index 布局。
- `ExchangePlan` 已经能按 writer layout 和 reader layout 的交集发现 peer-to-peer 传输。
- route IR 已经能描述一维 steady-state mapping，例如 `state[i] = next[i - 1]`。
- no-root-bridge steady-state path 已经能把 shell 输出发布到 distributed reader cache，并在 loop 后 materialize。

halo 交换的目标是在这些能力上做性能特化：

- 不硬编码 `rank +/- 1`。
- 不重新发明一套独立正确性规则。
- 仍然先用 global index / layout 判定“谁写、谁读、谁需要谁的数据”。
- 当判定结果在本地 slot 上表现为连续边界区间时，把传输压成 span。
- 任一条件不满足时，自动回退 generalized slot exchange，不关闭 Phase C。

一句话：halo 是 generalized exchange 的 compact 表示，不是另一条语义路径。

## 2. 核心数据判定

halo / exchange 都以 route 为单位。对一条 writer -> reader route，定义：

```text
W_p = rank p 本轮 authoritative writer globals
R_q = rank q 的 reader cache globals
δ   = route targetOffset = readerOffset - writerOffset
```

`p -> q` 需要发送的 writer global 集合是：

```text
Send(p, q) = { g in W_p | g + δ in R_q && p != q }
```

`q` 接收后要写入 reader cache 的 target global 是：

```text
target = g + δ
```

等价地，从接收侧看：

```text
Recv(q, p) = { target in R_q | target - δ in W_p && p != q }
```

这里的 `δ` 已经和现有 offset-aware helper 保持一致。例如：

```cpp
state[i] = next[i - 1];
```

reader offset 是 `0`，writer offset 是 `-1`，所以：

```text
targetOffset = 0 - (-1) = +1
writer global g -> reader global g + 1
```

## 3. 邻居判定

概念上，邻居完全由 layout 交集产生：

- 对本 rank `p`，遍历所有 `q != p`。
- 如果 `Send(p, q)` 非空，`q` 是 `p` 的 send neighbor。
- 对本 rank `q`，如果存在 `target in R_q`，且 `target - δ` 的唯一 writer owner 是 `p != q`，`p` 是 recv neighbor。
- `p == q` 是本地 self transfer，不走 MPI。

实现上，这一步不在 halo compact 层重复执行。现有 `build_exchange_plan_from_layouts(...)` / `build_exchange_plan_from_layouts_with_target_offset(...)` 已经完成上述交集和 peer 分组；halo 只消费 `ExchangePlan.send_transfers` / `ExchangePlan.recv_transfers`。

这个规则覆盖：

- 普通 1D 左右 halo。
- 宽 stencil 跨多个边界元素。
- 大 offset 导致跨越多个 rank 的依赖。
- 非均匀 rank 切分。
- 未来不规则访问退回 generalized exchange 的场景。

因此 v1 不应该写死 `rank - 1` / `rank + 1`。对于规则 1D stencil，layout 交集自然会只发现左右相邻 rank；对于更复杂的 item-space 布局，发现到的 peer 才是真正需要通信的邻居。

## 4. 本地 self transfer

self transfer 不属于 halo MPI 通信：

- 本 rank 写出的 writer global 如果映射到本 rank reader cache，继续通过 `local_write_slots -> local_target_slots` 本地回填。
- 只有 `p != q` 的 transfer 进入 `send_transfers` / `recv_transfers`。
- 这样可以避免同一 rank 内多走一次 pack/send/unpack，也和现有 `publish_local_writes_with_exchange(...)` 语义一致。

## 5. Runtime 设计

保留现有 `ExchangePlan` 作为正确性主路径。新增一组可选 compact 结构：

```cpp
struct SlotSpan {
    int32_t begin = 0;
    int32_t count = 0;
};

struct PeerHaloExchange {
    int peer_rank = -1;
    std::vector<SlotSpan> local_spans;
};

struct HaloExchangePlan {
    bool supported = false;
    std::string unsupported_reason;
    std::vector<PeerHaloExchange> send_transfers;
    std::vector<PeerHaloExchange> recv_transfers;
};
```

`DistributedTensorState<T>` 增加：

```cpp
std::vector<HaloExchangePlan> halo_plans_by_route;
HaloExchangePlan halo_plan;
```

`halo_plan` 是兼容旧单 route 字段；`halo_plans_by_route` 是主用 route 级状态，和当前 `exchange_plans_by_route` 对齐。

新增 helper 建议：

- `build_halo_plan_from_exchange_plan(const ExchangePlan&)`
- `exchange_values_by_halo_spans(...)`
- `publish_local_writes_with_halo_or_exchange(...)`

`HaloExchangePlan` 不重复保存 `globals`。它只保存 peer 和 slot span；global 级正确性仍由已经构建好的 `ExchangePlan` 负责。这样避免把 transfer metadata 再复制一份。

## 6. Halo supported 条件

v1 的 supported 条件必须保守：

- `ExchangePlan.supported == true`。
- 每个 peer 的 `local_slots` 非负，且都在对应本地 buffer 范围内。
- 每个 peer transfer 内的 `local_slots` 必须单调递增。
- 相邻 slot 如果连续，合并到同一个 `SlotSpan`。
- 如果允许 multi-span，则每个 peer 可以有多个连续 span。
- 如果 v1 实现先只启用 single-span，则出现多个 span 时 `HaloExchangePlan.supported=false`，回退 generalized exchange。

推荐 v1 直接支持 multi-span 数据结构，但只把 “span count 小、总 slots 连续性明确” 的情况标成 supported。这样后续扩展 2D tile halo 时不需要重做结构。

unsupported 不等于 Phase C unsupported：

- halo unsupported：只使用 generalized `ExchangePlan`。
- exchange plan unsupported：保持当前整站 fallback。

## 7. 复杂度约束

halo 不能重新做一遍邻居判定。当前 `ExchangePlan` 构建已经完成了 writer / reader layout 交集、writer owner 查询和 peer 分组，halo 只能从这个结果派生。

记号：

```text
P       = mpi_size
W_all   = all-rank writer layout globals 总数
W_local = 本 rank writer globals 数
R_local = 本 rank reader globals 数
R_max   = 单个 rank 最大 reader globals 数
T_send  = 本 rank outgoing transfer slot 总数
T_recv  = 本 rank incoming transfer slot 总数
N_peer  = 本 rank实际通信 peer 数
L_pack  = pack slot lookup 成本，DenseCover / g2l 为 O(1)，segment lookup 为 O(log segment_count)
```

现有初始化期主要成本：

- `validate_unique_writers(...)`：`O(W_all)`。
- 构建 `writer_owner`：`O(W_all)`。
- 接收侧邻居发现：`O(R_local * L_pack)`。
- 发送侧邻居发现：`O(W_local * P * log R_max + T_send * L_pack)`。
- peer vector sort：`O(N_peer log N_peer)`。

所以当前 route 级 exchange plan 初始化主项可写成：

```text
O(W_all + R_local * L_pack + W_local * P * log R_max + T_send * L_pack + N_peer log N_peer)
```

offset-aware helper 只是把 lookup key 从 `g` 换成 `g + targetOffset` 或 `target - targetOffset`，复杂度同阶。

halo compact 层的复杂度约束：

- 不访问 `writer_layout` / `reader_layout` 做新交集。
- 不重新构建 `writer_owner`。
- 不遍历 `P * W_local`。
- 只扫描现有 `ExchangePlan.send_transfers` 和 `recv_transfers` 的 slot list。

因此 `build_halo_plan_from_exchange_plan(...)` 只能是：

```text
O(T_send + T_recv + N_peer)
```

其中 `T_send + T_recv` 已经是现有 plan 产出的真实通信槽位数，且有：

```text
T_send <= W_local * (P - 1)
T_recv <= R_local
```

这不会改变初始化期 Big-O 主项，只增加一次线性 metadata 压缩，且不复制 globals，额外内存为：

```text
O(span_count + N_peer)
```

而不是 `O(T_send + T_recv)` 级别的第二份 global metadata。

运行期现有 generalized exchange 成本：

- self write pack / local unpack：`O(W_local)`。
- peer send pack：`O(T_send)`。
- peer recv unpack：`O(T_recv)`。
- MPI message 数：`O(N_peer)`。
- MPI payload：`O(T_send + T_recv)`。

halo span exchange 必须保持同阶：

```text
O(W_local + T_send + T_recv + span_count + N_peer)
```

`span_count <= T_send + T_recv`，所以不增加运行期 Big-O。规则 stencil 下通常 `span_count << T_send + T_recv`，但设计上只承诺不增加复杂度，不依赖这个优化成立。

结论：邻居判定只做一次，仍由现有 `ExchangePlan` 负责；halo 只做 plan-local 压缩和执行分支，不增加新的 layout 交集复杂度。

## 8. Exchange 执行语义

`publish_local_writes_with_halo_or_exchange(...)` 的行为：

1. 如果 `local_write_slots` 非空，从 kernel local output buffer pack 本 rank 写出的值。
2. 用 `local_target_slots` 把本 rank写出的 self-visible 值回填到目标 reader cache。
3. 如果 `halo_plan.supported`，按 halo spans 做 peer exchange。
4. 否则按现有 `ExchangePlan` 做 slot-list peer exchange。

halo exchange 内部仍使用非阻塞 MPI：

- 先为 `recv_transfers` post `MPI_Irecv`。
- 再为 `send_transfers` pack span 并 `MPI_Isend`。
- `MPI_Waitall` 后按 recv span unpack 到 reader cache。

MPI tag v1 可以沿用现有 exchange tag `0`，但如果后续同一阶段存在多条 route 并发执行，需要引入 route-specific tag 或保证 route exchange 严格串行。当前 codegen 是逐 route 串行 publish，因此可保持 tag `0`。

## 9. Codegen 接入点

`init()` 阶段：

- 先照旧为每条 route 构建 `exchange_plans_by_route[i]`。
- 如果 exchange plan unsupported，设置 `ctx.partial_exchange_disable_reason`，关闭 `ctx.use_partial_exchange`。
- 如果 exchange plan supported，再构建 `halo_plans_by_route[i]`。
- halo unsupported 只记录 reason，不关闭 partial exchange。
- 如果 `halo_plans_by_route` 非空，把首个 route 同步到兼容字段 `halo_plan`。

`run()` 阶段：

- kernel 后仍按 route 发布。
- route 有 explicit reader 时，目标 cache 是 `ctx.dist_<reader>.local_cache`。
- route 没有 explicit reader 时，保持现有兼容目标。
- 调用新 helper，由 helper 自己决定 halo 或 generalized exchange。
- root bridge 逻辑不变；helper 后 root -> distributed cache 刷新仍用已有 `root_bridge_plan`。

建议生成日志：

```text
[DACPP][MPI][PhaseC] halo-exchange enabled route=<writer>-><reader> peers=<N>
[DACPP][MPI][PhaseC] halo-exchange unavailable route=<writer>-><reader>: <reason>; using generalized exchange
```

日志只描述 halo compact path 是否启用，不改变当前 `partial-exchange enabled` / `partial-exchange disabled` 的主日志语义。

## 10. 和现有 generalized exchange 的关系

实现顺序应保持：

1. 先构建 generalized `ExchangePlan`。
2. 从 `ExchangePlan` 压缩出 optional `HaloExchangePlan`。
3. run path 优先尝试 halo。
4. halo 不可用时继续 generalized exchange。

这样有三个好处：

- halo 判断不需要重复实现 writer owner / reader intersection。
- 不规则访问天然保留正确 fallback。
- 测试可以同时断言 halo path 和 generalized fallback path。

## 11. v1 范围

v1 只覆盖：

- Phase C loop-lowered stencil site。
- 1D `dacpp::Vector`。
- transport mode 是 `WRITE` 的 writer。
- unique writer 能验证通过。
- route 已经能抽取为一维 affine mapping。
- no-root-bridge steady-state 优先；root-bridge case 不强制切 halo。

v1 不覆盖：

- 普通 one-shot MPI wrapper。
- 2D / `dacpp::Matrix` tile halo。
- 真 in-place `READ_WRITE`。
- 复杂 RHS 数据流 route。
- 精确 helper-written subset root bridge。
- 站内混合 partial/fallback。

## 12. 测试计划

新增 `mpiPhaseCHalo1D`：

- 1D `state -> next -> state` steady-state。
- route 形态：`state[i] = next[i - 1]`。
- 断言：
  - log 包含 `partial-exchange enabled`。
  - log 包含 `halo-exchange enabled`。
  - generated SYCL 包含 `build_halo_plan_from_exchange_plan`。
  - generated SYCL 包含 `publish_local_writes_with_halo_or_exchange`。
  - generated SYCL 不包含 `root_bridge_plan = dacpp::mpi::build_exchange_plan_from_layouts`。
  - MPI 输出和 baseline 一致。

新增 `mpiPhaseCHaloWide1D`：

- 读窗口跨多个边界元素，例如 5 点 stencil。
- 验证每 peer halo span 能覆盖宽边界。
- 输出和 baseline 一致。

更新现有测试：

- `mpiDistributedStencilSteady1D`
  - 断言生成 halo plan。
  - 保留 loop 后 materialize 断言。
- `mpiDistributedStencilMultiRoute1D`
  - 断言多 route 分别尝试 halo。
- `mpiDistributedStencilFanout1D`
  - 断言 fanout route 仍能发布到不同 reader cache。
- `mpiDistributedStencilRouteFallback1D`
  - 复杂 route 不误进 halo。
  - 保持 fallback / generalized behavior。
- `liuliang1.0`
  - 仍允许 root-bridge partial path。
  - 不要求 no-root-bridge halo。

建议回归命令：

```bash
cmake --build build --target translator -j8
bash test_mpi.sh mpiPhaseCHalo1D mpiPhaseCHaloWide1D
bash test_mpi.sh mpiDistributedStencilSteady1D mpiDistributedStencilMultiRoute1D mpiDistributedStencilFanout1D mpiDistributedStencilRouteFallback1D
bash test_mpi.sh liuliang1.0
bash test_mpi.sh
```

## 13. 实现顺序建议

建议按下面顺序实现：

1. 在运行时类型里增加 `SlotSpan` / `PeerHaloExchange` / `HaloExchangePlan`，不在 halo plan 中重复保存 globals。
2. 实现 `build_halo_plan_from_exchange_plan(...)`，只做 slot span 压缩，不做 MPI。
3. 给 helper 写小范围结构测试，先验证连续 / 不连续 / multi-span 判定，并确认只扫描现有 transfer slots。
4. 实现 `exchange_values_by_halo_spans(...)`。
5. 实现 `publish_local_writes_with_halo_or_exchange(...)`。
6. codegen `init()` 中预建 route 级 halo plan。
7. codegen `run()` 中替换 publish helper。
8. 加 `mpiPhaseCHalo1D` 和 `mpiPhaseCHaloWide1D`。
9. 跑 Phase C focused 回归和完整 MPI suite。

关键原则：每一步都保留 generalized exchange fallback，直到所有 halo 正向测试稳定。

## 14. 后续方向

- 2D / `dacpp::Matrix` tile halo，把 span 从 1D slot span 扩展为 row/column/tile region。
- 更通用 route IR，支持复杂 RHS 的数据流依赖分析。
- 更多安全 `READ_WRITE` 形态，尤其是可证明 step 边界一致的双 buffer / ping-pong 结构。
- 精确 helper-written subset bridge，减少 root bridge 的 dense payload。
- 对规则 stencil 自动选择 halo，对不规则访问保留 generalized index exchange。

## 15. 一句话总结

Phase C halo 的核心不是猜邻居，而是先用现有 global layout 精确发现 peer 依赖，再把连续 slot-list 压成 halo span。这样规则 stencil 会自动退化成传统 halo，不规则或暂时无法证明的场景仍然保留 generalized exchange 的正确性。
