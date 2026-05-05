# MPI Translator Stencil / Wrapper / Post-Shell Status

更新时间：2026-05-05

本文是 MPI translator 当前 stencil / wrapper / post-shell 优化状态的单一入口。它记录当前能做什么、怎么做、如何验证，以及下一步 TODO。

## 1. 当前结论

当前验收目标以 `tests/` 下 14 个非 `mpi*` 应用为主：

- `DFT1.0`
- `FOuLa1.0`
- `MDP1.0`
- `decay1.0`
- `gradientSum`
- `imageAdjustment1.0`
- `jacobi1.0`
- `liuliang1.0`
- `mandel1.0`
- `matMul1.0`
- `oddeven0.1`
- `stencil1.0`
- `vectorAddCombo`
- `waveEquation1.0`

当前状态：

- 完整 MPI suite：`28 tests | 28 passed | 0 failed | 0 skipped`。
- 14 个非 `mpi*` 应用均已纳入 MPI 验收；`oddeven0.1` 已加入 `test_mpi.sh`。
- `test_mpi.sh` 会全局检查生成 MPI SYCL 中不得残留 DACPP `<->`。
- `stencil1.0`、`liuliang1.0`、`waveEquation1.0` 的每步 root-centric 灾难慢路径已经移除。
- `jacobi1.0` 当前是明确的 mixed Vector/Matrix fallback correctness path，不进入 2D Matrix Phase C。
- `FOuLa1.0` 当前保持 loop 内临时 view 的 compatibility wrapper 路径；已加入高风险应用 benchmark smoke 和结构断言，当前小规模 smoke 未见灾难慢信号。

第一阶段性能口径是“消灭灾难慢”：

- 正确性必须全过。
- 先治理 10x/100x 级慢路径、每步 root dense gather/helper/bridge、每轮重复 scatter 大型 loop-invariant READ 输入。
- 暂不追求所有用例都在 1.3x / 2x 以内。

## 2. 生成结构

MPI stencil loop path 服务于 `--mode=buffer --mpi` 下位于 time-step loop 内的 `<->`。

目标生成形态：

```cpp
__dacpp_mpi_stencil_ctx_xxx ctx;
__dacpp_mpi_stencil_init_xxx(ctx, ...);
for (...) {
    __dacpp_mpi_stencil_run_xxx(ctx, ...);
}
__dacpp_mpi_stencil_materialize_xxx(ctx, ...);
```

含义：

- `ctx`：保存 MPI rank/size、SYCL queue、AccessPattern、PackPlan、ItemRange、cached layouts、distributed cache 和复用 buffer。
- `init()`：循环外执行一次，初始化稳定 metadata、rank-local item range、pack/layout/exchange plan，并 seed 初始 READ cache。
- `run()`：每步执行 kernel、必要通信、writer -> reader publish、boundary-local update、read-cache transition。
- `run()` 内的 read-cache transition 和 writer publish helper 复用 ctx 中的 SYCL queue，避免 helper 内部每步重复创建 queue。
- `DACPP_MPI_PROFILE=1` 时，`run()` 输出 wrapper 总耗时和 `input/dist_setup/kernel/read_transition/publish/boundary/root_bridge/writeback` 分段耗时；默认关闭，不影响正常测试输出。
- 规则 contiguous PackPlan 会启用 `ContiguousView1D/2D` kernel view，省掉 kernel 内 `slots/key_offsets` buffer 和间接寻址；guard 失败时自动回到通用 `View1D/2D` path。
- `materialize()`：loop 后执行一次，把 no-root distributed READ cache gather 回 root-visible tensor。root-bridge/fallback path 下直接返回或保持旧语义。
- compatibility wrapper：非 loop stencil site 仍可用 `ctx; init; run; materialize;` 一次性调用。

不能安全 hoist 的场景回退普通 MPI wrapper：

- shell 实参在循环体内部临时声明。
- 每次迭代实参形状或绑定语义可能变化。
- 无法把 `init(ctx, args...)` 安全提到外层 loop 之前。

## 3. 主要实现位置

核心入口：

- `clang/tools/translator/translator.cpp`
- `clang/tools/translator/rewriter/lib/Rewriter_MPI.cpp`
- `clang/tools/translator/rewriter/lib/Rewriter_MPI_Stencil.cpp`

Stencil / Phase C 分析和生成：

- `clang/tools/translator/rewriter/include/Rewriter_MPI_Common.h`
- `clang/tools/translator/rewriter/include/Rewriter_MPI_Stencil_Common.h`
- `clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Stencil_Analysis.cpp`
- `clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Stencil_Codegen.cpp`

输出一致性和 root-only 输出：

- `clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_OutputAnalysis.cpp`
- `clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_PrintRewrite.cpp`
- `clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_ParamAnalysis.cpp`

Post-shell root helper fallback：

- `clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_PostRegion_Analysis.cpp`
- `clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_PostRegion_Codegen.cpp`
- `clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_PostRegion_Internal.h`

Runtime helper：

- `clang/tools/translator/dpcppLib/include/MPIPlanner.h`
- `clang/tools/translator/dpcppLib/include/mpi/StencilTypes.h`
- `clang/tools/translator/dpcppLib/include/mpi/StencilLayout.h`
- `clang/tools/translator/dpcppLib/include/mpi/StencilExchange.h`
- `clang/tools/translator/dpcppLib/include/mpi/Views.h`
- `clang/tools/translator/dpcppLib/include/mpi/Wrapper*.h`

相关设计说明：

- `phase_c_impl_note_2026-05-03.md`
- `phase_c_routes_materialize_2026-05-04.md`
- `phase_c_halo_plan_2026-05-04.md`

## 4. 已完成能力

### 4.1 Stencil 接线和基础修复

已完成：

- 登记 loop 内 `<->` 为 `MpiStencilSite`，记录 `dacExpr` 和外层 loop。
- 入口分流：有安全 stencil site 时走 `rewriteMPIStencil()`，否则保留普通 MPI wrapper。
- 安全 hoist guard：shell 实参若是 loop 内临时 view/tensor，则不进入 stencil hoist。
- MPI 写回 gather 修复：普通 wrapper 和 stencil run 都按 `writeback_slots` 先构造 `writeback_values`，避免 `writeback_globals` 与 `local_*` 前缀错位。
- 同一 shell/calc 多个 `<->` occurrence 修复：MPI 模式不再因 `dacExprMap` 去重跳过后续 occurrence；stencil rewriter 会对未替换 occurrence 做 fallback wrapper rewrite。

`oddeven0.1` 当前验证了多 `<->` occurrence 都会被替换，生成 MPI SYCL 不残留 `<->`。

### 4.2 Phase 1 metadata hoist

Phase 1 只优化 root-centric MPI stencil 路径的重复 metadata，不改变语义：

- `init()` 中缓存 input/output layout、counts/displs、writeback slots、root pack globals。
- `run()` 删除每步重复 metadata gather，只保留数据相关通信。
- 本地/root 侧临时 buffer 复用。
- wrapper timing `MPI_Reduce` 只在 `DACPP_MPI_PROFILE` 开启时执行。

典型 ctx 字段：

```cpp
dacpp::mpi::GatheredIndexLayout input_layout_x;
std::vector<T> global_x;
std::vector<T> sendbuf_x;
dacpp::mpi::GatheredIndexLayout output_layout_y;
std::vector<int32_t> writeback_slots_y;
std::vector<T> writeback_values_y;
```

### 4.3 输出一致性分类

MPI 翻译后，tensor 一致性分为：

- root 结果一致性：通过 `MPI_Gatherv` 收集分布式输出，root 重建完整 tensor。
- 非 root 副本一致性：如果后续 all-rank host code 需要读该 tensor，再 `MPI_Bcast` root 完整结果。

当前分类：

- `RootOnly`：root 结果正确即可，不 broadcast。
- `AllRanksNeeded`：后续 all-rank host read，需要 `Gatherv + Bcast`。
- `RootCentricFollowup`：后续读落在 root-centric helper fallback 内，不先 broadcast shell 输出。
- `DistributedFollowup`：loop-lowered partial-exchange site 的 writer -> reader distributed path。

已验证：

- `decay1.0`：`local_A sync=root-only`，无 `MPI_Bcast`。
- `liuliang1.0`：`new_rho sync=distributed-followup`。
- `mpiDenseCoverSibling1.0`：`updates sync=all-ranks-needed`，保留 `MPI_Bcast`。

### 4.4 Root-only 输出 rewrite

不再用全局 stdout 重定向压掉非 root 输出，而是在输出语句周围插入 root guard：

```cpp
if (__dacpp_mpi_is_root_rank()) {
    tensor.print();
}
```

覆盖：

- DACPP tensor `.print()`，包括 member/subscript chain。
- `std::cout << ...` / `cout << ...`。
- main 外用户函数中的输出，例如 `mandel1.0` / `imageAdjustment1.0`。
- 无花括号 loop body 中的输出，例如 `gradientSum`。

## 5. Phase C 分布式缓存和部分交换

Phase C 的目标是把 root-centric 每步通信：

```text
root pack -> Scatterv input -> kernel -> Gatherv output -> root apply -> optional Bcast
```

逐步变成：

```text
persistent distributed read/write cache
kernel writes local subset
publish dirty overlap to remote reader cache
only materialize at observable point
```

### 5.1 Runtime / IR

已落地 runtime 结构：

- `AllRankIndexLayout`
- `PeerSlotExchange`
- `ExchangePlan`
- `DistributedTensorState<T>`
- `SlotSpan`
- `PeerHaloExchange`
- `HaloExchangePlan`

已落地 helper：

- `init_all_rank_index_layout()`
- `build_exchange_plan_from_layouts()`
- `build_target_slots_for_globals()`
- `build_target_slots_for_globals_with_offset()`
- `build_exchange_plan_from_layouts_with_target_offset()`
- `map_2d_global_with_offset()`
- `build_target_slots_for_globals_2d_offset()`
- `build_exchange_plan_from_layouts_2d_offset()`
- `build_halo_plan_from_exchange_plan()`
- `publish_local_writes_with_halo_or_exchange()`
- `publish_local_writes_with_halo_or_exchange_cache_only()`

Route IR 当前支持：

- 1D `reader[i + a] = writer[i + b]`，记录 `AffineIndex1D` 和 `targetOffset = readerOffset - writerOffset`。
- 一个 sibling loop 内多条 route。
- fanout：一个 writer 发布到多个 reader cache。
- V1 2D `reader[i+a][j+b] = writer[i+c][j+d]`，记录 row/col offset 和 writer/reader cols。
- 1D boundary-local 单点 update。
- 2D boundary-local update。
- 简单 2D READ-cache state transition。

整站 fallback 规则仍保留：任一条件不满足时，回到 root-centric correctness path；halo compact unsupported 只回退 generalized exchange，不关闭 Phase C。

### 5.2 当前支持的正向路径

1D Vector：

- `mpiDistributedStencilNoBridge1D`：no-root partial exchange，无 `root_bridge_plan`。
- `mpiDistributedStencilSteady1D`：`next -> state` steady-state route，loop 后 materialize。
- `mpiPhaseCHalo1D`：stride-1 route 使用 halo compact path。
- `mpiPhaseCHaloWide1D`：wide stride route 域不满足 guard，保守 root-bridge fallback。
- `mpiDistributedStencil1D`：当前已从 helper bridge 推进到 no-root route + boundary-local。
- `liuliang1.0`：`new_rho -> rho offset=1` + `rho[0] = new_rho[0]` boundary-local，无 root helper/bridge。

2D Matrix：

- `mpiDistributedStencil2DRowBlock`：V1 2D row-block no-root route。
- `stencil1.0`：`matOut -> matIn offset=(1,1)`，边界 copy 为 boundary-local，loop 后 materialize。
- `waveEquation1.0`：
  - `matCur -> matPrev offset=(-1,-1)` 为 READ-cache transition。
  - `matNext -> matCur offset=(1,1)` 为 distributed route。
  - 边界清零为 boundary-local update。
  - READ-cache transition 已改为复用 stencil slot-copy helper 和现有 SYCL queue，避免大规模 transition 固定落在生成的 CPU copy loop 上。
  - writer -> reader publish helper 也复用现有 SYCL queue，减少每步 helper 内部临时 queue 构造。
  - 不再生成 root helper / root bridge。

Jacobi：

- `jacobi1.0` 是 mixed Matrix/Vector solver，当前明确禁用 Phase C：
  - `partial-exchange disabled: phase-c does not support mixed Vector/Matrix sites`
- fallback 已优化：
  - `A` / `b` / `nums` 作为 loop-invariant READ input 在 `init()` scatter 一次。
  - `x` 通过 `x <- x_new` loop-carried assignment 识别为 init-cache + run-refresh。
  - 仍需要每轮 `x_new` gather/broadcast 和 host 收敛判断。

## 6. 应用状态矩阵

| 用例 | 当前 MPI 路径 | 关键结构 |
|---|---|---|
| `DFT1.0` | wrapper / correctness path | 正确性通过 |
| `FOuLa1.0` | compatibility wrapper | loop 内临时 view，不做 hoist |
| `MDP1.0` | 1D distributed route | `new_p -> p offset=1`，no-root steady-state |
| `decay1.0` | stencil fallback + root-only output | `local_A sync=root-only` |
| `gradientSum` | wrapper / root-only output guard | `std::cout` guard |
| `imageAdjustment1.0` | wrapper / root-only output guard | main 外输出 guard |
| `jacobi1.0` | mixed Matrix/Vector fallback | READ input cache + `x` refresh |
| `liuliang1.0` | no-root 1D distributed route | `new_rho -> rho offset=1` + boundary-local |
| `mandel1.0` | wrapper / root-only output guard | main 外输出 guard |
| `matMul1.0` | wrapper / correctness path | 正确性通过 |
| `oddeven0.1` | compatibility/fallback correctness path | 多 `<->` occurrence 全替换 |
| `stencil1.0` | no-root 2D row-block path | route + boundary-local + loop 后 materialize |
| `vectorAddCombo` | wrapper / correctness path | 正确性通过 |
| `waveEquation1.0` | no-root 2D multi-state path | read-cache transition + route + boundary-local |

## 7. 验证和结构断言

`test_mpi.sh` 支持每个测试目录的 `mpi_expect.txt`：

- `LOG_CONTAINS:<literal>`
- `LOG_NOT_CONTAINS:<literal>`
- `SYCL_CONTAINS:<literal>`
- `SYCL_NOT_CONTAINS:<literal>`

关键结构断言：

- 全局：生成 MPI SYCL 不得残留 `<->`。
- `oddeven0.1`：两个 call site 都必须替换。
- `liuliang1.0`：必须有 `route detected new_rho->rho offset=1` 和 boundary-local；不得有 root helper / root bridge。
- `mpiDistributedStencil1D`：必须有 `route detected next->state offset=1` 和 boundary-local；不得有 root bridge。
- `stencil1.0`：必须有 2D offset route、boundary-local、cache-only publish；不得有 root bridge/helper。
- `waveEquation1.0`：必须有 `matCur->matPrev` read-cache transition、`matNext->matCur` route、boundary-local；不得有 root bridge/helper。
- `jacobi1.0`：必须保持 mixed Vector/Matrix Phase C disabled fallback，不能误进 Matrix Phase C。

常用验证命令：

```bash
cd /Volumes/QUQ/working/dacpp
cmake --build build --target translator -j8

cd /Volumes/QUQ/working/dacpp/clang/tools/translator
bash test_mpi.sh FOuLa1.0 liuliang1.0 MDP1.0 oddeven0.1 stencil1.0 waveEquation1.0 jacobi1.0
bash test_mpi.sh
```

当前结果：

```text
28 tests | 28 passed | 0 failed | 0 skipped
```

## 8. Benchmark

### 8.1 `stencil1.0`

脚本：

```bash
cd /Volumes/QUQ/working/dacpp/clang/tools/translator
STENCIL_BENCH_RUNS=1 bash bench_stencil_mpi.sh
```

默认口径：

- `NX=NY=1024`
- `TIME_STEPS=200`
- `mpirun -np 4`
- SYCL buffer
- 只计 time-step loop runtime，含 loop 后 materialize

当前复核结果：

```text
translated MPI stencil: 0.199666s
hand-written coarse MPI+SYCL buffer: 0.145769s
ratio: 1.37x
```

结论：旧 `25.0584s / 0.177454s ≈ 141x` 是 root-bridge 热路径旧数字，不代表当前状态。当前剩余差距主要来自通用 slot/view 间接寻址和 SYCL buffer 构造。

最新 profile / direct-view 复核：

```text
DACPP_MPI_PROFILE=1 STENCIL_BENCH_RUNS=1 STENCIL_BENCH_NX=1024 STENCIL_BENCH_NY=1024 STENCIL_BENCH_TIME_STEPS=200 bash bench_stencil_mpi.sh

translated MPI stencil: 0.212349s
hand-written coarse MPI+SYCL buffer: 0.158242s
ratio: 1.34x

per-step median profile:
kernel=0.743ms publish=0.279ms boundary=0.006ms dist_setup=0.064ms
root_bridge=0.000ms read_transition=0.000ms
```

`ContiguousView1D/2D` direct-view path 已启用并保持 correctness，但大头仍在 generated kernel launch / buffer 生命周期和 publish/exchange 常数；它不是数量级优化。

### 8.2 `waveEquation1.0`

脚本：

```bash
cd /Volumes/QUQ/working/dacpp/clang/tools/translator
WAVE_BENCH_RUNS=1 bash bench_wave_mpi.sh
```

默认口径：

- `NX=NY=1024`
- `TIME_STEPS=200`
- `mpirun -np 4`
- SYCL buffer
- 只计 time-step loop runtime，含 loop 后 materialize

修复前灾难慢路径：

```text
translated MPI wave: 51.3173s
hand-written coarse MPI+SYCL wave: 0.103013s
ratio: 498.16x
```

当前结果：

```text
translated MPI wave: 0.288732s
hand-written coarse MPI+SYCL wave: 0.112175s
ratio: 2.57x
```

结论：每步 root gather/helper/dense bridge 已移除；剩余差距属于 generated code / exchange / buffer 常数，不属于第一阶段灾难慢。

最新 profile / direct-view 复核：

```text
DACPP_MPI_PROFILE=1 WAVE_BENCH_RUNS=1 WAVE_BENCH_NX=1024 WAVE_BENCH_NY=1024 WAVE_BENCH_TIME_STEPS=200 bash bench_wave_mpi.sh

translated MPI wave: 0.291630s
hand-written coarse MPI+SYCL wave: 0.101942s
ratio: 2.86x

per-step median profile:
kernel=0.981ms publish=0.309ms read_transition=0.207ms boundary=0.005ms dist_setup=0.054ms
root_bridge=0.000ms writeback=0.000ms
```

`ContiguousView1D/2D` direct-view path 已启用，但 1024 规模 kernel 中位数与之前基本持平；下一步应优先减少每步 SYCL buffer / launch 常数，或做更窄的规则 row-block direct kernel，而不是继续微调 slot-copy。

### 8.3 高风险应用粗粒度 benchmark

脚本：

```bash
cd /Volumes/QUQ/working/dacpp/clang/tools/translator
APP_BENCH_RUNS=1 bash bench_mpi_apps.sh FOuLa1.0 liuliang1.0 MDP1.0 oddeven0.1 stencil1.0 waveEquation1.0 jacobi1.0
```

结果：

```text
FOuLa1.0         ratio 0.86x
liuliang1.0      ratio 1.04x
MDP1.0           ratio 1.24x
oddeven0.1       ratio 1.00x
stencil1.0       ratio 0.99x
waveEquation1.0  ratio 1.10x
jacobi1.0        ratio 1.05x
```

这个口径包含较多进程启动和小规模固定成本，只用于 smoke；真实 stencil/wave 性能以专用大规模 benchmark 为准。

### 8.4 多规模 sweep

脚本：

```bash
cd /Volumes/QUQ/working/dacpp/clang/tools/translator
SWEEP_BENCH_RUNS=3 bash bench_stencil_wave_sweep.sh
```

默认覆盖 `512 1024 1536` 三个 `NX=NY` 规模，每个规模分别跑 `stencil1.0` 和 `waveEquation1.0`，取现有专用 benchmark 的中位数并汇总到 TSV。这个入口用于降低单点 benchmark 噪声，判断后续复杂度 / 常数优化是否稳定。

当前中等规模复核：

```text
case     nx    time_steps  runs  ratio
stencil  512   200         2     0.48x
wave     512   200         2     1.01x
stencil  1024  200         2     1.26x
wave     1024  200         2     2.57x
```

已试验但不保留：

- CPU contiguous slot copy fast path：`stencil1.0` / `waveEquation1.0` 1024 单点均变慢。
- READ-cache transition 预缓存 slot `sycl::buffer`：`waveEquation1.0` 1024 两次中位数变差，说明当前 buffer 生命周期 / 同步成本抵消收益。

已保留但收益有限：

- guarded `ContiguousView1D/2D` kernel view：只在 PackPlan 证明 `compact_slots == 0..N-1` 且 `item_key_offsets == item * partition_size` 时启用；correctness 通过，结构更简单，但 `stencil1.0` / `waveEquation1.0` 1024 profile 显示主要瓶颈仍是 kernel / buffer / publish 常数。

## 9. 尚未完成 / 风险

仍不能宣称：

- 支持所有 loop 内 `<->` 的 hoist。
- 支持 loop 内临时 view 跨迭代 ctx 复用。
- 完整别名、函数调用、复杂表达式数据流、完整 root-only observable 证明。
- 所有非 root stale 副本风险都已消除。
- 所有 stdout/stderr API 都纳入 root-only rewrite；当前只覆盖 `.print()` 和 `std::cout` / `cout`。
- 通用 Matrix / 二维 / 复杂 sibling loop 都能 distributed follow-up；当前 Matrix 分布式路径只覆盖可证明的 V1 route、boundary-local update 和 READ-cache transition。
- 通用复杂数据流 steady-state route；当前 route 仍以简单 assignment/copy/shift 为主。
- stride/window 与 writer index 域不一致的 distributed route；这类形状保守 root helper / root bridge fallback。
- 精确 helper-written subset bridge payload；root bridge fallback 仍是保守 dense-root-authoritative 方案。
- 规则 2D fast kernel / tile halo / corner halo / buffer-view 开销优化已经完成。
- `jacobi1.0` 的 distributed iterative solver 已完成；当前只是 correctness fallback + READ cache / `x` refresh。
- Phase C 已接入 one-shot MPI wrapper path。

## 10. 下一步 TODO

当前推进重心先集中在 `waveEquation1.0`。其它应用和泛化路线暂时只做 correctness / structure 回归，不主动展开新优化，避免多线并行把判断噪声放大。

1. `waveEquation1.0` 剩余差距治理
   - 当前从 `498x` 灾难 root-centric path 降到约 `2.57x`；多规模 sweep 显示 1024 规模仍是最值得优先看的热路径。
   - profile 显示 kernel 约 `0.98ms/step`、publish 约 `0.31ms/step`、read-transition 约 `0.21ms/step`；boundary 可忽略，root bridge 为 0。
   - 短期只围绕 wave 做复杂度和常数优化：减少每步 SYCL buffer / launch 常数，降低 publish / read-transition 开销，保持 `matCur -> matPrev` read-cache transition、`matNext -> matCur` route、boundary-local update 语义不变。
   - 不引入 root helper/root bridge，不放宽 Phase C eligibility guard；guard 不满足时继续走现有 correctness path。

2. `waveEquation1.0` 可考虑的非常数优化方向
   - multi-state transition：在 layout、shape、生命周期完全一致时，把 `matCur -> matPrev` 从 O(local slots) scatter 降为 cache swap/rename；不满足 guard 时保留当前 slot-copy path。
   - kernel/transition 融合：评估固定 `kernel -> read-transition -> publish` 链能否融合本地 state update 或延迟 publish，减少每步 SYCL launch 和 buffer 生命周期；必须保持 halo 边界语义。
   - wave 专用 row-block direct kernel：只覆盖当前可证明的规则 2D row-block wave 形态，直接以 row/col domain 生成 kernel，绕过 item-space PackPlan 和每 item View 构造；不泛化到任意 stencil。
   - wave halo / exchange 压缩：优先把当前 generalized exchange 中的 row-span halo 做轻量化，减少 per-step metadata traversal 和消息处理；不规则访问继续走 generalized fallback。

3. wave benchmark / profile 固化
   - `bench_wave_mpi.sh` 作为当前主要性能 smoke。
   - 使用 `WAVE_BENCH_RUNS>=3` 和多个 `NX=NY` 规模复核，避免按单点噪声调整实现。
   - `bench_stencil_wave_sweep.sh` 暂时只用于观察 wave 和确认 stencil 没退化；不以 stencil 提速作为当前目标。
   - 每次 wave 优化后至少跑：`cmake --build build --target translator -j8`、`bash test_mpi.sh waveEquation1.0 stencil1.0 jacobi1.0`、`WAVE_BENCH_RUNS=1 bash bench_wave_mpi.sh`；阶段收口再跑完整 `bash test_mpi.sh`。

4. 暂缓项，只保留回归
   - `stencil1.0`：当前约 `1.3x`，不是灾难慢；暂缓 fast-kernel work，只确认 no-root route / boundary-local / materialize 结构不退化。
   - `FOuLa1.0`：当前 correctness 通过；暂缓 loop 内临时 view hot path 方案，只保留 smoke 和结构 expectation。
   - `jacobi1.0`：继续保持 mixed Matrix/Vector Phase C disabled fallback；独立 distributed solver 暂缓设计。
   - 泛化 Phase C route / boundary / bridge、tile halo / corner halo、helper-written subset bridge：作为后续方向保留，不进入当前 wave-first 阶段。

## 11. 一句话总结

当前 MPI stencil 路径已经完成 Phase 1、输出一致性分类、root-only 输出 rewrite、loop-lowered `ctx/init/run/materialize`、Phase C distributed cache / route / halo 基础设施，以及 1D boundary-local、2D boundary-local、2D READ-cache state transition。`liuliang1.0`、`stencil1.0`、`waveEquation1.0` 已从灾难 root-centric 热路径迁移到 no-root distributed path；`jacobi1.0` 保持正确的 mixed Vector/Matrix fallback。14 个非 `mpi*` 应用和完整 MPI suite 当前均通过。下一阶段先把重心放在 `waveEquation1.0`：围绕 kernel / publish / read-transition 的常数和必要的非常数优化做窄口径推进；`stencil1.0`、`FOuLa1.0`、`jacobi1.0` 和 Phase C 泛化路线暂时只保留回归，不主动展开。
