# MPI Stencil 优化计划

## Context

当前 MPI 路径对时间迭代类程序（stencil、wave equation 等）效率低下：每个时间步都执行完整的 `root pack → MPI_Scatterv → 本地 SYCL 计算 → MPI_Gatherv → root writeback`，即每步都经过 root 做全量数据分发和回收。

用户的优化思路是：
1. **循环前一次性分发数据**，每个 rank 拿到所有自己需要的内存（含 halo 区）
2. **每个 rank 内部跑完整 for 循环**，消除每步的 scatter/gather 开销
3. **每步迭代后交换 halo 边界数据**，rank 间只传更新了的边界
4. 每个 rank 能自行计算它需要哪些邻居的哪些数据、哪些邻居需要它的数据
5. 已有的 pack map 基础设施可以直接复用，本地数组同时当 halo 缓冲区

**设计决策**：
- Halo exchange 按**参数+邻居方向双层**：外层遍历每个参数，内层遍历每个邻居方向（左/右），最精确的经典 halo 模式
- Sibling loops **一并处理**：在 submit 函数中生成 sibling loop helpers，让 stencil 的数据交换也在 rank 本地设备上执行

## 核心设计思路

### 为什么可行

现有 `build_input_pack_map(range, pattern)` 已经计算了一个 rank 所有 item 会读取的**全部**全局位置（包含 halo）。初始 scatter 后，每个 rank 的 `local_` 数组已经包含了 halo 数据。所以：

- **初始分发**：复用现有 scatter 逻辑，rank 拿到 interior + halo
- **Halo 交换**：每步迭代后，只需交换边界上被邻居更新的那部分数据
- **缓冲区**：`local_` 数组 + `g2l` 映射已经定位了 halo 在本地数组中的位置，无需额外分配

### Halo 计算（运行时，在 MPIPlanner 中）

对每个 rank r（item range `[begin_r, end_r)`）：
1. 计算 `my_outputs`：rank r 的 item 写入的全局位置（从 output pack map）
2. 计算 `my_inputs`：rank r 的 item 读取的全局位置（从 input pack map）
3. **Recv halo** = `my_inputs` 中不属于 `my_outputs` 的位置 → 这些由邻居 rank 产出
4. **Send halo** = `my_outputs` 中被邻居 rank 的 `my_inputs` 包含的位置

由于 item range 是连续的，halo 只涉及 rank-1 和 rank+1 两个邻居。每个 rank 可以独立计算自己的 send/recv halo（不需要跨 rank 通信），因为：
- 邻居的 item range 可以用 `get_rank_item_range()` 确定
- 邻居的 output 位置可以用 `collect_positions_for_item()` 确定

### 生成代码结构

```
__dacpp_mpi_ctx_<shell>_<calc>       // 持有 region 生命周期状态 + halo 信息
__dacpp_mpi_init_<shell>_<calc>()    // 一次性 scatter + 计算 halo 映射
for (t = 0; t < T; t++) {
    __dacpp_mpi_submit_<shell>_<calc>(ctx)      // SYCL kernel
    __dacpp_mpi_halo_<shell>_<calc>(ctx)         // halo exchange（D2H + MPI + H2D）
    __dacpp_mpi_submit_region_*(ctx)             // sibling loop helpers
}
__dacpp_mpi_sync_<shell>_<calc>()    // 最终 gather + writeback
```

## 实施步骤

### Step 1: MPIPlanner — 添加 Halo 数据结构和计算函数

**文件**: `dpcppLib/include/MPIPlanner.h`

添加 Halo 数据结构（**按参数+邻居方向组织**）：

```cpp
// 单个邻居方向的 halo 信息（针对一个参数）
struct HaloRegion {
    int neighbor_rank = -1;
    // 我需要从该 neighbor 接收的全局位置
    std::vector<int64_t> recv_globals;
    std::vector<int32_t> recv_local_slots;   // local_ 数组中的 slot
    // 我需要发送给该 neighbor 的全局位置
    std::vector<int64_t> send_globals;
    std::vector<int32_t> send_local_slots;   // local_ 数组中的 slot
};

// 一个参数的完整 halo 信息（最多左右两个邻居）
struct ParamHalo {
    std::vector<HaloRegion> regions;  // 1~2 个 neighbor
};
```

添加计算函数：

```cpp
// 为单个参数计算 halo
ParamHalo computeParamHalo(
    const AccessPattern& pattern,
    AccessMode mode,
    ItemRange my_range,
    int64_t total_items,
    int mpi_rank, int mpi_size);
```

**核心算法**：
```
for each neighbor_rank in {mpi_rank-1, mpi_rank+1} (如果合法):
    neighbor_range = get_rank_item_range(total_items, neighbor_rank, mpi_size)
    my_pack = build_input_pack_map(my_range, pattern)   // 我读的所有位置
    neighbor_pack = build_input_pack_map(neighbor_range, pattern)  // 邻居读的所有位置

    // recv: 我需要从 neighbor 接收的 = neighbor 的 output 中、我在 my_inputs 中也需要的位置
    // send: 我需要发送给 neighbor 的 = 我的 output 中、neighbor 的 inputs 中也包含的位置

    具体：
    if mode != WRITE:   // 有 read 行为，可能需要 recv
        neighbor_outputs = 遍历 neighbor_range 的每个 item 的 positions
        recv = my_pack.globals ∩ neighbor_outputs
        recv_local_slots = [my_pack.g2l[g] for g in recv]

    if mode != READ:    // 有 write 行为，可能需要 send
        my_outputs = 遍历 my_range 的每个 item 的 positions (output 部分)
        send = my_outputs ∩ neighbor_pack.globals
        send_local_slots = [my_pack.g2l[g] for g in send]
```

### Step 2: MPIPlanner — 添加 Halo 交换函数

**文件**: `dpcppLib/include/MPIPlanner.h`

```cpp
// 按参数+邻居双层执行 halo 交换
template <typename T>
void exchangeHalo(std::vector<T>& local_data,
                  const ParamHalo& halo,
                  MPI_Datatype mpi_type) {
    for (const auto& region : halo.regions) {
        if (region.neighbor_rank < 0) continue;
        // 打包发送
        std::vector<T> send_buf;
        send_buf.reserve(region.send_local_slots.size());
        for (int32_t slot : region.send_local_slots)
            send_buf.push_back(local_data[slot]);

        std::vector<T> recv_buf(region.recv_local_slots.size());

        MPI_Request reqs[2];
        MPI_Isend(send_buf.data(), send_buf.size(), mpi_type,
                  region.neighbor_rank, 0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(recv_buf.data(), recv_buf.size(), mpi_type,
                  region.neighbor_rank, 0, MPI_COMM_WORLD, &reqs[1]);
        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);

        // 解包接收
        for (std::size_t i = 0; i < region.recv_local_slots.size(); ++i)
            local_data[region.recv_local_slots[i]] = recv_buf[i];
    }
}
```

### Step 3: Rewriter_MPI — 扩展 ctx 结构体

**文件**: `rewriter/lib/Rewriter_MPI.cpp` — 修改 `buildMPIRegionCode()`

在现有 `ctx` 结构体中添加 halo 相关字段：

```cpp
bool has_halo = false;
// 每个参数的 halo 信息
dacpp::mpi::ParamHalo halo_<paramName>;
```

### Step 4: Rewriter_MPI — 修改 init 函数生成

**文件**: `rewriter/lib/Rewriter_MPI.cpp` — 修改 `buildMPIRegionCode()` 的 init 部分

在现有 scatter 逻辑之后，添加 halo 计算代码：

```cpp
code += "    ctx.has_halo = (ctx.mpi_size > 1);\n";
code += "    if (ctx.has_halo) {\n";
code += "        auto my_range = dacpp::mpi::get_rank_item_range(ctx.total_items, ctx.mpi_rank, ctx.mpi_size);\n";
for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
    const std::string& name = calc->getParam(paramIdx)->getName();
    code += "        ctx.halo_" + name + " = dacpp::mpi::computeParamHalo(\n";
    code += "            ctx.pattern_" + name + ",\n";
    code += "            ctx.pattern_" + name + ".mode,\n";
    code += "            my_range, ctx.total_items, ctx.mpi_rank, ctx.mpi_size);\n";
}
code += "    }\n";
```

### Step 5: Rewriter_MPI — 生成 submit 函数（SYCL kernel）

**文件**: `rewriter/lib/Rewriter_MPI.cpp` — 修改 `buildMPIRegionCode()` 的 submit 部分

submit 函数只负责 SYCL kernel 提交（现有逻辑不变），不做 wait：

```cpp
void __dacpp_mpi_submit_<shell>_<calc>(ctx) {
    if (ctx.local_item_count <= 0) return;
    // 现有的 kernel submit 逻辑（accessor + parallel_for）
}
```

### Step 6: Rewriter_MPI — 新增 halo exchange 函数

**文件**: `rewriter/lib/Rewriter_MPI.cpp` — 在 `buildMPIRegionCode()` 中新增

生成独立的 halo exchange 函数：

```cpp
void __dacpp_mpi_halo_<shell>_<calc>(ctx) {
    ctx.q.wait();  // 等待 kernel 完成

    if (!ctx.has_halo) return;

    // 按参数遍历
    for each param with mode != READ:  // 只有被 kernel 更新的参数需要交换
        // D2H: 从 SYCL buffer 读回 local_ 数组
        {
            auto host_acc = ctx.buf_<name>->get_access<sycl::access::mode::read>();
            auto* ptr = host_acc.get_multi_ptr<...>().get();
            for (i = 0; i < ctx.local_<name>.size(); ++i)
                ctx.local_<name>[i] = ptr[i];
        }
        // Halo exchange: 按邻居方向交换
        dacpp::mpi::exchangeHalo(ctx.local_<name>, ctx.halo_<name>, mpi_type);
        // H2D: 更新后的 halo 数据写回 SYCL buffer
        {
            auto host_acc = ctx.buf_<name>->get_access<sycl::access::mode::write>();
            auto* ptr = host_acc.get_multi_ptr<...>().get();
            for (i = 0; i < ctx.local_<name>.size(); ++i)
                ptr[i] = ctx.local_<name>[i];
        }
}
```

### Step 7: Rewriter_MPI — 处理 sibling loops

**文件**: `rewriter/lib/Rewriter_MPI.cpp`

从 `Rewriter_Buffer_new.cpp` 提取 sibling loop 处理模式，在 MPI submit 函数之后生成 sibling loop helpers：

1. 在 `DacppStructure.cpp` 的 `BufferRegionPlan` 中已有 `siblingForStmts` 列表
2. 对每个 sibling for loop，生成 `__dacpp_mpi_submit_region_<idx>` 函数
3. 该函数把 for loop 改写为 `parallel_for` 提交到 `ctx.q`
4. 操作 `ctx` 中的 SYCL buffer（而非单机路径的独立 buffer）

关键参考：
- `Rewriter_Buffer_new.cpp` 中的 `rewriteSiblingForsToParallel()` 和 `parallelizeNestedFor()` 函数
- `LoopVarAccessVisitor` 用于分析 sibling loop 中每个变量的读写模式
- 生成 accessor 时使用收紧后的 mode（read/write/read_write）

对于 MPI 路径的特殊处理：
- sibling loop 操作的是 `ctx` 中的 SYCL buffer（不是独立 buffer）
- accessor mode 按 sibling loop AST 真实读写集确定
- 不需要额外的 wait（halo exchange 中已经 wait 过）

### Step 8: translator.cpp — 生成 main 函数中的调用序列

**文件**: `rewriter/lib/Rewriter_MPI.cpp` + `rewriter/include/Rewriter.h`

实现 `rewriteMPI_Stencil()` 函数，在 main 函数中生成调用序列：

```cpp
// 在 main 的外层 for 循环处，替换为：
__dacpp_mpi_ctx_<shell>_<calc> ctx;
__dacpp_mpi_init_<shell>_<calc>(ctx, tensor_args...);
for (int t = 0; t < T; t++) {
    __dacpp_mpi_submit_<shell>_<calc>(ctx);
    __dacpp_mpi_halo_<shell>_<calc>(ctx);
    __dacpp_mpi_submit_region_0(ctx);  // sibling loop 0
    __dacpp_mpi_submit_region_1(ctx);  // sibling loop 1
    // ...
}
__dacpp_mpi_sync_<shell>_<calc>(ctx, tensor_args...);
```

具体插入位置由 `BufferRegionPlan` 中的 `outerFor`、`dacExpr` 等 AST 节点确定，复用 `Rewriter_Buffer_new.cpp` 中已有的插入逻辑。

### Step 9: translator.cpp — 激活 MPI Stencil 路由

**文件**: `translator.cpp` — 修改 `EndSourceFileAction()`

```cpp
if (MpiOpt) {
    if (dacppFile->hasBufferRegionPlan()) {
        rewriter->rewriteMPI_Stencil();  // 新入口
    } else {
        rewriter->rewriteMPI();          // 原有路径不变
    }
}
```

## 关键文件清单

| 文件 | 修改内容 |
|------|---------|
| `dpcppLib/include/MPIPlanner.h` | 添加 `HaloRegion`、`ParamHalo`、`computeParamHalo()`、`exchangeHalo()` |
| `rewriter/lib/Rewriter_MPI.cpp` | 扩展 `buildMPIRegionCode()` 添加 halo 逻辑；新增 sibling loop helpers；实现 `rewriteMPI_Stencil()` |
| `rewriter/include/Rewriter.h` | 声明 `rewriteMPI_Stencil()` |
| `translator.cpp` | 路由 MPI+region 到 `rewriteMPI_Stencil()` |

## 复用的现有代码

- `BufferRegionPlan`（`parser/include/DacppStructure.h`）：已有的时间迭代循环检测，含 `outerFor`、`siblingForStmts`、`capturedVars`
- `collect_positions_for_item()`（`MPIPlanner.h:195`）：计算每个 item 访问的全局位置
- `get_rank_item_range()`（`MPIPlanner.h:65`）：确定各 rank 的 item range
- `build_input_pack_map()` / `build_output_pack_map()`（`MPIPlanner.h:288-308`）：构建 pack map
- `g2l` 映射（`PackMap` 内）：定位 halo 数据在本地数组中的位置
- `buildMPIRegionCode()`（`Rewriter_MPI.cpp:929`）：已有的 init/submit/sync 代码框架
- `inferEffectiveParamModes()`（`Rewriter_MPI.cpp:256`）：参数真实读写模式推导
- `rewriteSiblingForsToParallel()`（`Rewriter_Buffer_new.cpp`）：sibling loop 处理模式
- `LoopVarAccessVisitor`（`buffer_template_new.cpp`）：sibling loop 变量读写分析

## 验证计划

### 编译验证
```bash
cd /Volumes/QUQ/working/dacpp
cmake --build build --target translator -j8
```

### 单机回归（确保无回归）
```bash
cd /Volumes/QUQ/working/dacpp/clang/tools/translator
bash test_local.sh
INCLUDE_EXTENDED_LOCAL_TESTS=1 bash test_local.sh
```

### MPI 回归（确保现有 MPI 逻辑不受影响）
```bash
bash test_mpi.sh
```

### MPI Stencil 测试（新增验证）
对 stencil 类样例单独测试 MPI 路径：
```bash
# stencil1.0
dacpp tests/stencil1.0/stencil.dac.cpp --mode=buffer --mpi
acpp-compile stencil.dac_sycl_buffer.cpp /tmp/stencil_mpi
DYLD_LIBRARY_PATH=... mpirun -np 2 /tmp/stencil_mpi

# waveEquation1.0
dacpp tests/waveEquation1.0/waveEquation.dac.cpp --mode=buffer --mpi
acpp-compile waveEquation.dac_sycl_buffer.cpp /tmp/wave_mpi
DYLD_LIBRARY_PATH=... mpirun -np 2 /tmp/wave_mpi

# MDP1.0
dacpp tests/MDP1.0/mdp.dac.cpp --mode=buffer --mpi
acpp-compile mdp.dac_sycl_buffer.cpp /tmp/mdp_mpi
DYLD_LIBRARY_PATH=... mpirun -np 2 /tmp/mdp_mpi
```
对比输出与单机 baseline 一致。

### 重点验证项
1. `stencil1.0` MPI 结果与单机一致
2. `waveEquation1.0` MPI 结果与单机一致
3. `MDP1.0` MPI 结果与单机一致
4. 非 stencil 样例（matMul、FOuLa 等）走原有 `rewriteMPI()` 路径不受影响
5. 单 rank（`mpirun -np 1`）也能正确运行（halo exchange 跳过）

## 实施优先级和边界

### 第一个版本（MVP）
- 只支持 1D 连续 item-space 分区（当前唯一分区方式）
- 只考虑左右邻居（rank-1, rank+1）
- 阻塞式 halo exchange（`MPI_Isend/Irecv + Waitall`）
- Sibling loop 只处理不含控制流的简单 for 循环（与单机 region 一致）
- 非 region 样例继续走原 `rewriteMPI()` 路径

### 后续可优化方向
- 非阻塞 halo exchange 与 kernel 执行重叠（computation-communication overlap）
- 多维 partitioning 的 halo（上下左右 8 邻居）
- 增量 halo（只传变化的边界）
- 多 `<->` 支持
- Device-direct halo exchange（GPUDirect RDMA 等）
