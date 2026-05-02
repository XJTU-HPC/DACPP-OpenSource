# MPI Stencil Phase 1 Status

更新时间：2026-05-02

## 1. 阶段 1 目标和边界

阶段 1 只优化当前 root-centric MPI Stencil 路径，不改变通信语义，不引入 halo 分布式 stencil。

目标：

- 把 `run()` 中每个 time step 重复计算的通信元数据搬到 `init()`
- 复用本地和 root 侧临时 buffer
- 保持当前 scatter / kernel / gather / broadcast 语义不变
- 不修改 DACPP 源码语法
- 不新增命令行开关
- 不改变普通 MPI wrapper 路径

不做：

- 不把 sibling update loops 合并进 stencil region
- 不做 rank 间 halo exchange
- 不改变 `FOuLa1.0` 这类循环内临时 view 的回退行为

## 2. 旧生成代码的问题

以 `waveEquation1.0` 为例，阶段 1 前的 MPI stencil `run()` 每次迭代都会重复执行：

- `MPI_Gather(&local_global_count_*)`
- `MPI_Gatherv(pack.globals)`
- `MPI_Gather(&send_count_*)`
- `MPI_Gatherv(writeback_globals)`
- counts / displs 计算
- root 侧 send / recv 临时 vector 重新分配
- 默认情况下每轮执行 wrapper timing `MPI_Reduce`

这些信息只依赖 shape、split、binding 和 rank item range，不依赖每步 tensor 数据，适合在 `init()` 中缓存。

## 3. 本轮完成项

### 3.1 Runtime helper

涉及文件：

- `clang/tools/translator/dpcppLib/include/mpi/Common.h`
- `clang/tools/translator/dpcppLib/include/mpi/Pack.h`

新增：

- `dacpp::mpi::GatheredIndexLayout`
  - `local_count`
  - `counts`
  - `displs`
  - `byte_counts`
  - `byte_displs`
  - `globals`
- `init_gathered_index_layout(...)`
  - 在 `init()` 中一次性收集 local count、displs 和 root 侧 globals
- `init_layout_byte_counts(...)`
  - 为 byte transport 类型预计算 byte counts / displs
- `pack_values_by_globals_parallel_range_into(...)`
  - root 侧复用 send buffer，按 cached globals 打包输入值
- `build_local_slots_for_globals(...)`
  - 为 writeback globals 预计算本地 slot
- `pack_values_by_slots_parallel_into(...)`
  - 每步按 cached slots 抽取 writeback values 到复用 buffer

### 3.2 Stencil codegen

涉及文件：

- `clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Stencil_Codegen.cpp`

已调整：

- `ctx` 为每个参数保存 `local_<name>`
- `READ/READ_WRITE` 参数保存：
  - `input_layout_<name>`
  - `global_<name>`
  - `sendbuf_<name>`
- `WRITE/READ_WRITE` 参数保存：
  - `output_layout_<name>`
  - `writeback_slots_<name>`
  - `writeback_values_<name>`
  - `global_recv_values_<name>`
  - `global_out_<name>`
- `init()` 构建 pack plan 后初始化 input / output layout
- `run()` 删除重复 metadata gather，只保留数据相关通信
- `WRITE` 参数每轮 kernel 前重置 local buffer，保持原 fresh vector 语义
- wrapper timing `MPI_Reduce` 只在 `DACPP_MPI_PROFILE` 开启时生成执行

## 4. 当前生成代码形态

`waveEquation.mpi.dac_sycl_buffer.cpp` 现在的 stencil context 中包含：

```cpp
dacpp::mpi::GatheredIndexLayout input_layout_cur;
std::vector<double> global_cur;
std::vector<double> sendbuf_cur;
dacpp::mpi::GatheredIndexLayout output_layout_next;
std::vector<int32_t> writeback_slots_next;
std::vector<double> writeback_values_next;
```

`init()` 中执行一次：

```cpp
dacpp::mpi::init_gathered_index_layout(ctx.input_layout_cur, ...);
dacpp::mpi::init_gathered_index_layout(ctx.output_layout_next, ...);
dacpp::mpi::build_local_slots_for_globals(...);
```

`run()` 中只保留：

```cpp
MPI_Scatterv(... input_layout_*.counts/displs ...);
MPI_Gatherv(writeback_values_*.data(), ... output_layout_*.counts/displs ...);
```

结构检查结果：

- `run()` 中不再出现 `MPI_Gather(&local_global_count_*)`
- `run()` 中不再出现 `MPI_Gatherv(...pack.globals...)`
- `run()` 中不再出现 `MPI_Gatherv(...writeback_globals...)`
- `MPI_Reduce` 仍存在于生成代码中，但已包在 `dacpp::mpi::profilingEnabled()` 分支内

## 5. 验证结果

构建：

```bash
cd /Volumes/QUQ/working/dacpp
cmake --build build --target translator -j8
```

结果：通过。

重点回归：

```bash
cd /Volumes/QUQ/working/dacpp/clang/tools/translator
bash test_mpi.sh waveEquation1.0 stencil1.0 FOuLa1.0 mpiDenseCoverSibling1.0
```

结果：

```text
4 tests | 4 passed | 0 failed | 0 skipped
```

完整 MPI 回归：

```bash
cd /Volumes/QUQ/working/dacpp/clang/tools/translator
bash test_mpi.sh
```

结果：

```text
13 tests | 13 passed | 0 failed | 0 skipped
```

## 6. 还需要做什么

### Phase 2: stencil region 化

目标：

- 识别以 `<->` 开头、后续包含状态更新 / 边界更新 sibling loops 的 time-step region
- 生成 `init_region / run_region / sync_region`
- root 侧在 region 内维护 dense state arrays
- 减少每步对 tensor 对象的 `array2Tensor / tensor2Array`
- 对 `waveEquation` 这类用例避免每步 broadcast 临时输出 tensor

### Phase 3: halo 分布式 stencil

目标：

- 识别规则 2D stencil
- rank 按行或 tile 持有局部网格
- 每步只做 neighbor halo exchange
- kernel 在本地 subdomain 上运行
- 最终按输出需求 gather / print

建议：

- Phase 3 使用显式 opt-in 开关，例如 `--mpi-stencil-distributed`
- 不和当前 root-centric 路径混在同一次改动中实现

## 7. 一句话总结

阶段 1 已完成：MPI Stencil 现在会在 `init()` 中缓存通信布局，并在 `run()` 中复用 layout 和 buffer，只保留每步数据通信；当前构建和完整 MPI 回归均通过。
