# DACPP Translator 四条翻译路径详解

DACPP Translator 是一个基于 Clang AST 的源到源翻译器，将 DACPP 语法（`<->` 数据关联表达式）翻译为 SYCL 并行代码。根据是否启用 MPI 和是否识别到 stencil/region 模式，翻译器分为四条路径。

## 路由总览

```
translator.cpp EndSourceFileAction() 中的路由逻辑：

                         ┌──────────┐
                         │ 解析完成  │
                         └────┬─────┘
                              │
                    ┌─────────▼──────────┐
                    │  --mpi 是否启用？    │
                    └──┬──────────────┬───┘
                       │ 否           │ 是
                       ▼              ▼
              ┌─────────────┐  ┌──────────────────┐
              │ --mode=     │  │ analyzeBuffer    │
              │ buffer      │  │ RegionPlan()     │
              └──────┬──────┘  └──┬───────────┬───┘
                     │            │ 真        │ 假
                     ▼            ▼           ▼
            ┌──────────────┐  ┌─────────┐ ┌──────────┐
            │ rewriteDac   │  │rewrite  │ │rewrite   │
            │ _Buffer()    │  │MPI_     │ │MPI()     │
            │              │  │Region() │ │wrapper   │
            └──────┬───────┘  └─────────┘ └──────────┘
                   │
         ┌─────────▼────────────┐
         │ hasBufferRegionPlan? │
         └──┬───────────────┬───┘
            │ 真            │ 假
            ▼               ▼
    ┌───────────────┐  ┌──────────────┐
    │ 单机 stencil  │  │ 单机非 stencil│
    │ (region 优化) │  │ (标准 buffer) │
    └───────────────┘  └──────────────┘
```

对应关系：

| 路径 | MPI | Stencil/Region | 入口函数 |
|------|-----|----------------|---------|
| 单机非 stencil | 否 | 否 | `rewriteDac_Buffer()` 标准分支 |
| 单机 stencil | 否 | 是 | `rewriteDac_Buffer()` region 分支 |
| 多机非 stencil | 是 | 否 | `rewriteMPI()` wrapper 路径 |
| 多机 stencil | 是 | 是 | `rewriteMPI_Region()` → `rewriteMPI_Stencil()` |

---

## 1. 单机非 stencil（标准 Buffer 路径）

### 触发条件

- 命令行：`dacpp <file> --mode=buffer`
- 不满足 `hasBufferRegionPlan()`（即不存在外层循环包裹的单一 `<->`，或存在多个 `<->`）

### 入口

[translator.cpp:551](translator.cpp#L551)：`rewriter->rewriteDac_Buffer()`

### 执行流程

#### 1.1 AST 信息收集

翻译器在 AST 匹配阶段已完成：
- 匹配 `<->` 二元运算符，识别出 shell 调用和 calc 函数
- 收集 shell 参数的类型和名称（`dacppFile->shellVars`）
- 收集外层循环信息（`dacppFile->forStatement`）
- 收集主函数的 `FunctionDecl`、`Stmt*` 和 `ASTContext`

#### 1.2 参数模式推导

[Rewriter_Buffer_new.cpp](rewriter/lib/Rewriter_Buffer_new.cpp) 中 `inferEffectiveBufferParamModes()` 分析 calc 函数体，推导每个参数的有效访问模式（`READ` / `WRITE` / `READ_WRITE`）：

- 遍历 calc 函数体的 AST
- 追踪每个参数的读、写和更新读（如 `a += b` 中 `a` 既读又写）
- 以 shell 参数声明的顺序为基准

#### 1.3 代码生成：计算函数（Calc）

为每个 `<->` 表达式生成 SYCL 计算函数：

```cpp
void <calc_name>(
    const float* param0,        // READ 参数加 const
    float* param1,              // WRITE / READ_WRITE 不加 const
    int param0_0, int param0_1, // 各维度的划分索引
    int param0_0_shape,         // 各维度的划分大小
    ...
    sycl::accessor<int, 1, read> info_param0_acc  // 划分信息
) {
    // calc 函数体，原样嵌入
}
```

函数签名包含：
- calc 参数指针（按访问模式加或不加 `const`）
- 每个参数各维度的索引变量
- 每个参数各维度的 shape 变量
- 划分信息的 accessor

#### 1.4 代码生成：Shell 包装函数

包装函数的签名包含 shell 参数和 for 循环中的非 shell 变量：

```cpp
void <shell>_<calc>(Tensor<float> param0, Tensor<float> param1, ...) {
```

包装函数内部的生成步骤：

**a) 算子初始化**

从 shell 参数的 split 信息初始化划分算子：
- `IndexSplit` → 生成索引划分初始化代码
- `RegularSplit` → 生成规则分区初始化代码（size + stride）

**b) 数据关联构建**

将算子与参数关联：
```
DataInfo → AddOps → DataOps
```
区分输入算子（In_Ops）和输出算子（Out_Ops）。

**c) Buffer 分配与数据搬迁**

对每个参数：
```cpp
// Host → Device
buffer<float> b_param0{param0.getData(), param0.getSize()};
```

**d) Accessor 初始化**

根据参数访问模式生成对应的 accessor：
- `READ` → `accessor<float, 1, access::mode::read>`
- `WRITE` → `accessor<float, 1, access::mode::discard_write>`
- `READ_WRITE` → `accessor<float, 1, access::mode::read_write>`

**e) Kernel 提交**

```cpp
q.submit([&](sycl::handler& h) {
    // 获取 accessor
    // 将指针传给参数生成器
    h.parallel_for(sycl::range<1>(Item_Size), [=](sycl::id<1> id) {
        // 通过 ParameterGenerate 生成参数视图
        // 调用 <calc_name>(...)
    });
});
```

工作项数量 `Item_Size` 由划分信息自动计算。嵌套 for 循环被转换为 2D `parallel_for`。

#### 1.5 主函数改写

- 删除 shell 和 calc 函数定义
- 在程序入口插入生成的函数定义
- 将 `<->` 表达式替换为 `shell_calc(param0, param1, ...)`

### 生成的代码骨架

```cpp
#include <sycl/sycl.hpp>
// ... 其他头文件

// 计算函数
void calc_func(const float* a, float* b, int a_0, int a_0_shape, ...) {
    // 原 calc 函数体
}

// Shell 包装函数
void shell_calc(Tensor<float> a, Tensor<float> b) {
    // 1. 算子初始化
    // 2. Buffer 创建
    sycl::queue q;
    buffer<float> b_a{a.getData(), a.getSize()};
    buffer<float> b_b{b.getData(), b.getSize()};
    // 3. Kernel 提交
    q.submit([&](sycl::handler& h) {
        accessor<float, 1, read> acc_a(b_a, h);
        accessor<float, 1, discard_write> acc_b(b_b, h);
        h.parallel_for(range<1>(N), [=](id<1> id) {
            calc_func(&acc_a[...], &acc_b[...], ...);
        });
    });
    q.wait();
    // 4. Device → Host 回收
}

int main() {
    // ...
    shell_calc(a, b);  // 替换了原来的 a <-> b
}
```

---

## 2. 单机 stencil（Buffer Region 优化路径）

### 触发条件

- 命令行：`dacpp <file> --mode=buffer`
- `hasBufferRegionPlan()` 为真

### Region 条件检测

[analyzeBufferRegionPlan()](parser/lib/DacppStructure.cpp) 检测以下条件：

1. **必须恰好有一个 `<->` 表达式**
2. **该 `<->` 必须位于一个外层 for 循环内**
3. **外层循环体必须是复合语句 `{...}`**
4. **`<->` 必须是循环体的第一条顶层语句**
5. **循环体内 `<->` 之后的语句（sibling stmts）不能包含不支持的控制流**（`while`、`switch`、`return`、`break`、`continue`、`goto`）
6. **sibling for 循环的内部也不能有上述控制流**

满足以上条件时，`BufferRegionPlan::enabled = true`，并记录：
- `exprIndex`：对应的表达式索引
- `outerFor`：外层 for 循环的 AST 节点
- `dacExpr`：`<->` 运算符的 AST 节点
- `siblingStmts`：`<->` 之后的同级语句列表
- `capturedVars` / `capturedNonShellVars`：循环体内捕获的变量

### 入口

[Rewriter_Buffer_new.cpp:1408](rewriter/lib/Rewriter_Buffer_new.cpp)：进入 region 分支，调用 `buildOptimizedBufferRegionCode()`。

### 执行模型

单机 stencil 路径采用 **ctx → init → submit → sync** 四段式模型（没有 halo，因为没有跨进程通信）：

```cpp
__dacpp_ctx_<shell>_<calc> ctx;
__dacpp_init_<shell>_<calc>(ctx, tensors...);
for (...) {
    __dacpp_submit_<shell>_<calc>(ctx);
    // sibling helpers（如果有）
}
__dacpp_sync_<shell>_<calc>(ctx, tensors...);
```

### Region 代码生成细节

#### 2.1 ctx 结构体

生成一个包含所有运行时状态的上下文结构体：

```cpp
struct __dacpp_ctx_<shell>_<calc> {
    sycl::queue q{};

    int64_t total_items = 1;
    int64_t local_item_count = 0;
    std::vector<int64_t> binding_split_sizes;

    // 每个参数的字段：
    AccessPattern pattern_<param>;
    PackMap pack_<param>;
    std::vector<int32_t> slots_<param>;
    std::vector<float> h_<param>;        // host 副本
    sycl::buffer<float> buf_<param>;     // SYCL buffer
    std::vector<int32_t> global_to_local_<param>;  // sibling 用

    // 非 shell 参数的捕获变量
    int t;  // 例如时间步变量
};
```

#### 2.2 init 函数

```cpp
void __dacpp_init_<shell>_<calc>(ctx_type& ctx, Tensor<float>& param0, ...) {
    // 1. 创建 SYCL queue
    ctx.q = sycl::queue(sycl::default_selector_v);

    // 2. 为每个参数构造 AccessPattern
    //    从 shell 的 split/binding 信息推导访问模式

    // 3. 计算工作项
    ctx.total_items = binding_split_sizes 的乘积;

    // 4. 构造 pack-map 和 slots
    //    按参数的 READ/WRITE/READ_WRITE 模式选择打包策略

    // 5. 将 tensor 数据打包到 host 副本 h_<param>
    //    init 决策来自 BufferRegionTransferPolicy：
    //    - needsInitCopy[param] 为真时执行复制

    // 6. 创建 SYCL buffer
    ctx.buf_<param> = sycl::buffer<float>(h_<param>.data(), h_<param>.size());
}
```

传输策略由 `analyzeBufferRegionTransferPolicy()` 决定：
- `needsInitCopy`：参数是否需要在 init 时复制数据（READ 和 READ_WRITE 需要）
- `needsSyncCopy`：参数是否需要在 sync 时回收数据（WRITE 和 READ_WRITE 需要）
- `writtenInRegion`：参数在 region 内是否被写入
- `hostUsedAfterLoop`：循环后 host 侧是否继续使用

#### 2.3 submit 函数

```cpp
void __dacpp_submit_<shell>_<calc>(ctx_type& ctx) {
    if (ctx.local_item_count <= 0) return;

    ctx.q.submit([&](sycl::handler& h) {
        // 为每个参数获取 accessor
        auto acc_<param> = ctx.buf_<param>.get_access<mode>(h);
        auto slots_acc = ctx.slots_buf_<param>.get_access<read>(h);

        h.parallel_for(sycl::range<1>(ctx.local_item_count), [=](sycl::id<1> idx) {
            int item_linear = idx[0];
            // 构造 View1D / View2D
            auto view_<param> = View1D<float>(data, slots, offset);
            // 调用 calc 函数
            calc_mpi_local(view_0, view_1, ...);
        });
    });
}
```

核心：submit 不等待 kernel 完成，允许异步执行。

#### 2.4 sibling helpers

如果 region 内 `<->` 之后有 for 循环（sibling for），生成 `__dacpp_submit_region_*` helper：

```cpp
void __dacpp_submit_region_<shell>_<calc>_stmt_0(ctx_type& ctx) {
    // 1. D2H：从 buf 回收到 host 副本
    // 2. 为 sibling 写入的参数构造全局可覆盖的 pack 和 global_to_local 查表
    // 3. 在设备侧提交 sibling kernel
    // 4. 从 dirty 位图提取真实写入位置
    // 5. 同步回 buf
}
```

单机场景下的 sibling helper 没有 MPI 通信，只有本地的 D2H / kernel / H2D 循环。

#### 2.5 sync 函数

```cpp
void __dacpp_sync_<shell>_<calc>(ctx_type& ctx, Tensor<float>& param0, ...) {
    ctx.q.wait();

    // 对需要写回的参数：
    // 1. D2H：从 buf 读回 host
    {
        sycl::host_accessor ha(ctx.buf_<param>, sycl::read_only);
        for (i...) ctx.h_<param>[i] = ha[i];
    }
    // 2. 将 host 数据写回 tensor
    param0.array2Tensor(ctx.h_<param>);
}
```

### 与非 stencil 路径的关键区别

| 特性 | 非 stencil | stencil (region) |
|------|-----------|-----------------|
| 执行模型 | 一次提交 | init → 循环 submit → sync |
| 数据生命周期 | 按次分配回收 | init 一次分配，循环内复用 |
| Buffer 创建 | 每次 `<->` 重新创建 | init 时创建，循环内复用 |
| sibling 语句 | 不支持 | 支持 for 和赋值语句 |
| 内存优化 | 标准 | 避免重复分配和搬迁 |

### 主函数改写

```
原始代码:
    for (int t = 0; t < T; t++) {
        A <-> B;
        for (int i = 0; i < N; i++) { C[i] = B[i] * 2; }
    }

翻译后:
    __dacpp_ctx_shell_calc ctx;
    __dacpp_init_shell_calc(ctx, A, B, C);
    for (int t = 0; t < T; t++) {
        __dacpp_submit_shell_calc(ctx);
        __dacpp_submit_region_shell_calc_stmt_0(ctx);
    }
    __dacpp_sync_shell_calc(ctx, A, B, C);
```

---

## 3. 多机非 stencil（MPI Wrapper 路径）

### 触发条件

- 命令行：`dacpp <file> --mode=buffer --mpi`
- `hasBufferRegionPlan()` 为假（没有外层循环包裹，或存在多个 `<->`）

### 入口

[translator.cpp:547](translator.cpp#L547)：`rewriter->rewriteMPI()`
[Rewriter_MPI.cpp:12](rewriter/lib/Rewriter_MPI.cpp)：`Rewriter::rewriteMPI()`

### 执行模型

wrapper 路径按**一次调用**完成 scatter、局部执行和 gather：

```
rank 0: 打包全量数据 → MPI_Scatterv → 局部执行 → MPI_Gatherv → apply writeback → Bcast
rank N:                接收局部数据 → 局部执行 → 发送写回数据
```

### 代码生成流程

#### 3.1 Prelude

`buildPrelude()` 生成头文件引用和全局辅助代码。

#### 3.2 Local Calc 函数

`buildLocalCalcCode()` 生成一个可直接在 SYCL kernel 内调用的本地计算函数：

```cpp
void <calc>_mpi_local(
    View1D<const float> view_param0,  // READ
    View1D<float> view_param1,        // WRITE
    ...
) {
    // 原 calc 函数体
}
```

参数用 `View1D` / `View2D` 包装，支持 packed 数组上的索引访问。

#### 3.3 Wrapper 函数

`buildWrapperCode()` 生成完整的 MPI wrapper 函数，分为五个阶段：

**a) 初始化阶段**

```cpp
void <shell>_<calc>(Tensor<float>& param0, ...) {
    int mpi_rank = 0, mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    sycl::queue q(sycl::default_selector_v);
```

**b) AccessPattern 与 Pack-map 构建**

```cpp
    // 为每个参数构造 AccessPattern
    dacpp::mpi::AccessPattern pattern_param0;
    pattern_param0.param_id = 0;
    pattern_param0.name = "param0";
    pattern_param0.mode = dacpp::mpi::AccessMode::Read;
    pattern_param0.data_info.dim = 2;
    pattern_param0.data_info.sizes = {rows, cols};
    // 添加 split 操作：dim_id, size, stride
    // 设置 bind_set_id, bind_offset_expr

    // 合并 binding_split_sizes
    binding_split_sizes = ...;
    total_items = binding_split_sizes 的乘积;

    // 计算 rank 的 item range
    auto item_range = dacpp::mpi::get_rank_item_range(total_items, mpi_rank, mpi_size);
```

Pack-map 选择规则：
| 参数模式 | 使用函数 |
|---------|---------|
| READ | `build_input_pack_map()` |
| WRITE | `build_output_pack_map()` |
| READ_WRITE | `build_rw_pack_map()` |

**c) Scatter 阶段（数据分发）**

```cpp
    // rank 0 执行打包
    if (mpi_rank == 0) {
        for (int r = 0; r < mpi_size; r++) {
            auto r_range = get_rank_item_range(total_items, r, mpi_size);
            auto r_slots = build_item_slots(r_range, pattern, pack);
            send_counts[r] = r_slots.size();
            // 从全量数据中提取该 rank 需要的元素
            for (auto idx : r_slots) {
                send_buf[r].push_back(global_data[pack.globals[idx]]);
            }
        }
    }

    // 分发
    MPI_Scatter(send_counts.data(), 1, MPI_INT, &recv_count, 1, MPI_INT, 0, ...);
    std::vector<float> local_data(recv_count);
    MPI_Scatterv(send_buf_packed, send_counts, displs, MPI_FLOAT,
                 local_data.data(), recv_count, MPI_FLOAT, 0, ...);
```

核心：**不是按 tensor 物理空间做规则切块，而是按 item-space 分配 rank 的逻辑工作集**。每个 rank 只接收它真正需要访问的元素。

**d) 局部 SYCL 执行**

```cpp
    if (item_range.size() > 0) {
        sycl::buffer<float> buf_param0(local_data.data(), local_data.size());
        sycl::buffer<int32_t> slots_buf(slots.data(), slots.size());

        q.submit([&](sycl::handler& h) {
            auto acc = buf_param0.get_access<mode>(h);
            auto slots_acc = slots_buf.get_access<read>(h);

            h.parallel_for(sycl::range<1>(item_range.size()), [=](sycl::id<1> idx) {
                int item_linear = idx[0];
                auto view0 = View1D<float>{&acc[0], &slots_acc[0],
                                           item_linear * partition_size};
                calc_mpi_local(view0, ...);
            });
        });
        q.wait();
    }
```

**e) Gather + Writeback + Broadcast**

```cpp
    // 构造写回值
    auto wb = build_writeback_values(local_data, pack);

    // Gather 到 rank 0
    MPI_Gather(&send_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, ...);
    MPI_Gatherv(wb.data(), ..., global_wb_data, recv_counts, displs, ..., 0, ...);

    // rank 0 执行写回
    if (mpi_rank == 0) {
        apply_writeback_by_globals(global_wb_values, pack.writeback_globals, global_tensor);
        param0.array2Tensor();
    }

    // 按 tensorNeedsBroadcast() 判断是否需要广播
    if (needs_bcast) {
        MPI_Bcast(param0.getData(), param0.getSize(), MPI_FLOAT, 0, ...);
        if (mpi_rank != 0) param0.array2Tensor();
    }
```

#### 3.4 MPI 初始化与清理

在 main 函数体开头插入 MPI_Init，在 return 和函数末尾插入 MPI_Finalize：

```cpp
int main() {
    MPI_Init(...);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    if (mpi_rank != 0) freopen("/dev/null", "w", stdout);
    // ... 用户代码 ...

    MPI_Finalize();
}
```

非 0 进程的 stdout 被重定向到 `/dev/null`，避免重复输出。

### Item-space 分发详解

wrapper 路径的核心创新在于基于 item-space 的分发：

1. **shell 的 split/binding 信息**定义了一个 item-space
2. 每个 item 代表一次独立的计算任务
3. `get_rank_item_range()` 把 item-space 按连续区间分给各 rank
4. 对每个 rank，用 `collect_positions_for_item()` 计算它需要访问哪些全局位置
5. 通过 pack-map 建立全局索引到局部索引的映射
6. 只 scatter 真正需要的数据

这种方式避免了按 tensor 物理边界切割导致的冗余传输。

---

## 4. 多机 stencil（MPI Region 路径）

### 触发条件

- 命令行：`dacpp <file> --mode=buffer --mpi`
- `hasBufferRegionPlan()` 为真（存在外层循环包裹的单一 `<->`）

### 入口

[translator.cpp:544](translator.cpp#L544)：`rewriter->rewriteMPI_Region()`
[Rewriter_MPI.cpp:198](rewriter/lib/Rewriter_MPI.cpp)：`rewriteMPI_Region()` → `rewriteMPI_Stencil()`

### 执行模型

MPI Region 路径采用 **ctx → init → 循环{submit → halo → sibling helpers} → sync** 的完整五段式模型：

```cpp
__dacpp_mpi_ctx_<shell>_<calc> ctx;
__dacpp_mpi_init_<shell>_<calc>(ctx, tensors...);
for (int t = 0; t < T; t++) {
    __dacpp_mpi_submit_<shell>_<calc>(ctx);
    __dacpp_mpi_halo_<shell>_<calc>(ctx);
    __dacpp_mpi_submit_region_<shell>_<calc>_stmt_0(ctx);  // sibling for
    __dacpp_mpi_submit_region_<shell>_<calc>_stmt_1(ctx);  // sibling assign
}
__dacpp_mpi_sync_<shell>_<calc>(ctx, tensors...);
```

### Region 代码生成细节

代码由 7 个模块协同生成：
- [Rewriter_MPI_Region_Codegen.cpp](rewriter/lib/mpi/Rewriter_MPI_Region_Codegen.cpp) — 编排层
- [Rewriter_MPI_Region_Ctx.cpp](rewriter/lib/mpi/Rewriter_MPI_Region_Ctx.cpp) — ctx
- [Rewriter_MPI_Region_Init.cpp](rewriter/lib/mpi/Rewriter_MPI_Region_Init.cpp) — init
- [Rewriter_MPI_Region_Submit.cpp](rewriter/lib/mpi/Rewriter_MPI_Region_Submit.cpp) — submit
- [Rewriter_MPI_Region_Halo.cpp](rewriter/lib/mpi/Rewriter_MPI_Region_Halo.cpp) — halo
- [Rewriter_MPI_Region_Sync.cpp](rewriter/lib/mpi/Rewriter_MPI_Region_Sync.cpp) — sync
- [Rewriter_MPI_Region_Sibling.cpp](rewriter/lib/mpi/Rewriter_MPI_Region_Sibling.cpp) — sibling

#### 4.1 ctx 结构体

```cpp
struct __dacpp_mpi_ctx_<shell>_<calc> {
    // MPI 基本信息
    int mpi_rank = 0;
    int mpi_size = 1;

    // 设备队列
    sycl::queue q{};

    // Item-space 信息
    int64_t total_items = 1;
    int64_t local_item_count = 0;
    std::vector<int64_t> binding_split_sizes;

    // Halo 标志
    bool has_halo = false;

    // 每个参数的完整状态：
    dacpp::mpi::AccessPattern pattern_<param>;
    dacpp::mpi::PackMap pack_<param>;
    std::vector<int32_t> slots_<param>;
    std::vector<float> local_<param>;                              // host 侧 packed 数组
    std::unique_ptr<sycl::buffer<float>> buf_<param>;              // device 侧 packed buffer
    std::unique_ptr<sycl::buffer<int32_t>> slots_buf_<param>;      // device 侧 slots buffer
    std::vector<int32_t> global_to_local_<param>;                  // sibling 全局索引查表
    std::unique_ptr<sycl::buffer<int32_t>> global_to_local_buf_<param>;
    int <param>_partition_size = 0;
    int <param>_cols = 0;                                          // rank > 1 时的列数
    dacpp::mpi::ParamHalo halo_<param>;                            // halo 信息

    // 非 shell 捕获变量
    int t;
};
```

#### 4.2 init 函数

```cpp
void __dacpp_mpi_init_<shell>_<calc>(ctx_type& ctx, Tensor<float>& param0, ...) {
    MPI_Comm_rank(MPI_COMM_WORLD, &ctx.mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ctx.mpi_size);
    ctx.q = sycl::queue(sycl::default_selector_v);

    // --- 为每个参数构造 AccessPattern ---
    ctx.pattern_<param>.param_id = <id>;
    ctx.pattern_<param>.mode = <READ/WRITE/READ_WRITE>;
    // 设置 operations, partition_shape, bind_set_id, bind_offset_expr

    // --- 合并 binding_split_sizes ---
    ctx.binding_split_sizes = ...;
    ctx.total_items = binding_split_sizes 的乘积;

    // --- 计算本 rank 的 item_range ---
    auto item_range = get_rank_item_range(ctx.total_items, ctx.mpi_rank, ctx.mpi_size);
    ctx.local_item_count = item_range.size();

    // --- 构造 pack-map 和 slots ---
    ctx.pack_<param> = build_input_pack_map(pattern, item_range);  // READ
    ctx.slots_<param> = build_item_slots(item_range, pattern, pack);

    // --- Scatter（一次性数据分发）---
    if (needsInitScatter[param]) {
        if (ctx.mpi_rank == 0) {
            // 打包各 rank 需要的数据
            for (int r = 0; r < ctx.mpi_size; r++) {
                auto r_range = get_rank_item_range(ctx.total_items, r, ctx.mpi_size);
                // 收集该 rank 需要的全局位置，打包数据
            }
        }
        MPI_Scatterv(...);  // 分发到各 rank
    }

    // --- 创建 local 和 buf ---
    ctx.local_<param>.resize(pack.size());
    ctx.buf_<param> = make_unique<sycl::buffer<float>>(ctx.local_<param>.data(), ...);
    ctx.slots_buf_<param> = make_unique<sycl::buffer<int32_t>>(ctx.slots_<param>.data(), ...);

    // --- 计算 Halo ---
    if (ctx.mpi_size > 1) {
        ctx.has_halo = true;
        ctx.halo_<param> = computeParamHalo(pattern, mode, item_range,
                                              total_items, mpi_rank, mpi_size);
    }

    // --- 广播非 shell 变量 ---
    MPI_Bcast(&ctx.t, 1, MPI_INT, 0, MPI_COMM_WORLD);
}
```

传输策略由 `analyzeMPIRegionTransferPolicy()` 决定：
- `needsInitScatter[param]`：READ 和 READ_WRITE 参数需要 scatter
- `needsSyncGather[param]`：WRITE 和 READ_WRITE 参数需要 gather

`inferMPIRegionStorageModes()` 会把 sibling loop 的读写行为合并到参数的存储模式中。

#### 4.3 submit 函数

```cpp
void __dacpp_mpi_submit_<shell>_<calc>(ctx_type& ctx) {
    if (ctx.local_item_count <= 0) return;

    ctx.q.submit([&](sycl::handler& h) {
        // 获取 accessor
        auto acc_<param> = ctx.buf_<param>->get_access<mode>(h);
        auto slots_acc = ctx.slots_buf_<param>->get_access<read>(h);

        h.parallel_for(sycl::range<1>(ctx.local_item_count), [=](sycl::id<1> idx) {
            int item_linear = idx[0];
            // 构造视图
            auto view_<param> = dacpp::mpi::View1D<float>{
                &acc_<param>[0], &slots_acc[0],
                item_linear * ctx.<param>_partition_size
            };
            // 或 View2D<float>{..., ctx.<param>_cols}
            <calc>_mpi_local(view_0, view_1, ...);
        });
    });
    // 不等待！异步提交
}
```

#### 4.4 halo 函数

```cpp
void __dacpp_mpi_halo_<shell>_<calc>(ctx_type& ctx) {
    ctx.q.wait();  // 等待 submit 完成
    if (!ctx.has_halo || ctx.local_item_count <= 0) return;

    // 对每个写参数：
    // 1. D2H：从 buf 读回 local
    {
        sycl::host_accessor ha(*ctx.buf_<param>, sycl::read_only);
        for (i...) ctx.local_<param>[i] = ha[i];
    }

    // 2. 执行 halo exchange
    MPI_Datatype mpi_dt = MPI_FLOAT;
    dacpp::mpi::exchangeHalo(ctx.local_<param>, ctx.halo_<param>, &mpi_dt);

    // 3. H2D：把更新后的 local 写回 buf
    {
        sycl::host_accessor ha(*ctx.buf_<param>, sycl::write_only);
        for (i...) ha[i] = ctx.local_<param>[i];
    }
}
```

**Halo 语义详解**：

`computeParamHalo()` 的计算逻辑：
- **recv halo** = 我的输入 ∩ 邻居的输出
  - 我需要读但自己不产生的数据，从产生它的邻居那里接收
- **send halo** = 我的输出 ∩ 邻居的输入
  - 我产生的但邻居需要读的数据，发送给邻居

邻居集合：按一维 item-space 连续分区，只考虑 `rank - 1` 和 `rank + 1`。

`exchangeHalo()` 的执行过程：
1. 对每个 `HaloRegion`，从 `local_data` 按 `send_local_slots` 提取发送数据
2. `MPI_Isend` + `MPI_Irecv` 非阻塞通信
3. `MPI_Waitall` 等待完成
4. 按 `recv_local_slots` 将接收数据写入 `local_data`

#### 4.5 sibling helpers

```cpp
void __dacpp_mpi_submit_region_<shell>_<calc>_stmt_<idx>(ctx_type& ctx) {
    ctx.q.wait();

    // 1. D2H：从 buf 回收 local
    {
        sycl::host_accessor ha(*ctx.buf_<param>, sycl::read_only);
        for (i...) ctx.local_<param>[i] = ha[i];
    }

    // 2. 为 sibling 写入的参数构造全局可覆盖的 pack 和 dense fallback
    std::vector<float> dense_<param>(lookup_size);
    std::vector<unsigned char> dirty_<param>(lookup_size, 0);
    std::vector<float> dense_shadow_<param>(lookup_size);

    // 3. 广播全局初值（init 阶段已广播）
    //    读参数走 dense fallback（无 dirty）
    //    写参数使用 PackedElementRef（带 dirty + shadow）

    // 4. 设备侧 sibling kernel
    ctx.q.submit([&](sycl::handler& h) {
        auto acc = ctx.buf_<param>->get_access<rw>(h);
        auto lookup_acc = ctx.global_to_local_buf_<param>->get_access<read>(h);

        h.parallel_for(sycl::range<1>(loop_count), [=](sycl::id<1> idx) {
            int i = loop_begin + idx[0];
            // PackedVectorView / PackedMatrixView
            auto view = PackedVectorView<float>{
                data, lookup, lookup_size, dirty, dense_fallback, dense_shadow
            };
            // sibling loop body（使用原语义索引访问）
            C[i] = B[i] * 2;
        });
    });
    ctx.q.wait();

    // 5. 从 dirty 位图提取真实写入位置
    std::vector<int64_t> written_globals;
    for (i...) {
        if (dirty[i]) written_globals.push_back(global_index_of(i));
    }

    // 6. 跨 rank 合并去重
    //    MPI_Allgatherv 收集所有 rank 的写入索引
    //    去重排序

    // 7. 稀疏 MPI_Allreduce 值同步
    //    对每个写入位置：
    //    MPI_Allreduce(&value, &sum, 1, MPI_FLOAT, MPI_SUM, ...)
    //    MPI_Allreduce(&present, &count, 1, MPI_INT, MPI_SUM, ...)
    //    result = sum / count  // 多 rank 重叠写取平均

    // 8. 写回 ctx.local_* 和 ctx.buf_*
}
```

**PackedElementRef** 统一三类语义：
- 命中本地 slot → 直接读写 `ctx.buf_*`
- 未命中 → 读 dense fallback
- 写入 → 更新 dense_shadow + 打 dirty 标记

#### 4.6 sync 函数

```cpp
void __dacpp_mpi_sync_<shell>_<calc>(ctx_type& ctx, Tensor<float>& param0, ...) {
    ctx.q.wait();

    // 对需要 sync 的每个参数：
    // 1. D2H
    {
        sycl::host_accessor ha(*ctx.buf_<param>, sycl::read_only);
        for (i...) ctx.local_<param>[i] = ha[i];
    }

    // 2. 构造 writeback
    auto wb = build_writeback_values(ctx.local_<param>, ctx.pack_<param>);
    auto wb_globals = ctx.pack_<param>.writeback_globals;

    // 3. Gather 到 rank 0
    int send_count = wb.size();
    MPI_Gather(&send_count, 1, MPI_INT, recv_counts, 1, MPI_INT, 0, ...);
    MPI_Gatherv(wb.data(), send_count, MPI_FLOAT,
                global_values, recv_counts, displs, MPI_FLOAT, 0, ...);
    MPI_Gatherv(wb_globals.data(), send_count, MPI_INT64_T,
                global_globals, recv_counts64, displs64, MPI_INT64_T, 0, ...);

    // 4. rank 0 apply writeback
    if (ctx.mpi_rank == 0) {
        apply_writeback_by_globals(global_values, global_globals, global_tensor);
        param0.array2Tensor();
    }

    // 5. Broadcast（如需要）
    if (needs_bcast) {
        MPI_Bcast(param0.getData(), size, MPI_FLOAT, 0, ...);
        if (ctx.mpi_rank != 0) param0.array2Tensor();
    }
}
```

### 与 wrapper 路径的关键区别

| 特性 | Wrapper 路径 | Region 路径 |
|------|-------------|-------------|
| 执行次数 | 一次 `<->` = 一次完整 scatter+calc+gather | init 一次，循环多次 submit+halo，最后 sync |
| Halo 交换 | 无 | 每次循环后执行邻居 halo 同步 |
| 数据驻留 | 每次从全量 scatter | init scatter 后数据驻留 local |
| Sibling 支持 | 不支持 | 支持（含 dirty 稀疏同步） |
| 上下文 | 无状态函数调用 | 有状态 ctx 结构体 |
| 适用场景 | 独立任务并行 | 时间推进、stencil 迭代 |

---

## 四条路径的对比总表

| | 单机非 stencil | 单机 stencil | 多机非 stencil | 多机 stencil |
|---|---|---|---|---|
| **命令行** | `--mode=buffer` | `--mode=buffer` | `--mode=buffer --mpi` | `--mode=buffer --mpi` |
| **Region 条件** | 不满足 | 满足 | 不满足 | 满足 |
| **入口函数** | `rewriteDac_Buffer()` 标准分支 | `rewriteDac_Buffer()` region 分支 | `rewriteMPI()` | `rewriteMPI_Region()` |
| **执行模型** | 单次 kernel 提交 | init→循环submit→sync | scatter→kernel→gather | init→循环{submit→halo→sibling}→sync |
| **数据分发** | 无（单机全量） | 无（单机全量） | MPI_Scatterv | MPI_Scatterv（init 时一次） |
| **数据回收** | kernel 后自动 | sync 时 D2H | MPI_Gatherv | MPI_Gatherv（sync 时一次） |
| **Halo 交换** | 不需要 | 不需要 | 不需要 | 每次循环后执行 |
| **Sibling 支持** | 不支持 | 支持 | 不支持 | 支持（含 MPI 稀疏同步） |
| **Buffer 生命周期** | 每次 `<->` 创建销毁 | init 创建，sync 销毁 | 函数内创建销毁 | init 创建，sync 销毁 |
| **典型用例** | 矩阵乘法、向量加 | Jacobi、stencil | 大规模矩阵乘法 | 大规模 stencil、波动方程 |

---

## 关键数据结构与运行时

### MPIPlanner.h 提供的运行时支持

| 类别 | 类/函数 | 用途 |
|------|---------|------|
| **Item-space** | `ItemRange` | 表示连续的 item 范围 |
| | `AccessPattern` | 描述参数的访问模式（维度、划分、绑定） |
| | `PackMap` | 全局索引到局部索引的映射（含 segments 优化） |
| **Pack-map 构建** | `get_rank_item_range()` | 按 rank 分配 item 范围 |
| | `build_input_pack_map()` | READ 参数的打包映射 |
| | `build_output_pack_map()` | WRITE 参数的打包映射 |
| | `build_rw_pack_map()` | READ_WRITE 参数的打包映射 |
| | `build_item_slots()` | 构建 item 到 slot 的映射 |
| **写回** | `build_writeback_values()` | 从局部数据提取写回值 |
| | `apply_writeback_by_globals()` | 按全局索引写回到原始 tensor |
| **设备视图** | `View1D` / `View2D` | packed 数组上的 1D/2D 访问视图 |
| **Dense 视图** | `DenseVectorView` / `DenseMatrixView` | sibling helper 的 dense 数组视图 |
| **Packed 视图** | `PackedVectorView` / `PackedMatrixView` | sibling 设备侧 packed 视图（含 dirty） |
| **Halo** | `HaloRegion` | 单个邻居的 halo 通信信息 |
| | `ParamHalo` | 单个参数的所有 halo 区域 |
| | `computeParamHalo()` | 计算 halo 区域 |
| | `exchangeHalo()` | 执行 halo 通信 |

---

## 代码模块索引

### 单机路径

| 文件 | 功能 |
|------|------|
| [Rewriter_Buffer_new.cpp](rewriter/lib/Rewriter_Buffer_new.cpp) | 单机 buffer 翻译主逻辑 |
| [buffer_template_new.cpp](rewriter/lib/buffer_template_new.cpp) | Buffer 模板（kernel、accessor、数据搬迁） |

### 多机路径

| 文件 | 功能 |
|------|------|
| [Rewriter_MPI.cpp](rewriter/lib/Rewriter_MPI.cpp) | MPI 入口与路由 |
| [Rewriter_MPI_Common.h](rewriter/include/Rewriter_MPI_Common.h) | 公共类型和接口声明 |
| [Rewriter_MPI_Wrapper_Codegen.cpp](rewriter/lib/mpi/Rewriter_MPI_Wrapper_Codegen.cpp) | Wrapper 路径代码生成 |
| [Rewriter_MPI_Analysis.cpp](rewriter/lib/mpi/Rewriter_MPI_Analysis.cpp) | 访问模式分析和广播判断 |
| [Rewriter_MPI_Types.cpp](rewriter/lib/mpi/Rewriter_MPI_Types.cpp) | MPI 数据类型映射和视图推导 |
| [Rewriter_MPI_Pattern.cpp](rewriter/lib/mpi/Rewriter_MPI_Pattern.cpp) | AccessPattern 构造和 pack-map 生成 |
| [Rewriter_MPI_Region_Codegen.cpp](rewriter/lib/mpi/Rewriter_MPI_Region_Codegen.cpp) | Region 编排层 |
| [Rewriter_MPI_Region_Ctx.cpp](rewriter/lib/mpi/Rewriter_MPI_Region_Ctx.cpp) | ctx 结构体生成 |
| [Rewriter_MPI_Region_Init.cpp](rewriter/lib/mpi/Rewriter_MPI_Region_Init.cpp) | init 函数生成 |
| [Rewriter_MPI_Region_Submit.cpp](rewriter/lib/mpi/Rewriter_MPI_Region_Submit.cpp) | submit 函数生成 |
| [Rewriter_MPI_Region_Halo.cpp](rewriter/lib/mpi/Rewriter_MPI_Region_Halo.cpp) | halo 函数生成 |
| [Rewriter_MPI_Region_Sync.cpp](rewriter/lib/mpi/Rewriter_MPI_Region_Sync.cpp) | sync 函数生成 |
| [Rewriter_MPI_Region_Sibling.cpp](rewriter/lib/mpi/Rewriter_MPI_Region_Sibling.cpp) | sibling helper 生成 |
| [Rewriter_MPI_Region_Policy.cpp](rewriter/lib/mpi/Rewriter_MPI_Region_Policy.cpp) | 传输策略分析 |
| [MPIPlanner.h](dpcppLib/include/MPIPlanner.h) | MPI 运行时库（所有视图、pack-map、halo） |

### 分析与解析

| 文件 | 功能 |
|------|------|
| [DacppStructure.cpp](parser/lib/DacppStructure.cpp) | `analyzeBufferRegionPlan()` — stencil 检测 |
| [translator.cpp](translator.cpp) | 入口、路由、命令行参数 |
