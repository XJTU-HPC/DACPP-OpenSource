# 优化方案：消除 MPI 路径的 O(N³) Slots 扩展

## 1. 问题背景

矩阵乘法 N=2048 时，MPI 生成的代码即使注释掉计算，数据准备阶段也要 20+ 秒。

用户在神威上的 profiling 数据（256×256, 32 rank）：

```
总执行时间(所有rank最大值): 23804.6 ms
collect_positions_for_item 总调用次数(所有rank求和): 524288
collect_positions_for_item 总耗时(所有rank求和): 46467.2 ms
collect_positions_for_item 最大rank耗时: 16279.7 ms
```

调用次数分析：`6 × N² + 2 × N² = 8 × 65,536 = 524,288`，其中：
- 每个参数（A/B/C）各做 pack + slots = 6 × N²（所有 rank）
- Root 为每个 rank 重算 A 和 B 的 pack = 2 × N²

## 2. 问题定位

当前 rewriter 已使用 `build_pack_plan`（含 bind key 去重），但仍有两个瓶颈：

### 瓶颈 A：O(N³) Slots 扩展（最严重）

`build_pack_plan`（Pack.h:236-239）的最终 slots expansion：

```cpp
for (int32_t key_index : item_key_indices) {      // N² 次迭代
    const auto& slots_for_key = key_slots[key_index]; // 每次取 partition_size 个 slots
    plan.slots.insert(plan.slots.end(), ...);          // 追加到 slots
}
```

对 matMul 的 vecA：
- `total_items` = N² = 4M
- `partition_size` = N = 2048（每个 item 取一整行）
- slots 数组 = N² × N = **8.6 billion** int32_t ≈ 32 GB

虽然 positions 已按 bind key 去重（vecA 只有 N 个 unique key），但 slots expansion 把每个 unique slot range 复制了 N 份。

### 瓶颈 B：旧代码残留

`build_input_pack_map` + `build_item_slots`（Pack.h:115-154）仍然存在，完全没有去重。旧测试文件和其他 code path 可能仍使用它们。每个 item 都调用 `collect_positions_for_item`，且同一 item 的位置集合至少算两遍（pack 一遍，slots 一遍）。

## 3. 1D 问题是否受影响

分析各 1D 场景：

| 场景 | partition_size | unique keys | 旧 slots 规模 | 是否有 N² 问题 |
|---|---|---|---|---|
| vectorAdd (`v[i]`) | 1 | N | N × 1 = N | **否** |
| VSHIFT broadcast (`bias[{}]`, size=1) | 1 | 1 | N × 1 = N | **否** |
| VSHIFT broadcast (`bias[{}]`, size=N) | N | 1 | N × N = N² | **是** |
| jacobi `x[{}]`（broadcast 全向量） | N | 1 | N × N = N² | **是** |
| jacobi `A[{idx1}][{}]`（取一整行） | N | N | N × N = N² | **部分**（无去重空间） |
| stencil `matIn[sp1][sp2]` (3×3 窗口) | 9 | O(N²) | 9N² | **否**（常数 partition） |
| oddeven `array[{S1}]` (size=2 split) | 2 | N/2 | N | **否** |

**结论**：1D 问题中的 broadcast 模式（`param[{}]`）确实存在 N² 扩展。本方案统一处理所有维度。

## 4. 方案设计：Compact Slots + Item-to-Key Offset Table

### 核心思路

把 O(N³) 的 slots 拆成两个 O(N²)（或更小）的数组：

- `compact_slots`：只存 unique key 的 slot range，大小 = `num_unique_keys × partition_size`
- `item_key_offsets`：每个 item 到 compact_slots 的偏移，大小 = `num_items`

Kernel 从：
```cpp
View1D<T>{data, slots, item_linear * partition_size}
```
改为：
```cpp
View1D<T>{data, compact_slots, item_key_offsets[item_linear]}
```

View1D / View2D **不需要改**。

### 内存对比

**matMul vecA (N=2048, 单 rank)**：

| | 改前 | 改后 |
|---|---|---|
| slots | N² × N × 4B = 32 GB | — |
| compact_slots | — | N × N × 4B = 16 MB |
| item_key_offsets | — | N² × 4B = 16 MB |
| **合计** | **32 GB** | **32 MB** |

**jacobi `x[{}]` (N=2048, 单 rank)**：

| | 改前 | 改后 |
|---|---|---|
| slots | N × N × 4B = 16 MB | — |
| compact_slots | — | 1 × N × 4B = 8 KB |
| item_key_offsets | — | N × 4B = 8 KB |
| **合计** | **16 MB** | **16 KB** |

## 5. 具体改动

### 5.1 Pack.h — `PackPlan` 和 `build_pack_plan`

**文件**：`dpcppLib/include/mpi/Pack.h`

#### a) 替换 `PackPlan` 的 slots 字段（约 line 79）

```cpp
struct PackPlan {
    PackMap pack;
    // std::vector<int32_t> slots;             // 删除
    std::vector<int32_t> compact_slots;         // 新增：unique key slot ranges
    std::vector<int32_t> item_key_offsets;      // 新增：item → offset 映射
};
```

#### b) 修改 `build_pack_plan`（lines 186–242）

把最后 O(N³) 的 slots expansion（lines 192, 236-239）替换为 compact 版本：

```cpp
// 删除 line 192: plan.slots.reserve(item_count * elem_count);
// 删除 lines 236-239: 整个旧 slots expansion loop

// 新增：构建 compact_slots（每个 unique key 只存一份）
plan.compact_slots.reserve(unique_positions.size() * elem_count);
for (const auto& slots_for_key : key_slots) {
    plan.compact_slots.insert(plan.compact_slots.end(),
                               slots_for_key.begin(), slots_for_key.end());
}

// 新增：构建 item_key_offsets
plan.item_key_offsets.reserve(item_key_indices.size());
for (int32_t key_index : item_key_indices) {
    plan.item_key_offsets.push_back(key_index * static_cast<int32_t>(elem_count));
}
```

#### c) 删除旧函数

以下函数无去重，每次都遍历所有 item 调用 `collect_positions_for_item`：

- `build_input_pack_map`（lines 115-122）→ 已被 `build_pack_plan` 替代
- `build_output_pack_map`（lines 124-129）→ 已被 `build_pack_plan` 替代
- `build_rw_pack_map`（lines 131-136）→ 已被 `build_pack_plan` 替代
- `build_item_slots`（lines 140-154）→ 旧的无去重 slots 构建

### 5.2 Rewriter_MPI_Wrapper_Codegen.cpp — Kernel 代码生成

**文件**：`rewriter/lib/mpi/Rewriter_MPI_Wrapper_Codegen.cpp`

#### a) 提取 compact_slots 和 item_key_offsets（约 line 68）

```cpp
// 旧:
code += "    auto& " + slotsName + " = plan_" + calcName + ".slots;\n";

// 新:
code += "    auto& " + slotsName + " = plan_" + calcName + ".compact_slots;\n";
code += "    auto& key_offsets_" + calcName + " = plan_" + calcName + ".item_key_offsets;\n";
```

#### b) 创建 SYCL buffer（约 lines 143-146）

在现有 `slots_buffer` 之后，增加 `key_offsets_buffer`：

```cpp
code += "            sycl::buffer<int32_t, 1> key_offsets_buffer_" + name +
        "(key_offsets_" + name + ".data(), sycl::range<1>(key_offsets_" + name + ".size()));\n";
```

#### c) 添加 accessor（约 lines 149-155）

```cpp
code += "                auto key_offsets_acc_" + name +
        " = key_offsets_buffer_" + name + ".get_access<sycl::access::mode::read>(h);\n";
```

#### d) kernel 内取指针 + 修改 View 构造（约 lines 157-177）

在 `auto* slots_XXX = ...` 之后添加：
```cpp
code += "                    auto* key_offsets_" + name +
        " = key_offsets_acc_" + name +
        ".template get_multi_ptr<sycl::access::decorated::no>().get();\n";
```

修改 View 构造（核心改动，涉及 lines 170-176 的 1D 和 2D 两种情况）：

```cpp
// 旧:
"{data_" + name + ", slots_" + name + ", item_linear * " + name + "_partition_size}"

// 新:
"{data_" + name + ", slots_" + name + ", key_offsets_" + name + "[item_linear]}"
```

View1D/View2D 结构体不需要任何改动。

### 5.3 Rewriter_MPI_Pattern.cpp — 清理废弃代码

`buildPackBuilderExpr`（line 173）和 `buildRemotePackBuilderExpr`（line 200）如果引用了旧的 `build_input_pack_map` / `build_item_slots`，需要一并清理。

### 5.4 向后兼容性

当所有 item 的 bind key 都唯一时（无去重）：
- `compact_slots` 等于旧 `slots`
- `item_key_offsets[i] = i * partition_size`
- 行为完全一致，无需特殊处理

## 6. 预期 `collect_positions_for_item` 调用次数

以 256×256, 32 rank 为例：

| | 旧代码 | 新代码 |
|---|---|---|
| vecA (256 unique rows) | 2 × 2048/rank × 32 = 131K | 256/rank × 32 = 8K |
| vecB (256 unique cols) | 2 × 2048/rank × 32 = 131K | 256/rank × 32 = 8K |
| dotProduct (2048 unique) | 2 × 2048/rank × 32 = 131K | 2048/rank × 32 = 66K |
| Root 重算 | 2 × 65536 = 131K | 0（使用 gathered_globals） |
| **合计** | **524K** | **82K** |

## 7. 需要修改的文件

| 文件 | 改动 |
|---|---|
| `dpcppLib/include/mpi/Pack.h` | `PackPlan` 去掉 `slots`，加 `compact_slots` + `item_key_offsets`；`build_pack_plan` 用 compact 版本；删除旧函数 |
| `rewriter/lib/mpi/Rewriter_MPI_Wrapper_Codegen.cpp` | 生成 compact_slots / key_offsets buffer + accessor + 新 View 构造 |
| `rewriter/lib/mpi/Rewriter_MPI_Pattern.cpp` | 清理引用旧函数的废弃代码 |

## 8. 验证步骤

1. 编译 translator：
   ```bash
   cd /Volumes/QUQ/working/dacpp && cmake --build build --target translator -j8
   ```

2. 重新翻译 matMul，检查生成代码：
   ```bash
   dacpp tests/matMul1.0/matMul.dac.cpp --mode=buffer --mpi
   ```
   验证：使用 `compact_slots` 和 `item_key_offsets`，不再出现 `plan_XXX.slots`。

3. 本地回归：
   ```bash
   cd /Volumes/QUQ/working/dacpp/clang/tools/translator && bash test_local.sh
   ```

4. MPI 回归：
   ```bash
   bash test_mpi.sh
   ```

5. 神威性能验证（256×256, 32 rank）：
   - 预期 `collect_positions_for_item` 调用次数：从 524,288 降到 ~81,920
   - 预期该函数总耗时：从 46,467ms 降到 ~7,000ms
   - 预期 slots 内存：从 O(N³) 降到 O(N²)

6. 神威 N=2048 验证：
   - 数据准备阶段应 < 2 秒
