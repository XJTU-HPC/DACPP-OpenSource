# MPI 翻译器测试报告

**日期**: 2026-04-12  
**测试环境**: macOS Tahoe | AdaptiveCpp (CPU fallback) | Homebrew OpenMPI | `mpirun -np 2`  
**测试命令**:

```bash
cd /Volumes/QUQ/working/dacpp/clang/tools/translator
bash test_mpi.sh
```

**脚本说明**:
- `test_mpi.sh` 的使用方式与 [macos_local_sycl_setup.md](/Volumes/QUQ/working/dacpp/clang/tools/translator/docs/macos_local_sycl_setup.md) 一致
- 脚本会先用 `--mode=buffer` 生成单机 baseline，再用 `--mode=usm --mpi` 生成 MPI 版本
- 中间日志与产物位于 `/tmp/dacpp_mpi_tests`

**本轮最终结果**: **10 tests | 10 passed | 0 failed | 0 skipped**

**说明**:
- 当前所有 non-stencil MPI 测试已经通过
- `MDP1.0` 最终确认不是精度问题，而是语义问题
- 最终修复由三部分组成：
  1. `binding()` 连通分量映射
  2. `READ_WRITE` 访问模式细化
  3. 保留主函数中的原始 `for` 循环结构

---

## 一、结果总览

| # | 测试名 | 结果 | 失败阶段 | 当前判断 |
|---|--------|------|----------|----------|
| 1 | matMul1.0 | PASS | - | 输出保持 `WRITE` 语义后稳定通过 |
| 2 | FOuLa1.0 | PASS | - | `binding()` 连通分量映射修复后恢复通过 |
| 3 | decay1.0 | PASS | - | 正常 |
| 4 | DFT1.0 | PASS | - | 正常 |
| 5 | liuliang1.0 | PASS | - | 正常 |
| 6 | MDP1.0 | PASS | - | `binding()` + access-mode + 循环保留三层修复后恢复通过 |
| 7 | mandel1.0 | PASS | - | 正常 |
| 8 | imageAdjustment1.0 | PASS | - | buffer 基线索引生成已修复 |
| 9 | vectorAddCombo | PASS | - | buffer 基线索引生成已修复 |
| 10 | gradientSum | PASS | - | 正常 |

---

## 二、通过的测试

通过的 9 个样例：

- `matMul1.0`
- `FOuLa1.0`
- `decay1.0`
- `DFT1.0`
- `liuliang1.0`
- `mandel1.0`
- `imageAdjustment1.0`
- `vectorAddCombo`
- `gradientSum`

从当前结果看，这些样例至少满足下面之一：

- 不依赖 `binding(idx, split)` 去把输入窗口和输出索引绑成同一组
- 或者虽然存在中间张量级联，但当前 buffer baseline 与 MPI 行为已经一致
- 或者即使有后续使用，也是在当前实现的广播判定范围内

---

## 三、失败样例与直接证据

### 3.1 MDP1.0

**现象**:

- baseline:
  `3.72665e-06`
- MPI:
  `9.9295e-06`

**源代码里的绑定关系**:

```cpp
binding(idx, sp);
```

**第一阶段发现的错误生成结果**:

```cpp
pattern_p.bind_set_id.push_back(0);
pattern_new_p.bind_set_id.push_back(1);
```

**最终结论**:

- 第一层问题和 `FOuLa1.0` 相同，都是 `binding()` 语义丢失
- 输入三点窗口 `p[sp]` 与输出单点 `new_p[idx]` 本应共享同一绑定集合
- 修完这层之后，`MDP1.0` 仍未完全恢复，说明它还叠加了第二层问题：MPI 路径把 shell 上的 `READ_WRITE` 直接当成真实访问模式
- 再继续排查发现，主函数重写还错误删除了外层时间推进 `for` 循环
- 因此它最终是三层语义问题叠加，不是精度问题

---

## 四、根因归类

失败根因最终收敛为三层：

### 4.1 第一阶段：`binding()` 语义没有进入 MPI planner

受影响样例：

- `MDP1.0`
- `FOuLa1.0`

具体表现：

- 源码里显式写了 `binding(i, s)` / `binding(idx, sp)`
- 但生成的 `bind_set_id` 仍然把输入 split 和输出 index 分成不同编号
- 导致 planner 把原本应共享的一维 item 空间扩成错误的笛卡尔积

### 4.2 第二阶段：shell 声明的 `READ_WRITE` 不等于 calc 的真实访问模式

具体表现：

- shell 把 `p` / `new_p` 都声明成 `READ_WRITE`
- 但 calc 实际是：
  - `p` 只读
  - `new_p` 只写
- 旧的 MPI 代码生成直接沿用 shell 声明，导致 pack / scatter / writeback 路径都按 `ReadWrite` 处理

这类问题更像语义错误，而不是浮点精度误差。

### 4.3 第三阶段：主函数重写错误删除了外层循环

当前重点样例：

- `MDP1.0`

具体表现：

- `<->` 位于 `for (int t = 0; t < T; ++t)` 内部
- 旧的 `rewriteMain()` 会直接删除整个外层 `for`
- 最终只剩一次 wrapper 调用，导致生成程序和原程序不再等价

---

## 五、为什么“吃掉整个 for”没有在所有测试里立刻暴露

这个 bug 很严重，但之所以没有在更早阶段把大量样例打红，主要有三类原因：

### 5.1 一部分样例根本不在 `for` 里

这类样例不会走旧的 `forStatement` 特殊路径，因此完全不受影响。

例如：

- `matMul1.0`
- `DFT1.0`
- `vectorAddCombo`
- `imageAdjustment1.0`
- `gradientSum`

### 5.2 一部分样例在循环里，但不是 `for`

旧逻辑只对 `ForStmt` 生效，不会吃掉 `while`。

例如：

- `decay1.0`

它的时间推进是 `while (t_tensor[0] <= T)`，因此不受这个特定 bug 影响。

### 5.3 还有一部分样例虽然受影响，但当前观测值不敏感

最典型的是：

- `liuliang1.0`

它原始确实在 `for (int t = 0; t < TIME_STEPS; ++t)` 中调用 `<->`，但最后打印的是 `rho[15]`。这个位置落在初始化的常值平台区，当前公式下单步和多步都可能保持同样输出，因此：

- 程序语义其实已经被改坏了
- 但当前测试的输出点没有把错误显性放大出来

### 5.4 这说明什么

所以这个 bug 当时没有大面积爆雷，并不是因为它轻微，而是因为：

- 有些样例没走到这条错误路径
- 有些样例的观测方式对它不敏感

`MDP1.0` 最终把它暴露出来，是因为它的输出对时间推进次数高度敏感，只跑一次 wrapper 和跑 1000 次迭代的结果会立刻分离。

---

## 六、与旧结论相比需要修正的点

旧版本报告里有四处判断需要修正：

- `FOuLa1.0` 和 `MDP1.0` 不是“多阶段 `<->` 流水线”问题，它们各自只有一个 `<->`
- `FOuLa1.0` 已经随着 `binding()` 连通分量修复恢复通过
- `vectorAddCombo` 和 `imageAdjustment1.0` 的失败来自 buffer baseline 生成错误，而不是 MPI 广播语义本身；修复 buffer 代码生成后，这两个样例已经通过
- `MDP1.0` 当前剩余问题不像精度问题，而更像访问模式语义没有正确下沉到 MPI planner
- `MDP1.0` 最后一层卡点不是 planner 数值本身，而是主函数循环结构在重写时被删掉了

---

## 七、修复建议

### 7.1 已完成：修 `binding()` 映射

已完成的修复要点：

- `binding(i, s)` 这类关系已经从 parser 传到 `Rewriter_MPI.cpp`
- `collectSplitBindMeta(...)` 不再只按 split 名字分配新编号
- 被绑定的 index/split 现在会共享同一个 `bind_set_id`

这一步修完后：

- `FOuLa1.0` 恢复通过
- `MDP1.0` 从“完全错误绑定”收敛为“剩余访问模式问题”

### 7.2 已完成：细化 access-mode

修复原则：

- 不再无条件使用 `shellParam->getRw()`
- 基于 calc AST 推断参数真实是 `READ` / `WRITE` / `READ_WRITE`
- 但只允许收窄 shell 原本声明为 `READ_WRITE` 的参数，避免误伤 `matMul1.0` 这类输出缓冲

这一步修完后：

- `MDP1.0` 的 `p` 会按 `READ`
- `MDP1.0` 的 `new_p` 会按 `WRITE`
- `matMul1.0` 重新稳定通过

### 7.3 已完成：保留主函数中的原始 `for` 循环

修复内容：

- `rewriteMain()` 不再删除包含 `<->` 的整个外层 `for`
- 现在只在原位置把 `<->` 替换成 wrapper 调用

这一步修完后：

- `MDP1.0` 的 1000 次时间推进循环被正确保留
- `MDP1.0` 恢复通过

### 7.4 已完成的 buffer 修复

本轮已修复两处 buffer 代码生成问题：

- `Rewriter_Buffer_new.cpp` 中 `clacparam` 元信息不再跨多个 `<->` 累积污染
- `Rewriter_Buffer_new.cpp` 重新使用 `BUFFER_TEMPLATE::CodeGen_DAC2SYCL2(...)`，修复了 buffer 包装器里 `q` / `dacpp_q` 不一致的问题

这两处修复后：

- `vectorAddCombo` 恢复通过
- `imageAdjustment1.0` 恢复通过

---

## 八、本轮测试结论

当前文档记录的最终完整回归状态是：

- 10 个 non-stencil 样例中 10 个通过
- 0 个失败
- `MDP1.0` 已确认通过三层语义修复解决
