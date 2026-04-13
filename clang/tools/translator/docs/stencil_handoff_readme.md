# Stencil MPI Handoff README

本文档写给下一个接手这个目录的模型或开发者，用来快速建立上下文，避免重复排查已经修完的 non-stencil 问题，并把注意力集中到当前真正剩下的 stencil MPI 问题上。

---

## 1. 这个项目在做什么

目录：

- `/Volumes/QUQ/working/dacpp/clang/tools/translator`

这是一个 DACPP 到 SYCL 的 source-to-source translator。核心语义是：

- 用户写 `shell(...) <-> calc` 这样的 DACPP 表达式
- translator 基于 Clang AST 解析：
  - shell 参数
  - split / index / binding 关系
  - calc 参数与函数体
  - `<->` 所在的控制流位置
- 然后把 DACPP 源文件改写成可编译的 SYCL C++ 代码

当前支持的几个主要路径：

- `--mode=buffer`
  - 生成单机 buffer 版 SYCL
- `--mode=usm`
  - 生成单机 USM 版 SYCL
- `--mode=usm --mpi`
  - 生成 MPI + SYCL 版本
  - 现在已经确认 non-stencil 路径 10/10 通过

要点是：这个 `--mpi` 目前不是“所有模式都完全成熟”，它的 non-stencil 路径已经修通，但 stencil 还没有真正收敛。

---

## 2. 从哪里开始看

如果要从零开始建立全局理解，建议优先读这些文件：

### 2.1 入口与整体流程

- `/Volumes/QUQ/working/dacpp/clang/tools/translator/translator.cpp`
  - AST matcher 入口
  - 识别 `<->`
  - 识别 shell / calc
  - 记录最外层 `for`
  - 根据命令行模式选择不同 rewriter

### 2.2 parser 层

- `/Volumes/QUQ/working/dacpp/clang/tools/translator/parser/lib/Shell.cpp`
  - `binding()` 图信息在这里被解析
  - 重点接口：`Shell::GetBindInfo(...)`
- `/Volumes/QUQ/working/dacpp/clang/tools/translator/parser/include/Shell.h`
- `/Volumes/QUQ/working/dacpp/clang/tools/translator/parser/lib/DacppStructure.cpp`
  - 记录 `<->` 外层 `for` 以及相关变量

### 2.3 rewriter 层

- `/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/Rewriter_MPI.cpp`
  - 当前 MPI 重写主逻辑
  - 最近 non-stencil 修复主要都在这里
- `/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/include/Rewriter.h`
  - `rewriteMain()` 在这里
  - 之前这里会把整个外层 `for` 删掉，现在已经修成“只替换 `<->`，不删循环”
- `/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/Rewriter_Buffer_new.cpp`
- `/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/buffer_template_new.cpp`
  - 这两处是之前 buffer baseline 出错时修过的地方

### 2.4 MPI planner 运行时

- `/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/MPIPlanner.h`
  - `AccessPattern`
  - `bind_set_id`
  - `bind_offset_expr`
  - `build_input_pack_map(...)`
  - `build_output_pack_map(...)`
  - `build_rw_pack_map(...)`
  - `build_item_slots(...)`
  - 这是 stencil 后续分析的高优先级区域

---

## 3. 当前已经修好的东西

截至 2026-04-12，non-stencil MPI 已经全部通过：

```bash
cd /Volumes/QUQ/working/dacpp/clang/tools/translator
bash test_mpi.sh
```

结果：

- `10 tests | 10 passed | 0 failed | 0 skipped`

这 10 个测试是：

- `matMul1.0`
- `FOuLa1.0`
- `decay1.0`
- `DFT1.0`
- `liuliang1.0`
- `MDP1.0`
- `mandel1.0`
- `imageAdjustment1.0`
- `vectorAddCombo`
- `gradientSum`

### 3.1 最近修复链条

这轮已经确认并修完的，是下面四类问题。

#### A. buffer baseline 生成错误

影响样例：

- `vectorAddCombo`
- `imageAdjustment1.0`

问题本质：

- `clacparam` 元信息跨多个 `<->` 表达式累积污染
- buffer 路径错误走了不匹配的模板逻辑，导致生成代码里的 `q` / `dacpp_q` 使用不一致

修复位置：

- `/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/Rewriter_Buffer_new.cpp`
- `/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/buffer_template_new.cpp`

#### B. `binding()` 语义没有进入 MPI planner

影响样例：

- `FOuLa1.0`
- `MDP1.0`

问题本质：

- parser 已经能通过 `Shell::GetBindInfo(...)` 算出 binding 连通关系
- 旧版 `Rewriter_MPI.cpp` 的 `collectSplitBindMeta(...)` 没有使用这份图信息
- 结果是被 `binding()` 绑定在一起的 index / split 被错误分成了不同 `bind_set_id`

修复位置：

- `/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/Rewriter_MPI.cpp`

#### C. shell 声明的 `READ_WRITE` 被错误当成真实访问模式

影响样例：

- `MDP1.0`

问题本质：

- shell 上写成 `READ_WRITE` 只是外层接口声明
- calc 函数里参数真实可能是只读或只写
- 旧 MPI 逻辑直接沿用 shell 模式，导致 pack / scatter / writeback 路径全错

现在的修法：

- 在 `Rewriter_MPI.cpp` 里对 calc AST 做访问模式推断
- 但只允许把 shell 原本声明为 `READ_WRITE` 的参数收窄成 `READ` / `WRITE` / `READ_WRITE`
- 不去覆盖 shell 原本就声明为 `READ` 或 `WRITE` 的参数

这是为了避免重新打坏 `matMul1.0`

#### D. `rewriteMain()` 把整个外层 `for` 吃掉

影响样例：

- `MDP1.0`

问题本质：

- 旧逻辑在 `<->` 位于 `for` 内部时，会删掉整个 `for`
- 最后只保留一条 wrapper 调用
- 对 time-stepping 程序来说，这会直接改坏语义

修复位置：

- `/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/include/Rewriter.h`

现在行为：

- 保留原循环
- 只把 `<->` 替换为 wrapper 调用

---

## 4. 为什么“把整个 for 吃掉”对有些测试没有影响

这个结论已经同步进下面两份文档里：

- `/Volumes/QUQ/working/dacpp/clang/tools/translator/docs/mpi_binding_fix_analysis.md`
- `/Volumes/QUQ/working/dacpp/clang/tools/translator/docs/mpi_test_report.md`

这里再压缩复述一次，方便后续接手时不重复走弯路。

### 4.1 有些样例根本不在 `for` 里

例如：

- `matMul1.0`
- `DFT1.0`
- `vectorAddCombo`
- `imageAdjustment1.0`
- `gradientSum`

这些样例不会触发“删整个 `for`”的旧分支。

### 4.2 有些样例在循环里，但不是 `for`

例如：

- `decay1.0`

它用的是 `while`，而旧 bug 只咬 `ForStmt`。

### 4.3 有些样例虽然语义已经坏了，但当前观测点不敏感

典型样例：

- `liuliang1.0`

它最后打印的是平台区里的 `rho[15]`。对当前公式来说，单步和多步在这个观测点上都可能一样，所以：

- 程序实际已经不等价
- 但测试没有把错误放大出来

这点很重要，因为它说明：

- “之前没炸”不代表实现没问题
- 只能说明测试覆盖的观测点刚好没有暴露这个 bug

---

## 5. 现在真正剩下的问题：stencil MPI

### 5.1 当前状态

non-stencil 已经全绿，但 stencil 还没有修完。

已经做过一次 stencil smoke test，结论是：

- baseline 可以编译运行
- MPI 版本也可以编译运行
- 但两者输出不一致

现有 probe 产物在：

- `/tmp/dacpp_stencil_probe/stencil.mpi.dac_sycl_usm.cpp`

这个 probe 已经说明两件事：

1. `MPIPlanner.h` 已经被正确 include 进 stencil 生成结果
2. `rewriteMain()` 修复后，外层时间推进 `for` 已经保留下来

所以 stencil 当前失败，不再是前面已经修掉的那三类 non-stencil 语义问题的简单复现。

### 5.2 已知现象

`stencil1.0` 的一组观测结果是：

- baseline 第一行前几个值：
  - `0.0854598 0.0854598 0.0854628 0.0854658 ...`
- MPI 第一行前几个值：
  - `0.0238285 0.0238285 0.0457072 0.070995 ...`

这不是末位抖动，而是明显的语义偏离。

### 5.3 为什么它不像之前的 MDP 问题

`stencil1.0` 当前生成代码里已经能看到：

- 输入 `matIn` 被推断成 `Read`
- 输出 `matOut` 被推断成 `Write`
- 2 个维度上的 binding：
  - `sp1 <-> idx1`
  - `sp2 <-> idx2`
- 外层时间推进循环已保留

因此接下来不要先入为主地把 stencil 问题继续归因到：

- binding 丢失
- access-mode 没推断
- `rewriteMain()` 吃循环

这些问题对 stencil 分析仍然值得验证，但它们已经不是当前最强嫌疑。

---

## 6. stencil 相关样例，先看谁

建议优先看下面三个：

- `/Volumes/QUQ/working/dacpp/clang/tools/translator/tests/stencil1.0/stencil.dac.cpp`
- `/Volumes/QUQ/working/dacpp/clang/tools/translator/tests/waveEquation1.0/waveEquation.dac.cpp`
- `/Volumes/QUQ/working/dacpp/clang/tools/translator/tests/jacobi1.0/jacobi.dac.cpp`

它们分别代表三种很有价值的形态。

### 6.1 `stencil1.0`

特点：

- 2D 3x3 输入窗口
- 2D 单点输出
- 每一步之后，用户代码把 `matOut` 回写到 `matIn` 内部区域
- 再手动处理边界
- 有时间推进循环

这最接近“典型 stencil”

### 6.2 `waveEquation1.0`

特点：

- 也是 2D stencil
- 但同时涉及：
  - `matCur`
  - `matPrev`
  - `matNext`
- 时间推进里存在多块状态张量的轮换

如果 `stencil1.0` 的问题和状态同步、子视图写回、时间步之间的数据一致性有关，`waveEquation1.0` 很可能也会暴露类似结构性问题。

### 6.3 `jacobi1.0`

特点：

- 名字上像迭代算法
- 但它不是典型的局部邻域 stencil
- 更像“每个输出点读取整行/整向量信息”

它能帮助区分：

- “所有迭代类问题都坏”
- 还是“只有带局部重叠窗口的 stencil 坏”

---

## 7. stencil 下一步应该重点怀疑什么

这里给的是“下一步分析方向”，不是已经确认的结论。

### 7.1 首先怀疑 MPIPlanner 对重叠窗口的建模

高优先级函数：

- `build_input_pack_map(...)`
- `build_output_pack_map(...)`
- `build_item_slots(...)`
- `apply_writeback_by_globals(...)`

为什么先怀疑这里：

- stencil 的核心不是普通 scatter/gather
- 而是“多个 item 读取重叠输入窗口，但输出只写各自中心点”
- non-stencil 全绿只能证明 planner 对非重叠或弱重叠模式大体可用
- 不能证明它对 2D 重叠窗口也正确

建议先检查：

1. 2D item 的线性化顺序是否和 `bind_set_id` / `partition_shape` 一致
2. 同一个输入元素被多个 item 共享时，`pack.globals` 和 `slots` 是否正确去重并回映射
3. 输出是子 tensor 时，writeback 的 global 索引是否仍和用户视角一致

### 7.2 其次怀疑子视图 tensor 的 gather / broadcast / 回写语义

`stencil1.0` 里：

- 输入是完整 `matIn`
- 输出不是完整矩阵，而是内部子视图 `matOut = u_next_tensor[{1,NX-1}][{1,NY-1}]`

这意味着 MPI wrapper 内部对 `matOut` 做的：

- `tensor2Array`
- gather
- `array2Tensor`
- broadcast 后再 `array2Tensor`

必须对“子视图而不是完整底层 tensor”的语义完全正确。

如果这里 global index 的参照系错了，就会出现：

- 每个点都被写到了“某个地方”
- 但不是用户认为的那个内部区域

### 7.3 再怀疑 2D split / 1D 指针输出的混合建模

`stencil1.0` 的 calc 形参是：

- `dacpp::Matrix<double>& mat`
- `double* out`

生成代码里会变成：

- `View2D<const double> mat`
- `View1D<double> out`

这本身未必错，但要继续确认：

- 2D 输入窗口对应的 slot 布局是否和 `mat[a][b]` 的访存次序一致
- 单点输出 `out[0]` 在 2D item 空间里是否始终对应唯一正确的 global output element

### 7.4 最后再看时间步之间的 rank 间状态一致性

当前 wrapper 在 root gather 之后，会把输出广播给所有 rank。

理论上这能保证每个 rank 在下一轮循环开始前看到一致的 `matOut`。

但还需要进一步确认：

- 用户在 wrapper 外部把 `matOut` 回写到 `matIn` 时，各 rank 是否真的都在操作同样的 tensor 内容
- 边界更新后的 `matIn` 是否在下一轮 wrapper 前仍保持全 rank 一致

这条不是首嫌疑，因为如果所有 rank 都执行同一份 host 代码且输入已经同步，它本来应该一致；但仍值得在 stencil 深挖时做一次显式验证。

---

## 8. 一个很实用的分析顺序

如果下一位要继续接手，我建议按这个顺序推进：

1. 先不要再回头修 non-stencil
2. 先固定 `stencil1.0` 做最小复现
3. 对比：
   - 原始 `stencil.dac.cpp`
   - 生成的 buffer baseline
   - 生成的 MPI 代码 `/tmp/dacpp_stencil_probe/stencil.mpi.dac_sycl_usm.cpp`
4. 重点看：
   - `pattern_mat`
   - `pattern_out`
   - `pack_mat`
   - `slots_mat`
   - `pack_out`
   - writeback 和 broadcast 的索引参照系
5. 如果 `stencil1.0` 根因收敛，再把结论套到：
   - `waveEquation1.0`
   - `jacobi1.0`

不要一开始就同时改多个方向，否则很容易把“stencil 专属问题”和“已经修好的旧问题”重新混在一起。

---

## 9. 当前可直接复现的命令

### 9.1 跑 non-stencil MPI 回归

```bash
cd /Volumes/QUQ/working/dacpp/clang/tools/translator
bash test_mpi.sh
```

### 9.2 单独翻译一个 stencil 样例

```bash
bash -lc 'source /Volumes/QUQ/working/dacpp/clang/tools/translator/env.sh && dacpp /Volumes/QUQ/working/dacpp/clang/tools/translator/tests/stencil1.0/stencil.dac.cpp --mode=usm --mpi'
```

### 9.3 编译 stencil 生成结果

```bash
bash -lc 'source /Volumes/QUQ/working/dacpp/clang/tools/translator/env.sh && acpp-compile /Volumes/QUQ/working/dacpp/clang/tools/translator/tests/stencil1.0/stencil.dac_sycl_usm.cpp /tmp/stencil_mpi_bin'
```

### 9.4 运行 stencil MPI

```bash
DYLD_LIBRARY_PATH=/Volumes/QUQ/working/sycl-install/lib mpirun -np 2 /tmp/stencil_mpi_bin
```

---

## 10. 相关文档

这几份文档已经记录了最近的 non-stencil 修复结论，接手 stencil 前最好先读：

- `/Volumes/QUQ/working/dacpp/clang/tools/translator/docs/buffer_template_bugfix.md`
- `/Volumes/QUQ/working/dacpp/clang/tools/translator/docs/mpi_binding_fix_analysis.md`
- `/Volumes/QUQ/working/dacpp/clang/tools/translator/docs/mpi_semantic_broadcast_analysis.md`
- `/Volumes/QUQ/working/dacpp/clang/tools/translator/docs/mpi_test_report.md`
- `/Volumes/QUQ/working/dacpp/clang/tools/translator/docs/macos_local_sycl_setup.md`

读这些文档的目的不是重复修那几处，而是明确边界：

- 哪些问题已经确认并解决
- 哪些判断已经被证伪
- stencil 现在到底是从什么状态出发继续分析

---

## 11. 一句话结论

当前项目状态可以概括为：

- DACPP translator 的 non-stencil MPI 已经修通
- `binding()` / access-mode / `rewriteMain()` / buffer baseline 这几类旧问题已经有明确结论
- 下一个真正要解决的问题，是 stencil 在 MPIPlanner 和子视图写回语义上的正确性

