# DACPP Translator 当前 MPI 主链路说明

本文档描述 2026-04-13 仓库里 `--mpi` 的真实行为，以及当前 stencil 相关讨论应该建立在哪个事实上。

先记住四句话：

- `--mpi` 统一走 `rewriteMPI()`，不会再进入普通的 `rewriteDac_Buffer()` / `rewriteDac_Usm()` 分支。
- 当前单步 MPI wrapper 已经能正确覆盖现有 non-stencil MPI 回归，也能跑通当前的 `stencil1.0` 和 `waveEquation1.0`。
- 因此 stencil 的当前主题已经不是“支不支持”，而是“能不能自动识别并优化成 halo / 子域路径”。
- 当前 MPI 主链路的抽象仍然是 item-space 的 Root pack / scatter / gather，而不是 stencil-aware 的 subdomain runtime。

---

## 1. 入口与当前模式事实

当用户传入 `--mpi` 时，`translator.cpp` 会：

1. 打开 MPI 相关头文件和同步选项
2. 调用 `rewriter->rewriteMPI()`
3. 再统一调用 `rewriteMain()`

这意味着：

- `--mpi` 是一条独立的重写主链路
- 它不再受普通 buffer / usm 重写器分支控制

这里有一个容易混淆但必须说清的事实：

- 当前回归脚本 `test_mpi.sh` 是用 `--mode=usm --mpi` 触发 MPI
- 生成文件名也仍然沿用 `_sycl_<mode>.cpp`
- 但 `rewriteMPI()` 生成的本地执行段内部，今天仍然显式构造 `sycl::buffer` 和 accessor

所以当前应该把它理解成：

- “一个独立的 MPI item-space backend”
- 而不是“完全等价于普通 usm backend 的 MPI 版本”

这点在看代码时尤其重要，否则会把命令行 `--mode` 字面值和 MPI 路径的真实本地执行模型混为一谈。

---

## 2. 当前已经验证过的状态

### 2.1 non-stencil MPI 回归

2026-04-13 已执行：

```bash
cd /Volumes/QUQ/working/dacpp/clang/tools/translator
bash test_mpi.sh
```

结果：

- `10 tests | 10 passed | 0 failed | 0 skipped`

当前通过的 non-stencil MPI 样例包括：

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

### 2.2 当前 stencil 样例

2026-04-13 另外手工验证了：

- `stencil1.0`：baseline 与 MPI 结果一致，`mpirun -np 2`
- `waveEquation1.0`：baseline 与 MPI 结果一致，`mpirun -np 2`
- `waveEquation1.0`：baseline 与 MPI 结果一致，`mpirun -np 4`

这件事改变了 stencil 讨论的前提。

现在的问题已经不是：

- “当前 MPI 主链路完全无法处理 stencil”

而是：

- “当前 MPI 主链路已经能正确跑现有 stencil，但它用的是通用 item-space 分发，不够 stencil-aware，也不够高效”

---

## 3. 当前生成结果的结构

对每个 `<->` 表达式，`rewriteMPI()` 仍然生成两层代码。

### 3.1 本地计算函数

形如：

- `xxx_mpi_local(...)`

特点：

- 直接复用原 `calc` 函数体
- 不重写 calc 算法语义
- 通过 `dacpp::mpi::View1D<T>` / `View2D<T>` 适配本地压缩数据访问

### 3.2 单步 MPI wrapper

形如：

- `ShellName_CalcName(...)`

wrapper 负责：

1. 推导参数的有效访问模式
2. 基于 shell / binding 初始化 `AccessPattern`
3. 计算统一 item 空间与每个 rank 的连续 item range
4. Root 为各 rank 打包输入 payload
5. 用 `MPI_Scatter` / `MPI_Scatterv` 分发本次调用需要的局部数据
6. 各 rank 本地提交线性 `parallel_for`
7. 对 `WRITE / READ_WRITE` 参数做 gather / writeback
8. 如有需要，再把 root 上更新后的 tensor broadcast 回其他 rank

### 3.3 `main()` 的 MPI 生命周期

`rewriteMPI()` 还会向 `main()` 注入：

- 条件性的 `MPI_Init`
- `MPI_Comm_rank / MPI_Comm_size`
- 非 root rank 的 `stdout` 重定向
- return 前的 `MPI_Finalize`

---

## 4. 当前主链路的核心抽象

当前 MPI 路径切分的不是“空间子域”，而是：

- shell / binding 定义出来的 item 空间

一个 item 会经历下面这条链：

1. 由 binding split 组合定义 item 坐标
2. planner 把这个 item 展开成一组全局线性下标
3. Root 根据这些全局下标做 pack
4. rank 本地只拿到自己本次调用会访问到的压缩数组
5. 本地 kernel 用 `slots` 把逻辑窗口映射回压缩数组

这条链路最擅长的是：

- “每次 wrapper 调用都可以把所需输入一次性准备好”的问题

它没有单独建模的东西包括：

- 长期持有的空间子域
- 邻接 rank 间的 halo / ghost exchange
- 跨时间步持久化的分布式状态
- frontier push / neighbor-only exchange

换句话说，当前 stencil 能跑，并不是因为今天已经有了 halo runtime，而是因为：

- 当前 item-space wrapper 仍然可以在每次 `<->` 调用前，把该次 stencil 需要的窗口数据从 Root 打包下发

---

## 5. 当前为什么能覆盖 `stencil1.0` / `waveEquation1.0`

关键原因不是 planner 变成了 stencil planner，而是 MPI 路径现在对 calc 的真实读写语义理解足够准确。

### 5.1 `inferEffectiveParamModes(...)` 现在看真实 `calc` 访问

当前 `Rewriter_MPI.cpp` 里的 `inferEffectiveParamModes(...)` 会：

1. 读取 `calc` 的 AST body
2. 用 `ParamAccessVisitor` 统计每个参数：
   - `Reads`
   - `UpdateReads`
   - `Writes`
3. 根据真实访问重新计算本次 MPI wrapper 使用的有效模式

这里的 visitor 已经能正确处理：

- 普通赋值
- 复合赋值
- `++` / `--`
- `CXXOperatorCallExpr` 形式的重载赋值与重载自增自减

### 5.2 这对 stencil 样例的直接效果

`stencil1.0` 中：

- `matIn` 在 shell 上是 `READ_WRITE`，但 calc 实际只读，MPI 路径现在会收紧成 `READ`
- `matOut` 在 shell 上是 `READ_WRITE`，但 calc 实际只写，MPI 路径现在会收紧成 `WRITE`

`waveEquation1.0` 中：

- `matCur` 在 shell 上是 `WRITE`，但 calc 实际通过窗口读取，MPI 路径现在会纠正成 `READ`
- `matPrev` 在 shell 上是 `READ_WRITE`，但 calc 实际只读，MPI 路径现在会收紧成 `READ`
- `matNext` 在 shell 上是 `WRITE`，calc 也只写，因此保持 `WRITE`

这使得当前单步 wrapper 至少能够：

- 正确 scatter 真正需要 copy-in 的张量
- 避免对只读输入走错误的 writeback
- 避免把纯输出参数误当成 read-write 输入

---

## 6. 当前 stencil 话题已经变成“优化问题”

基于上面的验证和修复，当前 stencil 相关结论应该更新成：

- 现有 stencil 样例已经可以沿用当前单步 MPI 主链路得到正确结果
- 当前不足不在“能否运行”，而在“运行方式还是通用 item-space，不是 stencil-aware”

所以今天真正要讨论的问题是：

- translator 能不能自动识别某个 `<->` 是规则 stencil
- 一旦识别出来，能不能把它从当前 Root pack / scatter / gather 路径，降成 halo / 子域 / 邻居交换路径

这也是后续 stencil 设计的正确起点。

---

## 7. 自动识别 stencil 时，当前主链路应该扮演什么角色

当前主链路不应该被废掉，原因有三个：

1. 它已经是现成可工作的 correctness baseline
2. 对混合程序来说，不是每个 `<->` 都应该被强行降成 stencil runtime
3. 自动识别一旦不够确定，就应该安全回退到当前通用路径

因此更合理的架构是：

- 当前 `rewriteMPI()` 单步 wrapper 继续作为保底路径
- stencil-aware 路径只在 translator 确认“这是规则 stencil”时才接管

如果未来要支持混合问题，前端可以再补一层分类开关，例如：

- 全局关闭 stencil 优化
- 用户显式把某个 `<->` 标成 stencil
- translator 自动识别后再决定是否切换路径

但这属于后续优化设计，不是当前仓库里已经存在的功能。

---

## 8. 自动识别 stencil 的目标，不是再造一个更复杂的 item planner

当前 item-space planner 的强项是：

- 给定一个 item，展开它会访问到哪些全局元素

而 stencil-aware 降级真正需要补的是另一层语义：

1. owner core 是什么
2. halo 宽度是什么
3. 哪些邻居需要交换数据
4. 是否存在时间推进循环
5. 是否值得在循环内保留分布式状态，而不是每步都回 root

因此下一个阶段的核心不是：

- 再把当前 `PackMap` 做得更复杂一些

而是：

- 在 translator 层先回答“这是 stencil 吗”
- 再决定是否切换到 block ownership + halo exchange 的另一条生成路径

---

## 9. 下一步设计时最值得继续看的文件

- `/Volumes/QUQ/working/dacpp/clang/tools/translator/translator.cpp`
  - `--mpi` 入口
  - 输出文件命名

- `/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/Rewriter_MPI.cpp`
  - 当前单步 wrapper 生成
  - `inferEffectiveParamModes(...)`
  - `buildWrapperCode(...)`

- `/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/MPIPlanner.h`
  - 当前 item-space pack / slot / writeback 运行时

- `/Volumes/QUQ/working/dacpp/clang/tools/translator/parser/lib/Shell.cpp`
  - binding 图与 offset 信息

- `/Volumes/QUQ/working/dacpp/clang/tools/translator/docs/stencil_handoff_readme.md`
  - 后续 stencil 自动识别与优化讨论的 handoff 文档

- `/Volumes/QUQ/working/dacpp/clang/tools/translator/docs/bugs_fix_log/mpi_param_mode_inference_fix.md`
  - 本次让单步 MPI 能正确覆盖当前 stencil 样例的关键修复说明

---

## 10. 当前结论

截至 2026-04-13，仓库中的 `--mpi` 应该这样理解：

- 它已经是一条可工作的单步 MPI 翻译链路
- 它已经覆盖现有 non-stencil MPI 回归
- 它也已经能正确执行当前 stencil 样例
- 但它仍然是 item-space 的通用分发路径
- 因此 stencil 的下一阶段问题已经收敛为：自动识别 + halo / 子域优化，而不是基本支持
