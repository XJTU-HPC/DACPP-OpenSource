# Stencil MPI Handoff README

本文档给下一位接手这个目录的开发者或模型使用，用来快速建立当前 stencil MPI 问题的真实上下文、代码入口和设计边界。

当前可以先记住一句话：

- stencil 样例今天会走和 non-stencil 一样的 `rewriteMPI()` 主链路
- non-stencil 已经完成回归验证
- stencil 需要的 halo / 子域语义还没有形成独立范式

---

## 1. 项目概况

目录：

- `/Volumes/QUQ/working/dacpp/clang/tools/translator`

这是一个 DACPP 到 SYCL 的 source-to-source translator。基本流程是：

- 用户写 `shell(...) <-> calc`
- translator 用 Clang AST 解析：
  - shell 参数
  - split / index / binding 关系
  - calc 参数与函数体
  - `<->` 所在的控制流位置
- 然后把 DACPP 源文件改写成可编译的 SYCL C++

常见入口有：

- `--mode=buffer`
- `--mode=usm`
- `--mode=usm --mpi`

其中 `--mpi` 当前统一走：

- `rewriteMPI()`

---

## 2. 建立全局理解时优先看的文件

### 2.1 入口与流程

- `/Volumes/QUQ/working/dacpp/clang/tools/translator/translator.cpp`
  - AST matcher 入口
  - `<->`、shell、calc 的识别
  - 最外层 `for` 的记录
  - `--mpi` 下的重写入口

### 2.2 parser 层

- `/Volumes/QUQ/working/dacpp/clang/tools/translator/parser/lib/Shell.cpp`
  - `binding()` 图的解析
  - `Shell::GetBindInfo(...)`
- `/Volumes/QUQ/working/dacpp/clang/tools/translator/parser/include/Shell.h`
- `/Volumes/QUQ/working/dacpp/clang/tools/translator/parser/lib/DacppStructure.cpp`
  - 外层 `for` 和相关变量信息

### 2.3 rewriter 层

- `/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/Rewriter_MPI.cpp`
  - 当前 MPI 主重写逻辑
  - 重点看：
    - `collectSplitBindMeta(...)`
    - `buildPatternInitCode(...)`
    - `buildWrapperCode(...)`
    - `inferEffectiveParamModes(...)`
- `/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/include/Rewriter.h`
  - `rewriteMain()`
  - `<->` 替换与宿主控制流保留逻辑

### 2.4 MPI planner 运行时

- `/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/MPIPlanner.h`
  - `AccessPattern`
  - `eval_bind_offset_expr(...)`
  - `collect_positions_for_item(...)`
  - `build_input_pack_map(...)`
  - `build_output_pack_map(...)`
  - `build_rw_pack_map(...)`
  - `build_item_slots(...)`
  - `build_writeback_values(...)`

---

## 3. 当前已经验证过的状态

我在 2026-04-13 重新执行了：

```bash
cd /Volumes/QUQ/working/dacpp/clang/tools/translator
bash test_mpi.sh
```

结果：

- `10 tests | 10 passed | 0 failed | 0 skipped`

当前通过回归的 non-stencil MPI 样例有：

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

因此，今天这条 `rewriteMPI()` 主链路对 non-stencil item 级分发场景是工作的。

---

## 4. 当前 stencil MPI 的问题定义

stencil 样例今天也会走同一条 MPI 主链路。它的抽象是：

1. 用 shell / binding 定义 item 空间
2. 把 `total_items` 切分给各 rank
3. Root 按全局线性下标集合为各 rank 打包输入
4. rank 本地用压缩数组、slot 映射和 `View1D / View2D` 执行 calc
5. Root gather 并写回输出

这条链路当前擅长的是：

- item-space planner
- Root pack / scatter / gather
- 本地压缩视图执行

stencil 需要的核心语义是：

- 子域所有权
- halo 宽度
- 邻接 rank 间的数据交换
- 时间步之间的 ghost region / 状态同步

这部分今天还没有单独建模。

因此，当前 stencil MPI 的问题可以直接表述为：

- 现有 MPI 主链路尚未提供 stencil 所需的 domain decomposition 与 halo exchange 语义

---

## 5. 需要直接记住的实现边界

### 5.1 item 空间是当前分发单位

当前分发单位是：

- item 空间中的 item

它对应的是 shell / binding 定义出来的工作单元，当前主链路围绕这个抽象做 Root 打包、rank 本地执行和 Root 写回。

### 5.2 offset 求值能力是常数级别

`bind_offset_expr` 当前支持的是：

- 纯整数
- 只含 `+` / `-` 的简单常数表达式

带变量或复杂符号的 offset 表达式不会被完整求值。

### 5.3 MPI 模式推断依赖 shell 标注

当前 `inferEffectiveParamModes(...)` 的规则是：

- 先以 shell 标注为基线
- 只收窄原本标成 `READ_WRITE` 的参数

因此 shell 上标成 `READ` 或 `WRITE` 的参数，会保持原标注。

### 5.4 输出同步由 host 侧后续使用分析主导

当前 broadcast 判定依赖：

- `TensorUseVisitor`

在通常存在 `main()` 的程序里，最终 `needsBcast` 由 host 侧后续使用分析主导。

### 5.5 本地 view 层当前覆盖到 1D / 2D

当前 `calc` 本地执行层正式提供的是：

- `View1D`
- `View2D`

---

## 6. 宿主控制流语义

`rewriteMain()` 当前负责：

- 把 `<->` 表达式替换成 wrapper 调用
- 保留宿主侧控制流结构

在正常链路下，如果 `<->` 位于最外层 `for` 中，生成结果会保留该宿主循环。

如果文件定义了：

- `DACPP_TRANSLATE_MODE 1`

则 `rewriteMain()` 会走简单替换分支。

这点在分析样例时要单独留意。

---

## 7. stencil 样例的代表性

建议优先看下面三个样例：

- `/Volumes/QUQ/working/dacpp/clang/tools/translator/tests/stencil1.0/stencil.dac.cpp`
- `/Volumes/QUQ/working/dacpp/clang/tools/translator/tests/waveEquation1.0/waveEquation.dac.cpp`
- `/Volumes/QUQ/working/dacpp/clang/tools/translator/tests/jacobi1.0/jacobi.dac.cpp`

### 7.1 `stencil1.0`

特点：

- 2D `3x3` 输入窗口
- 2D 单点输出
- 每一步后把 `matOut` 回写到 `matIn` 内部区域
- 再手动处理边界
- 有时间推进循环

它代表最典型的局部邻域 stencil。

### 7.2 `waveEquation1.0`

特点：

- 2D stencil
- 同时涉及：
  - `matCur`
  - `matPrev`
  - `matNext`
- 时间推进里有多块状态张量轮换

它同时包含：

- 局部邻域访问
- 多状态时间推进
- shell 标注与 calc 实际访问语义的对照问题

例如这个样例里：

- shell 把 `matCur` 标成 `WRITE`
- calc `waveEq(...)` 会读取 `cur`

因此它很适合用来观察 halo 需求和访问模式边界的叠加效果。

### 7.3 `jacobi1.0`

特点：

- 更像“每个输出点读取整行/整向量信息”
- 不属于典型的局部 halo stencil
- 文件里定义了 `DACPP_TRANSLATE_MODE 1`

它适合用来区分：

- 迭代类问题
- 局部邻域 stencil 问题

它更适合用于区分迭代类访问模式与局部邻域 stencil 的差异，而不是作为宿主时间循环保留逻辑的首选样例。

---

## 8. 设计 stencil MPI 时需要明确的语义层

如果下一步要为 stencil 单独设计 MPI 范式，至少需要明确下面几层语义：

1. 子域切分规则
   - 每个 rank 真正拥有哪一块主区域

2. halo 宽度推导
   - 如何从 shell / split / binding / 窗口形状推导 halo

3. 时间步中的交换时机
   - 每次 `<->` 前交换
   - 还是阶段性同步

4. 本地拥有区与写回区
   - 哪些点由本 rank 负责最终写回
   - 哪些点只是临时读取的 ghost / halo 数据

5. 多状态张量同步
   - `cur / prev / next` 轮换时哪些张量需要 halo
   - 哪些张量只需要拥有区

当前可以继续复用的更像是工具型能力：

- item 到位置集合的展开
- 压缩本地数组
- slot 映射
- 本地 view 访问

---

## 9. 相关文档

配合阅读这些文档会更快：

- 当前 MPI 主链路说明：
  `/Volumes/QUQ/working/dacpp/clang/tools/translator/docs/mpi_current_translation_logic.md`

- 最近的 bug / 修复结论：
  `/Volumes/QUQ/working/dacpp/clang/tools/translator/docs/bugs_fix_log`

- 当前环境配置：
  `/Volumes/QUQ/working/dacpp/clang/tools/translator/docs/macos_local_sycl_setup.md`
