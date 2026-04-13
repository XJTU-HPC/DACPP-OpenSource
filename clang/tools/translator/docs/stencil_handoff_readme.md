# Stencil MPI Handoff README

本文档面向后续继续设计 stencil MPI 优化方案的人。它的目标不是再证明“stencil 能不能跑”，而是帮助接手的人迅速建立下面这个新前提：

- 当前单步 MPI 主链路已经能正确覆盖现有 stencil 样例
- 后续 stencil 工作的核心问题变成了“如何自动识别，并在合适时降成 halo / 子域路径”

先记住五句话：

- `--mpi` 今天仍统一走 `rewriteMPI()`
- `stencil1.0` 和 `waveEquation1.0` 已经可以沿用当前单步 MPI wrapper 得到正确结果
- 当前 stencil 的主要短板是性能模型和通信模型，而不是基本正确性
- 当前 item-space wrapper 应继续保留为保底 fallback
- 真正新的工作是：自动识别规则 stencil，并把它们降成 halo-aware runtime

---

## 1. 当前项目上下文

目录：

- `/Volumes/QUQ/working/dacpp/clang/tools/translator`

这是 DACPP 到 SYCL 的 source-to-source translator。和本次 stencil 话题最相关的部分可以粗略分成三层：

1. parser
   - 负责抽取 shell / split / index / binding 语义

2. rewriter
   - 负责生成本地计算函数和 wrapper

3. runtime planner
   - 负责把 item 展开成局部需要访问的全局元素集合

今天 stencil 相关讨论默认站在这条现实上：

- 当前仓库已经有一条工作中的 MPI item-space 路径
- 这条路径已经足够支撑现有 stencil 样例的正确执行
- 因此后面新增的 stencil 方案，应该被视为“优化分支”，而不是“从零补支持”

---

## 2. 已确认的当前状态

### 2.1 non-stencil MPI 回归

2026-04-13 已执行：

```bash
cd /Volumes/QUQ/working/dacpp/clang/tools/translator
bash test_mpi.sh
```

结果：

- `10 tests | 10 passed | 0 failed | 0 skipped`

### 2.2 stencil 样例

同一天手工确认：

- `stencil1.0`：baseline 与 MPI 一致，`-np 2`
- `waveEquation1.0`：baseline 与 MPI 一致，`-np 2`
- `waveEquation1.0`：baseline 与 MPI 一致，`-np 4`

这说明当前单步 MPI wrapper 至少已经能覆盖两类典型 stencil：

- 单状态 2D 规则 stencil
- 多状态时间推进 stencil

---

## 3. 当前为什么能跑通现有 stencil

答案不是：

- “仓库里已经有 halo exchange runtime”

真正原因是：

- 当前 `rewriteMPI()` 生成的单步 wrapper 足够通用
- 再加上 `Rewriter_MPI.cpp` 现在能基于 `calc` AST 自动推导更真实的读写模式

本次关键修复后，MPI 路径不再死守 shell 表面的 `READ / WRITE / READ_WRITE`，而是会根据 `calc` 真正读了什么、写了什么来决定：

- 哪些参数要 scatter copy-in
- 哪些参数只需要本地输出缓冲
- 哪些参数需要 writeback

这对两个 stencil 样例的直接影响是：

### 3.1 `stencil1.0`

- `matIn`：shell 是 `READ_WRITE`，calc 实际只读，现在收紧成 `READ`
- `matOut`：shell 是 `READ_WRITE`，calc 实际只写，现在收紧成 `WRITE`

### 3.2 `waveEquation1.0`

- `matCur`：shell 是 `WRITE`，calc 实际读 `3x3` 窗口，现在纠正成 `READ`
- `matPrev`：shell 是 `READ_WRITE`，calc 实际只读，现在收紧成 `READ`
- `matNext`：保持 `WRITE`

因此今天的现实是：

- 当前 MPI 路径已经能正确处理“单步 stencil 调用”
- 但它仍然是 Root 每步 pack / scatter / gather 的通用方案
- 它还不是针对 stencil 通信结构优化过的方案

---

## 4. 当前真正的问题定义

现在 stencil 讨论应该改成下面这个问题：

- translator 有没有办法自动识别某个 `<->` 是规则 stencil
- 如果能识别，什么时候应该把它从当前单步 wrapper，降成 halo / 子域 / 邻居交换路径

这比“是否支持 stencil”更精确，因为：

1. 支持已经有了，至少对现有样例正确
2. 性能瓶颈来自当前仍然走 Root pack / scatter / gather
3. stencil 的价值在于局部规则依赖，如果能识别出来，就有机会用更便宜的通信模型

---

## 5. 建议的总体策略

推荐把后续设计分成两层：

### 5.1 第一层：分类

回答：

- 这个 `<->` 是不是一个规则 stencil
- 当前是否值得切到 stencil-aware MPI 路径

### 5.2 第二层：降级

只有分类成功后，才去生成：

- block ownership
- halo specification
- neighbor exchange
- 可选的 loop-level distributed state reuse

这样做的好处是：

- 保留当前单步 wrapper 作为 fallback
- 不会把混合程序里所有 `<->` 都强行推到 stencil runtime
- 自动识别不够确定时可以安全回退

---

## 6. 自动识别 stencil 时，translator 应该重点看什么

建议至少检查下面几类信息。

### 6.1 shell / binding 层

这里要回答的是：

- 输出 owner 是哪个 index 组合
- 输入窗口和输出 owner 是否由同一个 binding 组关联
- split 形状是不是规则窗口

对于 v1，更现实的限制是：

- 只考虑 1D / 2D
- 只考虑规则、常偏移、单位 stride 的窗口

### 6.2 calc 真实访问层

这里要回答的是：

- 哪些参数是只读窗口
- 哪些参数是 point write 输出
- 读访问是不是固定常偏移邻域
- 是否存在不规则 gather / scatter 或数据依赖逃逸

这一步应直接复用当前已经加强过的参数访问分析，而不是重新信任 shell 壳子标注。

### 6.3 宿主控制流层

这里要回答的是：

- `<->` 是否位于可识别的时间推进循环中
- 循环体里是否存在规律性的状态迁移
- 是否能识别 interior copy / tensor swap / 简单边界更新

如果识别不到稳定的时间推进语义，也不必判失败，可以先回退到单步 wrapper。

---

## 7. 建议的 lowering 目标

这里推荐保留两个目标形态，而不是一步到位只押一个。

### 7.1 单步 stencil wrapper

适用场景：

- `<->` 本身可以识别成规则 stencil
- 但外层时间推进循环识别不稳定，或暂时不想做跨步缓存

目标特征：

- rank ownership 改成规则 block，而不是线性 item range
- 通信从 Root pack / scatter 转成几何上可解释的 halo pull / exchange
- 本步结束后仍可 gather 回 root，保持和当前语义兼容

### 7.2 loop stencil wrapper

适用场景：

- `<->` 可识别成规则 stencil
- 外层时间推进循环也可识别

目标特征：

- 首次只分发 owned core
- 每个时间步只做邻居 halo / frontier exchange
- 循环内复用分布式状态，不再每步回 root
- 循环结束后再按 host 可见性做 gather / broadcast

从工程节奏上看，更合理的关系是：

- 当前单步通用 wrapper 是保底
- 单步 stencil wrapper 是第一层优化
- loop stencil wrapper 是性能上更有价值的第二层优化

---

## 8. 是否需要显式开关

从混合程序角度看，答案大概率是“需要保留显式分类手段”，但不一定一开始就强依赖它。

比较稳妥的策略是：

1. 自动识别作为主方向
2. 同时预留显式开关或前端标记作为兜底

显式分类的价值主要在这里：

- 混合程序里可能同时存在 stencil `<->` 和非-stencil `<->`
- 某些样例的自动识别边界可能一开始不稳
- 用户有时比编译器更清楚这个 `<->` 是否值得走 stencil 优化

但要明确：

- 当前仓库里还没有这层开关
- 这里只是在说明后续设计上值得预留这条接口

---

## 9. 自动识别成功后，建议生成哪些中间语义

相比直接在生成阶段拼代码，更建议先显式建几类中间描述：

1. `StencilShape`
   - 维度、owner 点、各维窗口范围

2. `OwnershipSpec`
   - 每个 rank 的规则 block 划分方式

3. `HaloSpec`
   - 每个读参数在各维的 halo minus / plus 宽度

4. `FrontierSpec`
   - 哪些本步新写出的 owner 输出，下一步需要推给邻居

5. `StateTransitionSpec`
   - 时间推进循环里的 copy / swap / boundary update 结构

有了这些中间语义后：

- 可以更清楚地区分“识别”和“生成”
- 也更容易让单步 wrapper 与 loop wrapper 共享分析结果

---

## 10. 为什么当前 item-space 路径仍然很重要

即使后面真的做 stencil-aware MPI，当前主链路依然有三个不可替代的作用。

### 10.1 correctness baseline

自动识别一旦不确定，必须能退回当前路径，先保证结果正确。

### 10.2 mixed-program fallback

不是所有 `<->` 都是 stencil，也不是所有 stencil 都值得专门优化。

### 10.3 对照参考

后续做 halo/block 路径时，当前单步 wrapper 可以作为：

- 结果对照
- 生成结构对照
- debug 退路

所以后续 stencil 优化的正确姿势不是“替换掉当前 MPI 主链路”，而是“在它之上分流出更适合 stencil 的优化路径”。

---

## 11. 推荐的推进顺序

如果要继续做实现，建议顺序是：

1. 先把“自动识别规则 stencil”的分析层做出来
2. 先保留当前单步 wrapper 作为完整 fallback
3. 再做单步 stencil wrapper 的 halo / block 降级
4. 最后再做 loop-level distributed state reuse

这样可以保证每一步都可验证：

- 先验证“识别对不对”
- 再验证“单步 halo 语义对不对”
- 最后再验证“跨时间步缓存值不值、稳不稳”

---

## 12. 当前最值得看的文件

- `/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/Rewriter_MPI.cpp`
  - 当前单步 MPI wrapper 生成
  - 参数模式推导

- `/Volumes/QUQ/working/dacpp/clang/tools/translator/dpcppLib/include/MPIPlanner.h`
  - 当前 item-space planner 与 pack / slot 运行时

- `/Volumes/QUQ/working/dacpp/clang/tools/translator/parser/lib/Shell.cpp`
  - binding 连通关系和 offset 信息

- `/Volumes/QUQ/working/dacpp/clang/tools/translator/tests/stencil1.0/stencil.dac.cpp`
  - 单状态 2D stencil 主样例

- `/Volumes/QUQ/working/dacpp/clang/tools/translator/tests/waveEquation1.0/waveEquation.dac.cpp`
  - 多状态时间推进 stencil 主样例

- `/Volumes/QUQ/working/dacpp/clang/tools/translator/docs/mpi_current_translation_logic.md`
  - 当前 MPI 主链路的真实边界

- `/Volumes/QUQ/working/dacpp/clang/tools/translator/docs/bugs_fix_log/mpi_param_mode_inference_fix.md`
  - 本次让单步 MPI 能正确覆盖当前 stencil 样例的关键修复

---

## 13. 当前 handoff 结论

截至 2026-04-13，stencil MPI 的 handoff 结论应该更新成：

- 当前 stencil 样例已经能走 `rewriteMPI()` 主链路并得到正确结果
- 因此下一个阶段的问题不再是“支持 stencil”
- 而是“能否自动识别规则 stencil，并在合适时降成 halo / 子域 / 循环复用路径”
- 当前单步 wrapper 应保留为 fallback 和 correctness baseline
