# MPI Param-Mode Inference Fix

本文档记录 2026-04-13 对 `Rewriter_MPI.cpp` 做的一个关键修复：让当前单步 MPI 路径可以根据 `calc` 的真实访问方式，自动识别并收紧参数的有效读写模式。

这次修复的意义非常直接：

- 它让当前单步 MPI wrapper 不再只依赖 shell 表面的 `READ / WRITE / READ_WRITE`
- 它让 `stencil1.0` 和 `waveEquation1.0` 这类 stencil 样例可以沿用现有 MPI 主链路得到正确结果
- 它也没有打坏现有 non-stencil MPI 回归

---

## 1. 问题背景

旧的 MPI 路径过度相信 shell 上的读写标注。

这在简单样例里问题不一定会暴露，但一到 stencil 和时间推进类样例，就会碰到一种很常见的情况：

- shell 上的 `READ / WRITE / READ_WRITE` 只是一个外层壳子
- calc 函数体里的真实访问方式可能更窄，甚至和壳子方向不一致

如果 MPI wrapper 直接沿用 shell 标注，就会连带影响：

- `AccessPattern.mode`
- 选用哪种 pack builder
- 哪些 tensor 需要 scatter copy-in
- 哪些 tensor 需要 gather / writeback
- 本地 view / accessor 的只读或可写形态

一旦这些地方和真实数据流不一致，单步 MPI 虽然还能编译，但结果很容易出错。

---

## 2. 旧问题具体是什么

旧实现的薄弱点有两层。

### 2.1 `ParamAccessVisitor` 的访问语义过粗

旧 visitor 的问题是：

- 对参数访问只做了比较粗的读写区分
- 没有显式区分“更新型读取”这件事
- 对 `operator=` / `operator+=` / `++` / `--` 这类语义覆盖不完整

这会导致：

- `a += x`
- `++a`
- 某些重载赋值形式

在 visitor 看来不够精确，进而影响有效模式判定。

### 2.2 `inferEffectiveParamModes(...)` 太依赖 shell 外壳

旧逻辑会过度保留 shell 标注，无法稳定地从 calc 的真实访问重新得到有效模式。

这会直接卡住一类很典型的 stencil 参数：

- shell 标了 `WRITE`
- 但 calc 实际把它当输入窗口读

`waveEquation1.0` 就是现成例子。

---

## 3. 为什么 `waveEquation1.0` 会暴露这个问题

`waveEquation1.0` 的 shell 是：

```cpp
shell dacpp::list waveEqShell(
    dacpp::Matrix<double>& matCur WRITE,
    dacpp::Matrix<double>& matPrev READ_WRITE,
    dacpp::Matrix<double>& matNext WRITE)
```

但 calc 实际是：

```cpp
calc void waveEq(dacpp::Matrix<double>& cur, double* prev, double* next) {
    double u_xx = (cur[2][1] - 2.0f * cur[1][1] + cur[0][1]) / (dx * dx);
    double u_yy = (cur[1][2] - 2.0f * cur[1][1] + cur[1][0]) / (dy * dy);
    next[0] = 2.0f * cur[1][1] - prev[0] + (c * c) * dt * dt * (u_xx + u_yy);
}
```

这里真实访问语义是：

- `matCur` / `cur`：读 `3x3` 窗口
- `matPrev` / `prev`：只读单点
- `matNext` / `next`：只写单点

如果 MPI 路径坚持用 shell 上的：

- `matCur = WRITE`
- `matPrev = READ_WRITE`

就会导致：

- `matCur` 没有正确 copy-in
- `matPrev` 被误判成需要 writeback

这和 calc 的真实数据流明显不一致。

---

## 4. 这次具体改了什么

修改文件：

- `/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/Rewriter_MPI.cpp`

### 4.1 扩展 `ParamAccessVisitor`

现在 visitor 会分别记录三个集合：

- `Reads`
- `UpdateReads`
- `Writes`

其中：

- `Reads` 表示普通读
- `UpdateReads` 表示“这个对象被更新时，先读了旧值”
- `Writes` 表示写目标

同时补上了更完整的 AST 覆盖：

- `TraverseBinaryOperator(...)`
  - 处理普通赋值

- `TraverseCompoundAssignOperator(...)`
  - 处理 `+=`、`-=` 这类复合赋值

- `TraverseUnaryOperator(...)`
  - 处理 `++`、`--`

- `TraverseCXXOperatorCallExpr(...)`
  - 处理重载赋值、重载复合赋值、重载自增自减

这样做之后，visitor 对“读旧值再写回”的模式不会再和普通只读或纯写混在一起。

### 4.2 重写 `inferEffectiveParamModes(...)`

现在 `inferEffectiveParamModes(...)` 的逻辑是：

1. 先拿到 shell 的初始模式
2. 再遍历 `calc` body 的真实 AST 访问
3. 对每个参数重新计算本次 MPI wrapper 使用的有效模式

当前规则可概括成：

- `reads && writes` -> `READ_WRITE`
- `writes && updateReads`
  - 如果 shell 原本就是 `WRITE`，则保持 `WRITE`
  - 否则提升为 `READ_WRITE`
- `writes` -> `WRITE`
- `reads || updateReads` -> `READ`

这里专门保留了一个很重要的例外：

- 对 shell 明确声明为 `WRITE` 的输出缓冲，如果它只是因为 `+=` 或类似更新语义出现 `updateReads`，就仍然允许保持 `WRITE`

这样可以避免把一类“输出累加缓冲”错误升级成必须 copy-in 的 `READ_WRITE`。

---

## 5. 修复后对 stencil 样例的直接效果

### 5.1 `stencil1.0`

源程序里：

- `matIn` 壳子是 `READ_WRITE`
- `matOut` 壳子是 `READ_WRITE`

calc 真实访问后，MPI 有效模式会变成：

- `matIn` -> `READ`
- `matOut` -> `WRITE`

这正是单步 stencil wrapper 真正需要的语义。

### 5.2 `waveEquation1.0`

修复后有效模式会变成：

- `matCur` -> `READ`
- `matPrev` -> `READ`
- `matNext` -> `WRITE`

这让当前单步 MPI wrapper 能正确：

- 下发 `matCur` 的窗口输入
- 下发 `matPrev` 的点输入
- 仅对 `matNext` 走输出写回

---

## 6. 为什么这次修复不会打坏已有 non-stencil 样例

这次修复虽然增强了自动识别能力，但没有简单地“只要出现读就一律升级成 `READ_WRITE`”。

相反，它保留了一个对现有样例很关键的约束：

- 如果某个参数本来就是典型输出缓冲，shell 明确给的是 `WRITE`
- 而 calc 里的“读”只是更新型读，例如 `+=`
- 那么 MPI 路径仍然可以保持它为 `WRITE`

这样做的目的是避免：

- 不必要的输入 scatter
- 把输出缓冲错误地当成 read-write 状态数组

这也是为什么修复之后，non-stencil MPI 回归没有重新被打坏。

---

## 7. 验证结果

### 7.1 构建

已执行：

```bash
cmake --build /Volumes/QUQ/working/dacpp/build --target translator -j8
```

构建通过。

### 7.2 non-stencil MPI 回归

已执行：

```bash
cd /Volumes/QUQ/working/dacpp/clang/tools/translator
bash test_mpi.sh
```

结果：

- `10 tests | 10 passed | 0 failed | 0 skipped`

### 7.3 stencil 样例

已确认：

- `stencil1.0`：baseline 与 MPI 一致，`-np 2`
- `waveEquation1.0`：baseline 与 MPI 一致，`-np 2`
- `waveEquation1.0`：baseline 与 MPI 一致，`-np 4`

---

## 8. 这次修复对后续 stencil 设计意味着什么

这次修复的意义不是“已经做完 stencil 优化”，而是先把一个更重要的前提补齐了：

- 当前单步 MPI 路径已经可以作为 stencil 的 correctness baseline

因此后续如果继续做 stencil-aware MPI，正确的推进方式应该是：

1. 把当前单步 wrapper 当作 fallback 和结果基线
2. 在此基础上继续讨论自动识别 stencil
3. 再讨论是否把可识别 stencil 降成 halo / 子域 / 循环复用路径

换句话说，这次修复把问题从：

- “单步 MPI 连当前 stencil 都跑不对”

推进成了：

- “单步 MPI 已经能跑对，下一步该优化 stencil 的通信模型”
