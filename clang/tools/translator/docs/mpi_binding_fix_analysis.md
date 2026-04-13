# MPI Binding And Access-Mode Fix Analysis

## 1. 结论先行

`MDP1.0` 之前的失败**不像精度问题**，更像语义问题。

原因有两个：

1. 失败是**稳定且可重复**的，不是浮点舍入那种随机微扰
2. 差异量级明显偏大，`3.72665e-06` 对 `9.9295e-06` 不是普通并行归约顺序变化能解释的误差级别

结合代码生成结果，`MDP1.0` 的问题最终来自三层语义偏差：

- 第一层：`binding()` 在 MPI 重写阶段没有被正确保留下来
- 第二层：MPI 代码生成完全沿用 shell 上声明的 `READ_WRITE`，没有区分 calc 里“实际只读”和“实际写出”的参数
- 第三层：主函数重写时错误删除了包含 `<->` 的外层循环，只保留了一次 wrapper 调用

这三层都修完之后，MPI 测试已经达到：

- `10 tests | 10 passed | 0 failed | 0 skipped`

---

## 2. 第一层问题：`binding()` 语义在 MPI 代码生成阶段丢失

### 2.1 原来的实现是什么

问题函数：

- [Rewriter_MPI.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/Rewriter_MPI.cpp)

原实现的核心逻辑是 `collectSplitBindMeta(Shell* shell)`。

它原本的做法非常简单：

- 遍历每个 `ShellParam`
- 遍历每个参数上的 split
- 只要看到一个 split，就给它按名字分配一个新的 `bind_set_id`
- `bind_offset_expr` 一律写死成 `"0"`

也就是说，原实现并**没有真正使用 parser 里已经解析好的 binding 图信息**。

### 2.2 为什么会错

parser 其实已经在下面这两处算出了 binding 连通关系：

- [Shell.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/parser/lib/Shell.cpp)
- [Shell.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/parser/include/Shell.h)

现成可用的信息包括：

- `Shell::GetBindInfo(...)`
- `Shell::search_symbol(...)`

`GetBindInfo(...)` 会给出：

- `icls`: 绑定连通分量编号
- `offset`: 相对偏移表达式

但是原来的 MPI 重写器没有消费这些信息，导致：

- `binding(i, s)` 明明要求 `i` 和 `s` 属于同一个绑定组
- 生成出来却还是两个独立的 `bind_set_id`

典型错误例子：

### `FOuLa1.0`

源代码：

```cpp
binding(i, s);
```

旧的 MPI 生成结果：

```cpp
pattern_u_kin.bind_set_id.push_back(0);
pattern_u_kout.bind_set_id.push_back(1);
```

### `MDP1.0`

源代码：

```cpp
binding(idx, sp);
```

旧的 MPI 生成结果：

```cpp
pattern_p.bind_set_id.push_back(0);
pattern_new_p.bind_set_id.push_back(1);
```

这会把原本应该共享的一维 item 空间，错误扩展成笛卡尔积或错误分片，最终导致 stencil / time-stepping 类样例数值明显偏离。

### 2.3 现在改成了什么

现在 `collectSplitBindMeta(...)` 改成了两段式逻辑：

1. 优先调用 `shell->GetBindInfo(...)`
2. 对每个 `BINDINFO`：
   - 通过 `shell->search_symbol(info.v)` 找到对应 root split
   - 按 `info.icls` 归并成统一 `bind_set_id`
   - 保留 `info.offset`，空字符串时归一成 `"0"`
3. 只有那些**没有出现在 binding 图里**的 split，才走兜底逻辑分配独立 bind id

对应代码已经落在：

- [Rewriter_MPI.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/Rewriter_MPI.cpp)

### 2.4 修复效果

这一改动之后，`FOuLa1.0` 已经恢复通过。

这说明第一层判断是正确的：至少有一批失败并不是 broadcast 或 buffer 基线问题，而是 `binding()` 在 MPI planner 里丢了。

---

## 3. 第二层问题：`MDP1.0` 不是精度问题，而是访问模式语义偏差

### 3.1 原来的实现是什么

在 MPI 重写器中，参数访问模式基本直接沿用：

- `shellParam->getRw()`

这会影响：

- `AccessPattern.mode`
- 选择 `build_input_pack_map / build_output_pack_map / build_rw_pack_map`
- scatter 阶段是否把原 tensor 数据发给 worker
- gather / writeback 阶段是否把局部数据拼回全局 tensor
- 生成 `View1D<const T>` 还是 `View1D<T>`

也就是说，原实现默认认为：

- shell 上标成 `READ_WRITE`，那 MPI 路径就完整按 `ReadWrite` 处理

### 3.2 为什么 `MDP1.0` 会出问题

`MDP1.0` 的 shell 是：

```cpp
shell dacpp::list mdp_shell(
    dacpp::Vector<double>& p READ_WRITE,
    dacpp::Vector<double>& new_p READ_WRITE)
```

但 calc 实际上是：

```cpp
calc void mdp(dacpp::Vector<double>& p, double* new_p){
    double diffusion = D * (p[2] - 2 * p[1] + p[0]) / (dx * dx);
    double drift = (-A) * (p[2] - p[0]) / (2 * dx);
    new_p[0] = p[1] + dt * (diffusion + drift);
}
```

这里真实的访问语义其实是：

- `p`: 只读
- `new_p`: 只写

但旧的 MPI 生成器会把两者都按 `ReadWrite` 处理。

这带来两个问题：

1. **对 `p` 做了不必要的读写双向 pack/writeback**
   - 这会把输入窗口也纳入回写路径
   - 对重叠窗口迭代来说风险很高

2. **planner 认为 `p` 的本地副本会被修改**
   - 这不是 calc 的真实行为
   - 会让生成逻辑和真实数据流不一致

这类偏差导致的结果通常不是“最后几位抖动”，而是**稳定的系统性偏离**，所以不应被归类为精度问题。

### 3.3 现在改成了什么

我在 [Rewriter_MPI.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/Rewriter_MPI.cpp) 里补了一层“实际访问模式推断”：

- 基于 `calc->getCalcLoc()->getBody()` 的 AST
- 遍历参数在 calc 中的真实使用方式
- 记录每个参数是否：
  - 被读
  - 被写

然后在 MPI 路径里把非 `READ` 参数进一步收窄成：

- `READ`
- `WRITE`
- `READ_WRITE`

这层推断会影响：

- `AccessPattern.mode`
- `build_*_pack_map(...)` 的选择
- `sycl accessor` 的访问模式
- `View1D<const T>` / `View1D<T>` 的生成
- 是否进入最终 writeback 阶段

### 3.4 最终修复策略

这层修复最终不是“完全用 AST 推断覆盖 shell 声明”，而是采用了更稳妥的规则：

1. 先以 shell 上声明的 `IOTYPE` 为基准
2. 只有 shell 原本声明成 `READ_WRITE` 的参数，才允许继续根据 calc AST 收窄成：
   - `READ`
   - `WRITE`
   - `READ_WRITE`
3. shell 原本就是 `READ` 或 `WRITE` 的参数，保持原声明不变

这样做是因为：

- `MDP1.0` 这类样例确实需要把 `READ_WRITE` 收窄成真实模式
- 但 `matMul1.0` 里的 `dotProduct[0] += ...` 说明，不能把 shell 上明确的 `WRITE` 输出，因为 `+=` 就升级成 `READ_WRITE`

如果错误升级，就会把输出初值也一起 scatter 进来，导致结果整体偏大。

### 3.5 修复结果

这层规则稳定后：

- `MDP1.0` 的 `p` 会按 `READ`
- `MDP1.0` 的 `new_p` 会按 `WRITE`
- `matMul1.0` 的输出仍保持 `WRITE`

这说明：

- `MDP1.0` 不是精度问题
- 它确实是访问模式语义错误
- 修复后没有重新打坏原本通过的 `matMul1.0`

---

## 4. 第三层问题：`rewriteMain()` 错误删除了包含 `<->` 的外层循环

### 4.1 原来的实现是什么

问题位置：

- [Rewriter.h](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/include/Rewriter.h)

旧的 `rewriteMain()` 在检测到 `<->` 位于 `for` 循环内部时，会走一条特殊分支：

1. 找到最外层包含 `<->` 的 `for`
2. 直接删除整个 `for` 语句
3. 在原位置只插入一条 wrapper 调用

也就是说，旧实现会把整个循环结构删掉。

### 4.2 为什么会错

这会直接改变程序语义。

在 `MDP1.0` 里，原始代码是：

```cpp
for (int t = 0; t < T; ++t) {
    mdp_shell(p, new_p) <-> mdp;
    for(int i = 0; i <= N-3; i++){
        p[i+1] = new_p[i];
    }
}
```

但错误生成后的 MPI 代码只剩：

```cpp
mdp_shell_mdp(p, new_p);
```

时间推进循环、循环体里的状态更新都丢了。

### 4.3 现在改成了什么

`rewriteMain()` 现在改为：

- 保留原有 `for` 循环和外层控制流
- 只在原位置把 `<->` 表达式替换成对应的 wrapper 调用

也就是把：

```cpp
mdp_shell(p, new_p) <-> mdp;
```

改写成：

```cpp
mdp_shell_mdp(p, new_p);
```

但不会删除其所在的 `for` 循环。

### 4.4 修复结果

这一步修完以后，`MDP1.0` 才真正恢复通过。

### 4.5 为什么这个 bug 对有些测试没有暴露出来

`rewriteMain()` 吃掉整个 `for` 循环这个问题，虽然本质上非常严重，但它并没有在所有测试上表现成“肉眼可见的失败”。原因主要有三类：

#### 情况 A：`<->` 根本不在 `for` 循环里

这类样例不会走旧的 `forStatement` 特殊分支，所以完全不受这个 bug 影响。

典型样例：

- `matMul1.0`
- `DFT1.0`
- `vectorAddCombo`
- `imageAdjustment1.0`
- `gradientSum`

#### 情况 B：计算在循环里，但不是 `for`

当前旧逻辑只对 `ForStmt` 走“删整个循环再插调用”的特殊路径。

因此如果外层控制流是：

- `while`
- 或其他没有被 `forStatement` 记录的结构

那这个 bug 也不会触发。

典型样例：

- `decay1.0`

它的时间推进写法是 `while (t_tensor[0] <= T)`，因此不受这个 `for` 专属 bug 影响。

#### 情况 C：虽然 `for` 被吃掉了，但测试输出刚好对这个错误不敏感

这是最容易误导判断的一类。

典型样例：

- `liuliang1.0`

这个样例原始确实位于 `for (int t = 0; t < TIME_STEPS; ++t)` 中，但测试最后打印的是 `rho[15]`。而初始化里，`rho[15]` 位于高密度常值平台区：

```cpp
if (i < WIDTH / 4) {
    rho[i] = 40;
}
```

更新公式：

```cpp
new_rho[0] = rho[1] - (DELTA_T / DELTA_X) * (q(rho[1]) - q(rho[0]));
```

在平台区内，相邻点相同，`q(rho[1]) - q(rho[0]) == 0`，因此一步和多步都可能保持同样的输出值。也就是说：

- 程序语义其实已经错了
- 但当前测试观测点没把这个错误放大出来

#### 结论

所以，`for` 被吃掉没有在所有测试上立刻爆雷，并不代表 bug 不严重，只说明当前测试集里有一部分：

- 没走到这条错误路径
- 或者观测值刚好不足以暴露问题

`MDP1.0` 之所以最终把这个问题暴露出来，是因为：

- 它的时间推进循环是核心语义，不是外围优化
- 打印的结果对迭代次数高度敏感
- 一旦只执行一次 wrapper，数值就会立即偏离 baseline

---

## 5. 和 Buffer 修复的关系

这一轮 MPI 修复和前面的 buffer 修复是两类不同问题：

### Buffer 修复

对应文档：

- [buffer_template_bugfix.md](/Volumes/QUQ/working/dacpp/clang/tools/translator/docs/buffer_template_bugfix.md)

修复的是：

- `Rewriter_Buffer_new.cpp` 的跨表达式参数污染
- buffer 路径误调用 USM 模板

直接解决的样例：

- `vectorAddCombo`
- `imageAdjustment1.0`

### MPI 修复

本文件记录的是：

- `binding()` 连通分量没有进入 MPI planner
- `READ_WRITE` 没有细化为 calc 的真实访问模式

直接影响的样例：

- `FOuLa1.0`
- `MDP1.0`

---

## 6. 最终结果

修完三层问题之后：

1. `binding()` 连通分量已经正确进入 MPI planner
2. `READ_WRITE` 参数已经按 calc 真实访问模式安全收窄
3. 主函数中的循环结构已经被正确保留

最终回归结果：

```bash
cd /Volumes/QUQ/working/dacpp/clang/tools/translator
bash test_mpi.sh
```

得到：

- `10 tests | 10 passed | 0 failed | 0 skipped`
