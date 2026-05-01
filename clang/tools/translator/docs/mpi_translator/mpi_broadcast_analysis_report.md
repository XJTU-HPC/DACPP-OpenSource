# DACPP MPI Broadcast 分析汇报

## 1. 问题背景

在 DACPP 的 MPI 翻译路径里，一个典型的 `Shell <-> Calc` 会被翻译成一套分布式执行流程：

1. Rank 0 保存完整的全局 Tensor。
2. Rank 0 根据 DACPP 的划分语义，把每个 rank 需要的数据片段分发出去。
3. 每个 rank 在本地用 SYCL kernel 执行自己的计算任务。
4. 对于被写入的输出 Tensor，各个 rank 把本地计算结果 gather 回 Rank 0。
5. Rank 0 把这些局部结果重新拼回完整 Tensor。

这个流程本身是正确的，但它带来一个关键问题：计算结束以后，最新的完整结果只在 Rank 0 上。其他 rank 本地的同名 Tensor 可能还是旧值。

如果后面的程序只是在 Rank 0 上打印结果，这通常没问题；但如果后续 Host 代码在所有 rank 上继续读取这个 Tensor，就会出现语义风险。

这里可以先看一个非 stencil 的例子。`decay1.0` 里每一次 DACPP 算子并不是做邻域 stencil，而是对每个同位素独立计算当前时间点的衰变量 `local_A`。算子结束后，Host 代码会把这一时刻的 `local_A` 写入历史结果矩阵 `A_tensor`：

```cpp
while (t_tensor[0] <= T) {
    DECAY(N0s_tensor, lambdas_tensor, local_A_tensor, t_tensor) <-> decay;
    A_tensor[10 * t_tensor[0]] = local_A_tensor;
    t_tensor[0] += dt;
}
```

这段代码里，`local_A_tensor` 是 MPI wrapper 写回的输出。写回完成后，最新完整结果只在 Rank 0 上。如果后续 Host 代码要在每个 rank 上读取 `local_A_tensor` 并更新别的 Host 数据结构，那么其他 rank 就也必须看到这份最新结果。否则 Rank 0 写入 `A_tensor` 的是新结果，而其他 rank 写入的可能是旧结果。

这里的重点不在于 stencil 或邻域依赖，而在于“算子输出离开了 DACPP wrapper，进入了普通 Host 更新逻辑”。下面这句就是典型的 Host 更新：

```cpp
A_tensor[10 * t_tensor[0]] = local_A_tensor;
```

这种问题不一定马上表现为死锁，也可能先表现为数值错误或输出不一致。但如果后续 Host 读取还参与了 `if / while` 这类控制流判断，就会进一步变成更危险的控制流分歧：一部分 rank 进入下一次 collective 通信，另一部分 rank 没有进入，程序就可能卡死。

因此，一个朴素但安全的做法是：只要某个输出 Tensor 被 Rank 0 gather 回来，就立刻用 `MPI_Bcast` 广播给所有 rank，让所有 rank 都拥有同一份最新结果。

这个做法能保证正确性，但代价很高。

## 2. 为什么不能总是 Broadcast

Broadcast 是一次全局通信。Tensor 越大，rank 越多，代价越明显。

在很多 DACPP 程序里，一个算子的输出并不会被普通 Host 代码读取，而是马上作为下一个 DACPP 算子的输入。例如：

```cpp
Step1_shell(A, B) <-> step1;
Step2_shell(B, C) <-> step2;
```

这里 `B` 虽然是第一个算子的输出，但它并不需要马上同步到每个 rank 的 Host 内存里。因为第二个算子执行时，MPI wrapper 本来就会重新根据全局 Tensor 做数据分发。也就是说，下一次 DACPP 计算会从 Rank 0 的最新全局结果出发，再把各个 rank 需要的局部数据 scatter 下去。

所以中间这次 `MPI_Bcast(B)` 在语义上是冗余的。

这个优化点的核心判断是：

如果输出 Tensor 在当前算子结束后，没有被普通 Host 代码读取，那么就不需要广播。

如果它只是继续出现在后面的 DACPP `<->` 表达式里，广播也可以省掉，因为后续 wrapper 会负责重新分发数据。

## 3. 我们要解决的问题

MPI Broadcast 分析要回答一个很具体的问题：

当前算子写回了一个输出 Tensor，Rank 0 已经拿到了最新结果。现在要不要把它广播给所有 rank？

这个判断不能只看 Tensor 是不是输出参数。输出参数只说明它被修改了，但不能说明后面有没有必要让所有 rank 立刻看到修改后的值。

真正要看的是后续程序语义：

- 后面普通 C++ 代码是否读取了这个 Tensor？
- 它是否参与了 `if`、`for`、计算表达式、打印、统计等 Host 行为？
- 它是否只是作为下一个 DACPP 算子的参数出现？
- 后面对它是纯写入，还是读旧值再更新？

只有把这些情况区分开，才能既保证正确性，又减少不必要通信。

## 4. 当前实现怎么做

当前实现主要分成分析和代码生成两步。分析部分在 `rewriter/lib/mpi/Rewriter_MPI_Analysis.cpp` 中，核心入口是 `tensorNeedsBroadcast(...)`。它会为当前输出 Tensor 创建一个 `TensorUseVisitor`，基于 Clang AST 遍历当前函数体，寻找这个 Tensor 在当前 `<->` 表达式之后的使用情况。遍历时，分析器会先识别“当前这一次 DACPP 表达式”的位置，避免把之前的代码误判成后续使用；之后如果发现该 Tensor 只是出现在后续另一个 DACPP `<->` 表达式内部，就认为它仍然处在 DACPP/MPI wrapper 管理范围内，后续 wrapper 会从 Rank 0 的最新全局数据重新分发局部数据，因此不需要立刻 broadcast。相反，如果它出现在普通 Host 代码里，例如 `print()`、`tensor2Array()`、普通计算表达式、`if / while` 条件，或者被用于 `+=`、`++` 这类会读取旧值的更新操作，分析器就认为非 0 rank 也必须看到最新结果，于是标记为需要 broadcast。为了区分“只是覆盖写”和“读取旧值后更新”，visitor 内部维护了 `WriteDepth` 和 `UpdateReadDepth`：纯赋值左值通常不需要同步旧结果，而普通读取、复合赋值、自增自减都需要同步。分析结束后，代码生成部分在 `rewriter/lib/mpi/Rewriter_MPI_Wrapper_Codegen.cpp` 中读取这个 `needsBcast` 结果：如果需要广播，就在 Rank 0 gather/writeback 完成后生成 `MPI_Bcast`，并让其他 rank 接收完整输出；如果不需要广播，就只保留 Rank 0 的完整写回，非 0 rank 跳过这一步。

如果需要广播，生成逻辑类似：

```cpp
if (mpi_rank == 0) {
    apply_writeback_by_globals(...);
    tensor.array2Tensor(global_out);
    MPI_Bcast(global_out.data(), count, mpiType, 0, MPI_COMM_WORLD);
} else {
    std::vector<T> global_out(tensor.getSize());
    MPI_Bcast(global_out.data(), count, mpiType, 0, MPI_COMM_WORLD);
    tensor.array2Tensor(global_out);
}
```

如果不需要广播，非 0 rank 不做这一步：

```cpp
if (mpi_rank == 0) {
    apply_writeback_by_globals(...);
    tensor.array2Tensor(global_out);
} else {
}
```

这样，输出 Tensor 的最新完整版本仍然保留在 Rank 0 上，后续 DACPP wrapper 依然可以从 Rank 0 分发正确数据；同时避免了把完整 Tensor 立刻同步到所有 rank。

## 5. 正确性保证

这个优化不是简单地删通信，而是基于语义判断删通信。

它遵循一个保守原则：

只要后续 Host 代码可能读取这个 Tensor，就保留 broadcast。

只有在分析器确认后续使用不依赖所有 rank 的本地副本时，才省掉 broadcast。

因此，优化后的程序仍然保持 MPI collective 调用的一致性：

- 如果生成了 `MPI_Bcast`，所有 rank 都会进入。
- 如果省掉了 `MPI_Bcast`，所有 rank 都不会进入。

这避免了最危险的情况：只有部分 rank 调用 collective。

## 6. 效率提升在哪里

这个优化带来的效率提升主要来自三个方面。

第一，减少全局通信次数。

Broadcast 是所有 rank 都参与的通信。对于大 Tensor，每次 broadcast 都要把完整结果从 Rank 0 传播到所有 rank。如果一个程序有多轮迭代，或者每一轮有多个中间 Tensor，那么冗余 broadcast 会被放大很多次。

按需 broadcast 后，中间结果如果只是流向下一个 DACPP 算子，就不再做全量同步。

第二，减少网络带宽压力。

MPI 程序在多节点上运行时，网络通信通常比本地计算更贵。特别是 Tensor 数据量较大时，一次 broadcast 的代价可能超过一次局部 kernel 计算。删掉不必要的 broadcast，可以直接降低网络传输量。

第三，减少同步等待。

Broadcast 不只是数据传输，也是同步点。所有 rank 都要等到这次 collective 完成才能继续。如果某些 rank 较慢，其他 rank 会被迫等待。

省掉 broadcast 后，程序可以减少这种全局等待，让后续流程更快进入下一阶段。

从执行模型上看，这个优化把原来的：

```text
Gather 写回 Rank 0
Broadcast 同步给所有 rank
下一次计算
```

缩短为：

```text
Gather 写回 Rank 0
下一次计算时再按需 Scatter
```

也就是说，我们把“立刻全量同步”改成了“后续真正需要时再分发”。

## 7. 两个直观例子

先看一个应该保留 broadcast 的例子。这个例子不是 stencil，而是“逐元素独立计算结果，然后由 Host 代码记录到另一个数据结构里”，和 `decay1.0` 的模式更接近：

```cpp
while (time_tensor[0] <= T) {
    Compute_shell(input_tensor, local_result_tensor, time_tensor) <-> compute;
    history_tensor[step_id] = local_result_tensor;
    time_tensor[0] += dt;
}
```

这里 `local_result_tensor` 是 MPI wrapper 的输出。后面的 Host 语句会读取它，并写入 `history_tensor`。如果没有 broadcast，那么只有 Rank 0 的 `local_result_tensor` 是完整新结果，其他 rank 读到的可能是旧值。于是各个 rank 上的 `history_tensor` 就会不一致。

所以这种“输出参与下一步 Host 更新”的场景需要 broadcast。它的重点不是打印，也不是 stencil 邻域，而是保证输出离开 wrapper 后，所有 rank 的 Host 侧状态一致。

再看一个可以省掉 broadcast 的例子。假设有三个连续算子：

```cpp
A_shell(input, tmp1) <-> calc_a;
B_shell(tmp1, tmp2) <-> calc_b;
C_shell(tmp2, output) <-> calc_c;

output.print();
```

在没有 broadcast 分析时，可能每一步输出都要同步：

```text
tmp1 gather 后 broadcast
tmp2 gather 后 broadcast
output gather 后 broadcast
```

但实际上：

- `tmp1` 只给下一个 DACPP 算子使用，不需要 broadcast
- `tmp2` 也只给下一个 DACPP 算子使用，不需要 broadcast
- `output` 后面被 Host 代码打印，需要 broadcast 或至少保证打印语义

优化后变成：

```text
tmp1 gather 后不 broadcast
tmp2 gather 后不 broadcast
output gather 后按语义需要 broadcast
```

如果中间 Tensor 很大，或者这个结构处在循环里，减少的通信量会非常明显。

## 8. 当前边界

当前 Broadcast 分析主要围绕已经捕获到的 AST 函数体进行后续引用分析。它适合处理主线测试里的直接控制流和常见 Host 使用方式。

对于更复杂的情况，比如：

- Tensor 被传入未知外部函数
- 指针或引用别名非常复杂
- 使用跨函数封装隐藏了真实读取
- 分析器无法确定某个表达式是否安全

更稳妥的策略应该是保守地保留 broadcast。

这类边界不是优化失败，而是编译器分析中的正常取舍：能确定安全时才优化，不能确定时优先保证正确性。

## 9. 当前测试中的覆盖情况

我用当前 translator 对 `test_mpi.sh` 中的 MPI 主线测试做了一次静态生成统计。这里的统计口径是：把每个 `.dac.cpp` 用 `--mode=buffer --mpi` 翻译后，检查生成的 `*_sycl_buffer.cpp` 中是否出现 `MPI_Bcast`。

需要说明一点：生成文件里一次逻辑 broadcast 通常会出现两处 `MPI_Bcast`，一处在 Rank 0 分支，一处在非 0 rank 分支。所以表里的 `BCAST=2` 通常表示“这个 case 的当前生成结果保留了一次输出同步 broadcast”。

### 9.1 当前会保留 Broadcast 的测试

这些 case 的生成结果中包含 `MPI_Bcast`。它们主要覆盖的是“输出结果后续会被 Host 代码读取或参与下一步 Host 更新，因此需要把 Rank 0 的最新结果同步给所有 rank”的场景。

| 测试 | 当前生成 | 为什么需要 Broadcast 分析 |
| --- | --- | --- |
| `mpiDenseCoverSibling1.0` | `BCAST=2` | `updates` 写回后，后续 Host 侧兄弟循环会读取它并更新 `state`，如果非 0 rank 看不到最新 `updates`，后续循环语义会不一致。 |
| `matMul1.0` | `BCAST=2` | `matC` 是输出矩阵，后面有 `matC.print()`，属于典型 Host 读取输出结果。 |
| `FOuLa1.0` | `BCAST=2` | 每个时间步输出 `u_kout` 后，Host 侧会把它写回 `u_tensor` 的下一层时间片，属于算子后 Host 更新。 |
| `liuliang1.0` | `BCAST=2` | 每轮 `new_rho` 写回后，Host 侧循环把它拷回 `rho`，并且最后读取 `rho[15]` 输出。 |
| `MDP1.0` | `BCAST=2` | 每轮 `new_p` 写回后，Host 侧把它更新到 `p`，最后还读取 `p[2]` 输出。这个 case 同时也覆盖了访问模式推断。 |
| `gradientSum` | `BCAST=2` | `matNeuronSum` 作为输出，后续 Host 侧直接读取前几个结果并打印。 |
| `stencil1.0` | `BCAST=2` | `matOut` 写回后，Host 侧把它拷回 `matIn` 并继续处理边界，最后打印 `matIn`。 |
| `waveEquation1.0` | `BCAST=2` | `matNext` 写回后，Host 侧更新 `matCur/matPrev`，后续时间步依赖这些 Host 更新。 |

这些测试的共同点是：输出 Tensor 不只是“给下一个 DACPP wrapper 使用”，而是在 wrapper 结束后被普通 C++ 代码读取或用于更新其他 Tensor。Broadcast 分析在这里的作用不是省通信，而是判断“这个同步不能删”，保证所有 rank 的 Host 视图一致。

### 9.2 当前生成中省掉 Broadcast 的测试

这些 case 的当前生成结果里没有 `MPI_Bcast`。它们覆盖的是另一半逻辑：分析器认为输出没有必要立刻同步到所有 rank，于是省掉广播。

| 测试 | 当前生成 | 对 Broadcast 分析的意义 |
| --- | --- | --- |
| `decay1.0` | `BCAST=0` | `local_A` 的结果主要被 Host 侧写入 `A_tensor` 的某个位置；当前生成没有插入 broadcast，体现了“不是所有输出都做全量同步”的路径。 |
| `DFT1.0` | `BCAST=0` | 当前生成没有对 `output` 做 broadcast，属于输出同步被省略的生成路径。 |
| `mandel1.0` | `BCAST=0` | `mandelbrot_flags` 写回后当前生成没有插入 broadcast，统计逻辑继续在 Host 侧读取结果。这个 case 可以作为后续检查“实参与形参名字不一致时是否还能准确识别 Host 读取”的观察点。 |
| `imageAdjustment1.0` | `BCAST=0` | 两段图像处理连续执行，当前生成没有插入 broadcast，比较适合观察连续算子之间是否存在冗余同步。 |
| `vectorAddCombo` | `BCAST=0` | 这是最典型的连续算子链：`tmp_tensor`、`shifted_tensor` 都只是中间结果，理论上不需要每一步都 broadcast。它很好地体现了省掉中间冗余通信的目标。 |

这里要稍微谨慎一点：`BCAST=0` 并不一定都代表“最终语义已经完全最优”。它说明当前生成结果走到了“不广播”的分支。像 `vectorAddCombo` 这种中间 Tensor 链路，是非常符合优化预期的；而 `mandel1.0` 这种输出后还有 Host 统计的 case，则可以作为后续完善分析精度的检查对象。

### 9.3 汇报时可以怎么讲

组会里可以把当前测试分成两类讲：

第一类是“Broadcast 必须保留”的测试，比如 `matMul`、`MDP`、`stencil`、`waveEquation`。这些 case 证明分析器不能只追求省通信，它还要识别 Host 后续读取，避免多 rank 看到不同数据。

第二类是“Broadcast 可以省掉”的测试，最有代表性的是 `vectorAddCombo`。它有连续三个 DACPP 算子，中间结果只在 DACPP 算子链内部流动。如果每一步都 broadcast，就会产生明显冗余通信；分析的价值就是把这种同步删掉。

所以这套测试并不是只验证一个方向，而是同时验证两个能力：

- 该保留时保留，保证正确性。
- 该省略时省略，提高效率。

## 10. 总结

MPI Broadcast 分析解决的是一个典型的分布式编译优化问题：不是所有写回 Rank 0 的结果，都需要立刻同步到所有 rank。

我们的做法是利用 Clang AST 理解后续代码语义，判断输出 Tensor 是否真的会被 Host 代码读取。

如果后续只是进入另一个 DACPP 算子，就省掉 broadcast；如果后续 Host 逻辑需要读取它，就保留 broadcast。

这样既保证了多 rank 语义一致，又减少了不必要的全局通信。效率提升主要体现在减少通信次数、降低网络带宽占用、减少 collective 同步等待。对于多轮迭代和大规模 Tensor 程序，这类优化会直接影响整体运行时间。
