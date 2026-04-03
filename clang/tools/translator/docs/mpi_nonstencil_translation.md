# MPI 非 stencil 说明

这份文档只保留 4 件事：

1. 当前 `translator --mpi` 的核心逻辑
2. 最重要的代码位置
3. 关键修复点
4. 最近一轮 non-stencil 测试结果

## 1. 核心逻辑

当前 MPI 翻译走的是“按工作项切分，再按访问模式打包数据”的路线。

整体流程：

1. 解析 `shell <-> calc`
2. 为每个参数生成 `AccessPattern`
3. 计算全局工作项数 `total_items`
4. 按 rank 切分工作项区间
5. root 按访问模式打包输入并发给各 rank
6. 各 rank 在本地构造 `slots` 和 `View1D` / `View2D`
7. rank 内部按工作项执行本地 `calc`
8. 输出参数写回 root
9. 对 `WRITE` / `READ_WRITE` 参数，再广播回所有 rank

最后这一步很重要。否则像 `jacobi1.0` 这类 shell 返回后还会继续读输出 tensor 的程序，会因为各 rank 状态不一致而在下一轮通信挂住。

## 2. 关键代码位置

- `clang/tools/translator/translator.cpp`
  这里定义 `--mpi` 开关。
- `clang/tools/translator/rewriter/lib/Rewriter_MPI.cpp`
  这里生成本地 `calc`、wrapper、scatter/gather/broadcast 代码。
- `clang/tools/translator/dpcppLib/include/MPIPlanner.h`
  这里放运行期映射逻辑：`ItemRange`、`AccessPattern`、`PackMap`、`View1D/2D`。

## 3. 关键修复点

当前这层已经补齐：

- 裸指针参数按 `View1D`
- `Vector` / `Matrix` 参数按 calc 参数类型推断本地 view
- shell 参数名和真实 tensor 名对齐
- 多参数共享 binding 时统一 `bind_split_sizes`
- `MPI_Finalize()` 插到每个 `return` 前
- 输出参数 gather 后再广播回所有 rank
- `MPI_BYTE` 路径按真实字节数做 `MPI_Send` / `MPI_Recv` / `MPI_Bcast`

最后一条是 `imageAdjustment1.0` 这类结构体数据能正确工作的关键。

## 4. 当前测试结果

2026-04-03 这台机器上的 `--mpi` non-stencil 样例测试状态（`mpirun -np 2`）：

| 样例 | 状态 | 备注 |
|------|------|------|
| `DFT1.0` | PASS | |
| `FOuLa1.0` | 浮点非确定性 | 两次 run 最大相对误差 ~2.6%，结构正确 |
| `MDP1.0` | PASS | |
| `decay1.0` | PASS | |
| `gradientSum` | SKIP | 无 result.out |
| `imageAdjustment1.0` | PASS | |
| `jacobi1.0` | PASS | |
| `liuliang1.0` | PASS | |
| `mandel1.0` | PASS | |
| `matMul1.0` | PASS | |
| `oddeven0.1` | PASS | |

注意：`result.out` 已于 2026-04-03 从实际 MPI 运行输出重新生成，之前的文件是 serial 参考输出，与 MPI 模式输出不一致。

日志目录：

```text
/tmp/dacpp-mpi-j1JAtl
```

## 5. 适用范围

当前实现更适合：

- 非 stencil
- 规则 split / binding
- 输出位置不发生跨工作项冲突

当前机器上验证的是：

- AdaptiveCpp
- OpenMP host backend
- OpenMPI

没有验证 Metal backend，也不依赖 Intel oneAPI。
