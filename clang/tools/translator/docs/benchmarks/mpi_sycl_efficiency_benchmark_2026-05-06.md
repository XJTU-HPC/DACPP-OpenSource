# MPI+SYCL Efficiency Benchmark 2026-05-06

## 目标

对 `clang/tools/translator/tests` 下 14 个非 `mpi*` 应用测试进行效率测试，只比较：

- 手写标准 MPI+SYCL：`*.MPI_StandardSycl.cpp`
- DAC 翻译生成的 MPI+SYCL：`*.large_dac.cpp` 经 `dacpp --mpi --mode=buffer` 翻译后编译运行

单机 SYCL / serial / DAC buffer baseline 不纳入本次结果。

## 测试环境

- 机器：`mzk-MacMini.local`
- OS：Darwin 25.3.0 arm64
- MPI：Open MPI 5.0.9
- SYCL 编译器：AdaptiveCpp 安装在 `/Volumes/QUQ/working/sycl-install`
- MPI 进程数：`mpirun -np 4`
- 结果目录：`/Volumes/QUQ/working/mpi_tmp/mpi_only_rescaled`
- 驱动脚本：`clang/tools/translator/bench_mpi_only_requested.py`

## 测试方法

1. 每个测试单独复制到临时目录，不直接修改 `tests/` 原始文件。
2. 在临时副本中统一调整数据规模，并压缩大规模输出，只打印少量 sample，避免 I/O 污染运行时间。
3. 对标准 MPI+SYCL 副本执行：

   ```bash
   acpp-compile case.MPI_StandardSycl.cpp standard_bin
   mpirun -np 4 standard_bin
   ```

4. 对 DAC 副本执行：

   ```bash
   dacpp case.mpi.dac.cpp --mode=buffer --mpi
   acpp-compile case.mpi.dac_sycl_buffer.cpp dac_mpi_bin
   mpirun -np 4 dac_mpi_bin
   ```

5. 表中时间只统计 `mpirun` 运行阶段的外部 wall-clock 时间，不包含翻译和编译。

## 修复项

上一轮 `decay1.0` 的 DAC-MPI 运行失败不是 translator 崩溃，而是临时规模改写把 `dt` 改成了 `1.0`，但源程序内部仍使用 `A_tensor[10*t]` 作为输出行索引，导致 Tensor 行越界。

修复方式：

- 保留源程序时间步语义：`dt = 0.1`
- 设定 `T = steps / 10.0`
- 将临时副本中的 `while(t_tensor[0] <= T)` 改为 `while(t_tensor[0] < T)`，避免多跑一个边界步

`gradientSum` 的手写标准 MPI+SYCL 参考文件使用了 `using namespace sycl;`，当前 AdaptiveCpp 头文件暴露的是 `cl::sycl`，因此在临时副本中加入了兼容声明；原始测试源未修改。

## 放大规模

上一轮很多样例运行时间低于 2s，因此本轮手动放大了这些样例的数据规模。`matMul1.0` 在 `2048x2048` 下已经超过 2s，所以保持不变。

| 测试样例 | 本轮规模 |
|---|---:|
| DFT1.0 | N=4096 |
| FOuLa1.0 | m=8192, n=600 |
| MDP1.0 | N=8192, T=600 |
| decay1.0 | numIsotopes=8192, steps=600 |
| gradientSum | 8192x4096 |
| imageAdjustment1.0 | 4096x4096 |
| jacobi1.0 | N=4096, iter=300 |
| liuliang1.0 | WIDTH=8192, steps=1000 |
| mandel1.0 | 4096x4096, max_iter=1000 |
| matMul1.0 | 2048x2048 |
| oddeven0.1 | N=4096 |
| stencil1.0 | 2048x2048, steps=600 |
| vectorAddCombo | N=8388608 |
| waveEquation1.0 | 2048x2048, steps=600 |

## 当前结果

| 测试样例 | 数据规模 | 标准 MPI+SYCL 时间(s) | DAC 翻译后 MPI 时间(s) | 状态 |
|---|---:|---:|---:|---|
| DFT1.0 | N=4096 | 1.044311 | 0.783150 | ok |
| FOuLa1.0 | m=8192, n=600 | 0.876812 | 2.345472 | ok |
| MDP1.0 | N=8192, T=600 | 0.851989 | 0.801768 | ok |
| decay1.0 | numIsotopes=8192, steps=600 | 0.839301 | 0.794610 | ok |
| gradientSum | 8192x4096 | 0.727669 | 1.631674 | ok |
| imageAdjustment1.0 | 4096x4096 | 3.125963 | 0.896448 | ok |
| jacobi1.0 | N=4096, iter=300 | 0.866027 | 0.775995 | ok |
| liuliang1.0 | WIDTH=8192, steps=1000 | 0.896834 | 0.949857 | ok |
| mandel1.0 | 4096x4096, max_iter=1000 | 2.530378 | 5.015036 | ok |
| matMul1.0 | 2048x2048 | 5.765554 | 7.576529 | ok |
| oddeven0.1 | N=4096 | 1.598514 | 4.529443 | ok |
| stencil1.0 | 2048x2048, steps=600 | 1.494438 | 4.614142 | ok |
| vectorAddCombo | N=8388608 | 0.675558 | 4.901816 | ok |
| waveEquation1.0 | 2048x2048, steps=600 | 1.474793 | 4.790831 | ok |

## 观察

- `decay1.0` 的运行错误已修复，标准 MPI+SYCL 和 DAC-MPI 均可完成。
- `DFT1.0`、`MDP1.0`、`decay1.0`、`jacobi1.0` 在本轮中 DAC-MPI 与标准实现接近，部分略快。
- `imageAdjustment1.0` 的标准 MPI+SYCL 参考实现已在 2026-05-07 对齐 DAC calc 语义：第一步只写 red channel，green/blue 保持输出初值；对齐后 DAC-MPI 为 0.29x 标准实现时间。
- `vectorAddCombo`、`stencil1.0`、`waveEquation1.0`、`oddeven0.1` 的 DAC-MPI 明显慢于标准实现。
- 仍有若干标准 MPI+SYCL 参考实现低于 2s，说明手写标准版在这些规模下仍较轻；但 DAC-MPI 侧除少数轻量用例外，多数已经超过 2s。
