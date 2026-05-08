# OR Followup Phase 4 Benchmark 2026-05-08

## 范围

本轮只覆盖 Phase 4 OR followup lowering 直接影响的样例：

- `MDP1.0`
- `liuliang1.0`
- `stencil1.0`
- `waveEquation1.0`

旧实现数据不重新运行，直接引用
`clang/tools/translator/docs/benchmarks/mpi_sycl_efficiency_benchmark_2026-05-06.md`
中的结果。新实现和手写 MPI+SYCL 为本次实测。

## 测试口径

- 分支：`tqc-2`
- MPI 进程数：`mpirun -np 4`
- translator build：`cmake --build build --target translator -j8`
- 驱动脚本：`clang/tools/translator/bench_mpi_only_requested.py`
- 临时结果目录：`/Volumes/QUQ/working/mpi_tmp/phase4_or_followup_bench`
- 计时：外部 wall-clock，仅统计 `mpirun` 运行阶段，不包含翻译和编译
- 新实现/手写 MPI+SYCL：每个样例跑 3 次，表中取 median
- 旧实现：使用 2026-05-06 文档中的单次记录

命令：

```bash
cmake --build build --target translator -j8
MPI_ONLY_BENCH_TMP_DIR=/Volumes/QUQ/working/mpi_tmp/phase4_or_followup_bench \
MPI_ONLY_BENCH_RANKS=4 \
MPI_ONLY_BENCH_TIMEOUT_SECONDS=1800 \
python3 clang/tools/translator/bench_mpi_only_requested.py \
  MDP1.0 liuliang1.0 stencil1.0 waveEquation1.0
```

## Median 对比

`自动/手写耗时倍率` 越接近 `1.0x` 越好；大于 `1.0x` 表示自动生成代码更慢。

| 样例 | 规模 | 手写 MPI+SYCL 本次 median(s) | 新实现 OR median(s) | 新实现 自动/手写 | 旧实现文档 DAC-MPI(s) | 旧实现文档 自动/手写 | 新实现/旧实现 DAC |
|---|---:|---:|---:|---:|---:|---:|---:|
| `MDP1.0` | N=8192, T=600 | 0.795349 | 1.205218 | 1.52x | 0.801768 | 0.94x | 1.50x |
| `liuliang1.0` | WIDTH=8192, steps=1000 | 0.714773 | 1.489254 | 2.08x | 0.949857 | 1.06x | 1.57x |
| `stencil1.0` | 2048x2048, steps=600 | 1.500077 | 18.675504 | 12.45x | 4.614142 | 3.09x | 4.05x |
| `waveEquation1.0` | 2048x2048, steps=600 | 1.532526 | 37.721034 | 24.61x | 4.790831 | 3.25x | 7.87x |

## 原始数据

| 样例 | 手写 run1 | 新实现 run1 | 手写 run2 | 新实现 run2 | 手写 run3 | 新实现 run3 |
|---|---:|---:|---:|---:|---:|---:|
| `MDP1.0` | 0.831805 | 1.205218 | 0.762571 | 1.532670 | 0.795349 | 0.988529 |
| `liuliang1.0` | 0.907605 | 1.550026 | 0.658455 | 1.489254 | 0.714773 | 1.483854 |
| `stencil1.0` | 1.500077 | 18.750247 | 1.842740 | 18.538254 | 1.385434 | 18.675504 |
| `waveEquation1.0` | 1.698328 | 38.463363 | 1.489260 | 37.721034 | 1.532526 | 37.263676 |

## 结论

- 4 个样例均可翻译、编译、运行成功。
- 当前生成代码确实走 OR wrapper，例如生成文件中包含
  `__dacpp_mpi_or_stencilShell_stencil_0` 和
  `__dacpp_mpi_or_waveEqShell_waveEq_0`，不是整文件退回 Phase-C。
- `MDP1.0` / `liuliang1.0` 的新实现已完成 OR 侧 distributed followup lowering，
  但为了保持非 root host-visible reader 语义，需要同步 followup reader tensor，
  因此相对旧实现有 1.50x / 1.57x 的 DAC 运行时增加。
- `stencil1.0` / `waveEquation1.0` 的 2D followup lowering 成本更明显。
  当前 OR wrapper 每步仍广播完整 reader、gather materialized writer，并同步
  followup/read-cache 目标 tensor；语义已经收口，但通信模式还不是手写 MPI+SYCL
  那种长期 resident halo 分片。
- 因此，本轮 benchmark 的性能结论是：Phase 4 followup lowering 功能可用，
  但 2D stencil 当前是语义优先的保守实现，性能明显慢于旧文档记录和手写版本。
