# macOS 本地运行

当前这台机器的固定环境：

- SYCL: `/Volumes/QUQ/working/sycl-install`
- MPI: Homebrew `open-mpi`
- translator: `/Volumes/QUQ/working/dacpp/build/bin/translator/translator`

## 常用流程

先加载脚本：

```bash
source /Volumes/QUQ/working/dacpp/clang/tools/translator/env.sh
```

更稳的写法是直接放进 `bash -lc`，因为脚本内部用了 `BASH_SOURCE`：

```bash
bash -lc 'source /Volumes/QUQ/working/dacpp/clang/tools/translator/env.sh && dacpp-translate-and-build /path/to/file.dac.cpp /tmp/my_bin'
```

只翻译：

```bash
dacpp /path/to/file.dac.cpp --mpi
```

如果程序只要求 root 持有最新输出，可以关闭 gather 之后的全 rank 广播：

```bash
dacpp /path/to/file.dac.cpp --mpi --mpi-output-sync=root-only
```

编译单个已生成文件：

```bash
acpp-compile /path/to/file.dac_sycl_usm.cpp /tmp/my_bin
```

一步完成翻译加编译：

```bash
dacpp-translate-and-build /path/to/file.dac.cpp /tmp/my_bin
```

运行 MPI：

```bash
DYLD_LIBRARY_PATH=/Volumes/QUQ/working/sycl-install/lib mpirun -np 2 /tmp/my_bin
```

## 当前已修好的点

- `env.sh` 已补齐 macOS 下的 `isysroot` / `resource-dir` / `libc++` 参数
- `Rewriter_MPI.cpp` 已补齐：
  - shell 参数名映射
  - `MPI_Finalize()` 插入
  - 裸指针按 `View1D`
  - 共享 binding 统一 `bind_split_sizes`
  - 输出参数 gather 后再广播回所有 rank
  - `MPI_BYTE` 传输按真实字节数计数

## 运行全量测试脚本

你可以使用内置的 `test_mpi.sh` 脚本一次性跑完 `tests/` 目录下的所有非 Stencil 算法，并对 MPI 版本与单机版 SYCL 输出进行正确性比对：

该测试脚本的工作流程为：
1. 遍历所有的非 Stencil 测试（如 `FOuLa1.0`, `decay1.0`, `matMul1.0` 等）
2. **生成基准**：采用默认 `--mode=buffer` 不带 `--mpi` 进行代码翻译，编译并执行获得单机的 `base.out`
3. **生成待测目标**：采用 `--mode=buffer --mpi` 生成支持 MPI 的 SYCL 代码，然后由 `mpirun -np 2` 分发执行，获得 `mpi.out`
4. 对两者的结果（去除 AdaptiveCpp 警告后）进行比对

**运行方法**：

```bash
cd /Volumes/QUQ/working/dacpp/clang/tools/translator
bash test_mpi.sh
```

所有的中间产物和日志会保存在 `/tmp/dacpp_mpi_tests` 目录下。

> **注意：** 这份文档里提到的旧版 `Rewriter_Buffer.cpp` 基线路径已经不再是主线；当前 baseline 走的是 `Rewriter_Buffer_new.cpp`。历史旧实现已被归档到 `translator/archive/`，以免和现行代码路径混淆。

## 运行本机 macOS 回归脚本

如果你想在这台 macOS 机器上做本地 buffer 模式回归，使用 `test_local.sh`：

- `test_mpi.sh`：验证 `--mode=usm --mpi` 路径，基线是单机 `--mode=buffer`
- `test_local.sh`：验证本机单进程 `--mode=buffer` 路径，适合快速 smoke test 和定位非 MPI 代码生成问题

默认 smoke suite：

```bash
cd /Volumes/QUQ/working/dacpp/clang/tools/translator
bash test_local.sh
```

只跑指定样例：

```bash
cd /Volumes/QUQ/working/dacpp/clang/tools/translator
bash test_local.sh gradientSum
bash test_local.sh oddeven0.1 stencil1.0
```

打开扩展样例集：

```bash
cd /Volumes/QUQ/working/dacpp/clang/tools/translator
INCLUDE_EXTENDED_LOCAL_TESTS=1 bash test_local.sh
```

调整单个样例超时时间：

```bash
cd /Volumes/QUQ/working/dacpp/clang/tools/translator
LOCAL_TEST_TIMEOUT_SEC=300 bash test_local.sh MDP1.0
```

脚本行为说明：

1. 自动挑选主 `.dac.cpp`，按 `--mode=buffer` 生成 `*.dac_sycl_buffer.cpp`
2. 优先使用 `StandardSycl.cpp` 作为参考实现；如果本机不兼容，再回退到 `serial.cpp`，最后才回退到 `result.out`
3. 运行前会对旧版 `StandardSycl.cpp` 做一层本机兼容适配，例如 `CL/sycl.hpp` 命名空间补齐、`gpu_selector_v -> default_selector_v`
4. 输出比对前会过滤 AdaptiveCpp 警告，避免误报

所有的中间产物和日志会保存在 `/tmp/dacpp_local_tests` 目录下。

## 当前测试状态

`--mpi` 的 non-stencil 样例已全部通过，最近一轮日志目录：

```text
/tmp/dacpp-mpi-final.gqTD5X
```

本机 `test_local.sh` 默认 smoke suite 最近一轮结果：

```text
8 tests | 8 passed | 0 failed | 0 skipped
```
