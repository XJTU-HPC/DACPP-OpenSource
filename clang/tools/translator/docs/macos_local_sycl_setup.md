# macOS 本地开发与全量测试

这份文档记录当前这台 macOS 机器上，如何：

- 重新编译 `translator`
- 翻译并编译单个 `.dac.cpp`
- 运行本地单进程回归
- 运行 MPI 全量回归
- 查看日志与常见排障

文档内容以当前仓库主线脚本和最近一次实测结果为准。

## 固定环境

当前这台机器的固定路径：

- `ACPP_ROOT=/Volumes/QUQ/working/sycl-install`
- `translator=/Volumes/QUQ/working/dacpp/build/bin/translator/translator`
- `translator_dir=/Volumes/QUQ/working/dacpp/clang/tools/translator`
- MPI: Homebrew `open-mpi`

## 先加载环境

建议每次先加载：

```bash
source /Volumes/QUQ/working/dacpp/clang/tools/translator/env.sh
```

如果你是从脚本、子 shell 或自动化命令里调用，推荐用 `bash -lc`，因为 `env.sh` 里依赖了 `BASH_SOURCE`：

```bash
bash -lc 'source /Volumes/QUQ/working/dacpp/clang/tools/translator/env.sh && dacpp /path/to/file.dac.cpp --mode=buffer'
```

`env.sh` 目前已经处理好了这台 macOS 上常见的编译细节，包括：

- `isysroot`
- `clang resource-dir`
- `libc++`
- Homebrew OpenMPI 头文件和链接参数
- AdaptiveCpp 运行时库路径

## 重新编译 translator

如果你修改了 `translator.cpp`、`Rewriter_*`、`parser` 或模板相关代码，先重新编译：

```bash
cd /Volumes/QUQ/working/dacpp
cmake --build build --target translator -j8
```

编译成功后，二进制位置是：

```text
/Volumes/QUQ/working/dacpp/build/bin/translator/translator
```

如果你只想确认二进制是否存在：

```bash
ls -l /Volumes/QUQ/working/dacpp/build/bin/translator/translator
```

## 单文件常用流程

### 1. 只做翻译

本机 buffer 路径：

```bash
dacpp /path/to/file.dac.cpp --mode=buffer
```

MPI 路径：

```bash
dacpp /path/to/file.dac.cpp --mode=buffer --mpi
```

如果你只需要 root rank 持有 gather 后的最终输出，可以关闭“回广播”：

```bash
dacpp /path/to/file.dac.cpp --mode=buffer --mpi --mpi-output-sync=root-only
```

### 2. 编译已生成的文件

当前主线生成物通常是：

- `*.dac_sycl_buffer.cpp`

直接编译：

```bash
acpp-compile /path/to/file.dac_sycl_buffer.cpp /tmp/my_bin
```

### 3. 一步完成“翻译 + 编译”

```bash
dacpp-translate-and-build /path/to/file.dac.cpp /tmp/my_bin
```

`dacpp-translate-and-build` 现在会优先识别当前真实生成的文件名，不再假定只能找到旧的 `_sycl_usm.cpp`。

### 4. 运行单进程程序

```bash
DYLD_LIBRARY_PATH=/Volumes/QUQ/working/sycl-install/lib /tmp/my_bin
```

### 5. 运行 MPI 程序

```bash
DYLD_LIBRARY_PATH=/Volumes/QUQ/working/sycl-install/lib mpirun -np 2 /tmp/my_bin
```

## 如何跑本地回归

`test_local.sh` 用来验证本机单进程 `--mode=buffer` 路径。

脚本位置：

```text
/Volumes/QUQ/working/dacpp/clang/tools/translator/test_local.sh
```

### 默认 smoke suite

```bash
cd /Volumes/QUQ/working/dacpp/clang/tools/translator
bash test_local.sh
```

默认样例当前包括：

- `matMul1.0`
- `FOuLa1.0`
- `decay1.0`
- `DFT1.0`
- `liuliang1.0`
- `mandel1.0`
- `gradientSum`
- `jacobi1.0`

### 扩展样例集

如果你想跑更接近“本地全量”的集合：

```bash
cd /Volumes/QUQ/working/dacpp/clang/tools/translator
INCLUDE_EXTENDED_LOCAL_TESTS=1 bash test_local.sh
```

这会额外包含：

- `imageAdjustment1.0`
- `oddeven0.1`
- `stencil1.0`
- `waveEquation1.0`
- `MDP1.0`
- `vectorAddCombo`

### 只跑指定样例

```bash
cd /Volumes/QUQ/working/dacpp/clang/tools/translator
bash test_local.sh gradientSum
bash test_local.sh oddeven0.1 stencil1.0
```

### 调整超时

默认超时是 120 秒。可以按需放宽：

```bash
cd /Volumes/QUQ/working/dacpp/clang/tools/translator
LOCAL_TEST_TIMEOUT_SEC=300 bash test_local.sh MDP1.0
```

### `test_local.sh` 的行为

脚本会：

1. 复制样例到临时目录
2. 自动挑选主 `.dac.cpp`
3. 用 `--mode=buffer` 生成 `*.dac_sycl_buffer.cpp`
4. 编译并运行生成程序
5. 优先用 `StandardSycl.cpp` 做参考实现
6. 如果 `StandardSycl.cpp` 本机不可用，则回退到 `serial.cpp`
7. 如果前两者都不可跑，最后回退到 `result.out`
8. 对输出做规范化并比对

其中还包含一层本机兼容适配，例如：

- `CL/sycl.hpp` 到本机命名空间桥接
- `gpu_selector_v -> default_selector_v`
- 过滤 AdaptiveCpp 警告，避免误报

### 本地回归产物与日志

默认输出目录：

```text
/Volumes/QUQ/working/local_tmp
```

每个样例目录下通常会有：

- `translate.log`
- `generated_compile.log`
- `generated.out`
- `reference*.out`
- `diff.log`

## 如何跑 MPI 全量回归

`test_mpi.sh` 用来验证 `--mode=buffer --mpi` 路径。

脚本位置：

```text
/Volumes/QUQ/working/dacpp/clang/tools/translator/test_mpi.sh
```

### 直接运行

```bash
cd /Volumes/QUQ/working/dacpp/clang/tools/translator
bash test_mpi.sh
```

### 当前 MPI 全量覆盖范围

当前脚本会跑 `tests/` 下这 10 个 non-stencil / 主线 MPI 回归样例：

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

### `test_mpi.sh` 的行为

对每个样例，脚本会执行：

1. 复制样例到临时目录
2. 生成单机 baseline：
   - `dacpp ... --mode=buffer`
   - `acpp-compile`
   - 运行单进程程序得到 `base.out`
3. 生成 MPI 版本：
   - `dacpp ... --mode=buffer --mpi`
   - `acpp-compile`
   - `mpirun -np 2`
4. 对 baseline 和 MPI 输出做清洗与比对

MPI 版本当前真实生成物是：

```text
*.dac_sycl_buffer.cpp
```

不再使用旧文档里提到的 `*.dac_sycl_buffer_mpi.cpp` 命名。

### MPI 回归产物与日志

默认输出目录：

```text
/Volumes/QUQ/working/mpi_tmp
```

每个样例目录下通常会有：

- `step1.log`
- `step2.log`
- `base.out`
- `mpi.out`
- `*.clean`

## 推荐的全量验证顺序

如果你刚改完 translator，建议按这个顺序做一次完整验证：

### 1. 重新编译

```bash
cd /Volumes/QUQ/working/dacpp
cmake --build build --target translator -j8
```

### 2. 本地默认回归

```bash
cd /Volumes/QUQ/working/dacpp/clang/tools/translator
bash test_local.sh
```

### 3. 本地扩展回归

```bash
cd /Volumes/QUQ/working/dacpp/clang/tools/translator
INCLUDE_EXTENDED_LOCAL_TESTS=1 bash test_local.sh
```

### 4. MPI 全量回归

```bash
cd /Volumes/QUQ/working/dacpp/clang/tools/translator
bash test_mpi.sh
```

这也是目前最接近“本机全量”的验证方式。

## 当前主线的真实情况

当前主线里，`--mpi` 仍然统一走 MPI wrapper 主链路，语义上已经稳定覆盖现有回归集。

需要特别说明的是：

- MPI 的 region 入口已经接回主线
- 但当前默认实现是一个 `region-safe` 版本
- 它优先保证编译和语义正确性
- 还没有把单机 buffer region 那套 device-resident 优化完整搬到 MPI 路径

也就是说，当前重点是“默认主线正确、测试全绿”，而不是“MPI region backend 已经做完所有性能优化”

## 近期实测结果

最近一次在这台机器上的实测结果：

### 本地默认 suite

```text
8 tests | 8 passed | 0 failed | 0 skipped
```

### 本地扩展 suite

```text
14 tests | 14 passed | 0 failed | 0 skipped
```

### MPI 全量 suite

```text
10 tests | 10 passed | 0 failed | 0 skipped
```

## 常见排障

### 1. `No such file: .../build/bin/translator/translator`

说明 `translator` 还没编出来，先执行：

```bash
cd /Volumes/QUQ/working/dacpp
cmake --build build --target translator -j8
```

### 2. `AdaptiveCpp compiler not found`

说明 `ACPP_ROOT` 没设对，或者 `sycl-install` 不完整。先检查：

```bash
echo "$ACPP_ROOT"
ls -l /Volumes/QUQ/working/sycl-install/bin/acpp
```

### 3. `mpirun` 失败，提示端口绑定或网络接口问题

如果你是在受限沙箱、CI 或容器里运行，OpenMPI 可能没有权限绑定本地端口。优先在正常终端里直接跑：

```bash
cd /Volumes/QUQ/working/dacpp/clang/tools/translator
bash test_mpi.sh
```

### 4. 生成文件名和文档不一致

当前主线以 `*.dac_sycl_buffer.cpp` 为主，不要再按旧习惯去找：

```text
*.dac_sycl_buffer_mpi.cpp
```

### 5. `test_local.sh` / `test_mpi.sh` 目录里有很多中间产物

这是脚本正常行为。统一看：

- `/Volumes/QUQ/working/local_tmp`
- `/Volumes/QUQ/working/mpi_tmp`

需要重跑时，脚本会先清自己的临时目录。

## 不再推荐的旧表述

下面这些说法已经过时，阅读旧文档或历史记录时需要自动替换理解：

- “MPI 测试走 `--mode=usm --mpi`”
  现在回归脚本实际走的是 `--mode=buffer --mpi`

- “MPI 生成文件名是 `*.dac_sycl_buffer_mpi.cpp`”
  当前主线实际生成并编译的是 `*.dac_sycl_buffer.cpp`

- “只跑 `test_local.sh` 默认 suite 就代表本地全量”
  现在更准确的说法是：
  默认 suite 是 smoke test；
  `INCLUDE_EXTENDED_LOCAL_TESTS=1 bash test_local.sh` 更接近本地全量

- “旧版 `Rewriter_Buffer.cpp` / archive 路径是当前主线”
  当前主线 buffer 路径是 `Rewriter_Buffer_new.cpp`

