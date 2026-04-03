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

## 当前测试状态

`--mpi` 的 non-stencil 样例已全部通过，最近一轮日志目录：

```text
/tmp/dacpp-mpi-final.gqTD5X
```
