# DACPP Translator

`clang/tools/translator` 是当前 DACPP 到 SYCL 的 source-to-source translator 主目录。
这份 README 只保留当前主线仍然有效、且和本机脚本一致的入口；更完整的环境说明与排障请看 `docs/macos_local_sycl_setup.md`。

## 当前推荐流程

先加载本机开发环境：

```bash
source /Volumes/QUQ/working/dacpp/clang/tools/translator/env.sh
```

如果你修改了 `translator.cpp`、`parser/`、`rewriter/` 或模板生成代码，先重新编译：

```bash
cd /Volumes/QUQ/working/dacpp
cmake --build build --target translator -j8
```

当前二进制位置：

```text
/Volumes/QUQ/working/dacpp/build/bin/translator/translator
```

## 单文件常用命令

本地 buffer 翻译：

```bash
dacpp /path/to/file.dac.cpp --mode=buffer
```

MPI 翻译：

```bash
dacpp /path/to/file.dac.cpp --mode=buffer --mpi
```

当前主线生成物默认是：

```text
*.dac_sycl_buffer.cpp
```

编译生成文件：

```bash
acpp-compile /path/to/file.dac_sycl_buffer.cpp /tmp/my_bin
```

## 回归测试入口

本地 smoke suite：

```bash
cd /Volumes/QUQ/working/dacpp/clang/tools/translator
bash test_local.sh
```

本地扩展回归：

```bash
cd /Volumes/QUQ/working/dacpp/clang/tools/translator
INCLUDE_EXTENDED_LOCAL_TESTS=1 bash test_local.sh
```

MPI 主线回归：

```bash
cd /Volumes/QUQ/working/dacpp/clang/tools/translator
bash test_mpi.sh
```

默认日志目录：

- `/Volumes/QUQ/working/local_tmp`
- `/Volumes/QUQ/working/mpi_tmp`

## 当前主线代码入口

如果你要理解现在真正生效的实现，建议先看：

- `translator.cpp`
- `parser/lib/DacppStructure.cpp`
- `parser/lib/Shell.cpp`
- `rewriter/include/Rewriter.h`
- `rewriter/lib/Rewriter_Buffer_new.cpp`
- `rewriter/lib/buffer_template_new.cpp`
- `rewriter/lib/Rewriter_MPI.cpp`
- `dpcppLib/include/MPIPlanner.h`

`archive/` 和 `docs/mpi_translator/archive/` 只保留历史材料，不代表当前主线。

## 推荐文档

- `docs/macos_local_sycl_setup.md`
- `docs/local_translator/buffer_region_device_residency.md`
- `docs/mpi_translator/mpi_current_translation_logic.md`
- `archive/README.md`

## 不要再按旧文档理解

下面这些表述已经过时：

- MPI 回归脚本当前走的是 `--mode=buffer --mpi`，不是 `--mode=usm --mpi`
- MPI 主线生成并编译的是 `*.dac_sycl_buffer.cpp`，不是 `*.dac_sycl_buffer_mpi.cpp`
- `bash test_local.sh` 默认是本地 smoke suite；`INCLUDE_EXTENDED_LOCAL_TESTS=1 bash test_local.sh` 才更接近本地全量
