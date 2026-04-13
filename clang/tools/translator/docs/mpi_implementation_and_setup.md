# DACPP Translator MPI 实现思路与环境配置指南

本文档全面梳理了目前 DACPP Translator 翻译为 MPI 多节点并行代码的核心实现思路，同时整理了如何编译、定位和运行翻译器的完整流程。

---

## 一、 环境配置与如何编译 Translator

DACPP Translator 基于 LLVM/Clang 开发，它是一个自定义的 Clang Frontend Action 工具。我们通过编译整个基于 LLVM 源码树的 Clang 体系来生成它。

### 1.1 编译与安装步骤

在 macOS 或 Linux 上，建议先配置并编译整个包含 `clang` 项目的 LLVM 环境。
如果代码已经在本地：
```bash
# 假设你在项目根目录 /Volumes/QUQ/working/dacpp 下
cd /Volumes/QUQ/working/dacpp

# 使用 CMake 配置工程
# 请确保将 CMAKE_INSTALL_PREFIX 改为你需要的路径，通常直接指定为当前环境构建路径下
cmake -S llvm -B build -G Ninja \
    -DLLVM_ENABLE_PROJECTS="clang" \
    -DCMAKE_INSTALL_PREFIX=/Volumes/QUQ/working/dacpp/clang_install \
    -DCMAKE_BUILD_TYPE=Debug

# 进入 build 目录进行编译，-j 参数根据你的 CPU 核心数调整
cd build
ninja -j8

# （可选）安装到 CMAKE_INSTALL_PREFIX 指定目录
ninja install
```

### 1.2 编译产物路径在哪？

DACPP 的特定工具位于源码树的 `dacpp/clang/tools/translator`，编译构建后生成的可执行文件（也就是真正的翻译器前端程序）的位置默认在构建目录下的 `bin` 中。

- **翻译器可执行文件路径：**  
  `/Volumes/QUQ/working/dacpp/build/bin/translator/translator`

> 注意：有时如果通过环境变量封装脚本（如 `env.sh`）调用，通常会被别名为 `dacpp` 命令。例如，在当前 macOS 测试机上加载 `env.sh` 后，输入 `dacpp <文件> --mpi` 就会自动调用到上述路径下的翻译器。

---

## 二、 翻译测试与运行流

一旦编译完成，就可以使用 `env.sh` 脚本和翻译器来测试一段 DACPP 代码，它将生成带 MPI 后端的 SYCL-USM C++ 代码，随后可以进行编译并用 `mpirun` 测试。

### 2.1 依赖环境与快捷脚本加载

使用翻译器进行 MPI 测试，首先需要在终端初始化环境。我们提供了集成化的 `env.sh` 脚本处理一切包含路径。

```bash
# 在含有工作环境的 Bash 会话中运行：
bash -lc 'source /Volumes/QUQ/working/dacpp/clang/tools/translator/env.sh && dacpp-translate-and-build /path/to/test.dac.cpp /tmp/output_bin'
```
这里 `dacpp-translate-and-build` 是一个自动化函数，背后做了两件事：
1. `dacpp /path/to/test.dac.cpp --mpi`  -> 生成 `test.dac_sycl_usm.cpp`
2. `acpp-compile test.dac_sycl_usm.cpp /tmp/output_bin` -> 生成可被 MPI 调用的 `output_bin`

### 2.2 运行生成的 MPI 程序

使用 `mpirun` 或者 `mpiexec` 直接调用即可。由于这是多节点的异构加速程序，需要传入底层 SYCL 实现（AdaptiveCpp）动态库路径：

```bash
bash -lc 'DYLD_LIBRARY_PATH=/Volumes/QUQ/working/sycl-install/lib mpirun -np 2 /tmp/output_bin'
```

---

## 三、 当前 MPI 翻译器的底层实现思路

翻译器核心通过 `Rewriter_MPI.cpp` 实现 `--mpi` 标志的逻辑，主要将以 `<->` 符号绑定的 `Shell`（逻辑拓扑、读写属性）和 `Calc`（具体数学计算）展开成适合 **Scatter-Compute-Gather** 的多节点运行代码。

### 3.1 总体执行架构

MPI 翻译器并不是盲目对所有数据复制。它会为每个具有绑定表达式的语句生成一个自动的 `Wrapper 函数`（形如 `ShellName_CalcName`）。整个计算流程分为三个部分：

1. **主节点切分分发（Scatter/Pack）**：
   - 所有的输入 Tensor（`Vector` 或 `Matrix`）最初都位于 Rank 0 节点。
   - Rank 0 利用原有的形如 `A[i]` (IndexSplit) 等绑定信息生成相应的 `AccessPattern`。
   - 翻译后的代码通过底层的 `dacpp::mpi::pack_values_by_globals` 结合 `build_input_pack_map` 将全局数据按照各节点需要计算的 Item 提取对应的局部区块，并调用 `MPI_Send` 散发给对应的 Worker Rank。
   - Worker 节点通过 `MPI_Recv` 将属于自己计算范围的切片接收到 `local_A` 之类的一维 `vector` 中。

2. **从节点并行计算（Compute via SYCL）**：
   - 每个 Rank（包括 Rank 0）都构建属于自身的 `sycl::buffer`，并在本机的计算队列中提交 `q.submit` 进行并行计算。
   - 底层的 `local` 数据被包装成如 `dacpp::mpi::View1D` 或者 `dacpp::mpi::View2D` 对象。这一步是为了掩盖因为分布式存储导致的数据碎片和偏移差异。
   - 从原先 `Calc` 提取的代码块将无缝地接收这些 `View`，像访问普通数组一样利用重载的 `operator[]` 处理局部运算 `vadd_mpi_local`。

3. **结果脏数据汇集回主节点（Gather & Writeback）**：
   - 若参数标志具有 `Write` 或者 `ReadWrite` 属性（如 `Vector<int>&` 非 const 修饰），翻译器会自动开启脏数据写回。
   - 各个 Worker 会通过 `dacpp::mpi::build_writeback_values` 和对应的目标序号 `writeback_globals` 向 Rank 0 传送改动过的数据元素（`MPI_Send`）。
   - Rank 0 利用 `apply_writeback_by_globals` 依次接收各个 Worker 修改的数据并写回全局唯一的存储空间中。
   - 最终调用 `MPI_Bcast` 向全网广播被更新以后的数据（根据具体 `--mpi-output-sync=all-ranks` 选项来决定是否同步到其余所有 Rank）。

### 3.2 细节优化与稳健性设计

1. **自动清理环境与过滤控制台**：在被替换的 `main` 开头和结尾自动安插了 `MPI_Init` 与 `MPI_Finalize`，其中包含了防止同个进程因为 `return` 被错误安插多次而导致 Segment Fault 的防御检查 `dacpp_mpi_finalize_needed = 0;`。并且，仅放行 `mpi_rank == 0` 的 `stdout` 标准输出，避免用户程序的 `printf` 产生刷屏轰炸。
2. **跨平台的结构体通信（字节传递）**：如果计算中使用了普通标量（如 `float`, `int`），重写器直接映射到标准 MPI 格式 `MPI_FLOAT`。如果是诸如自定义的结构体 `Pixel`，则自动退回到 `MPI_BYTE` 模式，并自动编译生成类似 `count * sizeof(Pixel)` 的真实传输量计算公式，解决了早期元素数目失真的问题。
3. **同名 Wrapper 安全**：通过翻译器级的全局记录簿（`std::set<std::string> generatedWrappers`），有效拦截并忽略了同一代码里多次调用同一绑定表达式（如两个 `VADD <-> vadd`），避免生成的 C++ 文件因重新定义发生编译报错（redefinition error）。
