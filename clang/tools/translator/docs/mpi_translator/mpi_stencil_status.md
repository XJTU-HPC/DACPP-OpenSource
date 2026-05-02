# MPI Stencil Loop Path Status

更新时间：2026-05-02

## 1. 目标声明

这条路径的目标是为 MPI 下的 stencil / time-step 类循环单独生成代码：

- 输入使用 `--mode=buffer --mpi`
- 源码中存在位于 `for` / `while` 内部的 `<->`
- shell 实参在外层循环之前已经声明，循环内每次迭代复用这些对象

目标生成形态是：

```cpp
__dacpp_mpi_stencil_ctx_xxx ctx;
__dacpp_mpi_stencil_init_xxx(ctx, ...);
for (...) {
    __dacpp_mpi_stencil_run_xxx(ctx, ...);
}
```

这样可以把稳定不变的 pattern / binding / pack plan / rank range 初始化从每次迭代中提出来，循环内只执行每步真正需要重复的 scatter、kernel、gather、writeback、broadcast。

明确不在这条路径优化范围内的场景：

- shell 实参是在循环体内部临时声明的 view / tensor
- 每次迭代 shell 实参的形状或绑定语义会变化
- 不能安全把 `init(ctx, args...)` 提到外层循环之前的 `<->`

这类场景会回退到普通 MPI wrapper 路径，保持原有语义优先。

## 2. 当前设计

当前采用 `ctx + init + run + compatibility wrapper` 结构：

- `ctx`
  - 保存 MPI rank / size、SYCL queue、AccessPattern、PackPlan、ItemRange 等稳定状态
- `init`
  - 在循环外执行一次
  - 初始化 pattern、binding split sizes、pack plan、本 rank item range
- `run`
  - 在每次循环迭代执行
  - 完成数据分发、本地 kernel、写回收集和必要 broadcast
- compatibility wrapper
  - 对非 loop stencil site 保留一次性调用形式
  - 内部执行 `ctx; init(ctx, ...); run(ctx, ...);`

## 3. 已完成实现

### 3.1 AST 侧记录 MPI stencil site

涉及文件：

- `clang/tools/translator/parser/include/DacppStructure.h`
- `clang/tools/translator/translator.cpp`

已加入 `MpiStencilSite`，记录：

- `<->` 的表达式编号
- 当前 `BinaryOperator* dacExpr`
- 包裹 `<->` 的外层 loop

当前登记条件已经收紧：

- `<->` 必须位于 `for` / `while` 中
- shell 调用实参引用的非全局变量必须声明在外层 loop 之前

这样可以避免把循环体内部临时变量错误 hoist 到循环外。

### 3.2 MPI 总入口分流

涉及文件：

- `clang/tools/translator/rewriter/lib/Rewriter_MPI.cpp`

逻辑：

```cpp
if (dacppFile && dacppFile->hasMPIStencilSites()) {
    rewriteMPIStencil();
    return;
}
```

普通 MPI wrapper 路径仍然保留；只有安全登记为 stencil site 的 loop `<->` 才进入新路径。

### 3.3 新增 stencil rewriter 文件

新增或接入文件：

- `clang/tools/translator/rewriter/include/Rewriter_MPI_Stencil_Common.h`
- `clang/tools/translator/rewriter/lib/Rewriter_MPI_Stencil.cpp`
- `clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Stencil_Analysis.cpp`
- `clang/tools/translator/rewriter/lib/mpi/Rewriter_MPI_Stencil_Codegen.cpp`
- `clang/tools/translator/rewriter/include/Rewriter.h`
- `clang/tools/translator/CMakeLists.txt`

`rewriteMPIStencil()` 当前负责：

1. 复用普通 MPI prelude
2. 生成 local calc
3. 生成 stencil ctx / init / run / compatibility wrapper
4. 删除原 shell / calc 声明
5. 在 `main` 中插入 MPI init / finalize
6. 在 loop 前插入 ctx/init
7. 将 loop 内 `<->` 替换为 `run(ctx, ...)`
8. 设置 `mainAlreadyRewritten`，避免通用 `rewriteMain()` 二次替换

## 4. 本轮修复

### 4.1 修复错误 hoist

问题：

`FOuLa1.0` 中 `<->` 位于循环内，但 shell 实参 `u_kin`、`u_kout`、`r` 都是在循环体内部临时声明的。旧逻辑只要看到 `<->` 在 loop 中就走 stencil 路径，导致生成：

```cpp
__dacpp_mpi_stencil_init_PDE_pde(ctx, u_kin, u_kout, r);
for (...) {
    dacpp::Vector<double> u_kout = ...;
    dacpp::Vector<double> r(...);
    dacpp::Vector<double> u_kin = ...;
    __dacpp_mpi_stencil_run_PDE_pde(ctx, u_kin, u_kout, r);
}
```

`init` 被插到变量声明之前，生成代码编译失败。

修复：

- 在登记 `MpiStencilSite` 前检查 shell 实参引用的变量声明位置
- 如果变量不是全局变量，且声明位置不在外层 loop 之前，则不登记 stencil site
- 该 `<->` 回退普通 MPI wrapper

### 4.2 修复 MPI 写回 gather 值错位风险

问题：

旧代码用：

```cpp
MPI_Gatherv(local_xxx.data(), send_count_xxx, ...);
```

但 `send_count_xxx` 对应的是 `writeback_globals` 数量，不一定等于 `local_xxx` 前缀中需要写回的元素。对于 compact pack / writeback globals 不等于 pack globals 前缀的情况，会把错误 slot 的值 gather 回 root。

修复：

普通 MPI wrapper 和 stencil run 都改为：

```cpp
std::vector<T> writeback_values_xxx =
    dacpp::mpi::build_writeback_values_parallel(local_xxx, pack_xxx);
MPI_Gatherv(writeback_values_xxx.data(), send_count_xxx, ...);
```

这样 gather 的值和 `writeback_globals` 一一对应。

## 5. 当前验证结果

已完成构建：

```bash
cd /Volumes/QUQ/working/dacpp
cmake --build build --target translator -j8
```

结果：通过。构建中仍有既有 Clang/本项目 warning，但没有新增编译错误。

重点回归：

```bash
cd /Volumes/QUQ/working/dacpp/clang/tools/translator
bash test_mpi.sh FOuLa1.0 stencil1.0 waveEquation1.0
```

结果：

```text
3 tests | 3 passed | 0 failed | 0 skipped
```

完整 MPI 回归：

```bash
cd /Volumes/QUQ/working/dacpp/clang/tools/translator
bash test_mpi.sh
```

结果：

```text
13 tests | 13 passed | 0 failed | 0 skipped
```

覆盖到的关键用例：

- `mpiDenseCoverSibling1.0`
- `FOuLa1.0`
- `stencil1.0`
- `waveEquation1.0`
- 以及普通 MPI 主线 case

## 6. 当前状态判断

可以认为当前已经完成：

- MPI stencil 路径接线
- ctx / init / run codegen 第一版
- loop stencil site 安全分流
- 普通 MPI 和 stencil MPI 写回值 gather 修复
- `translator` 构建验证
- `test_mpi.sh` 完整回归验证

当前不能宣称的是：

- 已支持所有 loop 内 `<->` 的 hoist 优化
- 已支持循环体内部临时 view 的跨迭代 ctx 复用
- 已完成更精细的 stencil 语义分类

## 7. 后续风险和建议

剩余风险：

1. 多个 `<->` 位于同一个 loop 时，ctx/init 插入顺序和作用域还需要专门验证。
2. 嵌套 loop 下，目前使用外层 loop 做 hoist 点，后续可能需要根据实参生命周期选择更精确的 loop。
3. `Rewriter_MPI_Stencil_Analysis.cpp` 目前仍偏占位，真正的 stencil 分类逻辑还可以继续下沉到这里。
4. 当前分流以“能否安全 hoist”为主，不等价于完整 stencil 语义识别。

建议下一步：

1. 增加专门覆盖多个 loop `<->` 的 MPI 测试。
2. 增加嵌套 loop 的 stencil/非 stencil 对照测试。
3. 把当前实参生命周期检查整理进 stencil analysis 模块。
4. 如果要支持循环内临时 view 的 stencil 优化，需要设计按迭代更新 pattern/plan 或局部 init 的新策略。

## 8. 一句话总结

当前 MPI stencil 路径已经从“骨架接好但未闭环”推进到“可构建、可回归、普通 MPI 不被误伤”的状态；本轮关键修复是收紧 hoist 条件，并修正写回 gather 的值与 `writeback_globals` 对齐问题。
