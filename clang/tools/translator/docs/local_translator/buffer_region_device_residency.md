# Buffer Region Device Residency

## 背景

在单机 `buffer` 模式下，原来的翻译结果会把每次 `shell <-> calc` 都展开成一整套：

- `queue` 初始化
- `ParameterGeneration`
- `DataInfo`
- `split/index`
- `buffer` 构造
- `tensor2Array`
- H2D
- kernel submit
- D2H
- `array2Tensor`

这在 `waveEquation`、`stencil` 这类时间迭代问题里代价很高，因为相同的数据和算子会在每个时间步反复初始化、反复搬运。

本次修改的目标是：

- 在可识别的时间迭代 region 内，把数据尽量常驻在设备上
- 把初始化外提到循环前
- 把同步压到循环后
- 把 `<->` 后面可识别的 sibling loops 一起设备化
- 在保证安全回退的前提下，减少不必要的 H2D / D2H / `wait()`

## 适用范围

当前实现只覆盖 v1 范围：

- 只考虑单机
- 只考虑 `buffer` 模式
- 只优化“单个外层时间循环 + 单个主 `<->`”
- 外层循环仍然保留在 host
- `<->` 后面只接受可识别的 sibling `for` loops

只要不满足条件，就会回退到原有 buffer wrapper 路径。

## 这次修改了哪些文件

### AST / plan 分析

- `clang/tools/translator/parser/include/DacppStructure.h`
- `clang/tools/translator/parser/lib/DacppStructure.cpp`
- `clang/tools/translator/translator.cpp`

这里新增了 `BufferRegionPlan`，用于描述“可优化时间迭代区域”。

`BufferRegionPlan` 当前记录：

- 外层 `ForStmt`
- 唯一主 `<->`
- `<->` 后面的 sibling `for` 列表
- 捕获变量
- 是否启用优化
- 禁用原因

`translator.cpp` 在 `buffer` 模式下会先跑 region 分析，并打印：

- `buffer region optimization enabled`
- 或 `buffer region optimization skipped: ...`

### 重写与代码生成

- `clang/tools/translator/rewriter/include/Rewriter.h`
- `clang/tools/translator/rewriter/lib/Rewriter_Buffer_new.cpp`

这里完成了 region 路径的主要生成逻辑：

- 生成 `__dacpp_ctx_*`
- 生成 `__dacpp_init_*`
- 生成 `__dacpp_submit_*`
- 生成 `__dacpp_submit_region_*`
- 生成 `__dacpp_sync_*`
- 在调用点插入 `init`
- 用 submit helper 替换循环体中的 `<->` 和 sibling loops
- 在循环后插入 `sync`

`rewriteMain()` 在命中 region plan 时直接跳过，避免旧主路径和新路径重复改写同一段循环。

### 兼容性修复

- `clang/tools/translator/rewriter/include/dacInfo.h`

`ctx` 里会持有 `RegularSlice` / `Index` 成员，所以补了默认构造函数，避免 region 生成代码链接失败。

## region 检测逻辑

当前 AST 侧的 region 检测规则是：

1. 文件里必须只有一个 `<->`
2. 外层时间循环里必须只有一个 `<->`
3. 外层循环体必须是 `CompoundStmt`
4. `<->` 必须是外层循环里的第一个顶层 statement
5. `<->` 后面只允许 sibling `for` 语句
6. sibling loop 里不允许：
   - `<->`
   - `if / while / switch`
   - `return / break / continue / goto`
7. sibling loop 不允许使用非 shell captured state

如果不满足，就不会进入 region 优化。

## 生成代码结构

优化后的翻译结果会把原来的“单次大 wrapper”拆成四层：

### 1. `__dacpp_ctx_<shell>_<calc>`

持有 region 生命周期内的常驻状态，包括：

- `sycl::queue`
- `ParameterGeneration`
- `DataInfo`
- `split/index`
- `Dac_Ops`
- `info_partition_*`
- launch geometry
- host shadow `std::vector<T>`
- persistent `sycl::buffer<T,1>`

### 2. `__dacpp_init_<shell>_<calc>(ctx, args...)`

只在循环前执行一次。

负责：

- 初始化 queue
- 初始化 `DataInfo`
- 初始化 split/index
- 初始化 `Dac_Ops`
- 初始化 `info_partition_*`
- 计算 work-item / launch geometry
- 构造 persistent buffer
- 只对真正需要的参数做首轮 H2D

### 3. `__dacpp_submit_<shell>_<calc>(ctx)`

每个时间步只提交主 `<->` 对应的 kernel。

负责：

- accessor 构造
- kernel submit

### 4. `__dacpp_submit_region_*`

每个可识别 sibling loop 对应一个 helper。

负责：

- accessor 构造
- loop device 化后的 kernel submit

### 5. `__dacpp_sync_<shell>_<calc>(ctx, args...)`

只在循环后执行一次。

负责：

- 单次 `queue.wait()`
- 只对需要回写的参数做 D2H
- `array2Tensor(...)`

## 这次新增的两项优化

### 优化 1：移除 region 内每步 `wait()`

之前 region 路径里虽然已经做到了 buffer 常驻，但每次：

- 主 `<->` submit 后 `wait()`
- 每个 sibling helper 后再 `wait()`

这样会把一整个时间步重新串行化，host 每一步都要阻塞。

现在的逻辑改成：

- `submit` 内不再 `.wait()`
- `submit_region_*` 内不再 `dacpp_q.wait()`
- 只在 `sync()` 里统一做一次 `ctx.dacpp_q.wait()`

这里依赖的是同一组 persistent `sycl::buffer` 带来的访问依赖：

- 后续 kernel 对同一 buffer 的访问会自动形成依赖
- host 只在 region 末尾需要一次显式同步

这减少了每一步的 host-device 同步开销。

### 优化 2：按真实读写集裁剪 H2D / D2H

之前 region 路径虽然把搬运挪到了循环前后，但仍然是偏保守的：

- 所有参数循环前都做 `tensor2Array + H2D`
- 所有写参数循环后都做 `D2H + array2Tensor`

现在新增了 `BufferRegionTransferPolicy`，专门分析每个参数的：

- `needsInitCopy`
- `needsSyncCopy`
- `writtenInRegion`
- `hostUsedAfterLoop`

### `needsInitCopy` 的判定

按 region 中的执行顺序分析：

1. 主 `<->` kernel
2. 后续 sibling loops

规则是：

- 如果某个参数在“第一次写入之前”就被读取过，则 `needsInitCopy = true`
- 否则就跳过初始 H2D，直接在设备上构造空 buffer

这等价于“read-before-first-write 需要首轮 H2D”。

例子：

- `waveEquation`
  - `matCur`：主 kernel 先读，所以要 H2D
  - `matPrev`：主 kernel 先读，所以要 H2D
  - `matNext`：主 kernel先写，后续 sibling loop 才读，所以不需要首轮 H2D

- `stencil`
  - `matIn`：先读，所以要 H2D
  - `matOut`：先写，所以不需要首轮 H2D

### `needsSyncCopy` 的判定

当前实现不是简单地“只要写过就回写”，而是：

- 该参数在 region 里必须真的被写过
- 并且满足以下之一：
  - 无法证明它是一个纯本地 dead local
  - 或者循环后 host 还会继续使用它

为了做到这一点，当前实现会：

1. 找到 shell call 对应的实参变量
2. 对本地变量递归追踪其初始化依赖
   - 例如 view 变量会追到其底层 tensor / vector
3. 分析外层时间循环之后的 statements
4. 只要这些后续 statements 使用了该参数或其底层依赖，就保留 D2H

这一步让出口同步从“所有写参数”收紧成“写过且 host 后面还会用的参数”。

例子：

- `waveEquation`
  - `matCur`：循环后会 `print()`，所以要 D2H
  - `matPrev`：后续 host 不再使用，所以不回写
  - `matNext`：后续 host 不再使用，所以不回写

- `stencil`
  - `matIn`：循环后 host 继续使用，所以要 D2H
  - `matOut`：循环后 host 不再使用，所以不回写

## 当前的内存搬运逻辑

这是现在 region 路径的完整搬运策略。

### 进入 region 前

一定会做：

- queue 初始化
- `DataInfo`
- split/index 初始化
- `Dac_Ops`
- `info_partition_*`
- launch geometry 计算
- persistent buffer 构造

只对 `needsInitCopy == true` 的参数做：

- `tensor2Array`
- 用 host shadow 构造 buffer

对 `needsInitCopy == false` 的参数：

- 直接用 `sycl::buffer(range)` 构造空 buffer
- 不做 `tensor2Array`
- 不做 H2D

### region 内每个时间步

只做：

- accessor 构造
- kernel submit

不再做：

- queue 重建
- buffer 重建
- `tensor2Array`
- `array2Tensor`
- 每步 `wait()`

### region 结束后

只做一次：

- `ctx.dacpp_q.wait()`

然后只对 `needsSyncCopy == true` 的参数做：

- `host_accessor`
- D2H
- `array2Tensor`

对 `needsSyncCopy == false` 的参数不做任何 host 回写。

## 当前行为总结

以 `waveEquation` 为例，现在实际生成的逻辑是：

- `matCur`
  - 入口 H2D：有
  - 循环内常驻：有
  - 出口 D2H：有

- `matPrev`
  - 入口 H2D：有
  - 循环内常驻：有
  - 出口 D2H：无

- `matNext`
  - 入口 H2D：无
  - 循环内常驻：有
  - 出口 D2H：无

并且：

- 主 `<->` 不再每步 `wait()`
- sibling helper 不再每步 `wait()`
- 只在 `sync()` 末尾统一等待

## 当前限制

这版仍然是 v1，实现上有几个明确边界：

- 只支持单个外层时间循环
- 只支持单个主 `<->`
- sibling 只支持可识别的 `for` loops
- 如果 shell call 实参不是简单变量，传输裁剪会退回更保守的策略
- 当前 host liveness 只分析“外层循环之后的本函数内语句”

也就是说，这一版已经把最主要的性能热点先拿掉，但还没有做更激进的跨语句别名 / 生命周期证明。

## 测试结果

本次修改后，按 `test_local.sh` 全量单机测试重新回归：

- 命令：
  - `INCLUDE_EXTENDED_LOCAL_TESTS=1 LOCAL_TEST_TIMEOUT_SEC=300 bash test_local.sh`
- 结果：
  - `14 tests | 14 passed | 0 failed | 0 skipped`

重点样例：

- `waveEquation1.0`：通过
- `stencil1.0`：通过
- `MDP1.0`：通过
- `vectorAddCombo`：通过

## 后续可继续优化的方向

如果下一步继续做，可以优先考虑：

1. sibling loop 访问模式分析再细化
   - 让 helper accessor 也按真实读写集收紧

2. host liveness / alias 分析继续增强
   - 让 D2H 裁剪覆盖更多“局部 view -> 底层 tensor -> 底层 vector”场景

3. 跨多个 `<->` 或更复杂 region 的推广
   - 当前仍然是单 `<->` v1
