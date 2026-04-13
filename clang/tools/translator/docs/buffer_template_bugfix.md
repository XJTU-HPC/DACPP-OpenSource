# Buffer Template Bug Fix Analysis

## 背景

在 MPI 测试集中，`vectorAddCombo` 和 `imageAdjustment1.0` 一度被归类为 MPI 翻译失败，但进一步对比生成代码后可以确认，这两个用例的根因并不在 MPI 语义，而是在 buffer 路径的基线翻译本身就生成错了代码。

修复前的整体表现是：

- MPI 测试总计 10 项
- 通过 6 项
- 失败 4 项

其中 `vectorAddCombo` 和 `imageAdjustment1.0` 属于“buffer 基线错误导致的假失败”。

## Bug 1: `clacparams` 跨表达式污染

问题位置：

- [Rewriter_Buffer_new.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/Rewriter_Buffer_new.cpp)

### 现象

buffer 重写阶段在处理多个数据关联表达式时，参数收集使用了文件级全局状态。前一个表达式留下来的 `clacparams` 会继续影响后一个表达式，导致 `calc->getBody(count, clacparams)` 获取到的参数映射不再只属于当前表达式。

这会直接影响生成出的下标替换结果，典型表现是：

- `vectorAddCombo` 中局部视图变量和真实 tensor 维度/偏移映射错位
- `imageAdjustment1.0` 中输出 tensor 的下标展开混入前序表达式的参数信息

### 根因

`Rewriter_Buffer_new.cpp` 原先维护了文件级全局变量：

- `std::vector<int> dim;`
- `std::vector<dacppTranslator::clacparam> clacparams;`

这些状态没有按单个表达式隔离，导致重写逻辑不是纯局部的。

### 修复

把参数收集改为“每个表达式单独构造局部 `calcParams`”，再传给：

- `calc->getBody(count, calcParams)`

这样每一次 body 生成都只依赖当前表达式对应的 `Shell/Calc` 信息，不会被前一次翻译污染。

## Bug 2: buffer 路径误调用 USM 包装层模板

问题位置：

- [Rewriter_Buffer_new.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/Rewriter_Buffer_new.cpp)
- [buffer_template_new.cpp](/Volumes/QUQ/working/dacpp/clang/tools/translator/rewriter/lib/buffer_template_new.cpp)

### 现象

buffer 翻译路径在生成外层包装代码时，错误调用了：

```cpp
USM_TEMPLATE::CodeGen_DAC2SYCL2(...)
```

而这里实际应该走 buffer 专用实现：

```cpp
BUFFER_TEMPLATE::CodeGen_DAC2SYCL2(...)
```

这会让 buffer 翻译结果混入 USM 模板的包装逻辑，出现队列变量、接口签名和运行时约定不一致的问题。

### 根因

这是一次模板分流错误：buffer 重写器进入了错误的代码生成分支。

同时，`buffer_template_new.cpp` 的实现签名与调用侧也存在不一致，调用实际只传 6 个参数，但实现仍保留旧的 7 参数版本。

### 修复

做了两处同步调整：

1. 在 `Rewriter_Buffer_new.cpp` 中把外层包装生成入口改回：
   - `BUFFER_TEMPLATE::CodeGen_DAC2SYCL2(...)`
2. 在 `buffer_template_new.cpp` 中把实现签名调整为和调用方一致的 6 参数版本，去掉多余的 `MemFree` 参数

## 修复后的影响

修复后重新检查生成代码，可以看到这两个用例的 buffer 基线已经恢复正常：

- `vectorAddCombo` 能正确生成 `vshift` 的逐元素访问表达式
- `imageAdjustment1.0` 能正确生成 `image_tensor3` 的多维下标写入

对应地，MPI 测试集结果从：

- `10 tests | 6 passed | 4 failed`

提升为：

- `10 tests | 8 passed | 2 failed`

## 结论

`vectorAddCombo` 和 `imageAdjustment1.0` 的失败不属于 MPI `broadcast`/`binding` 语义错误，而是 buffer 基线翻译错误造成的连带失败。

这次修复之后，MPI 方向剩余的真实问题主要收敛到了：

- `FOuLa1.0`
- `MDP1.0`

它们更像是 `binding()` 语义在 MPI 重写阶段没有被正确保留下来，后续应继续在 `Rewriter_MPI.cpp` 的绑定元数据生成逻辑上修复。
