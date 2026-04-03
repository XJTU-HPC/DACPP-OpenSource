# Translator Docs

`docs/` 里现在保留 3 份主文档：

- [mpi_nonstencil_translation.md](/Volumes/QUQ/working/dacpp/clang/tools/translator/docs/mpi_nonstencil_translation.md)
  说明当前 `translator --mpi` 的核心逻辑、关键代码位置、运行流程和测试状态。
- [macos_local_sycl_setup.md](/Volumes/QUQ/working/dacpp/clang/tools/translator/docs/macos_local_sycl_setup.md)
  说明这台 macOS 机器上怎么直接跑通翻译、编译和 MPI 执行。
- [image_adjustment_fix.md](/Volumes/QUQ/working/dacpp/clang/tools/translator/docs/image_adjustment_fix.md)
  记录 `imageAdjustment1.0` 的问题、修复点，以及为什么串行参考实现不会爆。

建议阅读顺序：

1. 先看 `macos_local_sycl_setup.md`
2. 再看 `mpi_nonstencil_translation.md`
3. 具体 case 再看 `image_adjustment_fix.md`
