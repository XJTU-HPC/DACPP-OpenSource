# imageAdjustment1.0 修复

这份文档只记录 `tests/imageAdjustment1.0/imageAdjustment.dac.cpp` 的修复点。

## 1. 原始问题

原来的样例有 3 个问题：

1. `image2` 传给 `dacpp::Matrix` 后又被继续复用
2. MPI 下所有 rank 都会读 `stdin`
3. `Pixel` 这类结构体走 `MPI_BYTE` 时，传输 count 按元素数算错了

## 2. 最核心的 bug

原始代码里有这一段：

```cpp
dacpp::Matrix<Pixel> image_tensor2({height, width}, image2);
...
std::vector<Pixel> image3 = image2;
dacpp::Tensor<Pixel, 2> image_tensor3({height, width}, image3);
```

`ReconTensor` 这里不是拷贝 `vector`，而是直接接管所有权。所以 `image_tensor2` 构造完以后，`image2` 已经是 moved-from 状态，再拿它去构造 `image3` 会导致：

```text
The number of elements in the vector does not correspond to the Shape.
```

## 3. 修复方式

### 3.1 不再复用 moved-from 的 `image2`

现在改成：

```cpp
std::vector<Pixel> image3(height * width, {0, 0, 0});
dacpp::Tensor<Pixel, 2> image_tensor3({height, width}, image3);
```

含义很简单：

- `image_tensor2` 有自己的底层存储
- `image_tensor3` 也有自己的底层存储

### 3.2 固定图像尺寸

原来 `main` 里会读：

```cpp
std::cin >> width;
std::cin >> height;
```

MPI 下每个 rank 都会执行 `main`，所以这里会抢 `stdin`。现在直接固定成：

```cpp
const int width = 4;
const int height = 4;
```

### 3.3 `MPI_BYTE` 按字节数传输

`Pixel` 没有专门的 MPI datatype，所以当前重写器退回到 `MPI_BYTE`。这时 `count` 必须是字节数，不是元素数。

现在重写器已经按下面的语义生成代码：

```cpp
MPI_Send(data, count * sizeof(Pixel), MPI_BYTE, ...);
MPI_Recv(data, count * sizeof(Pixel), MPI_BYTE, ...);
MPI_Bcast(data, count * sizeof(Pixel), MPI_BYTE, ...);
```

## 4. 为什么串行参考实现没问题

`imageAdjustment.serial.cpp` 不会触发这个 bug，因为它：

- 用的是 `std::vector<std::vector<Pixel>>`
- 没经过 `ReconTensor` 的 vector 所有权接管
- 没有 MPI 多 rank 读输入的问题

## 5. 当前结果

修复后，`imageAdjustment1.0` 的 MPI 版本已经跑通，输出结果是：

```text
Image After Brightness Enhancement:
{{(180, 30, 30), (180, 30, 30), (180, 30, 30), (180, 30, 30)}, {(180, 30, 30), (180, 30, 30), (180, 30, 30), (180, 30, 30)}, {(180, 30, 30), (180, 30, 30), (180, 30, 30), (180, 30, 30)}, {(180, 30, 30), (180, 30, 30), (180, 30, 30), (180, 30, 30)}}
```

最近一轮 non-stencil MPI 回归日志目录：

```text
/tmp/dacpp-mpi-final.gqTD5X
```
