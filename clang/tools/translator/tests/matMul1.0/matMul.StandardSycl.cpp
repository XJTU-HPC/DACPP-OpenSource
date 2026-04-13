#include <sycl/sycl.hpp>
#include <vector>
#include <iostream>

using namespace sycl;

int main() {
    constexpr int M = 4, K = 5, N = 4;

    // 矩阵 A (4×5) 按行存储
    std::vector<int> dataA{
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20
    };

    // 矩阵 B (5×4) 按列存储
    std::vector<int> dataB{
        1, 5, 9, 13,
        17, 2, 6, 10,
        14, 18, 3, 7,
        11, 15, 19, 4,
        8, 12, 16, 20
    };

    // 结果矩阵 C (4×4)
    std::vector<int> result(M * N, 0);

    {
        queue q;  // 创建 SYCL 队列

        // 分配 SYCL 设备内存
        buffer<int, 1> bufA(dataA.data(), range<1>(M * K));
        buffer<int, 1> bufB(dataB.data(), range<1>(K * N));
        buffer<int, 1> bufC(result.data(), range<1>(M * N));

        q.submit([&](handler& h) {
            accessor a(bufA, h, read_only);
            accessor b(bufB, h, read_only);
            accessor c(bufC, h, write_only, no_init);

            h.parallel_for(range<2>(M, N), [=](id<2> idx) {
                int i = idx[0];
                int j = idx[1];
                int sum = 0;

                for (int k = 0; k < K; k++) {
                    sum += a[i * K + k] * b[k * N + j];
                }

                c[i * N + j] = sum;
            });
        });

        q.wait();  // 等待计算完成
    }

    // 格式化输出：{{a, b, c, d}, {e, f, g, h}, ...}
    std::cout << "{";
    for (int i = 0; i < M; i++) {
        std::cout << "{";
        for (int j = 0; j < N; j++) {
            std::cout << result[i * N + j];
            if (j < N - 1) std::cout << ", ";
        }
        std::cout << "}";
        if (i < M - 1) std::cout << ", ";
    }
    std::cout << "}" << std::endl;

    return 0;
}
