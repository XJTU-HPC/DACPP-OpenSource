#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

using namespace sycl;

const int N = 1024;

// 一个 kernel：比较 array[i] 和 array[i+1]，必要时交换
void oddEvenSortSYCL(std::vector<int> &arr) {
    queue q;

    // 创建 SYCL 缓冲
    buffer<int> buf(arr.data(), range<1>(N));

    for (int phase = 0; phase < N; phase++) {

        // 偶数比较： (0,1), (2,3) ...
        // 奇数比较： (1,2), (3,4) ...
        int start = (phase % 2 == 0 ? 0 : 1);

        q.submit([&](handler &h) {
            auto acc = buf.get_access<access::mode::read_write>(h);

            h.parallel_for(range<1>((N - 1) / 2 + 1), [=](id<1> idx) {
                int i = start + idx[0] * 2;
                if (i + 1 < N) {
                    if (acc[i] > acc[i + 1]) {
                        int t = acc[i];
                        acc[i] = acc[i + 1];
                        acc[i + 1] = t;
                    }
                }
            });
        });

        q.wait();
    }
}

int main() {
    std::vector<int> arr(N);

    // 初始化
    for (int i = 0; i < N; i++)
        arr[i] = N - i;

    oddEvenSortSYCL(arr);

    // 输出
    std::cout << "{";
    for (int i = 0; i < N; i++) {
        std::cout << arr[i];
        if (i < N - 1) std::cout << ", ";
    }
    std::cout << "}" << std::endl;

    return 0;
}
