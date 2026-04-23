// dft_sycl.cpp
#include <CL/sycl.hpp>
#include <vector>
#include <iostream>
#include <cmath>

using namespace sycl;
using namespace std;

int main() {
    constexpr int N = 65536;

    // --- 准备输入：这里以实数序列示例（imag = 0），可按需修改 ---
    std::vector<double> in_real(N), in_imag(N);
    for (int n = 0; n < N; ++n) {
        in_real[n] = static_cast<double>(n); // 示例：0,1,2,...,N-1
        in_imag[n] = 0.0;
    }

    // 输出缓冲区（实部/虚部）
    std::vector<double> out_real(N, 0.0), out_imag(N, 0.0);

    // 创建 SYCL 队列（选择默认设备）
    queue q;

    {
        // 创建 buffer（RAII）
        buffer<double, 1> buf_in_real(in_real.data(), range<1>(N));
        buffer<double, 1> buf_in_imag(in_imag.data(), range<1>(N));
        buffer<double, 1> buf_out_real(out_real.data(), range<1>(N));
        buffer<double, 1> buf_out_imag(out_imag.data(), range<1>(N));

        // 提交任务：对每个 k 并行计算 X[k]
        q.submit([&](handler &h) {
            accessor a_in_r(buf_in_real, h, read_only);
            accessor a_in_i(buf_in_imag, h, read_only);
            accessor a_out_r(buf_out_real, h, write_only, no_init);
            accessor a_out_i(buf_out_imag, h, write_only, no_init);

            h.parallel_for<class dft_kernel>(range<1>(N), [=](id<1> idk) {
                int k = idk[0];
                double sum_r = 0.0;
                double sum_i = 0.0;

                // 内层 n 循环在 device 上顺序执行
                for (int n = 0; n < N; ++n) {
                    // 角度：angle = -2 * pi * k * n / N
                    double angle = -2.0 * M_PI * k * n / static_cast<double>(N);
                    double c = sycl::cos(angle);
                    double s = sycl::sin(angle);

                    double a = a_in_r[n];
                    double b = a_in_i[n];

                    // (a + i b) * (c + i s) = (a*c - b*s) + i(a*s + b*c)
                    sum_r += a * c - b * s;
                    sum_i += a * s + b * c;
                }

                a_out_r[k] = sum_r;
                a_out_i[k] = sum_i;
            });
        });

        // 等待队列完成，buffer 的析构会自动把数据拷回主机内存
        q.wait();
    } // buffers 离开作用域并同步回 out_real/out_imag

    // 格式化输出：每个元素用 (real, imag)
    std::cout << "{";
    for (int k = 0; k < N; ++k) {
        std::cout << "(" << out_real[k] << "," << out_imag[k] << ")";
        if (k < N - 1) std::cout << ", ";
    }
    std::cout << "}" << std::endl;

    return 0;
}
