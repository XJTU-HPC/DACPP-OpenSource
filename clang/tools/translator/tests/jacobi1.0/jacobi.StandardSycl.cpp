#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace sycl;

int main() {
    auto start_time = std::chrono::high_resolution_clock::now(); // 开始时间测量
    const int N = 100; // 矩阵大小
    const int max_iter = 100;
    const float tolerance = 1e-6;

    std::vector<float> A(N * N, 0.0f);
    std::vector<float> b(N, 0.0f);
    std::vector<float> x(N, 0.0f);     // 初始解
    std::vector<float> x_new(N, 0.0f); // 更新后的解

    // 初始化 A 和 b
    for (int i = 0; i < N; ++i) {
        A[i * N + i] = 4.0f; // 对角线元素
        if (i > 0) A[i * N + i - 1] = -1.0f; // 下三角
        if (i < N - 1) A[i * N + i + 1] = -1.0f; // 上三角
        b[i] = 1.0f; // 向量 b
    }

    queue q;

    buffer<float, 1> A_buf(A.data(), range<1>(N * N));
    buffer<float, 1> b_buf(b.data(), range<1>(N));
    buffer<float, 1> x_buf(x.data(), range<1>(N));
    buffer<float, 1> x_new_buf(x_new.data(), range<1>(N));

    bool converged = false;
    int iter = 0;

    while (!converged && iter < max_iter) {
        //std::cout << "迭代次数: " << iter << std::endl;

        // 提交计算任务
        q.submit([&](handler& h) {
            auto A_acc = A_buf.get_access<access::mode::read>(h);
            auto b_acc = b_buf.get_access<access::mode::read>(h);
            auto x_acc = x_buf.get_access<access::mode::read>(h);
            auto x_new_acc = x_new_buf.get_access<access::mode::discard_write>(h);

            h.parallel_for(range<1>(N), [=](id<1> i) {
                float sigma = 0.0f;
                for (int j = 0; j < N; ++j) {
                    if (j != i[0]) {
                        sigma += A_acc[i[0] * N + j] * x_acc[j];
                    }
                }
                x_new_acc[i] = (b_acc[i] - sigma) / A_acc[i[0] * N + i[0]];
            });
        });

        // 等待计算完成
        q.wait();

        // 计算收敛性
        auto x_acc_host = x_buf.get_access<access::mode::read>();
        auto x_new_acc_host = x_new_buf.get_access<access::mode::read>();

        float max_error = 0.0f;
        for (int i = 0; i < N; ++i) {
            max_error = std::max(max_error, std::fabs(x_new_acc_host[i] - x_acc_host[i]));
        }
        //std::cout << "最大误差: " << max_error << std::endl;

        if (max_error < tolerance) {
            converged = true;
        }

        // 更新 x，主机端进行更新
        for (int i = 0; i < N; ++i) {
            x[i] = x_new[i];
        }

        ++iter;
    }

    // 获取最终结果
    auto x_result = x_new_buf.get_access<access::mode::read>();
    std::cout << "迭代次数: " << iter << std::endl;
    std::cout << "解向量 x:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << x_result[i] << " ";
    }
    std::cout << std::endl;
    auto end_time = std::chrono::high_resolution_clock::now(); // 结束时间测量
    std::chrono::duration<double> duration = end_time - start_time; // 计算持续时间
    std::cout << "总执行时间: " << duration.count() << " 秒" << std::endl; // 输出执行时间
    return 0;
}
