#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>

// namespace sycl = cl::sycl;

void calculateDecay(const std::vector<double>& lambdas, const std::vector<double>& N0s, double dt, double T) {
    size_t numIsotopes = lambdas.size();  // 同位素的数量
    size_t numTimeSteps = static_cast<size_t>(T / dt);  // 计算总时间步数

    // 存储每个同位素在每个时间步的衰变数量
    std::vector<std::vector<double>> A(numTimeSteps, std::vector<double>(numIsotopes, 0.0));

    // 创建一个SYCL队列来选择设备（通常是GPU）
    sycl::queue q;

    // 使用SYCL buffer来分配设备内存
    sycl::buffer<double> lambdas_buf(lambdas);
    sycl::buffer<double> N0s_buf(N0s);
    sycl::buffer<double> A_buf(A.data(), sycl::range<2>(numTimeSteps, numIsotopes));

    // 创建时间步变量
    std::vector<double> time = {0.0};

    // 在设备上并行计算
    for (size_t step = 0; step < numTimeSteps; ++step) {
        q.submit([&](sycl::handler& h) {
            // 获取缓冲区的访问权限
            auto lambdas_acc = lambdas_buf.get_access<sycl::access::mode::read>(h);
            auto N0s_acc = N0s_buf.get_access<sycl::access::mode::read>(h);
            auto A_acc = A_buf.get_access<sycl::access::mode::write>(h);

            // 并行计算每个同位素的数量
            h.parallel_for(sycl::range<1>(numIsotopes), [=](sycl::id<1> i) {
                double lambda = lambdas_acc[i];    // 衰变常数
                double N0 = N0s_acc[i];            // 初始数量
                double A = N0 * std::exp(-lambda * time[0]);  // 衰变公式
                A_acc[step][i] = A;  // 将计算结果存储到A数组
            });
        }).wait();  // 等待内核完成

        // 更新时间
        time[0] += dt;  // 每次迭代后更新时间
    }

    // 打印整个二维数组的结果
    std::cout << "Decay results (rows: time steps, columns: isotopes):\n";
    for (size_t i = 0; i < numTimeSteps; ++i) {
        std::cout << "Time step " << i * dt << ": ";
        for (size_t j = 0; j < numIsotopes; ++j) {
            std::cout << A[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    size_t numIsotopes = 10;  // 假设有 10 个同位素

    // 随机生成衰变常数和初始数量
    std::vector<double> lambdas(numIsotopes);
    std::vector<double> N0s(numIsotopes, 1000.0);  // 初始数量为1000

    // 随机初始化衰变常数（例如，lambda 在 0.01 到 0.2 之间）
    for (size_t i = 0; i < numIsotopes; ++i) {
        lambdas[i] = 0.01 + static_cast<double>(rand()) / (RAND_MAX / (0.2 - 0.01));  // lambda 范围 [0.01, 0.2]
    }

    double dt = 0.1;       // 时间步长
    double T = 1.0;        // 总时间

    // 调用计算衰变函数
    calculateDecay(lambdas, N0s, dt, T);

    return 0;
}
