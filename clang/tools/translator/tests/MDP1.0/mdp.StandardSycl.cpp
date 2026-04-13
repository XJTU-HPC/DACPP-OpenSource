#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>

// 参数设置
const double A = 1.0;  // 吸引力系数
const double D = 0.1;  // 扩散系数
const double dx = 0.1; // 空间步长
const double dt = 0.01; // 时间步长
const int N = 150;     // 空间网格点数
const int T = 10000;    // 时间步数

// 初始化用户偏好分布
void initialize(std::vector<double>& p) {
    for (int i = 0; i < N; ++i) {
        // 假设初始偏好为高斯分布
        double x = i * dx;
        p[i] = exp(-pow(x - 5.0, 2) / 2.0); // 初始偏好分布中心在x=5
    }
}

// 归一化函数
void normalize(std::vector<double>& p) {
    double sum = 0.0;
    for (double value : p) {
        sum += value;
    }
    for (double& value : p) {
        value /= sum; // 归一化
    }
}

int main() {
    std::vector<double> p(N, 0.0); // 存储用户偏好分布
    std::vector<double> new_p(N, 0.0); // 存储下一时间步的分布

    // 初始化偏好分布
    initialize(p);

    // 使用SYCL进行并行计算
    {
        sycl::queue q;

        for (int t = 0; t < T; ++t) {
            // 对内部点进行更新，边界条件暂不考虑
            sycl::buffer<double, 1> p_buffer(p.data(), sycl::range<1>(N));
            sycl::buffer<double, 1> new_p_buffer(new_p.data(), sycl::range<1>(N));

            q.submit([&](sycl::handler& cgh) {
                auto p_access = p_buffer.get_access<sycl::access::mode::read>(cgh);
                auto new_p_access = new_p_buffer.get_access<sycl::access::mode::write>(cgh);

                cgh.parallel_for<class FokkerPlanck>(sycl::range<1>(N - 2), [=](sycl::item<1> item) {
                    int i = item.get_id(0) + 1; // 因为我们从1到N-2的索引进行计算

                    // 扩散项和漂移项
                    double diffusion = D * (p_access[i + 1] - 2 * p_access[i] + p_access[i - 1]) / (dx * dx);
                    double drift = -A * (p_access[i + 1] - p_access[i - 1]) / (2 * dx);
                    new_p_access[i] = p_access[i] + dt * (diffusion + drift);
                });
            });

            // 归一化分布
            //normalize(p);

            // 更新p，顺序执行，因为没有并发冲突
            for (int i = 1; i < N - 1; ++i) {
                p[i] = new_p[i];  // 更新内部点
            }
            // 设置边界条件
            //p[0] = 0.0;
            //p[N - 1] = 0.0;
        }
        // std::cout << "{";
        // for (int i = 2; i < 2; ++i) {
        //     std::cout << p[i];
        //     //if (i < N - 1) std::cout << ", ";
        // }
        // std::cout << "}" << std::endl;

        std::cout << p[2] << std::endl;
        

            
    }

    return 0;
}
