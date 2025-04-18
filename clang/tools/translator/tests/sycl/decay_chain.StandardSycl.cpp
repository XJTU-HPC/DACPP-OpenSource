#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace sycl;

constexpr int num_isotopes = 10;  // 同位素数量
constexpr int num_steps = 51;     // 时间步数量，基于 T 和 dt 计算 (T / dt = 5 / 0.1 = 50)

double decay_rate(int isotope_id) {
    // 假设每个同位素的衰变常数不同
    const double decay_constants[num_isotopes] = {0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10};
    return decay_constants[isotope_id];
}

// 衰变计算函数
void decay_computation(queue &q, std::vector<double> &flat_results, double dt, double T) {
    buffer<double, 1> result_buffer(flat_results.data(), range<1>(num_steps * num_isotopes));

    q.submit([&](handler &h) {
        // 读取结果缓冲区
        auto results_acc = result_buffer.get_access<access::mode::write>(h);

        h.parallel_for(range<2>(num_steps, num_isotopes), [=](id<2> idx) {
            int step = idx[0];
            int isotope_id = idx[1];

            // 计算当前时间 t
            double t = step * dt;

            // 初始数量为1，假设每个同位素的初始数量是1
            double initial_quantity = 1000.0;

            // 衰变公式 N(t) = N(0) * exp(-lambda * t)
            double lambda = decay_rate(isotope_id);
            double quantity = initial_quantity * std::exp(-lambda * t);

            // 存储在展平后的二维数组的相应位置
            results_acc[step * num_isotopes + isotope_id] = quantity;
        });
    }).wait();
}

int main() {
    // 创建队列
    queue q;

    // 用来存储结果的展平二维数组
    std::vector<double> flat_results(num_steps * num_isotopes, 0.0);

    // 时间步长和总时间
    double dt = 0.1;  // 时间步长
    double T = 5.0;   // 总时间

    // 进行衰变计算
    decay_computation(q, flat_results, dt, T);

    // 打印结果
    std::cout << "Time Step\tIsotope 1\tIsotope 2\tIsotope 3\tIsotope 4\tIsotope 5" << std::endl;
    for (int step = 0; step < num_steps; ++step) {
        std::cout << step * dt << "\t";  // 打印对应的时间步
        for (int isotope_id = 0; isotope_id < num_isotopes; ++isotope_id) {
            std::cout << flat_results[step * num_isotopes + isotope_id] << "\t\t";
        }
        std::cout << std::endl;
    }

    return 0;
}
