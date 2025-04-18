#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>

const int WIDTH = 100;       // 路段长度
const int TIME_STEPS = 200;  // 时间步数
const double DELTA_T = 0.01; // 时间步长
const double DELTA_X = 1.0;  // 空间步长

// 流量函数，考虑密度对流量的影响
double q1(double rho) {
    const double V_max = 30.0; // 最大速度
    const double rho_max = 50.0; // 最大密度
    return rho * V_max * (1 - rho / rho_max);
}

// 初始化密度，使用随机的分布
void initializeDensity(std::vector<double>& rho) {
    for (int i = 0; i < WIDTH; ++i) {
        if (i < WIDTH / 4) {
            rho[i] = 40.0; // 高密度区
        } else if (i < 3 * WIDTH / 4) {
            rho[i] = 20.0; // 中密度区
        } else {
            rho[i] = 10.0; // 低密度区
        }
    }
}

int main() {
    std::vector<double> rho(WIDTH, 0.0);
    std::vector<double> new_rho(WIDTH, 0.0);

    initializeDensity(rho);
    // std::cout << "chushihuahou " <<std::endl;

    // for(int i = 0;i < 100; i++){
    //     std::cout << rho[i] << ","  ;
    // }
    // SYCL queue 用于选择设备并执行并行任务
    sycl::queue q;

    for (int t = 0; t < TIME_STEPS; ++t) {
        // 使用 SYCL buffer 和 parallel_for 来并行更新密度
        {
            sycl::buffer<double, 1> rho_buf(rho.data(), sycl::range<1>(WIDTH));
            sycl::buffer<double, 1> new_rho_buf(new_rho.data(), sycl::range<1>(WIDTH));

            q.submit([&](sycl::handler& h) {
                // 获取rho和new_rho的访问权限
                auto rho_acc = rho_buf.get_access<sycl::access::mode::read>(h);
                auto new_rho_acc = new_rho_buf.get_access<sycl::access::mode::write>(h);

                h.parallel_for(sycl::range<1>(WIDTH - 2), [=](sycl::id<1> i) {
                    int x = i[0] + 1; // 跳过边界
                    double flow_left = q1(rho_acc[x - 1]);
                    double flow_right = q1(rho_acc[x]);
                    double delta_rho = (DELTA_T / DELTA_X) * (flow_right - flow_left);

                    new_rho_acc[x] = rho_acc[x] - delta_rho;
                    new_rho_acc[x] = std::max(0.0, new_rho_acc[x]);

                    // 扩散效应
                    // if (x > 1 && x < WIDTH - 2) {
                    //     new_rho_acc[x] += 0.1 * (rho_acc[x - 1] + rho_acc[x + 1] - 2 * rho_acc[x]);
                    // }
                });
            });
        }

        // 处理边界条件
        new_rho[0] = new_rho[1]; // 左边界无车流
        new_rho[WIDTH - 1] = new_rho[WIDTH - 2]; // 右边界无车流

        // 更新密度值
        rho = new_rho;

        
    }
    // 输出当前密度分布
    for (int x = 0; x < WIDTH; ++x) {
        std::cout << rho[x] << ",";
    }
    std::cout << std::endl;

    return 0;
}
