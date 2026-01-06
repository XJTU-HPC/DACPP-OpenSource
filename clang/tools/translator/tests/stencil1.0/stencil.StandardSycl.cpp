#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace sycl;
using namespace std;

// 网格参数
const int NX = 8;           // x方向网格数量
const int NY = 8;           // y方向网格数量
const double Lx = 10.0f;       // x方向长度
const double Ly = 10.0f;       // y方向长度
const double alpha = 0.01f;    // 热扩散系数
const int TIME_STEPS = 10;  // 时间步数

int main() {
    // 空间步长
    double dx = Lx / (NX - 1);
    double dy = Ly / (NY - 1);

    // 稳定性条件
    double dt_stability = (dx * dx * dy * dy) / (2.0f * alpha * (dx * dx + dy * dy));
    double delta_t = 0.4f * dt_stability; // 选择一个更严格的时间步长以确保稳定性

    //cout << "Grid size: " << NX << "x" << NY << "\n";
    //cout << "dx = " << dx << ", dy = " << dy << ", delta_t = " << delta_t << "\n";

    // 初始化温度场
    vector<double> u_prev(NX * NY, 0.0f); // 前一步（在热传导方程中，只有当前步和上一步）
    vector<double> u_curr(NX * NY, 0.0f); // 当前步

    // 初始条件：例如，中心有一个高斯分布的热源
    int cx = NX / 2;
    int cy = NY / 2;
    double sigma = 1.0f;
    for(int i = 0; i < NX; ++i) {
        for(int j = 0; j < NY; ++j) {
            double x = i * dx;
            double y = j * dy;
            // 高斯分布
            u_curr[i * NY + j] = std::exp(-((x - Lx/2.0f)*(x - Lx/2.0f) + (y - Ly/2.0f)*(y - Ly/2.0f)) / (2.0f * sigma * sigma));
        }
    }

    // SYCL队列
    queue q(gpu_selector_v);
    // cout << "Running on " 
    //      << q.get_device().get_info<info::device::name>() << "\n";

    // 分配设备内存
    double *d_u_prev = malloc_device<double>(NX * NY, q);
    double *d_u_curr = malloc_device<double>(NX * NY, q);
    double *d_u_next = malloc_device<double>(NX * NY, q);

    // 将初始数据复制到设备
    q.memcpy(d_u_prev, u_prev.data(), sizeof(double) * NX * NY).wait();
    q.memcpy(d_u_curr, u_curr.data(), sizeof(double) * NX * NY).wait();

    // 主迭代循环
    for(int t = 0; t < TIME_STEPS; ++t) {
        q.submit([&](handler &h) {
            // 使用parallel_for来更新内部点
            h.parallel_for<class heat_compute>(
                range<2>{NX - 2, NY - 2},
                [=](id<2> idx) {
                    int i = idx[0] + 1;
                    int j = idx[1] + 1;

                    // 二阶中心差分
                    double u_xx = (d_u_curr[(i + 1) * NY + j] - 2.0f * d_u_curr[i * NY + j] + d_u_curr[(i - 1) * NY + j]) / (dx * dx);
                    double u_yy = (d_u_curr[i * NY + (j + 1)] - 2.0f * d_u_curr[i * NY + j] + d_u_curr[i * NY + (j - 1)]) / (dy * dy);

                    // 显式更新公式
                    d_u_next[i * NY + j] = d_u_curr[i * NY + j] + alpha * delta_t * (u_xx + u_yy);
                }
            );
        });

        // 处理边界条件（简单的绝热边界，导数为零）
        // 可以根据具体需求修改边界条件类型

        // 上下边界（i = 0 和 i = NX-1）
        q.submit([&](handler &h) {
            h.parallel_for<class boundary_top_bottom>(range<1>{NY}, [=](id<1> j) {
                // 顶部边界 i = 0
                d_u_next[0 * NY + j] = d_u_next[1 * NY + j];
                // 底部边界 i = NX-1
                d_u_next[(NX - 1) * NY + j] = d_u_next[(NX - 2) * NY + j];
            });
        });

        // 左右边界（j = 0 和 j = NY-1）
        q.submit([&](handler &h) {
            h.parallel_for<class boundary_left_right>(range<1>{NX}, [=](id<1> i) {
                // 左边界 j = 0
                d_u_next[i[0] * NY + 0] = d_u_next[i[0] * NY + 1];
                // 右边界 j = NY-1
                d_u_next[i[0] * NY + (NY - 1)] = d_u_next[i[0] * NY + (NY - 2)];
            });
        });

        q.wait();

        // 交换指针以更新下一步
        swap(d_u_prev, d_u_curr);
        swap(d_u_curr, d_u_next);

        // 可选：每隔一定步数复制回主机并输出
        // if(t % 100 == 0) {
        //     q.memcpy(u_curr.data(), d_u_curr, sizeof(double) * NX * NY).wait();
        //     cout << "Step " << t << " completed.\n";
        //     // 这里可以添加保存或可视化代码
        // }
    }

    // 复制结果回主机
    q.memcpy(u_curr.data(), d_u_curr, sizeof(double) * NX * NY).wait();


    std::cout << "{";
    for(int i = 0; i < 1; i++){
        //std::cout << "{";
        for(int j = 0; j < NY; j++){
            std::cout << u_curr[i * NX + j]  ;
            if (j < NY - 1) std::cout << ", ";
        }
        //std::cout << "}";
        //if (i < NX - 1) std::cout << ", ";
        
    }
    std::cout << "}" << std::endl;


    // 释放设备内存
    free(d_u_prev, q);
    free(d_u_curr, q);
    free(d_u_next, q);

    // 输出最终结果的某些值作为示例
    //cout << "Final temperature at center: " << u_curr[(NX/2)*NY + (NY/2)] << "\n";

    return 0;
}
