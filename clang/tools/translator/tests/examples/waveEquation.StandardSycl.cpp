#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace sycl;
using namespace std;

// 网格参数
const int NX = 8;    // x方向网格数量
const int NY = 8;    // y方向网格数量
const float Lx = 10.0f; // x方向长度
const float Ly = 10.0f; // y方向长度
const float c = 1.0f;   // 波速
const int TIME_STEPS = 2; // 时间步数

int main() {
    auto start_time = std::chrono::high_resolution_clock::now(); // 开始时间测量
    // 网格步长
    float dx = Lx / (NX - 1);
    float dy = Ly / (NY - 1);
    
    // CFL条件
    float dt = 0.5f * std::fmin(dx, dy) / c; // 满足稳定性条件
    
    // 初始化波场
    vector<float> u_prev(NX * NY, 0.0f); // 前一步
    vector<float> u_curr(NX * NY, 0.0f); // 当前步
    vector<float> u_next(NX * NY, 0.0f); // 下一步
    
    // 初始条件：例如一个高斯脉冲
    int cx = NX / 2;
    int cy = NY / 2;
    float sigma = 0.5f;
    for(int i = 0; i < NX; ++i) {
        for(int j = 0; j < NY; ++j) {
            float x = i * dx;
            float y = j * dy;
            //u_prev[i * NY + j] = 1;
            u_prev[i * NY + j] = std::exp(-((x - Lx/2)*(x - Lx/2) + (y - Ly/2)*(y - Ly/2)) / (2 * sigma * sigma));
        }
    }

    
    // SYCL队列
    queue q(gpu_selector_v);
    // cout << "Running on "
    //      << q.get_device().get_info<info::device::name>() << "\n";
    
    // 分配设备内存
    float *d_u_prev = malloc_device<float>(NX * NY, q);
    float *d_u_curr = malloc_device<float>(NX * NY, q);
    float *d_u_next = malloc_device<float>(NX * NY, q);
    
    // 将初始数据复制到设备
    q.memcpy(d_u_prev, u_prev.data(), sizeof(float) * NX * NY).wait();
    q.memcpy(d_u_curr, u_curr.data(), sizeof(float) * NX * NY).wait();
    
    // 主迭代循环
    for(int t = 0; t < TIME_STEPS; ++t) {
        std::cout<<"step "<<t<<":\n";
        std::cout<<"now cur:\n";
        q.memcpy(u_curr.data(), d_u_curr, sizeof(float) * NX * NY).wait();
        for(int i = 0; i < NX; i++){
            for(int j = 0; j < NY; j++){
                std::cout << u_curr[i * NX + j] << " " ;
            }
            std::cout << std::endl;
        }
        q.submit([&](handler &h) {
            // 使用parallel_for来更新内部点
            h.parallel_for<class wave_compute>(
                range<2>{NX - 2, NY - 2},
                [=](id<2> idx) {
                    int i = idx[0] + 1;
                    int j = idx[1] + 1;
                    // 二阶中心差分
                    float u_xx = (d_u_curr[(i+1) * NY + j] - 2.0f * d_u_curr[i * NY + j] + d_u_curr[(i-1) * NY + j]) / (dx * dx);
                    float u_yy = (d_u_curr[i * NY + (j+1)] - 2.0f * d_u_curr[i * NY + j] + d_u_curr[i * NY + (j-1)]) / (dy * dy);
                    
                    // 显式更新公式
                    d_u_next[i * NY + j] = 2.0f * d_u_curr[i * NY + j] - d_u_prev[i * NY + j] + (c * c) * (u_xx + u_yy) * dt * dt;
                    // d_u_next[i * NY + j] = 2*d_u_prev[i * NY + j];
                    
                }
            );
        });
        
        // 处理边界条件（简单的固定边界，波值为0）
        q.submit([&](handler &h) {
            // 左右边界
            h.parallel_for<class boundary_left_right>(range<1>{NX}, [=](id<1> i) {
                d_u_next[i[0] * NY + 0] = 0.0f;
                d_u_next[i[0] * NY + (NY-1)] = 0.0f;
            });
        });
        
        q.submit([&](handler &h) {
            // 上下边界
            h.parallel_for<class boundary_up_down>(range<1>{NY}, [=](id<1> j) {
                d_u_next[0 * NY + j] = 0.0f;
                d_u_next[(NX-1) * NY + j] = 0.0f;
            });
        });
        
        q.wait();


        
        // 交换指针
        swap(d_u_prev, d_u_curr);
        swap(d_u_curr, d_u_next);
        
        // 可选：每隔一定步数复制回主机并输出
        // if(t % 100 == 0) {
        //     q.memcpy(u_curr.data(), d_u_curr, sizeof(float) * NX * NY).wait();
        //     cout << "Step " << t << " completed.\n";
        //     // 这里可以添加保存或可视化代码
        // }

        std::cout<<"next cur:\n";
        q.memcpy(u_curr.data(), d_u_curr, sizeof(float) * NX * NY).wait();
        for(int i = 0; i < NX; i++){
            for(int j = 0; j < NY; j++){
                std::cout << u_curr[i * NX + j] << " " ;
            }
            std::cout << std::endl;
        }
    }
    
    // 复制结果回主机
    q.memcpy(u_curr.data(), d_u_curr, sizeof(float) * NX * NY).wait();

    // 释放设备内存
    free(d_u_prev, q);
    free(d_u_curr, q);
    free(d_u_next, q);
    
    // 输出最终结果的某些值作为示例
    cout << "Final wave state at center: " << u_curr[(NX/2)*NY + (NY/2)] << "\n";
    auto end_time = std::chrono::high_resolution_clock::now(); // 结束时间测量
    std::chrono::duration<double> duration = end_time - start_time; // 计算持续时间
    std::cout << "总执行时间: " << duration.count() << " 秒" << std::endl; // 输出执行时间
    return 0;
}
