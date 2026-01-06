#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <any>
#include <queue>
#include "ReconTensor.h"
namespace dacpp {
    typedef std::vector<std::any> list;
}
// 参数设置
const double A = 1.0;  // 吸引力系数
const double D = 0.1;  // 扩散系数
const double dx = 0.1; // 空间步长
const double dt = 0.01; // 时间步长
const int N = 10000;     // 空间网格点数
const int T = 1;    // 时间步数

// 初始化用户偏好分布
void initialize(std::vector<double>& p) {
    for (int i = 0; i < N; ++i) {
        // 假设初始偏好为高斯分布
        double x = i * dx;
        p[i] = std::exp(-std::pow(x - 5.0, 2) / 2.0); // 初始偏好分布中心在x=5
    }
}

// 归一化函数
void normalize(dacpp::Vector<double>& p) {
    double sum = 0.0;
    for (int i = 0;i < N-2; i++) {
        sum += p[i];
    }
    for (int i = 0;i < N-2; i++) {
        p[i] /= sum; // 归一化    
    }
}

shell dacpp::list mdp_shell( dacpp::Vector<double>& p READ_WRITE, dacpp::Vector<double>& new_p READ_WRITE){
    dacpp::index idx;
    dacpp::split sp(3,1);
    binding(idx, sp);
    dacpp::list dataList{p[sp],new_p[idx]};
    return dataList;
}

calc void mdp(dacpp::Vector<double>& p, double* new_p){
    double diffusion = D * (p[2] - 2 * p[1] + p[0]) / (dx * dx) ;
    double drift = (-A) * (p[2] - p[0]) / (2 * dx);
    new_p[0] = p[1] + dt * (diffusion+ drift);
}

// 数值求解Fokker-Planck方程

int main() {
    std::vector<double> p1(N, 0.0); // 存储用户偏好分布
    // 初始化偏好分布
    initialize(p1);
    // 数值求解Fokker-Planck方程
    std::vector<double> new_p1(N-2, 0.0); // 存储下一时间步的分布
    dacpp::Vector<double> p(p1);
    dacpp::Vector<double> new_p(new_p1);
    for (int t = 0; t < T; ++t) {
        mdp_shell(p, new_p) <->  mdp;
        //normalize(new_p); // 归一化分布  
        // 更新分布
        for(int i = 0; i <= N-3; i++){
            p[i+1] = new_p[i];
        }
        // 设置边界条件
        //p[0] = 0.0;
        //p[N - 1] = 0.0;
        
    }
    std::cout << p[2] << std::endl;
    //p.print();
    return 0;
}
