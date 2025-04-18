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
const int N = 100;     // 空间网格点数
const int T = 1000;    // 时间步数

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
    for (int i = 0;i < N; i++) {
        sum += p[i];
    }
    for (int i = 0;i < N; i++) {
        p[i] /= sum; // 归一化    
    }
}

shell dacpp::list mdp_shell(const dacpp::Vector<double>& p, dacpp::Vector<double>& new_p){
    dacpp::index idx;
    dacpp::split sp(3,1);
    dacpp::list datalist{p[sp],new_p[idx]};
}

calc void mdp(dacpp::Vector<double>& p, double* new_p){
    new_p[0] = p[1] + dt * (D * (p[2] - 2 * p[1] + p[0]) / (dx * dx) + (-A) * (p[2] - p[0]) / (2 * dx));
}

// 数值求解Fokker-Planck方程
void solveFokkerPlanck(std::vector<double>& p) {
    std::vector<double> new_p(N-2, 0.0); // 存储下一时间步的分布
    dacpp::Vector<double> p_tensor(p);
    dacpp::Vector<double> new_p_tensor(new_p);
    for (int t = 0; t < T; ++t) {
        mdp_shell(p_tensor, new_p_tensor) <->  mdp;  
        // 更新分布
        for(int i = 0; i < N-2; i++){
            p_tensor[i+1] = new_p_tensor[i];
        }
        normalize(p_tensor); // 归一化分布
    }
    p_tensor.print();
}

int main() {
    std::vector<double> p(N, 0.0); // 存储用户偏好分布
    // 初始化偏好分布
    initialize(p);
    // 数值求解Fokker-Planck方程
    solveFokkerPlanck(p);
    return 0;
}
