#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

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

// 数值求解Fokker-Planck方程
void solveFokkerPlanck(std::vector<double>& p) {
    std::vector<double> new_p(N, 0.0); // 存储下一时间步的分布

    std::ofstream outFile("preference_distribution.csv");
    outFile << "TimeStep,";
    for (int i = 0; i < N; ++i) {
        outFile << "p(" << i * dx << "),";
    }
    outFile << std::endl;

    for (int t = 0; t < T; ++t) {
        // 对内部点进行更新，边界条件暂不考虑
        for (int i = 1; i < N - 1; ++i) {
            double diffusion = D * (p[i + 1] - 2 * p[i] + p[i - 1]) / (dx * dx);
            double drift = -A * (p[i + 1] - p[i - 1]) / (2 * dx);
            new_p[i] = p[i] + dt * (diffusion + drift);
        }

        // 更新分布
        p = new_p;
        normalize(p); // 归一化分布

        // 输出当前分布
        if (t % 100 == 0) {
            //std::cout << "Time step: " << t << std::endl;
            outFile << t << ",";
            for (int i = 0; i < N; ++i) {
                outFile << p[i] << ",";
            }
            outFile << std::endl;
        }
    }
    for (int i = 0; i < N; ++i) {
        std::cout << p[i] << ",";
    }          

    outFile.close();
}

int main() {
    std::vector<double> p(N, 0.0); // 存储用户偏好分布

    // 初始化偏好分布
    initialize(p);

    // 数值求解Fokker-Planck方程
    solveFokkerPlanck(p);

    return 0;
}
