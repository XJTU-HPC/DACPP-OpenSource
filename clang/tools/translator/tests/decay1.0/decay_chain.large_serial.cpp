#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
int main() {
    size_t numIsotopes = 10000; // 同位素数量
    double dt = 0.1;         // 时间步长
    double T = 5.0;         // 总时间
    size_t numSteps = static_cast<size_t>(T / dt); // 计算总步数

    // 初始化衰变常数和初始数量
    std::vector<double> lambdas(numIsotopes);
    std::vector<double> N0s(numIsotopes, 1000.0);  // 初始数量为1000
    for (size_t i = 0; i < numIsotopes; ++i) {
        lambdas[i] = 0.01 + 0.01 * i;  // lambda 线性递增 [0.01, 0.1]
    }

    // 存储所有时间步的数据
    std::vector<std::vector<double>> results(numSteps + 1, std::vector<double>(numIsotopes));
    results[0] = N0s; // 初始值

    // 计算同位素的衰变过程
    for (size_t t = 1; t <= numSteps; ++t) {
        for (size_t i = 0; i < numIsotopes; ++i) {
            results[t][i] = results[t - 1][i] * std::exp(-lambdas[i] * dt);
        }
    }

    // 以 "{{...}, {...}, ...}" 格式输出
    std::cout << "{";
    for (size_t t = 1; t <= 1; ++t) {
        //std::cout << "{";
        for (size_t i = 0; i < numIsotopes; ++i) {
            std::cout << results[t][i];
            if (i < numIsotopes - 1) std::cout << ", ";
        }
        //std::cout << "}";
        //if (t < numSteps) std::cout << ", ";
    }
    std::cout << "}" << std::endl;
    
    return 0;
}
