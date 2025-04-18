#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

// 计算每种同位素在时间 t 的数量
void calculateDecay(const std::vector<double>& lambdas, const std::vector<double>& N0s, double dt, double T, size_t numOutputSteps) {
    size_t numIsotopes = lambdas.size(); // 同位素的数量
    std::vector<std::vector<double>> A(numIsotopes);  // 存储每个同位素在不同时间点的数量
    std::vector<double> time;  // 时间序列
    double t = 0;

    // 串行计算每个同位素的衰变过程
    for (size_t i = 0; i < numIsotopes; ++i) {
        double N = N0s[i];  // 当前同位素的初始数量
        double lambda = lambdas[i];  // 当前同位素的衰变常数
        std::vector<double> local_A;  // 当前同位素的数量随时间变化的存储

        // 对当前同位素进行衰变计算
        while (t <= T) {
            local_A.push_back(N * exp(-lambda * t));  // 计算衰变后的数量
            t += dt;
        }

        // 存储当前同位素的结果
        A[i] = local_A;
        t = 0;  // 重置时间 t，以便下一个同位素使用正确的时间步
    }

    // 计算时间序列
    t = 0;
    while (t <= T) {
        time.push_back(t);
        t += dt;
    }

    // 只输出部分时间步的结果（例如，前 5 个时间步）
    size_t outputStepInterval = time.size() / numOutputSteps; // 计算输出间隔
    std::cout << "Time ";
    // 输出同位素的标签（A1, A2, A3...）
    for (size_t i = 0; i < numIsotopes; ++i) {
        std::cout << "A" << i + 1 << " ";
    }
    std::cout << "\n";

    // 输出每个时间步的数据（部分时间步）
    for (size_t i = 0; i < numOutputSteps; ++i) {
        size_t timeIndex = i * outputStepInterval;
        std::cout << time[timeIndex];
        for (size_t j = 0; j < numIsotopes; ++j) {
            std::cout << " " << A[j][timeIndex];
        }
        std::cout << "\n";
    }
}

int main() {
    size_t numIsotopes = 10000; // 设定大量同位素（例如，10000个）

    // 随机生成衰变常数和初始数量
    std::vector<double> lambdas(numIsotopes);
    std::vector<double> N0s(numIsotopes, 1000.0);  // 初始数量为1000

    // 随机初始化衰变常数（例如，lambda 在 0.01 到 0.2 之间）
    for (size_t i = 0; i < numIsotopes; ++i) {
        lambdas[i] = 0.01 + static_cast<double>(rand()) / (RAND_MAX / (0.2 - 0.01));  // lambda 范围 [0.01, 0.2]
    }

    double dt = 0.1;       // 时间步长
    double T = 50.0;       // 总时间
    size_t numOutputSteps = 10; // 输出的时间步数量

    calculateDecay(lambdas, N0s, dt, T, numOutputSteps);

    return 0;
}
