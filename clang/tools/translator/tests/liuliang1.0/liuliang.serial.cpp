#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <fstream>

const int WIDTH = 100;       // 路段长度
const int TIME_STEPS = 200;  // 时间步数
const double DELTA_T = 0.01; // 时间步长
const double DELTA_X = 1.0;  // 空间步长

// 流量函数，考虑密度对流量的影响
double q(double rho) {
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

    std::ofstream outputFile("density_output.csv");
    if (!outputFile) {
        std::cerr << "无法打开文件！" << std::endl;
        return 1;
    }

    // 写入表头
    outputFile << "Time Step";
    for (int x = 0; x < WIDTH; ++x) {
        outputFile << "," << x;
    }
    outputFile << std::endl;

    for (int t = 0; t < TIME_STEPS; ++t) {
        for (int x = 1; x < WIDTH - 1; ++x) {
            double flow_left = q(rho[x - 1]);
            double flow_right = q(rho[x]);
            double delta_rho = (DELTA_T / DELTA_X) * (flow_right - flow_left);
            
            // 更新密度，并确保不会产生负值
            new_rho[x] = rho[x] - delta_rho;

            // 确保密度不为负数
            new_rho[x] = std::max(0.0, new_rho[x]);

            // 引入扩散效应，增加密度的均匀性
            if (x > 1 && x < WIDTH - 2) {
                new_rho[x] += 0.1 * (rho[x - 1] + rho[x + 1] - 2 * rho[x]); // 简单的扩散模型
            }
        }

        // 处理边界条件
        new_rho[0] = new_rho[1]; // 左边界无车流
        new_rho[WIDTH - 1] = new_rho[WIDTH - 2]; // 右边界无车流
        
        // 更新密度值
        rho = new_rho;

        // 输出当前时间步的密度分布到文件
        outputFile << t; 
        for (int x = 0; x < WIDTH; ++x) {
            outputFile << "," << static_cast<int>(rho[x]);
        }
        outputFile << std::endl;
    }

    outputFile.close();
    return 0;
}
