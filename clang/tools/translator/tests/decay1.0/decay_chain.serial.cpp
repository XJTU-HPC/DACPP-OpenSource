#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <sstream>

// 计算每种同位素在时间 t 的数量
void calculateParallelDecay(double lambda1, double lambda2, double lambda3, double N10, double dt, double T, const std::string& filename) {
    std::vector<double> time, A1, A2, A3;
    double t = 0;

    // 初始化同位素数量
    double N1 = N10;
    double N2 = N10;
    double N3 = N10;

    while (t <= T) {
        // 计算当前时间点的数量
        A1.push_back(N1);
        A2.push_back(N2);
        A3.push_back(N3);
        time.push_back(t);

        // 更新同位素数量
        N1 = N10 * exp(-lambda1 * t);
        N2 = N10 * exp(-lambda2 * t);
        N3 = N10 * exp(-lambda3 * t);

        t += dt;
    }

    // 保存结果到文件
    std::ofstream outfile(filename);
    if (outfile.is_open()) {
        outfile << "Time A1 A2 A3\n";
        for (size_t i = 0; i < time.size(); ++i) {
            outfile << time[i] << " " << A1[i] << " " << A2[i] << " " << A3[i] << "\n";
        }
        outfile.close();
    } else {
        std::cerr << "Unable to open file for writing.\n";
    }
}

int main() {
    double lambda1 = 0.1;  // 衰变常数λ1
    double lambda2 = 0.05; // 衰变常数λ2
    double lambda3 = 0.02; // 衰变常数λ3
    double N10 = 1000.0;   // 初始数量N10
    double dt = 0.1;       // 时间步长
    double T = 50.0;       // 总时间

    std::string filename = "parallel_decay_data.txt";
    calculateParallelDecay(lambda1, lambda2, lambda3, N10, dt, T, filename);

    std::cout << "Simulation complete. Results saved to " << filename << std::endl;

    return 0;
}
