#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

shell dacpp::list DECAY(const dacpp::Vector<double>& N0s,
                        const dacpp::Vector<double>& lambdas,
                        dacpp::Vector<double>& local_A,
                        const dacpp::Vector<double>& t) {
    dacpp::index i;
    //dacpp::split s(3,1);
    //binding(i, s);
    dacpp::list dataList{N0s[i], lambdas[i], local_A[i], t[{}]};
    return dataList;
}

calc void decay(double* N0s,
                double* lambdas,
                double* local_A,
                double* t) {
    local_A[0] = N0s[0] * std::exp(-lambdas[0] * t[0]); 
}

// 计算每种同位素在时间 t 的数量
void calculateDecay(const std::vector<double>& lambdas, const std::vector<double>& N0s, double dt, double T) {
    size_t numIsotopes = lambdas.size(); // 同位素的数量
    std::vector<double> A(50*10, 0.0);  // 存储每个同位素在不同时间点的数量
    std::vector<double> time;  // 时间序列
    std::vector<double> t;
    t.push_back(static_cast<double>(0));

    // 串行计算每个同位素的衰变过程
    std::vector<double> local_A(10, 0.0);
    dacpp::Vector<double> local_A_tensor(local_A);
    dacpp::Vector<double> N0s_tensor(N0s);
    dacpp::Vector<double> lambdas_tensor(lambdas);
    dacpp::Vector<double> t_tensor(t);
    dacpp::Matrix<double> A_tensor({50, 10}, A);
    

    while(t_tensor[0] < T){        
        DECAY(N0s_tensor, lambdas_tensor, local_A_tensor,  t_tensor) <-> decay;
        A_tensor[10*t_tensor[0]] = local_A_tensor;
        t_tensor[0] += dt;
    }
    A_tensor.print();
}

int main() {
    size_t numIsotopes = 10; // 设定大量同位素（例如，10000个）

    // 随机生成衰变常数和初始数量
    std::vector<double> lambdas(numIsotopes);
    std::vector<double> N0s(numIsotopes, 1000.0);  // 初始数量为1000

    // 随机初始化衰变常数（例如，lambda 在 0.01 到 0.2 之间）
    for (size_t i = 0; i < numIsotopes; ++i) {
        lambdas[i] = 0.01 + 0.01*i;  // lambda 范围 [0.01, 0.2]
    }

    double dt = 0.1;       // 时间步长
    double T = 5.0;       // 总时间
    //size_t numOutputSteps = 10; // 输出的时间步数量

    calculateDecay(lambdas, N0s, dt, T);

    return 0;
}
