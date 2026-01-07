#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <any>
#include <queue>
#include "ReconTensor.h"


namespace dacpp {
    typedef std::vector<std::any> list;
}

const int WIDTH = 10;       // 路段长度
const double TIME_STEPS = 20;  // 时间步数
const double DELTA_T = 0.01; // 时间步长
const double DELTA_X = 1.0;  // 空间步长

// 流量函数，考虑密度对流量的影响
double q(double rho) {
    double V_max = 30; // 最大速度
    double rho_max = 50; // 最大密度
    return rho * V_max * (1 - rho / rho_max);
}

// 初始化密度，使用随机的分布
void initializeDensity(std::vector<double>& rho) {
    for (int i = 0; i < WIDTH; ++i) {
        if (i < WIDTH / 4) {
            rho[i] = 40; // 高密度区
        } else if (i < 3 * WIDTH / 4) {
            rho[i] = 20; // 中密度区
        } else {
            rho[i] = 10; // 低密度区
        }
    }
}

// 计算交通流量
calc void lwr(dacpp::Vector<double> & rho, double* new_rho) {
    new_rho[0] = rho[1] - (DELTA_T / DELTA_X) * (q(rho[1]) - q(rho[0]));
    new_rho[0] = std::max(0.0, new_rho[0]);
    //new_rho += 0.1 * (rho[0] + new_rho - 2 * rho[1]);
}

shell dacpp::list LWR_shell( dacpp::Vector<double> & rho READ_WRITE, 
                            dacpp::Vector<double> & new_rho READ_WRITE) {
    dacpp::index idx1;
    dacpp::split S1(2, 1);
    binding(idx1, S1);
    dacpp::list dataList{rho[S1], new_rho[idx1]};
    return dataList;
}

int main() {
    // 创建 Tensor 类型对象
    std::vector<double> rho1(WIDTH, 0.0);
    std::vector<double> new_rho1(WIDTH, 0.0);
    initializeDensity(rho1);
    dacpp::Vector<double> rho_tensor(rho1);
    dacpp::Vector<double> new_rho_tensor(new_rho1);
    dacpp::Vector<double> new_rho = new_rho_tensor[{1,WIDTH-1}];
    dacpp::Vector<double> rho = rho_tensor[{0,WIDTH-1}];
    for (int t = 0; t < TIME_STEPS; ++t) {
        LWR_shell(rho, new_rho) <-> lwr;
        for (int i = 1; i <= WIDTH-2; i++) {
            rho[i] = new_rho[i-1];
        }
        for(int i=0;i<1;i++){
        rho[0] = new_rho[0]; // 左边界无车流
        //middle_in_tensor[99] = middle_out_tensor[97];
    }
    }
    std::cout << rho[15] << std::endl;
    

    

    // 释放动态分配的内存

    return 0;
}
