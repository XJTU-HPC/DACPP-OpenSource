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

const int WIDTH = 100;
const double TIME_STEPS = 200;
const double DELTA_T = 0.01;
const double DELTA_X = 1.0;


double q(double rho) {
    double V_max = 30;
    double rho_max = 50;
    return rho * V_max * (1 - rho / rho_max);
}


void initializeDensity(std::vector<double>& rho) {
    for (int i = 0; i < WIDTH; ++i) {
        if (i < WIDTH / 4) {
            rho[i] = 40;
        } else if (i < 3 * WIDTH / 4) {
            rho[i] = 20;
        } else {
            rho[i] = 10;
        }
    }
}


calc void lwr(dacpp::Vector<double> & rho, double* new_rho) {
    new_rho[0] = rho[1] - (DELTA_T / DELTA_X) * (q(rho[1]) - q(rho[0]));
    new_rho[0] = std::max(0.0, new_rho[0]);

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
        rho[0] = new_rho[0];

    }
    }
    std::cout << rho[15] << std::endl;






    return 0;
}