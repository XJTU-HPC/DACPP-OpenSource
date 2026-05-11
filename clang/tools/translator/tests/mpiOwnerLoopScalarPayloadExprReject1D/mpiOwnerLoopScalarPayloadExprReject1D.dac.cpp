#include <any>
#include <iostream>
#include <vector>

#include "ReconTensor.h"

namespace dacpp {
typedef std::vector<std::any> list;
}

shell dacpp::list PDE(const dacpp::Vector<double>& u_kin,
                      dacpp::Vector<double>& u_kout,
                      const dacpp::Vector<double>& r) {
    dacpp::index i;
    dacpp::split s(3, 1);
    binding(i, s);
    dacpp::list dataList{u_kin[s], u_kout[i], r[{}]};
    return dataList;
}

calc void pde(dacpp::Vector<double>& u_kin, double* u_kout, double* r) {
    u_kout[0] =
        r[0] * u_kin[0] + (1 - 2 * r[0]) * u_kin[1] + r[0] * u_kin[2];
}

int main() {
    int n = 8;
    int m = 5;
    double r = 0.25;
    std::vector<double> u_flat((m + 1) * (n + 1), 0.0);
    dacpp::Matrix<double> u_tensor({m + 1, n + 1}, u_flat);

    for (int k = 0; k <= n - 1; k++) {
        dacpp::Vector<double> u_kout = u_tensor[{1, m}][k + 1];
        std::vector<double> r_data;
        r_data.push_back(r+0.0);
        dacpp::Vector<double> r(r_data);
        dacpp::Vector<double> u_kin = u_tensor[{}][k];
        PDE(u_kin, u_kout, r) <-> pde;
        for (int i = 1; i <= m - 1; i++) {
            u_tensor[i][k + 1] = u_kout[i - 1];
        }
    }

    u_tensor[1].print();
    return 0;
}
