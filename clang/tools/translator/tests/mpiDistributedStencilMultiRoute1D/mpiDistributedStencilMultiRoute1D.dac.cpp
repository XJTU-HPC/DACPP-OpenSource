#include <iomanip>
#include <iostream>
#include <vector>

#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

const int WIDTH = 72;
const int TIME_STEPS = 8;

calc void two_step(dacpp::Vector<double>& a, dacpp::Vector<double>& b,
                   double* next_a, double* next_b) {
    next_a[0] = 0.2 * a[0] + 0.6 * a[1] + 0.2 * a[2];
    next_b[0] = 0.1 * b[0] + 0.7 * b[1] + 0.2 * b[2];
}

shell dacpp::list twoShell(dacpp::Vector<double>& a READ,
                           dacpp::Vector<double>& b READ,
                           dacpp::Vector<double>& next_a WRITE,
                           dacpp::Vector<double>& next_b WRITE) {
    dacpp::split S1(3, 1);
    dacpp::index idx1;
    binding(S1, idx1);
    dacpp::list dataList{a[S1], b[S1], next_a[idx1], next_b[idx1]};
    return dataList;
}

int main() {
    std::vector<double> a_data(WIDTH, 0.0);
    std::vector<double> b_data(WIDTH, 0.0);
    std::vector<double> next_a_data(WIDTH, 0.0);
    std::vector<double> next_b_data(WIDTH, 0.0);

    for (int i = 0; i < WIDTH; ++i) {
        a_data[i] = static_cast<double>((i % 9) + 1);
        b_data[i] = static_cast<double>((i % 7) + 2);
    }

    dacpp::Vector<double> a_tensor(a_data);
    dacpp::Vector<double> b_tensor(b_data);
    dacpp::Vector<double> next_a_tensor(next_a_data);
    dacpp::Vector<double> next_b_tensor(next_b_data);
    dacpp::Vector<double> a = a_tensor[{0, WIDTH - 1}];
    dacpp::Vector<double> b = b_tensor[{0, WIDTH - 1}];
    dacpp::Vector<double> next_a = next_a_tensor[{1, WIDTH - 1}];
    dacpp::Vector<double> next_b = next_b_tensor[{1, WIDTH - 1}];

    for (int t = 0; t < TIME_STEPS; ++t) {
        twoShell(a, b, next_a, next_b) <-> two_step;

        for (int i = 1; i <= WIDTH - 2; ++i) {
            a[i] = next_a[i - 1];
            b[i] = next_b[i - 1];
        }
    }

    std::cout << std::fixed << std::setprecision(4)
              << a[6] << " " << b[8] << std::endl;
    return 0;
}
