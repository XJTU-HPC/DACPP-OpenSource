#include <iomanip>
#include <iostream>
#include <vector>

#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

const int WIDTH = 72;
const int TIME_STEPS = 7;

calc void fan_step(dacpp::Vector<double>& left, dacpp::Vector<double>& right,
                   double* next) {
    next[0] = 0.15 * left[0] + 0.5 * left[1] + 0.35 * right[2];
}

shell dacpp::list fanShell(dacpp::Vector<double>& left READ,
                           dacpp::Vector<double>& right READ,
                           dacpp::Vector<double>& next WRITE) {
    dacpp::split S1(3, 1);
    dacpp::index idx1;
    binding(S1, idx1);
    dacpp::list dataList{left[S1], right[S1], next[idx1]};
    return dataList;
}

int main() {
    std::vector<double> left_data(WIDTH, 0.0);
    std::vector<double> right_data(WIDTH, 0.0);
    std::vector<double> next_data(WIDTH, 0.0);

    for (int i = 0; i < WIDTH; ++i) {
        left_data[i] = static_cast<double>((i % 5) + (i / 10));
        right_data[i] = static_cast<double>((i % 6) + 3);
    }

    dacpp::Vector<double> left_tensor(left_data);
    dacpp::Vector<double> right_tensor(right_data);
    dacpp::Vector<double> next_tensor(next_data);
    dacpp::Vector<double> left = left_tensor[{0, WIDTH - 1}];
    dacpp::Vector<double> right = right_tensor[{0, WIDTH - 1}];
    dacpp::Vector<double> next = next_tensor[{1, WIDTH - 1}];

    for (int t = 0; t < TIME_STEPS; ++t) {
        fanShell(left, right, next) <-> fan_step;

        for (int i = 1; i <= WIDTH - 2; ++i) {
            left[i] = next[i - 1];
            right[i] = next[i - 1];
        }
    }

    std::cout << std::fixed << std::setprecision(4)
              << left[6] << " " << right[7] << std::endl;
    return 0;
}
